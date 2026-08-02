//! Ties the pieces together: `Model` (bound tensors), `llm_load`-equivalent
//! parsing, and `llm_forward` -- the full per-token decode step.
//!
//! Structural note on per-layer tensors: the C `Model` struct holds fixed
//! `[32]` arrays of `QT`/`float*` for the per-layer tensors, filled once at
//! load time. `QT<'a>` here borrows from the model buffer and isn't
//! `Default`, so a `[QT; 32]` array is awkward to build safely. Instead this
//! port records each layer's *byte offset* into the buffer at load time
//! (cheap, `[usize; MAX_LAYERS]`) and re-derives that layer's `Layer<'a>`
//! on demand in `Model::layer()` -- a few integer multiplies, not a data
//! copy, so it costs nothing next to the matvecs that dominate a forward
//! pass. Numerically identical to the C layout; only the in-memory
//! representation differs.

use crate::attention;
use crate::ops;
use crate::rope;
use crate::scratch::Scratch;
use crate::tensor::{Cfg, Cursor, FVec, LoadError, QT, MAGIC_CHECK};

/// Matches the C code's fixed per-layer array capacity (`Model.attn_norm[32]`
/// etc). `Model::load` rejects anything larger instead of silently
/// overflowing like the C fixed array would.
pub const MAX_LAYERS: usize = 32;

#[derive(Clone, Copy)]
pub struct Layer<'a> {
    pub attn_norm: FVec<'a>,
    pub qkv: QT<'a>,
    pub attn_proj: QT<'a>,
    pub ffn_norm: FVec<'a>,
    pub gate: QT<'a>,
    pub up: QT<'a>,
    pub down: QT<'a>,
    pub ple_gate: QT<'a>,
    pub ple_proj: QT<'a>,
    pub ple_norm: FVec<'a>,
}

fn bind_layer<'a>(base: &'a [u8], offset: usize, d: usize, f: usize, p: usize) -> Layer<'a> {
    let mut c = Cursor::at(base, offset);
    // Order matches llm_load's per-layer bind sequence exactly.
    let attn_norm = c.bind_f(d).expect("layer offset computed at load time");
    let qkv = c.bind_q(3 * d, d).expect("layer offset computed at load time");
    let attn_proj = c.bind_q(d, d).expect("layer offset computed at load time");
    let ffn_norm = c.bind_f(d).expect("layer offset computed at load time");
    let gate = c.bind_q(f, d).expect("layer offset computed at load time");
    let up = c.bind_q(f, d).expect("layer offset computed at load time");
    let down = c.bind_q(d, f).expect("layer offset computed at load time");
    let ple_gate = c.bind_q(p, d).expect("layer offset computed at load time");
    let ple_proj = c.bind_q(d, p).expect("layer offset computed at load time");
    let ple_norm = c.bind_f(d).expect("layer offset computed at load time");
    Layer {
        attn_norm,
        qkv,
        attn_proj,
        ffn_norm,
        gate,
        up,
        down,
        ple_gate,
        ple_proj,
        ple_norm,
    }
}

pub struct Model<'a> {
    pub cfg: Cfg,
    base: &'a [u8],
    tok_emb: QT<'a>,
    ple_model_proj: QT<'a>,
    ple_proj_norm: FVec<'a>,
    ple_table: QT<'a>,
    out_norm: FVec<'a>,
    layer_offsets: [usize; MAX_LAYERS],
}

impl<'a> Model<'a> {
    /// Parse header + bind all tensors. Ports `llm_load`.
    pub fn load(base: &'a [u8]) -> Result<Self, LoadError> {
        let mut c = Cursor::new(base);
        let magic = c.u32()?;
        if magic != MAGIC_CHECK {
            return Err(LoadError::BadMagic);
        }
        let vocab = c.i32()? as usize;
        let dim = c.i32()? as usize;
        let n_layers = c.i32()? as usize;
        let n_heads = c.i32()? as usize;
        let ffn = c.i32()? as usize;
        let ple_dim = c.i32()? as usize;
        let seq_len = c.i32()? as usize;
        let group = c.i32()? as usize;
        let rope_theta = c.f32()?;

        if n_layers > MAX_LAYERS {
            return Err(LoadError::TooManyLayers);
        }

        let cfg = Cfg {
            vocab,
            dim,
            n_layers,
            n_heads,
            ffn,
            ple_dim,
            seq_len,
            group,
            rope_theta,
        };
        let (d, l, p, f, v) = (dim, n_layers, ple_dim, ffn, vocab);

        let tok_emb = c.bind_q(v, d)?;
        let ple_model_proj = c.bind_q(l * p, d)?;
        let ple_proj_norm = c.bind_f(p)?;
        let ple_table = c.bind_q(v, l * p)?;

        let mut layer_offsets = [0usize; MAX_LAYERS];
        for i in 0..l {
            layer_offsets[i] = c.pos();
            // Validate this layer's bind sequence fits in the buffer now
            // (`try_bind_layer` returns Err instead of panicking, unlike
            // `bind_layer`/`Model::layer()`, which trust load-time
            // validation and can't fail once it's passed), then advance the
            // cursor past it.
            let end = try_bind_layer(base, layer_offsets[i], d, f, p)?;
            c = Cursor::at(base, end);
        }

        let out_norm = c.bind_f(d)?;

        Ok(Model {
            cfg,
            base,
            tok_emb,
            ple_model_proj,
            ple_proj_norm,
            ple_table,
            out_norm,
            layer_offsets,
        })
    }

    pub fn layer(&self, l: usize) -> Layer<'a> {
        bind_layer(self.base, self.layer_offsets[l], self.cfg.dim, self.cfg.ffn, self.cfg.ple_dim)
    }
}

/// Byte offset just past one layer's bind sequence, starting at `offset`.
/// Used at load time to advance the cursor without keeping `bind_layer`'s
/// internal `Cursor` borrowed past its own scope.
/// Fallible version of `bind_layer`'s bind sequence, used once at load time
/// to validate the buffer is long enough for this layer and report the
/// offset just past it. Returns the position after `ple_norm` (the last
/// field bound per layer).
fn try_bind_layer(
    base: &[u8],
    offset: usize,
    d: usize,
    f: usize,
    p: usize,
) -> Result<usize, LoadError> {
    let mut c = Cursor::at(base, offset);
    c.bind_f(d)?; // attn_norm
    c.bind_q(3 * d, d)?; // qkv
    c.bind_q(d, d)?; // attn_proj
    c.bind_f(d)?; // ffn_norm
    c.bind_q(f, d)?; // gate
    c.bind_q(f, d)?; // up
    c.bind_q(d, f)?; // down
    c.bind_q(p, d)?; // ple_gate
    c.bind_q(d, p)?; // ple_proj
    c.bind_f(d)?; // ple_norm
    Ok(c.pos())
}

#[inline]
fn matvec(t: &QT, x: &[f32], y: &mut [f32], iq: &mut [i8]) {
    #[cfg(feature = "int8-activations")]
    {
        t.matvec_int8_activations(x, y, iq);
    }
    #[cfg(not(feature = "int8-activations"))]
    {
        let _ = iq;
        t.matvec(x, y);
    }
}

/// One decode step: token at position `pos` -> logits (written into
/// `s.logits`). The KV cache in `s` persists across calls -- caller is
/// responsible for calling with increasing `pos` values, same contract as
/// the C `llm_forward`.
///
/// Platform-specific overrides (the ESP32 firmware's dual-core int8-staged
/// output head, `Model.head_matvec` in the C code) are deliberately not
/// modeled here -- that's a `llm-firmware` (Phase 3) concern, out of scope
/// for this platform-agnostic crate. This always runs the standard head
/// matvec, matching every host tool (gen_prompt.c, host_generate.c,
/// verify.c, ppl.c never set head_matvec either).
pub fn llm_forward(model: &Model, token: usize, pos: usize, s: &mut Scratch) {
    let cfg = &model.cfg;
    let d = cfg.dim;
    let l_layers = cfg.n_layers;
    let p = cfg.ple_dim;
    let f = cfg.ffn;
    let h_heads = cfg.n_heads;
    let dh = cfg.head_dim();
    let lp = l_layers * p;

    model.tok_emb.deq_row(token, s.x);

    // ---- per-layer input: (RMSNorm(proj(x)/sqrt(D)) + table[tok]*sqrt(P)) / sqrt(2)
    matvec(&model.ple_model_proj, s.x, &mut s.tmp_p[..lp], s.iq);
    let dscale = 1.0 / libm::sqrtf(d as f32);
    for v in s.tmp_p[..lp].iter_mut() {
        *v *= dscale;
    }
    for l in 0..l_layers {
        ops::rmsnorm_inplace(&mut s.tmp_p[l * p..l * p + p], model.ple_proj_norm, p);
    }
    model.ple_table.deq_row(token, &mut s.trow[..lp]);
    let sp = libm::sqrtf(p as f32);
    // Matches llm.h's `0.70710678f` literal exactly (see ops::gelu's doc
    // comment) rather than the stdlib FRAC_1_SQRT_2 constant.
    #[allow(clippy::approx_constant)]
    let inv2 = 0.70710678f32;
    for i in 0..lp {
        s.ple[i] = (s.tmp_p[i] + s.trow[i] * sp) * inv2;
    }

    // RoPE cos/sin for this position (see rope.rs for the dedicated-buffer note).
    rope::compute(pos, dh, cfg.rope_theta, s.rope_cos, s.rope_sin);

    for l in 0..l_layers {
        let layer = model.layer(l);

        // ---- attention
        ops::rmsnorm(s.x, layer.attn_norm, d, s.h);
        matvec(&layer.qkv, s.h, &mut s.qkv[..3 * d], s.iq);
        {
            let (q, rest) = s.qkv[..3 * d].split_at_mut(d);
            let (k, v) = rest.split_at_mut(d);
            for hh in 0..h_heads {
                rope::apply(&mut q[hh * dh..hh * dh + dh], s.rope_cos, s.rope_sin, dh);
                rope::apply(&mut k[hh * dh..hh * dh + dh], s.rope_cos, s.rope_sin, dh);
            }
            let seq = cfg.seq_len;
            let layer_kc = &mut s.kcache[l * seq * d..(l + 1) * seq * d];
            let layer_vc = &mut s.vcache[l * seq * d..(l + 1) * seq * d];
            attention::forward(
                q, k, v, layer_kc, layer_vc, s.att, s.scores, pos, d, h_heads, dh,
            );
        }
        matvec(&layer.attn_proj, s.att, s.h, s.iq);
        for i in 0..d {
            s.x[i] += s.h[i];
        }

        // ---- SwiGLU FFN
        ops::rmsnorm(s.x, layer.ffn_norm, d, s.h);
        matvec(&layer.gate, s.h, &mut s.g1[..f], s.iq);
        matvec(&layer.up, s.h, &mut s.g2[..f], s.iq);
        for i in 0..f {
            s.g1[i] = ops::silu(s.g1[i]) * s.g2[i];
        }
        matvec(&layer.down, &s.g1[..f], s.h, s.iq);
        for i in 0..d {
            s.x[i] += s.h[i];
        }

        // ---- PLE gate: x += RMSNorm(ple_proj(gelu(ple_gate(x)) * ple_l))
        matvec(&layer.ple_gate, s.x, &mut s.g2[..p], s.iq);
        for i in 0..p {
            s.g2[i] = ops::gelu(s.g2[i]) * s.ple[l * p + i];
        }
        matvec(&layer.ple_proj, &s.g2[..p], s.h, s.iq);
        ops::rmsnorm_inplace(&mut s.h[..d], layer.ple_norm, d);
        for i in 0..d {
            s.x[i] += s.h[i];
        }
    }

    ops::rmsnorm_inplace(s.x, model.out_norm, d);
    matvec(&model.tok_emb, s.x, s.logits, s.iq);
}
