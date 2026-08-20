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
use crate::profile::Profile;
use crate::rope;
use crate::scratch::Scratch;
use crate::tensor::{
    Cfg, Cursor, FVec, LoadError, QT, FLAG_TIED_HEAD_CHECK, MAGIC_CHECK, MAGIC_V2_CHECK,
    MAX_FORMAT_VERSION_CHECK,
};

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

        // Two header formats carry the identical tensor stream; only the
        // preamble differs.
        //
        //   "PLE1"  magic, then 8 int32 config fields, then rope_theta.
        //           No version, no flags, no output_vocab. This is what the
        //           model in model/ and the frozen C reference both use.
        //
        //   "PLE\0" magic, version, header_bytes, flags, input_vocab,
        //           output_vocab, then the same 7 remaining config fields
        //           and rope_theta. What the exporter writes today.
        //
        // header_bytes exists so a later version can append fields without
        // breaking this reader: seek past it rather than assuming a size.
        let (vocab, out_vocab, dim, n_layers, n_heads, ffn, ple_dim, seq_len, group, rope_theta) =
            if magic == MAGIC_CHECK {
                let vocab = c.i32()? as usize;
                let dim = c.i32()? as usize;
                let n_layers = c.i32()? as usize;
                let n_heads = c.i32()? as usize;
                let ffn = c.i32()? as usize;
                let ple_dim = c.i32()? as usize;
                let seq_len = c.i32()? as usize;
                let group = c.i32()? as usize;
                let rope_theta = c.f32()?;
                // No output_vocab in this format: score every embedding row,
                // exactly as the C reference does. Narrowing the head here
                // would silently change greedy decode against a file whose
                // author never declared the distinction.
                (
                    vocab, vocab, dim, n_layers, n_heads, ffn, ple_dim, seq_len, group, rope_theta,
                )
            } else if magic == MAGIC_V2_CHECK {
                let version = c.u32()?;
                if version > MAX_FORMAT_VERSION_CHECK {
                    return Err(LoadError::UnsupportedVersion(version));
                }
                let header_bytes = c.u32()? as usize;
                let flags = c.u32()?;
                if flags & FLAG_TIED_HEAD_CHECK == 0 {
                    return Err(LoadError::UntiedHead);
                }
                let vocab = c.u32()? as usize;
                let out_vocab = c.u32()? as usize;
                let dim = c.i32()? as usize;
                let n_layers = c.i32()? as usize;
                let n_heads = c.i32()? as usize;
                let ffn = c.i32()? as usize;
                let ple_dim = c.i32()? as usize;
                let seq_len = c.i32()? as usize;
                let group = c.i32()? as usize;
                let rope_theta = c.f32()?;
                if out_vocab == 0 || out_vocab > vocab {
                    return Err(LoadError::BadOutputVocab);
                }
                // Skip anything a future version appended.
                if header_bytes < c.pos() {
                    return Err(LoadError::TooShort);
                }
                c = Cursor::at(base, header_bytes);
                (
                    vocab, out_vocab, dim, n_layers, n_heads, ffn, ple_dim, seq_len, group,
                    rope_theta,
                )
            } else {
                return Err(LoadError::BadMagic);
            };

        if n_layers > MAX_LAYERS {
            return Err(LoadError::TooManyLayers);
        }

        let cfg = Cfg {
            vocab,
            out_vocab,
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

    /// The tied input/output embedding table, `[vocab, dim]`. `QT` is
    /// `Copy`, so this returns an independent value the caller can freely
    /// adjust -- e.g. `llm-firmware` caps `.rows` to the trained vocab
    /// count (padding rows above that have no vocab entry and are never
    /// scored) before staging it as int8, exactly like the C reference's
    /// `model.tok_emb.rows = VOCAB_N;` before `stage_head_int8(&model.tok_emb)`.
    /// Mutating the returned copy does not affect `llm_forward`, which
    /// always reads the original untruncated table for the embedding
    /// lookup (`deq_row(&m->tok_emb, token, ...)` in C never observes the
    /// row cap either -- only the output-head path does, and only because
    /// the platform routes it through `head_matvec`).
    pub fn tok_emb(&self) -> QT<'a> {
        self.tok_emb
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

/// `matvec` restricted to rows `0..row_end`, honouring the same
/// int8/fp32 feature split. Used only for the output head, where
/// `row_end` is `cfg.out_vocab` -- see `Model::load`.
#[inline]
fn matvec_rows(t: &QT, x: &[f32], y: &mut [f32], iq: &mut [i8], row_end: usize) {
    #[cfg(feature = "int8-activations")]
    {
        t.matvec_int8_activations_range(x, y, iq, 0, row_end);
    }
    #[cfg(not(feature = "int8-activations"))]
    {
        let _ = iq;
        t.matvec_range(x, y, 0, row_end);
    }
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

/// Dispatches one matvec call: `matvec_override` if given, otherwise the
/// default `matvec()` above. Every per-tensor matvec call site in
/// `llm_forward_impl` goes through this instead of calling `matvec()`
/// directly, so a platform can override *every* matvec (not just the
/// head's, which already had its own dedicated hook) without
/// `llm_forward_impl` itself knowing or caring what the override does.
///
/// `matvec_override`'s signature deliberately matches the existing `head`
/// override's (`&[f32] in, &mut [f32] out`, no `iq` parameter) -- an
/// override is responsible for its own activation handling internally
/// (quantizing, SIMD-unpacking, whatever), the same way `llm-firmware`'s
/// `head.rs` already does for the head specifically.
#[inline]
fn dispatch_matvec(
    matvec_override: &mut Option<&mut dyn FnMut(&QT, &[f32], &mut [f32])>,
    t: &QT,
    x: &[f32],
    y: &mut [f32],
    iq: &mut [i8],
) {
    match matvec_override.as_deref_mut() {
        Some(f) => f(t, x, y),
        None => matvec(t, x, y, iq),
    }
}

/// One decode step: token at position `pos` -> logits (written into
/// `s.logits`). The KV cache in `s` persists across calls -- caller is
/// responsible for calling with increasing `pos` values, same contract as
/// the C `llm_forward`.
///
/// This is the path every host tool uses (gen_prompt.c, host_generate.c,
/// verify.c, ppl.c never set `head_matvec` either) -- the output head runs
/// through the same matvec as everything else. For the ESP32 firmware's
/// dual-core int8-staged output head (`Model.head_matvec` in the C code),
/// see `llm_forward_with_head_override` below.
pub fn llm_forward(model: &Model, token: usize, pos: usize, s: &mut Scratch) {
    llm_forward_impl(model, token, pos, s, None, None, None);
}

/// Same decode step as `llm_forward`, but the final output-head matvec is
/// replaced by `head` when given -- ports the C code's optional
/// `Model.head_matvec` function-pointer override (`x: &[f32]` in,
/// `y: &mut [f32]` logits out, same shapes `llm_forward` would have used).
/// `None` behaves identically to plain `llm_forward` (same code path,
/// verified by `llm-host`'s test suite).
///
/// This is the hook `llm-firmware` (Phase 3) uses for the dual-core
/// int8-staged head: the closure captures the PSRAM-staged int8 weights
/// and the FreeRTOS task handle for the other core's half of the rows,
/// while every other tensor op in this function -- the part already
/// verified against PyTorch -- doesn't change at all.
pub fn llm_forward_with_head_override(
    model: &Model,
    token: usize,
    pos: usize,
    s: &mut Scratch,
    head: &mut dyn FnMut(&[f32], &mut [f32]),
) {
    llm_forward_impl(model, token, pos, s, Some(head), None, None);
}

/// Same decode step as `llm_forward_with_head_override`, plus `llm.h`'s
/// `#ifdef LLM_PROFILE` per-stage timing (see `profile.rs`'s doc comment for
/// why the timestamp source is a caller-supplied closure rather than a
/// baked-in clock). `now` is called once per stage boundary per layer
/// (input once, then attn/ffn/ple per layer, then head once) -- same
/// boundaries `llm.h` instruments, same accumulation into `prof`. Pass a
/// fresh (or `.reset()`) `Profile` and call this instead of
/// `llm_forward_with_head_override` only where you actually want the
/// breakdown; the plain functions above have zero profiling overhead since
/// they pass `None` straight through.
pub fn llm_forward_profiled(
    model: &Model,
    token: usize,
    pos: usize,
    s: &mut Scratch,
    head: &mut dyn FnMut(&[f32], &mut [f32]),
    now: &mut dyn FnMut() -> u64,
    prof: &mut Profile,
) {
    llm_forward_impl(model, token, pos, s, Some(head), None, Some((now, prof)));
}

/// Same decode step as `llm_forward_profiled`, plus `matvec_override`:
/// replaces every non-head matvec (`qkv`, `attn_proj`, FFN `gate`/`up`/
/// `down`, `ple_gate`/`ple_proj`, and the PLE input's `ple_model_proj`)
/// with the given closure, the same way `head` already replaces just the
/// output head's. `None` for either override behaves identically to
/// `llm_forward_profiled`/`llm_forward`.
///
/// This is the hook a platform needing SIMD-accelerated matvecs beyond
/// just the head would use -- e.g. an `esp-dsp`-backed path in
/// `llm-firmware`, on the ESP32-S3's 128-bit PIE vector unit. No such
/// path ships today: this hook, `QT::matvec_range_chunked`, and
/// `llm-host/tests/matvec_simd_tolerance.rs` are the scaffolding for
/// writing one (the hard part -- knowing how much numeric drift to
/// accept -- is already measured), not an implementation of one. See the
/// `BENCHMARKING.md`'s "What to optimize next".
///
/// Unlike the head override, a `matvec_override` here is NOT expected to
/// match the default fp32 path bit-for-bit -- see
/// `QT::matvec_range_chunked`'s doc comment in `tensor.rs` for why
/// reordering an `f32` accumulation (which any real SIMD dot product
/// does) is inherently lossy in a way the head's int8 accumulation isn't,
/// and how that's measured/tolerated instead of assumed away.
#[allow(clippy::too_many_arguments)]
pub fn llm_forward_profiled_with_matvec_override(
    model: &Model,
    token: usize,
    pos: usize,
    s: &mut Scratch,
    head: &mut dyn FnMut(&[f32], &mut [f32]),
    matvec_override: &mut dyn FnMut(&QT, &[f32], &mut [f32]),
    now: &mut dyn FnMut() -> u64,
    prof: &mut Profile,
) {
    llm_forward_impl(
        model,
        token,
        pos,
        s,
        Some(head),
        Some(matvec_override),
        Some((now, prof)),
    );
}

/// Same as `llm_forward_profiled_with_matvec_override`, without profiling
/// -- for a platform that wants `matvec_override` but not the per-stage
/// timing breakdown (e.g. a plain generation loop instead of a
/// measurement run).
pub fn llm_forward_with_matvec_override(
    model: &Model,
    token: usize,
    pos: usize,
    s: &mut Scratch,
    head: &mut dyn FnMut(&[f32], &mut [f32]),
    matvec_override: &mut dyn FnMut(&QT, &[f32], &mut [f32]),
) {
    llm_forward_impl(model, token, pos, s, Some(head), Some(matvec_override), None);
}

fn llm_forward_impl(
    model: &Model,
    token: usize,
    pos: usize,
    s: &mut Scratch,
    mut head: Option<&mut dyn FnMut(&[f32], &mut [f32])>,
    mut matvec_override: Option<&mut dyn FnMut(&QT, &[f32], &mut [f32])>,
    mut prof: Option<(&mut dyn FnMut() -> u64, &mut Profile)>,
) {
    // `last_t` tracks the timestamp at the previous stage boundary --
    // matches `llm.h`'s reuse of a single rolling `profile_tN` local across
    // stages (e.g. `profile_t1` is both "end of input prep" and, after
    // being reassigned inside the loop, "end of this layer's PLE stage").
    // Only ever read/written when `prof.is_some()`; the `0` default is
    // never observed otherwise.
    let mut last_t: u64 = 0;
    if let Some((now, _)) = prof.as_mut() {
        last_t = now();
    }
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
    dispatch_matvec(&mut matvec_override, &model.ple_model_proj, s.x, &mut s.tmp_p[..lp], s.iq);
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

    if let Some((now, p)) = prof.as_mut() {
        let t = now();
        p.input_us += t - last_t;
        last_t = t;
    }

    for l in 0..l_layers {
        let layer = model.layer(l);

        // ---- attention
        //
        // Split four ways when profiling. `attn` is a third of a token and was
        // the last stage reported as one number; the four parts have different
        // levers (see `Profile::attn_detail_ms_per_token`), and guessing which
        // one dominates is exactly what cost flashes on the head. Each `sub`
        // below reads the clock only when a `Profile` was supplied.
        let mut sub_t = last_t;
        macro_rules! sub {
            ($field:ident) => {
                if let Some((now, p)) = prof.as_mut() {
                    let t = now();
                    p.$field += t - sub_t;
                    sub_t = t;
                }
            };
        }
        ops::rmsnorm(s.x, layer.attn_norm, d, s.h);
        dispatch_matvec(&mut matvec_override, &layer.qkv, s.h, &mut s.qkv[..3 * d], s.iq);
        sub!(attn_qkv_us);
        {
            let (q, rest) = s.qkv[..3 * d].split_at_mut(d);
            let (k, v) = rest.split_at_mut(d);
            for hh in 0..h_heads {
                rope::apply(&mut q[hh * dh..hh * dh + dh], s.rope_cos, s.rope_sin, dh);
                rope::apply(&mut k[hh * dh..hh * dh + dh], s.rope_cos, s.rope_sin, dh);
            }
            sub!(attn_rope_us);
            let seq = cfg.seq_len;
            let layer_kc = &mut s.kcache[l * seq * d..(l + 1) * seq * d];
            let layer_vc = &mut s.vcache[l * seq * d..(l + 1) * seq * d];
            attention::forward(
                q, k, v, layer_kc, layer_vc, s.att, s.scores, pos, d, h_heads, dh, seq,
            );
        }
        sub!(attn_core_us);
        dispatch_matvec(&mut matvec_override, &layer.attn_proj, s.att, s.h, s.iq);
        for i in 0..d {
            s.x[i] += s.h[i];
        }
        sub!(attn_proj_us);
        // The last `sub!` advances `sub_t` and nothing reads it afterwards.
        // Consuming it here is cheaper than an `#[allow]`, which would have to
        // sit on the macro expansion rather than the binding to take effect.
        let _ = sub_t;

        if let Some((now, p)) = prof.as_mut() {
            let t = now();
            p.attn_us += t - last_t;
            last_t = t;
        }

        // ---- SwiGLU FFN
        ops::rmsnorm(s.x, layer.ffn_norm, d, s.h);
        dispatch_matvec(&mut matvec_override, &layer.gate, s.h, &mut s.g1[..f], s.iq);
        dispatch_matvec(&mut matvec_override, &layer.up, s.h, &mut s.g2[..f], s.iq);
        for i in 0..f {
            s.g1[i] = ops::silu(s.g1[i]) * s.g2[i];
        }
        dispatch_matvec(&mut matvec_override, &layer.down, &s.g1[..f], s.h, s.iq);
        for i in 0..d {
            s.x[i] += s.h[i];
        }

        if let Some((now, p)) = prof.as_mut() {
            let t = now();
            p.ffn_us += t - last_t;
            last_t = t;
        }

        // ---- PLE gate: x += RMSNorm(ple_proj(gelu(ple_gate(x)) * ple_l))
        dispatch_matvec(&mut matvec_override, &layer.ple_gate, s.x, &mut s.g2[..p], s.iq);
        for i in 0..p {
            s.g2[i] = ops::gelu(s.g2[i]) * s.ple[l * p + i];
        }
        dispatch_matvec(&mut matvec_override, &layer.ple_proj, &s.g2[..p], s.h, s.iq);
        ops::rmsnorm_inplace(&mut s.h[..d], layer.ple_norm, d);
        for i in 0..d {
            s.x[i] += s.h[i];
        }

        if let Some((now, p)) = prof.as_mut() {
            let t = now();
            p.ple_us += t - last_t;
            last_t = t;
        }
    }

    ops::rmsnorm_inplace(s.x, model.out_norm, d);
    match head.as_deref_mut() {
        Some(f) => f(s.x, s.logits),
        // `s.logits` is sized `out_vocab`, which is `vocab` for every PLE1
        // file. Where a PLE\0 file declares it smaller, the rows above it
        // are padding the tokenizer can never emit, so scoring them is
        // wasted work in the single most expensive stage. An override is
        // handed the same slice and must respect its length.
        None => match matvec_override.as_deref_mut() {
            Some(f) => f(&model.tok_emb, s.x, s.logits),
            // Explicitly row-bounded rather than `matvec`, which would run
            // every embedding row and write past a shorter `logits`.
            None => matvec_rows(&model.tok_emb, s.x, s.logits, s.iq, model.cfg.out_vocab),
        },
    }

    if let Some((now, p)) = prof.as_mut() {
        let t = now();
        p.head_us += t - last_t;
        p.calls += 1;
    }
}
