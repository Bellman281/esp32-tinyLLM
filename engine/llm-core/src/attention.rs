//! Causal self-attention with a persistent KV cache. Ports the attention
//! block inlined in `llm_forward` (the `for hh in 0..H` loop and the
//! kcache/vcache memcpy before it).
//!
//! PERFORMANCE NOTE: the two hot loops below (`dot += qh[i]*kt[i]` and
//! `ao[i] += w*vt[i]`) used raw `usize` indexing until this fix -- the
//! same shape `tensor.rs`'s `matvec_range`/`matvec_int8_activations` were
//! found to be doing needless per-element work in before their
//! byte-pair-unroll fix (see that fix's doc comment). `qh`/`kt`/`ao`/`vt`
//! are always freshly sliced to exactly `head_dim` right before these
//! loops, so LLVM *can* prove the indices are in range, but proving it
//! reliably at every optimization level is exactly what an indexed loop
//! makes it work harder for -- `.zip()` over two slices of a shared,
//! provable length hands the backend that fact directly instead of
//! leaving it to infer, the same reasoning `matvec_range`'s doc comment
//! gives for why it slices `actq`/`row` down before zipping. Attention is
//! this port's worst C-vs-Rust ratio (2.10x, see the top-level README's
//! Status section) and this is a pure loop-shape change -- same
//! left-to-right summation order, so bit-for-bit output is unaffected;
//! verified via `llm-host`'s golden/ppl/cli parity tests, which would
//! catch even a single bit of drift.

/// Write this position's post-RoPE k/v into the per-layer cache, then run
/// causal multi-head attention over `0..=pos` and write the result into
/// `att_out` (len == dim, all heads concatenated, same layout as q/k/v).
///
/// `kcache_layer` / `vcache_layer` are this layer's slice of the full
/// `[n_layers * seq_len * dim]` cache (`&mut [f32]` of length
/// `seq_len * dim`) -- the caller slices the right layer out before calling,
/// same as the C code's `kc = s->kcache + l*S*D`.
///
/// `scores` is caller-owned scratch of length >= seq_len, reused per head
/// exactly as in the C code.
#[allow(clippy::too_many_arguments)]
pub fn forward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    kcache_layer: &mut [f32],
    vcache_layer: &mut [f32],
    att_out: &mut [f32],
    scores: &mut [f32],
    pos: usize,
    dim: usize,
    n_heads: usize,
    head_dim: usize,
) {
    kcache_layer[pos * dim..pos * dim + dim].copy_from_slice(&k[..dim]);
    vcache_layer[pos * dim..pos * dim + dim].copy_from_slice(&v[..dim]);

    let scale = 1.0 / libm::sqrtf(head_dim as f32);

    for hh in 0..n_heads {
        let qh = &q[hh * head_dim..hh * head_dim + head_dim];
        let ao = &mut att_out[hh * head_dim..hh * head_dim + head_dim];
        for v in ao.iter_mut() {
            *v = 0.0;
        }

        let mut maxs = -1e30f32;
        for t in 0..=pos {
            let kt = &kcache_layer[t * dim + hh * head_dim..t * dim + hh * head_dim + head_dim];
            // Same summation order as the old `for i in 0..head_dim`
            // loop -- see this module's doc comment.
            let mut dot = 0.0f32;
            for (&a, &b) in qh.iter().zip(kt.iter()) {
                dot += a * b;
            }
            dot *= scale;
            scores[t] = dot;
            if dot > maxs {
                maxs = dot;
            }
        }

        let mut denom = 0.0f32;
        for t in 0..=pos {
            let w = libm::expf(scores[t] - maxs);
            denom += w;
            let vt = &vcache_layer[t * dim + hh * head_dim..t * dim + hh * head_dim + head_dim];
            // Same summation order as the old `for i in 0..head_dim`
            // loop -- see this module's doc comment.
            for (av, &vv) in ao.iter_mut().zip(vt.iter()) {
                *av += w * vv;
            }
        }
        for v in ao.iter_mut() {
            *v /= denom;
        }
    }
}
