//! Causal self-attention with a persistent KV cache. Ports the attention
//! block inlined in `llm_forward` (the `for hh in 0..H` loop and the
//! kcache/vcache memcpy before it).

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
            let mut dot = 0.0f32;
            for i in 0..head_dim {
                dot += qh[i] * kt[i];
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
            for i in 0..head_dim {
                ao[i] += w * vt[i];
            }
        }
        for i in 0..head_dim {
            ao[i] /= denom;
        }
    }
}
