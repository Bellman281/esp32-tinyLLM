//! Elementwise ops. Ports `rmsnorm`, `gelu`, `silu` from llm.h, verbatim.

use crate::tensor::FVec;

const RMS_EPS: f32 = 1e-6;

#[inline]
fn rms_inv(x: &[f32], n: usize) -> f32 {
    let mut ss = 0f32;
    for i in 0..n {
        ss += x[i] * x[i];
    }
    1.0 / libm::sqrtf(ss / n as f32 + RMS_EPS)
}

/// out[i] = w[i] * x[i] * rsqrt(mean(x^2) + eps), x and out disjoint
/// buffers. Ports `rmsnorm(x, w, n, out)` for the C call sites where `x !=
/// out` (e.g. `rmsnorm(s->x, attn_norm, D, s->h)`).
pub fn rmsnorm(x: &[f32], w: FVec, n: usize, out: &mut [f32]) {
    let inv = rms_inv(x, n);
    for i in 0..n {
        out[i] = w.get(i) * x[i] * inv;
    }
}

/// Same as `rmsnorm`, but in place (buf[i] = w[i] * buf[i] * rsqrt(...)).
/// Ports the C call sites where `x == out` (the same pointer passed twice) --
/// Rust can't alias `&[f32]`/`&mut [f32]` over the same memory, so those
/// call sites get this single-mutable-borrow version instead. Identical
/// arithmetic, same result.
pub fn rmsnorm_inplace(buf: &mut [f32], w: FVec, n: usize) {
    let inv = rms_inv(buf, n);
    for i in 0..n {
        buf[i] = w.get(i) * buf[i] * inv;
    }
}

/// 0.5*x*(1+erf(x/sqrt(2))). Ports `gelu` (erf-based, matches
/// `torch.nn.functional.gelu`'s default, not the tanh approximation).
// The literal below matches llm.h's `0.70710678f` exactly rather than
// `core::f32::consts::FRAC_1_SQRT_2` (which f32-rounds slightly
// differently) -- this port targets bit-for-bit parity with the C
// reference over idiomatic style here.
#[inline]
#[allow(clippy::approx_constant)]
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + libm::erff(x * 0.70710678f32))
}

/// x / (1 + exp(-x)). Ports `silu`.
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + libm::expf(-x))
}
