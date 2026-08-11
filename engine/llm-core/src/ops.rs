//! Elementwise ops. Ports `rmsnorm`, `gelu`, `silu` from llm.h, verbatim.
//!
//! PERFORMANCE NOTE: `rms_inv`/`rmsnorm`/`rmsnorm_inplace` used raw `usize`
//! indexing (`x[i]`, `out[i]`, `buf[i]`) until this fix -- same category of
//! needless per-element overhead `tensor.rs`'s `matvec_range` and
//! `attention.rs`'s dot-product loops were found to have (see their own
//! doc comments). Slicing `x[..n]`/`out[..n]`/`buf[..n]` down to a common,
//! provable length before iterating (`quantize_activations` in `tensor.rs`
//! already does this for its own single-slice loop) lets the backend
//! elide bounds checks it otherwise has to re-derive from the separately
//! passed `n: usize` on every element. `FVec::get(i)` itself still needs
//! an explicit index (it's not slice-backed), so these loops keep an
//! `.enumerate()` for it rather than a pure `.zip()` -- that call's own
//! four-byte-at-a-time indexing is a separate, smaller, not-yet-addressed
//! cost (see `tensor.rs`'s `FVec::get` doc comment). Same left-to-right
//! summation order as before in every loop below, so this is a pure
//! loop-shape change, not a numerics change -- verified via `llm-host`'s
//! golden/ppl/cli parity tests.

use crate::tensor::FVec;

const RMS_EPS: f32 = 1e-6;

#[inline]
fn rms_inv(x: &[f32], n: usize) -> f32 {
    let mut ss = 0f32;
    for &v in &x[..n] {
        ss += v * v;
    }
    1.0 / libm::sqrtf(ss / n as f32 + RMS_EPS)
}

/// out[i] = w[i] * x[i] * rsqrt(mean(x^2) + eps), x and out disjoint
/// buffers. Ports `rmsnorm(x, w, n, out)` for the C call sites where `x !=
/// out` (e.g. `rmsnorm(s->x, attn_norm, D, s->h)`).
pub fn rmsnorm(x: &[f32], w: FVec, n: usize, out: &mut [f32]) {
    let inv = rms_inv(x, n);
    for (i, (&xv, ov)) in x[..n].iter().zip(out[..n].iter_mut()).enumerate() {
        *ov = w.get(i) * xv * inv;
    }
}

/// Same as `rmsnorm`, but in place (buf[i] = w[i] * buf[i] * rsqrt(...)).
/// Ports the C call sites where `x == out` (the same pointer passed twice) --
/// Rust can't alias `&[f32]`/`&mut [f32]` over the same memory, so those
/// call sites get this single-mutable-borrow version instead. Identical
/// arithmetic, same result.
pub fn rmsnorm_inplace(buf: &mut [f32], w: FVec, n: usize) {
    let inv = rms_inv(buf, n);
    for (i, bv) in buf[..n].iter_mut().enumerate() {
        *bv = w.get(i) * *bv * inv;
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
