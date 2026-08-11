//! Optional SIMD-accelerated fp32 matvec, wired in as `llm_core::model`'s
//! `matvec_override` hook (`llm_forward_with_matvec_override` /
//! `llm_forward_profiled_with_matvec_override`) in place of the default
//! `QT::matvec_range`. Behind the `matvec-simd` feature, default OFF.
//!
//! Unlike `head.rs`'s int8 `dot_i8` (`simd_dot.rs`), this is NOT expected to
//! reproduce `matvec_range`'s output bit-for-bit -- `matvec_range`'s
//! accumulation is `f32`, which is not associative, so any real SIMD dot
//! product's different summation order can move the result in the last few
//! bits. See `llm_core::tensor::QT::matvec_range_chunked`'s doc comment for
//! the full reasoning, and `llm-host/tests/matvec_simd_tolerance.rs` for the
//! actual measured bound this is held to: on real validation data (2048
//! predictions, `data/val_v32768.bin`), reordering a 128-element group's sum
//! into 4 or 8 interleaved partial sums moved cross-entropy by ~4e-8 --
//! about 45,000x smaller than the 0.002 noise floor `ppl_parity.rs` already
//! tolerates for int8-activations' own reordering. That measurement is for
//! `matvec_range_chunked`'s specific reduction shape (portable, host-side),
//! not a guarantee about whatever reduction order `esp-dsp`'s real
//! implementation uses -- rerun `matvec_simd_tolerance.rs` if that shape
//! turns out to matter (e.g. after point 2 below is confirmed), and compare
//! this module's actual on-device output against `matvec_range` over a
//! real prompt before trusting it changes generation quality by nothing
//! visible.
//!
//! Design: `QT`'s packed int4 codes and per-group scale are private fields
//! (`codes`/`scales`/`code()`/`scale()` are all crate-private in
//! `tensor.rs`) -- this module only has `QT`'s public API to work with, same
//! as any other crate. Rather than widen `llm-core`'s private surface for a
//! single unverified caller, this reuses the existing public `QT::deq_row`
//! (already verified bit-for-bit against the C reference by
//! `int8_head_staging.rs`'s `unpack_row_int8_matches_deq_row` test) to
//! dequantize a full row into a scratch buffer -- scale already folded into
//! each element -- then does ONE whole-row dot product against `x` via
//! esp-dsp, instead of per-group dot products the way `matvec_range` itself
//! is structured. Mathematically equivalent (per-group `group_acc * scale`
//! summed across groups == per-element `code*scale` summed across the whole
//! row, since scale is constant within a group and float multiplication
//! distributes over the sum up to reordering) -- but a DIFFERENT reordering
//! than `matvec_range_chunked` measures, so it got its own, separate host
//! measurement: `llm-host/tests/matvec_simd_tolerance.rs`'s
//! `matvec_deqrow_dot_within_measured_tolerance` test runs this exact
//! algorithm (scalar dot, `matvec-simd` off) against the real validation
//! set and measured drift = 0.000000027910 (~2.8e-8) from the exact
//! `matvec_range` reference -- same order of magnitude as, and slightly
//! smaller than, `matvec_range_chunked`'s LANES=4/8 results (~4e-8), well
//! inside the same 1e-5 bound. That covers this module's own reordering
//! (`deq_row` + whole-row dot, scalar); it does NOT cover whatever
//! additional reordering `esp-dsp`'s real internal reduction does on top --
//! that part still needs an on-device A/B once `matvec-simd` actually
//! builds and links (see point 5 below).
//!
//! VERIFICATION STATUS: not compiled, not linked, not run -- same tier as
//! `simd_dot.rs`, for the same reason (no ESP-IDF/hardware access from this
//! sandbox). Everything in `head.rs`'s three-point checklist (function
//! name/signature, accumulation semantics, whether `idf_component.yml` gets
//! picked up) applies here too, PLUS:
//!
//! 4. **`dsps_dotprod_f32`'s real name/signature.** Guessed by analogy with
//!    `simd_dot.rs`'s `dsps_dotprod_s8` guess and esp-dsp's documented
//!    `f32` API surface (`dsps_dotprod_f32_ansi`/`_ae32`/`_arp4` variants
//!    dispatched through a plain `dsps_dotprod_f32` entry point per esp-dsp's
//!    own pattern for other kernels) -- NOT confirmed against a real vendored
//!    copy. Check `managed_components/espressif__esp-dsp/dotprod/include/
//!    dotprod.h` after pulling the dependency, same as `simd_dot.rs` point 1.
//! 5. **Performance is unproven, not just numerics.** `matvec_range` is
//!    called ~8x per layer * 6 layers = far more often per token than the
//!    head's single call, over rows as short as `cols=66` (the FFN down-
//!    projection) up to `cols=128` (`ple_proj`) -- an FFI call per ROW (this
//!    module's granularity) has real per-call overhead that may or may not
//!    net positive at these small lengths, unlike the head's one dot product
//!    over up to ~25k rows where per-call overhead amortizes trivially. This
//!    needs a real on-hardware A/B (same `git stash`/`stash pop` methodology
//!    already used for `perf/attention-rmsnorm-zip`), not an assumption that
//!    "SIMD is faster" transfers from the head's case.
//!
//! With `matvec-simd` off (the default), this file's dequant-then-dot path
//! still runs (unlike `simd_dot.rs`, there's no free "identical to before"
//! branch here, since nothing called this override before it existed) --
//! but the dot product itself falls back to a plain scalar loop, so the
//! only behavior change vs. not installing this override at all is the
//! `deq_row`-then-single-dot reordering described above, not an esp-dsp
//! dependency. Only install this override (see main.rs) behind
//! `matvec-simd`; leave `llm_forward_profiled`/`llm_forward_with_head_override`
//! as the default wiring otherwise.

use llm_core::QT;

#[cfg(feature = "matvec-simd")]
mod ffi {
    extern "C" {
        // BEST-EFFORT SIGNATURE, UNVERIFIED -- see this module's doc
        // comment, point 4. Expected contract, by analogy with
        // `simd_dot.rs`'s `dsps_dotprod_s8`: writes sum(src1[i] * src2[i])
        // for i in 0..len as a plain f32 to *dest, returns esp_err_t (0 =
        // ESP_OK).
        pub fn dsps_dotprod_f32(src1: *const f32, src2: *const f32, dest: *mut f32, len: i32) -> i32;
    }
}

/// Computes `sum(a[i] * b[i])` over `a.len()` elements. Same
/// verified/unverified split as `simd_dot::dot_i8`: scalar loop with
/// `matvec-simd` off, esp-dsp FFI call with it on.
#[inline]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(feature = "matvec-simd")]
    {
        let mut acc: f32 = 0.0;
        // SAFETY: `a`/`b` point to `a.len()`/`b.len()` valid `f32`s (equal
        // lengths per the debug_assert above), `acc` is a valid `&mut f32`
        // on this stack frame for the call's duration. Whether the callee
        // actually honors "writes a plain f32 sum, no OOB read past `len`"
        // is exactly what's unverified -- see this module's doc comment,
        // point 4.
        let rc = unsafe { ffi::dsps_dotprod_f32(a.as_ptr(), b.as_ptr(), &mut acc, a.len() as i32) };
        debug_assert_eq!(rc, 0, "dsps_dotprod_f32 returned a non-zero esp_err_t");
        acc
    }

    #[cfg(not(feature = "matvec-simd"))]
    {
        let mut acc = 0f32;
        for (&x, &y) in a.iter().zip(b.iter()) {
            acc += x * y;
        }
        acc
    }
}

/// Largest `cols` value across every tensor `dispatch_matvec` can be called
/// with on this model (`ple_proj`'s cols == `ple_dim`, 128 on the shipped
/// config -- the largest of qkv/attn_proj/gate/up/down/ple_gate/ple_proj/
/// ple_model_proj/tok_emb's `cols`). 256 gives headroom for a larger future
/// model without silently truncating -- checked at the call site, not
/// assumed, same as `head.rs`'s `MAX_HEAD_COLS` pattern.
const MAX_COLS: usize = 256;

/// `matvec_override` closure body: `y[0..t.rows] = t * x`, via
/// `QT::deq_row` + a whole-row `dot_f32` per row instead of `matvec_range`'s
/// per-group accumulation. See this module's doc comment for why that's a
/// different (not necessarily smaller) source of float-reorder drift than
/// `matvec_range_chunked` measures, and why it wasn't separately host-
/// tested before this was written.
///
/// `scratch` is caller-owned, reused across every call (one per matvec call
/// site per layer, ~8 per token) -- avoids a heap allocation per row per
/// call the way a fresh `Vec` here would cost. Sized `>= t.cols` is the
/// caller's contract (checked in debug); `main.rs` should size it to
/// `MAX_COLS` once, up front, same as `head.rs`'s `actq` scratch.
pub fn matvec_range_simd(t: &QT, x: &[f32], y: &mut [f32], scratch: &mut [f32]) {
    debug_assert!(x.len() >= t.cols);
    debug_assert!(y.len() >= t.rows);
    debug_assert!(
        t.cols <= scratch.len() && t.cols <= MAX_COLS,
        "matvec_range_simd: tensor cols ({}) exceeds scratch/MAX_COLS ({})",
        t.cols,
        MAX_COLS.min(scratch.len())
    );
    let cols = t.cols;
    for r in 0..t.rows {
        t.deq_row(r, &mut scratch[..cols]);
        y[r] = dot_f32(&scratch[..cols], &x[..cols]);
    }
}

/// Fixed-size scratch owning the buffer `matvec_range_simd` needs, so
/// `main.rs` can build the override closure without a heap allocation per
/// call. Mirrors `head.rs`'s `Head::actq` field in spirit (a reusable
/// fixed-size scratch sized to this model's known-max dimension) but has no
/// dual-core concurrency concerns -- `matvec_override` only ever runs on
/// the single inference-loop core, sequentially, so a plain owned array
/// (no `UnsafeCell`/atomics) is sufficient.
pub struct MatvecSimdScratch {
    buf: [f32; MAX_COLS],
}

impl Default for MatvecSimdScratch {
    fn default() -> Self {
        MatvecSimdScratch { buf: [0.0; MAX_COLS] }
    }
}

impl MatvecSimdScratch {
    /// Runs `matvec_range_simd` using this scratch's owned buffer. Intended
    /// call shape at the wiring site (see `main.rs`):
    /// `let mut matvec_override = |t: &QT, x: &[f32], y: &mut [f32]| scratch.run(t, x, y);`
    pub fn run(&mut self, t: &QT, x: &[f32], y: &mut [f32]) {
        matvec_range_simd(t, x, y, &mut self.buf);
    }
}
