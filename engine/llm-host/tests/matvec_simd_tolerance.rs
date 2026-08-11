//! Measures how much a SIMD-shaped (lane-reduced) reordering of
//! `QT::matvec_range`'s float accumulation actually moves validation-set
//! perplexity, so a real on-device `esp-dsp` implementation has an
//! empirically-measured tolerance to be held to instead of a guessed one.
//!
//! Why this exists (see `QT::matvec_range_chunked`'s doc comment in
//! `tensor.rs` for the full reasoning): `matvec_range`'s accumulation is
//! `f32`, and float addition is not associative. Any real SIMD dot product
//! reorders the summation (`LANES` independent partial sums instead of one
//! running total), so it is NOT expected to reproduce `matvec_range`'s
//! output bit-for-bit the way the head's int8 accumulation could. This test
//! is the fp32-path analogue of `ppl_parity.rs`'s int8-activations noise-
//! floor test: same idea (measure real drift against real validation data,
//! bound it with headroom), different source of reordering (SIMD lane
//! width here, int8 quantization there).
//!
//! Only meaningful for the fp32-activations path -- `matvec_range_chunked`
//! is a reordered variant of `matvec_range`, which the int8-activations
//! build doesn't call at all (see `model.rs`'s private `matvec()` dispatch).
//! No `#[cfg(feature = "int8-activations")]` variant of this file exists for
//! that reason, same as `ppl_parity.rs`'s fp32 test is unconditional and its
//! int8 test is feature-gated, not the other way round -- instead this
//! whole file is compiled out under that feature (see the crate-level `cfg`
//! below), so it never accidentally compares an int8-quantized CE against a
//! fp32-chunked one.
#![cfg(not(feature = "int8-activations"))]

use llm_core::{llm_forward, llm_forward_with_matvec_override, Model, QT};
use llm_host::support::ScratchOwned;
use std::fs;

const MODEL_BIN: &str = "../../reference-c/esp32-llm-lab/model.bin";
const VAL_BIN: &str = "../../data/val_v32768.bin";
const WINDOWS: usize = 4;

fn load_fixtures() -> (Model<'static>, Vec<u16>) {
    let bytes: &'static [u8] = fs::read(MODEL_BIN)
        .unwrap_or_else(|e| panic!("read {MODEL_BIN}: {e} (run from llm-host/)"))
        .leak();
    let model = Model::load(bytes).expect("model.bin should parse");
    let val_bytes =
        fs::read(VAL_BIN).unwrap_or_else(|e| panic!("read {VAL_BIN}: {e} (run from llm-host/)"));
    let val: Vec<u16> = val_bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    (model, val)
}

/// Same cross-entropy computation as `ppl_parity.rs`'s `mean_ce`, but always
/// runs the plain (non-overridden) forward pass -- the exact `matvec_range`
/// reference this file measures divergence *from*, not against a captured
/// C-reference constant.
fn mean_ce_exact(model: &Model, val: &[u16]) -> f64 {
    let mut owned = ScratchOwned::new(&model.cfg);
    let mut s = owned.as_scratch();
    let seq = model.cfg.seq_len;
    let vocab = model.cfg.vocab;

    let mut nll = 0f64;
    let mut count = 0u64;
    for w in 0..WINDOWS {
        let base = w * seq;
        if base + seq + 1 > val.len() {
            break;
        }
        for pos in 0..seq {
            let tok = val[base + pos] as usize;
            llm_forward(model, tok, pos, &mut s);
            let target = val[base + pos + 1] as usize;
            nll += neg_log_prob(s.logits, vocab, target);
            count += 1;
        }
    }
    nll / count as f64
}

/// Same as `mean_ce_exact`, but every matvec is replaced by
/// `QT::matvec_range_chunked::<LANES>` -- the lane-reduced reordering a real
/// SIMD dot product would perform. `matvec_override` (per its doc comment in
/// `model.rs`) only covers the 8 non-head call sites, NOT the output head --
/// so a separate `head` closure is passed here too, doing the exact same
/// chunked matvec against `model.tok_emb()`, to get a full-model divergence
/// measurement rather than one that silently leaves the head on the exact
/// path.
fn mean_ce_chunked<const LANES: usize>(model: &Model, val: &[u16]) -> f64 {
    let mut owned = ScratchOwned::new(&model.cfg);
    let mut s = owned.as_scratch();
    let seq = model.cfg.seq_len;
    let vocab = model.cfg.vocab;

    let mut head = |x: &[f32], y: &mut [f32]| {
        let t = model.tok_emb();
        let rows = t.rows;
        t.matvec_range_chunked::<LANES>(x, y, 0, rows);
    };
    let mut matvec_override = |t: &QT, x: &[f32], y: &mut [f32]| {
        let rows = t.rows;
        t.matvec_range_chunked::<LANES>(x, y, 0, rows);
    };

    let mut nll = 0f64;
    let mut count = 0u64;
    for w in 0..WINDOWS {
        let base = w * seq;
        if base + seq + 1 > val.len() {
            break;
        }
        for pos in 0..seq {
            let tok = val[base + pos] as usize;
            llm_forward_with_matvec_override(
                model,
                tok,
                pos,
                &mut s,
                &mut head,
                &mut matvec_override,
            );
            let target = val[base + pos + 1] as usize;
            nll += neg_log_prob(s.logits, vocab, target);
            count += 1;
        }
    }
    nll / count as f64
}

/// Same idea as `mean_ce_chunked`, but exercises the actual algorithm
/// `llm-firmware`'s (draft, unverified-on-hardware) `matvec_simd.rs` uses:
/// dequantize each row whole via the public `QT::deq_row` (scale already
/// folded into each element), then one whole-row dot product against `x`,
/// instead of `matvec_range`'s per-group accumulate-then-scale. This is a
/// DIFFERENT reordering of the same underlying sum than
/// `matvec_range_chunked`'s per-group lanes -- see `matvec_simd.rs`'s
/// module doc comment for why that file reuses `deq_row` (only public API
/// `QT` exposes) instead of a lane-chunked approach, and why this
/// measurement matters on its own rather than assuming
/// `matvec_range_chunked`'s bound transfers.
fn mean_ce_deqrow_dot(model: &Model, val: &[u16]) -> f64 {
    let mut owned = ScratchOwned::new(&model.cfg);
    let mut s = owned.as_scratch();
    let seq = model.cfg.seq_len;
    let vocab = model.cfg.vocab;

    // Deliberately plain Vec-backed scratch, sized to this model's largest
    // `cols` value it will ever be called with (ple_proj's `ple_dim`) --
    // matches `matvec_simd.rs`'s `MatvecSimdScratch` in spirit (a reusable
    // buffer, sized once) without needing that crate's ESP32-only
    // dependencies just to exercise its algorithm on the host.
    let max_cols = model.cfg.dim.max(model.cfg.ffn).max(model.cfg.ple_dim);
    let mut scratch = vec![0f32; max_cols];

    let deqrow_dot = |t: &QT, x: &[f32], y: &mut [f32], scratch: &mut [f32]| {
        let cols = t.cols;
        for r in 0..t.rows {
            t.deq_row(r, &mut scratch[..cols]);
            let mut acc = 0f32;
            for (&a, &b) in scratch[..cols].iter().zip(x[..cols].iter()) {
                acc += a * b;
            }
            y[r] = acc;
        }
    };

    let mut head = |x: &[f32], y: &mut [f32]| {
        deqrow_dot(&model.tok_emb(), x, y, &mut scratch);
    };
    // Two closures both need `&mut scratch` at different times (never
    // simultaneously -- `llm_forward_impl` only ever has one of `head`/
    // `matvec_override` active per call site), so this can't borrow the
    // same `scratch` from both closures at once the way a single-purpose
    // buffer normally would. Two independent scratch buffers instead --
    // still just two heap allocations total for the whole run, not
    // per-call.
    let mut scratch2 = vec![0f32; max_cols];
    let mut matvec_override = |t: &QT, x: &[f32], y: &mut [f32]| {
        deqrow_dot(t, x, y, &mut scratch2);
    };

    let mut nll = 0f64;
    let mut count = 0u64;
    for w in 0..WINDOWS {
        let base = w * seq;
        if base + seq + 1 > val.len() {
            break;
        }
        for pos in 0..seq {
            let tok = val[base + pos] as usize;
            llm_forward_with_matvec_override(
                model,
                tok,
                pos,
                &mut s,
                &mut head,
                &mut matvec_override,
            );
            let target = val[base + pos + 1] as usize;
            nll += neg_log_prob(s.logits, vocab, target);
            count += 1;
        }
    }
    nll / count as f64
}

fn neg_log_prob(logits: &[f32], vocab: usize, target: usize) -> f64 {
    let mut mx = f32::NEG_INFINITY;
    for &lv in logits[..vocab].iter() {
        if lv > mx {
            mx = lv;
        }
    }
    let mut sum = 0f64;
    for &lv in logits[..vocab].iter() {
        sum += ((lv - mx) as f64).exp();
    }
    sum.ln() - (logits[target] - mx) as f64
}

/// The exact algorithm `llm-firmware`'s draft `matvec_simd.rs` uses
/// (`deq_row` + whole-row dot, see `mean_ce_deqrow_dot`'s doc comment) --
/// run with its own real measurement rather than assuming
/// `matvec_range_chunked`'s bound transfers to a different reordering.
///
/// Measured: exact CE = 2.667096968813, deq_row+dot CE = 2.667096996723,
/// drift = 0.000000027910 (~2.8e-8) -- same order of magnitude as, and
/// slightly smaller than, both `matvec_range_chunked` LANES results above.
/// Same 1e-5 bound and reasoning.
#[test]
fn matvec_deqrow_dot_within_measured_tolerance() {
    let (model, val) = load_fixtures();
    let exact = mean_ce_exact(&model, &val);
    let deqrow = mean_ce_deqrow_dot(&model, &val);
    let drift = (deqrow - exact).abs();
    eprintln!(
        "deq_row+dot: exact CE = {exact:.12}, deqrow CE = {deqrow:.12}, drift = {drift:.12}"
    );
    assert!(
        drift < 1e-5,
        "deq_row+whole-row-dot matvec diverged more than the measured float-reorder \
         tolerance allows: exact {exact:.12}, deqrow {deqrow:.12}, drift {drift:.12} \
         (bound 1e-5)"
    );
}

/// LANES=4: the width a 128-bit Xtensa PIE `f32` vector op actually holds
/// (128 bits / 32 bits per lane). This is the realistic value for a real
/// `esp-dsp` `dsps_dotprod_f32` call.
///
/// Measured (run via `cargo test --release --test matvec_simd_tolerance --
/// --nocapture`), full 4-window / 2048-prediction validation set: exact CE
/// = 2.667096968813, LANES=4 chunked CE = 2.667097012669, drift =
/// 0.000000043856 (~4.4e-8). That's over 45,000x smaller than the 0.002
/// noise floor `ppl_parity.rs` already tolerates for int8-activations'
/// reordering sensitivity -- splitting one 128-element group sum into 4
/// interleaved partial sums barely moves the result at all for this model's
/// weight/activation distribution. 1e-5 gives >200x headroom over the
/// observed 4.4e-8 while still catching a real regression (e.g. a
/// scale/group-boundary bug in a future SIMD implementation would produce a
/// jump many orders of magnitude larger than reorder noise, not a
/// last-few-bits difference).
#[test]
fn matvec_range_chunked_lanes4_within_measured_tolerance() {
    let (model, val) = load_fixtures();
    let exact = mean_ce_exact(&model, &val);
    let chunked = mean_ce_chunked::<4>(&model, &val);
    let drift = (chunked - exact).abs();
    eprintln!(
        "LANES=4: exact CE = {exact:.12}, chunked CE = {chunked:.12}, drift = {drift:.12}"
    );
    assert!(
        drift < 1e-5,
        "LANES=4 chunked-reduction matvec diverged more than the measured \
         float-reorder tolerance allows: exact {exact:.12}, chunked {chunked:.12}, \
         drift {drift:.12} (bound 1e-5)"
    );
}

/// LANES=8: a wider grouping than any single 128-bit Xtensa vector op could
/// do in one instruction (would need two ops combined), included as a
/// second data point on how drift scales with lane count -- if `esp-dsp`
/// turns out to internally accumulate across a wider software-pipelined
/// stride than the raw 128-bit width, this bounds that case too instead of
/// only ever having measured the narrowest one.
///
/// Measured: exact CE = 2.667096968813, LANES=8 chunked CE =
/// 2.667097008994, drift = 0.000000040181 (~4.0e-8) -- essentially the same
/// order of magnitude as LANES=4, not a growing trend with lane count for
/// this model. Same 1e-5 bound and reasoning as LANES=4 above.
#[test]
fn matvec_range_chunked_lanes8_within_measured_tolerance() {
    let (model, val) = load_fixtures();
    let exact = mean_ce_exact(&model, &val);
    let chunked = mean_ce_chunked::<8>(&model, &val);
    let drift = (chunked - exact).abs();
    eprintln!(
        "LANES=8: exact CE = {exact:.12}, chunked CE = {chunked:.12}, drift = {drift:.12}"
    );
    assert!(
        drift < 1e-5,
        "LANES=8 chunked-reduction matvec diverged more than the measured \
         float-reorder tolerance allows: exact {exact:.12}, chunked {chunked:.12}, \
         drift {drift:.12} (bound 1e-5)"
    );
}
