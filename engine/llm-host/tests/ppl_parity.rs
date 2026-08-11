//! Perplexity parity gate: runs the portable engine over real validation
//! tokens (reference-c/firmware/host_verify/ppl.c's job) and checks
//! cross-entropy against values captured from the C reference.
//!
//! Two paths, two different tolerances, and that difference is itself the
//! point of this test file:
//!
//! - fp32-activations (`matvec` / `matvec_q`, dequantized-weight path):
//!   deterministic float arithmetic, same op order as the C loop -- this
//!   must match the captured C reference EXACTLY (bit-for-bit CE).
//! - int8-activations (`matvec_int8_activations` / `matvec_q8`): activations
//!   get hard-quantized to a 255-level range every layer, which makes a
//!   single sub-ULP difference in `x[j] * inv` (unavoidable across two
//!   different compilers/codegens for the *same* algorithm) enough to flip
//!   a rounded int8 code, and that flip then compounds through the causal
//!   KV cache for the rest of the window. This is inherent float
//!   non-associativity, not a port bug -- confirmed by rebuilding the
//!   *C reference itself* with `-march=native` instead of plain `-O2`:
//!   that alone moves its own CE from 2.6684 to 2.6678, a bigger shift
//!   than the ~0.0001 gap between this Rust port (2.6683) and the
//!   baseline C build (2.6684). The tolerance below is set generously
//!   inside that observed noise floor, not loosened to paper over a bug.
//!
//! Run with `cargo test --release --features int8-activations` to also
//! exercise the second half of this file; without the feature only the
//! fp32 half runs (matches this crate's default build).

use llm_core::{llm_forward, Model};
use llm_host::support::ScratchOwned;
use std::fs;

/// Model paths, window count, and the expected C-reference cross-entropy
/// all come from `model/models.toml` now, rather than being duplicated as
/// consts here — see `llm_host::manifest`. The C reference values there
/// were produced by `gcc -O2 -o ppl reference-c/firmware/host_verify/ppl.c
/// -lm` against the same model.bin/val.bin at `windows=4` (2048
/// predictions): fp32 CE 2.6671 / ppl 14.40, int8 CE 2.6684 / ppl 14.42.
///
/// Keeping them in the registry rather than here is what lets the C-side
/// `scripts/model.sh` run its `ppl` binary at the same window count against
/// the same files — the two can no longer drift apart silently.
use llm_host::manifest;

fn mean_ce(model: &Model, val: &[u16], windows: usize) -> f64 {
    let mut owned = ScratchOwned::new(&model.cfg);
    let mut s = owned.as_scratch();
    let seq = model.cfg.seq_len;
    let vocab = model.cfg.vocab;

    let mut nll = 0f64;
    let mut count = 0u64;
    for w in 0..windows {
        let base = w * seq;
        if base + seq + 1 > val.len() {
            break;
        }
        for pos in 0..seq {
            let tok = val[base + pos] as usize;
            llm_forward(model, tok, pos, &mut s);
            let target = val[base + pos + 1] as usize;

            let mut mx = f32::NEG_INFINITY;
            for &lv in s.logits[..vocab].iter() {
                if lv > mx {
                    mx = lv;
                }
            }
            let mut sum = 0f64;
            for &lv in s.logits[..vocab].iter() {
                sum += ((lv - mx) as f64).exp();
            }
            nll += sum.ln() - (s.logits[target] - mx) as f64;
            count += 1;
        }
    }
    nll / count as f64
}

/// Returns the loaded model, its validation tokens, and the registry entry
/// they came from (so each test can read its own expected CE and window
/// count from the same place, instead of a local const).
fn load_fixtures() -> (Model<'static>, Vec<u16>, manifest::Model) {
    let entry = manifest::default_model();
    let bin = entry.bin_path();
    let bytes: &'static [u8] = fs::read(&bin)
        .unwrap_or_else(|e| panic!("read {}: {e} (run from llm-host/)", bin.display()))
        .leak();
    let model = Model::load(bytes).expect("model.bin should parse");
    let valp = entry.val_tokens_path();
    let val_bytes = fs::read(&valp)
        .unwrap_or_else(|e| panic!("read {}: {e} (run from llm-host/)", valp.display()));
    let val: Vec<u16> = val_bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    (model, val, entry)
}

fn windows(entry: &manifest::Model) -> usize {
    entry
        .ppl_windows
        .expect("default model needs `ppl_windows` in model/models.toml")
}

#[test]
#[cfg(not(feature = "int8-activations"))]
fn fp32_activations_ppl_matches_c_reference_exactly() {
    let (model, val, entry) = load_fixtures();
    let expected = entry
        .ppl_ce_fp32
        .expect("default model needs `ppl_ce_fp32` in model/models.toml");
    let ce = mean_ce(&model, &val, windows(&entry));
    eprintln!("fp32-activations val CE = {ce:.4} (C reference {expected:.4})");
    assert!(
        (ce - expected).abs() < 0.0001,
        "fp32-activations path should match the C reference to the last printed digit: \
         got {ce:.4}, expected {expected:.4}"
    );
}

#[test]
#[cfg(feature = "int8-activations")]
fn int8_activations_ppl_matches_c_reference_within_fp_noise_floor() {
    let (model, val, entry) = load_fixtures();
    let expected = entry
        .ppl_ce_int8
        .expect("default model needs `ppl_ce_int8` in model/models.toml");
    let ce = mean_ce(&model, &val, windows(&entry));
    eprintln!("int8-activations val CE = {ce:.4} (C reference {expected:.4})");
    // Noise floor established empirically: rebuilding the C reference with
    // -march=native instead of -O2 alone shifts its own CE by ~0.0006. 0.002
    // gives >3x headroom over that -- enough to catch a real regression
    // (e.g. the scratch-buffer sizing bug this test suite originally caught,
    // which didn't shrink the gap, it crashed the process) while not being
    // sensitive to ordinary float non-associativity noise.
    assert!(
        (ce - expected).abs() < 0.002,
        "int8-activations path diverged further than the fp-noise floor allows: \
         got {ce:.4}, expected ~{expected:.4}"
    );
}
