//! Port of reference-c/firmware/host_verify/verify.c: the port's actual
//! correctness gate. Runs the Rust engine over the same fixed golden
//! prompt PyTorch's export.py used, and checks the last-position logits
//! against PyTorch's reference to the same tolerance the C code requires
//! (max abs diff < 0.02). If this test passes, the Rust port is numerically
//! equivalent to both the C reference and the original PyTorch model.
//!
//! That 0.02 tolerance is a property of the fp32-dequantized-weight path
//! only (`matvec_q` / plain `matvec`) -- golden.txt was dumped from
//! PyTorch's full-precision forward pass, and int8-activation quantization
//! is a genuinely lossy approximation of it by construction. Proof: the
//! *C reference itself*, rebuilt with `-DLLM_INT8_ACT` and run against this
//! exact model.bin/golden.txt, also fails this check --
//! `max abs diff = 0.13001   rms diff = 0.029788` -- so a fp32-tolerance
//! failure under int8-activations would be expected, not a regression.
//! The second test below checks the int8-activations build against THAT
//! C measurement instead (tight tolerance -- see its own doc comment).

use llm_core::{llm_forward, Model};
use llm_host::support::ScratchOwned;
use std::fs;

/// This test deliberately does NOT use the registry's default model. It is
/// the only direct PyTorch parity gate in the project, and a golden-logit
/// check needs a `golden.txt` — which exists only for this smaller fixture
/// model, not for the 28.9M-param model everything else uses. Named
/// explicitly (rather than read from `default`) so that pointing `default`
/// at a new model can never silently move this check onto a model that has
/// no golden reference. See `model/models.toml`.
const FIXTURE_MODEL: &str = "golden-v4096";

#[cfg(not(feature = "int8-activations"))]
const TOLERANCE: f32 = 0.02;

#[test]
#[cfg(not(feature = "int8-activations"))]
fn golden_logits_match_pytorch() {
    let fixture = llm_host::manifest::model(FIXTURE_MODEL);
    let (bin, gold) = (fixture.bin_path(), fixture.golden_path());
    let bytes = fs::read(&bin)
        .unwrap_or_else(|e| panic!("read {}: {e} (run from llm-host/)", bin.display()));
    let model = Model::load(&bytes).expect("model.bin should parse (bad magic?)");

    let gold_text =
        fs::read_to_string(&gold).unwrap_or_else(|e| panic!("read {}: {e}", gold.display()));
    let mut lines = gold_text.lines();

    let plen: usize = lines
        .next()
        .expect("golden.txt line 1: prompt length")
        .trim()
        .parse()
        .expect("prompt length should be an integer");
    let prompt: Vec<usize> = lines
        .next()
        .expect("golden.txt line 2: prompt ids")
        .split_whitespace()
        .map(|s| s.parse().expect("prompt id should be an integer"))
        .collect();
    assert_eq!(
        prompt.len(),
        plen,
        "golden.txt's declared prompt length doesn't match the id list"
    );

    let reference: Vec<f32> = lines
        .map(|l| l.trim().parse().expect("logit line should be a float"))
        .collect();
    assert_eq!(
        reference.len(),
        model.cfg.vocab,
        "golden.txt has a different vocab size than model.bin's header"
    );

    let mut owned = ScratchOwned::new(&model.cfg);
    let mut s = owned.as_scratch();
    for (pos, &tok) in prompt.iter().enumerate() {
        llm_forward(&model, tok, pos, &mut s);
    }

    let mut max_abs = 0f32;
    let mut sum_sq = 0f64;
    for i in 0..model.cfg.vocab {
        let d = s.logits[i] - reference[i];
        if d.abs() > max_abs {
            max_abs = d.abs();
        }
        sum_sq += (d as f64) * (d as f64);
    }
    let rms = (sum_sq / model.cfg.vocab as f64).sqrt();
    eprintln!("max abs diff = {max_abs:.5}   rms diff = {rms:.6}   (tolerance {TOLERANCE})");

    let mut c_top = 0usize;
    let mut r_top = 0usize;
    for i in 0..model.cfg.vocab {
        if s.logits[i] > s.logits[c_top] {
            c_top = i;
        }
        if reference[i] > reference[r_top] {
            r_top = i;
        }
    }
    eprintln!("logits: Rust top={c_top}  PyTorch top={r_top}");

    assert!(
        max_abs < TOLERANCE,
        "Rust port diverges from the PyTorch golden reference: max abs diff {max_abs} >= {TOLERANCE}"
    );
    assert_eq!(
        c_top, r_top,
        "Rust and PyTorch disagree on the argmax token"
    );
}

/// int8-activations counterpart. Not "does this match PyTorch fp32" (see
/// the module doc comment for why that's the wrong question here) but
/// "does this match the C reference's OWN int8-activations numerics" --
/// measured once via:
///   gcc -O2 -DLLM_INT8_ACT -o verify reference-c/host_tools/verify.c -lm
///   ./verify reference-c/model/model_1p87mb.bin reference-c/model/golden.txt
///   -> max abs diff = 0.13001   rms diff = 0.029788
/// A tight tolerance here (not the 0.02 golden tolerance) is deliberate:
/// this is checking cross-language agreement on the exact same lossy
/// algorithm, not agreement with an independent fp32 ground truth, so it
/// should be nearly exact modulo ordinary float non-associativity.
#[test]
#[cfg(feature = "int8-activations")]
fn golden_logits_int8_activations_matches_c_int8_numerics() {
    const C_INT8_MAX_ABS: f32 = 0.13001;
    const C_INT8_RMS: f64 = 0.029788;
    const NOISE_TOLERANCE: f32 = 0.001;

    let fixture = llm_host::manifest::model(FIXTURE_MODEL);
    let (bin, gold) = (fixture.bin_path(), fixture.golden_path());
    let bytes = fs::read(&bin)
        .unwrap_or_else(|e| panic!("read {}: {e} (run from llm-host/)", bin.display()));
    let model = Model::load(&bytes).expect("model.bin should parse (bad magic?)");

    let gold_text =
        fs::read_to_string(&gold).unwrap_or_else(|e| panic!("read {}: {e}", gold.display()));
    let mut lines = gold_text.lines();
    let plen: usize = lines.next().unwrap().trim().parse().unwrap();
    let prompt: Vec<usize> = lines
        .next()
        .unwrap()
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    assert_eq!(prompt.len(), plen);
    let reference: Vec<f32> = lines.map(|l| l.trim().parse().unwrap()).collect();

    let mut owned = ScratchOwned::new(&model.cfg);
    let mut s = owned.as_scratch();
    for (pos, &tok) in prompt.iter().enumerate() {
        llm_forward(&model, tok, pos, &mut s);
    }

    let mut max_abs = 0f32;
    let mut sum_sq = 0f64;
    for i in 0..model.cfg.vocab {
        let d = s.logits[i] - reference[i];
        max_abs = max_abs.max(d.abs());
        sum_sq += (d as f64) * (d as f64);
    }
    let rms = (sum_sq / model.cfg.vocab as f64).sqrt();
    eprintln!(
        "int8-activations: max abs diff = {max_abs:.5} (C: {C_INT8_MAX_ABS})   \
         rms diff = {rms:.6} (C: {C_INT8_RMS})"
    );

    assert!(
        (max_abs - C_INT8_MAX_ABS).abs() < NOISE_TOLERANCE,
        "int8-activations max abs diff {max_abs} diverged from the C reference's own \
         int8 numerics {C_INT8_MAX_ABS} by more than the fp noise floor"
    );
    assert!(
        (rms - C_INT8_RMS).abs() < NOISE_TOLERANCE as f64,
        "int8-activations rms diff {rms} diverged from the C reference's own int8 \
         numerics {C_INT8_RMS} by more than the fp noise floor"
    );
}
