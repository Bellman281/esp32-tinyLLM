//! What an fp16 KV cache would cost, measured before it is built.
//!
//! WHY. The dual-core bandwidth probe
//! (`llm-firmware`'s `Head::probe_weight_bandwidth_dual`) found that a second
//! LX7 buys only **1.27x** more PSRAM bandwidth, not 2x: 72 MB/s with one core
//! reading, 91 MB/s aggregate with two. Against that ceiling, attention's
//! `core` sub-stage reads the whole KV cache every token — at the benchmark
//! settings that is `219 positions x 96 floats x 4 bytes x 2 x 6 layers`
//! ≈ **1.01 MB**, nearly as much as the entire output head — so
//!
//! ```text
//! attn core = 11.1 ms of transfer + 4.8 ms of everything else = 15.9 ms
//! ```
//!
//! **70% of the stage is moving bytes**, against 21% for the head. Halving the
//! cache is the largest win left in the engine: ~5.5 ms/token, ~5% of a token.
//!
//! WHY MEASURE FIRST. It would also be the first change in this port that is
//! not bit-exact. Every one of the twelve device-measured speedups so far has
//! been exact, `golden.rs`/`cli_parity.rs` hold the engine to the C reference
//! with no tolerance, and the firmware now prints a digest of its own token
//! stream. Trading that away is a real decision, and it should be made against
//! a number rather than against "fp16 is probably fine" —
//! `matvec_simd_tolerance.rs` set that precedent for SIMD reordering, and the
//! answer there (~3–4e-8 CE, 45,000x inside the noise floor) is the only
//! reason that hook was safe to ship.
//!
//! WHAT IS SIMULATED. `llm_forward_fp16_kv` rounds every k/v through `f16` on
//! the way into the cache and leaves the buffers `f32`. That reproduces an
//! `f16` cache's arithmetic *exactly* — a real one rounds on store, widens on
//! load, and runs the same `f32` dot products over the same values — while
//! changing no buffer types, so the experiment carries none of the rewrite's
//! risk. It does not simulate the speed; that is a device property and the
//! probe already bounds it.
//!
//! These tests report rather than gate. There is no threshold to assert
//! against until someone decides what this engine is willing to spend.

// Gated off for the same reason `token_stream_digest.rs` is: this feature
// quantizes every activation, not just the head's, so the fp32-KV control arm
// no longer reproduces `gen_digest` and the comparison would be measuring two
// changes at once. The firmware never enables it.
#![cfg(not(feature = "int8-activations"))]

use llm_core::{
    activation_sum, dot_int4_int8, llm_forward, llm_forward_fp16_kv, quantize_activations, Model,
    TokenDigest,
};
use llm_host::manifest;
use llm_host::support::ScratchOwned;
use llm_host::vocab::Vocab;

fn load() -> (Vec<u8>, manifest::Model) {
    let entry = manifest::default_model();
    (std::fs::read(entry.bin_path()).expect("model.bin"), entry)
}

/// Greedy decode over the benchmark prompt/length, bounded to the real vocab
/// exactly as the firmware and the C reference are. Returns every emitted id.
///
/// NOTE ON THE HEAD. This runs the *default* fp32 output head, not the
/// int8-activation/int4-weight head the firmware uses. Both are reported,
/// because they can disagree about where an argmax flips and only one of them
/// is what a board would do — see `report_fp16_kv_on_the_firmware_head`.
fn generate(model: &Model, entry: &manifest::Model, vocab_n: usize, fp16: bool) -> Vec<usize> {
    let n_generate = entry.n_generate.expect("n_generate");
    let mut owned = ScratchOwned::new(&model.cfg);
    let mut s = owned.as_scratch();
    let mut out = Vec::with_capacity(entry.prompt_ids.len() + n_generate);
    let mut pos = 0usize;

    let step = |s: &mut llm_core::Scratch, tok: usize, pos: usize| {
        if fp16 {
            llm_forward_fp16_kv(model, tok, pos, s)
        } else {
            llm_forward(model, tok, pos, s)
        }
    };

    for &t in &entry.prompt_ids {
        out.push(t);
        step(&mut s, t, pos);
        pos += 1;
    }
    for _ in 0..n_generate {
        if pos >= model.cfg.seq_len {
            break;
        }
        let mut best = 0usize;
        let mut best_v = s.logits[0];
        for v in 1..vocab_n {
            if s.logits[v] > best_v {
                best_v = s.logits[v];
                best = v;
            }
        }
        out.push(best);
        step(&mut s, best, pos);
        pos += 1;
    }
    out
}

/// How far into the run the two paths agree, and how much they diverge.
///
/// The number that matters is not "how many tokens differ" but "where does the
/// first one differ" — greedy decode is a feedback loop, so one flipped argmax
/// at step `n` makes everything after it a different sequence, not a slightly
/// wrong one. A late first divergence and a large final difference is the
/// expected shape; an early one is what would make fp16 unusable.
#[test]
fn report_fp16_kv_token_divergence() {
    let (bytes, entry) = load();
    let model = Model::load(&bytes).expect("load");
    let vocab = Vocab::load(entry.vocab_path().to_str().expect("utf-8")).expect("vocab.h");

    let base = generate(&model, &entry, vocab.n, false);
    let fp16 = generate(&model, &entry, vocab.n, true);
    assert_eq!(base.len(), fp16.len(), "both arms must run the same length");

    let first = base.iter().zip(&fp16).position(|(a, b)| a != b);
    let differing = base.iter().zip(&fp16).filter(|(a, b)| a != b).count();

    let mut d = TokenDigest::new();
    for &t in &fp16 {
        d.push(t);
    }

    eprintln!("\n--- fp16 KV cache, token stream ---");
    eprintln!("  tokens                {}", base.len());
    eprintln!("  prompt (identical)    {}", entry.prompt_ids.len());
    match first {
        None => eprintln!("  first divergence      none — streams are identical"),
        Some(i) => eprintln!(
            "  first divergence      step {i} (generated token {})",
            i as isize - entry.prompt_ids.len() as isize
        ),
    }
    eprintln!(
        "  differing tokens      {differing} of {} ({:.1}%)",
        base.len(),
        100.0 * differing as f64 / base.len() as f64
    );
    eprintln!("  fp16 digest           0x{:016x}", d.value());
    eprintln!("  (a new gen_digest would be needed if this ever shipped)\n");
}

/// Cross-entropy over the real validation set — the measure that says whether
/// the *model* got worse, as opposed to whether one greedy path forked.
///
/// This is the number to weigh against `ppl_ce_fp32 = 2.6671` and
/// `ppl_ce_int8 = 2.6684` in the registry. The gap between those two is the
/// noise floor this project already accepts on the int8-activation path, so it
/// is the natural yardstick: an fp16-KV delta well inside it is a change of
/// the same order as one already shipping, and a delta well outside it is a
/// genuinely worse model.
#[test]
fn report_fp16_kv_cross_entropy() {
    let (bytes, entry) = load();
    let model = Model::load(&bytes).expect("load");
    let val_raw = std::fs::read(entry.val_tokens_path()).expect("val.bin");
    let val: Vec<u16> = val_raw
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    let windows = entry.ppl_windows.expect("ppl_windows");

    let ce = |fp16: bool| -> f64 {
        let mut owned = ScratchOwned::new(&model.cfg);
        let mut s = owned.as_scratch();
        let seq = model.cfg.seq_len;
        let vocab = model.cfg.vocab;
        let (mut nll, mut count) = (0f64, 0u64);
        for w in 0..windows {
            let base = w * seq;
            if base + seq + 1 > val.len() {
                break;
            }
            for pos in 0..seq {
                let tok = val[base + pos] as usize;
                if fp16 {
                    llm_forward_fp16_kv(&model, tok, pos, &mut s);
                } else {
                    llm_forward(&model, tok, pos, &mut s);
                }
                let target = val[base + pos + 1] as usize;
                let mx = s.logits[..vocab]
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max);
                let sum: f64 = s.logits[..vocab]
                    .iter()
                    .map(|&lv| ((lv - mx) as f64).exp())
                    .sum();
                nll += sum.ln() - (s.logits[target] - mx) as f64;
                count += 1;
            }
        }
        nll / count as f64
    };

    let base = ce(false);
    let f16 = ce(true);
    let int8_floor = (entry.ppl_ce_int8.unwrap_or(0.0) - entry.ppl_ce_fp32.unwrap_or(0.0)).abs();

    eprintln!("\n--- fp16 KV cache, validation cross-entropy ---");
    eprintln!("  fp32 KV (shipping)    {base:.6}");
    eprintln!("  fp16 KV               {f16:.6}");
    eprintln!("  delta                 {:+.6}", f16 - base);
    eprintln!("  int8-activation floor {int8_floor:.6}  (registry ppl_ce_int8 - ppl_ce_fp32)");
    eprintln!(
        "  ratio to that floor   {:.2}x\n",
        if int8_floor > 0.0 {
            (f16 - base).abs() / int8_floor
        } else {
            f64::NAN
        }
    );
}

/// The same divergence question, through the head the firmware actually runs.
///
/// `generate` above uses the default fp32 output head. `llm-firmware` does
/// not: it quantizes the activation to int8 and dots it against the packed
/// int4 codes, which is different arithmetic and different logits, and the
/// argmax it produces can flip at a different place. `token_stream_digest.rs`
/// exists because of exactly that distinction; the fp16-KV question inherits
/// it.
///
/// The fp32-KV arm here must reproduce `models.toml`'s `gen_digest`. That is
/// the control: if it does not, this test is measuring something other than
/// what the board runs, and its answer about fp16 means nothing.
#[test]
fn report_fp16_kv_on_the_firmware_head() {
    let (bytes, entry) = load();
    let model = Model::load(&bytes).expect("load");
    let vocab = Vocab::load(entry.vocab_path().to_str().expect("utf-8")).expect("vocab.h");
    let n_generate = entry.n_generate.expect("n_generate");
    let head_t = model.tok_emb();
    let cols = head_t.cols;

    let run = |fp16: bool| -> Vec<usize> {
        let mut owned = ScratchOwned::new(&model.cfg);
        let mut s = owned.as_scratch();
        let mut actq = vec![0i8; cols];
        // Ports `head.rs::rows_range` exactly, same as token_stream_digest.rs.
        let mut head = |x: &[f32], y: &mut [f32]| {
            let acts = quantize_activations(&x[..cols], &mut actq);
            let asum = activation_sum(&actq);
            for (r, slot) in y.iter_mut().enumerate().take(vocab.n) {
                let (codes, row_scale) = head_t.packed_row(r);
                *slot = dot_int4_int8(codes, &actq, asum) as f32 * row_scale * acts;
            }
        };
        let mut out = Vec::with_capacity(entry.prompt_ids.len() + n_generate);
        let mut pos = 0usize;
        for &t in &entry.prompt_ids {
            out.push(t);
            if fp16 {
                llm_core::llm_forward_fp16_kv_with_head_override(&model, t, pos, &mut s, &mut head);
            } else {
                llm_core::llm_forward_with_head_override(&model, t, pos, &mut s, &mut head);
            }
            pos += 1;
        }
        for _ in 0..n_generate {
            if pos >= model.cfg.seq_len {
                break;
            }
            let mut best = 0usize;
            let mut best_v = s.logits[0];
            for v in 1..vocab.n {
                if s.logits[v] > best_v {
                    best_v = s.logits[v];
                    best = v;
                }
            }
            out.push(best);
            if fp16 {
                llm_core::llm_forward_fp16_kv_with_head_override(
                    &model, best, pos, &mut s, &mut head,
                );
            } else {
                llm_core::llm_forward_with_head_override(&model, best, pos, &mut s, &mut head);
            }
            pos += 1;
        }
        out
    };

    let digest_of = |ts: &[usize]| {
        let mut d = TokenDigest::new();
        for &t in ts {
            d.push(t);
        }
        d.value()
    };

    let base = run(false);
    let f16 = run(true);
    let expected = entry.gen_digest.expect("gen_digest");
    let base_d = digest_of(&base);
    assert_eq!(
        base_d, expected,
        "control failed: the fp32-KV arm must reproduce models.toml's gen_digest \
         (got 0x{base_d:016x}, expected 0x{expected:016x}). Until it does, this \
         test is not measuring the firmware's path."
    );

    let first = base.iter().zip(&f16).position(|(a, b)| a != b);
    let differing = base.iter().zip(&f16).filter(|(a, b)| a != b).count();
    eprintln!("\n--- fp16 KV cache, FIRMWARE head (int8 act x int4 weights) ---");
    eprintln!("  control: fp32-KV digest matches models.toml gen_digest");
    match first {
        None => eprintln!("  first divergence      none -- streams are identical"),
        Some(i) => eprintln!(
            "  first divergence      step {i} of {} (generated token {})",
            base.len(),
            i as isize - entry.prompt_ids.len() as isize
        ),
    }
    eprintln!(
        "  differing tokens      {differing} of {} ({:.1}%)",
        base.len(),
        100.0 * differing as f64 / base.len() as f64
    );
    // The percentage on its own reads as "62% of tokens are wrong", which is
    // the wrong picture. Greedy decode is a feedback loop: one flipped argmax
    // makes every later position a *different continuation*, not a corrupted
    // one. Reporting how much of the tail differs separates "one fork" from
    // "many independent errors" -- they call for opposite conclusions.
    if let Some(i) = first {
        let tail = base.len() - i;
        let tail_diff = base[i..]
            .iter()
            .zip(&f16[i..])
            .filter(|(a, b)| a != b)
            .count();
        eprintln!(
            "  after the fork        {tail_diff} of {tail} differ ({:.0}%) -- \
             {}",
            100.0 * tail_diff as f64 / tail as f64,
            if tail_diff as f64 / tail as f64 > 0.9 {
                "no re-convergence, so this is ONE flipped argmax and a different\n\
                 \x20                       continuation, not many independent errors"
            } else {
                "partial re-convergence, so the streams keep meeting again"
            }
        );
    }
    eprintln!("  fp16 digest           0x{:016x}\n", digest_of(&f16));
}
