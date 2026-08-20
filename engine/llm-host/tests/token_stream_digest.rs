//! Gate for the firmware's on-device token stream.
//!
//! WHAT WAS MISSING. Every device figure in `benchmarks/device.toml` is
//! reported alongside "byte-identical output", and until this file nothing
//! checked it. The host parity suite is thorough — `golden.rs` against
//! PyTorch, `cli_parity.rs` byte-for-byte against the compiled C binary,
//! `ppl_parity.rs` on exact fp32 cross-entropy — but none of it runs the
//! firmware's code path, and `cli_parity.rs` in particular uses a *different*
//! model and `N=120`, not the benchmark settings the board runs. The only
//! thing comparing the board's 400 tokens to anything was a person reading the
//! story and deciding it looked the same.
//!
//! WHAT THIS DOES. It replays exactly what `llm-firmware`'s `main` does —
//! prime on `prompt_ids`, then `n_generate` steps of greedy decode with a
//! `vocab.n`-bounded argmax — through the plain `llm_forward`, and folds every
//! emitted id into `llm_core::TokenDigest`. The result is pinned in
//! `model/models.toml` as `gen_digest`. The firmware prints the same number at
//! the end of a run, so "byte-identical" becomes one value to compare instead
//! of 400 tokens of prose to eyeball.
//!
//! WHAT IT DOES AND DOES NOT PROVE. It pins the *reference* side: the digest
//! is what this engine's ordinary, parity-tested path produces on the
//! benchmark settings, and this test fails if any change moves it. Combined
//! with the board printing the same value, it says the firmware's int4 head,
//! dual-core matvec, per-core attention heads and two-candidate argmax all
//! land on the same tokens as the single-core reference.
//!
//! It does not, by itself, tie those 400 tokens to the C firmware — no
//! reference capture of C's own 400-token run exists in this repo. Closing
//! that would mean either capturing one, or teaching the C firmware the same
//! digest. Worth doing; this is the part that can be done without hardware.
//!
//! ON UPDATING THE EXPECTED VALUE. If this fails after a change you believe is
//! correct, the digest moving *is* the finding: this path is supposed to be
//! bit-exact, so a legitimate reason to edit `gen_digest` is close to
//! nonexistent. Establish why the tokens changed before touching it.

// The firmware does not enable this feature -- like the C sketch, it never
// defines `LLM_INT8_ACT`, so every tensor except the output head runs the
// plain fp32 path. Quantizing every activation *does* move this digest (it
// flips at least one argmax over 400 tokens, which `ppl_parity.rs`'s
// tolerance already implies is possible), so asserting the firmware's value
// under a configuration the firmware never runs would be asserting a
// coincidence. Skipped rather than given a second expected value, because a
// second value would imply someone is checking it against something.
#![cfg(not(feature = "int8-activations"))]

use llm_core::{
    activation_sum, dot_int4_int8, llm_forward_with_head_override, quantize_activations, Model,
    TokenDigest,
};
use llm_host::manifest;
use llm_host::support::ScratchOwned;
use llm_host::vocab::Vocab;

/// The firmware's greedy pick, reproduced: scan `0..vocab_n` with a strict
/// `>`, so the lowest index wins a tie.
///
/// Bounded by the vocab entry count, NOT by `cfg.out_vocab`. For a PLE1 file
/// `out_vocab == vocab == 32,768`, while the tokenizer only ever learned
/// 25,353 entries; the C reference, `host_generate`, and the firmware all
/// bound the argmax to the latter. Scanning the padding rows here would score
/// tokens that cannot be emitted and could pick one.
fn argmax(logits: &[f32], vocab_n: usize) -> usize {
    let mut best = 0usize;
    let mut best_v = logits[0];
    for v in 1..vocab_n {
        if logits[v] > best_v {
            best_v = logits[v];
            best = v;
        }
    }
    best
}

#[test]
fn the_benchmark_run_has_the_expected_token_digest() {
    let m = manifest::default_model();
    let expected = m
        .gen_digest
        .expect("models.toml: default model needs a gen_digest");
    let n_generate = m
        .n_generate
        .expect("models.toml: default model needs n_generate");
    assert!(
        !m.prompt_ids.is_empty(),
        "models.toml: default model needs prompt_ids"
    );

    let bytes = std::fs::read(m.bin_path()).expect("model.bin");
    let model = Model::load(&bytes).expect("load");
    let vocab = Vocab::load(m.vocab_path().to_str().expect("utf-8 path")).expect("vocab.h");

    // THE OUTPUT HEAD MUST BE THE FIRMWARE'S, NOT THE DEFAULT ONE.
    //
    // This is the part that makes the digest mean something. `llm_forward`
    // runs the tied embedding through the ordinary fp32 `matvec_range`;
    // `llm-firmware` instead quantizes the activation to int8 and dots it
    // against the packed int4 codes (`head.rs::rows_range`), exactly as the C
    // firmware does. Those are different arithmetic and give different
    // logits, so a reference built on the fp32 head would be a number the
    // board has no reason to reproduce -- and the first mismatch would have
    // been blamed on the dual-core split.
    //
    // `llm_forward_with_head_override` is the hook the firmware itself uses.
    // Reproducing `rows_range` through it here is what makes this a reference
    // for the device rather than for the host.
    //
    // Not reproduced, deliberately: the split of those rows across two cores.
    // That is bit-exact and separately gated by `matvec_split.rs` and
    // `attn_split.rs`, and no host test can run FreeRTOS anyway.
    let head_t = model.tok_emb();
    assert_eq!(
        head_t.n_groups, 1,
        "the firmware's head path assumes one scale per row"
    );
    let head_cols = head_t.cols;
    let mut actq = vec![0i8; head_cols];
    let mut head = |x: &[f32], y: &mut [f32]| {
        let acts = quantize_activations(&x[..head_cols], &mut actq);
        let asum = activation_sum(&actq);
        for (r, slot) in y.iter_mut().enumerate().take(vocab.n) {
            let (codes, row_scale) = head_t.packed_row(r);
            *slot = dot_int4_int8(codes, &actq, asum) as f32 * row_scale * acts;
        }
    };

    let mut owned = ScratchOwned::new(&model.cfg);
    let mut s = owned.as_scratch();
    let mut digest = TokenDigest::new();
    let mut pos = 0usize;
    let mut tok;

    // Prime. The firmware emits each prompt id as it feeds it, so they are
    // part of the stream the digest covers.
    for &t in &m.prompt_ids {
        tok = t;
        digest.push(tok);
        llm_forward_with_head_override(&model, tok, pos, &mut s, &mut head);
        pos += 1;
    }

    // Generate. Same loop shape and same `pos >= seq_len` guard as the
    // firmware, so a model whose prompt plus generation overruns the context
    // stops at the same place on both sides.
    for _ in 0..n_generate {
        if pos >= model.cfg.seq_len {
            break;
        }
        tok = argmax(s.logits, vocab.n);
        digest.push(tok);
        llm_forward_with_head_override(&model, tok, pos, &mut s, &mut head);
        pos += 1;
    }

    assert_eq!(
        digest.count() as usize,
        m.prompt_ids.len() + n_generate,
        "token count changed: the run stopped early or the loop shape drifted \
         from the firmware's"
    );
    assert_eq!(
        digest.value(),
        expected,
        "token stream digest is 0x{:016x}, models.toml says 0x{expected:016x}. \
         This path is bit-exact by design, so the tokens moving is the finding \
         -- do not update models.toml until you know why.",
        digest.value()
    );
}
