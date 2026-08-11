//! Wiring test for `llm_core::llm_forward_with_head_override` -- the hook
//! `llm-firmware` (Phase 3) uses to replace the output-head matvec with its
//! dual-core int8-staged version (`Model.head_matvec` in the C code).
//!
//! This crate has no access to `Model`'s private `tok_emb` field, so it
//! can't replicate the real staged-head computation here (that lives in
//! `llm-firmware`, which can). What this test *can* prove, without any
//! ESP32 hardware: the override closure actually gets called instead of
//! the default head matvec, it receives `x`/`y` at the shapes
//! `llm_forward` documents (`x.len() == cfg.dim`, `y.len() == cfg.vocab`),
//! and whatever it writes to `y` is exactly what ends up in `s.logits` --
//! i.e. the plumbing `llm-firmware` depends on is correct, independent of
//! what the eventual firmware head implementation does with it.

use llm_core::{llm_forward_with_head_override, Model};
use llm_host::support::ScratchOwned;
use std::fs;

#[test]
fn head_override_receives_correct_shapes_and_its_output_becomes_logits() {
    let bin = llm_host::manifest::default_model().bin_path();
    let bytes = fs::read(&bin)
        .unwrap_or_else(|e| panic!("read {}: {e} (run from llm-host/)", bin.display()));
    let model = Model::load(&bytes).expect("model.bin should parse");

    let mut owned = ScratchOwned::new(&model.cfg);
    let mut s = owned.as_scratch();

    let mut calls = 0usize;
    let mut seen_x_len = 0usize;
    let mut seen_y_len = 0usize;
    let mut head = |x: &[f32], y: &mut [f32]| {
        calls += 1;
        seen_x_len = x.len();
        seen_y_len = y.len();
        // A pattern that could never come from the real head matvec, so
        // finding it in s.logits afterward proves this closure -- not the
        // default path -- produced it.
        for (i, v) in y.iter_mut().enumerate() {
            *v = 1000.0 + i as f32;
        }
    };

    llm_forward_with_head_override(&model, /* token */ 0, /* pos */ 0, &mut s, &mut head);

    assert_eq!(calls, 1, "override should run exactly once per llm_forward call");
    assert_eq!(seen_x_len, model.cfg.dim, "head override's x should be the D-dim pre-head activation");
    assert_eq!(seen_y_len, model.cfg.vocab, "head override's y should be the full vocab-size logit buffer");

    for i in 0..model.cfg.vocab {
        assert_eq!(
            s.logits[i],
            1000.0 + i as f32,
            "s.logits[{i}] doesn't match what the override wrote -- the hook isn't wired to the real output"
        );
    }
}
