//! Proves `QT::unpack_row_int8` -- the int4->int8 staging step
//! `llm-firmware`'s dual-core output head uses at boot (mirrors the C
//! reference's `stage_head_int8` in `firmware/esp32_llm/esp32_llm.ino`) --
//! reconstructs exactly the same per-row values as the already-verified
//! `deq_row` fp32 dequant path.
//!
//! `unpack_row_int8` and `deq_row` both read the same private `code`/
//! `scale` methods on `QT`, so this is mostly a guard against a future
//! refactor breaking that equivalence -- but it's cheap insurance for code
//! that runs on a chip this crate can't build for (see llm-firmware's
//! README for why: no execution environment available to this port has
//! the Xtensa/ESP-IDF toolchain).

use llm_core::Model;
use std::fs;

#[test]
fn unpack_row_int8_matches_deq_row() {
    let bytes = {
        let bin = llm_host::manifest::default_model().bin_path();
        fs::read(&bin)
            .unwrap_or_else(|e| panic!("read {}: {e} (run from llm-host/)", bin.display()))
    };
    let model = Model::load(&bytes).expect("model.bin should parse");
    let tok_emb = model.tok_emb();
    assert_eq!(
        tok_emb.n_groups, 1,
        "int8 head staging assumes a single group per row (group >= cols), matching the C \
         reference's `// n_groups==1` comment at its stage_head_int8 call site -- if this ever \
         fails, the model's `group` config shrank below `dim` and head.rs's staging needs to \
         handle multiple scales per row"
    );

    let mut codes = vec![0i8; tok_emb.cols];
    let mut deq = vec![0f32; tok_emb.cols];
    // Sample rows spread across the table rather than all of them -- the
    // staging loop is identical for every row, so this is representative
    // without making the test slow.
    let sample_rows = [0usize, 1, 2, 100, 1000, tok_emb.rows / 2, tok_emb.rows - 1];
    for &r in &sample_rows {
        let scale = tok_emb.unpack_row_int8(r, &mut codes);
        tok_emb.deq_row(r, &mut deq);
        for j in 0..tok_emb.cols {
            let reconstructed = codes[j] as f32 * scale;
            assert_eq!(
                reconstructed, deq[j],
                "row {r} col {j}: staged int8*scale doesn't match deq_row's fp32 dequant"
            );
        }
    }
}

#[test]
fn tok_emb_rows_can_be_capped_independently_of_the_model() {
    // Mirrors the C reference's `model.tok_emb.rows = VOCAB_N;` before
    // staging: the padded rows above the trained vocab size have no vocab
    // entry and are never scored, so llm-firmware caps its own copy of
    // tok_emb before staging it, without touching the Model used for the
    // rest of the forward pass.
    let bytes = {
        let bin = llm_host::manifest::default_model().bin_path();
        fs::read(&bin)
            .unwrap_or_else(|e| panic!("read {}: {e} (run from llm-host/)", bin.display()))
    };
    let model = Model::load(&bytes).expect("model.bin should parse");
    let full_rows = model.tok_emb().rows;
    assert!(full_rows > 1, "sanity: test model should have more than one vocab row");

    let mut capped = model.tok_emb();
    capped.rows = 1;
    assert_eq!(capped.rows, 1, "QT::rows should be freely mutable on an owned copy");
    assert_eq!(model.tok_emb().rows, full_rows, "capping a copy must not affect the Model's own tok_emb");
}
