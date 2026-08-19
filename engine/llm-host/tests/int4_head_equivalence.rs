//! Gate for staging the output head as int4 instead of int8.
//!
//! `llm-firmware` currently unpacks the tied embedding to int8 in PSRAM at
//! boot (`stage_and_spawn` -> `QT::unpack_row_int8`) so the per-token head is
//! a plain `i8 * i8` dot. That doubles what streams per token — 2.43 MB rather
//! than 1.21 MB on the shipped model — and spends 2.4 MB of PSRAM to do it.
//! `llm_core::dot_int4_int8` reads the packed nibbles directly instead.
//!
//! The firmware change is only safe if the two produce *identical* logits, and
//! no host test can exercise FreeRTOS or PSRAM — so this pins the part that
//! can be pinned: the arithmetic. It reconstructs exactly what
//! `head.rs::rows_range` computes, both ways, over every row of the real
//! model, and requires bit-for-bit agreement. Integer addition is associative
//! and every term is the same term, so "close" would mean a bug, not rounding.
//!
//! What this does NOT cover, and what the device run is for: whether halving
//! the stream actually beats the unpack cost on an LX7. That is a memory-system
//! question this machine cannot answer.

use llm_core::{dot_int4_int8, quantize_activations, Model};
use llm_host::manifest;

fn model_bytes() -> Vec<u8> {
    std::fs::read(manifest::default_model().bin_path()).expect("model.bin")
}

/// Ports `head.rs::rows_range`'s scaling, so both paths below differ only in
/// how the row's codes are read.
fn row_value(dot: i32, row_scale: f32, act_scale: f32) -> f32 {
    dot as f32 * row_scale * act_scale
}

#[test]
fn int4_head_matches_the_staged_int8_head_bit_for_bit() {
    let bytes = model_bytes();
    let model = Model::load(&bytes).expect("load");
    let head = model.tok_emb();
    let cols = head.cols;
    assert_eq!(head.n_groups, 1, "this path assumes one scale per row");

    // A few activation vectors with different shapes, including the extremes
    // that stress `quantize_activations`' max-abs scaling.
    let cases: [&dyn Fn(usize) -> f32; 4] = [
        &|j| (j as f32 * 0.37).sin(),
        &|j| if j % 2 == 0 { 1.0 } else { -1.0 },
        &|j| (j as f32 - 48.0) * 1e-4,
        &|_| 0.0,
    ];

    for (ci, f) in cases.iter().enumerate() {
        let x: Vec<f32> = (0..cols).map(|j| f(j)).collect();
        let mut actq = vec![0i8; cols];
        let act_scale = quantize_activations(&x, &mut actq);

        let mut staged = vec![0i8; cols];
        for r in 0..head.rows {
            // Path A: what ships today — unpack to int8 at boot, plain dot.
            let row_scale_a = head.unpack_row_int8(r, &mut staged);
            let mut dot_a: i32 = 0;
            for (&a, &w) in actq.iter().zip(staged.iter()) {
                dot_a += a as i32 * w as i32;
            }

            // Path B: read the packed nibbles directly.
            let (codes, row_scale_b) = head.packed_row(r);
            let dot_b = dot_int4_int8(codes, &actq);

            assert_eq!(row_scale_a.to_bits(), row_scale_b.to_bits(),
                       "case {ci} row {r}: row scale differs");
            assert_eq!(dot_a, dot_b, "case {ci} row {r}: int32 dot differs");
            assert_eq!(
                row_value(dot_a, row_scale_a, act_scale).to_bits(),
                row_value(dot_b, row_scale_b, act_scale).to_bits(),
                "case {ci} row {r}: logit differs"
            );
        }
    }
}

/// The saving the firmware change is being made for, asserted rather than
/// asserted-in-a-comment: the packed form is half the bytes.
#[test]
fn int4_staging_halves_what_the_head_streams() {
    let bytes = model_bytes();
    let model = Model::load(&bytes).expect("load");
    let head = model.tok_emb();

    // Firmware caps rows to the trained vocab before staging; use the full
    // table here since the ratio is what matters.
    assert_eq!(head.row_bytes, head.cols.div_ceil(2), "row_bytes is ceil(cols/2)");
    let int8_bytes = head.rows * head.cols;
    let int4_bytes = head.rows * head.row_bytes;
    assert!(
        int4_bytes * 2 <= int8_bytes + head.rows,
        "packed head ({int4_bytes}) must be ~half the staged int8 head ({int8_bytes})"
    );
    eprintln!(
        "head staging: int8 {:.2} MB -> int4 {:.2} MB (rows={} cols={})",
        int8_bytes as f64 / 1e6,
        int4_bytes as f64 / 1e6,
        head.rows,
        head.cols
    );
}
