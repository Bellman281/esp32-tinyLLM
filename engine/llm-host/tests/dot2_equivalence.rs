//! Gate for computing two head rows in one pass.
//!
//! WHY. The head's inner loop measured **issue-bound** on the ESP32-S3: 8.96
//! cycles/MAC with the weights resident in the data cache against 9.64
//! streaming from PSRAM. Only 8% apart, so the four accumulator chains already
//! cover load-use latency and software pipelining would buy nothing. The only
//! lever left is fewer instructions per multiply-accumulate.
//!
//! The activation vector is the same for all 25,353 rows, and the one-row-at-
//! a-time loop reloads it for every one of them — 2.43M `l16si` per token to
//! read 96 distinct values. `dot2_int4_int16` loads each activation once and
//! multiplies it into two rows.
//!
//! WHAT THIS ASSERTS. That both returned values equal what `dot_int4_int16`
//! returns for those rows individually, on every consecutive row pair of the
//! shipped model, across several activation shapes. Each row's accumulation is
//! untouched — same terms, same order, same lanes, same hoisted bias — so this
//! must be exact, and "close" would mean the two rows are contaminating each
//! other.
//!
//! WHAT IT DOES NOT ANSWER: whether the LX7 actually halves the activation
//! loads. That is `scripts/disasm_head.sh` followed by the stopwatch.

use llm_core::{
    activation_sum_i16, dot2_int4_int16, dot_int4_int16, quantize_activations_i16, Model,
};
use llm_host::manifest;

#[test]
fn two_rows_at_once_match_two_rows_one_at_a_time() {
    let bytes = std::fs::read(manifest::default_model().bin_path()).expect("model.bin");
    let model = Model::load(&bytes).expect("load");
    let head = model.tok_emb();
    let cols = head.cols;

    let cases: [&dyn Fn(usize) -> f32; 4] = [
        &|j| (j as f32 * 0.37).sin(),
        &|j| if j % 2 == 0 { 1.0 } else { -1.0 },
        &|_| 0.0,
        &|j| if j % 3 == 0 { 1e6 } else { -1e6 },
    ];

    for (ci, f) in cases.iter().enumerate() {
        let x: Vec<f32> = (0..cols).map(|j| f(j)).collect();
        let mut a = vec![0i16; cols];
        let scale = quantize_activations_i16(&x, &mut a);
        let asum = activation_sum_i16(&a);

        // Every consecutive pair, plus the odd tail row on its own — the
        // firmware walks rows in pairs and must handle an odd row count.
        let mut r = 0usize;
        while r + 1 < head.rows {
            let (ca, _) = head.packed_row(r);
            let (cb, _) = head.packed_row(r + 1);
            let (d2a, d2b) = dot2_int4_int16(ca, cb, &a, asum);
            assert_eq!(
                d2a,
                dot_int4_int16(ca, &a, asum),
                "case {ci} row {r}: first of pair"
            );
            assert_eq!(
                d2b,
                dot_int4_int16(cb, &a, asum),
                "case {ci} row {}: second of pair",
                r + 1
            );
            // And the scaled logits the firmware actually writes.
            let (_, sa) = head.packed_row(r);
            let (_, sb) = head.packed_row(r + 1);
            assert_eq!(
                (d2a as f32 * sa * scale).to_bits(),
                (dot_int4_int16(ca, &a, asum) as f32 * sa * scale).to_bits()
            );
            assert_eq!(
                (d2b as f32 * sb * scale).to_bits(),
                (dot_int4_int16(cb, &a, asum) as f32 * sb * scale).to_bits()
            );
            r += 2;
        }
        assert!(
            r >= head.rows - 1,
            "case {ci}: pair walk left more than one row"
        );
    }
}

/// Feeding the same row twice must give the same answer twice — the cheapest
/// possible check that the two accumulator sets are genuinely independent
/// rather than aliasing each other.
#[test]
fn the_same_row_twice_gives_the_same_answer_twice() {
    let bytes = std::fs::read(manifest::default_model().bin_path()).expect("model.bin");
    let model = Model::load(&bytes).expect("load");
    let head = model.tok_emb();
    let cols = head.cols;
    let x: Vec<f32> = (0..cols).map(|j| (j as f32 * 0.11).cos()).collect();
    let mut a = vec![0i16; cols];
    quantize_activations_i16(&x, &mut a);
    let asum = activation_sum_i16(&a);

    for r in [0, 1, 7, 1000, head.rows - 1] {
        let (c, _) = head.packed_row(r);
        let (p, q) = dot2_int4_int16(c, c, &a, asum);
        let single = dot_int4_int16(c, &a, asum);
        assert_eq!(p, single, "row {r}: lane set A");
        assert_eq!(q, single, "row {r}: lane set B");
    }
}
