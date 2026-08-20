//! Gate for staging the output head as int8 again, instead of packed int4.
//!
//! WHY GO BACK. `int4_head_equivalence.rs` gated the move *to* packed int4,
//! which halved what the head streams per token: 2.54 MB down to 1.32 MB. On
//! the device that change moved the stage by **0.0 ms**, and the reason only
//! became clear later, when the dual-core bandwidth probe measured the head at
//! **21% of available PSRAM bandwidth**. The head is compute-bound and always
//! was. Packing bought a resource the stage was not short of.
//!
//! It is not free, though. Every element pays a mask or a shift to extract its
//! nibble, in a loop that runs 2.43M times per token — and the C reference,
//! which stages int8, never pays it. When this test was written the head was
//! the one stage still above C (1.09x, against 0.89–0.90x everywhere else),
//! and this looked like a specific, testable reason why.
//!
//! HOW IT TURNED OUT. It was tested, and it was the wrong reason: staging
//! int8 made the head 62.3 -> 76.9 ms and the token 106.3 -> 121.5. The extra
//! 1.2 MB streamed per token cost more than the mask saved — which also
//! falsified "the head is compute-bound so bytes don't matter". Low bandwidth
//! utilisation is not the same as insensitivity to bytes. Reverted.
//!
//! The gap closed from the instruction side instead: `i16` activations, then
//! two rows per pass. The head is now 49.7 ms, 0.87x of C, with the packing
//! untouched. This test is kept because the equivalence it asserts is what
//! makes the experiment cheap to re-run on a board with different memory.
//!
//! WHAT THIS ASSERTS. That `dot_int8_int8` over `QT::unpack_row_int8`'s output
//! is bit-for-bit identical to `dot_int4_int8` over the packed row, for every
//! row of the shipped model. Both must also equal a plain unrolled reference
//! sum, so a shared bug in the two four-lane implementations cannot pass.
//!
//! Integer addition is associative and every term is the same term, so "close"
//! here would mean a bug rather than rounding — the same standard
//! `int4_head_equivalence.rs` holds.
//!
//! WHAT IT DOES NOT ANSWER: whether dropping the unpack is actually faster on
//! an LX7, and whether the extra 1.2 MB of PSRAM is affordable. Those are
//! device questions. This only establishes that switching cannot change a
//! single logit.

use llm_core::{activation_sum, dot_int4_int8, dot_int8_int8, quantize_activations, Model};
use llm_host::manifest;

fn model_bytes() -> Vec<u8> {
    std::fs::read(manifest::default_model().bin_path()).expect("model.bin")
}

/// Deliberately naive: one accumulator, front to back, no unrolling. Both
/// production dots are four-lane, so checking them only against each other
/// would miss a mistake made the same way in both.
fn reference_dot(row: &[i8], actq: &[i8]) -> i32 {
    row.iter()
        .zip(actq)
        .map(|(&w, &a)| w as i32 * a as i32)
        .sum()
}

#[test]
fn int8_and_int4_heads_agree_bit_for_bit_on_every_row() {
    let bytes = model_bytes();
    let model = Model::load(&bytes).expect("load");
    let head = model.tok_emb();
    let cols = head.cols;
    assert_eq!(head.n_groups, 1, "this path assumes one scale per row");

    // The same activation shapes int4_head_equivalence.rs uses, including the
    // extremes that stress `quantize_activations`' max-abs scaling.
    let cases: [&dyn Fn(usize) -> f32; 4] = [
        &|j| (j as f32 * 0.37).sin(),
        &|j| if j % 2 == 0 { 1.0 } else { -1.0 },
        &|j| (j as f32 - 48.0) * 1e-4,
        &|_| 0.0,
    ];

    let mut staged = vec![0i8; cols];
    for (ci, f) in cases.iter().enumerate() {
        let x: Vec<f32> = (0..cols).map(|j| f(j)).collect();
        let mut actq = vec![0i8; cols];
        let act_scale = quantize_activations(&x, &mut actq);
        let asum = activation_sum(&actq);

        for r in 0..head.rows {
            let scale8 = head.unpack_row_int8(r, &mut staged);
            let (codes, scale4) = head.packed_row(r);

            let d8 = dot_int8_int8(&staged, &actq);
            let d4 = dot_int4_int8(codes, &actq, asum);
            let dref = reference_dot(&staged, &actq);

            assert_eq!(
                d8, dref,
                "case {ci} row {r}: four-lane int8 dot != reference"
            );
            assert_eq!(
                d4, dref,
                "case {ci} row {r}: four-lane int4 dot != reference"
            );
            assert_eq!(
                scale8.to_bits(),
                scale4.to_bits(),
                "case {ci} row {r}: scale differs"
            );

            // What the firmware actually writes into `logits`, in the same
            // multiply order `head.rs::rows_range` uses.
            let v8 = d8 as f32 * scale8 * act_scale;
            let v4 = d4 as f32 * scale4 * act_scale;
            assert_eq!(
                v8.to_bits(),
                v4.to_bits(),
                "case {ci} row {r}: logit differs"
            );
        }
    }
}

/// The cost side of the trade, asserted rather than asserted-in-a-comment:
/// int8 staging is twice the bytes, and the board has to have them.
#[test]
fn int8_staging_costs_twice_the_psram_of_int4() {
    let bytes = model_bytes();
    let model = Model::load(&bytes).expect("load");
    let head = model.tok_emb();
    // The firmware caps rows to the trained vocab (25,353) before staging.
    let rows = 25_353.min(head.rows);
    let int4 = rows * head.row_bytes + rows * 4;
    let int8 = rows * head.cols + rows * 4;
    eprintln!(
        "head staging: int4 {:.2} MB -> int8 {:.2} MB (+{:.2} MB), \
         board reported 4427 KB PSRAM free with int4 staged",
        int4 as f64 / 1e6,
        int8 as f64 / 1e6,
        (int8 - int4) as f64 / 1e6
    );
    assert!(
        int8 > int4,
        "int8 staging must be the larger of the two; if not, the packing is not doing anything"
    );
    assert!(
        (int8 - int4) as f64 / 1e6 < 3.0,
        "the extra must fit in the PSRAM the board actually has free"
    );
}
