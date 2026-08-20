//! Gate for splitting a matvec's rows across two cores.
//!
//! `llm-firmware` runs the output head on both LX7 cores and everything else
//! on one, so `input`/`attn`/`ffn`/`ple` — 55% of a token — execute with the
//! worker task blocked. Splitting each matvec's rows across both is the
//! obvious fix, and it is bit-exact in principle: `y[r]` depends only on row
//! `r` and `x`, so which core computes it cannot change the value.
//!
//! "In principle" is not the bar this project uses. `QT::matvec_rows_into`
//! addresses the output relative to `out` rather than absolutely, which is
//! what lets two cores hold disjoint `&mut [f32]` halves from `split_at_mut`
//! instead of both holding `&mut` to the whole buffer. This asserts that the
//! re-addressing changed nothing, at every split point, on real tensors.
//!
//! What this does NOT cover: the FreeRTOS handoff, which no host test can
//! reach. It covers the arithmetic, so that a device failure would have to be
//! the plumbing.

use llm_core::{Model, QT};
use llm_host::manifest;

fn model_bytes() -> Vec<u8> {
    std::fs::read(manifest::default_model().bin_path()).expect("model.bin")
}

fn activations(n: usize) -> Vec<f32> {
    let mut st = 12345u32;
    (0..n)
        .map(|_| {
            st = st.wrapping_mul(1664525).wrapping_add(1013904223);
            (st >> 8) as f32 / 8388608.0 - 1.0
        })
        .collect()
}

/// Every split point of a tensor must reproduce the unsplit result exactly.
fn check_all_splits(name: &str, t: &QT) {
    let x = activations(t.cols);
    let mut whole = vec![0f32; t.rows];
    t.matvec(&x, &mut whole);

    for split in 0..=t.rows {
        let mut out = vec![0f32; t.rows];
        {
            let (lo, hi) = out.split_at_mut(split);
            // Two disjoint &mut, exactly as the two cores will hold them.
            t.matvec_rows_into(&x, lo, 0);
            t.matvec_rows_into(&x, hi, split);
        }
        for r in 0..t.rows {
            assert_eq!(
                whole[r].to_bits(),
                out[r].to_bits(),
                "{name}: split at {split}, row {r} differs \
                 ({} vs {})",
                whole[r],
                out[r]
            );
        }
    }
}

#[test]
fn splitting_a_matvec_by_rows_is_bit_exact() {
    let bytes = model_bytes();
    let model = Model::load(&bytes).expect("load");
    let l0 = model.layer(0);
    // Every distinct shape the forward pass runs through matvec_range, except
    // the head — narrow (down: cols=66), square (attn_proj), tall (qkv), and
    // the one whose cols exceed dim (ple_proj).
    check_all_splits("qkv", &l0.qkv);
    check_all_splits("attn_proj", &l0.attn_proj);
    check_all_splits("gate", &l0.gate);
    check_all_splits("down", &l0.down);
    check_all_splits("ple_gate", &l0.ple_gate);
    check_all_splits("ple_proj", &l0.ple_proj);
}

/// The head is far too big to check every split, so check the one the firmware
/// would actually use plus the degenerate ends.
#[test]
fn splitting_the_head_by_rows_is_bit_exact() {
    let bytes = model_bytes();
    let model = Model::load(&bytes).expect("load");
    let t = model.tok_emb();
    let x = activations(t.cols);
    let mut whole = vec![0f32; t.rows];
    t.matvec(&x, &mut whole);

    for split in [0, 1, t.rows / 2, t.rows - 1, t.rows] {
        let mut out = vec![0f32; t.rows];
        {
            let (lo, hi) = out.split_at_mut(split);
            t.matvec_rows_into(&x, lo, 0);
            t.matvec_rows_into(&x, hi, split);
        }
        assert!(
            whole.iter().zip(&out).all(|(a, b)| a.to_bits() == b.to_bits()),
            "head: split at {split} differs"
        );
    }
}
