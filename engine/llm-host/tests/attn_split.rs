//! Gate for splitting attention's heads across two cores.
//!
//! `attn` is the largest stage that the matvec split could not reach --
//! attention is not a matvec, so `matvec_override` never sees it. Instrumented
//! four ways on the ESP32-S3 it came out `qkv 7.8 | rope 0.14 | core 28.3 |
//! proj 2.7` ms/token: the `core` part alone is 24% of a token, and it was
//! running on one LX7 while the worker task sat blocked.
//!
//! Heads are independent -- each reads the whole (already stored) KV cache and
//! writes only its own `head_dim` slice of the output -- so dividing them
//! across cores should be bit-exact for the same reason splitting a matvec by
//! rows is: no dot product changes and nothing reorders.
//!
//! "Should be" is not the bar. `attention::heads` addresses `att_out`
//! *relative* to the head range it was given (the same re-addressing
//! `QT::matvec_rows_into` does for rows), which is what lets two cores hold
//! disjoint `&mut [f32]` halves instead of both holding `&mut` to the whole
//! buffer. This asserts the re-addressing changed nothing, at every split
//! point, through the real forward pass on the real model.
//!
//! Two things make this stronger than a one-shot check on a single call:
//!
//!   * It runs a multi-token generation, so the KV cache accumulates. A split
//!     that corrupted one head's cache region would show up at the next
//!     position even if the current one looked right.
//!   * Nothing but attention is overridden, so a difference in the logits can
//!     only be the split. `llm_forward_with_attn_override` exists for exactly
//!     that reason.
//!
//! What this does NOT cover: the FreeRTOS handoff in `llm-firmware`'s
//! `head.rs`, which no host test can reach. It covers the arithmetic, so that
//! a device mismatch would have to be the plumbing.

use llm_core::{llm_forward, llm_forward_with_attn_override, HeadsJob, Model};
use llm_host::manifest;
use llm_host::support::ScratchOwned;

const STEPS: usize = 24;

fn model_bytes() -> Vec<u8> {
    std::fs::read(manifest::default_model().bin_path()).expect("model.bin")
}

/// A short deterministic token walk. Greedy decode would work too, but a fixed
/// sequence keeps every split point driving *identical* input, which is the
/// property being compared -- if one split diverged and then greedy decode
/// followed a different branch, later positions would differ for two reasons
/// at once.
fn token_at(step: usize, out_vocab: usize) -> usize {
    (step * 2654435761) % out_vocab
}

/// Logits after each of `STEPS` decode steps, with the heads divided at
/// `split` (`None` = the unmodified `llm_forward` path).
fn run(model: &Model, split: Option<usize>) -> Vec<Vec<f32>> {
    let mut owned = ScratchOwned::new(&model.cfg);
    let mut s = owned.as_scratch();
    let out_vocab = model.cfg.out_vocab;
    let seq = model.cfg.seq_len;
    let mut trace = Vec::with_capacity(STEPS);

    // Per-call scratch for the two ranges. On the device these are two
    // buffers on two different cores; here they are two buffers used one
    // after the other, which exercises the same thing that matters -- that
    // neither range depends on what the other left in `scores`.
    let mut scores_a = vec![0f32; seq];
    let mut scores_b = vec![0f32; seq];

    for step in 0..STEPS {
        let tok = token_at(step, out_vocab);
        match split {
            None => llm_forward(model, tok, step, &mut s),
            Some(sp) => {
                let mut heads = |job: HeadsJob| {
                    assert_eq!(job.pos, step, "job carries the position it was called for");
                    let dh = job.head_dim;
                    let n = job.n_heads;
                    // Exactly the shape `Head::attn_parallel` uses: two
                    // disjoint `&mut` halves of one output buffer, each range
                    // addressed from its own start.
                    let (lo, hi) = job.att_out.split_at_mut(sp * dh);
                    llm_core::attention::heads(
                        job.q,
                        job.kcache_layer,
                        job.vcache_layer,
                        lo,
                        &mut scores_a,
                        job.pos,
                        dh,
                        job.seq_len,
                        0,
                        sp,
                    );
                    llm_core::attention::heads(
                        job.q,
                        job.kcache_layer,
                        job.vcache_layer,
                        hi,
                        &mut scores_b,
                        job.pos,
                        dh,
                        job.seq_len,
                        sp,
                        n,
                    );
                };
                llm_forward_with_attn_override(model, tok, step, &mut s, &mut heads);
            }
        }
        trace.push(s.logits[..out_vocab].to_vec());
    }
    trace
}

#[test]
fn splitting_attention_by_heads_is_bit_exact() {
    let bytes = model_bytes();
    let model = Model::load(&bytes).expect("load");
    let n_heads = model.cfg.n_heads;
    let reference = run(&model, None);

    for sp in 0..=n_heads {
        let got = run(&model, Some(sp));
        for (step, (want, have)) in reference.iter().zip(&got).enumerate() {
            for (v, (a, b)) in want.iter().zip(have).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "split at head {sp}, step {step}, logit {v}: {a} vs {b}"
                );
            }
        }
    }
}

/// The degenerate splits are the ones most likely to be quietly wrong: `0`
/// gives the first range no heads at all, `n_heads` gives the second none.
/// Both must still write every element of `att_out` -- nothing else does, so a
/// skipped range shows up as stale scratch rather than as zeros, which is
/// exactly the failure a "looks reasonable" check would miss.
#[test]
fn an_empty_head_range_still_leaves_the_output_fully_written() {
    let bytes = model_bytes();
    let model = Model::load(&bytes).expect("load");
    let n_heads = model.cfg.n_heads;
    let reference = run(&model, None);
    for sp in [0, n_heads] {
        let got = run(&model, Some(sp));
        assert!(
            reference
                .iter()
                .zip(&got)
                .all(|(w, h)| w.iter().zip(h).all(|(a, b)| a.to_bits() == b.to_bits())),
            "degenerate split at {sp} differs"
        );
    }
}
