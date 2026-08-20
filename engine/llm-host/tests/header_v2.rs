//! The `"PLE\0"` header gate.
//!
//! Two header formats exist for the same tensor stream:
//!
//!   `"PLE1"`  — magic, then 8 int32 config fields, then rope_theta. What
//!              `model/tinystories-v32768/model.bin` and the frozen C
//!              reference (`reference-c/.../llm.h`) both use.
//!   `"PLE\0"` — magic, version, header_bytes, flags, input_vocab,
//!              output_vocab, then the same 7 remaining fields and
//!              rope_theta. What the exporter in `training/` writes today.
//!
//! Before this, `Model::load` accepted only the first, so anything produced
//! by following `TRAINING.md` died with `BadMagic` — the pipeline was
//! documented but could not actually round-trip. See TRAINING.md's "The
//! export format gap".
//!
//! Every fixture here is built by *rewriting the header of the real
//! model.bin* and leaving its tensor bytes untouched, so what's under test
//! is the header parse alone, against weights whose correct output is
//! already pinned by `golden.rs` / `ppl_parity.rs` / `cli_parity.rs`.

use llm_core::{llm_forward, LoadError, Model};
use llm_host::support::ScratchOwned;
use std::fs;

const MAGIC_V2: u32 = 0x0045_4C50; // b"PLE\0" little-endian
const V1_HEADER_BYTES: usize = 4 + 8 * 4 + 4; // magic + 8 i32 + f32
const V2_HEADER_BYTES: usize = 4 * 6 + 7 * 4 + 4; // magic..output_vocab + 7 i32 + f32
const FLAG_TIED_HEAD: u32 = 1;

fn real_model_bytes() -> Vec<u8> {
    let m = llm_host::manifest::default_model();
    fs::read(m.bin_path()).expect("model.bin")
}

/// Re-emit `v1` with a `"PLE\0"` header. `tensors` are copied verbatim.
fn to_v2(v1: &[u8], out_vocab: u32, flags: u32, version: u32) -> Vec<u8> {
    let i32_at = |off: usize| i32::from_le_bytes(v1[off..off + 4].try_into().unwrap());
    let vocab = i32_at(4) as u32;

    let mut out = Vec::with_capacity(v1.len() + 16);
    out.extend_from_slice(&MAGIC_V2.to_le_bytes());
    out.extend_from_slice(&version.to_le_bytes());
    out.extend_from_slice(&(V2_HEADER_BYTES as u32).to_le_bytes());
    out.extend_from_slice(&flags.to_le_bytes());
    out.extend_from_slice(&vocab.to_le_bytes());
    out.extend_from_slice(&out_vocab.to_le_bytes());
    // dim, n_layers, n_heads, ffn, ple_dim, seq_len, group — v1 offsets 8..36
    out.extend_from_slice(&v1[8..36]);
    // rope_theta
    out.extend_from_slice(&v1[36..40]);
    assert_eq!(out.len(), V2_HEADER_BYTES);
    out.extend_from_slice(&v1[V1_HEADER_BYTES..]);
    out
}

fn logits_for(bytes: &[u8], tokens: &[u32]) -> Vec<f32> {
    let model = Model::load(bytes).expect("load");
    let mut owned = ScratchOwned::new(&model.cfg);
    let mut s = owned.as_scratch();
    for (pos, &t) in tokens.iter().enumerate() {
        llm_forward(&model, t as usize, pos, &mut s);
    }
    s.logits.to_vec()
}

/// A v2 file that declares `output_vocab == input_vocab` must produce
/// bit-identical logits to the v1 file it was rewritten from. The header is
/// the only thing that differs, so anything else would be a parse bug.
#[test]
fn v2_header_with_full_output_vocab_is_bit_identical_to_v1() {
    let v1 = real_model_bytes();
    let vocab = i32::from_le_bytes(v1[4..8].try_into().unwrap()) as u32;
    let v2 = to_v2(&v1, vocab, FLAG_TIED_HEAD, 1);

    let m1 = Model::load(&v1).expect("v1 loads");
    let m2 = Model::load(&v2).expect("v2 loads");
    assert_eq!(m1.cfg.vocab, m2.cfg.vocab);
    assert_eq!(m1.cfg.dim, m2.cfg.dim);
    assert_eq!(m1.cfg.n_layers, m2.cfg.n_layers);
    assert_eq!(m1.cfg.n_heads, m2.cfg.n_heads);
    assert_eq!(m1.cfg.ffn, m2.cfg.ffn);
    assert_eq!(m1.cfg.ple_dim, m2.cfg.ple_dim);
    assert_eq!(m1.cfg.seq_len, m2.cfg.seq_len);
    assert_eq!(m1.cfg.group, m2.cfg.group);
    assert_eq!(m1.cfg.rope_theta, m2.cfg.rope_theta);
    assert_eq!(m1.cfg.out_vocab, m2.cfg.out_vocab);

    let prompt = [433u32, 447, 259, 405];
    let a = logits_for(&v1, &prompt);
    let b = logits_for(&v2, &prompt);
    assert_eq!(a.len(), b.len());
    assert!(
        a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()),
        "v2 header changed the logits; the header parse is wrong"
    );
}

/// A v1 file must keep scoring every embedding row. It has no way to say
/// otherwise, and the frozen C reference scores all of them — narrowing the
/// head here would silently change greedy decode against a file whose author
/// never declared the distinction.
#[test]
fn v1_scores_every_embedding_row() {
    let v1 = real_model_bytes();
    let m = Model::load(&v1).expect("load");
    assert_eq!(
        m.cfg.out_vocab, m.cfg.vocab,
        "a PLE1 file must not narrow its own head"
    );
}

/// The point of the format: with `output_vocab < input_vocab` the head skips
/// the padded rows the tokenizer can never emit. The logits it *does*
/// produce must be bit-identical to the corresponding prefix of the full
/// run — skipping rows may not perturb the rows that remain.
#[test]
fn narrowed_output_vocab_matches_the_prefix_of_the_full_head() {
    let v1 = real_model_bytes();
    let vocab = i32::from_le_bytes(v1[4..8].try_into().unwrap()) as u32;
    let narrowed = vocab / 2;
    assert!(narrowed > 0);

    let full = to_v2(&v1, vocab, FLAG_TIED_HEAD, 1);
    let part = to_v2(&v1, narrowed, FLAG_TIED_HEAD, 1);

    let m = Model::load(&part).expect("load");
    assert_eq!(m.cfg.vocab, vocab as usize, "embedding still has all rows");
    assert_eq!(m.cfg.out_vocab, narrowed as usize, "head is narrowed");

    let prompt = [433u32, 447, 259, 405];
    let a = logits_for(&full, &prompt);
    let b = logits_for(&part, &prompt);
    assert_eq!(b.len(), narrowed as usize, "scratch sized by out_vocab");
    assert!(
        a[..narrowed as usize]
            .iter()
            .zip(&b)
            .all(|(x, y)| x.to_bits() == y.to_bits()),
        "narrowing the head changed the logits it still computes"
    );
}

/// An untied head is a different weight layout with an extra tensor this
/// port does not read, and no model exists here to verify it against.
/// Refused loudly rather than silently misparsed.
#[test]
fn untied_head_is_refused() {
    let v1 = real_model_bytes();
    let vocab = i32::from_le_bytes(v1[4..8].try_into().unwrap()) as u32;
    let v2 = to_v2(&v1, vocab, 0, 1); // flags = 0 -> not tied
    assert!(matches!(Model::load(&v2), Err(LoadError::UntiedHead)));
}

/// `header_bytes` lets a future version append fields, but a version this
/// reader doesn't know may also have changed what it already reads.
#[test]
fn future_format_version_is_refused() {
    let v1 = real_model_bytes();
    let vocab = i32::from_le_bytes(v1[4..8].try_into().unwrap()) as u32;
    let v2 = to_v2(&v1, vocab, FLAG_TIED_HEAD, 99);
    assert!(matches!(
        Model::load(&v2),
        Err(LoadError::UnsupportedVersion(99))
    ));
}

/// The tied head is the first `output_vocab` rows of the embedding, so a
/// larger `output_vocab` would read past that tensor. Caught at load time,
/// where it can still be reported, rather than as a panic mid-forward.
#[test]
fn output_vocab_larger_than_the_embedding_is_refused() {
    let v1 = real_model_bytes();
    let vocab = i32::from_le_bytes(v1[4..8].try_into().unwrap()) as u32;
    for bad in [0, vocab + 1] {
        let v2 = to_v2(&v1, bad, FLAG_TIED_HEAD, 1);
        assert!(
            matches!(Model::load(&v2), Err(LoadError::BadOutputVocab)),
            "output_vocab={bad} should be refused"
        );
    }
}

/// A file that is neither magic is still rejected — this change widened what
/// is accepted, and that must not have widened it to "anything".
#[test]
fn unknown_magic_is_still_bad_magic() {
    let mut v1 = real_model_bytes();
    v1[0..4].copy_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
    assert!(matches!(Model::load(&v1), Err(LoadError::BadMagic)));
}
