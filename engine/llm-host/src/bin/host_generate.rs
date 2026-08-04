//! Port of reference-c/esp32-llm-lab/host_generate.c: run the exact chip
//! inference engine on a laptop and print a story. Greedy decode only, no
//! PyTorch, no emulator, no hardware -- just this binary.
//!
//! Usage: host-generate <model.bin> <vocab.h> [N]
//!
//! Deviation from the C original: vocab.h is a runtime argument here
//! instead of a `#include` baked in at compile time (see vocab.rs).
//! Everything else -- the hardcoded prompt, the greedy decode loop, the
//! VOCAB_N-bounded argmax -- matches host_generate.c exactly.

use llm_core::{llm_forward, Model};
use llm_host::{support::ScratchOwned, vocab::Vocab};
use std::fs;
use std::io::Write;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args.get(1).map(String::as_str).unwrap_or("model.bin");
    let vocab_path = args.get(2).map(String::as_str).unwrap_or("vocab.h");
    let n: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(120);

    let bytes = fs::read(model_path).unwrap_or_else(|e| {
        eprintln!("{model_path}: {e}");
        std::process::exit(1);
    });
    let model = Model::load(&bytes).unwrap_or_else(|e| {
        eprintln!("bad model magic: {e:?}");
        std::process::exit(1);
    });
    let vocab = Vocab::load(vocab_path).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });

    let c = &model.cfg;
    eprintln!(
        "model: V={} D={} L={} H={} F={} P={}  ({:.2} MB)",
        c.vocab,
        c.dim,
        c.n_layers,
        c.n_heads,
        c.ffn,
        c.ple_dim,
        bytes.len() as f64 / 1e6
    );

    let mut owned = ScratchOwned::new(&model.cfg);
    let mut s = owned.as_scratch();

    // "Once upon a time" (this tokenizer) -- reference-c/esp32-llm-lab/host_generate.c's
    // own hardcoded prompt array, kept verbatim for parity. (Note: with the
    // vocab.h bundled alongside it in this snapshot, these ids do NOT
    // actually decode to "Once upon a time" -- see MIGRATION_PLAN.md notes.
    // Reproducing the C reference exactly, not "fixing" it, is the point of
    // this binary.)
    let prompt = [336usize, 337, 258, 338];

    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    write!(out, ">>> ").unwrap();

    let mut pos = 0usize;
    // Matches C's `int tok=0;` -- always overwritten before use in both
    // loops below, same as the original. Kept as a plain dead initializer
    // (not restructured into an Option/loop-scoped binding) for structural
    // fidelity to host_generate.c.
    #[allow(unused_assignments)]
    let mut tok = 0usize;
    for &t in &prompt {
        tok = t;
        emit(&mut out, &vocab, tok);
        llm_forward(&model, tok, pos, &mut s);
        pos += 1;
    }
    for _ in 0..n {
        if pos >= model.cfg.seq_len {
            break;
        }
        let mut best = 0usize;
        let mut bv = f32::NEG_INFINITY;
        for v in 0..vocab.n {
            if s.logits[v] > bv {
                bv = s.logits[v];
                best = v;
            }
        }
        tok = best;
        emit(&mut out, &vocab, tok);
        llm_forward(&model, tok, pos, &mut s);
        pos += 1;
    }
    writeln!(out).unwrap();
}

fn emit<W: Write>(out: &mut W, vocab: &Vocab, tok: usize) {
    if let Some(bytes) = vocab.bytes(tok) {
        out.write_all(bytes).ok();
        out.flush().ok();
    }
}
