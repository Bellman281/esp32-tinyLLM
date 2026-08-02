//! Port of reference-c/esp32-llm-lab/gen_prompt.c: generate from an
//! arbitrary prompt (comma-separated token ids). Greedy by default
//! (temp=0, deterministic, matches the chip); temp>0 samples with top-k=40.
//!
//! Usage: gen-prompt <model.bin> <vocab.h> <N> <id,id,...> [temp]
//!
//! Deviation from the C original: vocab.h is a runtime argument (see
//! vocab.rs), and sampled output (temp>0) uses a different PRNG than C's
//! `rand()` so it will NOT match the C binary token-for-token in that mode
//! -- see the `Xorshift32` doc comment. Greedy mode (temp=0, the default,
//! and what the parity tests use) has no randomness and matches exactly.

use llm_core::{llm_forward, Model};
use llm_host::{support::ScratchOwned, vocab::Vocab};
use std::fs;
use std::io::Write;

/// Small xorshift32 PRNG. NOT the same stream as C's `rand()` -- there is
/// no portable way to reproduce glibc's rand() from Rust without
/// reimplementing it byte-for-byte, and doing so would buy nothing: the
/// golden-logit and greedy-decode parity tests never exercise this path
/// (they use temp=0). Only here so `gen-prompt`'s sampling mode is
/// functionally complete, matching gen_prompt.c's *algorithm*, not its
/// exact random bit stream.
struct Xorshift32(u32);
impl Xorshift32 {
    fn next_unit(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        (x as f32) / (u32::MAX as f32)
    }
}

fn pick(logits: &[f32], vocab_n: usize, temp: f32, rng: &mut Xorshift32) -> usize {
    if temp <= 0.0 {
        let mut best = 0usize;
        let mut bv = f32::NEG_INFINITY;
        for v in 0..vocab_n {
            if logits[v] > bv {
                bv = logits[v];
                best = v;
            }
        }
        return best;
    }
    let k = core::cmp::min(40, vocab_n);
    let mut idx = [0usize; 40];
    let mut val = [f32::NEG_INFINITY; 40];
    let mut used = vec![false; vocab_n];
    for slot in idx.iter_mut().take(k) {
        let mut b = 0usize;
        let mut bv = f32::NEG_INFINITY;
        for x in 0..vocab_n {
            if !used[x] && logits[x] > bv {
                bv = logits[x];
                b = x;
            }
        }
        used[b] = true;
        *slot = b;
    }
    for (slot, &i) in idx.iter().enumerate().take(k) {
        val[slot] = logits[i];
    }
    let mx = val[0];
    let mut sum = 0f32;
    for v in val.iter_mut().take(k) {
        *v = ((*v - mx) / temp).exp();
        sum += *v;
    }
    let r = rng.next_unit() * sum;
    let mut acc = 0f32;
    for slot in 0..k {
        acc += val[slot];
        if r <= acc {
            return idx[slot];
        }
    }
    idx[k - 1]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!(
            "usage: {} model.bin vocab.h N id,id[,id...] [temp]",
            args.first().map(String::as_str).unwrap_or("gen-prompt")
        );
        std::process::exit(1);
    }
    let bytes = fs::read(&args[1]).unwrap_or_else(|e| {
        eprintln!("{}: {e}", args[1]);
        std::process::exit(1);
    });
    let model = Model::load(&bytes).unwrap_or_else(|e| {
        eprintln!("bad magic: {e:?}");
        std::process::exit(1);
    });
    let vocab = Vocab::load(&args[2]).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });
    let n: usize = args[3].parse().expect("N must be an integer");
    let prompt: Vec<usize> = args[4]
        .split(',')
        .map(|s| s.parse().expect("prompt ids must be integers"))
        .collect();
    let temp: f32 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(0.0);

    let mut owned = ScratchOwned::new(&model.cfg);
    let mut s = owned.as_scratch();
    let mut rng = Xorshift32(0x1234_5678);

    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    write!(out, ">>> ").unwrap();

    let mut pos = 0usize;
    for &t in prompt.iter() {
        if pos >= model.cfg.seq_len {
            break;
        }
        emit(&mut out, &vocab, t);
        llm_forward(&model, t, pos, &mut s);
        pos += 1;
    }
    for _ in 0..n {
        if pos >= model.cfg.seq_len {
            break;
        }
        let tok = pick(&s.logits, vocab.n, temp, &mut rng);
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
