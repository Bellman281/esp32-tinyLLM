//! One-shot converter: `vocab.h` (C array-literal source) -> two flat
//! binary files `llm-firmware` embeds via `include_bytes!`. Rerun this any
//! time `vocab.h` changes upstream (new export, retrained tokenizer).
//!
//! Usage: vocab-to-bin <vocab.h> <blob.bin> <off.bin>

use llm_host::vocab::Vocab;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "usage: {} vocab.h blob.bin off.bin",
            args.first().map(String::as_str).unwrap_or("vocab-to-bin")
        );
        std::process::exit(1);
    }
    let vocab = Vocab::load(&args[1]).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });
    vocab.dump_binary(&args[2], &args[3]).unwrap_or_else(|e| {
        eprintln!("write failed: {e}");
        std::process::exit(1);
    });
    eprintln!("wrote {} ({} tokens)", args[2], vocab.n);
    eprintln!("wrote {} ({} bytes)", args[3], (vocab.n + 1) * 4);
}
