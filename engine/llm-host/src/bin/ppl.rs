//! Port of reference-c/firmware/host_verify/ppl.c: run the Rust inference
//! engine over real validation tokens and report cross-entropy /
//! perplexity. Build with `--features int8-activations` to measure the
//! quality cost of int8-activation quantization, same as the C code's
//! `-DLLM_INT8_ACT` build.
//!
//! Usage: ppl <model.bin> <val.bin> [windows]

use llm_core::{llm_forward, Model};
use llm_host::support::ScratchOwned;
use std::fs;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("firmware/model/model.bin");
    let val_path = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("model/tinystories-v32768/val.bin");
    let windows: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(8);

    let bytes = fs::read(model_path).unwrap_or_else(|e| {
        eprintln!("{model_path}: {e}");
        std::process::exit(1);
    });
    let model = Model::load(&bytes).unwrap_or_else(|e| {
        eprintln!("bad magic: {e:?}");
        std::process::exit(1);
    });

    let val_bytes = fs::read(val_path).unwrap_or_else(|e| {
        eprintln!("{val_path}: {e}");
        std::process::exit(1);
    });
    let val: Vec<u16> = val_bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();

    let mut owned = ScratchOwned::new(&model.cfg);
    let mut s = owned.as_scratch();

    let seq = model.cfg.seq_len;
    let vocab = model.cfg.out_vocab;
    let mut nll = 0f64;
    let mut count = 0u64;

    for w in 0..windows {
        let base = w * seq;
        if base + seq + 1 > val.len() {
            break;
        }
        for pos in 0..seq {
            let tok = val[base + pos] as usize;
            llm_forward(&model, tok, pos, &mut s);
            let target = val[base + pos + 1] as usize;

            let mut mx = f32::NEG_INFINITY;
            for &lv in s.logits[..vocab].iter() {
                if lv > mx {
                    mx = lv;
                }
            }
            let mut sum = 0f64;
            for &lv in s.logits[..vocab].iter() {
                sum += ((lv - mx) as f64).exp();
            }
            let ce = sum.ln() - (s.logits[target] - mx) as f64;
            nll += ce;
            count += 1;
        }
    }

    let mean = nll / count as f64;
    let mode = if cfg!(feature = "int8-activations") {
        "int8-activations"
    } else {
        "fp32-activations"
    };
    println!(
        "{mode:<18}  val CE {mean:.4}  ppl {:.2}   ({count} predictions)",
        mean.exp()
    );
}
