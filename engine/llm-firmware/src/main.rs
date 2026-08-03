//! ESP32-S3 PLE TinyLM firmware -- Rust port of `esp32_llm.ino`'s
//! `setup()`/`loop()`. The 28.9M-param model (14.9MB, 4-bit) lives in a
//! flash `model` partition, memory-mapped so the ~25k-row embedding table
//! is read a row at a time from flash; the hot tied output head plus
//! scratch and KV cache sit in PSRAM. Same `llm_core` engine already
//! verified against PyTorch and the C reference on the host
//! (`llm-host`'s golden/ppl/cli parity tests) -- only the platform glue
//! differs here: partition mmap (`partition.rs`), PSRAM allocation
//! (`psram.rs`), the dual-core int8-staged head (`head.rs`), and the
//! embedded vocab decode table (`vocab.rs`).
//!
//! Scope note: this port intentionally does NOT include the C
//! reference's display support (`USE_DISPLAY`/`display.h`, an ST7789 TFT
//! panel driver) or its `blink()` RGB LED status indicator -- both are
//! deferred to Phase 4 (`llm-display`) per this project's migration plan.
//! This binary is serial-output-only, which is also all the C reference
//! does when `USE_DISPLAY` is set to 0.
//!
//! VERIFICATION STATUS: not compiled -- see the crate README for why (no
//! execution environment available to this port has the Xtensa/ESP-IDF
//! toolchain) and the per-module doc comments for each piece's individual
//! confidence tier.

mod head;
mod partition;
mod psram;
mod vocab;

use esp_idf_svc::hal::sys::esp_timer_get_time;
use llm_core::{llm_forward_with_head_override, Model};
use std::io::Write;
use std::time::Duration;

/// "Once upon a time" -- matches `esp32_llm.ino`'s `PROMPT_IDS` exactly
/// (cross-checked against the embedded vocab table in `vocab.rs`'s doc
/// comment: token 433 decodes to "Once", 447 to " upon", etc).
const PROMPT_IDS: [usize; 4] = [433, 447, 259, 405];
const N_GENERATE: usize = 200;

fn main() {
    // Standard esp-idf-svc std-app boilerplate (matches the confirmed-
    // working `cargo generate esp-rs/esp-idf-template` skeleton this
    // crate's Cargo.toml/`.cargo/config.toml` were built from): patches
    // libc/newlib hooks ESP-IDF needs, then wires up the `log` crate to
    // route through ESP-IDF's own logger.
    esp_idf_svc::sys::link_patches();
    esp_idf_svc::log::EspLogger::initialize_default();

    println!("\n=== ESP32-S3 PLE TinyLM ===");
    // Matches C's `delay(1500)`: gives a USB-CDC host time to enumerate
    // before the first print, so early boot logs aren't lost.
    std::thread::sleep(Duration::from_millis(1500));

    let flash = match partition::map_model_partition() {
        Ok(f) => f,
        Err(e) => {
            log::error!("{e}");
            halt();
        }
    };

    let model = match Model::load(flash) {
        Ok(m) => m,
        Err(e) => {
            log::error!("bad model: {e:?}");
            halt();
        }
    };
    let c = &model.cfg;
    println!(
        "model: V={} D={} L={} H={} F={} P={}  (mapped {:.1} MB)",
        c.vocab,
        c.dim,
        c.n_layers,
        c.n_heads,
        c.ffn,
        c.ple_dim,
        flash.len() as f64 / 1e6
    );
    let _ = std::io::stdout().flush();

    // Cap the staged head to the trained vocab BEFORE staging (see
    // head::stage_and_spawn's doc comment): the tokenizer learned
    // vocab::vocab_n() entries; padded rows above that can never be
    // emitted and have no decode entry, so they're neither staged nor
    // scored.
    let vocab_n = vocab::vocab_n();
    let head = match head::stage_and_spawn(model.tok_emb(), vocab_n) {
        Ok(h) => h,
        Err(e) => {
            log::error!("head staging: {e}");
            halt();
        }
    };

    let mut scratch = psram::alloc_scratch_psram(&model.cfg);
    println!("PSRAM free after alloc: {} KB\n", psram::free_psram_bytes() / 1024);
    let _ = std::io::stdout().flush();

    // The override closure IS `Model.head_matvec` from the C reference --
    // see head::stage_and_spawn's doc comment for why this port wires it
    // in as a closure parameter instead of a function-pointer struct
    // field.
    let mut head_matvec = |x: &[f32], y: &mut [f32]| head.matvec(x, y);

    print!(">>> ");
    let _ = std::io::stdout().flush();
    let mut pos = 0usize;
    let mut tok = 0usize;

    for &t in PROMPT_IDS.iter() {
        tok = t;
        emit(tok);
        llm_forward_with_head_override(&model, tok, pos, &mut scratch, &mut head_matvec);
        pos += 1;
    }

    // No equivalent of C's `llm_profile_reset(&s)` / the per-stage
    // `s.profile.{input,attn,ffn,ple,head}_us` breakdown printed at the
    // end: llm_core's Scratch doesn't carry that instrumentation (it
    // wasn't ported -- see llm-core/src/scratch.rs), only this decode
    // loop's own wall-clock timing around each `llm_forward_with_head_override`
    // call, which is external to llm_core and needs no change there. The
    // overall throughput number (the one the C reference's README
    // actually reports, 102.9ms/step) is still fully and faithfully
    // measured below; only the finer-grained internal breakdown is
    // omitted. Worth adding back to llm_core as a follow-up if the
    // per-stage numbers turn out to matter for on-device tuning.
    let t_start = unsafe { esp_timer_get_time() };
    let mut decode_us: i64 = 0;
    let mut decoded = 0usize;

    for step in 0..N_GENERATE {
        if pos >= model.cfg.seq_len {
            break;
        }
        // Greedy: argmax over the trained vocab only (NOT model.cfg.vocab
        // -- see head::stage_and_spawn's doc comment on why rows at/above
        // vocab_n in s.logits are never written and must not be scored).
        let mut best = 0usize;
        let mut bv = f32::NEG_INFINITY;
        for v in 0..vocab_n {
            if scratch.logits[v] > bv {
                bv = scratch.logits[v];
                best = v;
            }
        }
        tok = best;
        emit(tok);

        let d0 = unsafe { esp_timer_get_time() };
        llm_forward_with_head_override(&model, tok, pos, &mut scratch, &mut head_matvec);
        pos += 1;
        decode_us += unsafe { esp_timer_get_time() } - d0;
        decoded += 1;

        if step % 8 == 0 {
            // Feeds the task watchdog roughly every 8 tokens, matching
            // C's `delay(0)` in the same spot -- a zero-length sleep still
            // yields to the scheduler/WDT reset path without meaningfully
            // slowing generation.
            std::thread::sleep(Duration::from_millis(0));
        }
    }
    let total_us = unsafe { esp_timer_get_time() } - t_start;
    let _ = tok; // last generated token; only its side effects (emit) matter past this point

    println!("\n\n--- {decoded} tokens in {:.2} s ---", total_us as f64 / 1e6);
    if decoded > 0 {
        println!(
            "throughput: {:.2} tok/s   ({:.1} ms/token)",
            decoded as f64 * 1e6 / total_us as f64,
            decode_us as f64 / 1000.0 / decoded as f64
        );
    }
    let _ = std::io::stdout().flush();

    // Matches C's `void loop() { delay(10000); }` -- setup() (this whole
    // function) runs once; loop() just idles forever afterward.
    halt();
}

/// Ports `emit()`: token id -> decode bytes -> write to the serial
/// console. Simplification vs the C reference: C's `emit()` checks
/// `Serial.availableForWrite() >= len` first and skips the write if the
/// USB-CDC TX buffer is full, so generation never stalls when no host is
/// draining it (relevant once a display is attached and the device runs
/// standalone, showing output on-panel instead of over serial -- Phase
/// 4's territory). This port always blocks on the write for now; revisit
/// alongside Phase 4's display module, when running without an attached
/// USB host becomes a real scenario instead of a hypothetical one.
fn emit(tok: usize) {
    if let Some(bytes) = vocab::decode(tok) {
        let _ = std::io::stdout().write_all(bytes);
    }
}

/// Idles forever. Used both where the C reference's `setup()` hits an
/// early `return;` on a fatal init error (after which its `loop()` would
/// just spin anyway) and for this program's own actual `loop()` at the
/// end of `main`.
fn halt() -> ! {
    loop {
        std::thread::sleep(Duration::from_secs(10));
    }
}
