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
use llm_core::{
    llm_forward_profiled_with_matvec_override, llm_forward_with_head_override, Model, Profile, QT,
};
use std::io::Write;
use std::time::Duration;

/// "Once upon a time, there was a little girl who loved her dolls and her
/// baby brother" -- swapped from the original "Once upon a time"
/// (esp32_llm.ino's hardcoded `PROMPT_IDS`, ids [433, 447, 259, 405]) purely
/// because stdin still delivers zero bytes on real hardware (see
/// `read_prompt_ids`'s doc comment below), so this fallback is currently the
/// *only* prompt that ever actually runs. IDs found by greedy longest-match
/// encoding against the real vocab table
/// (`reference-c/esp32-llm-lab/vocab.h`) and round-trip-verified to decode
/// back to this exact string byte-for-byte. Swap this array (and only this
/// array) to try a different topic until the stdin bug is fixed.
const DEFAULT_PROMPT_IDS: [usize; 18] = [
    433, 447, 259, 405, 12, 406, 282, 259, 403, 450, 591, 504, 309, 1719, 265, 309, 1512, 1222,
];
/// Wait on stdin for a prompt before generating?
///
/// `false`, and that is the configuration this port is actually verified
/// against rather than a workaround being papered over: the C reference never
/// reads stdin either -- `esp32_llm.ino` decodes the fixed `PROMPT_IDS` baked in
/// via `vocab.h`. `DEFAULT_PROMPT_IDS` above is that same mechanism, so parity
/// with `golden.txt` is reachable without stdin working at all.
///
/// Left on, as it was, every boot spent 20 seconds waiting on a UART path that
/// has never delivered a byte on this board -- dead time in front of every
/// timing measurement, gating Phase 3 on a console bug Phase 3 does not depend
/// on. The interactive prompt is a feature beyond the C reference's behaviour;
/// it belongs after parity, not in front of it.
///
/// Flip to `true` to test the UART0 driver-mode fix (see `install_uart_driver`).
const STDIN_PROMPT: bool = false;

const N_GENERATE: usize = 400; // bumped from 200 -- still safely under the model's 512-token
                                // hard context ceiling (seq_len, from the model file's own
                                // header) even with a short prompt on top.

/// Reads one line of comma-separated token IDs from stdin (wired to the
/// USB-CDC console by ESP-IDF's std-app console VFS driver -- the same
/// mechanism that already makes `println!`/`print!` reach the serial
/// monitor, just the read direction, which nothing in this project has
/// exercised before this).
///
/// Companion piece: `esp32-ai/esp32-llm-lab/chat_device.py`, a new script
/// living next to (not replacing) `chat.py` -- it reuses `chat.py`'s own
/// `encode()` untouched (per MIGRATION_PLAN.md, that tokenizer stays
/// Python-side, off-chip) and sends its output here instead of piping it
/// into a host-compiled `gen_prompt` the way `chat.py` itself does.
///
/// VERIFICATION STATUS UPDATE (first on-hardware runs): originally built on
/// `std::io::stdin().read_line()`, polled in a loop (a single blocking call
/// returned instantly empty, matching how this whole ecosystem generally
/// does serial I/O -- the C reference's own `emit()` checks
/// `Serial.availableForWrite()` rather than blocking). That polling loop
/// ran cleanly -- no errors, no panics -- but never once received a byte,
/// even after 20s with `chat_device.py` writing within milliseconds of
/// `READY`. Two theories were tested in order:
///
/// 1. Console routing: ESP-IDF's stdio guide says a *secondary* console is
///    output-only ("stdin will only contain data sent by the host to the
///    *primary* console"), and this board's generated sdkconfig had UART0
///    primary / USB-Serial-JTAG secondary -- exactly backwards from the
///    `/dev/cu.usbmodem*` port actually in use. Forcing
///    `CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG=y` as primary (see
///    `sdkconfig.defaults`) reproducibly hung at boot instead (right after
///    `cpu_start: Multicore app`, both after a soft DTR reset and a full
///    power cycle) -- consistent with this board's USB connector actually
///    being wired to an external bridge chip on the UART0 pins rather than
///    to the chip's native USB D+/D- pins, so USB-Serial-JTAG-as-primary
///    pointed the console at a peripheral nothing is attached to. Reverted.
/// 2. `std::io::Stdin` itself: esp-rs/esp-idf-hal#200 documents this
///    directly -- Rust's std stdin implementation is not correctly wired
///    for ESP-IDF's console VFS regardless of which channel is primary
///    ("Rust's std does not allow it... not implemented there for esp"),
///    and its documented workaround is a raw `read(2)` on fd 0 via `libc`
///    instead of going through `std::io::Stdin` at all. That's what this
///    function does now. Still not yet confirmed on hardware -- if this
///    also receives nothing, the next place to look is whether the UART
///    driver needs an explicit `uart_driver_install`/interrupt-mode call
///    that ESP-IDF's std-app skeleton doesn't make automatically.
///
/// Falls back to `DEFAULT_PROMPT_IDS` once the wait budget is exhausted
/// with nothing usable received, so a bare reset with no companion script
/// attached still behaves exactly like every previous run of this
/// firmware -- it just now takes up to `WAIT_BUDGET` to get there instead
/// of returning immediately.
fn read_prompt_ids() -> std::vec::Vec<usize> {
    const WAIT_BUDGET: Duration = Duration::from_secs(20);
    const POLL_INTERVAL: Duration = Duration::from_millis(100);

    println!("READY (send comma-separated token ids, or blank for the default prompt)");
    let _ = std::io::stdout().flush();

    // Bypass std::io::stdin() entirely -- see the doc comment above
    // (esp-rs/esp-idf-hal#200). `read(2)` straight on fd 0 via `libc`,
    // accumulating bytes ourselves until we see a '\n'.
    let mut read_buf = [0u8; 256];
    let mut line: std::vec::Vec<u8> = std::vec::Vec::new();
    let deadline = std::time::Instant::now() + WAIT_BUDGET;
    loop {
        // SAFETY: `read_buf` is a valid, correctly-sized stack buffer that
        // outlives this call; this is a direct, single read(2) syscall with
        // the standard libc contract (returns bytes read, 0 at EOF, -1 with
        // errno set on error/would-block).
        let n = unsafe {
            libc::read(
                libc::STDIN_FILENO,
                read_buf.as_mut_ptr() as *mut libc::c_void,
                read_buf.len(),
            )
        };
        if n > 0 {
            line.extend_from_slice(&read_buf[..n as usize]);
            if let Some(nl_pos) = line.iter().position(|&b| b == b'\n') {
                let trimmed = std::str::from_utf8(&line[..nl_pos])
                    .unwrap_or("")
                    .trim();
                if !trimmed.is_empty() {
                    let ids: std::vec::Vec<usize> = trimmed
                        .split(',')
                        .filter_map(|s| s.trim().parse::<usize>().ok())
                        .collect();
                    if !ids.is_empty() {
                        return ids;
                    }
                    log::warn!("got a line, none parsed as token ids: {trimmed:?}");
                }
                line.clear();
            }
        } else if n < 0 {
            // Use std's own errno decoding (matches what read_line() used to
            // report) rather than poking libc's errno accessor ourselves --
            // WouldBlock/EAGAIN means "nothing arrived yet, ask again", any
            // other kind is unexpected and worth logging.
            let err = std::io::Error::last_os_error();
            if err.kind() != std::io::ErrorKind::WouldBlock {
                log::warn!("stdin read error (still polling): {err:?}");
            }
        }
        // n == 0: nothing this attempt -- keep polling.
        if std::time::Instant::now() >= deadline {
            log::warn!("no usable prompt within {WAIT_BUDGET:?}, using default");
            return DEFAULT_PROMPT_IDS.to_vec();
        }
        std::thread::sleep(POLL_INTERVAL);
    }
}

/// Installs the real UART0 driver and points the console VFS at it.
///
/// Only called when `STDIN_PROMPT` is set. Unproven on hardware, and it
/// reconfigures the same peripheral the working `println!` path uses -- so it
/// stays off the default boot path until confirmed, rather than sitting between
/// every flash and every measurement.
fn install_uart_driver() {
    // Force UART0's console VFS into driver-managed (interrupt-driven) mode.
    //
    // Root cause, established across three independent host-side tests this
    // debugging session (a clean pyserial write from `chat_device.py`, a raw
    // `libc::read` swap on the device side making zero difference, and
    // typing straight into espflash's own verified-bidirectional monitor):
    // stdin never received a single byte on this board's application
    // firmware, in ANY combination -- while `espflash flash` succeeding
    // repeatedly the same session proves RX definitely works at the ROM
    // bootloader stage over this exact link. So the fault is specific to
    // how the *application* console initializes UART0, not the wiring.
    //
    // ESP-IDF's default console mode (no driver installed) has `uart_vfs`
    // read bytes one at a time straight from the hardware FIFO via a ROM
    // helper, with no interrupt handler ever pulling the receiver out of
    // whatever state application-level startup leaves it in. Installing the
    // real UART driver and switching the VFS to use it
    // (`uart_vfs_dev_use_driver`) is ESP-IDF's own documented fix for
    // exactly this class of "stdout fine, stdin silent" symptom. `rx_buffer_size`
    // must exceed the hardware FIFO (128 bytes on this SoC) per
    // `uart_driver_install`'s own contract. `tx_buffer_size = 0` keeps TX
    // blocking/unbuffered, i.e. leaves the println! path that already works
    // untouched. Not yet confirmed on hardware -- next flash is the test.
    unsafe {
        use esp_idf_svc::hal::sys::{uart_driver_install, uart_vfs_dev_use_driver};
        // Real signatures (confirmed against this project's own generated
        // bindings.rs, not guessed): `uart_driver_install`'s uart_num is
        // `uart_port_t` (a `c_uint`/u32), but `uart_vfs_dev_use_driver`'s is
        // a plain `c_int`/i32 -- two different integer types for "which
        // UART" across two functions in the same header.
        let err = uart_driver_install(
            0u32, // uart_port_t == c_uint
            256,  // rx_buffer_size -- must exceed the 128-byte HW FIFO
            0,    // tx_buffer_size -- 0 = blocking/unbuffered, matches current TX behavior
            0,    // queue_size -- not using the UART event queue
            std::ptr::null_mut(),
            0, // intr_alloc_flags
        );
        if err != 0 {
            log::warn!("uart_driver_install failed (err {err}); stdin will likely still be silent");
        } else {
            uart_vfs_dev_use_driver(0i32); // c_int == i32
        }
    }
}

fn main() {
    // Standard esp-idf-svc std-app boilerplate (matches the confirmed-
    // working `cargo generate esp-rs/esp-idf-template` skeleton this
    // crate's Cargo.toml/`.cargo/config.toml` were built from): patches
    // libc/newlib hooks ESP-IDF needs, then wires up the `log` crate to
    // route through ESP-IDF's own logger.
    esp_idf_svc::sys::link_patches();
    esp_idf_svc::log::EspLogger::initialize_default();

    if STDIN_PROMPT {
        install_uart_driver();
    }

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
    println!("PSRAM free after alloc: {} KB", psram::free_psram_bytes() / 1024);

    // One cold sequential pass over the staged head weights -- the same buffer,
    // direction and access pattern the head uses every token, and far larger
    // than the data cache, so nothing is reused. This is the minimum useful
    // part of reference-c/firmware/bandwidth_bench/ and it converts "the head
    // is probably memory-bound" into a bound: at `bw` MB/s, streaming `n` MB
    // per token cannot cost less than 1000*n/bw ms, and whatever the head
    // spends above that is arithmetic that shrinking the weights will never
    // reach. Staging the head as int4 halved `n` and moved the head by 0.0 ms,
    // which is the observation this line exists to explain.
    {
        let (bytes, bw) = head.probe_weight_bandwidth();
        println!(
            "PSRAM read: {:.0} MB/s  ({:.2} MB of head weights => {:.1} ms/token floor)",
            bw,
            bytes as f64 / 1e6,
            bytes as f64 / bw / 1000.0
        );
    }
    println!();
    let _ = std::io::stdout().flush();

    // The override closure IS `Model.head_matvec` from the C reference --
    // see head::stage_and_spawn's doc comment for why this port wires it
    // in as a closure parameter instead of a function-pointer struct
    // field.
    // Locked once for the whole run. See `emit`.
    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    let mut head_matvec = |x: &[f32], y: &mut [f32]| head.matvec(x, y);
    // Every NON-head matvec also goes to both cores. The hook has shipped
    // since Phase 3 with no user -- it was written for esp-dsp, and a
    // dual-core row split needs the same shape. input/attn/ffn/ple were 55% of
    // a token running on one core with the worker parked; splitting by rows is
    // bit-exact (llm-host/tests/matvec_split.rs asserts every split point).
    let mut split_matvec = |t: &QT, x: &[f32], y: &mut [f32]| head.matvec_parallel(t, x, y);

    let prompt_ids = if STDIN_PROMPT {
        read_prompt_ids()
    } else {
        // Same path the C reference takes: a fixed, baked-in prompt.
        println!("prompt: baked-in default ({} ids)", DEFAULT_PROMPT_IDS.len());
        DEFAULT_PROMPT_IDS.to_vec()
    };

    print!(">>> ");
    let _ = std::io::stdout().flush();
    let mut pos = 0usize;
    let mut tok = 0usize;

    for &t in prompt_ids.iter() {
        tok = t;
        emit(&mut out, tok);
        llm_forward_with_head_override(&model, tok, pos, &mut scratch, &mut head_matvec);
        pos += 1;
        // The C reference doesn't feed the watchdog here either (see
        // esp32_llm.ino's own prompt-priming loop) -- fine there, because on
        // real hardware 4 forward passes finish in well under a second. This
        // port's first on-hardware run tripped the 5s task watchdog multiple
        // times just during these 4 iterations, meaning per-token compute is
        // currently far above that baseline (see Cargo.toml's `lto`/
        // `codegen-units` comment -- this and that are both part of chasing
        // the same gap, not independent fixes). A zero-length sleep is a
        // near-free yield back to the scheduler/WDT reset path, same
        // mechanism the generation loop below already uses every 8 steps.
        std::thread::sleep(Duration::from_millis(0));
    }

    // Matches C's `llm_profile_reset(&s)`, called right here (after
    // priming, before the timed generation loop) -- see llm_core::Profile's
    // doc comment for why this is a caller-owned value instead of a field
    // baked into Scratch the way C's `#ifdef LLM_PROFILE` embeds it. Timing
    // starts clean for the generation loop only, same as C: priming's KV
    // cache is empty/growing in a way steady-state decode isn't, so C
    // deliberately excludes it from the averaged breakdown too.
    let mut prof = Profile::default();
    head.profile_reset(); // same boundary, same reason -- exclude priming
    let t_start = unsafe { esp_timer_get_time() };
    let mut decode_us: i64 = 0;
    let mut decoded = 0usize;
    // The remainder outside the profiled stages, split. Wall clock minus the
    // five stages is ~3 ms/token and had never been attributed to anything.
    let mut argmax_us: i64 = 0;
    let mut emit_us: i64 = 0;

    for step in 0..N_GENERATE {
        if pos >= model.cfg.seq_len {
            break;
        }
        // Greedy pick.
        // No scan. Both cores tracked the maximum over their own rows while
        // computing the logits, so this reads two candidates instead of
        // 25,353 floats back out of PSRAM. Same token either way -- see
        // `Head::last_argmax` for why the tie-breaking is identical to a
        // front-to-back scan.
        //
        // `vocab_n` is not passed here because the head was already staged
        // with `rows = vocab_n`, so rows at or above it were never computed
        // and cannot be the maximum. That is the same invariant the old scan
        // relied on, enforced at staging time instead of at pick time.
        let a0 = unsafe { esp_timer_get_time() };
        tok = head.last_argmax();
        let a1 = unsafe { esp_timer_get_time() };
        emit(&mut out, tok);
        let a2 = unsafe { esp_timer_get_time() };
        argmax_us += a1 - a0;
        emit_us += a2 - a1;

        let d0 = unsafe { esp_timer_get_time() };
        llm_forward_profiled_with_matvec_override(
            &model,
            tok,
            pos,
            &mut scratch,
            &mut head_matvec,
            &mut split_matvec,
            &mut || unsafe { esp_timer_get_time() as u64 },
            &mut prof,
        );
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
    // Matches esp32_llm.ino's closing `if (s.profile.calls) { ... }` block
    // exactly, same format string and same divisor (calls*1000, converting
    // an accumulated-microseconds total into an average ms/token per
    // stage) -- see Profile::ms_per_token's doc comment.
    if let Some([input, attn, ffn, ple, head]) = prof.ms_per_token() {
        println!(
            "profile ms/token: input {input:.1} | attn {attn:.1} | ffn {ffn:.1} | ple {ple:.1} | head {head:.1}"
        );
    }
    // attn is a third of a token and was the last stage reported as one
    // number. qkv/proj are the same fp32 matvec that drives ffn and ple; core
    // is the attention proper and the only part that grows with sequence
    // length, since it scans the KV cache for every position so far.
    if let Some([qkv, rope, core, proj]) = prof.attn_detail_ms_per_token() {
        println!(
            "  attn detail:    qkv {qkv:.1} | rope {rope:.2} | core {core:.1} | proj {proj:.1}"
        );
    }
    // The head is ~52% of a token, so it gets its own breakdown. `own + wait` should account for nearly all of the `head`
    // figure above; see Head::profile_ms_per_token for how to read the rest.
    if let Some([quant, own, wait, worker]) = head.profile_ms_per_token() {
        println!(
            "  head detail:    quant {quant:.2} | own-half {own:.1} | wait-worker {wait:.1} | worker-half {worker:.1}"
        );
    }
    // Sync cost of splitting the non-head matvecs. `wait` is the sum of every
    // block on the worker across ~43 matvecs per token: it is the price the
    // split pays, and the thing that decides whether it was worth paying.
    if let Some((calls, wait)) = head.matvec_profile() {
        println!("  split matvecs:  {calls:.0} calls/token | wait-worker {wait:.1} ms/token");
    }
    if decoded > 0 {
        let n = decoded as f64 * 1000.0;
        let stages = decode_us as f64 / n;
        let wall = total_us as f64 / n;
        println!(
            "  outside stages: argmax {:.2} | emit {:.2} | other {:.2} ms/token  (wall {wall:.1} vs forward {stages:.1})",
            argmax_us as f64 / n,
            emit_us as f64 / n,
            wall - stages - (argmax_us + emit_us) as f64 / n
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
/// Takes the already-locked handle rather than calling `std::io::stdout()`
/// itself. That call returns a handle whose every `write_all`/`flush` takes a
/// reentrant lock, so doing it per token paid for two lock acquisitions 400
/// times for no reason. C's `emit` writes to a bare `FILE *`. The flush stays
/// -- see below.
fn emit<W: Write>(out: &mut W, tok: usize) {
    if let Some(bytes) = vocab::decode(tok) {
        let _ = out.write_all(bytes);
        // Explicit flush, added after the first on-hardware run: without it,
        // every token's bytes just accumulate in stdout's internal buffer
        // with no guaranteed flush point in between, so a whole run's worth
        // of output can show up as one delayed burst on the serial monitor
        // instead of token-by-token -- fine for the final measured
        // throughput number (that's wall-clock timed separately, around
        // llm_forward_with_head_override, not around this write), but
        // confusing when eyeballing live pacing or correlating output
        // against the task watchdog's timestamps, which is exactly what
        // debugging the current performance gap needs. Cost is one syscall
        // per token, negligible next to a multi-millisecond forward pass.
        let _ = out.flush();
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
