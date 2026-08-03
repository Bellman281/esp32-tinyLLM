# llm-firmware — the on-device binary

Phase 3 of the migration: the ESP32-S3 firmware itself, ported from
`reference-c/firmware/esp32_llm/esp32_llm.ino`. Depends on `llm-core` (the
already-verified, platform-independent inference engine) plus
`esp-idf-hal`/`esp-idf-svc`/`esp-idf-sys` for everything platform-specific:
flash partition mmap, PSRAM allocation, and the dual-core int8-staged output
head.

## Status: written, host-logic-checked, **not yet built for the target**

This crate cannot be compiled or run in any environment available to whoever
(human or Claude) is writing this from the cloud sandbox side: the
Xtensa/ESP-IDF toolchain isn't installable there (no access to the
GitHub-hosted release assets `espup` needs), and there's no other execution
environment with `cargo`/`rustc` at all, let alone the ESP-IDF C SDK. **Only a
machine with `espup`/ESP-IDF actually installed — confirmed working on this
project's own dev machine, where the `esp-idf-template` skeleton already
built successfully — can compile-check this crate.**

Given that constraint, this code was written and checked as rigorously as
possible without a target build:

- Every module's tensor/numerics logic reuses functions `llm-host`'s
  existing test suite already verified bit-for-bit against the C reference
  on real model weights (golden logits, perplexity, CLI byte-for-byte) —
  `QT::unpack_row_int8` and `quantize_activations` in particular, both
  factored out of `llm-core` specifically so this crate's int8 head math is
  the *same code*, not a second hand-written copy that could silently
  drift.
- The crate's own Rust logic — types, borrow checking, the `unsafe` blocks'
  internal consistency, the `FnMut` closure wiring into
  `llm_core::llm_forward_with_head_override` — was host-checked by
  compiling it against a hand-written stub `esp-idf-svc` crate (matching
  the real crate's module paths and function signatures as researched, but
  with dummy bodies) on the ordinary host target. This caught and fixed one
  real bug (`head.rs`'s `actq` slicing tripped Rust's
  `dangerous_implicit_autorefs` lint on a raw-pointer deref) before this
  code ever reached a real machine. It cannot, by construction, catch a
  wrong *real* ESP-IDF API signature — only that this crate's own code is
  internally sound once linked against whatever the stub claims.
- Every place this port had to guess at something un-verifiable from here —
  a bindgen-generated constant's exact spelling, an FFI function's exact
  signature — is flagged inline with a `VERIFICATION STATUS` doc comment
  naming the specific risk and what to trust instead (ESP-IDF's own
  generated bindings, always) if the guess is wrong.

### What to do with a real build

Run `cargo build` from this directory on a machine with `espup`/ESP-IDF
installed (same setup already confirmed working for the
`esp-idf-template` skeleton earlier in this project). Report back whichever
of these happens:

1. **It builds clean.** Flash it (`cargo run`, or `espflash flash --monitor`)
   against a real ESP32-S3 with the `model` partition programmed, compare
   boot diagnostics + measured tok/s against
   `reference-c/firmware/esp32_llm/README.md`'s numbers (102.9 ms/step,
   9.72 tok/s compute-only) — should match within noise.
2. **It fails to build.** The error message is the actual authority on
   whatever this port guessed wrong — most likely candidates, ranked by
   estimated risk:
   - `partition.rs`: the bindgen-mangled spelling of
     `ESP_PARTITION_TYPE_DATA`/`ESP_PARTITION_MMAP_DATA` (guessed as
     `esp_partition_type_t_ESP_PARTITION_TYPE_DATA` /
     `esp_partition_mmap_memory_t_ESP_PARTITION_MMAP_DATA`).
   - `head.rs`/`psram.rs`: `esp_idf_hal::task`'s exact API (`create`,
     `current`, `notify`, `wait_notification`) and `esp_idf_hal::cpu::Core`
     — researched from esp-idf-hal's own source, not from a build, so the
     types/argument order should be right but the exact module path or a
     since-changed signature is possible.
   - `sdkconfig.defaults`: any `CONFIG_*` key ESP-IDF's `menuconfig`
     doesn't recognize (PSRAM octal mode, flash size, custom partition
     table, CPU freq — see that file's own comments for what each guess is
     based on).
   Fix the specific line the compiler points at, using ESP-IDF's own
   generated bindings/headers as ground truth; everything else in the
   file was written to need no other changes.

Either way, report back (build log, or the diagnostics/tok-s output) before
Phase 4 (`llm-display`) starts — this crate is the foundation the display
work builds on, so it's worth confirming solid first.

## What's ported here (and what isn't yet)

| esp32_llm.ino | This crate |
|---|---|
| `esp_partition_find_first` / `esp_partition_mmap` | `partition.rs` |
| `heap_caps_malloc(MALLOC_CAP_SPIRAM)`, the `ps()` helper, all of `setup()`'s `s.x = ps(...)` calls | `psram.rs` |
| `stage_head_int8`, `head_worker_main`, `head_matvec_int8`, `dot_i8`, `head_rows_range`, the dual-core task/job plumbing | `head.rs` |
| `vocab.h`'s `VOCAB_BLOB`/`VOCAB_OFF`/`VOCAB_N`, `emit()`'s token lookup | `vocab.rs` (embeds `assets/vocab_blob.bin`/`vocab_off.bin`, generated from the real `vocab.h` by `llm-host`'s `vocab-to-bin` tool and round-trip-verified — see that tool's own doc comment) |
| `setup()`'s boot sequence, prompt priming, greedy-decode loop, throughput printout | `main.rs` |
| `Serial.begin/write` (non-blocking, `availableForWrite()`-gated) | `main.rs`'s `emit()` — **simplified**: always blocks on the write for now. The non-blocking guard exists in C so generation doesn't stall when no USB host is attached (relevant once a display is running standalone); revisiting this is bundled with Phase 4 since that's when "no attached host" becomes a real scenario. |
| `#ifdef LLM_PROFILE` per-stage timing (`input_us`/`attn_us`/`ffn_us`/`ple_us`/`head_us`) | **not ported** — `llm-core`'s `Scratch` doesn't carry this instrumentation (see `llm-core/src/scratch.rs`). Overall throughput/ms-per-token *is* still measured and printed, via wall-clock timing in `main.rs`'s decode loop, which needs no change inside `llm-core` to work. Could be added to `llm-core` later if the finer breakdown turns out to matter for on-device tuning. |
| `USE_DISPLAY`/`display.h` (ST7789 TFT), `blink()` (RGB status LED) | **deferred to Phase 4** (`llm-display`), per this project's migration plan — this binary is serial-output-only, same as the C reference with `USE_DISPLAY` set to 0. |

## Phase 6 (later, separate): re-platform onto esp-hal, no_std, no FreeRTOS

Once Phase 3 is confirmed working on real hardware, swap the platform layer
only — `llm-core` does not change — for `esp-hal`: no FreeRTOS, no ESP-IDF C
SDK anywhere in the build. The dual-core split becomes `esp-hal`'s core-1
closure spawn instead of a FreeRTOS task. Same exit gate as Phase 3.
