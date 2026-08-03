# llm-firmware — the on-device binary

Phase 3 of the migration: the ESP32-S3 firmware itself, ported from
`reference-c/firmware/esp32_llm/esp32_llm.ino`. Depends on `llm-core` (the
already-verified, platform-independent inference engine) plus
`esp-idf-hal`/`esp-idf-svc`/`esp-idf-sys` for everything platform-specific:
flash partition mmap, PSRAM allocation, and the dual-core int8-staged output
head.

## Status: **builds clean against the real Xtensa/ESP-IDF toolchain**

`cargo build` (dev profile) succeeds on the project's own dev machine (the
only place with `espup`/ESP-IDF actually installed — this can't be compiled
anywhere the cloud-sandbox/Claude side of this project has execution access,
see below). Every FFI guess this port made — the bindgen-mangled partition
enum constants, `esp_idf_hal::task`'s API, the `sdkconfig.defaults` keys —
turned out correct on the first real attempt, after fixing two
build-plumbing issues that had nothing to do with the ported logic itself:

1. `llm-firmware` needed an empty `[workspace]` table in its own `Cargo.toml`
   to opt out of `engine/Cargo.toml`'s workspace (Cargo doesn't auto-detect
   "not a member" from a parent directory).
2. `CONFIG_PARTITION_TABLE_CUSTOM`/`_FILENAME` in `sdkconfig.defaults` don't
   work with `esp-idf-sys`'s build (a known, documented limitation --
   [esp-rs/esp-idf-sys#395](https://github.com/esp-rs/esp-idf-sys/issues/395)):
   its synthetic CMake project resolves a custom partition table's filename
   against its own `OUT_DIR`, not the crate root, so a relative path can
   never be found there. Fix: don't set those keys at all; pass the real
   table at flash time instead (`.cargo/config.toml`'s `runner` now does
   `espflash flash --partition-table partitions.csv --monitor`).

**Not yet done:** a release-profile build (`cargo build --release` — the
profile the ms/token measurement in `main.rs` actually depends on), and
anything involving real hardware (no board is attached to this project yet
-- see "What's next" below).

Before those two fixes, this crate couldn't be compiled or run in any
environment available to whoever (human or Claude) was writing the original
port from the cloud sandbox side: the Xtensa/ESP-IDF toolchain isn't
installable there (no access to the GitHub-hosted release assets `espup`
needs), and there's no other execution environment with `cargo`/`rustc` at
all, let alone the ESP-IDF C SDK. Given that constraint, the original port
was written and checked as rigorously as possible without a target build,
before ever reaching a real compiler:

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

### What's next

1. `cargo build --release` from this directory — same code, the profile
   that actually matters for the tok/s comparison (`opt-level = 3`, matching
   the C reference's `-O3`). Should also build clean; if it doesn't, that's
   new information the dev-profile build didn't surface.
2. Once there's an ESP32-S3 board to flash (none is attached to this
   project yet): `cargo run` (or `espflash flash --partition-table
   partitions.csv --monitor` directly) against a board with the `model`
   partition programmed, then compare boot diagnostics + measured tok/s
   against `reference-c/firmware/esp32_llm/README.md`'s numbers (102.9
   ms/step, 9.72 tok/s compute-only) — should match within noise. This is
   the real exit gate for Phase 3; everything above it is necessary but not
   sufficient.

Only after hardware confirms working should Phase 4 (`llm-display`) start —
it builds directly on this crate.

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
