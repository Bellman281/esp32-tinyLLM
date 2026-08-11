# llm-firmware — the on-device binary

Phase 3 of the migration: the ESP32-S3 firmware itself, ported from
`reference-c/firmware/esp32_llm/esp32_llm.ino`. Depends on `llm-core` (the
already-verified, platform-independent inference engine) plus
`esp-idf-hal`/`esp-idf-svc`/`esp-idf-sys` for everything platform-specific:
flash partition mmap, PSRAM allocation, and the dual-core int8-staged output
head.

## Status: **runs on real hardware**

Built with `cargo run --release`, flashed to an ESP32-S3-DevKitC (N16R8),
and generating 400 tokens of correct text with no watchdog trips. The
strongest result: at matched settings on the same board and the same
`model.bin`, this firmware's output is **byte-identical to the C
firmware's** — the same 400 tokens, cut off at the same word. Two
independent implementations of the same math, including the dual-core
int8-staged head, agreeing token for token on real silicon.

Measured throughput and the full per-stage C-vs-Rust breakdown live in
[`BENCHMARKING.md`](../../BENCHMARKING.md); they are deliberately not
duplicated here.

Every FFI guess this port made — the bindgen-mangled partition enum
constants, `esp_idf_hal::task`'s API, the `sdkconfig.defaults` keys — turned
out correct on the first real attempt, after fixing two build-plumbing
issues that had nothing to do with the ported logic itself:

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

### How this was written before it could be compiled

For most of its life this crate could not be built or run anywhere the port
was being written: the Xtensa/ESP-IDF toolchain was not installable in that
environment, and there was no other machine with `cargo`/`rustc` available,
let alone the ESP-IDF C SDK. That constraint is why the source carries so
many `VERIFICATION STATUS` doc comments, and it is worth keeping the record
of how the code was checked before it ever reached a real compiler:

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

Phase 3's exit gate is met — it builds, flashes, runs, and matches the C
firmware's output exactly. Remaining work on this crate:

1. **A clean A/B on `llm-core`'s `matvec_override` hook.** It adds a branch
   to every matvec call and currently has no user here (the `esp-dsp` path
   it exists for is not written). Its cost looks like roughly 2 ms/token,
   but that has not been isolated by a before/after on one build. Worth
   measuring before anyone optimizes around it.
2. **esp-dsp SIMD for the fp32 matvec.** The hook, `QT::matvec_range_chunked`,
   and a measured drift tolerance all ship in `llm-core`; what is missing is
   the ESP-IDF-side work. See `BENCHMARKING.md`'s "What to optimize next"
   for the full picture, including why this is not a guaranteed win.
3. **Phase 4 (`llm-display`)** can start whenever — it builds directly on
   this crate, and hardware has now confirmed the foundation works.

Running it: `cargo run --release` from this directory, against a board whose
`model` partition is programmed (see `BENCHMARKING.md`).

## What's ported here (and what isn't yet)

| esp32_llm.ino | This crate |
|---|---|
| `esp_partition_find_first` / `esp_partition_mmap` | `partition.rs` |
| `heap_caps_malloc(MALLOC_CAP_SPIRAM)`, the `ps()` helper, all of `setup()`'s `s.x = ps(...)` calls | `psram.rs` |
| `stage_head_int8`, `head_worker_main`, `head_matvec_int8`, `dot_i8`, `head_rows_range`, the dual-core task/job plumbing | `head.rs` |
| `vocab.h`'s `VOCAB_BLOB`/`VOCAB_OFF`/`VOCAB_N`, `emit()`'s token lookup | `vocab.rs` (embeds `assets/vocab_blob.bin`/`vocab_off.bin`, generated from the real `vocab.h` by `llm-host`'s `vocab-to-bin` tool and round-trip-verified — see that tool's own doc comment) |
| `setup()`'s boot sequence, prompt priming, greedy-decode loop, throughput printout | `main.rs` |
| `Serial.begin/write` (non-blocking, `availableForWrite()`-gated) | `main.rs`'s `emit()` — **simplified**: always blocks on the write for now. The non-blocking guard exists in C so generation doesn't stall when no USB host is attached (relevant once a display is running standalone); revisiting this is bundled with Phase 4 since that's when "no attached host" becomes a real scenario. |
| `#ifdef LLM_PROFILE` per-stage timing (`input_us`/`attn_us`/`ffn_us`/`ple_us`/`head_us`) | `llm-core`'s `Profile` + `llm_forward_profiled`, printed by `main.rs` as `profile ms/token: input .. \| attn .. \| ffn .. \| ple .. \| head ..` — the same stage boundaries and the same output format the C reference uses, so the two can be compared line for line. The timestamp source is a caller-supplied closure rather than a baked-in clock, keeping `llm-core` platform-agnostic (see `llm-core/src/profile.rs`). |
| `USE_DISPLAY`/`display.h` (ST7789 TFT), `blink()` (RGB status LED) | **deferred to Phase 4** (`llm-display`), per this project's migration plan — this binary is serial-output-only, same as the C reference with `USE_DISPLAY` set to 0. |

## Phase 6 (later, separate): re-platform onto esp-hal, no_std, no FreeRTOS

Once Phase 3 is confirmed working on real hardware, swap the platform layer
only — `llm-core` does not change — for `esp-hal`: no FreeRTOS, no ESP-IDF C
SDK anywhere in the build. The dual-core split becomes `esp-hal`'s core-1
closure spawn instead of a FreeRTOS task. Same exit gate as Phase 3.
