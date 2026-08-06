# esp32-Rust

A ground-up Rust port of the on-device inference engine for the **ESP32-S3
PLE-TinyLM** — a 28.9M-parameter decoder-only transformer with Gemma-style
Per-Layer Embeddings (25M of those params live in a flash-mapped lookup
table), running at ~9.5 tok/s on an $8 chip with 512KB SRAM / 8MB PSRAM /
16MB flash. The original C/C++ implementation lives in
[`vendor/esp32-ai`](https://github.com/Bellman281/esp32-ai) (Python
training/quantization/export + the C inference engine + the Arduino
firmware); this repo is where the inference engine and the ESP32 platform
code are being rewritten in Rust, function for function, with every step
checked against the C version's own output.

## Scope

**In scope:** everything that runs on (or verifies) the chip — the
quantized-transformer math, the ESP32 firmware, the host-side correctness
tooling.

**Out of scope, deliberately:** training, quantization, and model export
stay exactly where they are, in Python, in `vendor/esp32-ai`. Nothing here
regenerates `model.bin`, `vocab.h`, or `golden.txt` — they're consumed
as-is, produced upstream. See [`MIGRATION_PLAN.md`](./MIGRATION_PLAN.md)
for the full reasoning and the ground rules this port follows.

## Status

| Phase | What | Status |
|---|---|---|
| 0 | Workspace scaffold, frozen C reference copied in | ✅ done |
| 1 | `llm-core` — portable `no_std` model math | ✅ done, parity-verified |
| 2 | `llm-host` — Rust CLI tools + correctness test suite | ✅ done, parity-verified |
| 3 | `llm-firmware` — on-device build (`esp-idf-hal`, std, FreeRTOS) | ✅ runs on an ESP32-S3-DevKitC — generates correct text; performance being closed against the C reference (see below) |
| 4 | `llm-display` — optional OLED/TFT demo screen | not started, deferred |
| 5 | `bandwidth_bench` port | not started, optional |
| 6 | Drop FreeRTOS — re-platform onto `esp-hal` (no_std, bare metal) | not started, after Phase 3 |

**Phases 0–2 are verified on a laptop** against the C reference, with no
ESP32 involved — that is what the parity table below covers.

**Phase 3 now runs on real hardware.** First on-silicon run: 400 tokens of
correct text on an ESP32-S3-DevKitC (N16R8), at 3.21 tok/s against the C
reference's 9.5. The per-token profile pointed at the output head — 193.8 ms
of a 307.4 ms token — and comparing the two sources found two causes, both
now fixed and awaiting a re-measure:

- **The dual-core head split was not actually dual-core.** `head.rs` pins its
  worker to CPU0, faithfully copying the C reference's
  `xTaskCreatePinnedToCore(..., 0)`. But the C reference is an Arduino sketch,
  and arduino-esp32 runs `loop()` on CPU1; ESP-IDF instead pins the main task
  to CPU0 by default. So both halves of the head matvec were queueing on one
  core, with a task switch between them. `CONFIG_ESP_MAIN_TASK_AFFINITY_CPU1`
  in `sdkconfig.defaults` restores the C topology. This is also why the
  backtraces showed IDLE0 starving.
- **`QT::matvec_range` was doing two loads per multiply-accumulate.** C's
  `matvec_q_range` unpacks both nibbles of each packed byte in one pass; the
  Rust port called a `code(r, j)` helper per column, re-reading the same byte
  and branching on `j & 1` each time — on the path that owns attention, FFN
  and PLE. Now unrolled the same way C does it, and asserted bit-identical to
  the naive form by `llm-core`'s own `byte_pair_unroll_matches_naive_bit_for_bit`.

## Verified parity (Phases 1–2)

Every claim below is a test in `engine/llm-host/tests/`, not a spot check.
Run them yourself:

```bash
cd engine/llm-host
cargo test --release                              # fp32-activations path
cargo test --release --features int8-activations  # int8-activations path
```

- **Golden logits vs. PyTorch** (`golden.rs`): the fp32-activations build
  matches the model's own `golden.txt` reference to float32 precision —
  max abs diff `0.00001`, the same precision the C reference's own
  `verify.c` achieves. The int8-activations build is checked differently
  (see the test's doc comment): against the C reference's *own* int8
  numerics, which it also fails to match PyTorch's fp32 golden by
  construction. Rust matches C's int8 divergence to 5 decimal places.
- **CLI output vs. the compiled C binary** (`cli_parity.rs`): `host-generate`'s
  stdout is byte-for-byte identical to `host_generate.c`'s output, same
  model, same prompt.
- **Perplexity vs. the C reference** (`ppl_parity.rs`), over real validation
  tokens: fp32-activations matches the C reference's printed cross-entropy
  exactly (`2.6671`). int8-activations lands within a noise floor measured
  directly — rebuilding the *C reference itself* with `-march=native`
  instead of `-O2` shifts its own result by more than the Rust-vs-C gap.

Two real bugs were caught by this suite during the port (not eyeballing):
an out-of-bounds scratch-buffer size for one model config, and a
rounding-mode mismatch against C's `lrintf` (round-half-to-even vs.
round-half-away-from-zero). Both are fixed; see `MIGRATION_PLAN.md` for
detail.

## Repo layout

```
esp32-Rust/
  README.md              <- this file
  MIGRATION_PLAN.md       <- phase-by-phase plan, ground rules, full detail
  reference-c/             <- frozen, read-only copy of the C/C++ engine
                               (verified against this, never edited)
  data/                     <- validation tokens for the ppl parity test
  engine/                  <- the Rust workspace
    llm-core/                 no_std, platform-agnostic model math (Phase 1)
    llm-host/                 std CLI tools + correctness tests (Phase 2)
    llm-firmware/              on-device build (Phase 3 — runs on hardware)
    llm-display/                optional demo screen (Phase 4 — not started, README only)
```

## Building

```bash
cd engine/llm-host
cargo build --release
```

No network access is needed beyond the first `cargo build` (`half` and
`libm` are the only dependencies `llm-core` has, and they'll be cached in
`Cargo.lock` after that). No ESP32 hardware or toolchain is needed for any of
that — `engine/llm-firmware` is the part that needs both, and it builds and
flashes on its own (`cd engine/llm-firmware && cargo run --release`).

## Reviewing this PR

The interesting reading, in order: `MIGRATION_PLAN.md` for why the port is
scoped and sequenced this way, `engine/llm-core/src/tensor.rs` and
`model.rs` for the actual ported math (cross-reference against
`reference-c/firmware/common/llm.h`), and `engine/llm-host/tests/` for how
correctness is actually being enforced rather than assumed.
