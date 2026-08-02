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
| 3 | `llm-firmware` — on-device build (`esp-idf-hal`, std, FreeRTOS) | ⏳ not started — needs the Xtensa/ESP-IDF toolchain and physical hardware |
| 4 | `llm-display` — optional OLED/TFT demo screen | not started, deferred |
| 5 | `bandwidth_bench` port | not started, optional |
| 6 | Drop FreeRTOS — re-platform onto `esp-hal` (no_std, bare metal) | not started, after Phase 3 |

**This PR covers Phases 0–2.** Nothing here has run on real hardware yet —
everything below was verified on a laptop, against the C reference, with
no ESP32 involved. Phase 3 is tracked separately.

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
    llm-firmware/              on-device build (Phase 3 — not started, README only)
    llm-display/                optional demo screen (Phase 4 — not started, README only)
```

## Building

```bash
cd engine/llm-host
cargo build --release
```

No network access is needed beyond the first `cargo build` (`half` and
`libm` are the only dependencies `llm-core` has, and they'll be cached in
`Cargo.lock` after that). No ESP32 hardware or toolchain is needed for
anything in this repo yet — that starts with Phase 3.

## Reviewing this PR

The interesting reading, in order: `MIGRATION_PLAN.md` for why the port is
scoped and sequenced this way, `engine/llm-core/src/tensor.rs` and
`model.rs` for the actual ported math (cross-reference against
`reference-c/firmware/common/llm.h`), and `engine/llm-host/tests/` for how
correctness is actually being enforced rather than assumed.
