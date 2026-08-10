# esp32-tinyLLM


A ground-up Rust port of the on-device inference engine for the **ESP32-S3
PLE-TinyLM** — a 28.9M-parameter decoder-only transformer with Gemma-style
Per-Layer Embeddings (25M of those params live in a flash-mapped lookup
table), running at ~9.5 tok/s on an $8 chip with 512KB SRAM / 8MB PSRAM /
16MB flash. The original C/C++ implementation lives upstream in
[slvDev/esp32-ai](https://github.com/slvDev/esp32-ai) (Python
training/quantization/export + the C inference engine + the Arduino
firmware); this repo is where the inference engine and the ESP32 platform
code are being rewritten in Rust, function for function, with every step
checked against the C version's own output.

Trained model, dataset, and architecture: [`model/`](./model/).

## Scope

**In scope:** everything that runs on (or verifies) the chip — the
quantized-transformer math, the ESP32 firmware, the host-side correctness
tooling.

**Out of scope, deliberately:** training, quantization, and model export
stay exactly where they are, in Python, in
[`slvDev/esp32-ai`](https://github.com/slvDev/esp32-ai). Nothing
here regenerates `model.bin`, `vocab.h`, or `golden.txt` — they're consumed
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

**Phase 3 now runs on real hardware** on an ESP32-S3-DevKitC (N16R8):
400 tokens of correct text, no watchdog trips. Performance is being closed
against the C reference, which publishes its own per-stage profile in
`reference-c/firmware/esp32_llm/README.md` (102.9 ms/step, 9.72 tok/s
compute-only) -- so every number below is a like-for-like comparison, not an
estimate.

| ms/token | input | attn | ffn | ple | head | total | tok/s |
|---|---|---|---|---|---|---|---|
| C reference | 4.4 | 25.6 | 6.9 | 8.5 | 57.6 | 102.9 | 9.72 |
| Rust, first run | 10.6 | 66.7 | 15.8 | 20.4 | 193.8 | 307.4 | 3.21 |
| + dual-core, byte-pair unroll | 8.0 | 58.9 | 11.8 | 15.1 | 110.9 | 204.8 | 4.79 |
| + wider caches | 7.4 | 53.7 | 11.2 | 14.3 | 87.6 | 174.2 | 5.64 |

Three changes got it from 3.0x slower than C to 1.69x. In order of what they
were worth:

1. **The dual-core head split was not actually dual-core.** `head.rs` pins its
   worker to CPU0, faithfully copying the C reference's
   `xTaskCreatePinnedToCore(..., 0)`. But the C reference is an Arduino sketch,
   and arduino-esp32 runs `loop()` on CPU1; ESP-IDF instead pins the main task
   to CPU0 by default. Both halves of the head matvec were queueing on one
   core, with a task switch between them. `CONFIG_ESP_MAIN_TASK_AFFINITY_CPU1`
   restores the C topology. Same root cause as the IDLE0-starvation backtraces,
   which are now gone.
2. **`QT::matvec_range` did two loads per multiply-accumulate.** C's
   `matvec_q_range` unpacks both nibbles of each packed byte in one pass; the
   Rust port called a `code(r, j)` helper per column, re-reading the same byte
   and branching on `j & 1`. Now unrolled the same way, and asserted
   bit-identical to a naive reference. (`matvec_int8_activations` had the same
   gap and got the identical fix — see "Open perf thread" below.)
3. **The external-memory caches were at ESP-IDF's defaults**, not
   arduino-esp32's: 32KB data cache with a 32-byte line, 16KB instruction
   cache. Doubling all three (48 KB of internal SRAM) took the head from
   110.9 to 87.6 ms -- the head streams 2.43 MB of PSRAM sequentially, which is
   exactly the access pattern a wider cache line helps.

**What is left is not a port bug.** `attention.rs` is now the worst ratio
(2.10x) and it is a line-for-line translation of the same loop in
`reference-c/firmware/common/llm.h` -- same passes, same accumulation order,
same strided kcache/vcache reads. Its access pattern is strided rather than
sequential, which is why the cache change barely moved it (-9%, against -21%
for the head). The remaining gap is most likely GCC's Xtensa backend against
LLVM's on scalar float and integer dot-product loops, and the lever that would
actually beat it is the S3's SIMD unit, not more flag-tuning.

### Open perf thread: real SIMD for the int8 matvec

`QT::matvec_int8_activations`'s inner loop (`g += self.code(r, j) * iq[j] as
i32`) is a straight int8×int8→i32 dot product — the exact shape the ESP32-S3's
128-bit PIE vector unit accelerates via Espressif's `esp-dsp` C library
(`dsps_dp_s8_aes3`, ~4x on comparable dot products). That's not reachable from
`core::simd`/`std::simd` here — Xtensa's LLVM backend doesn't autovectorize to
those instructions, they're only exposed via hand-written assembly or the
esp-dsp C library. Plan: keep `llm-core` platform-agnostic (per
`MIGRATION_PLAN.md`) and add this the same way the dual-core head split
already works — an optional override hook in `llm-firmware` that FFIs into
`dsps_dp_s8_aes3` per group, falling back to the scalar loop when absent.
Needs the row's packed nibbles unpacked to contiguous `i8` first (cost worth
measuring against the MAC savings), and real ESP-IDF hardware to benchmark —
neither available in this sandbox. **Good first issue for a contributor with
ESP-IDF hardware access — see [CONTRIBUTING.md](./CONTRIBUTING.md).**

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
esp32-tinyLLM/
  README.md              <- this file
  CONTRIBUTING.md          <- how to build/test, and what's open to work on
  LICENSE                  <- Apache License 2.0
  MIGRATION_PLAN.md       <- phase-by-phase plan, ground rules, full detail
  model/                    <- trained model + dataset docs
  demo/                     <- interactive CLI demo (Python REPL over the C engine)
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

## Reviewing this repo

The interesting reading, in order: `MIGRATION_PLAN.md` for why the port is
scoped and sequenced this way, `engine/llm-core/src/tensor.rs` and
`model.rs` for the actual ported math (cross-reference against
`reference-c/firmware/common/llm.h`), and `engine/llm-host/tests/` for how
correctness is actually being enforced rather than assumed.

## Contributing

The main open work is closing the Rust-vs-C performance gap in the table
above — see [CONTRIBUTING.md](./CONTRIBUTING.md) for the ground rules
(`reference-c/` is frozen, `llm-core` stays platform-agnostic), how to run
the parity tests, and the current best starting point (the SIMD thread
above).

## References

- [slvDev/esp32-ai](https://github.com/slvDev/esp32-ai) — the
  upstream Python training/export pipeline + original C/Arduino firmware this
  repo ports from.
- Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c) /
  [llm.c](https://github.com/karpathy/llm.c) — the minimal single-file C
  inference style `llm.h` and the host tooling follow.
- TinyStories dataset — Eldan & Li (2023), [arXiv:2305.07759](https://arxiv.org/abs/2305.07759).
- Per-Layer Embeddings — Google's Gemma 3n on-device architecture.

## License

Apache License 2.0 covers this repo's own code — see [LICENSE](./LICENSE).
`reference-c/` is a vendored, unmodified copy of
[slvDev/esp32-ai](https://github.com/slvDev/esp32-ai)'s C/C++/Arduino code
and keeps its original MIT terms — see
[`reference-c/LICENSE`](./reference-c/LICENSE).
