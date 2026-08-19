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

That pipeline is now *documented and vendored* here for reference, without
becoming part of any build: [`TRAINING.md`](./TRAINING.md) walks it end to end
(dataset → tokenizer → training → 4-bit PTQ → export → `vocab.h`), and
[`training/`](./training/) holds the upstream Python unmodified, under the same
refresh-don't-patch rule `reference-c/` lives under. It also records a real open
blocker: the current upstream exporter writes a header format this repo's
readers reject.

## Status

| Phase | What | Status |
|---|---|---|
| 0 | Workspace scaffold, C reference copied in | ✅ done |
| 1 | `llm-core` — portable `no_std` model math | ✅ done, parity-verified |
| 2 | `llm-host` — Rust CLI tools + correctness test suite | ✅ done, parity-verified |
| 3 | `llm-firmware` — on-device build (`esp-idf-hal`, std, FreeRTOS) | ✅ runs on an ESP32-S3-DevKitC — correct text, performance being closed |
| 4 | `llm-display` — optional OLED/TFT demo screen | not started, deferred |
| 5 | `bandwidth_bench` port | not started, optional |
| 6 | Drop FreeRTOS — re-platform onto `esp-hal` (no_std, bare metal) | not started, after Phase 3 |

Phases 0–2 are verified on a laptop against the C reference. Phase 3 runs on
real hardware: 400 tokens of correct text, no watchdog trips.

### C vs. Rust, measured

Both rebuilt and flashed to the same board, same `model.bin`, same 400-token
run — not quoted from upstream. Full methodology, hardware spec, and
reproduction steps: [`BENCHMARKING.md`](./BENCHMARKING.md).

| ms/token | input | attn | ffn | ple | head | total | tok/s |
|---|---|---|---|---|---|---|---|
| **C reference** | 4.4 | 42.9 | 6.9 | 8.5 | 57.1 | **119.8** | 8.20 |
| **Rust** | 7.8 | 54.4 | 11.9 | 15.2 | 90.3 | **179.6** | 5.47 |
| ratio | 1.77x | 1.27x | 1.72x | 1.79x | 1.58x | **1.50x** | |
| absolute gap | +3.4 | +11.5 | +5.0 | +6.7 | **+33.2** | +59.8 | |

**The two firmwares emit byte-identical output** — the same 400 tokens, same
punctuation, cut off at the same word. Two independent implementations of
the same math agreeing token for token on real silicon; the device-level
counterpart to the host parity suite.

**Where the gap is:** the head is 56% of it in absolute terms and is
PSRAM-bandwidth-bound; `ple`/`input`/`ffn` have the worst ratios and all run
through the same fp32 matvec. Attention is the *closest* stage to parity at
1.27x — an earlier README claimed 2.10x and named it the top target, but
that was a benchmark artifact from comparing runs at different token counts
(the C sketch defaults to 200 tokens/4-token prompt, Rust used 400/18).
`benchmark_settings_match.rs` now fails the build if that drifts again. See
[`BENCHMARKING.md`](./BENCHMARKING.md) for the C-vs-C measurement isolating
it, the optimization history, and what to optimize next.

## Running it

```bash
# host, no hardware: Rust
cd engine/llm-host && cargo test --release

# host, no hardware: the C reference, driven by the model registry
cc -O3 -o /tmp/gen_prompt reference-c/esp32-llm-lab/gen_prompt.c -lm
/tmp/gen_prompt "$(scripts/model.sh bin)" \
                "$(scripts/model.sh n_generate)" \
                "$(scripts/model.sh prompt_ids)" 0

# host, no hardware: time the two engines against each other
scripts/bench_host.py

# on hardware
cd engine/llm-firmware && cargo run --release
```

Host and device output diverge after ~100 tokens on both ports, and that is
expected: the firmware stages the output head as int8 while the host tools
run it in fp32, and greedy decode eventually turns that into a different
token pick. `ppl_parity.rs` bounds the difference.

Flashing the C firmware, and the full set of host C tools:
[`BENCHMARKING.md`](./BENCHMARKING.md).

## The model registry

[`model/models.toml`](./model/models.toml) names the model files, benchmark
settings, and expected parity values once. Both languages read it — Rust via
`llm_host::manifest`, C-side tooling via `scripts/model.sh` — so paths and
settings cannot drift apart between them:

```bash
scripts/model.sh list                # tinystories-v32768, golden-v4096
scripts/model.sh bin                 # absolute path to the default model
scripts/model.sh prompt_ids          # 433,447,259,405,...
```

Model *dimensions* are deliberately not in it: `model.bin`'s header already
carries them and both engines parse it at load time. Adding a model, and why
the parser is hand-written rather than a TOML crate:
[`model/README.md`](./model/README.md).

## Verified parity (Phases 1–2)

Every claim below is a test in `engine/llm-host/tests/`, not a spot check.
Run them yourself:

```bash
cd engine/llm-host
cargo test --release                              # fp32-activations path
cargo test --release --features int8-activations  # int8-activations path
```

- **Golden logits vs. PyTorch** (`golden.rs`): fp32 logits match the
  model's `golden.txt` to max abs diff `0.00001` — the same precision the C
  reference's own `verify.c` achieves. The int8 build is checked against C's
  *own* int8 numerics instead (5 decimal places); see the test's doc comment
  for why that's the right bar.
- **CLI output vs. the compiled C binary** (`cli_parity.rs`): `host-generate`'s
  stdout is byte-for-byte identical to `host_generate.c`'s output, same
  model, same prompt.
- **Perplexity vs. the C reference** (`ppl_parity.rs`), over real validation
  tokens: fp32-activations matches the C reference's printed cross-entropy
  exactly (`2.6671`). int8-activations lands within a noise floor measured
  directly — rebuilding the *C reference itself* with `-march=native`
  instead of `-O2` shifts its own result by more than the Rust-vs-C gap.
- **The registry itself** (`manifest.rs`): every path `model/models.toml`
  declares must resolve, and unknown keys are an error rather than a silent
  skip.
- **Benchmark settings** (`benchmark_settings_match.rs`): both firmwares'
  `PROMPT_IDS`/`N_GENERATE` must match the registry — the regression gate
  for the artifact described above.
- **SIMD-reordering tolerance** (`matvec_simd_tolerance.rs`): measures how
  far a SIMD-shaped reordering of `matvec_range`'s `f32` accumulation moves
  validation cross-entropy (~3–4e-8), so a future `esp-dsp` implementation
  has an empirical bound rather than a guessed one.

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
  BENCHMARKING.md          <- how the C-vs-Rust numbers were measured, and
                               what to optimize next
  ROADMAP.md               <- what to do next, in order, and what's still
                               unexplained
  TRAINING.md              <- how model.bin was made: dataset -> tokenizer ->
                               training -> 4-bit PTQ -> export -> vocab.h
  training/                <- that pipeline's Python, vendored unmodified from
                               slvDev/esp32-ai (reference only, not built)
  model/                    <- trained model, dataset docs, and models.toml
                               (the registry both C and Rust read)
  scripts/                  <- model.sh, the C side's registry reader
  demo/                     <- interactive CLI demo (Python REPL over either
                               engine — chat.py drives C, chat_rust.py Rust)
  reference-c/             <- the C/C++ engine this port is checked against
                               (build-fix edits only — see its own README)
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

The main open work is closing the Rust-vs-C performance gap.
[`ROADMAP.md`](./ROADMAP.md) sequences it — two cheap unknowns gate most of
the expensive work, so start there rather than with the SIMD items —
and [`BENCHMARKING.md`](./BENCHMARKING.md)'s "What to optimize next" has the
per-lever detail. See
[CONTRIBUTING.md](./CONTRIBUTING.md) for the ground rules (`reference-c/`
takes build fixes only, `llm-core` stays platform-agnostic) and how to run
the parity tests.

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
`reference-c/` is a vendored copy of
[slvDev/esp32-ai](https://github.com/slvDev/esp32-ai)'s C/C++/Arduino code
(build-configuration edits only, all recorded in its README) and keeps its
original MIT terms — see
[`reference-c/LICENSE`](./reference-c/LICENSE).
