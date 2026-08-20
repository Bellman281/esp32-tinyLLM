# esp32-tinyLLM

**A 28.9M-parameter language model running on an $8 microcontroller — rewritten in Rust,
and faster than the C it was ported from.**

The ESP32-S3 has 512 KB of SRAM. This model has 28.9M parameters. It fits because ~25M of
them live in a flash-mapped Per-Layer Embedding table (Gemma 3n's trick, three orders of
magnitude down) and the rest are 4-bit quantized. It writes short children's stories at
<!--fig:tok_s-->10.50<!--/fig--> tok/s — it cannot answer questions or follow
instructions, which is the point.

![The model generating stories from the CLI](./png/1.jpg)

## C vs Rust, on the same board

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./png/benchmark-stages-dark.svg">
  <img alt="Per-stage inference cost, C vs Rust" src="./png/benchmark-stages.svg">
</picture>

<!-- BEGIN device-table (generated: scripts/plot_stages.py --inject) -->

| ms/token | input | attn | ffn | ple | head | **wall** | tok/s |
|---|---|---|---|---|---|---|---|
| **C reference** | 4.4 | 42.9 | 6.9 | 8.5 | 57.1 | **122.0** | 8.20 |
| **Rust** | 4.0 | 27.1 | 6.5 | 7.9 | 49.7 | **95.2** | 10.50 |
| **Rust, `--fast-mspi`** ‡ | 3.9 | 25.6 | 6.3 | 7.7 | 48.2 | **91.8** | 10.89 |
| ratio | 0.91x | 0.63x | 0.94x | 0.93x | 0.87x | **0.78x** | |

**28.2% faster per token than the C reference it was ported from**, on the same board, with byte-identical output — and faster on 5 of the 5 stages.

‡ **91.8 ms / 10.89 tok/s (32.9% faster than C)** with one flag — `./scripts/run_device.sh --fast-mspi` runs the memory bus at 120 MHz. Same engine, same bit-exact output; not the default because it needs an experimental ESP-IDF flag and Espressif documents 120 MHz as temperature-sensitive outside the commercial range. The ratio row above is the build you get by cloning. Full history — every run, every dead end — in [BENCHMARKING.md](./BENCHMARKING.md).

<!-- END device-table -->

Every figure above is generated from [`benchmarks/device.toml`](./benchmarks/device.toml)
by `scripts/plot_stages.py`. Nothing here is typed by hand, so nothing here can go stale
while the code moves.

## Byte-identical, and the board says so

Both firmwares emit the same 400 tokens, cut off at the same word. The board proves it
rather than being taken at its word: it folds every token id it emits into a digest and
prints it, and [`model/models.toml`](./model/models.toml) pins the value that
`llm-host/tests/token_stream_digest.rs` recomputes on the host. A run that diverges by one
token says so, loudly, in the same output as the timings.

That mattered more than expected. Two optimizations were reverted for being *slower*, and
the digest is what certified each as purely a cost rather than a silent change in what the
model computes. A firmware that is fast because it computes something else is not a result.

Sixteen test files, 38 tests: golden logits against PyTorch, byte-identical CLI output
against the compiled C binary, cross-entropy over real validation tokens, and a bit-exact
equivalence gate for every optimization in the hot path —
[`engine/llm-host/tests/README.md`](./engine/llm-host/tests/README.md).

## Quick start

No hardware needed:

```bash
cd engine/llm-host && cargo test --release   # verify Rust against the C reference
cd demo && sh run_cli.sh                     # chat with the model on your laptop
scripts/bench_host.py                        # time the two engines against each other
```

On a board: `./scripts/run_device.sh` flashes, monitors, and logs the run.

## Docs

| | |
|---|---|
| [**The port story** (PDF)](./docs/esp32-tinyLLM-port-story.pdf) | how the port went — the wrong turns, the measurements that settled them, and the wins |
| [BENCHMARKING.md](./BENCHMARKING.md) | method, full run history, and every dead end with its number |
| [TRAINING.md](./TRAINING.md) | how `model.bin` was made — dataset → training → 4-bit PTQ → export |
| [model/README.md](./model/README.md) | the model registry, provenance, and how to add another |
| [CONTRIBUTING.md](./CONTRIBUTING.md) | ground rules, and how to land a measured change |
| [engine/llm-firmware/README.md](./engine/llm-firmware/README.md) | what runs on the device, and how it maps to the C |

The story's source is [`docs/story.html`](./docs/story.html) — the PDF is rendered from it,
so it can be re-rendered when a number changes rather than redrawn.

## Repo layout

```
engine/            the Rust workspace
  llm-core/          no_std, platform-agnostic model math
  llm-host/          CLI tools + correctness tests
  llm-firmware/      on-device build (needs ESP-IDF + hardware)
  llm-display/       optional demo screen (not started)
model/             trained models, one folder per registry entry, + models.toml
reference-c/       the C engine this port is checked against (build fixes only)
training/          the Python training pipeline, vendored from upstream
demo/              interactive CLI over either engine
scripts/           run_device.sh, plot_stages.py, disasm_*.sh, bench_host.py
```

[`model/models.toml`](./model/models.toml) is the registry: it names model files, benchmark
settings and expected parity values *once*, and both languages read it — Rust via
`llm_host::manifest`, C-side tooling via `scripts/model.sh` — so they cannot drift apart.

## Credits

Ported function for function from [slvDev/esp32-ai](https://github.com/slvDev/esp32-ai).
Also built on [llama2.c](https://github.com/karpathy/llama2.c) (the single-file C inference
style `llm.h` follows), TinyStories (Eldan & Li 2023,
[arXiv:2305.07759](https://arxiv.org/abs/2305.07759)), and Google's Gemma 3n Per-Layer
Embeddings.

## License

Apache 2.0 for this repo's code ([LICENSE](./LICENSE)). `reference-c/` and `training/` are
vendored from upstream and keep their original MIT terms.
