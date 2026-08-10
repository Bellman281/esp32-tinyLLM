# model/ — the trained model and its dataset

This folder bundles the trained weights and their supporting artifacts in one
documented place, so you don't have to go spelunking through `reference-c/`
to find them. It's a copy for reference and convenience — the artifacts
themselves are produced upstream (see "Where this comes from" below), and
the copies actually consumed by the build still live where the rest of this
repo expects them (`reference-c/esp32-llm-lab/`, `data/`); nothing here is
wired into the Rust build.

**This is the model**: `model.bin` here is `reference-c/esp32-llm-lab/model.bin`
(14.91 MB) — the 28.9M-parameter, 32,768-vocab PLE TinyLM that the top-level
README and every performance number in this repo actually describe. It is
**not** the same file as `reference-c/firmware/model/model.bin` (1.87 MB,
V=4096, ~3.5M params) — an earlier attempt at this folder mistakenly copied
that smaller file instead. See "A real inconsistency in this repo" below.

## What's in this folder

| File | What it is |
|---|---|
| `model.bin` | The trained, quantized model weights. Group-128 ragged int4 weights + fp16 group scales, tied input/output embedding. Config `V=32768 D=96 L=6 H=4 F=66 P=128 seq_len=512 group=128`, parsed directly from the file header. 14.91 MB. |
| `vocab.h` | The tokenizer's token-id → UTF-8 bytes table, 32,768 entries — matches this `model.bin`'s vocab size. Generated data, not logic. |
| `val_v32768.bin` | A stream of validation tokens (vocab size 32,768) used to compute cross-entropy/perplexity in the `ppl` parity tests (`engine/llm-host/tests/ppl_parity.rs`), which load this exact model. |

SHA-256 of the `model.bin` in this folder:

```
3e5870b42d54e9f3a58b7b7d2a483b6f85170b5732bc1af6ac21a488a7fa7bbe
```

## A real inconsistency in this repo (unresolved, flagging rather than guessing)

`reference-c/firmware/esp32_llm/README.md`'s flash command points at
`firmware/model/model.bin` — but that file's actual header is `V=4096 D=128
L=6 H=4 F=415 P=64` (~3.5M params, 1.87 MB), while that same README's own
recorded boot diagnostics text reads `V=32768 D=96 L=6 H=4 F=66 P=128` — the
config of *this* folder's model, not the one the flash command points to.
Those two don't match each other. Likely explanations, not confirmed:
`firmware/model/model.bin` is stale (from an earlier/smaller test config)
and was never refreshed after the real 28.9M-param model was adopted, or the
flash command's path is wrong. Worth resolving upstream in `reference-c/`
(refreshed from `slvDev/esp32-ai`, not hand-edited, per `MIGRATION_PLAN.md`'s
own rule) — not something guessed at here.

One consequence: **this repo has no PyTorch golden-logit reference
(`golden.txt`/`golden.npz`) for the model in this folder.** The only
`golden.txt` that exists (`reference-c/firmware/model/golden.txt`) is paired
with the *other*, smaller V=4096 model — `engine/llm-host/tests/golden.rs`
currently checks Rust's logits against that smaller model, not the one
actually flashed to hardware. The perplexity and CLI parity tests
(`ppl_parity.rs`, `cli_parity.rs`) do correctly exercise this folder's model
(they already point at `esp32-llm-lab/model.bin`), so numeric correctness
for *this* model is covered — just not by the fp32-precision golden-logit
check. A golden run for this model (dump PyTorch logits for a fixed prompt
via `slvDev/esp32-ai`'s export tooling) would close that gap.

## Architecture

A 28.9M-parameter decoder-only transformer, config `V=32768 D=96 L=6 H=4
F=66 P=128` (vocab / hidden dim / layers / heads / FFN dim / per-layer-embed
dim), split-half RoPE, causal self-attention with a persistent KV cache,
SwiGLU FFN, RMSNorm, and a tied input/output embedding — the full math is
implemented once in `reference-c/firmware/common/llm.h` and ported
function-for-function to `engine/llm-core`.

The distinguishing piece is **Gemma-style Per-Layer Embeddings (PLE)**: of
the 28.9M total parameters, roughly 25M (`V * P * L` = 32768 * 128 * 6) live
in a flash-mapped per-layer lookup table rather than a single shared
embedding, and are gated into each layer as `x +=
RMSNorm(ple_proj(gelu(ple_gate(x)) * ple_l))`. This is the same technique
Google introduced in **Gemma 3n** to keep most of a model's parameters out
of the memory that has to hold "hot" transformer weights — here it's
flash/PSRAM standing in for the accelerator-memory constraint the technique
was designed around, which is what makes a 28.9M-parameter model fit and
run on a 512KB-SRAM / 8MB-PSRAM chip at all. See:

- Gemma 3n developer guide (Google): <https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/>
- Gemma model documentation: <https://ai.google.dev/gemma/docs>

Weights are int4, group-wise quantized (group size 128) with fp16 group
scales; the output head can additionally run in int8-staged mode for the
dual-core split described in the top-level README's status table.

Note: the sequence length in the header (`seq_len=512`) is this model's
context window — 512 tokens, not 32K. The "32K" in this model is its
**vocabulary** size (32,768 tokens), not its context length.

## Dataset

The model is trained on **TinyStories** — a synthetic corpus of very short,
simple stories written in a vocabulary a small child would understand,
purpose-built for training and evaluating small language models:

> Eldan, R. & Li, Y. (2023). *TinyStories: How Small Can Language Models Be
> and Still Speak Coherent English?* arXiv:2305.07759.
> <https://arxiv.org/abs/2305.07759>

The upstream training pipeline (in
[slvDev/esp32-ai](https://github.com/slvDev/esp32-ai)'s `research/tinystories/`)
can pull the real TinyStories dataset directly, or fall back to a synthetic
TinyStories-style corpus generated when the real dataset isn't reachable
from the training environment (this was the case for at least one earlier
demo build — see `reference-c/esp32-llm-lab/README.md`). Check the training
run that produced this `model.bin` if you need to know which corpus it
actually used; both produce the same short-story-style output at inference
time.

## Where this comes from

Training, tokenization, quantization, and export are all out of scope for
*this* repo by design (see the top-level `MIGRATION_PLAN.md`) and live
entirely upstream, in Python, in
**[slvDev/esp32-ai](https://github.com/slvDev/esp32-ai)**. Nothing in this
repo regenerates those files; they're consumed as-is. A full from-scratch
walkthrough (including the pure-C host demo and the QEMU device emulator)
is in `reference-c/esp32-llm-lab/README.md`.

## Where it's actually used from

This `model/` folder is a single, documented copy for reference — the build
and test tooling reads the same artifacts from their original location:

- `reference-c/esp32-llm-lab/model.bin` and `vocab.h` — the frozen,
  read-only C reference copy (never hand-edited; refreshed from upstream if
  the model changes). `cli_parity.rs` reads directly from here.
- `data/val_v32768.bin` — read directly by `engine/llm-host`'s perplexity
  parity test (`ppl_parity.rs`), which also loads `esp32-llm-lab/model.bin`.

## Try it yourself (host, no ESP32 needed)

This model + tokenizer also run as a plain C program on a laptop — no
Python ML dependencies, no pip. `reference-c/esp32-llm-lab/` (refreshed to
include `chat.py`, `run_cli.sh`, `bpe32768.json`, `START-HERE.md` alongside
the existing `llm.h`/`gen_prompt.c`/`model.bin`/`vocab.h`) builds the exact
ESP32 inference code and wraps it in a small interactive REPL:

```bash
cd reference-c/esp32-llm-lab
sh run_cli.sh
```

Verified output from this exact `model.bin` (`:temp 0.8`, the default):

```
prompt> Once upon a time
>>> Once upon a time, there was a boy named Timmy. Timmy loved to play with
his toys. One day, Timmy wanted to make a plan. He had a big box of tape
and he wanted to make something new. So, he started his own tape and tried
to fix it. Timmy was very careful and listened to his mom. He decided to
make a new toy with the tape and it went by. The tape was fixed and the
tape was fixed. Timmy was so happy and thanked his mom for
```

`:temp 0` reproduces the chip's exact greedy decode; `:n <N>` sets the
generation length. It writes simple children's-story text — no
question-answering, no instruction-following, by design (see "Dataset"
above).

## Measured performance (real hardware)

Per `reference-c/firmware/esp32_llm/README.md`'s recorded boot diagnostics
(config matches this folder's model, per the note above):

```
model: V=32768 D=96 L=6 H=4 F=66 P=128
head staged int8: 2.53 MB
PSRAM free after alloc: ~5100 KB
```

102.9 ms/model-step (9.72 tok/s compute-only); ~9.5 tok/s including serial
output. Per-stage breakdown: 57.6ms output head, 25.6ms attention, 8.5ms
PLE path, 6.9ms FFN, 4.4ms input. See the top-level README's status table
for how the Rust port's own measurements compare.
