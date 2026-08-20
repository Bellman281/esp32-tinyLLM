# model/ — this repo's trained models

**This folder is canonical.** The registry points here, the parity tests read
from here, and `scripts/model.sh` resolves here. It is not a convenience copy
of something else — an earlier version of this file said it was, and that is
no longer true.

## Layout

One folder per registry entry, named exactly as the `models.toml` section:

```
model/
  models.toml                    <- the registry (see below)
  README.md                      <- this file
  tinystories-v32768/            <- one folder per model, named as its
    model.bin                       models.toml section
    vocab.h
    val.bin
```

Adding a model means dropping in a folder and adding a `[section]` to
`models.toml` — nothing else in the repo needs editing, which is the whole
point of the registry. `manifest.rs`'s `real_registry_parses_and_its_files_exist`
test fails on a path that doesn't resolve, so a typo can't ship.

Conventional filenames inside a model folder (declare in `models.toml`
whichever the model actually has):

| file | what it is |
|---|---|
| `model.bin` | the exported, quantized weights |
| `vocab.h` | token-id → UTF-8 bytes decode table |
| `val.bin` | validation tokens for the perplexity parity test |
| `golden.txt` | PyTorch reference logits, if one was produced |

One exception, deliberate: **`golden-v4096` has no folder here.** It is
upstream's own 3.5M-param fixture living in `reference-c/firmware/model/`, and
`reference-c/` is a frozen mirror that is refreshed wholesale, not reorganized
(CONTRIBUTING ground rule 1). It stays registered by its path over there.

## The model that ships

`tinystories-v32768/model.bin` (14.91 MB) — the 28.9M-parameter PLE TinyLM
that the top-level README and every performance number in this repo describe.
It is **not** `reference-c/firmware/model/model.bin` (1.87 MB, V=4096, ~3.5M
params) — an earlier attempt at this folder mistakenly copied that smaller
file instead. See "A real inconsistency in this repo" below.

**Provenance:** it was trained on 2 August 2026 on a rented VPS GPU by this
repo's author, using the upstream pipeline (now vendored at
[`training/`](../training/) and walked through in
[`TRAINING.md`](../TRAINING.md)). It is *not* a download of upstream's
published release — that artifact is a different file:

| | bytes | sha256 |
|---|---|---|
| this repo's `model.bin` | 14,912,332 | `3e5870b4…7fa7bbe` |
| `slvDev/esp32-ai-tinystories` on HuggingFace | 14,912,348 | `1d8326c0…b45477e` |

The 16-byte difference is the header, not the weights: upstream's current
exporter writes a 56-byte versioned header, this file carries the older
40-byte one. Same architecture, so the tensor payload is the same size either
way. See [`TRAINING.md`](../TRAINING.md)'s "The two header formats".

`reference-c/esp32-llm-lab/` also contains a `model.bin` and `vocab.h`. Those
belong to the frozen C mirror — `gen_prompt.c` `#include`s that `vocab.h` at
compile time — and are left alone on purpose. Nothing in the registry points
at them.

## What's in `tinystories-v32768/`

| File | What it is |
|---|---|
| `model.bin` | The trained, quantized model weights. Group-128 ragged int4 weights + fp16 group scales, tied input/output embedding. Config `V=32768 D=96 L=6 H=4 F=66 P=128 seq_len=512 group=128`, parsed directly from the file header. 14.91 MB. |
| `vocab.h` | The tokenizer's token-id → UTF-8 bytes table, **25,353** entries (`VOCAB_N`). Generated data, not logic. Note this is *not* the 32,768 in `model.bin`'s header — see "Two different vocab numbers" below. |
| `val.bin` | A stream of validation tokens (vocab size 32,768) used to compute cross-entropy/perplexity in the `ppl` parity tests (`engine/llm-host/tests/ppl_parity.rs`), which load this exact model. |
| `models.toml` (in `model/`, not the model folder) | **The model registry** — the single place naming which model files exist, where they live, and what each is expected to produce. Read by Rust (`llm_host::manifest`) and by the C-side tooling (`scripts/model.sh`), so both get their paths, prompt ids, window counts, and expected cross-entropy from one source. See the top-level README's "The model registry". |

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

## Two different vocab numbers

`model.bin`'s header says `V=32768`. `vocab.h` says `VOCAB_N 25353`. Both are
right, and they mean different things:

| number | what it is |
|---|---|
| **32,768** | rows in the token embedding and the PLE table. This is what `V` in the header means, and what `V * P * L` = 25M PLE parameters is computed from. |
| **25,353** | entries the tokenizer actually has — the only ids that can ever be produced or decoded. |

The upstream pipeline asks for a 32,768-token BPE (`prepare.py --vocab 32768`,
trained on a 40 MB slice) but the tokenizer it produced has 25,353 entries. The
model is still built with `vocab_size=32768`, leaving rows 25353–32767 padded
and untrained. The likely cause is the BPE trainer exhausting useful merge
candidates before reaching the target — TinyStories' effective vocabulary is
small — but that is inferred from the artifacts, not measured; re-running
`prepare.py` would confirm it. They are never scored on-device because no id above 25,352 can be
emitted — but the output head does compute all 32,768 logits, so roughly 23% of
the head's work is spent on rows that can never win. The head is already the
largest single term in the C-vs-Rust gap (see
[`BENCHMARKING.md`](../BENCHMARKING.md)), which makes this worth knowing.

Upstream's current export format carries `output_vocab` separately from
`input_vocab` precisely so a runtime can skip that work, and `llm-core` now
reads it — but this file is in the older format, which cannot express the
distinction, so all 32,768 rows are still scored for *this* model. See
[`TRAINING.md`](../TRAINING.md)'s "The two header formats".

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

The *pipeline* — training, tokenization, quantization, export — is written in
Python and originates upstream in
**[slvDev/esp32-ai](https://github.com/slvDev/esp32-ai)**. It is vendored here
unmodified at [`training/`](../training/), with a walkthrough in
[`TRAINING.md`](../TRAINING.md), for reference only: no build target runs it.

The *artifacts* in this folder were produced by running that pipeline —
trained 2 August 2026 on a rented VPS GPU by this repo's author, not
downloaded from upstream's release. Regenerating them is a manual act
performed outside any build here; see `TRAINING.md`, including the open
export-format blocker that currently prevents a clean re-run.

A full from-scratch walkthrough of the C side (the pure-C host demo and the
QEMU device emulator) is in `reference-c/esp32-llm-lab/README.md`.

## Where it's used from

Every consumer goes through [`models.toml`](./models.toml). Nothing hardcodes
a path:

- `model/tinystories-v32768/{model.bin, vocab.h, val.bin}` — registered as
  `tinystories-v32768`'s `bin`, `vocab` and `val_tokens`. This is the
  `default`, so `scripts/model.sh bin` and `llm_host::manifest::default()`
  resolve here.
- `reference-c/firmware/model/model.bin` and `golden.txt` — upstream's
  smaller fixture, registered separately as `golden-v4096` because it is the
  only model with a PyTorch golden-logit reference (see the inconsistency
  note above). `golden.rs` names it explicitly rather than following
  `default`, so repointing `default` at a new model can't silently move that
  check onto a model with no golden file.

So moving a file means editing `models.toml`, not hunting through test files
for hardcoded paths — which is exactly what it exists to prevent.

## Try it yourself (host, no ESP32 needed)

This model + tokenizer also run as a plain C program on a laptop — no
Python ML dependencies, no pip. [`demo/`](../demo/) builds the exact ESP32
inference code from `reference-c/esp32-llm-lab/` and wraps it in a small
interactive REPL:

```bash
cd demo
sh run_cli.sh
```

See [`demo/README.md`](../demo/README.md) for a screenshot and more detail.

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

**The timing numbers live in [BENCHMARKING.md](../BENCHMARKING.md)** —
that is the single place they are maintained (the README carries a summary
table pointing there). They are
deliberately not repeated here; having them in two files is how the two
copies ended up disagreeing.

For the record, what this model actually prints when the C firmware boots
on an ESP32-S3-DevKitC N16R8 — captured from a real run on that board,
not quoted from upstream:

```
=== ESP32-S3 PLE TinyLM ===
model: V=32768 D=96 L=6 H=4 F=66 P=128  (mapped 15.6 MB)
head staged int8: 2.54 MB
PSRAM free after alloc: 3141 KB
```

An earlier version of this section instead quoted
`reference-c/firmware/esp32_llm/README.md`'s recorded diagnostics
(`2.53 MB` staged, `~5100 KB` free) alongside its 102.9 ms/step figure.
Those were upstream's own measurements, and the PSRAM figure in particular
does not reproduce here — this build has roughly 2 MB less free after
allocation. Treat upstream's recorded numbers as historical rather than as
something to check a fresh build against.
