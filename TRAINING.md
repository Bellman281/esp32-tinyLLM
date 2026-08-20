# Training a TinyLM and preparing it for this repo

How the `model.bin` this project runs was actually made: dataset → tokenizer →
training → quantization → export → `vocab.h`. The Python that does it is
vendored under [`training/`](./training/) (provenance and the refresh rule are
in [`training/README.md`](./training/README.md)); this file is the walkthrough.

Nothing in the Rust engine depends on any of this. `model.bin` and `vocab.h` are
consumed as static inputs, and the registry
([`model/models.toml`](./model/models.toml)) is what points at them. This
document exists so that "where did these files come from, and how do I make
another one?" has an answer inside the repo.

**The shipped model was trained for this repo**, not downloaded: 2 August 2026,
on a rented VPS GPU, by this repo's author, running the pipeline below. It is a
different artifact from upstream's published release — see
[`model/README.md`](./model/README.md) for the two checksums and why they differ
by exactly 16 bytes.

> **Note on the exporter.** It writes a newer header format than the shipped
> `model.bin` uses. `llm-core` reads both, so the pipeline below round-trips —
> but the frozen C reference still only reads the old one, so a freshly
> exported model runs under Rust and not under the C tools. See
> [The two header formats](#the-two-header-formats).

## What the shipped model actually is

Read straight out of `model/tinystories-v32768/model.bin`'s header and
`vocab.h`, not quoted from anywhere:

| | |
|---|---|
| Architecture | Decoder-only transformer, Gemma-style **Per-Layer Embeddings (PLE)** |
| Params | 28.9M total; ~25M of them (`V·P·L` = 32768·128·6) in the flash-resident PLE table |
| Config | `V=32768 D=96 L=6 H=4 F=66 P=128 seq_len=512 group=128` |
| **Context window** | **512 tokens.** The "32K" in this project is the *vocabulary*, not the context. |
| Embedding rows | 32,768 (padded) |
| **Tokenizer entries** | **25,353** — `vocab.h`'s `VOCAB_N`. The embedding is padded above the tokenizer; rows 25353–32767 are never scored. |
| Weights | int4, symmetric group-wise (group=128), fp16 group scales |
| Norms | fp32 |
| Head | tied to the input embedding |
| Corpus | [TinyStories](https://arxiv.org/abs/2305.07759) (Eldan & Li, 2023), first 300 MiB of the HF train split |
| Trained | 2 August 2026, rented VPS GPU, by this repo's author |
| File | `model/tinystories-v32768/model.bin`, 14.91 MB, sha256 `3e5870b4…7fa7bbe` |

## Setup

```bash
cd training
uv sync            # or: pip install "numpy>=2.5.1" "requests" "tokenizers" "torch" "tqdm"
```

Python ≥3.12. Torch on CPU works; MPS and CUDA are auto-detected by `train.py`.

**Run every `-m research.tinystories.*` command from inside `training/`.** The
vendored scripts resolve their root as `parents[2]`, which is `training/` — so
`data/`, `runs/` and `artifacts/` are created there.

## 1. Dataset + tokenizer

```bash
cd training
uv run python -m research.tinystories.prepare --vocab 32768
```

Downloads the first 300 MiB of
[`roneneldan/TinyStories`](https://huggingface.co/datasets/roneneldan/TinyStories)
(CDLA-Sharing-1.0, downloaded not redistributed), trains a byte-level BPE at the
requested size, and writes into `training/data/tinystories/vocab-32768/`:

- `tokenizer.json` — HuggingFace `tokenizers` format
- `train.bin`, `val.bin` — `uint16` token streams (so vocab can never exceed 65,536)

Each `--vocab` gets its own directory, because cross-entropy is only comparable
between models sharing a tokenizer.

**Reproducibility is approximate**, and upstream says so: the fetch is not
pinned to a dataset revision and training sets no determinism flags. You will
land on comparable numbers, not identical bytes.

`model/tinystories-v32768/val.bin` — the validation tokens `ppl_parity.rs`
reads — is the `val.bin` this step produces.

## 2. Train

The exact configuration that produced the shipped model:

```bash
uv run python -m research.tinystories.train \
  --arm ple --vocab 32768 --d-model 96 --n-layers 6 --ple-dim 128 \
  --target-core 560000 --batch-size 16 --seq-len 256 --steps 5000 \
  --seed 0 --tag cleandeploy
```

Note `--seq-len 256` at train time even though the exported header says
`seq_len=512`: RoPE extrapolates, and upstream's comment says bs32/sl512 stalls
on MPS. The 512 is the runtime KV-cache budget, not the training window.

`--arm` selects the embedding strategy under test. The five arms exist because
this was an ablation, and all are held to the same *dense core* parameter budget
so the comparison is fair:

| arm | what it tests |
|---|---|
| `ple` | **what ships** — per-layer adapters + flash-resident lookup table |
| `baseline` | no table at all; a conventional tiny LM |
| `ple_notable` | the adapters with no table — isolates the plumbing from the table |
| `fatembed` | the same table budget spent on one wide input embedding |
| `bigcore` | the table budget spent on a wider core instead |

The published cohort is a script:

```bash
bash research/tinystories/scripts/run_deploy_ablation.sh
```

Checkpoints and per-run JSON go to `training/runs/`. Each records the SHA-256 of
the tokenizer that trained it — the exporter and sampler both refuse a mismatch,
because a *smaller* wrong tokenizer passes a size check and silently yields a
model with the wrong output vocabulary.

Headline result at vocab 32,768, core-matched ~559k, two seeds: PLE beats
baseline by **+0.098 nats** (ppl 12.58 → 11.41), and the gain survives 4-bit
PTQ. At vocab 4,096 the same comparison gives only +0.025 — the gain is
vocabulary-dependent. Full tables in
[`training/RESULTS.upstream.md`](./training/RESULTS.upstream.md).

Sanity-check a checkpoint before spending time on export:

```bash
uv run python -m research.tinystories.sample \
  --run runs/ple-cleandeploy-s0.pt \
  --tokenizer data/tinystories/vocab-32768/tokenizer.json
```

## 3. Quantization

Two layers, and they are separate on purpose:

**The math** — `training/src/quantize.py`. Symmetric group-wise quantization
along the last dim, `qmax = 2^(bits-1) - 1`, one scale per `group` weights.
`fp16_scales=True` rounds the scale to fp16 *before* dequantizing, so the result
matches what a runtime reading fp16 scales computes. Only ≥2D parameters are
quantized; norms and biases stay fp32. This is what the shipped model uses:
**4-bit, group 128, fp16 scales**.

**The evaluation** — measure what PTQ costs before shipping it:

```bash
# what the exporter actually writes and the ESP32 reads
for s in 0 1; do
  uv run python -m research.tinystories.quantize_eval --tag cleandeploy --seed $s \
    --group 128 --fp16-scales
done
```

Reports fp32 vs post-training-quantized validation loss per arm, and fails
rather than reporting a partial comparison if a checkpoint is missing.

Note `export.py` does **not** call `src/quantize.py`. It carries its own copy of
the same arithmetic because it also needs the packed-nibble byte layout. The two
must agree or the golden reference stops matching — that is a real coupling to
respect if you touch either.

## 4. Export to `model.bin`

```bash
uv run python -m research.tinystories.export ple-cleandeploy-s0 \
  --tokenizer data/tinystories/vocab-32768/tokenizer.json
```

Writes `training/artifacts/tinystories/`:

- `model.bin` — the flat, mmap-able binary
- `tokenizer.json` — copied beside the weights so the artifact dir is self-contained
- `golden.txt` / `golden.npz` — last-position logits for a fixed 8-token prompt,
  forwarded through the **dequantized** weights. This is what isolates *port*
  correctness from *quantization* error, and it is exactly what this repo's
  `engine/llm-host/tests/golden.rs` checks against.

Tensor order is hard-coded and must match `llm.h`'s `llm_load` /
`llm-core`'s `Model::load`:

```
tok_emb, ple_model_proj, ple_proj_norm, ple_table,
  per layer × L: attn_norm, attn.qkv, attn.proj, ffn_norm,
                 ffn.gate, ffn.up, ffn.down, ple_gate, ple_proj, ple_norm,
out_norm
```

Quantized tensors are written as `int32 group`, then int4 codes packed two per
byte, then fp16 scales. Norms are raw fp32.

### The two header formats

The exporter and this repo's readers disagree on the *preamble*. The tensor
stream after it is byte-for-byte the same in both.

| | magic | layout |
|---|---|---|
| Shipped `model.bin`, and the frozen C reference (`llm.h` `LLM_MAGIC`) | `0x504C4531` = `"PLE1"` | `magic, V, D, L, H, F, P, seq_len, group, rope_theta` — 8 ints, no version field |
| `export.py` today | `0x00454C50` = `"PLE\0"` | `magic, version, header_bytes, flags, input_vocab, output_vocab,` then config |

The newer format is strictly better: `header_bytes` lets a later version append
fields without breaking existing readers, `flags` bit 0 records whether the head
is tied to the embedding, and `output_vocab` is separate from `input_vocab` —
which is what lets a runtime score only the 25,353 real tokens instead of all
32,768 padded embedding rows.

**`llm-core` reads both.** `Model::load` accepts either magic, honours
`output_vocab` where it is declared, and refuses an untied head, an unknown
format version, or an `output_vocab` larger than the embedding, rather than
misparsing them. `engine/llm-host/tests/header_v2.rs` is the gate: it rewrites
the real `model.bin`'s header into the newer form and asserts the logits come
out bit-identical, then that a narrowed `output_vocab` matches the prefix of the
full run. A `PLE1` file keeps scoring every embedding row, because that format
cannot express the distinction and the C reference scores all of them —
narrowing there would silently change greedy decode.

**Two caveats, both real:**

1. **The C reference still only reads `PLE1`.** `reference-c/` is a frozen
   mirror refreshed wholesale from upstream, not patched here (CONTRIBUTING
   ground rule 1), so a `PLE\0` model runs under Rust but not under the C
   tools — which means `cli_parity.rs` and the `ppl` C binary cannot check it.
   Fixing that belongs upstream.
2. **No real exporter output has been round-tripped yet.** The test fixtures
   are the shipped weights with a rewritten header, which exercises the parse
   but not `export.py` end to end. That needs a trained checkpoint; when
   someone produces one, run it through and confirm.

Still open, and unchanged by this: this repo has **no PyTorch golden-logit
reference for the 28.9M model it actually flashes** — `golden.rs` checks the
smaller V=4096 fixture instead. A completed export run produces that missing
golden as a side effect.

## 5. Generate `vocab.h`

```bash
cd training
uv run python tools/generate_vocab.py \
  --tokenizer ../demo/bpe32768.json \
  --out ../model/tinystories-v32768/vocab.h
```

Pass both flags — the vendored script's defaults point at the upstream sketch
layout, which does not exist here.

It decodes every single-id sequence to raw UTF-8 bytes and emits a flat blob plus
offsets, so the device just `Serial.write()`s a token's bytes (llama2.c style).
It also prints the token ids for `"Once upon a time"`.

**Verified when this was vendored:** run against `demo/bpe32768.json`, it
reproduces the shipped `reference-c/esp32-llm-lab/vocab.h` **byte-for-byte**
(modulo a two-line generator banner) — `VOCAB_N 25353`, blob `164388` bytes — and
prints `PROMPT_IDS = {433, 447, 259, 405}`, the first four ids of `models.toml`'s
`prompt_ids`. That is a closed loop: the tokenizer in `demo/` is provably the one
that produced the shipped decode table.

## 6. Register it

Per CONTRIBUTING ground rule 4, nothing hardcodes a path. Add a section to
[`model/models.toml`](./model/models.toml):

Drop the artifacts into `model/<name>/`, matching the section name:

```toml
[my-model]
description = "..."
bin         = "model/my-model/model.bin"
bin_sha256  = "..."
vocab       = "model/my-model/vocab.h"
val_tokens  = "model/my-model/val.bin"
golden      = "model/my-model/golden.txt"   # if you have one
prompt_ids  = [433, 447, 259, 405]
n_generate  = 400
ppl_windows = 4
ppl_ce_fp32 = ...
ppl_ce_int8 = ...
```

Config is deliberately *not* repeated — it is parsed from `model.bin`'s own
header by both readers. `manifest.rs`'s `real_registry_parses_and_its_files_exist`
test will fail on a typo'd path. Then:

```bash
cd engine/llm-host
cargo test --release
cargo test --release --features int8-activations
```

## Cost, roughly

The deploy ablation is 6 runs (3 arms × 2 seeds) × 5,000 steps at batch 16,
seq 256 — ~123M tokens seen per run. That is a laptop-scale job, not a frontier
one: upstream developed it on MPS. `prepare` downloads 300 MiB and writes token
bins of a few hundred MB. Exact wall-clock is not recorded upstream and is not
guessed at here — time one run before launching all six.

## Sources

- Upstream pipeline: [slvDev/esp32-ai](https://github.com/slvDev/esp32-ai) @ `97018d5`, MIT
- TinyStories: Eldan & Li (2023), [arXiv:2305.07759](https://arxiv.org/abs/2305.07759)
- PLE: [Gemma 3n developer guide](https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/)
