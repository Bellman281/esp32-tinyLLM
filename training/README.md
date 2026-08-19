# training/ — the Python pipeline that produces `model.bin`

Vendored, unmodified, from **[slvDev/esp32-ai](https://github.com/slvDev/esp32-ai)**
at commit `97018d5` (2026-08-07), MIT licensed — see [`LICENSE.upstream`](./LICENSE.upstream).

This repo ports the *inference* engine to Rust. It does not train models. Until
now the training, tokenization, quantization and export code lived only upstream,
which meant "how was `model.bin` made?" could not be answered without leaving the
repo. This folder answers it. The step-by-step walkthrough is
[`../TRAINING.md`](../TRAINING.md); this file is the provenance record.

## Ground rule: this folder is refreshed, not patched

Same rule `reference-c/` lives under (CONTRIBUTING.md ground rule 1). Every file
here is byte-identical to upstream. If something needs to change, it changes
upstream and this folder is re-copied wholesale. Local edits are recorded in the
table below, and the table is currently empty **by design**.

### Local changes vs. upstream

| File | Change | Why |
|---|---|---|
| _(none)_ | | |

## What was copied, and what was left behind

| Path here | Upstream path |
|---|---|
| `src/model.py` | `src/model.py` |
| `src/quantize.py` | `src/quantize.py` |
| `research/tinystories/*.py`, `README.md`, `scripts/*.sh` | same |
| `tools/generate_vocab.py` | `firmware/esp32_tinystories/tools/generate_vocab.py` |
| `pyproject.toml` | `pyproject.toml` |
| `RESULTS.upstream.md` | `RESULTS.md` |

Left behind on purpose: the Arduino/C firmware sketches (this repo's target is
Rust), the Barista model's tooling (a different model), `scripts/deploy.sh` and
`scripts/benchmark_device.py` (device flashing/serial, superseded here by
`engine/llm-firmware` and `BENCHMARKING.md`), and the upstream test suite.

## Path note

The vendored scripts compute their repository root as `Path(__file__).parents[2]`,
which resolves to **this folder**, not the repo root. So `data/`, `runs/` and
`artifacts/` land under `training/`. Run every `-m research.tinystories.*`
command from inside `training/`.

`tools/generate_vocab.py` is the exception — it derives its defaults from the
upstream *sketch* layout, which does not exist here, so always pass
`--tokenizer` and `--out` explicitly. See `../TRAINING.md`.

## Two things verified when this was vendored

1. **`tools/generate_vocab.py` reproduces the shipped `vocab.h` byte-for-byte**
   from `demo/bpe32768.json`, and emits `PROMPT_IDS = {433, 447, 259, 405}` —
   the first four ids of `models.toml`'s `prompt_ids`. The tokenizer in `demo/`
   really is the one that produced the shipped decode table.

2. **The current upstream `export.py` writes a header this repo cannot read.**
   See `../TRAINING.md`'s "The export format gap" — this is a real, open blocker
   on regenerating `model.bin`, not a doc nit.
