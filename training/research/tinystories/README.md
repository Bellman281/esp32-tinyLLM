# TinyStories research

This directory contains the TinyStories training and evaluation pipeline behind
[`RESULTS.md`](../../RESULTS.md): data preparation, ablation runs, quantization
evaluation, sampling, and export of a deployable PLE artifact.

It is not part of the ESP32 runtime or firmware build. Use it to reproduce the
TinyStories methodology or to generate local artifacts from a checkpoint.

Only models whose research is public get a directory here.

## Requirements

```bash
uv sync
```

Installs the shared PLE modules (`model`, `quantize`) so `from model import ...`
resolves anywhere.

**Run every command below from the repository root.** This directory is
reproduction code and is deliberately not packaged, so `-m
research.tinystories...` resolves only when the root is the working
directory. The scripts in `scripts/` derive the root themselves and can be run
from anywhere.

## Dataset

The first 300 MB of
[roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories),
licensed **CDLA-Sharing-1.0**. It is downloaded, not redistributed here.

```bash
uv run python -m research.tinystories.prepare --vocab 32768
```

Downloads the slice into `data/tinystories/raw/`, then writes the tokenizer and
`uint16` token bins into `data/tinystories/vocab-32768/`. Each `--vocab` value
gets its own directory, keeping the tokenizer beside the bins it produced.
The 4,096-vocab control used in `RESULTS.md` is `--vocab 4096`.

**Reproducibility is approximate.** The script fetches the dataset's `main`
branch without pinning a revision and records no hash of the raw slice, and
training sets no determinism flags. A re-run reproduces the methodology and
should land on comparable numbers; it will not reproduce these bytes.

## Ablation arms

Five arms, all built to the same dense-core parameter budget so the comparison
is fair:

| arm | what it tests |
|---|---|
| `baseline` | no table at all, what a tiny LM looks like today |
| `ple` | per-layer adapters + flash lookup table |
| `ple_notable` | the adapters with **no** table, isolating the plumbing |
| `fatembed` | the same table budget as `ple`, spent on a wide input embedding |
| `bigcore` | the table budget spent on a wider core instead |

```bash
uv run python -m research.tinystories.train --arm ple --vocab 4096 \
  --steps 3000 --target-core 1500000 --seed 0
```

Checkpoints and per-run JSON records go to `runs/` at the repository root.

Three scripts run the published configurations. Each derives the repository
root itself, so they work from any directory, and each fails immediately if a
run fails - a partial ablation is not a result.

| script | what it produces |
|---|---|
| `run_small_vocab_ablation.sh` | vocab 4096, five arms, seeds 0 and 1, core-matched 1.5M |
| `run_deploy_ablation.sh` | vocab 32768, three arms, seeds 0 and 1, core-matched ~559k - the headline, and the checkpoint the exporter consumes |
| `run_table_sweep.sh` | `ple` vs `ple_notable` at ple_dim 64/128/256/512, FFN fixed at 256 |

```bash
bash research/tinystories/scripts/run_deploy_ablation.sh
```

## Reading the results

```bash
uv run python -m research.tinystories.analyze --tag cleandeploy \
  --expect-arms baseline,ple,fatembed --expect-seeds 2

uv run python -m research.tinystories.analyze --tag clean \
  --expect-arms baseline,ple,ple_notable,fatembed,bigcore --expect-seeds 2
```

`--tag`, `--expect-arms` and `--expect-seeds` are all required. `runs/`
accumulates several cohorts plus records from other models, and averaging across
them reports a number that describes no cohort: the same archive that yields the
published +0.098 nats within the deploy cohort yields +0.074 when mixed.

The analyzer refuses a cohort whose runs disagree on vocabulary, sequence
length, batch size, steps or learning rate. Model shape is deliberately not
checked, because arms vary width to hit the same core budget and `bigcore` is
defined by being wider.

`--expect-arms` and `--expect-seeds` make completeness explicit and are
mandatory, so a comparison cannot quietly shrink. A missing arm, an unexpected
extra arm, duplicate records at one seed, or arms run on different seed sets
are all errors.

Fields that no record in the cohort carries are listed as *not verifiable*
rather than skipped silently. Runs predating the `training` section and the
tokenizer hash cannot be checked on those fields, and the output says so.

### The table sweep is four cohorts

One per `ple_dim`, and it must be read as four comparisons. Averaging them
erases the curve the sweep exists to measure:

```bash
for pd in 64 128 256 512; do
  uv run python -m research.tinystories.analyze --tag "fix-d$pd" \
    --expect-arms ple,ple_notable --expect-seeds 1
done
```

| ple_dim | ple vs ple_notable |
|---|---|
| 64 | +0.045 nats |
| 128 | +0.071 |
| 256 | +0.094 |
| 512 | +0.087 |

## Expected headline numbers

At vocab 32,768, core-matched at ~559k, two seeds:

| | |
|---|---|
| PLE vs baseline | **+0.098 nats**, +/-0.006 |
| perplexity | 12.58 -> 11.41 |
| survives 4-bit PTQ | yes, two seeds |
| vocab-4096 control | +0.025 nats, so the gain is vocabulary-dependent |

Full tables, seeds and caveats are in [`RESULTS.md`](../../RESULTS.md).

## Quantization evaluation

`RESULTS.md` publishes two quantization tables, and both need both seeds:

```bash
# group 64, fp32 scales - the first published PTQ table
for s in 0 1; do
  uv run python -m research.tinystories.quantize_eval --tag cleandeploy --seed $s
done

# group 128, fp16 scales - what the exporter writes and the runtime reads
for s in 0 1; do
  uv run python -m research.tinystories.quantize_eval --tag cleandeploy --seed $s \
    --group 128 --fp16-scales
done
```

Measures fp32 and post-training-quantized validation loss per arm, and fails if
any arm's checkpoint is missing rather than reporting a partial comparison. One
invocation covers one seed.

The reusable tensor math lives in `src/quantize.py`; this module holds the
checkpoint loading, validation data and arm loop.

## Sampling

```bash
uv run python -m research.tinystories.sample \
  --run runs/ple-cleandeploy-s0.pt \
  --tokenizer data/tinystories/vocab-32768/tokenizer.json
```

`--tokenizer` is required, and is verified against the hash the run recorded.
Legacy checkpoints predate that hash and only warn. Decoding a 32,768-vocab run
with the 4,096 tokenizer yields fluent text from the wrong token ids rather than
an error, so the mismatch has to be rejected up front. `--seed` fixes the
sampling stream.

## Export

The published checkpoint predates tokenizer hashing, so reproducing that exact
artifact requires the exception to be explicit:

```bash
uv run python -m research.tinystories.export ple-cleandeploy-s0 \
  --tokenizer data/tinystories/vocab-32768/tokenizer.json \
  --allow-unverified-tokenizer
```

Runs trained since then record the hash and need no flag.

Reads `runs/<tag>.pt` and writes `artifacts/tinystories/`: `model.bin` in the
PLE format, `tokenizer.json`, and the `golden.txt` / `golden.npz` references the
host verifier checks the C runtime against.

`--tokenizer` selects the one the checkpoint was trained with; its size becomes
`output_vocab`. Two guards:

- the export fails if `output_vocab` exceeds the checkpoint's embedding rows,
  because the tied head is the first `output_vocab` rows and the runtime would
  otherwise read past the tensor;
- runs record the SHA-256 of their tokenizer, and both the exporter and the
  sampler refuse one that does not match. A size check alone is not enough: a
  *smaller* wrong tokenizer passes it and silently produces a model with the
  wrong output vocabulary. Checkpoints predating that hash cannot be checked,
  and the exporter treats that as an error unless
  `--allow-unverified-tokenizer` is passed, because it produces a distributable
  binary. Sampling only warns.

`scripts/deploy.sh` does **not** invoke this. It consumes whatever is already in
`artifacts/tinystories/`, which this exporter is one way to fill.

## What is not here

Generated data, checkpoints and exported artifacts are all ignored: `data/`,
`runs/` and `artifacts/`. Nothing here is required to build or flash the
firmware.
