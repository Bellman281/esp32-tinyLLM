#!/usr/bin/env bash
# Deploy headline at vocab 32768: the configuration that ships. Three arms, both
# seeds, core-matched at ~559k so every arm fits the SRAM budget the original
# design targeted, with the 25M table in flash.
#
# Produces the +0.098 nat result in RESULTS.md and the checkpoint the exporter
# turns into a deployable artifact.
#
# bs16/sl256 keeps 32k-class cross-entropy off the MPS memory cliff; at
# bs32/sl512 it stalls.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

log() { echo "[$(date '+%m-%d %H:%M')] $*"; }

for seed in 0 1; do
  for arm in baseline ple fatembed; do
    log "RUN $arm seed$seed"
    uv run python -m research.tinystories.train \
      --arm "$arm" --vocab 32768 --d-model 96 --n-layers 6 --ple-dim 128 \
      --target-core 560000 --batch-size 16 --seq-len 256 --steps 5000 \
      --seed "$seed" --tag cleandeploy
  done
done

log "deploy ablation complete"
