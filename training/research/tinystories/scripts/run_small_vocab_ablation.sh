#!/usr/bin/env bash
# Main ablation at vocab 4096: all five arms, both seeds, core-matched at 1.5M.
# Produces the small-vocabulary table in RESULTS.md, where PLE's edge is +0.025
# nats - the control showing the gain is vocabulary-dependent.
#
# Fails immediately if any run fails: a partial ablation is not a result.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

log() { echo "[$(date '+%m-%d %H:%M')] $*"; }

for seed in 0 1; do
  for arm in baseline ple ple_notable fatembed bigcore; do
    log "RUN $arm seed$seed"
    uv run python -m research.tinystories.train \
      --arm "$arm" --vocab 4096 --target-core 1500000 --steps 3000 \
      --seed "$seed" --tag clean
  done
done

log "small-vocab ablation complete"
