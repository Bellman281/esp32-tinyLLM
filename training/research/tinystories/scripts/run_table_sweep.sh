#!/usr/bin/env bash
# Does the gain scale with table size? At vocab 4096, the same control the
# small-vocabulary ablation uses.
#
# FFN is fixed at 256 so the core does not shrink as ple_dim grows. Without that
# the table cannibalises compute and the sweep measures the wrong thing.
#
# ple and ple_notable are run at each ple_dim with identical core, ffn and
# plumbing, differing ONLY by the presence of the lookup table, so the table's
# isolated effect is ple - ple_notable at each point.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

log() { echo "[$(date '+%m-%d %H:%M')] $*"; }

for pd in 64 128 256 512; do
  for arm in ple ple_notable; do
    log "RUN $arm ple_dim=$pd"
    uv run python -m research.tinystories.train \
      --arm "$arm" --vocab 4096 --fixed-ffn 256 --ple-dim "$pd" --steps 3000 \
      --seed 0 --tag "fix-d$pd"
  done
done

log "table sweep complete"
