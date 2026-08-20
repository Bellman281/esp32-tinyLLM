#!/usr/bin/env bash
# Flash the firmware, monitor it, and tee everything to runs/.
#
# WHY: the board's output is the only place several of this project's figures
# exist, and it was being moved around by hand -- which loses the boot lines,
# and loses them selectively, because the interesting ones (PSRAM bandwidth,
# the attn split line, the token digest) are far apart in a 150-line log.
# Writing it to a file means a run can be re-read later, diffed against the
# previous one, and quoted exactly.
#
#   scripts/run_device.sh             # flash + monitor, log to runs/<stamp>.log
#   scripts/run_device.sh --no-flash  # monitor only (Ctrl+R on the board to reset)
#   scripts/run_device.sh --fast-mspi # + sdkconfig.defaults.fast-mspi (120 MHz)
#   scripts/run_device.sh --probe     # + the dual-core bandwidth probe
#
# Ctrl+C when the run finishes; the trap trims the log to runs/latest.txt,
# starting at the firmware banner so the ESP-IDF build noise is dropped.
#
# runs/ is gitignored: these are measurements of one binary on one board at one
# moment, and the ones worth keeping belong in benchmarks/device.toml with a
# commit next to them, not as loose logs.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs
STAMP=$(date +%Y%m%d-%H%M%S)
FULL="runs/$STAMP.log"
LATEST="runs/latest.txt"

trim() {
  # Everything from the firmware's own banner onward. If the board never got
  # that far -- a hang, a brownout, a boot loop -- fall back to the whole log,
  # because then the ESP-IDF output IS the finding.
  if grep -q '=== ESP32-S3 PLE TinyLM ===' "$FULL" 2>/dev/null; then
    awk '/=== ESP32-S3 PLE TinyLM ===/,0' "$FULL" > "$LATEST"
  else
    cp "$FULL" "$LATEST"
    echo "(no firmware banner in this run -- kept the full log)" >> "$LATEST"
  fi
  printf '\n--- wrote %s and %s ---\n' "$FULL" "$LATEST"
}
trap 'trim; exit 0' INT TERM

FEATURES=""
EXTRA_SDKCONFIG=""
for arg in "$@"; do
  case "$arg" in
    --probe)     FEATURES="--features bandwidth-probe" ;;
    # esp-idf-sys reads ESP_IDF_SDKCONFIG_DEFAULTS as a ';'-separated list and
    # prepends its own generated file. Naming the base explicitly keeps it in
    # play; the overlay comes second so its keys win.
    --fast-mspi) EXTRA_SDKCONFIG="sdkconfig.defaults;sdkconfig.defaults.fast-mspi" ;;
    --no-flash)  ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done
[ -n "$FEATURES" ] && echo "features: $FEATURES"
[ -n "$EXTRA_SDKCONFIG" ] && echo "sdkconfig: $EXTRA_SDKCONFIG"

(
  cd engine/llm-firmware
  [ -n "$EXTRA_SDKCONFIG" ] && export ESP_IDF_SDKCONFIG_DEFAULTS="$EXTRA_SDKCONFIG"
  # shellcheck disable=SC2086
  cargo run --release $FEATURES
) 2>&1 | tee "$FULL"
trim
