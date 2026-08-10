#!/bin/sh
# Interactive CLI: type a prompt, watch the ESP32 model respond. No pip needed.
# gen_prompt.c/llm.h/model.bin live in ../reference-c/esp32-llm-lab/ (the frozen
# C reference) -- this folder only holds the demo/tokenizer convenience layer,
# not C code, so it isn't part of reference-c/ itself. See demo/README.md.
set -e
cd "$(dirname "$0")"
LAB=../reference-c/esp32-llm-lab
cc -O3 -I "$LAB" -o gen_prompt "$LAB/gen_prompt.c" -lm   # always rebuild (picks up sampling support)
exec python3 chat.py "${1:-100}"
