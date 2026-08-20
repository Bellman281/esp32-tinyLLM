#!/usr/bin/env bash
# Disassemble the firmware's hot loops, so an optimization guess can be checked
# against the instructions instead of against a stopwatch.
#
# WHY: the head's arithmetic is ~48.8 ms/token against the C reference's ~30.4
# for the same algorithm over the same data -- C moves TWICE the bytes and is
# still faster overall. A gap that size is not scheduling noise; something
# concrete differs in what the two compilers emit. The LX7 has a MAC16
# multiply-accumulate unit that GCC's Xtensa backend is known to use and LLVM's
# may not, which would explain it, but that is a guess until someone reads the
# output.
#
#   scripts/disasm_head.sh          # dump to runs/disasm.txt
#
# Writes the symbol table sorted by size (so the hot code is easy to find) and
# the full disassembly. Nothing here touches the board.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs
ELF=$(find engine/llm-firmware/target/xtensa-esp32s3-espidf/release \
        -maxdepth 1 -name 'llm-firmware' -type f 2>/dev/null | head -1)
[ -z "$ELF" ] && { echo "no ELF -- run a build first" >&2; exit 1; }

OBJDUMP=$(find "$HOME/.rustup/toolchains/esp" \
               "engine/llm-firmware/.embuild/espressif/tools/xtensa-esp-elf" \
               -name 'xtensa-esp32s3-elf-objdump' -type f 2>/dev/null | head -1)
[ -z "$OBJDUMP" ] && OBJDUMP=$(command -v xtensa-esp32s3-elf-objdump)
[ -z "$OBJDUMP" ] && { echo "no xtensa objdump found" >&2; exit 1; }

NM=${OBJDUMP%objdump}nm
OUT=runs/disasm.txt
{
  echo "=== elf: $ELF"
  echo "=== objdump: $OBJDUMP"
  echo
  echo "=== largest .text symbols (hot code lives here) ==="
  "$NM" --print-size --size-sort --radix=d -C "$ELF" 2>/dev/null \
    | grep -iE ' [tT] ' | tail -40
  echo
  echo "=== symbols matching the head/dot/matvec path ==="
  "$NM" --print-size --size-sort --radix=d -C "$ELF" 2>/dev/null \
    | grep -iE 'dot_int|rows_range|matvec|llm_core' | tail -40
  echo
  echo "=== full disassembly follows ==="
  "$OBJDUMP" -d -C "$ELF"
} > "$OUT" 2>&1
echo "wrote $OUT ($(wc -l < "$OUT") lines)"
echo "MAC16 instructions present? $("$OBJDUMP" -d "$ELF" | grep -cE '\bmula\.|\bmul16|\bmull\b' || true)"
