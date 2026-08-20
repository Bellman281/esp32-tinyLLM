#!/usr/bin/env bash
# Compile the C reference's hot functions for Xtensa and disassemble them, so
# the Rust port can be compared against C instruction-for-instruction instead
# of only stage-by-stage.
#
# WHY THIS EXISTS. This repo verifies VALUES exactly -- golden.rs against
# PyTorch, cli_parity.rs against the compiled C binary, ppl_parity.rs on exact
# cross-entropy, and a token digest the board prints. It measures COST at
# five-stage resolution, plus two sub-splits. Between those two layers there
# was nothing: no comparison of `matvec_range` against `matvec_q`, or of
# `rmsnorm` against `rmsnorm`. Every optimization was aimed at a stage and
# judged by a stage.
#
# The first time anyone read the instructions instead (scripts/disasm_head.sh)
# it found a `sext` on every one of 2.43M elements per token, because Xtensa
# has no signed byte load. Stage timing cannot name that. This closes the gap
# from the other side: the same loops, as GCC compiles them.
#
#   scripts/disasm_c.sh        # writes runs/disasm-c.txt
#
# CAVEAT, and it is not small: the shipped C firmware is built by
# arduino-esp32's GCC, not by the ESP-IDF GCC this finds. Same backend family
# and the same -O3, but not the same build. Read the OUTPUT SHAPE -- does the
# inner loop use MAC16, how many instructions per multiply-accumulate, are
# there sign-extends -- and do not read small instruction-count differences as
# meaningful.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs

LLM_H=$(find reference-c -name 'llm.h' | head -1)
[ -z "$LLM_H" ] && { echo "no reference-c/**/llm.h found" >&2; exit 1; }
CC=$(find "$HOME/.rustup/toolchains/esp" \
          engine/llm-firmware/.embuild/espressif/tools/xtensa-esp-elf \
          -name 'xtensa-esp32s3-elf-gcc' -type f 2>/dev/null | head -1)
[ -z "$CC" ] && CC=$(command -v xtensa-esp32s3-elf-gcc)
[ -z "$CC" ] && { echo "no xtensa gcc found" >&2; exit 1; }
OBJDUMP=${CC%gcc}objdump

TU=$(mktemp /tmp/cref_XXXX.c)
OBJ=${TU%.c}.o
# Wrappers, not calls into a main: `static inline` bodies get inlined into
# whatever calls them, so each wrapper becomes a named symbol containing the
# loop we want to read. `noinline` on the wrapper keeps them separate.
cat > "$TU" <<EOF
#include <stdint.h>
#include <math.h>
#include <string.h>
#include "$(cd "$(dirname "$LLM_H")" && pwd)/$(basename "$LLM_H")"
#define W __attribute__((noinline,used))
W void c_matvec_q(const QT*t,const float*x,float*y){ matvec_q(t,x,y); }
W void c_matvec_q8(const QT*t,const float*x,float*y){ matvec_q8(t,x,y); }
W void c_deq_row(const QT*t,int r,float*o){ deq_row(t,r,o); }
W void c_rmsnorm(const float*x,const float*w,int n,float*o){ rmsnorm(x,w,n,o); }
W void c_quantize_act(const float*x,int n,int8_t*q,float*s){ quantize_act(x,n,q,s); }
EOF

OUT=runs/disasm-c.txt
{
  echo "=== cc: $CC"
  echo "=== llm.h: $LLM_H"
  echo
  if "$CC" -O3 -mlongcalls -std=gnu17 -w -c "$TU" -o "$OBJ" 2>&1; then
    echo "=== instruction mix in the C reference's hot functions ==="
    for i in mula.dd mula.da mula.ad mula.aa mul16s mul16u mull sext l8ui l16si; do
      printf "  %-9s %s\n" "$i" "$("$OBJDUMP" -d "$OBJ" | grep -c "\b$i" || true)"
    done
    echo
    echo "=== per-wrapper size ==="
    "${CC%gcc}nm" --print-size --size-sort --radix=d "$OBJ" | grep -iE ' [tT] c_' || true
    echo
    "$OBJDUMP" -d "$OBJ"
  else
    echo "!! compile failed -- GCC is the authority on what llm.h needs."
    echo "!! Fix the translation unit above rather than guessing twice."
  fi
} > "$OUT" 2>&1
rm -f "$TU" "$OBJ"
echo "wrote $OUT ($(wc -l < "$OUT") lines)"
grep -E '^  (mull|mula|sext|l8ui|l16si)' "$OUT" || sed -n '1,12p' "$OUT"
