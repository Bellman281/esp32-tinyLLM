#!/bin/sh
# Read one field out of the model registry (model/models.toml).
#
# This is the C side's reader. The Rust side parses the same file via
# engine/llm-host/src/manifest.rs; this script exists so the C tools --
# which take plain argv and have no business linking a TOML parser -- get
# their paths, window counts, and prompt ids from that same single source
# instead of the compiled-in defaults they ship with.
#
#   usage: scripts/model.sh <field> [model]
#          scripts/model.sh list
#
# With no [model], the registry's `default` is used.
#
# Path-valued fields (bin, vocab, val_tokens, golden) are printed as
# absolute paths, resolved against the repo root, so the output can be
# handed straight to a tool regardless of the directory you ran this from.
# Every other field is printed verbatim; `prompt_ids` comes out as
# `433,447,...`, which is exactly the form gen_prompt expects.
#
# Examples:
#   MODEL=$(scripts/model.sh bin)
#   cc -O3 -o /tmp/ppl reference-c/firmware/host_verify/ppl.c -lm
#   /tmp/ppl "$MODEL" "$(scripts/model.sh val_tokens)" "$(scripts/model.sh ppl_windows)"
#
#   cc -O3 -o /tmp/gen_prompt reference-c/esp32-llm-lab/gen_prompt.c -lm
#   /tmp/gen_prompt "$MODEL" "$(scripts/model.sh n_generate)" \
#                   "$(scripts/model.sh prompt_ids)" 0
#
# Parsing note: this reads the same restricted subset manifest.rs does --
# `key = value` inside `[section]`, values being quoted strings, numbers, or
# single-line integer arrays, `#` starting a comment. If you extend the
# format beyond that subset, both readers need updating; see the header of
# model/models.toml.
set -eu

ROOT=$(cd "$(dirname "$0")/.." && pwd)
MANIFEST="$ROOT/model/models.toml"

[ -f "$MANIFEST" ] || { echo "model.sh: no manifest at $MANIFEST" >&2; exit 1; }

field=${1:-}
[ -n "$field" ] || { echo "usage: $0 <field> [model]   |   $0 list" >&2; exit 2; }

if [ "$field" = "list" ]; then
    awk '/^[[:space:]]*\[/ { gsub(/[][[:space:]]/, ""); print }' "$MANIFEST"
    exit 0
fi

model=${2:-}
if [ -z "$model" ]; then
    model=$(awk -F= '
        /^[[:space:]]*\[/ { exit }                      # stop at the first section
        /^[[:space:]]*default[[:space:]]*=/ {
            sub(/#.*/, "", $2); gsub(/["[:space:]]/, "", $2); print $2; exit
        }' "$MANIFEST")
    [ -n "$model" ] || { echo "model.sh: no 'default' in $MANIFEST" >&2; exit 1; }
fi

value=$(awk -v sec="$model" -v key="$field" '
    /^[[:space:]]*\[/ {
        line = $0; gsub(/[][[:space:]]/, "", line)
        insec = (line == sec)
        next
    }
    !insec { next }
    {
        line = $0
        sub(/#.*/, "", line)
        if (line !~ "^[[:space:]]*" key "[[:space:]]*=") next
        sub(/^[^=]*=[[:space:]]*/, "", line)
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", line)      # trim
        gsub(/^"|"$/, "", line)                            # unquote
        if (line ~ /^\[.*\]$/) {                           # array -> a,b,c
            gsub(/^\[|\]$/, "", line)
            gsub(/[[:space:]]/, "", line)
        }
        print line
        found = 1
        exit
    }
    END { if (!found) exit 3 }
' "$MANIFEST") || {
    echo "model.sh: no field '$field' for model '$model' in $MANIFEST" >&2
    echo "          known models: $(awk '/^[[:space:]]*\[/ { gsub(/[][[:space:]]/, ""); printf "%s ", $0 }' "$MANIFEST")" >&2
    exit 3
}

case "$field" in
    bin|vocab|val_tokens|golden) printf '%s\n' "$ROOT/$value" ;;
    *)                           printf '%s\n' "$value" ;;
esac
