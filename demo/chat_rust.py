#!/usr/bin/env python3
"""Interactive CLI for the ESP32-S3 TinyLM, driven by the RUST engine.

The Rust counterpart to chat.py: same prompt loop, same tokenizer, but it
subprocesses `engine/target/release/gen-prompt` (llm-core) instead of the
compiled C `gen_prompt` (llm.h). Run both at `:temp 0` on the same prompt
and they emit identical text -- that is `cli_parity.rs`'s assertion, by
hand.

Two deliberate differences from chat.py:

  * ARGV. The engines do not take the same arguments:
        C:    gen_prompt  <model.bin>            <N> <ids> [temp]
        Rust: gen-prompt  <model.bin> <vocab.h>  <N> <ids> [temp]
    C `#include`s vocab.h at compile time (run_cli.sh's `-I` flag); Rust
    loads it at runtime. See engine/llm-host/src/vocab.rs.

  * MODEL PATHS. chat.py hardcodes them; this reads them from the registry
    via scripts/model.sh, per CONTRIBUTING.md ground rule 4, so it cannot
    drift from what the Rust tests and C tools use. Set TINYLLM_MODEL to
    name a non-default registry entry (`scripts/model.sh list`).

The tokenizer is imported from chat.py rather than copied -- there is one
BPE implementation in this project and this is not a second one.

Commands:
  :temp X   sampling temperature (0 = greedy, exactly the chip; try 0.8)
  :n N      tokens to generate (default 100)
It writes simple stories; it cannot answer questions.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

sys.path.insert(0, HERE)
from chat import encode  # noqa: E402  -- the tokenizer, unmodified

try:
    import readline  # arrow-key history + inline line editing
except Exception:
    pass

GEN = os.path.join(REPO, "engine", "target", "release", "gen-prompt")
BUILD_HINT = "build it first:  cd engine/llm-host && cargo build --release"


def registry(field):
    """One field out of model/models.toml, via the same reader the C tools use."""
    argv = [os.path.join(REPO, "scripts", "model.sh"), field]
    model = os.environ.get("TINYLLM_MODEL")
    if model:
        argv.append(model)
    try:
        out = subprocess.run(argv, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(f"model.sh {field}: {e.stderr.strip()}")
    return out.stdout.strip()


def repl(n_tok=100, temp=0.8):
    model, vocab = registry("bin"), registry("vocab")
    if not os.path.exists(GEN):
        sys.exit(f"missing {GEN}\n{BUILD_HINT}")

    print("=== ESP32-S3 TinyLM CLI -- Rust engine (llm-core) ===")
    print("Type a prompt. Commands:  :temp 0.8  (randomness, 0=greedy)   :n 150  (length)   Ctrl-C quits")
    print(f"[temp={temp}  (varied). Use :temp 0 for exact greedy chip behavior.  n={n_tok}]\n")

    while True:
        try:
            p = input("prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not p:
            continue
        if p.startswith(":temp"):
            temp = float(p.split()[1])
            print(f"[temp={temp}]\n")
            continue
        if p.startswith(":n"):
            n_tok = int(p.split()[1])
            print(f"[n={n_tok}]\n")
            continue
        ids = encode(p)
        if not ids:
            print("(couldn't tokenize)\n")
            continue
        subprocess.run([GEN, model, vocab, str(n_tok), ",".join(map(str, ids)), str(temp)])
        print()


if __name__ == "__main__":
    repl(int(sys.argv[1]) if len(sys.argv) > 1 else 100, 0.8)
