# demo/ — try the model interactively, no ESP32 needed

A ~29-million-parameter language model that writes little stories, running
natively on your laptop via the *exact* ESP32-S3 inference code (in plain
C) — no internet, no ML framework, no install beyond a C compiler and
python3.

This folder is deliberately **not** part of `reference-c/` — nothing here
is C/C++/Arduino code. `reference-c/` stays a strict, frozen mirror of
[slvDev/esp32-ai](https://github.com/slvDev/esp32-ai)'s C source; this
folder is the Python/shell convenience layer on top of it (the tokenizer
encoder and an interactive prompt loop), which was never ported to C or
Rust anywhere in this project by design — see `MIGRATION_PLAN.md`. The
actual inference code it runs (`gen_prompt.c`, `llm.h`, `model.bin`,
`vocab.h`) lives in `../reference-c/esp32-llm-lab/`, untouched; this folder
only adds `bpe32768.json` (the tokenizer) and the REPLs (`chat.py`,
`chat_rust.py`) on top.

## Run it (macOS or Linux)

You need only **python3** (already on Mac/Linux) and a **C compiler**:
- macOS: if it complains, run `xcode-select --install` once.
- Linux: `sudo apt install build-essential` if `cc`/`gcc` is missing.
- Windows: do it inside **WSL**, then follow the Linux steps.

```sh
cd demo
sh run_cli.sh
```

You'll get a `prompt>`. Try:

    Once upon a time
    The little robot
    One rainy day, Sara

Commands: `:temp 0.8` = randomness (`0` = deterministic/greedy, exactly the
chip's own decode), `:n 150` = length. Ctrl-C to quit.

## Run it against the Rust engine instead

`run_cli.sh` above drives the **C** engine (`llm.h`). To type at the **Rust**
engine (`llm-core`) instead, build the host tools once and use `chat_rust.py`
— same prompt loop, same tokenizer, no C compiler needed:

```sh
cd engine/llm-host && cargo build --release   # once
python3 demo/chat_rust.py                     # from the repo root
```

The two are worth running side by side. At `:temp 0` they emit **identical**
text from the same prompt — the same thing `cli_parity.rs` asserts, checkable
by hand. At the default `:temp 0.8` they diverge, and that is expected rather
than a bug: sampled mode uses a different PRNG in each engine (see
`gen_prompt.rs`'s `Xorshift32` doc comment), so only greedy mode is
token-for-token comparable.

Unlike `chat.py`, which hardcodes its model paths, `chat_rust.py` reads them
from the registry via `scripts/model.sh` (CONTRIBUTING.md ground rule 4);
`TINYLLM_MODEL=<name>` picks a non-default entry from `scripts/model.sh list`.

To time the two engines against each other, `scripts/bench_host.py` does it
at matched settings with a parity gate — see BENCHMARKING.md's "Host-side
comparison", which also spells out the equivalent commands by hand.

## Sample output

![Sample CLI output](./sample-output.jpg)

(Verified independently in this repo's own CI-less test run too — see
`model/README.md`'s "Try it yourself" section for a plain-text transcript.)

## What it can and can't do

It writes simple, coherent children's-story text with names, dialogue, and
a beginning-middle-end. It does **not** answer questions, follow
instructions, or know facts — it's a story model, not a chatbot. That's the
whole point: it's small enough to run on a microcontroller.

## What's inside
- `chat.py` — the prompt loop + a self-contained pure-Python BPE tokenizer (no pip), over the C engine
- `chat_rust.py` — the same loop over the Rust engine (`engine/target/release/gen-prompt`); imports `chat.py`'s tokenizer rather than copying it
- `run_cli.sh` — builds `gen_prompt` from `../reference-c/esp32-llm-lab/gen_prompt.c` and launches `chat.py`
- `bpe32768.json` — the tokenizer (BPE, HuggingFace `tokenizers` format), paired with `../reference-c/esp32-llm-lab/vocab.h`
- `sample-output.jpg` — a real transcript from this exact model

Trained on the TinyStories dataset. Architecture:
[slvDev/esp32-ai](https://github.com/slvDev/esp32-ai) (Per-Layer Embeddings
for microcontrollers) — see `model/README.md` for the full writeup.
