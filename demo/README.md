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
only adds `bpe32768.json` (the tokenizer) and `chat.py` (the REPL) on top.

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
- `chat.py` — the prompt loop + a self-contained pure-Python BPE tokenizer (no pip)
- `run_cli.sh` — builds `gen_prompt` from `../reference-c/esp32-llm-lab/gen_prompt.c` and launches `chat.py`
- `bpe32768.json` — the tokenizer (BPE, HuggingFace `tokenizers` format), paired with `../reference-c/esp32-llm-lab/vocab.h`
- `sample-output.jpg` — a real transcript from this exact model

Trained on the TinyStories dataset. Architecture:
[slvDev/esp32-ai](https://github.com/slvDev/esp32-ai) (Per-Layer Embeddings
for microcontrollers) — see `model/README.md` for the full writeup.
