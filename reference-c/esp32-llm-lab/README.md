# esp32-ai — test it on your own machine

Everything in the chat demo ran in a **cloud sandbox, not on your laptop**. This
bundle lets you run it yourself. Pick a level.

The included `model.bin` was trained on a **synthetic TinyStories-style corpus**
(the real dataset was blocked in the sandbox). It writes simple, coherent little
stories. For real-TinyStories quality, use Option B — your laptop can download
the real dataset.

Note: this folder is strictly the C/C++ inference code and its data
(`llm.h`, `gen_prompt.c`, `host_generate.c`, `model.bin`, `vocab.h`) — the
interactive Python demo that builds on top of it lives in
[`../../demo/`](../../demo/) instead, one level up, to keep this folder
free of non-C files. See Option A below.

---

## Option A — Instant, interactive (recommended)

Runs the *exact* ESP32 inference code (`llm.h`/`gen_prompt.c`) as a plain C
program, wrapped in a small Python REPL. Needs only **python3** and a **C
compiler**. On macOS, once: `xcode-select --install`.

```sh
cd ../../demo    # from this folder; or just `cd demo` from the repo root
sh run_cli.sh
```

You'll get a `prompt>`. Try `Once upon a time`, `The little robot`, `One
rainy day, Sara`. Commands: `:temp 0.8` (randomness, `0` = exact greedy chip
behavior), `:n 150` (length), Ctrl-C to quit. See `../../demo/README.md`
for more.

---

## Option A' — Instant, one-shot (no interaction)

The same C program, run once against a fixed prompt baked in at compile time:

```sh
sh run_host.sh          # generates ~120 tokens
# or:  cc -O3 -o gen host_generate.c -lm && ./gen model.bin 200
```

You should see: `Once upon a time, there was a little dog named Ella...`

---

## Option B — From scratch with the REAL TinyStories (best quality)

In your cloned repo (`/Users/zerocode/mygit3/ml-baremetal/vendor/esp32-ai`):

```sh
pip install torch numpy tokenizers tqdm requests   # or: uv sync
python data/prepare.py --vocab 4096                # downloads real TinyStories
python src/train.py --arm ple --vocab 4096 --steps 4000 --tag mine
python src/export.py ple-mine-s0                   # writes firmware/model/model.bin
# verify the C runtime matches PyTorch exactly, then generate:
cc -O3 -o verify firmware/host_verify/verify.c -lm
./verify firmware/model/model.bin firmware/model/golden.txt
python src/sample.py --run runs/ple-mine-s0.pt --prompt "Once upon a time"
```

That trains the real thing (a few hours on CPU, minutes on a GPU/Apple-Silicon MPS).

---

## Option C — Device-faithful emulator (ESP32-S3 QEMU)

Boot the real firmware on an emulated ESP32-S3. Requires Espressif's QEMU
(`qemu-system-xtensa` with the `esp32s3` machine — install via ESP-IDF:
https://docs.espressif.com/projects/esp-idf/en/stable/esp32s3/api-guides/tools/qemu.html).

```sh
sh run_qemu.sh
# = qemu-system-xtensa -nographic -machine esp32s3 -m 8M \
#     -drive file=flash.bin,if=mtd,format=raw
```

Neither `flash.bin` (the prebuilt 16 MB image) nor `run_qemu.sh` are
currently checked into this folder — both were dropped as unused (see
`../README.md`'s "Removed" note). To use this option, regenerate both from
`slvDev/esp32-ai`. Note: the tok/s QEMU reports is emulated time, not real
silicon.

---

## Option D — Real hardware (~$8)

Buy an **ESP32-S3-DevKitC-1 N16R8** and follow
`firmware/esp32_llm/README.md` in the repo (build with `arduino-cli`, flash the
app, then flash `model.bin` to `0x110000`). This is the only way to get the real
~9.5 tok/s speed number.

---

### Files (this folder — C/C++ and data only)
- `model.bin` — trained 28.9M-param model, 4-bit (~15 MB)
- `vocab.h` — token → bytes table for the C/Rust runtimes (generated from the tokenizer in `../../demo/bpe32768.json`)
- `llm.h` — the portable inference runtime (identical to the firmware's)
- `gen_prompt.c` — host runner with sampling (greedy or top-k+temperature); built by `../../demo/run_cli.sh`
- `host_generate.c` — host runner (the chip's exact `llm.h`, greedy decode only, fixed baked-in prompt)

The Python/shell demo layer (`chat.py`, `run_cli.sh`, `bpe32768.json`,
sample output) lives in [`../../demo/`](../../demo/), not here.

Note: `run_host.sh`, `run_qemu.sh`, and `flash.bin` (Options A' and C above)
are referenced here but aren't currently checked into this folder. If you
need them, regenerate from `slvDev/esp32-ai` or ask whoever last exported
this bundle.
