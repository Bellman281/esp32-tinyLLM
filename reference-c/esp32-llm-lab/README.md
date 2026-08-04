# esp32-ai — test it on your own machine

Everything in the chat demo ran in a **cloud sandbox, not on your laptop**. This
bundle lets you run it yourself. Pick a level.

The included `model.bin` was trained on a **synthetic TinyStories-style corpus**
(the real dataset was blocked in the sandbox). It writes simple, coherent little
stories. For real-TinyStories quality, use Option B — your laptop can download
the real dataset.

---

## Option A — Instant (10 seconds, no hardware, no Python)

Runs the *exact* ESP32 inference code (`llm.h`) as a plain C program.
Needs only a C compiler. On macOS, once: `xcode-select --install`.

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

`flash.bin` is the prebuilt 16 MB image (bootloader + app + this model). Note: the
tok/s QEMU reports is emulated time, not real silicon.

---

## Option D — Real hardware (~$8)

Buy an **ESP32-S3-DevKitC-1 N16R8** and follow
`firmware/esp32_llm/README.md` in the repo (build with `arduino-cli`, flash the
app, then flash `model.bin` to `0x110000`). This is the only way to get the real
~9.5 tok/s speed number.

---

### Files
- `host_generate.c` — host runner (the chip's exact `llm.h`, greedy decode)
- `llm.h` — the portable inference runtime (identical to the firmware's)
- `vocab.h` — token → bytes table for this tokenizer
- `model.bin` — trained 4-bit model (1.87 MB)
- `flash.bin` — prebuilt ESP32-S3 QEMU flash image (16 MB)
- `run_host.sh`, `run_qemu.sh` — one-line runners
