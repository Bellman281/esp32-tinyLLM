# Benchmarking

How the C-vs-Rust numbers in the [README](./README.md) were produced, how to
reproduce them, and what to optimize next. The short version lives in the
README; this is the detail behind it.

## Configuration

Identical on both sides. That matters more than it sounds — see
[Why matched settings matter](#why-matched-settings-matter) below.

| | |
|---|---|
| Board | ESP32-S3-DevKitC N16R8, chip rev. v0.2, dual LX7 @ 240 MHz |
| Memory | 8 MB PSRAM (OPI), 16 MB flash (QIO), 512 KB SRAM |
| Model | `reference-c/esp32-llm-lab/model.bin`, 14,912,332 bytes |
| SHA-256 | `3e5870b42d54e9f3a58b7b7d2a483b6f85170b5732bc1af6ac21a488a7fa7bbe` |
| Config | `V=32768 D=96 L=6 H=4 F=66 P=128 seq_len=512 group=128` |
| Prompt | 18 tokens, `433,447,259,405,12,406,282,259,403,450,591,504,309,1719,265,309,1512,1222` |
| Generated | 400 tokens, greedy decode |
| Output | serial only, no display |

The prompt and token count come from
[`model/models.toml`](./model/models.toml), and
`engine/llm-host/tests/benchmark_settings_match.rs` fails the build if
either firmware drifts from them.

## Results

| ms/token | input | attn | ffn | ple | head | total | tok/s |
|---|---|---|---|---|---|---|---|
| **C reference** | 4.4 | 42.9 | 6.9 | 8.5 | 57.1 | **119.8** | 8.20 |
| **Rust** | 7.8 | 54.4 | 11.9 | 15.2 | 90.3 | **179.6** | 5.47 |
| ratio | 1.77x | 1.27x | 1.72x | 1.79x | 1.58x | **1.50x** | |
| absolute gap | +3.4 | +11.5 | +5.0 | +6.7 | **+33.2** | +59.8 | |

Both measured the same evening on the same board. C: 400 tokens in 48.78 s,
reproduced bit-identically across three independent runs (same total to
0.01 s, same per-stage figures to 0.1 ms). Rust: 400 tokens in 73.07 s from
`cargo run --release` on the current tree. Neither is quoted from anywhere.

Upstream publishes its own C figures, and they are not used here: they were
measured against a different model file, and upstream's two READMEs disagree
with each other (94.9 vs 102.9 ms/token). Rebuilding and re-measuring on
this board settled it at 119.8 at these settings — and 103.1 at upstream's
own 200-token/4-token-prompt defaults, which is where their figure came
from.

**The two firmwares emit byte-identical output** for all 400 tokens — same
story, same punctuation, cut off at the same word. Two independent
implementations of the same math, including the dual-core int8-staged head,
agreeing token for token on real silicon. Boot diagnostics agree too
(`head staged int8: 2.54 MB`; PSRAM free 3141 KB for C, 3211 KB for Rust —
different allocators, same model).

## Reproducing

### Rust

```bash
cd engine/llm-firmware
cargo run --release        # builds, flashes, opens the monitor
```

### C

Needs `arduino-cli` and the pinned ESP32 core:

```bash
brew install arduino-cli   # or see arduino.github.io/arduino-cli
arduino-cli config init
arduino-cli config add board_manager.additional_urls \
  https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
arduino-cli core update-index
arduino-cli core install esp32:esp32@3.3.10
```

Then build, flash the model partition, flash the app, and watch:

```bash
cd reference-c
FQBN='esp32:esp32:esp32s3:UploadSpeed=921600,CPUFreq=240,FlashMode=qio,FlashSize=16M,PartitionScheme=custom,PSRAM=opi,DebugLevel=info'
PORT=/dev/cu.usbmodem<YOURS>          # ls /dev/cu.usbmodem*
ESPTOOL=~/Library/Arduino15/packages/esp32/tools/esptool_py/5.3.0/esptool

arduino-cli compile --fqbn "$FQBN" \
  --build-property compiler.optimization_flags=-O3 \
  --build-path /tmp/esp32-llm-build firmware/esp32_llm

"$ESPTOOL" --chip esp32s3 --port "$PORT" --baud 921600 \
  write-flash 0x110000 esp32-llm-lab/model.bin

arduino-cli upload -p "$PORT" --fqbn "$FQBN" \
  --input-dir /tmp/esp32-llm-build firmware/esp32_llm

arduino-cli monitor -p "$PORT" --config baudrate=115200
```

Three things that will bite you if you deviate:

- **Flash `esp32-llm-lab/model.bin`, not `firmware/model/model.bin`.** The
  latter is a smaller, differently-trained fixture model. Only the former
  matches the config and SHA-256 above.
- **Omit `USBMode`/`CDCOnBoot` from the FQBN.** The C README's documented
  command sets `USBMode=hwcdc,CDCOnBoot=cdc`, routing `Serial` through the
  S3's native USB-Serial-JTAG peripheral. On DevKitC boards whose USB
  connector is wired to an external UART bridge, that produces unreadable
  output. Dropping both falls back to UART0, which works.
- **A burst of garbage before the first token is normal** — ROM bootloader
  chatter at a different baud rate, before the sketch's `Serial`
  initializes.

To re-run without reflashing (the sketch runs once at boot, then idles):

```bash
"$ESPTOOL" --chip esp32s3 --port "$PORT" --after hard-reset chip-id
arduino-cli monitor -p "$PORT" --config baudrate=115200
```

## Host-side comparison (no hardware)

The numbers above are the ones that matter and they need a board. For a
30-second check on a laptop — after a `llm-core` change, before reaching for
the ESP32 — `scripts/bench_host.py` times the same two engines the demo
REPLs drive (`chat.py` → C `gen_prompt`, `chat_rust.py` → Rust `gen-prompt`):

```bash
cd engine/llm-host && cargo build --release    # once
scripts/bench_host.py                          # from the repo root
```

It takes prompt and token count from `model/models.toml`, runs greedy only,
and **gates on parity before printing any timing** — if the two engines stop
agreeing byte-for-byte it reports that and refuses to report speed. Per-token
cost comes from the slope between a 50-token and a 400-token run, which
cancels the fixed ~15 MB model load; each point is best-of-K wall clock.

Measured on an x86-64 Linux host, `cc -O3` vs. `cargo build --release`,
best of 9, reproduced across three runs (1.48x / 1.47x / 1.47x):

| host, ms/token | per token | tok/s | ratio |
|---|---|---|---|
| **C reference** | 1.15 | 867 | |
| **Rust** | 1.70 | 589 | **1.47x** |

### By hand

The script is only a wrapper — nothing stops you driving the two binaries
directly, and it's worth doing once to see there's no magic in it. Build
both engines, pull the settings from the registry so the two runs match,
then time each:

```bash
cd engine/llm-host && cargo build --release && cd ../..
cc -O3 -I reference-c/esp32-llm-lab -o /tmp/gen_prompt \
   reference-c/esp32-llm-lab/gen_prompt.c -lm

MODEL=$(scripts/model.sh bin);  VOCAB=$(scripts/model.sh vocab)
IDS=$(scripts/model.sh prompt_ids);  N=$(scripts/model.sh n_generate)

time /tmp/gen_prompt "$MODEL" "$N" "$IDS" 0 > /tmp/c.txt
time ./engine/target/release/gen-prompt "$MODEL" "$VOCAB" "$N" "$IDS" 0 > /tmp/rust.txt

cmp /tmp/c.txt /tmp/rust.txt && echo "parity: identical"
```

Mind the argv difference — C takes no `vocab.h` (it's `#include`d at compile
time), Rust loads it at runtime. Run the `cmp` before believing the timings,
for the reason given above. A representative result:

```
--- C ---     real 0m0.495s
--- Rust ---  real 0m0.753s
parity: identical
```

That raw wall clock is 1.52x, slightly above the script's 1.47x, because it
still contains the ~30–40 ms of startup and model load that the script's
two-point subtraction removes. To do that subtraction by hand, time a short
run too and take the slope:

```bash
time /tmp/gen_prompt "$MODEL" 50 "$IDS" 0 > /dev/null   # then: (t400 - t50) / 350
```

Take the *minimum* of several runs at each point rather than one sample or a
mean — on a laptop the floor is the signal and everything above it is other
processes.

Two caveats on reading this table:

- **These are not device numbers and must not enter the README's table** —
  CONTRIBUTING.md's rule that a host-only timing claim doesn't belong there.
  An x86-64 core has caches and memory bandwidth an LX7 does not, and the
  device gap is dominated by the PSRAM-bound output head, which has no host
  analogue.
- **The per-token figure is a mean over tokens 50–400**, not the
  instantaneous cost at token 400 that the device table reports — `attn`
  grows across that range. Both engines are measured identically, so the
  *ratio* is sound; the absolute ms/token is not comparable to the table
  above.

That the host ratio (1.47x) lands so close to the device's (1.50x) is
**not yet explained and should not be assumed causal.** On the device, the
head is 56% of the absolute gap and is bandwidth-bound; on the host it
cannot be. The agreement may mean the shared fp32 matvec dominates both for
the same reason, or it may be coincidence. Nothing here has isolated it
per-stage — `bench_host.py` times whole runs, not stages. Worth resolving
before anyone treats the host as a proxy for the board.

## Why matched settings matter

The C sketch ships with `N_GENERATE = 200` and a 4-token prompt; the Rust
firmware uses 400 and 18. `attn` is the **only** stage whose cost scales
with sequence position — it scans every cached position so far (`0..=pos`),
while input/ffn/ple/head are fixed-size matvecs that do not care how long
the sequence has grown. Running the C firmware at both settings isolates it
exactly:

| C reference | input | attn | ffn | ple | head | total | tok/s |
|---|---|---|---|---|---|---|---|
| 200 tokens, 4-token prompt (sketch default) | 4.4 | 26.2 | 6.9 | 8.5 | 57.1 | 103.1 | 9.51 |
| 400 tokens, 18-token prompt (matches Rust) | 4.4 | **42.9** | 6.9 | 8.5 | 57.1 | 119.8 | 8.20 |

Every other stage is bit-identical between those two runs; only `attn`
moves, by 64%.

Earlier revisions of this project's README compared Rust at 400/18 against
C at 200/4 and concluded attention was the port's worst stage at 2.10x,
recommending it as the top optimization target. It is not — at matched
settings it is **1.27x, the closest stage to parity** — and the overall gap
is 1.50x rather than 1.69x. The measurement was never wrong; the two runs
were simply not comparable.

`benchmark_settings_match.rs` is the regression gate for this, asserting
both firmwares' constants against the registry.

## Optimization history (Rust)

| ms/token | input | attn | ffn | ple | head | total | tok/s |
|---|---|---|---|---|---|---|---|
| first run | 10.6 | 66.7 | 15.8 | 20.4 | 193.8 | 307.4 | 3.21 |
| + dual-core, byte-pair unroll | 8.0 | 58.9 | 11.8 | 15.1 | 110.9 | 204.8 | 4.79 |
| + wider caches | 7.4 | 53.7 | 11.2 | 14.3 | 87.6 | 174.2 | 5.64 |

All at 400 tokens / 18-token prompt, so comparable to each other and to the
C row above. Three changes, in order of what they were worth:

1. **The dual-core head split was not actually dual-core.** `head.rs` pins
   its worker to CPU0, faithfully copying the C reference's
   `xTaskCreatePinnedToCore(..., 0)`. But the C reference is an Arduino
   sketch, and arduino-esp32 runs `loop()` on CPU1; ESP-IDF instead pins
   the main task to CPU0 by default. Both halves of the head matvec were
   queueing on one core with a task switch between them.
   `CONFIG_ESP_MAIN_TASK_AFFINITY_CPU1` restores the C topology.
2. **`QT::matvec_range` did two loads per multiply-accumulate.** C's
   `matvec_q_range` unpacks both nibbles of each packed byte in one pass;
   the Rust port called a `code(r, j)` helper per column, re-reading the
   same byte and branching on `j & 1`. Now unrolled the same way, asserted
   bit-identical to a naive reference.
3. **The external-memory caches were at ESP-IDF's defaults**, not
   arduino-esp32's: 32KB data cache with a 32-byte line, 16KB instruction
   cache. Doubling all three (48 KB of internal SRAM) took the head from
   110.9 to 87.6 ms — it streams 2.43 MB of PSRAM sequentially, exactly the
   access pattern a wider cache line helps.

The current tree measures **179.6**, not that last row's 174.2. Two things
landed since: `attention.rs`/`ops.rs` zip-based loops (an improvement when
measured), and the `matvec_override` hook, which adds a branch to every
matvec call. That hook has no shipping user yet, and its cost looks like
roughly 2 ms/token — but that has not been isolated by a clean before/after
on one build, so treat it as a suspect rather than a measurement. Worth an
A/B before anyone optimizes around it.

**What is left is not a port bug.** Every stage is a line-for-line
translation of the same loop in `reference-c/firmware/common/llm.h`, and
the host parity suite proves the arithmetic is bit-identical. The remaining
gap is most likely GCC's Xtensa backend against LLVM's on scalar float and
integer dot-product loops, and the lever that would actually beat it is the
S3's SIMD unit, not more flag-tuning.

## What to optimize next

Two levers, ranked by what the measured table says.

### 1. The fp32 matvec — worst ratios, scaffolding already in `main`

`ple` (1.79x), `input` (1.77x) and `ffn` (1.72x) are furthest from parity,
and all three run through `QT::matvec_range`, the fp32-dequantized-weight
path. So do `qkv` and `attn_proj`. Together, +15.1 ms/token.

> An earlier version of this section named `QT::matvec_int8_activations` as
> the SIMD target. The firmware never calls it — like the C sketch, it does
> not define `LLM_INT8_ACT` / enable the `int8-activations` feature, so
> every tensor except the output head runs the plain fp32 `matvec_range`.
> The int8 activation path exists only on the host, for the parity tests.

The ESP32-S3's 128-bit PIE vector unit is the lever, reachable only through
hand-written assembly or Espressif's `esp-dsp` C library — Xtensa's LLVM
backend will not autovectorize these loop shapes, and `core::simd` does not
target them. What already ships to make this attemptable without breaking
parity:

- **`llm_forward_with_matvec_override`** /
  **`llm_forward_profiled_with_matvec_override`** (`llm-core/src/model.rs`)
  replace every non-head matvec with a platform-supplied closure — the same
  pattern the dual-core head override already uses, keeping `llm-core`
  platform-agnostic per `MIGRATION_PLAN.md`.
- **`QT::matvec_range_chunked`** (`llm-core/src/tensor.rs`) +
  **`llm-host/tests/matvec_simd_tolerance.rs`** answer the hard part: *how
  much numeric drift is acceptable*. `matvec_range` accumulates in `f32`,
  and any real SIMD dot product reorders that sum — unlike the head's int8
  accumulation, where reordering is provably lossless. That drift is
  measured, not guessed: on the real validation set, SIMD-shaped
  reorderings (4-lane, 8-lane, and whole-row-dequant) move cross-entropy by
  ~3–4e-8, roughly 45,000x inside the noise floor the int8 path already
  tolerates.

What remains is ESP-IDF-side: vendor `esp-dsp` (an IDF Component Registry
component, not a crates.io crate — needs an `idf_component.yml` plus
confirmation your `embuild`/`esp-idf-sys` versions pick it up), confirm the
real `dsps_dotprod_f32` signature against the vendored header rather than
assuming it, unpack each row's packed nibbles into a contiguous buffer
first, and benchmark. An FFI call per row has real overhead at these row
lengths (66–128 elements), so this is **not** a guaranteed win — it needs a
before/after on hardware, not an assumption that SIMD is faster.

### 2. The output head — biggest absolute gap, but diagnose first

At +33.2 ms/token the head is **56% of the entire Rust-vs-C gap**, far more
than any other stage in absolute terms, despite a middling 1.58x ratio. It
is already int8-staged and split across both cores, and it streams 2.43 MB
of PSRAM per token. The `+ wider caches` change moved it more than any
other stage (-21%), pointing at memory bandwidth rather than instruction
throughput. Worth establishing which of the two the remaining gap actually
is *before* assuming SIMD is the answer —
`reference-c/firmware/bandwidth_bench/` exists for exactly this measurement
and has not been ported (Phase 5).

### Not a priority: `attn` (1.27x)

Closest stage to parity. Earlier revisions recommended it as the top target
on the strength of a 2.10x figure that was a benchmark artifact — see
[Why matched settings matter](#why-matched-settings-matter).

Both levers need real ESP-IDF hardware to benchmark. Good first issues for
a contributor who has a board — see [CONTRIBUTING.md](./CONTRIBUTING.md).
