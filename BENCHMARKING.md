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

<!-- BEGIN device-table (generated: scripts/plot_stages.py --inject) -->

| ms/token | input | attn | ffn | ple | head | stages | **wall** | tok/s |
|---|---|---|---|---|---|---|---|---|
| **C reference** | 4.4 | 42.9 | 6.9 | 8.5 | 57.1 | 119.8 | **122.0** | 8.20 |
| **Rust, first run** | 10.6 | 66.7 | 15.8 | 20.4 | 193.8 | 307.3 | **311.5** | 3.21 |
| **+ dual-core head, unroll** | 8.0 | 58.9 | 11.8 | 15.1 | 110.9 | 204.7 | **208.8** | 4.79 |
| **+ wider caches** | 7.4 | 53.7 | 11.2 | 14.3 | 87.6 | 174.2 | **177.3** | 5.64 |
| **Rust, before** | 7.8 | 54.4 | 11.9 | 15.2 | 94.6 | 183.9 | **187.2** | 5.34 |
| **Rust, nibble LUT** | 6.6 | 50.8 | 10.1 | 12.7 | 98.0 | 178.2 | **181.6** | 5.51 |
| **Rust, + KV layout** | 6.6 | 47.0 | 10.1 | 12.7 | 94.8 | 171.2 | **174.6** | 5.73 |
| **+ int4 head** | 6.6 | 47.0 | 10.1 | 12.7 | 94.8 | 171.2 | **174.6** | 5.73 |
| **+ 4 accumulators** | 6.6 | 47.0 | 10.1 | 12.7 | 81.3 | 157.7 | **161.0** | 6.21 |
| **+ bias hoist** | 6.6 | 47.0 | 10.1 | 12.7 | 61.5 | 137.9 | **140.9** | 7.09 |
| **+ dual-core matvec** | 3.9 | 40.2 | 6.2 | 7.5 | 61.6 | 119.4 | **122.7** | 8.15 |
| **+ argmax/stdout** | 3.8 | 39.8 | 6.0 | 7.3 | 61.5 | 118.4 | **121.1** | 8.25 |
| **+ argmax in head** | 3.9 | 40.0 | 6.1 | 7.5 | 61.5 | 119.0 | **119.0** | 8.40 |
| **+ attn instrumented** | 3.9 | 39.0 | 6.2 | 7.6 | 61.9 | 118.6 | **118.5** | 8.43 |
| **+ attn heads split** | 3.9 | 26.4 | 6.2 | 7.6 | 62.7 | 106.8 | **106.8** | 9.36 |
| **+ token digest** | 3.9 | 26.3 | 6.2 | 7.6 | 62.3 | 106.3 | **106.3** | 9.41 |
| **+ MSPI bus 120 MHz** ‡ | 3.6 | 24.5 | 5.7 | 7.0 | 62.1 | 102.9 | **102.9** | 9.71 |
| **+ int8 head (reverted)** ‡ | 4.0 | 26.8 | 6.2 | 7.6 | 76.9 | 121.5 | **121.6** | 8.23 |
| **+ probe, always on** ‡ | 3.8 | 26.2 | 6.0 | 7.3 | 63.2 | 106.5 | **106.6** | 9.38 |
| **Rust, now** | 3.9 | 26.3 | 6.2 | 7.6 | 62.2 | 106.2 | **106.3** | 9.41 |
| **Rust, --fast-mspi** ‡ | 3.8 | 24.7 | 6.0 | 7.3 | 60.9 | 102.7 | **102.6** | 9.75 |
| ratio vs C reference | 0.89x | 0.61x | 0.90x | 0.89x | 1.09x | 0.89x | **0.87x** | |
| absolute gap | -0.5 | -16.6 | -0.7 | -0.9 | +5.1 | -13.6 | **-15.7** | |
| **change this brought** | +0.0 | +0.0 | +0.0 | +0.0 | -0.1 | -0.1 | **+0.0** | |

‡ **+ MSPI bus 120 MHz is not the shipping configuration** and is excluded from the ratios and the summary below. Requires CONFIG_IDF_EXPERIMENTAL_FEATURES; Espressif documents 120 MHz as temperature-sensitive and not recommended across the industrial range, and this board's boya flash part refused it and fell back. Real, reproducible, bit-exact -- and not what you get by cloning this repo. Available as a +3.3% opt-in; see BENCHMARKING.md.
‡ **+ int8 head (reverted) is not the shipping configuration** and is excluded from the ratios and the summary below. A negative result, kept because it corrects a claim this file made. Staging the head as unpacked int8 instead of packed int4 -- the C reference's format, and one fewer operation per element -- made the head 62.3 -> 76.9 ms and the token 106.3 -> 121.5. Reverted. Bit-exact throughout: the digest still printed OK, so this is purely a cost.
‡ **+ probe, always on is not the shipping configuration** and is excluded from the ratios and the summary below. Superseded by [run.rust-findings-off]. Kept because it is the measurement that justified making the probe opt-in: identical arithmetic, 0.3 ms slower, purely from ~50 lines sitting in head.rs.
‡ **Rust, --fast-mspi is not the shipping configuration** and is excluded from the ratios and the summary below. One flag from the default build -- scripts/run_device.sh --fast-mspi -- and not the default, for reasons about other boards rather than this one: CONFIG_IDF_EXPERIMENTAL_FEATURES, Espressif documenting 120 MHz as temperature-sensitive, and this board's boya flash part refusing it and falling back. Bit-exact: 102.6 ms/token, 9.75 tok/s, 18.9% faster than the C reference, digest 0x327578cb136fd6aa OK.

**`attn` broken down** — the sub-stages the five-stage profile hides. `qkv`/`proj` are the fp32 matvec that also drives `ffn` and `ple`; `core` is attention proper, the only part that grows with sequence position. The C reference has no equivalent instrumentation, so it is absent rather than zero.

| attn ms/token | qkv | rope | core | proj | sum | both cores |
|---|---|---|---|---|---|---|
| **+ attn instrumented** | 7.8 | 0.14 | 28.3 | 2.7 | 38.9 | `qkv`, `proj` |
| **+ attn heads split** | 7.9 | 0.15 | 15.6 | 2.7 | 26.3 | `qkv`, `proj`, `core` |
| **+ token digest** | 7.8 | 0.15 | 15.6 | 2.7 | 26.2 | `qkv`, `proj`, `core` |
| **+ MSPI bus 120 MHz** | 7.2 | 0.15 | 14.7 | 2.5 | 24.6 | `qkv`, `proj`, `core` |
| **+ int8 head (reverted)** | 7.9 | 0.15 | 16.0 | 2.7 | 26.8 | `qkv`, `proj`, `core` |
| **+ probe, always on** | 7.6 | 0.15 | 15.9 | 2.6 | 26.2 | `qkv`, `proj`, `core` |
| **Rust, now** | 7.8 | 0.15 | 15.6 | 2.7 | 26.2 | `qkv`, `proj`, `core` |
| **Rust, --fast-mspi** | 7.5 | 0.15 | 14.5 | 2.6 | 24.8 | `qkv`, `proj`, `core` |

**Rust is 14.8% faster per token than the C reference it was ported from** — 106.3 ms against 122.0, 9.41 tok/s against 8.20, on the same board with byte-identical output. It beats C on 4 of the 5 stages.

<!-- END device-table -->

`total` is the sum of the profiled stages; `tok/s` comes from wall clock, which
runs 2–3 ms/token above that sum on both engines — token sampling, argmax and
serial output sit outside the profiled stages. The two agree on deltas: wall
went 187.2 → 181.6 ms/token across the nibble-LUT change, against the −5.7 ms
the stage sum shows.

This table and `png/benchmark-stages.svg` are both generated from
[`benchmarks/device.toml`](./benchmarks/device.toml) by
`scripts/plot_stages.py --inject`. Before that existed the chart was a
hand-drawn PNG with no generator and the table was hand-maintained in two
files, so both aged silently every time someone reflashed. Record a new run as
its own `[run.<id>]` section rather than editing an existing one — the point is
that what was measured, when, and against which commit survives.

The C row and the first Rust row were measured the same evening on the same board. C: 400 tokens in 48.78 s,
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

## Where the time goes now

Every stage except the head is faster than C, and the head is 1.08x. Outside
the profiled stages there is now essentially nothing:

```
outside stages: argmax 0.00 | emit 0.03 | other 0.00 ms/token  (wall 119.0 vs forward 119.0)
```

It did not start there. It was `argmax 2.74`, and the fix was not a faster
loop: the greedy pick was reading 25,353 logits back out of PSRAM, 101 KB at
roughly half the measured bus rate, because a data-dependent branch per element
leaves nothing to pipeline. Both cores already hold each logit in a register the
moment `rows_range` computes it, so each now tracks the maximum over its own
rows and the decode loop chooses between two candidates. Tie-breaking is
identical — first-strictly-greater within a half, `hi > lo ? hi : lo` between
halves, which yields the lower index on a tie exactly as a `0..n` scan does.

**The C reference still pays that cost**: 122.0 ms wall against 119.8 profiled.
Its `pick()` scans all 25,353 logits too. That 2.2 ms is most of the margin
between the two engines, and it is the one place this port is not doing the same
work as its reference — it is doing less of it, for the same output.

What remains, in order of size: `attn` at 40.0 ms is a third of a token and has
never been instrumented the way the head was; the fp32 `matvec_range` runs at
~21 cycles/MAC against the head's 12.1 and carries the same single-accumulator
chain the head shed, though fp32 reordering needs the tolerance
`matvec_simd_tolerance.rs` has already measured.

## Instrumentation

Two things the firmware prints beyond the five-stage line, both added after the
five stages proved too coarse twice in one day — once failing to separate a
code-placement shift from a real regression, once failing to explain why
halving the head's memory traffic changed nothing.

**A PSRAM bandwidth probe**, at boot, over the head's own staged weights — same
allocation, same direction, same cold-stream pattern, far larger than the data
cache:

```
PSRAM read: 72 MB/s  (1.22 MB of head weights => 16.9 ms/token floor)
```

Read it as a floor. Streaming `n` MB at `b` MB/s cannot cost less than
`1000n/b` ms, and whatever a stage spends above that floor is arithmetic no
amount of compression will reach. This is the minimum useful part of
`reference-c/firmware/bandwidth_bench/` (Phase 5, still unported) — enough to
choose between attacking bytes and attacking instructions, which is the choice
this project got wrong once.

**Head sub-stage timers**, under the profile line:

```
head detail: quant 0.04 | own-half 61.4 | wait-worker 0.0 | worker-half 61.4
```

- `own + wait` should account for nearly all of the `head` figure.
- `wait` near zero means the calling core is the slow half and the row split is
  fine. `wait` large would mean the worker's half takes materially longer and
  moving `split` off `rows / 2` is free throughput. It has measured 0.0 every
  time, so the even row split is also an even time split — worth knowing, and
  cheap to stop wondering about.
- `worker` is the other half timed on the worker itself, so `wait` much larger
  than `worker - own` would be scheduling latency rather than compute.
- `quant` should be negligible (96 elements). It is 0.04.

Cost: four `esp_timer_get_time` calls per token on the inference core, two on
the worker, against stages measured in tens of milliseconds. Not feature-gated
— the point is that it is on when someone reflashes and wonders where the time
went.

## Reproducibility and the layout tax

**A given binary reproduces exactly on this board.** The C firmware was already
known to (three runs, same total to 0.01 s, same per-stage figures to 0.1 ms);
both Rust binaries in the table above were then re-run from a hard reset and
did the same. One run per binary is therefore enough, and `--repeat`-style
averaging buys nothing.

The consequence is less comfortable, and it caught this project out once
already. Because there is no run-to-run noise to absorb it, **a difference
between two binaries in a stage neither of them touches is real.** `head` has
now moved twice that way:

| head | binary | what changed |
|---|---|---|
| 90.3 | pre-`c7f2ce7` | — |
| 94.6 | `feat/ple-v2-header` | PLE\0 header parse, `out_vocab`; `head.rs` untouched |
| 98.0 | `perf/nibble-lut` | two lookup tables, four reshaped loops; `head.rs` untouched |

Neither commit changes a line the head executes. What they change is how much
code and `.rodata` `llm-core` occupies, and the head streams 2.43 MB through a
64 KB cache every token — a stage where instruction and data placement is
worth several percent. Budget for it: an `llm-core` change that leaves the head
alone can still cost the head 3–4 ms, and a change measured only on the stages
it targets will overstate its net.

The first reading of the 90.3 → 94.6 → 98.0 sequence here was thermal drift
across back-to-back runs. It was not; the repeat runs killed that. Recorded
because "the board was warming up" is the comfortable explanation and it was
the wrong one.

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

**Resolved: it was coincidence.** The nibble-LUT change took the host from
1.47x to 1.11x while the device moved 1.50x to ~1.49x — the two ratios came
apart the moment anything changed, so their earlier agreement carried no
information. The paragraph below is kept because the reasoning that raised the
suspicion was right; the suspicion itself is now answered.

That the host ratio (1.47x) landed so close to the device's (1.50x) was
**never explained and should not have been assumed causal.** On the device, the
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

The full history is the generated table at the top of this file — it is the
same measurements, from `benchmarks/device.toml`, and there is deliberately
no second copy here. A hand-written duplicate lived at this spot until it
went two runs stale, which is the whole argument for the generator: three
of its rows (`first run`, `+ dual-core head, unroll`, `+ wider caches`)
existed **only** here and have been migrated into `device.toml`, where they
are flagged as migrated with `wall_ms` derived from `tok/s` rather than
measured.

All rows are at 400 tokens / 18-token prompt, so they are comparable to each
other and to the C reference. Three changes, in order of what they were worth:

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

Before the nibble-LUT row the tree measured **179.6**, not the wider-caches
row's 174.2. Two things had landed in between: `attention.rs`/`ops.rs`
zip-based loops (an improvement when measured), and the `matvec_override`
hook, which adds a branch to every matvec call. That hook has no shipping user yet, and its cost looks like
roughly 2 ms/token — but that has not been isolated by a clean before/after
on one build, so treat it as a suspect rather than a measurement. Worth an
A/B before anyone optimizes around it.

**What is left is not a port bug** — and the nibble-LUT row is evidence for
that reading rather than against it: it did not fix a translation error, it
removed per-element work the C compiler never had to do, and the output stayed
bit-identical throughout. Every stage is a line-for-line
translation of the same loop in `reference-c/firmware/common/llm.h`, and
the host parity suite proves the arithmetic is bit-identical. The remaining
gap is most likely GCC's Xtensa backend against LLVM's on scalar float and
integer dot-product loops, and the lever that would actually beat it is the
S3's SIMD unit, not more flag-tuning.

## What to optimize next

Ranked by what the measured table says, as of `Rust, now`:

| | ms/token | share of a token | vs C |
|---|---|---|---|
| **output head** | 62.7 | 59% | 1.10x — the only stage still behind |
| fp32 matvec (`input` + `ffn` + `ple` + `qkv` + `proj`) | 28.3 | 26% | 0.89x |
| attention `core` | 15.6 | 15% | not separately instrumented in C |

The ordering has inverted since this section was written. The fp32 matvec was
the top lever when `ple`/`input`/`ffn` sat at ~1.5x of C; they are now 0.89x
and the head is 59% of a token on its own. **The head is the target**, and it
is the one place where the remaining gap is a like-for-like instruction-count
difference rather than a structural one: at 62.7 ms across two cores it runs
~12.4 cycles per multiply-accumulate against C's ~11.3, so roughly 10% of the
head is scheduling rather than work.

A third item is now open that was not before — see
[Is attention's `core` bandwidth-bound?](#is-attentions-core-bandwidth-bound)
below. It is a measurement, not an optimization, and it is cheap.

### The fp32 matvec — partly done; SIMD still open

**Update.** `ple` (1.79x), `input` (1.77x) and `ffn` (1.72x) were the three
furthest from parity, all running through `QT::matvec_range`. Two bit-exact
changes to that function — a 16-entry nibble→f32 table replacing a per-element
subtract-and-convert, and `zip`-driven steady-state loops replacing indexed
ones — moved them to **1.49x / 1.50x / 1.46x**, and `attn` (which also routes
`qkv` and `attn_proj` through it) from 1.27x to 1.18x. Together −9.1 ms/token,
−10.2% across those four stages, output bit-identical.

That was the cheap half, and it did not need SIMD. The rest of this section —
the ESP32-S3's PIE vector unit — is still open and still the larger prize.

For the record, the pre-change framing: `ple` (1.79x), `input` (1.77x) and
`ffn` (1.72x) were furthest from parity, and all three run through
`QT::matvec_range`, the fp32-dequantized-weight path. So do `qkv` and
`attn_proj`. Together, +15.1 ms/token.

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

### Skipping head rows without reading them — measured, and it cannot work

The firmware computes all 25,353 logits and uses exactly one number from them:
`tok = head.last_argmax()`. Nothing reads `logits`. So in principle a row whose
value *cannot* beat the running maximum need never be read at all — and since
each row's scale lives in a separate 101 KB array, the test costs no PSRAM read
of the row's 48 bytes. At 13.4 ms of exposed memory time, skipping most rows
would be the largest remaining win in the stage.

Two exact bounds were measured on the host, over real decode steps:

| bound | rows skipped |
|---|---|
| max-abs: `\|dot\| <= 8 * sum\|a_j\|` | **0 of 25,353** |
| Cauchy–Schwarz: `\|dot\| <= \|\|c-8\|\|₂ * \|\|a\|\|₂`, row norm precomputed | **0 of 25,353** |

Not "few" — none, on every step tried. The reason is dimensional and does not
depend on tuning. Both bounds are worst-case: they assume every element of the
row aligns in sign with the activation. A real dot over `d = 96` elements with
mixed signs lands around `sqrt(d)` below that, so any such bound is ~10x loose.
The row scales span only **12.1x** (2.011e-2 to 2.432e-1), which is not enough
spread for a 10x-loose bound to separate anything.

Recorded so nobody re-derives it. Making this work would need a bound tight to
within the scale spread, which for exact argmax over high-dimensional inner
products is not available cheaply. An *approximate* pruner would work and would
change the token stream, which is the same trade the fp16 KV cache offers at a
better rate.

### The output head — the arithmetic fixes are done; the diagnosis was the whole story

**Resolved. The head went 94.6 -> 61.5 ms, 1.66x -> 1.08x of C.** What follows
is kept because the reasoning that got there is reusable and the reasoning that
preceded it was wrong.

This section used to say the head was bandwidth-bound, on the strength of the
`+ wider caches` change moving it 21% — more than any change had moved any
stage. Acting on that produced the int4 staging, which halved what the head
streams (2.54 MB -> 1.32 MB), returned 1.2 MB of PSRAM, and moved the stage
**0.0 ms**.

The null result is what forced the stage to be instrumented, and the
instruments answered in one run:

```
PSRAM read: 72 MB/s  (1.22 MB of head weights => 16.9 ms/token floor)
head detail: quant 0.04 | own-half 90.9 | wait-worker 0.0 | worker-half 90.9
```

`wait-worker 0.0` with both halves equal means each core independently
completes its rows in 90.9 ms while the other does the same — so aggregate
throughput was 1.22 MB in 91 ms, **13.4 MB/s against 72 MB/s available**. The
head was using 19% of its own bandwidth. If memory had been the constraint,
two cores could not both have hit 90.9. It was compute-bound the whole time,
and `+ wider caches` helped for some other reason.

What was actually wrong was two operations per element in a loop that runs
2.43M times per token:

1. **One accumulator.** `acc += a[i] * b[i]` makes every add wait for the
   previous one. Four independent accumulators: 91.0 -> 81.3. Bit-exact —
   `i32` addition is associative.
2. **A subtract and a table load that should not have existed.** Codes are
   stored biased, and `sum (c-8)a == sum ca - 8 sum a`, where `sum a` is
   identical for all 25,353 rows. Hoisting it deletes the subtract; deleting
   the subtract also deletes the `NIBBLE_I8` lookup that had been standing in
   for it. 81.3 -> 61.5. Also exact — the identity holds over the integers.

That second one is worth dwelling on. `NIBBLE_I8` was introduced alongside
`NIBBLE_F32`, where the table genuinely wins: it replaces an integer-to-float
*conversion* with a load, measured -24% on the host head. On the integer path
it replaced a *subtract* with a load, which is a worse instruction on any
machine. The pattern was copied without re-deriving whether it applied.

Neither operation was visible at five-stage resolution. Both were obvious once
the stage was split four ways. **The instrument was worth more than any guess
about the stage.**

### 2b. Historical: the output head — biggest absolute gap, but diagnose first

At +40.9 ms/token the head is now **roughly two-thirds of the entire
Rust-vs-C gap** — it was 56% before the nibble-LUT change shrank everything
else — and far more than any other stage in absolute terms. (Its figure was marked
provisional at the time; it no longer is, and the run it referred to is
`+ wider caches` in the generated table.) It
is already int8-staged and split across both cores, and it streams 2.43 MB
of PSRAM per token. The `+ wider caches` change moved it more than any
other stage (-21%), pointing at memory bandwidth rather than instruction
throughput. Worth establishing which of the two the remaining gap actually
is *before* assuming SIMD is the answer —
`reference-c/firmware/bandwidth_bench/` exists for exactly this measurement
and has not been ported (Phase 5).

### Is attention's `core` bandwidth-bound? — yes, and the bus does not scale

**Settled.** The probe ran, and the answer changed what the next lever is.

| | 1 core | 2 cores, disjoint halves | scaling |
|---|---|---|---|
| PSRAM @ 80 MHz | 72 MB/s | 91 MB/s | **1.27x** |
| PSRAM @ 120 MHz | 81 MB/s | 131 MB/s | **1.62x** |

The probe is behind a Cargo feature and **off by default**:

```
cargo run --release --features bandwidth-probe
```

It runs once at boot and still costs 0.3 ms/token — not from executing, but
from ~50 more lines sitting in `head.rs` and moving the code around it
(106.6 with it, 106.3 without, reproducible across three runs). A measurement
tool that taxes every token to do nothing during any of them does not belong
in the default build, however useful it was on the day.

A second LX7 buys 27% more bandwidth, not 100%. Both cores finished within
0.1 ms of each other, so this is the bus, not imbalance. Against that ceiling:

| | streams/token | stage | rate | of available |
|---|---|---|---|---|
| output head | 1.22 MB | 63.2 ms | 19.3 MB/s | **21%** |
| attention `core` | 1.01 MB | 15.9 ms | 63.3 MB/s | **70%** |

**Do not read the head's 21% as "compute-bound, so bytes do not matter".**
That inference was made here and then falsified: staging the head as unpacked
int8 rather than packed int4 doubled its bytes to 2.43 MB and cost **14.6 ms**,
against the 13.3 ms those bytes take at 91 MB/s. Its memory time is additive
and fully exposed. Low utilization means it cannot *saturate* the bus, not that
it is insensitive to how much it asks the bus for.

Attention reads the whole KV cache every token — `219 positions x 96 floats x
4 bytes x 2 x 6 layers` at the benchmark settings, almost as much as the entire
output head. That is the exact inverse of what this document assumed about the
two stages a day earlier.

**Raising the MSPI clock is the bit-exact lever, and it is worth 3.4 ms.**
PSRAM 80 -> 120 MHz took the token 106.3 -> 102.9 with the digest unchanged.
It forces flash along with it: the two share the MSPI peripheral, and at the
240 MHz core clock that 120M PSRAM implies, ESP-IDF ships no 80M-flash timing
table — the build fails outright saying so.

**Two things that were guessed wrong here, both cheaply.** First, the win was
predicted at ~7 ms by modelling each stage as `memory + compute` and halving
the memory term. The head moved **0.3%** while its memory floor dropped from
13.4 to 9.3 ms — which is what compute-bound actually means: the transfer was
already hidden behind arithmetic, so making it faster changes nothing. The
additive model was wrong for the stage it mattered most for.

Second, the per-stage shape (`input -7.7%`, `qkv -7.7%`, `ffn -8.1%`,
`ple -7.9%`, `proj -7.4%`, `core -5.8%`, `head -0.3%`) looked like a *flash*
win, because everything that moved reads weights through the flash mmap and
the head does not. A follow-up raising flash 40 -> 80 MHz with PSRAM left at
80 measured **106.6 ms — nothing at all**, ruling that out. The remaining
explanation, inferred rather than measured: the shared MSPI *core* clock going
160 -> 240 MHz cuts per-miss latency for flash and PSRAM alike, which helps
stages that take many scattered cache misses and does nothing for a stage that
streams sequentially and is compute-bound anyway.

**The caveat that decides whether to keep it.** 120 MHz needs
`CONFIG_IDF_EXPERIMENTAL_FEATURES`, is documented by Espressif as
temperature-sensitive and not recommended across the industrial range, and
this board's flash logs `High performance mode of this flash model hasn't been
supported` — the boya part refused 120 MHz flash and fell back. It is a
reasonable setting for a benchmark board on a desk and an unreasonable one for
a product.

**What is left for memory.** An fp16 KV cache would halve attention's 1.01 MB.
Measured cost, before building it: validation cross-entropy +1.1e-5 (one
hundred and eighteenth of the int8-activation noise floor this project already
ships with), but the token stream forks at generated token 136 under the
firmware's head — one flipped argmax, then a different continuation, no
re-convergence. See `llm-host/tests/fp16_kv_tolerance.rs`. It would retire
"byte-identical output" as a claim.

#### The original open question, for the record

Open, and cheap to settle. Splitting the four attention heads two-per-core
took `core` from 28.3 to 15.6 ms. A perfect halving would be 14.15, plus the
0.1 ms of measured sync = 14.25. **1.35 ms, 9.5%, did not come back**, and
this run cannot say which of two causes it is:

- **Bus contention.** Both cores now scan the same KV cache at the same time.
  The head was shown to be compute-bound precisely because two cores each
  independently completed their half at the same speed as one — attention may not have
  that property, since `core` reads far more memory per unit of arithmetic.
- **The layout tax.** This binary's boot-time PSRAM probe reads **68 MB/s
  against the previous binary's 72** — the same probe, the same buffer, the
  same code, before generation starts. That is a 5.6% slower memory path from
  placement alone, and `head` moved +0.8 ms in the same binary without
  `head.rs` changing. Most of the 9.5% may simply be this.

The experiment that separates them: run `psram::probe_read_bandwidth` on both
cores simultaneously and compare against one core. If aggregate throughput
does not scale, attention is contending and the fix is a smaller working set
(fp16 KV cache halves it); if it does scale, the shortfall was placement and
there is nothing to chase. `reference-c/firmware/bandwidth_bench/` is the
C-side prior art and is still unported (Phase 5).

All three items need real ESP-IDF hardware to benchmark. Good first issues for
a contributor who has a board — see [CONTRIBUTING.md](./CONTRIBUTING.md).
