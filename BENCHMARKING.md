# Benchmarking

How the C-vs-Rust numbers in the [README](./README.md) were produced, how to
reproduce them, and what is left. The short version lives in the README; this is
the detail, including the measurements that went the wrong way.

## What is measured, and how

Identical settings on both sides -- see
[Why matched settings matter](#why-matched-settings-matter).

| | |
|---|---|
| Board | ESP32-S3-DevKitC N16R8, chip rev. v0.2, dual LX7 @ 240 MHz |
| Memory | 8 MB PSRAM (OPI), 16 MB flash (QIO), 512 KB SRAM |
| Model | `model/tinystories-v32768/model.bin`, 14,912,332 bytes |
| SHA-256 | `3e5870b42d54e9f3a58b7b7d2a483b6f85170b5732bc1af6ac21a488a7fa7bbe` |
| Config | `V=32768 D=96 L=6 H=4 F=66 P=128 seq_len=512 group=128` |
| Prompt | 18 tokens, `433,447,259,405,12,406,282,259,403,450,591,504,309,1719,265,309,1512,1222` |
| Generated | 400 tokens, greedy decode |
| Output | serial only, no display |

That model path is canonical, declared in
[`model/models.toml`](./model/models.toml) and documented in
[`model/README.md`](./model/README.md): `model/` holds the weights,
`reference-c/` is a frozen mirror of the C sketch. Prompt and token count come
from the same registry, and `engine/llm-host/tests/benchmark_settings_match.rs`
fails the build if either firmware drifts from them.

### Why matched settings matter

The C sketch ships with `N_GENERATE = 200` and a 4-token prompt; the Rust
firmware uses 400 and 18. `attn` is the **only** stage whose cost scales with
sequence position -- it scans every cached position so far (`0..=pos`), while
input/ffn/ple/head are fixed-size matvecs. Running C at both settings isolates
it:

| C reference | input | attn | ffn | ple | head | total | tok/s |
|---|---|---|---|---|---|---|---|
| 200 tokens, 4-token prompt (sketch default) | 4.4 | 26.2 | 6.9 | 8.5 | 57.1 | 103.1 | 9.51 |
| 400 tokens, 18-token prompt (matches Rust) | 4.4 | **42.9** | 6.9 | 8.5 | 57.1 | 119.8 | 8.20 |

Every other stage is bit-identical between those runs; only `attn` moves, by 64%.
Earlier revisions of this project's README compared Rust at 400/18 against C at
200/4, called attention the port's worst stage at 2.10x, and made it the top
target. It was not: at matched settings it was **1.27x, the closest stage to
parity**, and the overall gap 1.50x rather than 1.69x. (Both historical, from the
`Rust, before` era; attention now runs 0.63x of C and the token 0.78x.) The
measurement was never wrong -- the two runs were not comparable.

### Reproducing: Rust

```bash
cd engine/llm-firmware && cargo run --release   # builds, flashes, opens the monitor
```

Or `scripts/run_device.sh`, which tees the whole log to `runs/` -- worth it,
because several figures below exist only in boot lines that scroll past.
`--probe` adds the opt-in bandwidth and stall probes; `--fast-mspi` overlays
`sdkconfig.defaults.fast-mspi` (120 MHz MSPI).

### Reproducing: C

Needs `arduino-cli` and the pinned core: `config init`, add
`https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`
to `board_manager.additional_urls`, `core update-index`, then
`core install esp32:esp32@3.3.10`.

```bash
cd reference-c
FQBN='esp32:esp32:esp32s3:UploadSpeed=921600,CPUFreq=240,FlashMode=qio,FlashSize=16M,PartitionScheme=custom,PSRAM=opi,DebugLevel=info'
PORT=/dev/cu.usbmodem<YOURS>          # ls /dev/cu.usbmodem*
ESPTOOL=~/Library/Arduino15/packages/esp32/tools/esptool_py/5.3.0/esptool

arduino-cli compile --fqbn "$FQBN" \
  --build-property compiler.optimization_flags=-O3 \
  --build-path /tmp/esp32-llm-build firmware/esp32_llm

"$ESPTOOL" --chip esp32s3 --port "$PORT" --baud 921600 \
  write-flash 0x110000 ../model/tinystories-v32768/model.bin

arduino-cli upload -p "$PORT" --fqbn "$FQBN" \
  --input-dir /tmp/esp32-llm-build firmware/esp32_llm

arduino-cli monitor -p "$PORT" --config baudrate=115200
```

Three things that will bite you. **Flash the canonical
`model/tinystories-v32768/model.bin`**, not `reference-c/firmware/model/model.bin`
-- the latter is a smaller, differently-trained fixture (1.87 MB, V=4096), and
only the former matches the config and SHA-256 above. **Omit
`USBMode`/`CDCOnBoot` from the FQBN**: the C README's command sets
`USBMode=hwcdc,CDCOnBoot=cdc`, which routes `Serial` through the S3's native
USB-Serial-JTAG and is unreadable on DevKitC boards wired to an external UART
bridge. And **a burst of garbage before the first token is normal** -- ROM
bootloader chatter at another baud rate. The sketch runs once at boot then idles,
so to re-run without reflashing: `esptool ... --after hard-reset chip-id`, then
re-open the monitor.

### Host-side comparison (no hardware)

`scripts/bench_host.py` times the same two engines the demo REPLs drive, taking
prompt and token count from `model/models.toml`. It **gates on parity before
printing any timing**, and per-token cost is the slope between a 50-token and a
400-token run, cancelling the fixed ~15 MB model load.

**Historical, pre-nibble-LUT:** x86-64 Linux, `cc -O3` vs `cargo build
--release`, best of 9, three runs at 1.48x / 1.47x / 1.47x -- C 1.15 ms/token
(867 tok/s), Rust 1.70 (589), **1.47x**. Not re-measured since: the nibble-LUT
change alone took the host ratio to **1.11x**, and eleven device-side changes
landed after it. Host figures must not enter the README's table
(CONTRIBUTING.md's rule), the per-token figure is a mean over tokens 50-400
rather than the instantaneous cost at token 400, and **the host is not a proxy
for the board** -- the two ratios once landed suspiciously close (1.47x against
1.50x) and it was never causal.

### What the firmware prints

Added after the five stages proved too coarse twice in one day: once failing to
separate a code-placement shift from a real regression, once failing to explain
why halving the head's memory traffic changed nothing.

**Always on, in every build** -- the `profile ms/token` line, the `attn`
sub-stage breakdown (`qkv`, `rope`, `core`, `proj`), head sub-stage timers, and
one single-core PSRAM bandwidth line at boot:

```
PSRAM read: 72 MB/s  (1.22 MB of head weights => 16.9 ms/token floor)
head detail: quant 0.04 | own-half 49.7 | wait-worker 0.0 | worker-half 49.7
```

Read the bandwidth line as a floor: `n` MB at `b` MB/s cannot cost less than
`1000n/b` ms, and whatever a stage spends above that is arithmetic no compression
will reach. It streams the head's own staged weights, cold, far larger than the
data cache. The figure moves with code placement (68 / 72 / 73 / 75 / 81 MB/s
across binaries), so read no bandwidth number on this board to better than ~5%.
In the head line `own + wait` should account for nearly all of `head`, and `wait`
near zero means the calling core is the slow half so the row split is fine; a
large `wait` would make moving `split` off `rows / 2` free throughput. It has
measured 0.0 every time. `quant` should be negligible (96 elements) and is 0.04.
Six `esp_timer_get_time` calls per token against stages measured in tens of
milliseconds, and they stay on deliberately.

**Behind `bandwidth-probe`, off by default** (`cargo run --release --features
bandwidth-probe`, or `scripts/run_device.sh --probe`): the *dual-core* bandwidth
probe, which separates a saturated bus from a per-core one, and the head's
cycles/MAC stall probe (`Head::probe_dot_cycles`). Both run once at boot and
neither executes during generation -- and they still cost 0.3 ms/token, see
[The instrument's own cost](#the-instruments-own-cost).

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
| **+ probe opt-in** | 3.9 | 26.3 | 6.2 | 7.6 | 62.2 | 106.2 | **106.3** | 9.41 |
| **Rust, --fast-mspi (pre-i16)** ‡ | 3.8 | 24.7 | 6.0 | 7.3 | 60.9 | 102.7 | **102.6** | 9.75 |
| **+ i16 activations** | 3.9 | 26.8 | 6.2 | 7.7 | 55.3 | 99.9 | **100.0** | 10.00 |
| **Rust, now** | 4.0 | 27.1 | 6.5 | 7.9 | 49.7 | 95.2 | **95.2** | 10.50 |
| **+ four rows per pass** ‡ | 3.9 | 27.0 | 6.1 | 7.4 | 52.8 | 97.2 | **97.1** | 10.30 |
| **Rust, --fast-mspi** ‡ | 3.9 | 25.6 | 6.3 | 7.7 | 48.2 | 91.7 | **91.8** | 10.89 |
| ratio vs C reference | 0.91x | 0.63x | 0.94x | 0.93x | 0.87x | 0.79x | **0.78x** | |
| absolute gap | -0.4 | -15.8 | -0.4 | -0.6 | -7.4 | -24.6 | **-26.8** | |
| **change this brought** | +0.1 | +0.3 | +0.3 | +0.2 | -5.6 | -4.7 | **-4.8** | |

‡ **Not the shipping configuration** — real, reproducible and bit-exact, but excluded from the ratios, the headline and the chart, because they describe what you get by cloning this repo.

- **+ MSPI bus 120 MHz** — Requires CONFIG_IDF_EXPERIMENTAL_FEATURES; Espressif documents 120 MHz as temperature-sensitive and not recommended across the industrial range, and this board's boya flash part refused it and fell back. Real, reproducible, bit-exact -- and not what you get by cloning this repo. It was a +3.3% opt-in AT THE TIME; the engine has moved twice since, and [run.rust-fast-mspi-dot2] is the current measurement of the same flag.
- **+ int8 head (reverted)** — A negative result, kept because it corrects a claim this file made. Staging the head as unpacked int8 instead of packed int4 -- the C reference's format, and one fewer operation per element -- made the head 62.3 -> 76.9 ms and the token 106.3 -> 121.5. Reverted. Bit-exact throughout: the digest still printed OK, so this is purely a cost.
- **+ probe, always on** — Superseded by [run.rust-findings-off]. Kept because it is the measurement that justified making the probe opt-in: identical arithmetic, 0.3 ms slower, purely from ~50 lines sitting in head.rs.
- **Rust, --fast-mspi (pre-i16)** — One flag from the default build -- scripts/run_device.sh --fast-mspi -- and not the default, for reasons about other boards rather than this one: CONFIG_IDF_EXPERIMENTAL_FEATURES, Espressif documenting 120 MHz as temperature-sensitive, and this board's boya flash part refusing it and falling back. Bit-exact: 102.6 ms/token, 9.75 tok/s, 18.9% faster than the C reference, digest 0x327578cb136fd6aa OK. SUPERSEDED as a description of the flag -- this was measured before i16 activations and two rows per pass, so it now reads slower than the shipping default. See [run.rust-fast-mspi-dot2] for the same flag on the current engine.
- **+ four rows per pass** — A negative result, and a predicted one. Two rows per pass took the head's activation loads to 0.5 per multiply-accumulate; four would take them to 0.25, and on an issue-bound loop that should be the whole win. It is not: four rows need four lane sets, and at DOT_LANES = 4 that is sixteen live accumulators on a machine with sixteen visible address registers. The allocator spilled. Head 49.3 -> 52.8 ms and the token 93.7 -> 97.1, measured in the probe build against the probe build; cycles/MAC went the wrong way, 8.13 -> 8.41, which is the spill showing up directly. Reverted. Bit-exact throughout -- the digest still printed OK -- so this is purely a cost. The arithmetic is kept in llm_core::dot4_int4_int16, tested, off the hot path.
- **Rust, --fast-mspi** — The opt-in 120 MHz MSPI clock, re-measured on the current engine -- the earlier [run.rust-fast-mspi] row was taken before i16 activations and two rows per pass, and had gone stale enough to read as a downgrade. 91.8 ms/token, 10.89 tok/s: 3.7% faster than the shipping default's 95.2 and 32.9% faster than the C reference (ref/new - 1, the convention every other row here uses -- as elapsed time it is 24.8% less, which is a different number and not the one this file quotes). The two effects compose, which is what you would expect of a memory-side change against two instruction-side ones -- the bus went 68 -> 75 MB/s on the probe and the head, which is compute-bound, moved only 49.7 -> 48.2 while attn (the stage that actually streams) took 27.1 -> 25.6. Still not the default: it needs CONFIG_IDF_EXPERIMENTAL_FEATURES, Espressif documents 120 MHz as temperature-sensitive outside the commercial range, and this board's boya flash part logs 'High performance mode of this flash model hasn't been supported' and stays at its own clock while PSRAM alone goes to 120. Bit-exact: digest 0x327578cb136fd6aa OK.

**`attn` broken down** — the sub-stages the five-stage profile hides. `qkv`/`proj` are the fp32 matvec that also drives `ffn` and `ple`; `core` is attention proper, the only part that grows with sequence position. The C reference has no equivalent instrumentation, so it is absent rather than zero.

| attn ms/token | qkv | rope | core | proj | sum | both cores |
|---|---|---|---|---|---|---|
| **+ attn instrumented** | 7.8 | 0.14 | 28.3 | 2.7 | 38.9 | `qkv`, `proj` |
| **+ attn heads split** | 7.9 | 0.15 | 15.6 | 2.7 | 26.3 | `qkv`, `proj`, `core` |
| **+ token digest** | 7.8 | 0.15 | 15.6 | 2.7 | 26.2 | `qkv`, `proj`, `core` |
| **+ MSPI bus 120 MHz** | 7.2 | 0.15 | 14.7 | 2.5 | 24.6 | `qkv`, `proj`, `core` |
| **+ int8 head (reverted)** | 7.9 | 0.15 | 16.0 | 2.7 | 26.8 | `qkv`, `proj`, `core` |
| **+ probe, always on** | 7.6 | 0.15 | 15.9 | 2.6 | 26.2 | `qkv`, `proj`, `core` |
| **+ probe opt-in** | 7.8 | 0.15 | 15.6 | 2.7 | 26.2 | `qkv`, `proj`, `core` |
| **Rust, --fast-mspi (pre-i16)** | 7.5 | 0.15 | 14.5 | 2.6 | 24.8 | `qkv`, `proj`, `core` |
| **+ i16 activations** | 7.9 | 0.15 | 16.0 | 2.7 | 26.8 | `qkv`, `proj`, `core` |
| **Rust, now** | 8.2 | 0.15 | 15.8 | 2.9 | 27.0 | `qkv`, `proj`, `core` |
| **+ four rows per pass** | 7.7 | 0.15 | 16.5 | 2.7 | 27.1 | `qkv`, `proj`, `core` |
| **Rust, --fast-mspi** | 8.0 | 0.15 | 14.8 | 2.8 | 25.8 | `qkv`, `proj`, `core` |

**Rust is 28.2% faster per token than the C reference it was ported from** — 95.2 ms against 122.0, 10.50 tok/s against 8.20, on the same board with byte-identical output. It beats C on 5 of the 5 stages.

<!-- END device-table -->

`stages` sums the profiled stages; `tok/s` comes from wall clock. On C, wall runs
2.2 ms/token above the stage sum (122.0 against 119.8) -- sampling, argmax and
serial output sit outside the profiled stages. On the shipping Rust build the two
agree exactly at 95.2, because argmax moved into the head. They agree on deltas
too: wall went 187.2 -> 181.6 across the nibble-LUT change, against -5.7 on the
stage sum.

This table and `png/benchmark-stages.svg` are generated from
[`benchmarks/device.toml`](./benchmarks/device.toml) by
`scripts/plot_stages.py --inject`. **Do not hand-edit between the markers.**
Before the generator the chart was a hand-drawn PNG with no source and the table
was hand-maintained in two files, so both aged silently every time someone
reflashed. Record a new run as its own `[run.<id>]` section rather than editing
an existing one.

**Provenance of the two anchor rows.** Measured the same evening on the same
board. C: 400 tokens in 48.78 s, reproduced bit-identically across three runs
(same total to 0.01 s, same per-stage figures to 0.1 ms). Rust: 400 tokens in
124.6 s, the 311.5 ms/token in the table. Neither is quoted from anywhere. (An
earlier version of this paragraph gave "73.07 s" for the first Rust row. That is
182.7 ms/token -- the nibble-LUT era, `Rust, nibble LUT` at 181.6, eight rows
later.) Upstream's own C figures are unused: different model file, and its two
READMEs disagree with each other (94.9 vs 102.9 ms/token). Re-measuring here
settled it at 119.8 at these settings and 103.1 at upstream's 200-token/4-token
defaults, which is where their figure came from.

**The two firmwares emit byte-identical output** for all 400 tokens, verified
since `+ token digest` rather than eyeballed: digest `0x327578cb136fd6aa` over
418 tokens, matching `models.toml`'s `gen_digest`. They no longer even stage the
head the same way -- C keeps unpacked int8, Rust packs to int4 (`head staged
int4: 1.32 MB (int8 staging would be 2.54 MB)`), which
`int4_head_equivalence.rs` asserts dequantizes to the same values. (A boot line
once quoted here, PSRAM free 3141 KB for C against 3211 KB for Rust, dates from
when Rust staged int8 too; int4 has since handed back 1.2 MB.)

**Where the time goes now.** All five stages are faster than C, so what is left
is a distribution, not a deficit. The head is 49.7 ms, 52% of a token; `attn` is
27.1 ms, 28%, of which `core` -- the only part that grows with sequence position
-- is 15.8. The biggest absolute wins against C are `attn` (-15.8) and the head
(-7.4).

## Reproducibility and the layout tax

**A given binary reproduces exactly on this board.** C was known to (three runs,
same total to 0.01 s, same per-stage figures to 0.1 ms); Rust binaries re-run
from a hard reset do the same, including `+ probe, always on`, which re-measured
at exactly 106.6 / head 63.2 to 0.0 ms. One run per binary is enough and
`--repeat`-style averaging buys nothing.

The consequence caught this project out once. With no run-to-run noise to absorb
it, **a difference between two binaries in a stage neither of them touches is
real.** Every `head` movement below comes from a binary that changes no line the
head executes:

| head | binary | what else changed |
|---|---|---|
| 90.3 -> 94.6 | `feat/ple-v2-header` | PLE\0 header parse, `out_vocab` |
| 94.6 -> 98.0 | `perf/nibble-lut` | two lookup tables, four reshaped loops |
| 98.0 -> 94.8 | `+ KV layout` | attention only -- this time in our favour |
| +3.8 | `+ 4 accumulators` baseline | the head's own instrumentation |
| +0.8 | `+ attn heads split` | attention only |
| -0.5 | `+ token digest` | a digest accumulator and a print |
| +0.3 | `+ probe, always on` | ~50 lines of probe in `head.rs` |

What changes is how much code and `.rodata` `llm-core` occupies, and the head
streams over a megabyte of PSRAM through a 64 KB cache every token. Budget for
it: an `llm-core` change that leaves the head alone can still cost it 3-4 ms, and
a change measured only on the stages it targets will overstate its net. The
instruments pay it too -- the same boot probe, identical buffer, identical code,
has read 68 / 72 / 73 / 75 / 81 MB/s. The first reading of 90.3 -> 94.6 -> 98.0
was thermal drift; it was not, and the repeat runs killed that. Recorded because
"the board was warming up" is the comfortable explanation and it was wrong.

**One unresolved provenance gap.** Before the nibble-LUT row the tree measured
179.6, not the wider-caches row's 174.2, with `attention.rs`/`ops.rs` zip loops
and the `matvec_override` hook landed in between. The hook's cost was once
guessed at ~2 ms/token and never isolated by a clean before/after -- a suspect,
not a measurement, and moot as one now that it has a shipping user at `+
dual-core matvec` worth 18.6 ms. The discrepancy is still unexplained; see
the open-questions section below.

## The investigation notebook

One section per lever, oldest first. Past tense is history -- what was believed,
and what the measurement said back. Present tense is only for what is true of the
shipping build. Everything is at 400 tokens / 18-token prompt, and every change
that shipped kept the output bit-identical.

### The first three: an affinity, a doubled load, a cache config

All shipped; head 193.8 -> 87.6 between them.

1. **The dual-core head split was not actually dual-core.** `head.rs` pinned its
   worker to CPU0, copying the C reference's `xTaskCreatePinnedToCore(..., 0)`.
   But arduino-esp32 runs `loop()` on CPU1 while ESP-IDF pins the main task to
   CPU0, so both halves of the head matvec queued on one core with a task switch
   between them. `CONFIG_ESP_MAIN_TASK_AFFINITY_CPU1` restores the C topology.
2. **`QT::matvec_range` did two loads per multiply-accumulate.** C's
   `matvec_q_range` unpacks both nibbles of a packed byte in one pass; the Rust
   port called a `code(r, j)` helper per column, re-reading the byte and
   branching on `j & 1`. Unrolled the same way, asserted bit-identical to a naive
   reference. With (1): head 193.8 -> 110.9.
3. **The external-memory caches were at ESP-IDF's defaults**, not
   arduino-esp32's: 32 KB data cache, 32-byte line, 16 KB instruction cache.
   Doubling all three (48 KB of internal SRAM) took the head 110.9 -> 87.6.

Item 3 is also the origin of a wrong belief that cost three days: moving one
stage 21%, more than any change had moved any stage, was read as proof the head
was bandwidth-bound. It was not.

### The nibble LUT: right for float, wrong for integers

**Shipped for the fp32 paths; the integer copy was later deleted.** `ple`
(1.79x), `input` (1.77x) and `ffn` (1.72x) were furthest from parity, all running
through `QT::matvec_range`. Two bit-exact changes -- a 16-entry nibble->f32 table
replacing a per-element subtract-and-convert, and `zip`-driven steady-state loops
replacing indexed ones -- moved them to 1.49x / 1.50x / 1.46x and `attn` 1.27x ->
1.18x, together -9.1 ms/token. (Historical ratios, from `Rust, nibble LUT`; those
stages now run 0.91x to 0.94x of C.) Net on the token was only -5.7 ms: the same
commit cost the head +3.4 ms of layout tax without touching `head.rs`.

The table was also copied onto a path it did not fit. `NIBBLE_I8` arrived
alongside `NIBBLE_F32`, where it genuinely wins by replacing an integer-to-float
*conversion* with a load (-24% on the host head). On the head's integer path it
replaced a *subtract* with a memory load, a worse instruction on any machine, and
deleting it was worth most of a 19.8 ms win a few runs later. The pattern was
copied without re-deriving whether it applied.

### The output head, in six episodes

94.6 ms and 1.66x of C at `Rust, before`; 49.7 ms and 0.87x now, still 52% of a
token and still the largest stage. The two episodes that shipped nothing taught
the most.

**1. int4 staging: 0.0 ms, kept anyway.** Acting on the bandwidth-bound belief,
the staged weights went from unpacked int8 to packed int4: 2.54 MB -> 1.32 MB
streamed per token, 1.2 MB of PSRAM returned, stage **0.0 ms**. That null result
is the most useful thing in this file -- it forced the stage to be instrumented.

**2. The instruments answered in one run.**

```
PSRAM read: 72 MB/s  (1.22 MB of head weights => 16.9 ms/token floor)
head detail: quant 0.04 | own-half 90.9 | wait-worker 0.0 | worker-half 90.9
```

`wait-worker 0.0` with both halves equal means each core independently completed
its rows in 90.9 ms while the other did the same: aggregate 1.22 MB in 91 ms,
**13.4 MB/s against 72 MB/s available**, 19% of its own bandwidth. Had memory
been the constraint, two cores could not both have hit 90.9. It was compute-bound
the whole time, and what was wrong was two operations per element in a loop that
runs 2.43M times per token. **One accumulator**: `acc += a[i] * b[i]` makes every
add wait for the previous one; four independent accumulators took it 91.0 ->
81.3, bit-exact because `i32` addition is associative. (Baseline 91.0, not the
table's 94.8 -- the instrumentation itself moved the head 3.8 ms on placement.)
**A subtract and a table load that should not have existed**: codes are stored
biased, `sum (c-8)a == sum ca - 8 sum a`, and `sum a` is identical for all 25,353
rows, so hoisting it deletes the subtract and with it the `NIBBLE_I8` lookup
standing in for the subtract. 81.3 -> 61.5, also exact. Neither was visible at
five-stage resolution. **The instrument was worth more than any guess about the
stage.**

**3. Reading both compilers' output: -6.9 ms from one instruction.**
`scripts/disasm_head.sh` and `scripts/disasm_c.sh` compile both sides for the
same target and dump the hot functions -- the only way to answer "are the
operations the same". First look: an `l8ui` + `sext` pair on every activation
read, because Xtensa has no signed byte load. Widening the activation to `i16` so
`l16si` folds the extension into the load took the head **62.2 -> 55.3 ms**, the
run at which it first beat C.

| the head's inner loop | C (GCC `-O3`) | Rust (LLVM) |
|---|---|---|
| accumulators | **1** -- serial dependency chain | **4** independent |
| stored-bias removal | `addi -8` **per element** | hoisted, one correction per row |
| activation load | `l8ui` + `sext` | `l16si` |
| instructions per MAC | ~8 | ~5 |
| MAC16 (`mula.*`) | **none** | **none** |

Three hypotheses died here. **MAC16 is not the answer**: neither compiler emits
it (814 `mull` against 19 `mula.*`), so it cannot explain any gap between them.
**The head was not behind C on arithmetic**: that came from treating C's 26.7 ms
of memory traffic as additive with its compute, which would put C at 6.0
cycles/MAC on an eight-instruction serial loop, and nothing in-order does that --
C's single accumulator stalls, and the stalls absorb its memory latency. Third,
already written into this file as a conclusion: 41.9 ms of non-memory time for
~6.1M instructions is ~1.65 cycles per instruction on a core that issues one,
therefore the loop must be stalling, therefore **the lever is software
pipelining**. It was testable, so it was tested rather than acted on.

**4. Two rows per pass: -5.6 ms.** `Head::probe_dot_cycles` reruns `rows_range`
over a row range small enough to stay in the 64 KB data cache and compares it
against one streaming pass. Matching rates mean issue-bound; a much faster
cache-resident rate means waiting for memory. It read **8.96 cycles/MAC
cache-resident against 9.64 streaming**, 8% apart -- issue-bound, so the four
accumulator chains already covered load-use latency and pipelining would have
bought nothing. Fewer instructions, then: the activation vector is identical for
all 25,353 rows and was reloaded for each, 2.43M `l16si` per token to read 96
distinct values. `dot2_int4_int16` loads each once and multiplies it into two
rows -- activation loads 1.0 -> 0.5 per MAC, head **55.3 -> 49.7 ms**, token
100.0 -> 95.2, cycles/MAC 8.96 -> 8.13 resident and 9.01 streaming, close to the
~11% the instruction count predicted. Bit-exact (same terms, order, lanes,
hoisted bias), asserted on every consecutive row pair by
`llm-host/tests/dot2_equivalence.rs`.

**5. Four rows per pass: reverted.** It takes activation loads to 0.25/MAC and on
an issue-bound loop should have been the whole win. **It was not.** Four rows
need four lane sets, and at `DOT_LANES = 4` that is sixteen live accumulators on
a machine with sixteen visible address registers: the allocator spilled. Head
**49.3 -> 52.8 ms**, token 93.7 -> 97.1, probe-build against probe-build,
cycles/MAC the wrong way **8.13 -> 8.41** -- the spill showing up directly.
Bit-exact, so purely a cost. `llm_core::dot4_int4_int16` is kept, tested and off
the hot path, because deleting it would delete the evidence of where the wall is.

**6. int8 staging: reverted.** Tried because the head was then the only stage
above C and pays a mask-or-shift per element that C does not. It lost: the extra
1.21 MB/token costs 13.3 ms at the measured 91 MB/s and the switch cost 14.6, so
the unpack is worth at most 1.3 ms across 2.43M elements. Head **62.3 -> 76.9
ms**, token 106.3 -> 121.5, digest still OK, purely a cost. It corrects a claim
this file made -- that the head uses only ~19% of available bandwidth, is
therefore compute-bound, and packing therefore bought a resource it was not short
of. **Utilization is not insensitivity**: a stage can be unable to saturate the
bus and still pay full price per byte. It also explains episode 1. int4 landed
when the head was 94.8 ms with a single-accumulator dot; that chain stalled the
loop and the stalls absorbed the memory latency. Four accumulators did not only
speed the arithmetic, they *exposed* the memory -- which is why a byte reduction
worth nothing then was worth 13 ms later.

### Skipping head rows without reading them: measured, and it cannot work

**Not attempted, and will not be.** The firmware computes all 25,353 logits and
uses one number, `tok = head.last_argmax()`, so a row that cannot beat the
running maximum need never be read -- and since each row's scale lives in a
separate 101 KB array, the test costs no PSRAM read of the row's 48 bytes. Two
exact bounds, measured on the host over real decode steps:

| bound | rows skipped |
|---|---|
| max-abs: `\|dot\| <= 8 * sum\|a_j\|` | **0 of 25,353** |
| Cauchy-Schwarz: `\|dot\| <= \|\|c-8\|\|₂ * \|\|a\|\|₂`, row norm precomputed | **0 of 25,353** |

Not "few" -- none, on every step tried, for a dimensional reason no tuning
touches. Both bounds are worst-case, assuming every element aligns in sign with
the activation, while a real dot over `d = 96` mixed-sign elements lands around
`sqrt(d)` below that: any such bound is ~10x loose, and the row scales span only
**12.1x** (2.011e-2 to 2.432e-1). Recorded so nobody re-derives it. An
*approximate* pruner would work and would change the token stream -- the same
trade the fp16 KV cache offers at a better rate.

### Dual-core everything else, and argmax

**Shipped: -18.6 ms on the matvecs, -12.7 ms on attention `core`, -2.7 ms on
argmax.** The head had been the only stage using both LX7 cores. Splitting each
matvec by rows is bit-exact -- `y[r]` depends only on row `r` and `x` -- and
`llm_forward_with_matvec_override`, shipped since Phase 3 with no user, was
already the right hook. Sync measured 0.5 ms/token across 43 matvecs (~11.6 us
per round trip) against ~18.6 ms saved; `head` moved +0.1 ms, so the control held.

Attention was instrumented next, identifying `core` at 28.3 ms, 24% of a token,
as the last stage on one LX7. Two heads per core took `core` **28.3 -> 15.6** and
the token 118.5 -> 106.8, sync 0.1 ms/token across 6 calls. A perfect halving
would have been 14.15 + 0.1 = 14.25, so **1.35 ms, 9.5%, did not come back** --
either bus contention, since unlike the head both cores now stream the same KV
cache at once, or the layout tax, since that binary's boot probe read 68 MB/s
against the previous binary's 72 while `head` moved +0.8 ms untouched.

Splitting the out-of-stage remainder then showed `emit 0.03`, `other 0.00`,
`argmax 2.74` ms/token. The fix was not a faster loop: the greedy pick was
reading 25,353 logits back out of PSRAM, 101 KB at roughly half the measured bus
rate, because a data-dependent branch per element leaves nothing to pipeline.
Both cores already hold each logit in a register the moment `rows_range` computes
it, so each now tracks the maximum over its own rows and the decode loop chooses
between two candidates. Tie-breaking is identical -- first-strictly-greater
within a half, `hi > lo` between halves, which yields the lower index on a tie
exactly as a `0..n` scan does -- and output stayed byte-identical, the only
available gate, since this path is firmware-only and no host test reaches it.
Afterwards `argmax 0.00 | emit 0.03 | other 0.00`, and wall clock equals the
stage sum. C still pays that cost: its `pick()` scans all 25,353 logits, which is
why its wall is 122.0 against 119.8 profiled. (When this was written that 2.2 ms
was most of the margin between the engines; it is now 8% of a 26.8 ms margin.)

### Is attention's `core` bandwidth-bound?

**Settled: yes, and the bus does not scale.** The dual-core probe ran.

| | 1 core | 2 cores, disjoint halves | scaling |
|---|---|---|---|
| PSRAM @ 80 MHz | 72 MB/s | 91 MB/s | **1.27x** |
| PSRAM @ 120 MHz | 81 MB/s | 131 MB/s | **1.62x** |

A second LX7 buys 27% more bandwidth, not 100%, and both cores finished within
0.1 ms of each other, so this is the bus and not imbalance: part of attention's
missing 1.35 ms is contention, and the fix is a smaller working set. Against that
ceiling, measured on `+ probe, always on`:

| | streams/token | stage (then) | rate | of available |
|---|---|---|---|---|
| output head | 1.22 MB | 63.2 ms | 19.3 MB/s | **21%** |
| attention `core` | 1.01 MB | 15.9 ms | 63.3 MB/s | **70%** |

Both stage figures are historical -- the head has since gone to 49.7 ms and draws
proportionally more of the same bus -- but the shape holds: the head cannot
saturate the bus and attention nearly can. Attention reads the whole KV cache
every token, `219 positions x 96 floats x 4 bytes x 2 x 6 layers` at these
settings, almost as much as the entire output head. That is the exact inverse of
what this document assumed about the two stages a day earlier.

### The instrument's own cost

**Made opt-in.** The dual-core probe and the stall probe run once at boot and
execute nothing during generation, and they still cost 0.3 ms/token -- 106.6 with
them against 106.3 without, across three runs, purely from ~50 more lines in
`head.rs`. The same layout tax as everywhere else, charged by the tool that
discovered it. Behind `bandwidth-probe` now; compiling it out recovered the
0.3 ms exactly.

### The MSPI clock: an opt-in, not a default

**Shipped as `--fast-mspi`, off by default.** PSRAM 80 -> 120 MHz is config-only
and bit-exact: **91.8 ms/token, 10.89 tok/s** on the current engine, 3.7% faster
than the shipping default's 95.2 and **32.9% faster than C**, digest
`0x327578cb136fd6aa` OK. (Percentages here are `ref/new - 1`, the convention the
whole file uses. As elapsed time it is 24.8% less -- a different number, and not
the one quoted anywhere in this repo.) It
composes with the instruction-side wins, as a memory-side change should -- the
bus went 68 -> 75 MB/s on the probe, the head (compute-bound) moved only 49.7 ->
48.2, and `attn`, the stage that actually streams, 27.1 -> 25.6.

**Why it is not the default.** It needs `CONFIG_IDF_EXPERIMENTAL_FEATURES`,
Espressif documents 120 MHz as temperature-sensitive and not recommended across
the industrial range, and this board's boya flash part logs `High performance
mode of this flash model hasn't been supported` and stays at its own clock while
PSRAM alone goes to 120. That corrects an earlier claim here that raising PSRAM
"forces flash along with it and the build fails outright": it does not, the build
succeeds and the overlay ships. Reasonable for a benchmark board on a desk,
unreasonable for a product.

**Two things guessed wrong, both cheaply.** The win was predicted at ~7 ms by
modelling each stage as `memory + compute` and halving the memory term; the head
moved **0.3%** while its memory floor dropped 13.4 -> 9.3 ms, which is what
compute-bound actually means. And the per-stage shape (`input -7.7%`, `qkv
-7.7%`, `ffn -8.1%`, `ple -7.9%`, `proj -7.4%`, `core -5.8%`, `head -0.3%`)
looked like a *flash* win, since everything that moved reads weights through the
flash mmap and the head does not -- but raising flash 40 -> 80 MHz with PSRAM
left at 80 measured **106.6 ms, nothing at all**. The remaining explanation,
inferred rather than measured: the shared MSPI *core* clock going 160 -> 240 MHz
cuts per-miss latency for flash and PSRAM alike.

### SIMD: scaffolding shipped, case weakened

**Not attempted.** This file long said the remaining gap was GCC's Xtensa backend
against LLVM's, and that the lever was the S3's 128-bit PIE vector unit. Both
halves read differently now: there is no remaining gap, and every win that closed
it was instruction-level and scalar -- an unroll, four accumulators, an algebraic
hoist, one wider load, one shared load across two rows.

The scaffolding is worth recording. `llm_forward_with_matvec_override`
(`llm-core/src/model.rs`) replaces every non-head matvec with a platform-supplied
closure, keeping `llm-core` platform-agnostic, and now
has a shipping user in the dual-core matvec split. `QT::matvec_range_chunked`
plus `llm-host/tests/matvec_simd_tolerance.rs` answer the hard part, *how much
numeric drift is acceptable*: `matvec_range` accumulates in `f32` and any real
SIMD dot product reorders that sum, unlike the head's integer accumulation where
reordering is provably lossless. Measured, not guessed -- SIMD-shaped reorderings
(4-lane, 8-lane, whole-row-dequant) move cross-entropy by ~3-4e-8 on the real
validation set, roughly 45,000x inside the noise floor the int8 path already
tolerates. What remains is ESP-IDF-side: vendor `esp-dsp` (an IDF Component
Registry component, not a crates.io crate), confirm the real `dsps_dotprod_f32`
signature against the vendored header rather than assuming it, unpack each row's
nibbles into a contiguous buffer, and benchmark. An FFI call per row has real
overhead at row lengths of 66-128 elements, so it is not a guaranteed win. (An
earlier version of this section named `QT::matvec_int8_activations` as the
target. The firmware never calls it: like the C sketch it does not enable
`int8-activations`, so every tensor except the head runs `matvec_range`.)

## What is left to optimize next

Four items, ranked by how much is known about each. None is a gap to close: the
port is 28.2% faster per token than the reference it was translated from, on all
five stages.

**1. fp16 KV cache -- measured, and undecided on purpose.** It halves attention's
1.01 MB/token, the one stage the bus measurement puts near its ceiling. Cost
measured before building it: validation cross-entropy +1.1e-5, one hundred and
eighteenth of the int8-activation noise floor this project already ships with --
but the token stream forks at generated token 136 under the firmware's head, one
flipped argmax and then a different continuation with no re-convergence
(`llm-host/tests/fp16_kv_tolerance.rs`). It would retire "byte-identical output"
as a claim, which is doing more work than a millisecond would. Needs a decision,
not a board.

**2. `l32i` weight loads.** Two rows per pass halved the *activation* loads; the
weight side still reads one packed byte at a time. Reading four packed bytes as
one 32-bit load is the same shape of win on the other operand, and the loop is
known issue-bound, so fewer instructions is known to be the right lever. It needs
`unsafe`: a 32-bit load requires the row pointer to be 4-byte aligned, and the
staged `w4` rows are byte-addressed with nothing in the type system asserting it.
Unmeasured -- and the four-row result is a standing reminder that "fewer loads"
is a prediction, not a measurement.

**3. `DOT_LANES = 2` with four rows.** Four rows at `DOT_LANES = 4` spilled the
register file; two lanes with four rows is eight live accumulators for the same
0.25 activation loads per MAC. `dot4_int4_int16` is kept, tested and off the hot
path precisely so this stays cheap to try. Unmeasured.

**4. Port `bandwidth_bench` (Phase 5).** `reference-c/firmware/bandwidth_bench/`
is still unported. The minimum useful part -- one cold sequential pass over the
head's own staged weights -- was written by hand, ships at boot, and settled both
questions it was wanted for. A full port would give per-pattern numbers (stride,
working-set size, read vs write) that nothing currently needs.

**Not on this list: SIMD**, ranked first here for months on the strength of a
compiler-backend hypothesis the disassembly then contradicted.

Items 2, 3 and 4 need real ESP-IDF hardware -- good first issues for a
contributor who has a board, see [CONTRIBUTING.md](./CONTRIBUTING.md). Item 1
needs a judgement call about what this project is for.
