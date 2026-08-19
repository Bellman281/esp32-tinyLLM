# Results

All numbers below are under one consistent three-tier accounting (head excluded
from core — see `param_budget()` in `src/model.py`). Earlier mixed-accounting runs are archived in
`runs/_archive_old_accounting/` and should not be cited.

**Status (2026-07-21): validated end to end.** PLE beats baseline (2 seeds), the
gain survives 4-bit PTQ (2 seeds), on-chip bandwidth confirms the flash table is
nearly free, and the complete 28.9M-parameter model now generates coherent text
on an ESP32-S3 N16R8. The C runtime generates text at **~9.5 tok/s end to end**
(what a viewer sees; includes serial output), equivalent to 102.9ms/model step or
9.72 tok/s of pure compute, using an int8-staged output head with int8 activations
(host-validated, val perplexity delta ~0). Lead public numbers with the end-to-end
figure.

## Headline (the deployable config)

Vocab 32768, core-matched across arms at ~559K params (273KB at 4-bit, small
enough for the ESP32-S3's 512KB internal SRAM; the design constraint that sized
it. The polished firmware in fact leaves the core flash-mapped XIP, which
measured fast enough — see the on-chip section), 25M-param PLE table (12MB
flash), d_model=96, 6 layers, ple_dim=128. Two seeds.

| arm | core | total | ppl | vs baseline |
|---|---:|---:|---:|---:|
| `baseline` | 559K | 3.7M | 12.58 | — |
| **`ple`** | 558K | **28.9M** | **11.41** | **+0.098 nats / 9.3% ppl** |
| `fatembed` | 559K | 28.9M | 11.94 | +0.052 nats |

- **PLE beats a same-core, SRAM-fitting baseline by 0.098 nats (2 seeds, ±0.006).
  ~16x the seed noise.** ppl 12.58 -> 11.41.
- **Per-layer injection beats bottom injection by 0.046 nats** (`ple` vs
  `fatembed`, both 2 seeds). At realistic vocab the *where* of injection is worth
  roughly 2x the params-at-the-bottom approach.
- As comparison context (prior independent work, no relation to this codebase):
  DaveBben's esp32-llm ran 260K params on an ESP32-S3. This is ~110x the stored
  parameter count, on a tighter fast-memory budget, with the extra params living
  in flash as a per-token-sparse table.

**What "28.9M params" does and does not mean.** It is 28.9M *stored* parameters:
a 559K dense core (SRAM), a 3.1M output head (streamed sequentially), and a 25M
lookup table (flash, one row per token). It is still a TinyStories-domain model —
better coherence and consistency, not new capability. Quote it as "parameters
resident via a memory-hierarchy split," never as a capability multiple.

## Why vocab matters (and why the small-vocab number is modest)

The same architecture at vocab 4096, core-matched at ~1.5M, 2 seeds:

| arm | ppl | vs baseline |
|---|---:|---:|
| `baseline` | 8.21 | — |
| `ple_notable` | 8.35 | **-0.017 (worse)** |
| `fatembed` | 8.26 | -0.006 (~nothing) |
| `ple` | 8.00 | +0.025 |
| `bigcore` (2x core) | 6.93 | +0.170 |

At vocab 4096 PLE's edge is only +0.025 nats. At vocab 32768 it is +0.098 — **4x
larger.** This is the memory-tiering thesis working: a large vocabulary makes the
table both huge and cheap (more rows, each a sparse per-token lookup), and that
is precisely the regime PLE was designed for. The deployable config lives in the
favourable regime; the small-vocab ablation is a control, not the product.

## What the controls establish

- **The table does the work, not the plumbing.** `ple_notable` (all of PLE's
  per-layer adapters and projection, no lookup table) is *worse* than baseline at
  vocab 4096 (-0.017): the plumbing spends core params on machinery that returns
  nothing unless a table feeds it. The isolated table contribution
  (`ple` - `ple_notable`) is +0.043. So the flash-resident table is the entire
  source of the gain, which is exactly what the ESP32 premise needs.
- **Row-width saturates; vocab-rows do not.** The corrected sweep (`runs/*fix-d*`,
  fixed FFN, vocab 4096) shows the isolated table benefit peaking around a 6M
  table when scaling `ple_dim` (row width): +0.045 (d64) -> +0.094 (d256) ->
  +0.087 (d512). But the deploy config scales the table a different way — more
  *rows* via a 32k vocab — and a 25M table there still clearly pays off (+0.098).
  Widening rows plateaus; adding rows does not, over the range tested.
- **PLE is not free capacity.** `bigcore` (2x the dense core) gets +0.170 at vocab
  4096. PLE recovers ~15% of that at vocab 4096, more at vocab 32768. On a desktop
  you would just buy the core; on an ESP32 the core is fixed silicon and flash is
  abundant, which is the whole point.

## Hardware: bandwidth measured on the N16R8 (2026-07-21)

Measured on the real ESP32-S3 (`firmware/benchmarks/bandwidth`, cycle-accurate timing
via the Xtensa cycle counter). This is what turns the estimated tok/s into a real
one, and it is the number the whole approach rested on.

| measurement | value |
|---|---|
| PSRAM sequential read | 60.7 MB/s |
| internal SRAM sequential read | 240 MB/s |
| flash random-read, 512B row | 20.3 us |
| **per-token TABLE cost** (6 random rows) | **~0.12 ms** |
| **per-token HEAD cost** (1.5MB PSRAM scan) | **~17.3 ms** |
| **bandwidth-only tok/s ceiling** | **~58 tok/s** |

**The core bet is confirmed on silicon.** In the isolated bandwidth benchmark,
the 25M-param flash table costs ~0.7% of the per-token memory time — nearly free,
exactly as designed. The output HEAD (which the baseline pays too) dominates the
memory traffic. This is not yet a measured end-to-end PLE-vs-baseline speed
comparison, so quote the 0.7% specifically as the table's synthetic bandwidth
share rather than total inference overhead.

Caveats: this is a bandwidth-only ceiling, not observed inference throughput.
The first complete scalar port showed that unpacking int4 and doing millions of
scalar float operations dominate well before raw bandwidth does. The 20us
random-read latency is real, not zero — it stays negligible at this ple_dim/layer
count but would grow with a wider table.

## On-chip generation: complete model running (2026-07-21)

The exported model is 14,912,332 bytes and fits in a custom 15,597,568-byte
flash partition with 685,236 bytes spare. The 619KB application occupies a
separate 1MB partition. The portable C runtime matches the exported PyTorch
golden across all 32,768 logits (`max abs diff = 0.00001`) before the same code
is compiled for the device.

On the ESP32-S3, the 1.64MB tied embedding/output head is copied to PSRAM at
boot, the 25M-parameter PLE table stays memory-mapped in flash, and scratch plus
KV cache live in PSRAM. After all allocations, 5,228KB of PSRAM remains free.
Greedy generation from `Once upon a time` produces coherent TinyStories text.
One captured on-device continuation begins:

> Once upon a time, there was a little girl named Lily. She loved to play
> outside in the sunshine. One day, she saw a big tree with a hole in it. She
> was curious and wanted to see what was inside.

| implementation | 200-token result | model-step time |
|---|---:|---:|
| first correct portable port | 0.57 tok/s end to end | 1,757.2 ms |
| PSRAM head + scalar cleanup | 4.61-4.77 tok/s end to end | 193.9 ms |
| exact dot/RoPE/attention cleanup | — | 172.9 ms |
| dual-core exact head | 5.67-6.22 tok/s end to end | 139.4 ms |
| **int8-staged head + int8 activations** | **~9.5 tok/s end to end** | **102.9 ms** |

The int8 head is the current runtime: **9.72 tok/s compute-only** (102.9 ms/step),
a further 1.35x over the exact fp32 dual-core head. The output head is staged as
int8 in PSRAM once at boot (int4 nibbles unpacked once), and activations are
quantized to int8 per token, so each output row is a plain int8xint8 -> int32 dot
with no per-token unpacking. **The int8-activation change was validated on host
val perplexity (delta ~0, see runtime/host_verify/ppl.c) before shipping**, and
on-chip text stays coherent. The scalar fp32 head (139.4ms) remains the exact
baseline; the fp32 host golden still matches PyTorch to 1e-5.

Profile after the int8 head (ms/token, dual-core wall): head 57.6 | attn 25.6 |
ple 8.5 | ffn 6.9 | input 4.4. The head is now **PSRAM-bandwidth-bound, not
compute-bound**: it reads 2.43MB of int8 weights per token, which at 60.7 MB/s is
a ~40ms floor, so only ~17ms is compute. Literal S3 vector-SIMD would cut that
17ms but not the 40ms bandwidth floor (bounded ~15% further gain). The bigger
levers from here are reducing bytes-read (int4 head + SIMD unpack) or a
smaller/factorised output head (a model change) - not vectorising harder.

The earlier fp32 exact optimizations (139.4ms baseline) were: staging the head
in PSRAM, converting each fp16 group scale once, unpacking both int4 values per
byte, applying a group scale after its dot product, compiling at `-O3`, skipping
7,415 unreachable padded vocabulary rows, computing RoPE values once per token,
caching attention scores, and splitting independent output rows across both
LX7 cores.

Historical profile of that earlier exact fp32-head path (139.4ms/step, superseded
by the int8 head; the current runtime's profile is the 57.6 | 25.6 | 8.5 | 6.9 |
4.4 breakdown above), averaged over 200 generated tokens (wall-time share; the
head runs on both cores, so its *compute* share is higher — ~80% — than its
wall share):

| stage | ms/token (wall) | wall share |
|---|---:|---:|
| output head (dual-core) | 93.2 | 66.9% |
| attention | 26.4 | 18.9% |
| PLE input + per-layer path | 12.9 | 9.3% |
| FFN | 6.9 | 4.9% |

An earlier core-staging configuration - the 0.29MB quantized core in PSRAM and
norms in internal RAM - improved latency by 2.0ms/token (1.4%).

### Current runtime

The current configuration enables staging, layer parallelism and SRAM hot-set
placement together:

1. every per-position tensor staged to int8 in PSRAM,
2. a dual-core `layer_matvec` hook, applied to tensors of >=128 rows
   (`ple_model_proj` 768, `qkv` 288, `ple_gate` 128),
3. the hot working set - all scratch buffers except the 99 KiB logits array,
   plus the norm vectors - in internal SRAM.

| step | published | current |
|---|---:|---:|
| output head | 57.6 | 59.4 |
| attention | 25.6 | 20.5 |
| PLE path | 8.5 | 6.4 |
| FFN | 6.9 | 6.5 |
| input | 4.4 | 2.2 |
| **total ms/token** | **102.9** | **94.9** |

The combined configuration reduces compute latency from 102.9 to 94.9ms/token
(9.88 tok/s attached). Individual contributions were not isolated.

Managed hot set: 29,320B internal (21,128 dynamic + 8,192 static), 294KB
internal free, 4.22MB PSRAM.

`matvec_i8_range` must stay out of line: inlining it regressed inference from
95.0 to 155.2ms/token on Arduino ESP32 3.3.10 at -O3.

Global `-O3` is set by `scripts/deploy.sh` and is the entire optimization
configuration; the runtime carries no per-function optimization attributes.

There is still bounded exact work—parallel attention, precomputed RoPE
frequencies, and a one-group-specialized head loop—but the profile caps the
entire attention opportunity at 26.4ms and the other items at low single-digit
milliseconds. They are intentionally deferred rather than presented as another
large scalar speedup.

The measured throughput is much lower than the 58 tok/s bandwidth ceiling and
disproves the earlier 20-40 tok/s compute estimate for the naive scalar kernel.
It is nevertheless in the same practical speed range as the prior independent
260K-parameter ESP32 project (comparison context only) while holding roughly
110x as many stored parameters. With int8 activations now shipped and the head
PSRAM-bandwidth-bound, the next large speed step is reducing bytes read (an
int4-in-PSRAM head with SIMD unpack) or a factorized/smaller output head — not
more scalar cleanup and not a question about PLE table bandwidth.

## 4-bit quantization: the gain survives (2026-07-21)

Group-wise symmetric int4 PTQ (group 64), the GGUF-Q4-style format you would flash
(`research/tinystories/quantize_eval.py`). Every large weight quantized, including the 25M table. Two
seeds, vocab 32768 deploy models.

| arm | fp32 -> 4-bit degradation | 
|---|---|
| `baseline` | +0.079 / +0.088 nats |
| `ple` | +0.055 / +0.061 nats |
| `fatembed` | +0.046 / +0.050 nats |

PLE vs baseline edge: fp32 +0.101/+0.095 -> 4-bit +0.125/+0.121, i.e. **fully
retained (124-128%)**. Read carefully: all arms degrade under 4-bit (~ppl +1);
PLE degrades *less*, because a large redundant lookup table with per-group scales
is inherently more quantization-robust than a small dense model where every weight
is critical. So the part we bet the flash budget on is also the most 4-bit-robust.
**No QAT needed for the headline.**

The final flash artifact uses a tighter group-128 format with ragged rows and
fp16 scales to fit 16MB flash. The exact storage scheme was also validated on
both seeds:

| arm | fp32 -> shipping-format degradation |
|---|---|
| `baseline` | +0.089 / +0.109 nats |
| `ple` | +0.063 / +0.089 nats |
| `fatembed` | +0.056 / +0.061 nats |

PLE's edge is +0.101/+0.095 nats in fp32 and +0.127/+0.115 in the shipping
format: **126%/121% retained**. Thus the exact bytes flashed to the board have
the same two-seed conclusion as the group-64 headline PTQ check.

## Remaining limitations

- **The ESP32-S3's SIMD instructions remain unused.** The shipping runtime
  (~9.5 tok/s end to end) quantizes activations to int8, host-validated, but its
  dot products are still scalar. The head is PSRAM-bandwidth-bound (a ~40ms
  read floor inside its 57.6ms), so SIMD alone buys a bounded ~15%; the real
  levers are reducing bytes read (int4 head + SIMD unpack) or a smaller head.
  The 58 tok/s number remains only a bandwidth ceiling.
- **Domain is TinyStories.** World knowledge, arithmetic, and multi-step reasoning
  remain absent; that ceiling is set by the dense core, not moved by the table.
- **Provenance.** This is independent work. Its actual dependencies are the
  TinyStories dataset (Eldan & Li, Microsoft Research, arXiv:2305.07759) and
  Google's published Gemma Per-Layer Embeddings design (reproduced from the
  `transformers` Gemma implementation and Google's documentation). No code,
  model, checkpoint, or method derives from llama2.c (Karpathy) or from
  DaveBben's esp32-llm; both are prior independent work in the tiny-LM /
  microcontroller space and appear in this document only as comparison context.
  The novel claim here is narrowly: applying Gemma-style Per-Layer Embeddings
  to a microcontroller SRAM/flash hierarchy so a larger stored model fits than
  fast memory allows.

## Next

1. Add interactive serial prompting/tokenization, then dialogue fine-tuning for
   the simple-conversation milestone.
2. Record the on-chip text-generation demo and publish the measured result.
3. Treat an ESP32-S3 SIMD/int8 head as a separate experiment and compare its
   quality and speed against the exact 139.4ms/token baseline.
