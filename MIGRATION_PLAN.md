# ESP32 PLE-TinyLM — C/C++ → Rust migration plan

Scope: **only** the on-device inference engine and the ESP32 platform code.
Python (training, quantization, export, the `chat.py` tokenizer/CLI) is
untouched and out of scope, per instruction. `model.bin`, `vocab.h`, and
`golden.txt` remain Python-produced artifacts, consumed as-is — nothing
here regenerates them.

## Folder layout

```
esp32-Rust/
  MIGRATION_PLAN.md      <- this file
  reference-c/             <- frozen, read-only copy of the working C/C++ engine
                               (ground truth + regression oracle; see its own README.md)
  data/                     <- validation token data for the ppl parity test
                               (val_v32768.bin), added in Phase 2
  engine/                  <- the new Rust workspace; all porting work happens here
    Cargo.toml               (workspace manifest)
    llm-core/                <- no_std, platform-agnostic model math (Phase 1 — DONE)
    llm-host/                <- std host CLI + correctness/perplexity tests (Phase 2 — DONE)
    llm-firmware/            <- the chip binary (Phase 3, then Phase 6) — not
                                 scaffolded yet, see llm-firmware/README.md
    llm-display/             <- optional OLED/TFT demo screen (Phase 4) — not
                                 scaffolded yet, see llm-display/README.md
```

`llm-core` is deliberately its own crate with zero ESP32/FreeRTOS/esp-hal
dependency. `llm-host` and `llm-firmware` both depend on it and add only
platform glue. This is what lets Phase 6 (dropping FreeRTOS) touch only
`llm-firmware` later, without re-touching or re-verifying the model math.

## Ground rules

- `reference-c/` is read-only for the duration of the port. Every phase's
  Rust output is checked against it, never the other way around. If it ever
  needs to change, that means the upstream Python project changed (retrain,
  new vocab, new export format) — refresh the copy, don't hand-edit it.
- `model.bin`, `vocab.h`, `golden.txt` are Python-produced. Not regenerated
  or modified by this plan.
- The chip never encodes text today — `esp32_llm.ino` decodes a fixed
  `PROMPT_IDS` array baked in at build time via `vocab.h`. So this plan does
  not port a tokenizer/encoder; that stays exactly where it is (Python,
  host-side, `chat.py`), untouched.
- Every phase ends at the same pass/fail line the C code already uses:
  `llm-host`'s port of `verify.c` must match `golden.txt` to the same
  tolerance (max abs diff < 0.02) before the next phase starts. (Phase 2
  refined this: that tolerance is an fp32-activations-only guarantee — see
  "Parity test suite" below for why, and how int8-activations is checked
  instead.)
- Decided target stack (see chat): **phased**. Phase 3 lands on
  `esp-idf-hal` (std, FreeRTOS underneath) for the fastest path to a
  verified working Rust binary. Phase 6, later and separate, re-platforms
  onto `esp-hal` (no_std, no FreeRTOS, no ESP-IDF C SDK) once the algorithm
  port is already proven correct. Two different problems, solved one at a
  time instead of together.

## Phases

### Phase 0 — Workspace scaffold — DONE
- `reference-c/` populated and frozen.
- `engine/` workspace created.

### Phase 1 — `llm-core`: port the portable math — DONE
Ported, in order, from `reference-c/firmware/common/llm.h`: the binary
header/tensor-cursor parser (`Cursor`, `bind_q`/`bind_f` equivalents),
fp16 scale decode (`half` crate), int4 group-wise dequantize (`deq_row`),
both matvec kernels (`matvec` for the fp32-dequantized-weight path,
`matvec_int8_activations` for the int8-activation SIMD-friendly path,
selected by the `int8-activations` Cargo feature — same role as the C
`#ifdef LLM_INT8_ACT` build switch), `rmsnorm`/`gelu`/`silu`, split-half
RoPE, causal self-attention with the persistent KV cache, SwiGLU FFN, the
PLE per-layer gate, and `llm_forward()` chaining all of it. `#![no_std]`,
zero allocator dependency (`core` + `libm` + `half` only) — compiles
unchanged for `llm-host` today and `llm-firmware` later.

One real bug was caught and fixed by the parity tests, not eyeballing: the
`iq` (quantized-activation) scratch buffer was sized
`max(cfg.dim, cfg.ffn)`, which omits `cfg.ple_dim` — for the 14.9MB model
`ple_dim=128` exceeds both `dim=96` and `ffn=66`, since `ple_proj` is bound
with `cols=ple_dim`. Running the int8-activations `ppl` binary against that
model panicked with an out-of-bounds index until this was fixed to
`max(cfg.dim, max(cfg.ffn, cfg.ple_dim))` (`llm-core/src/scratch.rs`).

A second, smaller precision gap was found and closed the same way: C's
`quantize_act` uses `(int)lrintf(...)`, which rounds half-to-even under the
default FPU rounding mode; the initial Rust port used `libm::roundf`, which
rounds half-away-from-zero. These disagree on exact `.5` ties, and the
disagreement compounds through the causal KV cache across a generation
window. Replaced with a hand-written `round_ties_even()`
(`llm-core/src/tensor.rs`) since neither `core` nor `libm` expose the
tie-to-even behavior in `no_std`. This closed most, not all, of a ~0.0001
cross-entropy gap against the C reference — see "Parity test suite" below
for why the remainder is expected, not a bug.

### Phase 2 — `llm-host`: replace the C host tooling — DONE
- `gen-prompt`, `host-generate`, `ppl` binaries: direct ports of
  `gen_prompt.c`, `host_generate.c`, `ppl.c`.
- Vocab loading (`vocab.rs`): `vocab.h` is read as a runtime argument
  instead of being `#include`d at compile time — the one deliberate CLI
  ergonomics deviation from the C originals (documented in each binary's
  own doc comment).

#### Parity test suite (`llm-host/tests/`, run under both `cargo test
--release` and `cargo test --release --features int8-activations`):

- **`golden.rs`** — last-position logits vs PyTorch's `golden.txt`.
  fp32-activations: passes at the *same precision the C reference's own
  `verify.c` gets* (max abs diff 0.00001, rms 0.000002 — both bottom out at
  float32 epsilon, not just "under the 0.02 tolerance"). int8-activations
  is checked differently, against the C reference's *own* int8 numerics
  (0.13001 / 0.029788), not against the fp32 PyTorch golden — because int8
  activation quantization is a genuinely lossy approximation of the fp32
  math by construction, and the fp32-tolerance failing under it is expected
  in C too. Proof: rebuilding `verify.c` itself with `-DLLM_INT8_ACT` and
  running it against this exact `model.bin`/`golden.txt` produces the
  *identical* divergence (0.13001 / 0.029788) — so the Rust int8 path
  matches the C int8 path to 5 decimal places; it just also (correctly)
  fails to match an fp32 reference it was never going to match.
- **`cli_parity.rs`** — `host-generate`'s stdout is byte-for-byte identical
  to a captured run of the compiled C `host_generate.c` binary (same
  model, same hardcoded prompt, default N=120 continuation).
- **`ppl_parity.rs`** — cross-entropy over `data/val_v32768.bin`.
  fp32-activations: exact match to the C reference's printed digits
  (CE 2.6671). int8-activations: within a tolerance set from an empirically
  measured float non-associativity noise floor — rebuilding the *C
  reference itself* with `-march=native` instead of plain `-O2` alone
  shifts its own CE from 2.6684 to 2.6678, a bigger swing than the ~0.0001
  gap between this Rust port (2.6683) and the baseline C build (2.6684).
  That's the standard "two different compilers, same algorithm, different
  instruction scheduling" float-precision story, not a logic error.

**Exit gate, met:** on the same `model.bin`, `reference-c`'s C host tools
and `llm-host`'s Rust host tools produce identical generated text and
perplexity (fp32-activations exactly; int8-activations within the measured
fp noise floor, itself smaller than the C reference's own build-to-build
variance). This is the proof the port is correct, independent of any ESP32
hardware — Phase 3 can start whenever hardware/toolchain access is
available.

### Phase 3 — `llm-firmware` on esp-idf-hal (std, FreeRTOS) — NOT STARTED
Blocked on tooling this environment doesn't have: the ESP-IDF/Xtensa
toolchain (or RISC-V, depending which chip variant this targets) and
physical ESP32-S3 hardware to flash and measure tok/s against. Everything
that can be verified without hardware (Phases 1-2) is done; this phase
needs to run wherever that toolchain and the board actually are.

Create the crate (deferred from Phase 0, see `llm-firmware/README.md` for
the full API mapping table). In short: partition mmap via `esp-idf-svc`,
PSRAM alloc via `esp-idf-sys` heap_caps bindings, the dual-core int8 head
split via FreeRTOS task bindings (same worker-on-core-0 design as today),
UART emit and boot/profile timing via `esp-idf-hal`/`esp-idf-svc`.

**Exit gate:** flash it, compare boot diagnostics and measured tok/s against
`reference-c/firmware/esp32_llm/README.md`'s recorded numbers (102.9ms/step,
9.72 tok/s compute-only). Should match within noise — a meaningful slowdown
is a regression to chase before calling this phase done, not something to
wave off as "Rust overhead."

### Phase 4 — `llm-display` (optional, deferred, lowest priority)
Port of `display.h`. Not required for "does it generate text" — do this
last, only if the on-device demo screen matters to you.

### Phase 5 — `bandwidth_bench` port (optional, independent, any time)
Port of the raw cycle-counter PSRAM/SRAM/flash benchmark. Confirms the same
memory numbers hold under the Rust build. Off the critical path.

### Phase 6 — drop FreeRTOS: re-platform `llm-firmware` onto esp-hal
Only after Phase 3 is verified. Swap `esp-idf-hal`/`esp-idf-sys` for
`esp-hal`: no_std, no FreeRTOS, no ESP-IDF C SDK anywhere in the build.
Dual-core split becomes esp-hal's core-1 closure spawn. `llm-core` does not
change at all in this phase — that's the entire reason it was split out in
Phase 0.

**Exit gate:** same as Phase 3 (boot diagnostics + tok/s match), now with
zero C, zero FreeRTOS, zero ESP-IDF anywhere in the build or the toolchain.

## Explicitly out of scope (left exactly as-is)

- `src/*.py` (train.py, quantize.py, export.py, sample.py, budget.py,
  analyze.py, gen_assets.py, model.py) — untouched.
- `esp32-llm-lab/chat.py` — untouched (Python, not C/C++; also the only
  place text ever gets tokenized, and it stays off-chip).
- `data/` under the original esp32-ai project, `runs/` — untouched. (The
  `data/val_v32768.bin` under *this* repo's own `data/` is a copy staged
  purely for the Rust ppl parity test to read; nothing writes to it.)
- The binary model format itself (ragged int4, fp16 scales, magic header) —
  unchanged; `llm-core` reads exactly what `export.py` already produces.

## Running the parity tests yourself

```
cd engine/llm-host
cargo test --release                              # fp32-activations path
cargo test --release --features int8-activations  # int8-activations path
```

Both must stay green before Phase 3 (or any further) work lands. There is
no network access needed to build or run these — `llm-core`'s only
dependencies are `half` and `libm`, both already in `Cargo.lock` once
fetched once.
