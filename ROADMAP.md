# Roadmap

What to do next on this project, in order, and why that order. Written at
`v0.1.0` (2026-08-11), when the port first ran verified on hardware.

Performance *targets* and their measurements live in
[`BENCHMARKING.md`](./BENCHMARKING.md); this file is about **sequencing** —
which work is blocked on what, and which questions are still open. Read
that one for the numbers, this one for the plan.

## Where things stand

- Runs on an ESP32-S3-DevKitC (N16R8) and produces **byte-identical output
  to the C firmware** over 400 tokens.
- Host parity suite green: 17 tests, 15 under `--features int8-activations`.
- **Faster than C** overall: 119.0 vs 122.0 ms/token wall clock, 8.40 vs 8.20
  tok/s (+2.4%), byte-identical output. Was 1.53x slower before the nibble-LUT,
  KV-layout, head-arithmetic, dual-core and argmax work — measured on one
  board, same evening, same `model.bin`.
- Phases 0–3 done. Phases 4 (display), 5 (bandwidth bench), 6 (drop
  FreeRTOS) not started.

## Do these two first — they gate everything expensive

Both are cheap. Both change what you'd do next. Doing SIMD work before them
means optimizing on a guess.

### 1. Bisect the unexplained 174.2 → 179.6 regression

**5.4 ms/token — 9% of the entire Rust-vs-C gap — is unexplained.**

`BENCHMARKING.md`'s optimization history ends at 174.2 ms for the
`+ wider caches` step. The tree at `v0.1.0` measures 179.6. Two things
landed in between: the `attention.rs`/`ops.rs` zip loops (which measured as
an *improvement* at the time, 182.1 → 177.3) and `llm-core`'s
`matvec_override` hook. We suspect the hook costs ~2 ms, but that was never
isolated, and 174.2 itself was never reproduced — so there may be a second
regression hiding, or the original 174.2 may have been measured under
conditions nobody recorded.

**Method:** check out a few historical commits, `cargo run --release` from
`engine/llm-firmware` on each, record `profile ms/token`. This workload is
deterministic to 0.1 ms/stage on repeated flashes, so differences are real
signal, not noise. Four flashes should localize it.

**Payoff:** either recover 5.4 ms, or retire a number that was never
reproducible. Also settles the hook's cost as a side effect — see below.

### 2. Port `bandwidth_bench` (Phase 5)

**The head was +40.9 ms and is now +4.4 (1.08x of C).** It was never
bandwidth-bound — instrumenting it showed 19% of available PSRAM bandwidth in
use and a stalled dependency chain plus two redundant operations per element.
See BENCHMARKING.md's "The output head — done". The gap is now spread almost
evenly across all five stages, with no stage above 1.50x.

The belief that it's PSRAM-bandwidth-bound rests on a single data point:
widening the external-memory caches moved it −21%, more than any other
stage. That's suggestive, not conclusive.

This matters because the two answers imply opposite work:

- **Bandwidth-bound** → SIMD will do nothing. The levers are data layout,
  prefetch, cache configuration, maybe restructuring the staged int8 table.
- **Compute-bound** → `head-simd` (below) is the right move.

`reference-c/firmware/bandwidth_bench/bandwidth_bench.ino` already exists
and is unported. It's a standalone cycle-counter benchmark — no dependency
on the rest of the port.

**Payoff:** turns the single biggest lever in the project from a guess into
a decision.

## Then: optimization

Order depends on what #2 says.

| # | Work | Targets | Notes |
|---|---|---|---|
| 3 | **`head-simd`** — esp-dsp int8 dot product for the output head | head, +40.9 ms (1.72x) | Only worth doing if #2 says compute-bound. Much simpler than the fp32 case: int8 accumulation is associative, so SIMD reordering is provably lossless — no tolerance question at all. Was drafted once and dropped from `main` as unverified; see `git log` around v0.1.0. |
| 4 | **`matvec-simd`** — esp-dsp fp32 matvec | ple 1.79x, input 1.77x, ffn 1.72x — together +15.1 ms | Worst *ratios*. Scaffolding already ships: the `matvec_override` hook, `QT::matvec_range_chunked`, and a measured drift bound (`matvec_simd_tolerance.rs`). Real risk: an FFI call per row, at rows of only 66–128 elements, may cost more than it saves. Not a guaranteed win. |
| 5 | **`FVec::get`** | input + ple stages | Cheapest concrete item here. `tensor.rs`'s own doc comment flags it: four separately bounds-checked byte loads plus `from_le_bytes` per element, inside `rmsnorm`'s hot loop — which lives in the two stages with the worst ratios. Correctness is host-testable; the win needs hardware. |

Both esp-dsp items share one prerequisite: **vendor `esp-dsp` and confirm the
real `dsps_dotprod_*` signatures against the vendored header.** The drafts
that were stripped from `main` guessed those signatures and were never
compiled — do not trust them, and don't re-add unverified FFI to `main` (see
`CONTRIBUTING.md`).

### The `matvec_override` hook decision

`llm-core`'s `matvec_override` / `dispatch_matvec` adds a branch to every
matvec call and **currently has no user** — the esp-dsp path it exists for
isn't written. If #1 confirms it costs ~2 ms/token for nothing, there's a
real choice: keep it as scaffolding, or revert it and re-add it alongside an
actual implementation. `QT::matvec_range_chunked` and the tolerance test are
test-only and cost nothing either way — they can stay regardless.

## Correctness debt

**6. A PyTorch golden reference for the shipped model.** `golden.rs` today
validates `golden-v4096`, a small fixture model that is never shipped,
flashed, or benchmarked — because it's the only one with a `golden.txt`.
The 28.9M model everything else describes has no direct PyTorch check; its
correctness is established against the C reference, which was itself checked
against PyTorch upstream. That transitive chain is decent but not
first-hand. Generating a golden run for `tinystories-v32768` via
`slvDev/esp32-ai`'s export tooling closes it. See `model/README.md`'s
"A real inconsistency in this repo".

**7. Nothing bounds host-vs-device divergence directly.** Both ports'
host tools and firmware diverge after ~100 tokens because the firmware
stages the output head as int8 while the host runs it in fp32. That's
expected and `ppl_parity.rs` bounds the int8 path in aggregate, but no test
pins the specific device head path against its fp32 equivalent on real
weights. `int8_head_staging.rs` covers the staging, not the full path.

## Infrastructure

**8. CI.** Nothing runs the parity suite automatically. A GitHub Action
running `cargo test --release` both ways on PRs is the difference between
having a parity suite and having one that's actually enforced — which
matters more now that the repo is set up for contributors. The 14.9 MB
tracked `model.bin` makes checkout heavy but workable.

**9. Generate instead of detect.** `benchmark_settings_match.rs` *catches*
drift between `model/models.toml` and the two firmwares' hardcoded
`PROMPT_IDS`/`N_GENERATE`. A build step generating those constants from the
registry would make the drift impossible instead of merely detectable. Same
applies to the embedded vocab (`assets/vocab_*.bin`, currently regenerated
by hand via `vocab-to-bin`). Bigger change; eliminates a bug class that has
already bitten this project once.

**10. The 16.5 MB duplication — resolved, partly.** `model/` is now the
canonical home: `model/tinystories-v32768/{model.bin, vocab.h, val.bin}`, one
folder per registry entry, and `data/` is gone. What remains duplicated is
`reference-c/esp32-llm-lab/`'s own `model.bin` and `vocab.h` (~15.7 MB), and
that is deliberate — `reference-c/` is a frozen mirror refreshed wholesale
(CONTRIBUTING ground rule 1), and `gen_prompt.c` `#include`s that `vocab.h` at
compile time. Nothing in the registry points at them. Removing it would mean
either patching the mirror or breaking the C build, so it stays until the C
reference itself is retired.

## Phases 4 and 6

**Phase 4 — `llm-display`.** ST7789/OLED demo screen, ported from
`reference-c/firmware/esp32_llm/display.h`. Unblocked: it builds on
`llm-firmware`, which hardware has now confirmed. Also the natural moment to
revisit `main.rs`'s `emit()`, which currently always blocks on the serial
write; the C reference gates on `Serial.availableForWrite()` so generation
doesn't stall with no USB host attached — which only becomes a real scenario
once a display exists.

**Phase 6 — drop FreeRTOS, re-platform onto `esp-hal`.** no_std, no ESP-IDF
C SDK. The dual-core split becomes an `esp-hal` core-1 closure spawn instead
of a FreeRTOS task. This is where the `llm-core` / `llm-firmware` split pays
off: **`llm-core` should not change at all.** If it has to, that's a sign
platform assumptions leaked into it. Same exit gate as Phase 3 — boot
diagnostics and measured tok/s.

## Open questions worth recording

- Why is the published 174.2 not reproducible? (→ #1)
- Is the head bandwidth- or compute-bound? (→ #2)
- Does an FFI call per row pay for itself at 66–128 element rows? (→ #4)
- Upstream's two READMEs report 94.9 and 102.9 ms/token for the C firmware
  and never reconcile. Neither matched this board (119.8 at matched
  settings, 103.1 at upstream's own 200-token defaults). Not worth chasing —
  recorded so nobody re-derives it.
- `PSRAM free after alloc` differs between the firmwares (C 3141 KB, Rust
  3211 KB). Small and plausibly just allocator behaviour, but never
  explained.
