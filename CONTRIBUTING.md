# Contributing

The main open work right now is closing the Rust-vs-C performance gap — see
the README's "Status" section for the current numbers and the "Open perf
thread" subsection for the specific next target (`matvec_int8_activations`,
real SIMD via `esp-dsp`). Start there.

## Ground rules

1. **`reference-c/` is frozen.** Never hand-edit it to "fix" something —
   it's a verbatim mirror of [slvDev/esp32-ai](https://github.com/slvDev/esp32-ai),
   refreshed wholesale from upstream when it changes, not patched locally.
   If you find a bug in it, that's a fact about the port's target, not
   something to correct here.
2. **`llm-core` stays platform-agnostic** (`no_std`, no ESP32-specific code).
   Platform-specific speedups (like the dual-core head split, or the planned
   `esp-dsp` SIMD hook) belong in `llm-firmware`, wired in behind a fallback
   to the portable scalar path. See `MIGRATION_PLAN.md` for the reasoning.
3. **Every change to `llm-core` math must stay parity-verified**, not just
   "looks right." See below.

## Building and testing

```bash
cd engine/llm-host
cargo test --release                              # fp32-activations path
cargo test --release --features int8-activations  # int8-activations path
```

Both must pass before and after your change. If you touch anything in
`engine/llm-core/src/tensor.rs` or `model.rs`, run both — they exercise
different code paths (`matvec_int8_activations` is only compiled under
`--features int8-activations`).

For firmware-side changes, `engine/llm-firmware` needs the ESP-IDF
toolchain and real hardware; `cd engine/llm-firmware && cargo run --release`
builds and flashes. There's no host-side substitute for confirming a
timing claim — a change to `llm-firmware` isn't "done" until it's measured
on an actual ESP32-S3, not just compiled.

## Making a performance change

1. Confirm what you're changing has a parity test covering it (`golden.rs`,
   `ppl_parity.rs`, `cli_parity.rs` in `engine/llm-host/tests/`). If it
   doesn't, that's worth fixing first — an unverified speedup is a
   correctness bug waiting to happen.
2. Measure before and after on real hardware if the change touches
   `llm-firmware`; a host-only "should be faster" claim doesn't belong in
   the README's numbers table.
3. Update the README's status table and per-stage timing table with your
   measured numbers, and say what you changed and why in one or two
   sentences (see the existing "Three changes got it from 3.0x slower..."
   section for the format).

## Branches / PRs

Branch names like `perf/<what>` for performance work are a reasonable
default (see `perf/int8-matvec-byte-pair-unroll` as an example). Small,
single-purpose PRs are easier to parity-check than large ones — a change
that touches both `llm-core` math and `llm-firmware` platform code is
usually two PRs, not one.
