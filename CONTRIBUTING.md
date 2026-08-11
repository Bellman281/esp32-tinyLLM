# Contributing

The main open work right now is closing the Rust-vs-C performance gap — see
the README's "Status" section for the current measured numbers, and
[BENCHMARKING.md](./BENCHMARKING.md)'s "What to optimize next" for the two
real levers, ranked, with what is already built for each. Start there.

## Ground rules

1. **`reference-c/`'s inference logic is never edited.** It mirrors
   [slvDev/esp32-ai](https://github.com/slvDev/esp32-ai) and is refreshed
   wholesale from upstream, not patched locally. If you find a bug in its
   math, that is a fact about the port's target, not something to correct
   here. The one carve-out is build and benchmark configuration — toggles
   and constants the sketch already exposes for that purpose
   (`USE_DISPLAY`, `N_GENERATE`, `PROMPT_IDS`) — set so the C build
   compiles and so its measurements match the Rust ones. Every such change
   must be recorded in `reference-c/README.md`'s "Local changes vs.
   upstream" table.
2. **`llm-core` stays platform-agnostic** (`no_std`, no ESP32-specific code).
   Platform-specific speedups (like the dual-core head split, or the planned
   `esp-dsp` SIMD hook) belong in `llm-firmware`, wired in behind a fallback
   to the portable scalar path. See `MIGRATION_PLAN.md` for the reasoning.
3. **Every change to `llm-core` math must stay parity-verified**, not just
   "looks right." See below.
4. **Model paths and expected values live in `model/models.toml`**, not in
   consts scattered across test files. If you need a model file, a prompt,
   a window count, or an expected cross-entropy, read it from the registry
   — `llm_host::manifest` in Rust, `scripts/model.sh` in shell/C. Adding a
   new hardcoded `const MODEL_BIN` is how the C and Rust sides drift apart.
   See the README's "The model registry".

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
   correctness bug waiting to happen. If your change deliberately gives up
   bit-exactness (any SIMD reordering of an `f32` accumulation does), it
   needs a *measured* tolerance, not a guessed one —
   `matvec_simd_tolerance.rs` is the worked example.
2. Measure before and after on real hardware if the change touches
   `llm-firmware`; a host-only "should be faster" claim doesn't belong in
   the README's numbers table. `git stash` / `git stash pop` around a
   same-session A/B is the cheapest way to keep both sides honest — this
   workload is deterministic enough that two flashes of identical code
   reproduce to 0.1 ms/stage, so small deltas are real, not noise.
3. **Never compare runs taken at different benchmark settings.** `attn` is
   the only stage whose cost scales with sequence position, so a different
   `N_GENERATE` or prompt length silently changes it while leaving every
   other stage identical. A published 2.10x attention ratio in this repo
   turned out to be entirely this artifact. The settings live in
   `model/models.toml`, and `benchmark_settings_match.rs` asserts both
   firmwares agree with it — if you change the prompt or token count,
   change it there and let that test tell you what else needs updating,
   rather than editing one firmware and moving on. The measurement
   methodology is in [BENCHMARKING.md](./BENCHMARKING.md).
4. Update the per-stage timing tables in the README and
   [BENCHMARKING.md](./BENCHMARKING.md) with your measured numbers, and say
   what you changed and why in one or two sentences (see BENCHMARKING.md's
   "Optimization history" for the format).

## Branches / PRs

Branch names like `perf/<what>` for performance work are a reasonable
default (`perf/attention-rmsnorm-zip`, `perf/matvec-simd-dsps`). Small,
single-purpose PRs are easier to parity-check than large ones — a change
that touches both `llm-core` math and `llm-firmware` platform code is
usually two PRs, not one.

`main` holds only code that has actually been built and measured.
Speculative or unverified work — an FFI binding whose signature hasn't
been confirmed against a real vendored header, a kernel that's never been
compiled for the target — belongs on a branch until it has been, however
well-commented it is. Feature-gating it `default = []` is not a substitute:
a reader of `main` should be able to assume everything there is real.
