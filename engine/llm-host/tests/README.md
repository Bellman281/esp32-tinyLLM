# llm-host/tests — the port's correctness gate

Phase 1-2 status: **done**. Six gates, run under both build configs
(default = fp32-activations, `--features int8-activations`):

- `golden.rs` — last-position logits vs PyTorch's golden.txt.
  fp32 build: exact-precision match (max abs diff 0.00001, same as the C
  reference's own `verify.c` run). int8-activations build: checked against
  the C reference's *own* int8 numerics instead (0.13001/0.029788), since
  int8 quantization is expected to diverge from the fp32 PyTorch reference
  by construction — proved by rebuilding `verify.c` itself with
  `-DLLM_INT8_ACT`, which shows the identical divergence.
- `cli_parity.rs` — `host-generate`'s stdout vs a byte-for-byte capture of
  the compiled C `host_generate.c` binary (same model, same hardcoded
  prompt, default N=120). Passes under both build configs.
- `ppl_parity.rs` — cross-entropy over `model/tinystories-v32768/val.bin` vs captured C
  reference values. fp32: exact match. int8: within a tolerance set from
  the empirically measured float non-associativity noise floor (rebuilding
  the C reference with `-march=native` instead of `-O2` alone shifts its
  own CE by more than the Rust-vs-C gap).
- `manifest.rs` — every path `model/models.toml` declares must resolve, and
  an unknown key is an error rather than a silent skip. A typo'd path in the
  registry fails here rather than surfacing as a confusing runtime error.
- `benchmark_settings_match.rs` — both firmwares' `PROMPT_IDS` / `N_GENERATE`
  must agree with the registry. This is the regression gate for a real
  incident: a published "2.10x attention" figure turned out to be an artifact
  of comparing runs taken at different token counts (`attn` is the only stage
  whose cost scales with sequence position). See BENCHMARKING.md, "Why matched
  settings matter".
- `matvec_simd_tolerance.rs` — measures how far a SIMD-shaped reordering of
  `matvec_range`'s `f32` accumulation moves validation cross-entropy
  (~3-4e-8), so a future `esp-dsp` implementation has an empirically measured
  bound to work against rather than a guessed one.

Two real bugs were caught by this suite during the port, not by inspection: an
out-of-bounds scratch-buffer size for one model config, and a rounding-mode
mismatch against C's `lrintf` (round-half-to-even vs round-half-away-from-zero).
Both are fixed; `MIGRATION_PLAN.md` has the detail.

Run:
```
cd llm-host
cargo test --release                              # fp32-activations path
cargo test --release --features int8-activations  # int8-activations path
```

Both must stay green before any further phase (3+) work lands.
