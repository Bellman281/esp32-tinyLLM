# llm-host/tests — the port's correctness gate

16 integration test files, 38 `#[test]` functions, run under both build configs
(default = fp32-activations, `--features int8-activations`). Each file's `//!`
header carries the full reasoning; this is the map. Nothing in `llm-firmware`
can be tested here — no host has the Xtensa toolchain, FreeRTOS or PSRAM — so
the rule is: pin the *arithmetic* of every device optimization on the host,
exactly, so a mismatch on the board can only be the plumbing.

## Exactness gates — the engine vs. its references

- `golden.rs` — last-position logits vs PyTorch's `golden.txt`. fp32 build:
  max abs diff 0.00001, same as the C reference's own `verify.c`.
  int8-activations build: checked against the C reference's *own* int8
  numerics (0.13001/0.029788) instead, since int8 quantization diverges from
  the fp32 PyTorch reference by construction — proved by rebuilding `verify.c`
  with `-DLLM_INT8_ACT` and seeing the identical divergence.
- `cli_parity.rs` — `host-generate`'s stdout vs a byte-for-byte capture of the
  compiled C `host_generate.c` (same model, same prompt, default N=120). Proves
  the whole CLI — decode loop, vocab lookup, argmax bound — matches end to end,
  not just the raw logits. Passes under both build configs.
- `ppl_parity.rs` — cross-entropy over `model/tinystories-v32768/val.bin` vs
  captured C values. fp32: exact match. int8: within a tolerance set from the
  measured float non-associativity noise floor (rebuilding the C reference with
  `-march=native` shifts its own CE by more than the Rust-vs-C gap).
- `token_stream_digest.rs` — replays exactly what `llm-firmware`'s `main` does
  (prime on `prompt_ids`, then `n_generate` greedy steps with a
  `vocab.n`-bounded argmax) and folds every emitted id into
  `llm_core::TokenDigest`, pinning the value in `model/models.toml`
  (`gen_digest`). The board prints the same number, so "byte-identical output"
  became one value to compare — `0x327578cb136fd6aa` — instead of 400 tokens of
  prose to eyeball. Every device figure claimed it; until this file, nothing
  checked it.
- `header_v2.rs` — the `"PLE\0"` header gate. Two formats exist for the same
  tensor stream: `"PLE1"` (the shipped `model.bin` and the frozen C reference)
  and `"PLE\0"` (what `training/`'s exporter writes). `Model::load` once
  accepted only the first, so anything produced by following TRAINING.md died
  with `BadMagic`. Each fixture rewrites the *real* `model.bin`'s header and
  leaves its tensor bytes alone, so the parse alone is under test.

## Equivalence gates — every device optimization, pinned exactly

Each was written *before* its firmware change, asserting bit-for-bit equality
against the path it replaces on every row of the shipped model. Integer addition
is associative and every term is the same term, so "close" would mean a bug.

- `int8_head_staging.rs` — `QT::unpack_row_int8` (the int4→int8 boot-time
  staging the C reference's `stage_head_int8` does) reconstructs the same
  per-row values as the verified `deq_row` fp32 dequant path.
- `int4_head_equivalence.rs` — gated staging the head as *packed int4* instead,
  reading nibbles directly via `dot_int4_int8` and halving what streams per
  token (2.54 MB → 1.32 MB).
- `int8_head_equivalence.rs` — gated going back to int8, on the theory that the
  per-element nibble mask was why the head was the last stage above C. Wrong
  reason: head 62.3 → 76.9 ms, token 106.3 → 121.5. Reverted; kept because the
  equivalence makes it cheap to re-run on a board with different memory.
- `int16_act_equivalence.rs` — gated holding the head's quantized activation as
  `i16`. The LX7 has no signed byte load, so every `i8` read cost a load *and* a
  `sext`, 2.43M times per token; `l16si` folds the extension into the load.
  Asserts `quantize_activations_i16` and `dot_int4_int16` return exactly what
  the `i8` versions do.
- `dot2_equivalence.rs` — gated computing two head rows per pass. The activation
  vector is identical for all 25,353 rows and the one-row loop reloaded it for
  each; `dot2_int4_int16` loads each activation once and multiplies it into two
  rows. Asserts both returned values match the single-row results, on every
  consecutive row pair.
- `matvec_simd_tolerance.rs` — the one deliberate *non*-exactness. Measures how
  far a SIMD-shaped lane-reduced reordering of `QT::matvec_range`'s `f32`
  accumulation moves validation cross-entropy (~3–4e-8), so a future `esp-dsp`
  implementation has an empirically measured bound rather than a guessed one.
  fp32-path only; compiled out under `int8-activations`.
- `fp16_kv_tolerance.rs` — the same discipline ahead of a change nobody has
  committed to. Attention's `core` sub-stage is ~70% bytes, so halving the KV
  cache is the largest win left — and would be the port's first non-bit-exact
  change. Simulates an fp16 cache by rounding k/v through `f16` on store, and
  **reports rather than gates**: no threshold to assert until someone decides
  what this engine is willing to spend.

## Split-correctness gates — the dual-core work

Splitting is bit-exact in principle (`y[r]` depends only on row `r` and `x`;
each attention head reads the whole KV cache and writes only its own output
slice). "In principle" is not the bar — what these check is the *re-addressing*
that lets two cores hold disjoint `&mut` halves instead of both holding `&mut`
to the whole buffer.

- `matvec_split.rs` — asserts `QT::matvec_rows_into`'s relative addressing
  changed nothing, at every split point, on real tensors.
- `attn_split.rs` — the same for `attention::heads`, through a real multi-token
  generation so the KV cache accumulates (a split that corrupted one head's
  cache region would surface at the *next* position). Nothing but attention is
  overridden, so any logit difference can only be the split.
- `head_override.rs` — wiring, not arithmetic: proves
  `llm_forward_with_head_override`'s closure is actually called instead of the
  default head matvec, receives `x`/`y` at the documented shapes, and that what
  it writes to `y` is what lands in `s.logits`.

Neither split test reaches the FreeRTOS handoff in `llm-firmware`'s `head.rs`.
That is the point.

## Settings gate

- `benchmark_settings_match.rs` — both firmwares' `PROMPT_IDS` / `N_GENERATE`
  must agree with `model/models.toml`. Regression gate for a real incident: a
  published "2.10x attention" figure was an artifact of comparing runs taken at
  different token counts (`attn` is the only stage whose cost scales with
  sequence position); at matched settings it was 1.25x, the *best* stage. It
  parses source text because neither firmware can read TOML at runtime. See
  BENCHMARKING.md, "Why matched settings matter".

The registry itself is gated by unit tests in `src/manifest.rs`
(`real_registry_parses_and_its_files_exist`): every path `models.toml` declares
must resolve, and an unknown key is an error rather than a silent skip.

## Run

```
cd llm-host
cargo test --release                              # fp32-activations path
cargo test --release --features int8-activations  # int8-activations path
```

Both must stay green. Two real bugs were caught by this suite during the port,
not by inspection: an out-of-bounds scratch-buffer size for one model config,
and a rounding-mode mismatch against C's `lrintf` (round-half-to-even vs
round-half-away-from-zero). Both fixed; detail in
[the port story](../../../docs/esp32-tinyLLM-port-story.pdf).
