# llm-host/tests — the port's correctness gate

Phase 1-2 status: **done**. Four parity checks, run under both build
configs (default = fp32-activations, `--features int8-activations`):

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
- `ppl_parity.rs` — cross-entropy over `data/val_v32768.bin` vs captured C
  reference values. fp32: exact match. int8: within a tolerance set from
  the empirically measured float non-associativity noise floor (rebuilding
  the C reference with `-march=native` instead of `-O2` alone shifts its
  own CE by more than the Rust-vs-C gap).

Run:
```
cd llm-host
cargo test --release                              # fp32-activations path
cargo test --release --features int8-activations  # int8-activations path
```

Both must stay green before any further phase (3+) work lands.
