# reference-c/ — frozen ground truth, read-only

Verbatim copy of the C/C++ inference engine and ESP32 firmware from
vendor/esp32-ai, taken 2026-08-02. This folder is never edited during the
Rust port. It exists for two reasons:

1. Every phase of the port is checked against it (same inputs, same outputs).
2. It's self-contained here so the whole migration workspace — old and new —
   lives in one place instead of jumping between two repos.

The real source of truth remains vendor/esp32-ai. If that project changes
(model retrained, vocab changed, format tweaked), refresh this folder from
there — do not hand-edit files here to "fix" something; that means the
Rust port's target moved, not that this copy is wrong.

## What's here, and where each piece goes in engine/

| File | What it does | Ports to |
|---|---|---|
| `esp32-llm-lab/llm.h` = `firmware/common/llm.h` | Portable inference engine: binary model parser, int4/int8 matvec kernels, rmsnorm/gelu/silu, RoPE, causal attention + KV cache, SwiGLU FFN, PLE gate, `llm_forward()` | `engine/llm-core` |
| `esp32-llm-lab/gen_prompt.c` | CLI: prompt token ids -> generated text, greedy or top-k+temperature | `engine/llm-host` (`gen-prompt` binary) |
| `esp32-llm-lab/host_generate.c` | CLI: greedy-only smoke test | `engine/llm-host` (`host-generate` binary) |
| `esp32-llm-lab/vocab.h` | Generated data table (token id -> UTF-8 bytes). Not logic — consumed as-is. | data dependency of `engine/llm-host` and `engine/llm-firmware`, not ported |
| `esp32-llm-lab/model.bin`, `firmware/model/model.bin` | Trained, quantized model weights (Python-produced, untouched) | data dependency, not ported |
| `firmware/model/golden.txt`, `golden.npz` | PyTorch reference logits for a fixed prompt | test fixture for `engine/llm-host`'s golden-logit test |
| `firmware/host_verify/verify.c` | Host correctness gate: compares C's logits to golden.txt | `engine/llm-host` golden-logit test (Phase 2) |
| `firmware/host_verify/ppl.c` | Host perplexity/quality harness, with/without int8 activations | `engine/llm-host` perplexity binary (Phase 2) |
| `firmware/esp32_llm/esp32_llm.ino` | ESP32-S3 firmware: flash partition mmap, PSRAM alloc, dual-core int8 head split, boot diagnostics | `engine/llm-firmware` (Phase 3, then Phase 6) |
| `firmware/esp32_llm/display.h` | OLED (SH1106/SSD1306) / TFT (ST7789) demo screen driver | `engine/llm-display` (Phase 4, optional) |
| `firmware/esp32_llm/partitions.csv` | Flash partition table (bootloader/app/model layout) | reused as-is by `engine/llm-firmware`'s build |
| `firmware/bandwidth_bench/bandwidth_bench.ino` | Raw cycle-counter PSRAM/SRAM/flash bandwidth benchmark | standalone port (Phase 5, optional) |
| `esp32-llm-lab/flash.bin` | Prebuilt 16MB QEMU flash image of the *C* firmware | reference only, for side-by-side QEMU comparison — not ported |

Not copied here (out of scope, stays Python, see /MIGRATION_PLAN.md):
`esp32-llm-lab/chat.py` (Python tokenizer + CLI wrapper — Python, not C/C++),
and everything under `src/`, `data/`, `runs/` in vendor/esp32-ai.
