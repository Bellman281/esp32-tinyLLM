# reference-c/ — frozen ground truth, read-only

Verbatim copy of the C/C++ inference engine and ESP32 firmware from
[slvDev/esp32-ai](https://github.com/slvDev/esp32-ai), taken 2026-08-02.
This folder is never hand-edited during the Rust port. It exists for two
reasons:

1. Every phase of the port is checked against it (same inputs, same outputs).
2. It's self-contained here so the whole migration workspace — old and new —
   lives in one place instead of jumping between two repos.

The real source of truth remains `slvDev/esp32-ai`. If that project changes
(model retrained, vocab changed, format tweaked), refresh this folder from
there — do not hand-edit files here to "fix" something; that means the
Rust port's target moved, not that this copy is wrong. Adding files that
are genuinely part of the upstream bundle but were missing from an earlier
copy (as happened with `esp32-llm-lab/chat.py` and friends, see below) counts
as a refresh, not a hand-edit.

## Two different `model.bin` files here — don't mix them up

- **`esp32-llm-lab/model.bin`** (14.9 MB, `V=32768 D=96 L=6 H=4 F=66 P=128`,
  ~28.9M params) — **the real model.** This is what every performance
  number, the top-level README, and `model/` describe. Confirmed against an
  independently-supplied training export with a matching SHA-256.
- **`firmware/model/model.bin`** (1.87 MB, `V=4096 D=128 L=6 H=4 F=415 P=64`,
  ~3.5M params) — a much smaller, different model. It's only used as the
  fixture for `engine/llm-host/tests/golden.rs` (it has a matching
  `golden.txt`/`golden.npz` PyTorch reference; the big model currently
  doesn't — see `model/README.md`'s "A real inconsistency" section). Don't
  treat this as "the" model — it isn't what gets described anywhere else in
  this repo, and `firmware/esp32_llm/README.md`'s own flash instructions
  pointing at this file while its boot-diagnostics text describes the
  *other* model's config is an unresolved inconsistency inherited from
  upstream, not something fixed here.

## What's here, and where each piece goes in engine/

| File | What it does | Ports to |
|---|---|---|
| `esp32-llm-lab/llm.h` = `firmware/common/llm.h` | Portable inference engine: binary model parser, int4/int8 matvec kernels, rmsnorm/gelu/silu, RoPE, causal attention + KV cache, SwiGLU FFN, PLE gate, `llm_forward()` | `engine/llm-core` |
| `esp32-llm-lab/gen_prompt.c` | CLI: prompt token ids -> generated text, greedy or top-k+temperature | `engine/llm-host` (`gen-prompt` binary) |
| `esp32-llm-lab/host_generate.c` | CLI: greedy-only smoke test | `engine/llm-host` (`host-generate` binary) |
| `esp32-llm-lab/vocab.h` | Generated data table (token id -> UTF-8 bytes). Not logic — consumed as-is. | data dependency of `engine/llm-host` and `engine/llm-firmware`, not ported |
| `esp32-llm-lab/model.bin`, `firmware/model/model.bin` | Trained, quantized model weights (Python-produced, untouched) — **two different models, see above** | data dependency, not ported |
| `firmware/model/golden.txt`, `golden.npz` | PyTorch reference logits for a fixed prompt — paired with `firmware/model/model.bin` (the small model), not `esp32-llm-lab/model.bin` | test fixture for `engine/llm-host`'s golden-logit test |
| `firmware/host_verify/verify.c` | Host correctness gate: compares C's logits to golden.txt | `engine/llm-host` golden-logit test (Phase 2) |
| `firmware/host_verify/ppl.c` | Host perplexity/quality harness, with/without int8 activations | `engine/llm-host` perplexity binary (Phase 2) |
| `firmware/esp32_llm/esp32_llm.ino` | ESP32-S3 firmware: flash partition mmap, PSRAM alloc, dual-core int8 head split, boot diagnostics | `engine/llm-firmware` (Phase 3, then Phase 6) |
| `firmware/esp32_llm/display.h` | OLED (SH1106/SSD1306) / TFT (ST7789) demo screen driver | `engine/llm-display` (Phase 4, optional) |
| `firmware/esp32_llm/partitions.csv` | Flash partition table (bootloader/app/model layout) | reused as-is by `engine/llm-firmware`'s build |
| `firmware/bandwidth_bench/bandwidth_bench.ino` | Raw cycle-counter PSRAM/SRAM/flash bandwidth benchmark | standalone port (Phase 5, optional) |

Not copied here (out of scope, stays entirely in Python, upstream only, see
`/MIGRATION_PLAN.md`): the training/quantization/export pipeline itself
(`src/*.py`, `data/prepare.py` and friends) and everything under
`data/`, `runs/` in `slvDev/esp32-ai`. `chat.py` and its tokenizer JSON are
also not here — this folder stays strictly C/C++/Arduino. That interactive
demo layer lives in [`demo/`](../demo/) instead, one level up, which builds
against `esp32-llm-lab/gen_prompt.c` here but keeps its own Python/shell
files out of this folder entirely.

## Removed: `esp32-llm-lab/flash.bin`

A prebuilt 16 MB QEMU flash image used to live here. It's been dropped —
its only consumer, `run_qemu.sh`, was never actually checked into this
repo, so the file was dead weight with nothing exercising it. If you want
the QEMU device-emulator path back, regenerate `flash.bin` and `run_qemu.sh`
from `slvDev/esp32-ai` rather than re-adding a stale binary here.
