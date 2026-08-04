# llm-firmware — the on-device binary (not scaffolded yet, by design)

This crate isn't created yet because its Cargo.toml and target config depend
on a decision made per-phase, not up front:

## Phase 3 (first): esp-idf-hal, std, FreeRTOS underneath
Closest 1:1 port of `reference-c/firmware/esp32_llm/esp32_llm.ino`. Depends on
`esp-idf-hal` / `esp-idf-svc` / `esp-idf-sys`, targets
`xtensa-esp32s3-espidf`, and needs the ESP-IDF C SDK installed as a build
dependency (via `espup`). Maps onto the C code as:

| esp32_llm.ino | Rust (Phase 3) |
|---|---|
| `esp_partition_find_first` / `esp_partition_mmap` | `esp-idf-svc::partition` |
| `heap_caps_malloc(MALLOC_CAP_SPIRAM)` | `esp-idf-sys` heap_caps bindings |
| `xTaskCreatePinnedToCore` + `ulTaskNotifyTake`/`xTaskNotifyGive` (dual-core int8 head split) | `esp-idf-hal`/`esp-idf-sys` FreeRTOS task bindings, same worker-on-core-0 / decode-on-core-1 design |
| `Serial.begin/write` | `esp-idf-hal` UART / USB-CDC driver |
| `esp_timer_get_time` (per-stage profiling) | `esp-idf-svc` timer bindings |

Exit gate: flash it, compare boot diagnostics + measured tok/s against
`reference-c/firmware/esp32_llm/README.md` (102.9ms/step, 9.72 tok/s
compute-only). Should match within noise.

## Phase 6 (later, separate): re-platform onto esp-hal, no_std, no FreeRTOS
Once Phase 3 is verified working, swap the platform layer only — llm-core
does not change — for `esp-hal`: no FreeRTOS, no ESP-IDF C SDK anywhere in
the build. Dual-core split becomes esp-hal's core-1 closure spawn instead of
a FreeRTOS task. Same exit gate as Phase 3.

Nothing is written here yet on purpose: committing to esp-idf-hal's exact
dependency versions before Phase 3 actually starts would just go stale.
