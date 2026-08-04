# llm-display — on-device demo screen (Phase 4, optional, deferred)

Rust port of `reference-c/firmware/esp32_llm/display.h`: the OLED
(SH1106/SSD1306, I2C) / TFT (ST7789, SPI) demo screen. Toggled by
`USE_DISPLAY` in the C code — genuinely optional, the model generates text
over serial with or without it.

Not started. Lowest priority: do this only after llm-firmware's Phase 3
exit gate (boot + generate text + match the recorded tok/s) is met. Likely
crates: `sh1106` / `ssd1306` / `st7789` (embedded-graphics ecosystem) —
their exact compatibility with whichever HAL llm-firmware lands on
(esp-idf-hal in Phase 3, or esp-hal in Phase 6) needs to be checked when
this phase actually starts; not verified yet.
