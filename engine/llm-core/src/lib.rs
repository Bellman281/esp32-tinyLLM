//! Portable core inference engine for the ESP32 PLE-TinyLM.
//!
//! Rust port of `reference-c/firmware/common/llm.h`. Owns every tensor
//! operation the model needs and nothing about where it runs -- no ESP32,
//! no FreeRTOS, no esp-hal/esp-idf dependency. `llm-host` (and later
//! `llm-firmware`) depend on this crate and add only platform glue.
//!
//! no_std: uses `core` + `libm` (portable math) + `half` (fp16 decode) only.

#![no_std]

pub mod attention;
pub mod model;
pub mod ops;
pub mod rope;
pub mod scratch;
pub mod tensor;

pub use model::{llm_forward, Layer, Model, MAX_LAYERS};
pub use scratch::{sizes as scratch_sizes, Scratch, ScratchSizes};
pub use tensor::{Cfg, LoadError, FVec, QT};
