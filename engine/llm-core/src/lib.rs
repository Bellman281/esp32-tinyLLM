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
pub mod digest;
pub mod model;
pub mod ops;
pub mod profile;
pub mod rope;
pub mod scratch;
pub mod tensor;

pub use attention::HeadsJob;
pub use digest::TokenDigest;
pub use model::{
    llm_forward, llm_forward_profiled, llm_forward_profiled_with_matvec_override,
    llm_forward_profiled_with_overrides, llm_forward_with_attn_override,
    llm_forward_with_head_override, llm_forward_with_matvec_override, Layer, Model, MAX_LAYERS,
};
pub use profile::Profile;
pub use scratch::{sizes as scratch_sizes, Scratch, ScratchSizes};
pub use tensor::{activation_sum, dot_int4_int8, quantize_activations, Cfg, FVec, LoadError, QT};
