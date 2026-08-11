//! Shared host-side support code for the gen-prompt / host-generate / ppl
//! binaries and the golden-logit parity test. Not part of the portable
//! model math -- see `llm_core` for that.

pub mod manifest;
pub mod support;
pub mod vocab;
