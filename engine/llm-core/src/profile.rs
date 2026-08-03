//! Optional per-stage timing, ported from `llm.h`'s `#ifdef LLM_PROFILE`
//! block (`s->profile.{input,attn,ffn,ple,head}_us`, `llm_profile_reset`).
//!
//! `llm-core` is `no_std` and has no timer of its own -- neither the host
//! build nor the firmware build can assume the same clock source (host
//! tests don't care about wall-clock at all; the firmware uses
//! `esp_timer_get_time`). So unlike the C macro (`LLM_PROFILE_NOW()`,
//! defined by whichever `.c`/`.ino` file includes `llm.h`), this port takes
//! the timestamp source as a caller-supplied closure (`now: &mut dyn FnMut()
//! -> u64`, one call per boundary) instead of baking in a specific clock.
//!
//! Deliberately NOT stored as a field on `Scratch` the way C embeds it in
//! the struct (`#ifdef LLM_PROFILE` inside `Scratch`) -- keeping it a
//! separate value the caller owns means `Scratch`'s layout and every
//! existing call site (`llm_forward`, `llm_forward_with_head_override`, and
//! every test in `llm-host` that constructs a `Scratch`) is completely
//! unaffected. Opting in is calling `llm_forward_profiled` instead of the
//! plain functions; opting out costs nothing -- `llm_forward_impl` takes an
//! `Option`, and `None` skips every timing call.
#[derive(Debug, Clone, Copy, Default)]
pub struct Profile {
    pub input_us: u64,
    pub attn_us: u64,
    pub ffn_us: u64,
    pub ple_us: u64,
    pub head_us: u64,
    pub calls: u32,
}

impl Profile {
    /// Ports `llm_profile_reset` -- zero everything, keep accumulating from
    /// here. C calls this once, right after the prompt-priming loop, so the
    /// printed averages cover only the timed generation steps, not the
    /// (differently-shaped, KV-cache-empty-at-first) priming calls.
    pub fn reset(&mut self) {
        *self = Profile::default();
    }

    /// `stage_us / (calls * 1000)` -- matches `llm.h`'s own divisor exactly
    /// (`float n = (float)s.profile.calls * 1000.f;`): converts an
    /// accumulated microsecond total across `calls` tokens into an average
    /// milliseconds-*per-token* figure for that one stage.
    pub fn ms_per_token(&self) -> Option<[f32; 5]> {
        if self.calls == 0 {
            return None;
        }
        let n = self.calls as f32 * 1000.0;
        Some([
            self.input_us as f32 / n,
            self.attn_us as f32 / n,
            self.ffn_us as f32 / n,
            self.ple_us as f32 / n,
            self.head_us as f32 / n,
        ])
    }
}
