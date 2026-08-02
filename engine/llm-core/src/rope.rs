//! Rotary position embedding. Ports the RoPE frequency computation and
//! split-half rotation inlined in `llm_forward`.
//!
//! Structural note: the C code stores `rope_c`/`rope_s` by reusing the
//! `trow` scratch buffer (dead after the PLE input is built) purely to save
//! a small scratch allocation. This port gives them their own dedicated
//! `Scratch` fields instead (see `scratch.rs`) -- same numeric result,
//! clearer ownership, and the memory cost is negligible (head_dim/2 floats).

/// cos/sin at `pos` for every frequency pair `i` in `0..head_dim/2`.
/// Ports the `rope_c[i] = cosf(pos*freq); rope_s[i] = sinf(pos*freq)` loop.
pub fn compute(pos: usize, head_dim: usize, theta: f32, cos_out: &mut [f32], sin_out: &mut [f32]) {
    let half = head_dim / 2;
    for i in 0..half {
        let freq = libm::powf(theta, -2.0 * i as f32 / head_dim as f32);
        let angle = pos as f32 * freq;
        cos_out[i] = libm::cosf(angle);
        sin_out[i] = libm::sinf(angle);
    }
}

/// In-place split-half rotation of one head's q or k vector (len ==
/// head_dim). Ports the `qh[i] = q1*c - q2*sn; qh[i+Dh/2] = q2*c + q1*sn`
/// pair, applied identically to q and k in the caller.
pub fn apply(vec: &mut [f32], cos: &[f32], sin: &[f32], head_dim: usize) {
    let half = head_dim / 2;
    for i in 0..half {
        let c = cos[i];
        let s = sin[i];
        let v1 = vec[i];
        let v2 = vec[i + half];
        vec[i] = v1 * c - v2 * s;
        vec[i + half] = v2 * c + v1 * s;
    }
}
