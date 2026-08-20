//! Scratch buffers for one decode step. Ports C's `Scratch` struct.
//!
//! The C struct holds raw pointers the caller mallocs (host) or PSRAM-allocs
//! (device). This port holds borrowed `&mut [f32]` slices instead -- same
//! caller-owns-the-memory design, but llm-core itself never allocates, so
//! it stays no_std/allocator-free. `llm_host::alloc_scratch` (std, Vec-backed)
//! is the allocator for the host binaries; llm-firmware will do the PSRAM
//! equivalent later.
//!
//! Two fields don't exist in the C struct: `rope_cos`/`rope_sin`, which the
//! C code stores by reusing the (by then dead) `trow` buffer. See
//! `rope.rs` for why this port uses dedicated fields instead.

use crate::tensor::Cfg;

pub struct Scratch<'a> {
    pub x: &'a mut [f32],        // [dim]
    pub h: &'a mut [f32],        // [max(ffn, dim)]
    pub qkv: &'a mut [f32],      // [3*dim]
    pub att: &'a mut [f32],      // [dim]
    pub g1: &'a mut [f32],       // [ffn]
    pub g2: &'a mut [f32],       // [max(ple_dim, ffn)]
    pub ple: &'a mut [f32],      // [n_layers*ple_dim]
    pub tmp_p: &'a mut [f32],    // [n_layers*ple_dim]
    pub trow: &'a mut [f32],     // [n_layers*ple_dim]
    pub rope_cos: &'a mut [f32], // [head_dim/2]
    pub rope_sin: &'a mut [f32], // [head_dim/2]
    pub logits: &'a mut [f32],   // [vocab]
    pub scores: &'a mut [f32],   // [seq_len]
    pub kcache: &'a mut [f32],   // [n_layers*seq_len*dim]
    pub vcache: &'a mut [f32],   // [n_layers*seq_len*dim]
    pub iq: &'a mut [i8],        // [max(dim, ffn)] -- int8-activation quant scratch
}

/// Required length of each `Scratch` buffer for a given config. Pure
/// arithmetic, no allocation -- callers use this to size their own buffers
/// (`Vec<f32>` on the host, a PSRAM region on the chip).
#[derive(Debug, Clone, Copy)]
pub struct ScratchSizes {
    pub x: usize,
    pub h: usize,
    pub qkv: usize,
    pub att: usize,
    pub g1: usize,
    pub g2: usize,
    pub ple: usize,
    pub tmp_p: usize,
    pub trow: usize,
    pub rope_cos: usize,
    pub rope_sin: usize,
    pub logits: usize,
    pub scores: usize,
    pub kcache: usize,
    pub vcache: usize,
    pub iq: usize,
}

pub fn sizes(cfg: &Cfg) -> ScratchSizes {
    let lp = cfg.n_layers * cfg.ple_dim;
    let half_head = cfg.head_dim() / 2;
    ScratchSizes {
        x: cfg.dim,
        h: core::cmp::max(cfg.ffn, cfg.dim),
        qkv: 3 * cfg.dim,
        att: cfg.dim,
        g1: cfg.ffn,
        g2: core::cmp::max(cfg.ple_dim, cfg.ffn),
        ple: lp,
        tmp_p: lp,
        trow: lp,
        rope_cos: half_head,
        rope_sin: half_head,
        logits: cfg.out_vocab,
        scores: cfg.seq_len,
        kcache: cfg.n_layers * cfg.seq_len * cfg.dim,
        vcache: cfg.n_layers * cfg.seq_len * cfg.dim,
        // Must cover every tensor's `cols`, since matvec_int8_activations
        // indexes iq[j] up to self.cols. ple_proj is bound as (rows=dim,
        // cols=ple_dim), so ple_dim can exceed both dim and ffn (as it
        // does for the 14.9MB model: ple_dim=128 > dim=96, ffn=66) --
        // omitting it here caused a real out-of-bounds panic.
        iq: core::cmp::max(cfg.dim, core::cmp::max(cfg.ffn, cfg.ple_dim)),
    }
}
