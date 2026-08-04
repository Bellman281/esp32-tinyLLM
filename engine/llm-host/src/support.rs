//! Owns the `Vec`-backed scratch buffers a CLI/test needs and lends them to
//! `llm_core::Scratch` as borrowed slices. `llm-core` itself never
//! allocates (see scratch.rs's doc comment there) -- this is the std-side
//! allocator, the equivalent of the C host tools' `malloc()` calls.

use llm_core::{Cfg, Scratch};

pub struct ScratchOwned {
    x: Vec<f32>,
    h: Vec<f32>,
    qkv: Vec<f32>,
    att: Vec<f32>,
    g1: Vec<f32>,
    g2: Vec<f32>,
    ple: Vec<f32>,
    tmp_p: Vec<f32>,
    trow: Vec<f32>,
    rope_cos: Vec<f32>,
    rope_sin: Vec<f32>,
    logits: Vec<f32>,
    scores: Vec<f32>,
    kcache: Vec<f32>,
    vcache: Vec<f32>,
    iq: Vec<i8>,
}

impl ScratchOwned {
    pub fn new(cfg: &Cfg) -> Self {
        let sz = llm_core::scratch_sizes(cfg);
        ScratchOwned {
            x: vec![0.0; sz.x],
            h: vec![0.0; sz.h],
            qkv: vec![0.0; sz.qkv],
            att: vec![0.0; sz.att],
            g1: vec![0.0; sz.g1],
            g2: vec![0.0; sz.g2],
            ple: vec![0.0; sz.ple],
            tmp_p: vec![0.0; sz.tmp_p],
            trow: vec![0.0; sz.trow],
            rope_cos: vec![0.0; sz.rope_cos],
            rope_sin: vec![0.0; sz.rope_sin],
            logits: vec![0.0; sz.logits],
            scores: vec![0.0; sz.scores],
            kcache: vec![0.0; sz.kcache],
            vcache: vec![0.0; sz.vcache],
            iq: vec![0; sz.iq],
        }
    }

    pub fn as_scratch(&mut self) -> Scratch<'_> {
        Scratch {
            x: &mut self.x,
            h: &mut self.h,
            qkv: &mut self.qkv,
            att: &mut self.att,
            g1: &mut self.g1,
            g2: &mut self.g2,
            ple: &mut self.ple,
            tmp_p: &mut self.tmp_p,
            trow: &mut self.trow,
            rope_cos: &mut self.rope_cos,
            rope_sin: &mut self.rope_sin,
            logits: &mut self.logits,
            scores: &mut self.scores,
            kcache: &mut self.kcache,
            vcache: &mut self.vcache,
            iq: &mut self.iq,
        }
    }
}
