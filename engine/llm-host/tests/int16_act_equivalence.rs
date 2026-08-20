//! Gate for holding the head's quantized activation as `i16` instead of `i8`.
//!
//! WHY. Disassembling the shipped firmware (`scripts/disasm_head.sh`) showed
//! the head's inner loop spending twelve instructions on two
//! multiply-accumulates, and two of the twelve on nothing:
//!
//! ```text
//! l8ui  a10, a9, 0      ; weight byte
//! srli  a7,  a10, 4     ; high nibble
//! l8ui  a15, a3, 0      ; activation
//! sext  a15, a15, 7     ; <-- pure overhead
//! mull  a15, a15, a7
//! and   a10, a10, a4    ; low nibble
//! addi  a7,  a3, -1
//! l8ui  a7,  a7, 0      ; activation
//! sext  a7,  a7, 7      ; <-- pure overhead
//! mull  a10, a10, a7
//! add   a8,  a10, a8
//! add   a8,  a8, a15
//! ```
//!
//! The LX7 has no signed byte load. `l8ui` zero-extends, so every `i8`
//! activation read costs a load *and* a sign-extend — on 2.43M elements per
//! token. `l16si` is a signed 16-bit load, so widening the activation folds
//! the extension into the load and deletes the instruction. The scratch grows
//! by 96 bytes, read once per token.
//!
//! This matters more than it looks. The head's arithmetic is ~48.8 ms/token
//! against the C reference's ~30.4 for the same algorithm over the same data —
//! and C moves *twice* the bytes and is still faster overall. A 60% gap on
//! scalar integer multiply-accumulate is a code-generation difference, and
//! this is the first concrete piece of it that the disassembly named.
//!
//! WHAT THIS ASSERTS. That `quantize_activations_i16` produces exactly the
//! values `quantize_activations` does, and that `dot_int4_int16` returns
//! exactly what `dot_int4_int8` returns, on every row of the shipped model.
//! Not approximately — the quantizer clamps to `-127..=127`, representable in
//! both widths, and `i16 as i32` equals `i8 as i32` throughout.
//!
//! WHAT IT DOES NOT ANSWER: whether the LX7 actually emits `l16si` and drops
//! the `sext`. That is a device question, answered by re-running
//! `scripts/disasm_head.sh` and looking, and then by the stopwatch.

use llm_core::{
    activation_sum, activation_sum_i16, dot_int4_int16, dot_int4_int8, quantize_activations,
    quantize_activations_i16, Model,
};
use llm_host::manifest;

fn model_bytes() -> Vec<u8> {
    std::fs::read(manifest::default_model().bin_path()).expect("model.bin")
}

#[test]
fn i16_activations_give_bit_identical_head_logits() {
    let bytes = model_bytes();
    let model = Model::load(&bytes).expect("load");
    let head = model.tok_emb();
    let cols = head.cols;
    assert_eq!(head.n_groups, 1, "this path assumes one scale per row");

    // The activation shapes the other head tests use, including the extremes
    // that stress `quantize_activations`' max-abs scaling and the all-zero
    // case that drives it to its `1e-8` floor.
    let cases: [&dyn Fn(usize) -> f32; 5] = [
        &|j| (j as f32 * 0.37).sin(),
        &|j| if j % 2 == 0 { 1.0 } else { -1.0 },
        &|j| (j as f32 - 48.0) * 1e-4,
        &|_| 0.0,
        // Straddles the clamp: values that quantize to exactly +/-127.
        &|j| if j % 3 == 0 { 1e6 } else { -1e6 },
    ];

    for (ci, f) in cases.iter().enumerate() {
        let x: Vec<f32> = (0..cols).map(|j| f(j)).collect();

        let mut a8 = vec![0i8; cols];
        let mut a16 = vec![0i16; cols];
        let s8 = quantize_activations(&x, &mut a8);
        let s16 = quantize_activations_i16(&x, &mut a16);

        assert_eq!(
            s8.to_bits(),
            s16.to_bits(),
            "case {ci}: activation scale differs"
        );
        for j in 0..cols {
            assert_eq!(
                a8[j] as i32, a16[j] as i32,
                "case {ci} element {j}: quantized activation differs"
            );
        }

        let sum8 = activation_sum(&a8);
        let sum16 = activation_sum_i16(&a16);
        assert_eq!(sum8, sum16, "case {ci}: activation sum differs");

        for r in 0..head.rows {
            let (codes, scale) = head.packed_row(r);
            let d8 = dot_int4_int8(codes, &a8, sum8);
            let d16 = dot_int4_int16(codes, &a16, sum16);
            assert_eq!(d8, d16, "case {ci} row {r}: int32 dot differs");
            // What `head.rs::rows_range` writes, in the same multiply order.
            let v8 = d8 as f32 * scale * s8;
            let v16 = d16 as f32 * scale * s16;
            assert_eq!(
                v8.to_bits(),
                v16.to_bits(),
                "case {ci} row {r}: logit differs"
            );
        }
    }
}

/// The cost side: 96 more bytes of scratch, once per token.
#[test]
fn i16_activations_cost_one_extra_byte_per_element() {
    let bytes = model_bytes();
    let model = Model::load(&bytes).expect("load");
    let cols = model.tok_emb().cols;
    // `head.rs` holds a fixed [_; MAX_HEAD_COLS] scratch, not a `cols`-sized one.
    const MAX_HEAD_COLS: usize = 128;
    assert!(cols <= MAX_HEAD_COLS);
    eprintln!(
        "head activation scratch: {} B -> {} B (+{} B, read once per token, \
         against 2.43M sext instructions removed)",
        MAX_HEAD_COLS,
        MAX_HEAD_COLS * 2,
        MAX_HEAD_COLS
    );
}
