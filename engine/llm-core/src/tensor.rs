//! Model config, quantized-tensor view, and the binary-cursor binder.
//!
//! Ports `struct Cfg`, `struct QT`, `bind_q`, `bind_f` from
//! reference-c/firmware/common/llm.h. Everything here borrows from the raw
//! model.bin byte buffer instead of copying it, same as the C code's
//! bind-in-place pointers -- just as `&'a [u8]` slices instead of raw
//! pointers, so out-of-range reads are caught instead of silently
//! corrupting memory.

const MAGIC: u32 = 0x504C_4531;

/// The exporter's current header: `b"PLE\0"` little-endian. Same tensor
/// stream as `MAGIC`, different preamble -- it carries a format version, a
/// self-describing header length, a flags word, and an `output_vocab`
/// distinct from the embedding's row count. See `Model::load`.
const MAGIC_V2: u32 = 0x0045_4C50;

/// `MAGIC_V2` flags, bit 0: the output head *is* the token embedding and no
/// separate head tensor follows. The only layout this port implements.
const FLAG_TIED_HEAD: u32 = 1 << 0;

/// Highest `MAGIC_V2` format version this reader understands.
const MAX_FORMAT_VERSION: u32 = 1;

/// Round-half-to-even, matching C's `lrintf` under the IEEE754 default
/// rounding mode (`FE_TONEAREST`). `core`/`libm` don't expose this for
/// no_std (`f32::round_ties_even` is std-only), so it's implemented by
/// hand from `libm::floorf`.
#[inline]
fn round_ties_even(x: f32) -> f32 {
    let floor = libm::floorf(x);
    let diff = x - floor;
    if diff < 0.5 {
        floor
    } else if diff > 0.5 {
        floor + 1.0
    } else {
        // Exactly .5: round to the even neighbor.
        if (floor as i64) & 1 == 0 {
            floor
        } else {
            floor + 1.0
        }
    }
}

/// Quantizes `x[..iq.len()]` to int8 by max-abs scaling. Ports the C
/// reference's free-standing `quantize_act` in llm.h, which both
/// `matvec_q8` (host/lib) and, on-device, `head_matvec_int8`
/// (`firmware/esp32_llm/esp32_llm.ino`'s dual-core staged output head)
/// call -- pulled out to a public function here for the same reason: two
/// call sites, one quantization implementation, so `llm-firmware`'s head
/// module gets the exact int8-activation numerics `llm-host`'s ppl parity
/// tests already verified against the C reference, instead of a
/// hand-rolled second copy that could silently drift (e.g. by using
/// `libm::roundf`'s round-half-away-from-zero instead of `lrintf`'s
/// round-to-nearest-even -- see the tie-breaking note below).
///
/// Returns the dequant scale (`x_scale` in C): `x[j] ~= iq[j] as f32 * scale`.
pub fn quantize_activations(x: &[f32], iq: &mut [i8]) -> f32 {
    let n = iq.len();
    debug_assert!(x.len() >= n);
    let mut xmax = 1e-8f32;
    for &v in &x[..n] {
        let a = libm::fabsf(v);
        if a > xmax {
            xmax = a;
        }
    }
    let inv = 127.0f32 / xmax;
    for j in 0..n {
        // C uses `(int)lrintf(...)`, which rounds with the CPU's current
        // rounding mode -- round-to-nearest-even under the IEEE754
        // default, NOT round-half-away-from-zero. `libm::roundf` is the
        // latter, so it silently disagrees with C on exact .5 ties; over a
        // few thousand activations that's enough to shift the
        // int8-activation perplexity in the last printed digit.
        // round_ties_even() below reproduces lrintf's tie-breaking.
        let q = round_ties_even(x[j] * inv) as i32;
        iq[j] = q.clamp(-127, 127) as i8;
    }
    xmax / 127.0f32
}

/// Signed int4 code (`nibble - 8`, i.e. -8.0..=7.0) as `f32`, indexed by the
/// raw 0..=15 nibble.
///
/// PERFORMANCE NOTE: every fp32 unpack site below used to compute this as
/// `(nibble as i32 - 8) as f32` -- an integer subtract plus an
/// integer-to-float conversion, per element, on loops that run
/// `rows * cols` times per token (3.1M times for the output head alone on
/// the shipped model). The table replaces both with a single load from 64
/// bytes that stay resident in L1 (and, on the ESP32-S3, in the same cache
/// the model already streams through). Bit-identical by construction: every
/// value -8..=7 is exactly representable in `f32`, so the table entry and
/// the arithmetic it replaces are the same float, not merely close.
///
/// Measured on an x86-64 host against the previous arithmetic form, whole
/// tensors, output asserted bit-for-bit identical: output head 4.04ms ->
/// 3.07ms per call (-24%), `qkv` 35.2us -> 27.0us, `gate` 7.9us -> 6.2us,
/// `ple_proj` 15.4us -> 12.8us. NOT measured on hardware -- the Xtensa
/// backend and the LX7's cache are a separate question, and BENCHMARKING.md
/// is the place that answer belongs once a board has run it.
const NIBBLE_F32: [f32; 16] = [
    -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
];

/// Integer counterpart of `NIBBLE_F32`, for the int8-activation path and
/// `unpack_row_int8` (which want the code as an integer, not a float).
/// Saves the `- 8` per element; same values, so bit-identical.
const NIBBLE_I8: [i8; 16] = [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadError {
    TooShort,
    BadMagic,
    TooManyLayers,
    /// A `PLE\0` file declaring a format version newer than this reader.
    UnsupportedVersion(u32),
    /// A `PLE\0` file whose head is a separate tensor rather than a view of
    /// the token embedding. Its weights are laid out differently and this
    /// port has no model to verify such a file against, so it is refused
    /// rather than guessed at.
    UntiedHead,
    /// `output_vocab` outside `1..=input_vocab`. The tied head is the first
    /// `output_vocab` rows of the embedding, so a larger value would read
    /// past that tensor.
    BadOutputVocab,
}

/// Mirrors C's `Cfg`. All dims stored as `usize` (C uses `int`, always
/// non-negative in this file format).
#[derive(Debug, Clone, Copy)]
pub struct Cfg {
    /// Rows in the token embedding and the PLE table.
    pub vocab: usize,
    /// Logits actually scored: the tokenizer's real entry count. Equal to
    /// `vocab` for `PLE1` files, which cannot express the distinction; a
    /// `PLE\0` file may declare it smaller, because the embedding is padded
    /// above the tokenizer and those rows can never be emitted.
    pub out_vocab: usize,
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub ffn: usize,
    pub ple_dim: usize,
    pub seq_len: usize,
    pub group: usize,
    pub rope_theta: f32,
}

impl Cfg {
    pub fn head_dim(&self) -> usize {
        self.dim / self.n_heads
    }
}

/// A group-wise int4 tensor viewed in place: ragged packed nibbles
/// (row-aligned to a byte) + fp16 group scales. Per-tensor group, no
/// padding. Ports C's `QT`.
#[derive(Clone, Copy)]
pub struct QT<'a> {
    codes: &'a [u8],  // rows*row_bytes, nibble = value+8
    scales: &'a [u8], // rows*n_groups fp16, raw little-endian bytes
    pub rows: usize,
    pub cols: usize,
    pub group: usize,
    pub n_groups: usize,
    pub row_bytes: usize,
}

impl<'a> QT<'a> {
    #[inline]
    fn scale(&self, r: usize, gi: usize) -> f32 {
        let off = (r * self.n_groups + gi) * 2;
        let bits = u16::from_le_bytes([self.scales[off], self.scales[off + 1]]);
        half::f16::from_bits(bits).to_f32()
    }

    /// Dequantize row `r` into `out` (len == self.cols). Ports `deq_row`.
    pub fn deq_row(&self, r: usize, out: &mut [f32]) {
        debug_assert!(out.len() >= self.cols);
        let row = &self.codes[r * self.row_bytes..(r + 1) * self.row_bytes];
        // Two elements per packed byte, the same shape `matvec_range` uses
        // and for the same reason -- this went through a `code(r, j)`
        // helper once per element, re-deriving `r * row_bytes`, re-loading
        // the same byte for each nibble of a pair, and branching on `j & 1`
        // every step. Same values, same order, bit-identical output.
        for gi in 0..self.n_groups {
            let begin = gi * self.group;
            let end = core::cmp::min(begin + self.group, self.cols);
            let scale = self.scale(r, gi);
            let mut j = begin;
            if j & 1 == 1 && j < end {
                out[j] = NIBBLE_F32[(row[j >> 1] >> 4) as usize] * scale;
                j += 1;
            }
            let pairs = (end - j) / 2;
            if pairs > 0 {
                let bytes = &row[j >> 1..(j >> 1) + pairs];
                let outs = &mut out[j..j + 2 * pairs];
                for (b, op) in bytes.iter().zip(outs.chunks_exact_mut(2)) {
                    op[0] = NIBBLE_F32[(b & 0xF) as usize] * scale;
                    op[1] = NIBBLE_F32[(b >> 4) as usize] * scale;
                }
                j += 2 * pairs;
            }
            if j < end {
                let byte = row[j >> 1];
                let nib = if j & 1 == 1 { byte >> 4 } else { byte & 0xF };
                out[j] = NIBBLE_F32[nib as usize] * scale;
            }
        }
    }

    /// y[row_begin..row_end] = W * x, fp32-dequantized-weight path. Ports
    /// `matvec_q_range` / `matvec_q` -- including its 2-elements-per-byte
    /// steady-state loop, not just its math.
    ///
    /// PERFORMANCE NOTE: the original port of this function called
    /// `self.code(r, j)` once per scalar element instead of matching this
    /// structure. On real ESP32-S3 hardware that measured ~2-2.4x slower
    /// per token than the C reference at every stage that calls this
    /// function (attention, FFN, PLE, input embedding -- everything except
    /// the head, which has its own int8-staged path in `llm-firmware`).
    /// Root cause, per-element `code(r, j)`: (a) recomputed `r *
    /// row_bytes` on every single `j` instead of once per row, (b)
    /// branched on `j & 1` every element instead of only at group
    /// boundaries, and (c) loaded the same packed byte twice for every
    /// adjacent (even, odd) pair instead of once. This version fixes all
    /// three by porting the C loop's actual shape, not just its per-element
    /// arithmetic.
    ///
    /// Same summation order as before (left-to-right, one term at a time,
    /// via `group_acc +=`) -- this model's tensors all have even
    /// `cols`/`group` (verified against `model.bin`'s header), so the
    /// steady-state 2-at-a-time loop below covers `begin..end` in the same
    /// order the old one-at-a-time loop did; the odd-boundary branches
    /// exist only for a hypothetical odd-cols model, exactly mirroring why
    /// the C reference has them. Verified bit-for-bit unchanged output via
    /// the existing golden/cli-parity/ppl-parity tests (byte-for-byte
    /// greedy-decode CLI parity in particular would catch even a single
    /// bit of summation-order drift) -- this is a pure performance
    /// rewrite, not a numerics change.
    ///
    /// NOTE ON THIS FUNCTION vs. `matvec_range_chunked` below: this is the
    /// only fp32-path matvec used by any verified/shipped code. It is NOT
    /// currently SIMD-accelerated on-device (unlike the head, which has
    /// its own int8-staged path in `llm-firmware::head`) -- see
    /// `matvec_range_chunked`'s doc comment for why a real SIMD version of
    /// *this* function is a harder, riskier change than the head's, and
    /// what has to be measured before trusting one.
    /// `out[i] = dot(row `row_begin + i`, x)` for `i` in `0..out.len()`.
    ///
    /// The same arithmetic as `matvec_range`, addressed relative to `out`
    /// instead of absolutely. That difference is what makes a row split safe
    /// in ordinary Rust: two cores computing disjoint halves of one output can
    /// each hold a real `&mut [f32]` from `split_at_mut`, rather than both
    /// holding a `&mut` to the whole buffer and promising not to overlap.
    /// `llm-firmware`'s dual-core matvec is the caller; `matvec_range`
    /// delegates here so there is one loop, not two.
    ///
    /// BIT-EXACT against `matvec_range` by construction -- same rows, same
    /// group order, same left-to-right accumulation within each row, only the
    /// destination index differs. `llm-host/tests/matvec_split.rs` asserts it
    /// over every split point of a real tensor.
    pub fn matvec_rows_into(&self, x: &[f32], out: &mut [f32], row_begin: usize) {
        debug_assert!(x.len() >= self.cols);
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = self.dot_row(x, row_begin + i);
        }
    }

    /// One row's fp32 dot product. Factored out of `matvec_range` so the
    /// row-relative and row-absolute entry points cannot drift apart.
    #[inline]
    fn dot_row(&self, x: &[f32], r: usize) -> f32 {
        let row = &self.codes[r * self.row_bytes..(r + 1) * self.row_bytes];
        let mut acc = 0f32;
        for gi in 0..self.n_groups {
            let begin = gi * self.group;
            let end = core::cmp::min(begin + self.group, self.cols);
            let scale = self.scale(r, gi);
            let mut group_acc = 0f32;
            let mut j = begin;
            if j & 1 == 1 && j < end {
                group_acc += NIBBLE_F32[(row[j >> 1] >> 4) as usize] * x[j];
                j += 1;
            }
            let pairs = (end - j) / 2;
            if pairs > 0 {
                let bytes = &row[j >> 1..(j >> 1) + pairs];
                let xs = &x[j..j + 2 * pairs];
                for (b, xp) in bytes.iter().zip(xs.chunks_exact(2)) {
                    group_acc += NIBBLE_F32[(b & 0xF) as usize] * xp[0];
                    group_acc += NIBBLE_F32[(b >> 4) as usize] * xp[1];
                }
                j += 2 * pairs;
            }
            if j < end {
                let byte = row[j >> 1];
                let nib = if j & 1 == 1 { byte >> 4 } else { byte & 0xF };
                group_acc += NIBBLE_F32[nib as usize] * x[j];
            }
            acc += group_acc * scale;
        }
        acc
    }

    pub fn matvec_range(&self, x: &[f32], y: &mut [f32], row_begin: usize, row_end: usize) {
        debug_assert!(x.len() >= self.cols);
        for r in row_begin..row_end {
            y[r] = self.dot_row(x, r);
        }
    }

    pub fn matvec(&self, x: &[f32], y: &mut [f32]) {
        let rows = self.rows;
        self.matvec_range(x, y, 0, rows);
    }

    /// Same math as `matvec_range`, but reduces each group's dot product
    /// across `LANES` independent parallel accumulators (`j`'s value goes
    /// into lane `(j - begin) % LANES`) instead of one strictly
    /// sequential running sum, then combines the `LANES` partial sums at
    /// the end. This is the reduction *shape* a real SIMD dot-product
    /// instruction uses (process `LANES` elements per step into `LANES`
    /// registers, horizontal-sum once at the end) -- NOT a claim that any
    /// particular `esp-dsp` function reduces in exactly this order, which
    /// is unverified and can't be confirmed without disassembling it.
    ///
    /// Why this function exists at all: `matvec_range`'s accumulation is
    /// `f32`, and float addition is not associative -- reordering it (as
    /// any real SIMD dot product necessarily does) can change the result
    /// in the last bit or two, unlike the head's int8 path where integer
    /// addition's associativity makes reordering provably lossless. That
    /// means a real on-device SIMD version of this function is NOT
    /// expected to match `matvec_range`'s output bit-for-bit -- it needs
    /// its own tolerance, not the exact-match bar `golden.rs`/
    /// `cli_parity.rs` hold the scalar path to. This function is how that
    /// tolerance gets measured, on the host, before any FFI/hardware is
    /// involved: `llm-host/tests/matvec_simd_tolerance.rs` runs this at a
    /// couple of `LANES` values against real validation data and checks
    /// how far the resulting perplexity drifts from the exact
    /// `matvec_range` reference, the same style `ppl_parity.rs` already
    /// uses to bound the (also inherently reorder-sensitive)
    /// int8-activations path.
    ///
    /// Not used by any shipped/default code path -- test-only, and
    /// intentionally not perf-tuned (plain per-element unpack, no
    /// byte-pair-unroll) since clarity matters more than speed for a
    /// function whose only job is measuring a numeric property.
    pub fn matvec_range_chunked<const LANES: usize>(
        &self,
        x: &[f32],
        y: &mut [f32],
        row_begin: usize,
        row_end: usize,
    ) {
        debug_assert!(x.len() >= self.cols);
        debug_assert!(LANES > 0, "LANES must be nonzero");
        for r in row_begin..row_end {
            let row = &self.codes[r * self.row_bytes..(r + 1) * self.row_bytes];
            let mut acc = 0f32;
            for gi in 0..self.n_groups {
                let begin = gi * self.group;
                let end = core::cmp::min(begin + self.group, self.cols);
                let scale = self.scale(r, gi);
                let mut lanes = [0f32; LANES];
                for j in begin..end {
                    let byte = row[j >> 1];
                    let nib = if j & 1 == 1 { byte >> 4 } else { byte & 0xF };
                    lanes[(j - begin) % LANES] += NIBBLE_F32[nib as usize] * x[j];
                }
                let mut group_acc = 0f32;
                for l in lanes.iter() {
                    group_acc += *l;
                }
                acc += group_acc * scale;
            }
            y[r] = acc;
        }
    }

    /// Unpacks the int4 codes of row `r` into `out[..self.cols]` as raw
    /// -8..=7 signed bytes (no scale applied), and returns that row's
    /// dequant scale. Ports the per-row body of
    /// `firmware/esp32_llm/esp32_llm.ino`'s `stage_head_int8`, which
    /// unpacks the tied output-embedding head from int4 to int8 once at
    /// boot (into PSRAM) instead of dequantizing to fp32 on every token --
    /// `llm-firmware`'s dual-core int8 head uses this.
    ///
    /// Assumes `self.n_groups == 1` for this row (only the group-0 scale is
    /// read), matching the C code's own `// n_groups==1` comment at the
    /// call site -- true for the shipped models' tok_emb (`group=128 >=
    /// cols=96`, see `llm-host`'s tests), but this is the caller's
    /// contract to uphold, not something this method checks. A tensor with
    /// `group < cols` (more than one group per row) has more than one
    /// scale per row and can't be represented by a single `f32` return;
    /// use `deq_row` for that case instead.
    pub fn unpack_row_int8(&self, r: usize, out: &mut [i8]) -> f32 {
        debug_assert!(out.len() >= self.cols);
        let row = &self.codes[r * self.row_bytes..(r + 1) * self.row_bytes];
        // Two elements per packed byte, same shape as `deq_row`/`matvec_range`
        // (this was the last remaining per-element `code(r, j)` site). Runs
        // `out_vocab * cols` times at boot when `llm-firmware` stages the
        // output head into PSRAM, so this is boot latency, not per-token
        // cost. Same values, same order.
        let pairs = self.cols / 2;
        for (b, op) in row[..pairs]
            .iter()
            .zip(out[..2 * pairs].chunks_exact_mut(2))
        {
            op[0] = NIBBLE_I8[(b & 0xF) as usize];
            op[1] = NIBBLE_I8[(b >> 4) as usize];
        }
        if self.cols & 1 == 1 {
            let j = self.cols - 1;
            let byte = row[j >> 1];
            let nib = if j & 1 == 1 { byte >> 4 } else { byte & 0xF };
            out[j] = NIBBLE_I8[nib as usize];
        }
        self.scale(r, 0)
    }

    /// One row's packed int4 codes (`row_bytes` of them) and its group-0
    /// scale — what `llm-firmware` copies into PSRAM when it stages the head
    /// as int4 rather than unpacking it to int8. Same `n_groups == 1`
    /// contract as `unpack_row_int8`.
    pub fn packed_row(&self, r: usize) -> (&'a [u8], f32) {
        (
            &self.codes[r * self.row_bytes..(r + 1) * self.row_bytes],
            self.scale(r, 0),
        )
    }

    /// int8-activation path (SIMD-friendly form used on-device). Ports
    /// `matvec_q8_range` / `matvec_q8` / `quantize_act`. `iq` is caller-owned
    /// scratch (>= self.cols), replacing the C static buffer so this is
    /// reentrant/thread-safe instead of relying on a shared global.
    ///
    /// PERFORMANCE NOTE: this had the same per-element `self.code(r, j)` gap
    /// `matvec_range` was fixed for (see that method's doc comment) -- one
    /// byte load + `j & 1` branch per element instead of two elements per
    /// packed byte. Same fix applied here: unpack both nibbles of each byte
    /// in one pass. Pure integer arithmetic (`i32`), summed in the same
    /// left-to-right element order as before, so this is exactly associative
    /// -- not just "close" -- to the old loop; no new float-rounding
    /// question the way the fp32 `matvec_range` change had to reason about.
    /// Verified bit-for-bit unchanged output via `golden.rs` and
    /// `ppl_parity.rs`'s int8-activations tests (both still require exact
    /// agreement with the previously-captured C-reference numbers) and
    /// `cli_parity.rs`'s int8 greedy-decode byte-for-byte check.
    pub fn matvec_int8_activations(&self, x: &[f32], y: &mut [f32], iq: &mut [i8]) {
        let rows = self.rows;
        self.matvec_int8_activations_range(x, y, iq, 0, rows);
    }

    /// `matvec_int8_activations` over `row_begin..row_end` only.
    ///
    /// The activation quantization is over `x`, not over rows, so it is
    /// computed identically whatever range is asked for -- the full-range
    /// call is bit-for-bit what the single-loop version produced, which is
    /// what lets `golden.rs` and `ppl_parity.rs` keep their exact-match
    /// assertions against the captured C int8 numerics.
    pub fn matvec_int8_activations_range(
        &self,
        x: &[f32],
        y: &mut [f32],
        iq: &mut [i8],
        row_begin: usize,
        row_end: usize,
    ) {
        let n = self.cols;
        let x_scale = quantize_activations(&x[..n], &mut iq[..n]);

        for r in row_begin..row_end {
            let row = &self.codes[r * self.row_bytes..(r + 1) * self.row_bytes];
            let mut acc = 0f32;
            for gi in 0..self.n_groups {
                let begin = gi * self.group;
                let end = core::cmp::min(begin + self.group, self.cols);
                let mut g: i32 = 0;
                let mut j = begin;
                if j & 1 == 1 && j < end {
                    g += NIBBLE_I8[(row[j >> 1] >> 4) as usize] as i32 * iq[j] as i32;
                    j += 1;
                }
                let pairs = (end - j) / 2;
                if pairs > 0 {
                    let bytes = &row[j >> 1..(j >> 1) + pairs];
                    let acts = &iq[j..j + 2 * pairs];
                    for (b, ap) in bytes.iter().zip(acts.chunks_exact(2)) {
                        g += NIBBLE_I8[(b & 0xF) as usize] as i32 * ap[0] as i32;
                        g += NIBBLE_I8[(b >> 4) as usize] as i32 * ap[1] as i32;
                    }
                    j += 2 * pairs;
                }
                if j < end {
                    let byte = row[j >> 1];
                    let nib = if j & 1 == 1 { byte >> 4 } else { byte & 0xF };
                    g += NIBBLE_I8[nib as usize] as i32 * iq[j] as i32;
                }
                acc += g as f32 * self.scale(r, gi);
            }
            y[r] = acc * x_scale;
        }
    }
}

/// int8-activation dot product against a row of **packed int4 codes**, without
/// unpacking the row first. Returns the raw `i32` accumulation; the caller
/// applies the row scale and the activation scale, exactly as
/// `matvec_int8_activations` does.
///
/// WHY THIS EXISTS. `llm-firmware` stages the tied output head into PSRAM as
/// int8 at boot (`stage_head_int8` in the C reference), unpacking every int4
/// nibble once so the per-token loop is a plain `i8 * i8` dot. That trade buys
/// simpler arithmetic and pays for it in bytes: it **doubles** what the head
/// streams per token, 2.43 MB instead of 1.21 MB on the shipped model, and it
/// costs 2.4 MB of PSRAM to hold.
///
/// The trade was struck when unpacking a nibble meant a mask, a subtract and a
/// conversion. It now means one indexed load from `NIBBLE_I8` (64 bytes, always
/// resident), and the `+ wider caches` result in BENCHMARKING.md — which moved
/// this stage 21%, more than any other change moved any other stage — says the
/// head is bound by memory, not by arithmetic. Halving the stream is therefore
/// worth more than the unpack costs, *if* that reading is right. This function
/// is how that gets tested rather than argued about.
///
/// EXACT, not approximate: `i32` addition is associative and every term is
/// identical to the one the unpacked path produces, so this returns bit-for-bit
/// what `dot_i8` over the staged int8 row returns. `llm-host`'s
/// `int4_head_equivalence.rs` asserts that over every row of the real model,
/// which is the whole safety argument for the firmware change.
///
/// `row_codes` is one row of `QT::codes` (`row_bytes = ceil(cols/2)` bytes) and
/// `actq` the quantized activations (`cols` of them). Assumes `n_groups == 1`
/// for the row — the same contract `unpack_row_int8` documents, and true of
/// every shipped model's `tok_emb` (`group = 128 >= cols = 96`).
/// Independent accumulators in `dot_int4_int8`. Four -- enough to break the
/// dependency chain without needing more live registers than the LX7 has to
/// spare.
///
/// Both this and the C reference's `dot_i8` originally accumulated into ONE
/// variable, so every iteration's add waited on the previous one. Splitting it
/// four ways took the head 91.0 -> 81.3 ms on device (-10.7%). Real, but far
/// short of what the stall theory predicted, which is what prompted looking at
/// the rest of the loop body -- see `dot_int4_int8`.
const DOT_LANES: usize = 4;

/// Sum of the quantized activations, as `i32`. Hoisted out of the per-row dot
/// by the identity in `dot_int4_int8`; compute once per token and pass it to
/// every row.
#[inline]
pub fn activation_sum(actq: &[i8]) -> i32 {
    let mut s = 0i32;
    for &a in actq {
        s += a as i32;
    }
    s
}

/// `quantize_activations`, storing the result as `i16` instead of `i8`.
///
/// Identical numbers -- the quantizer clamps to `-127..=127`, which is
/// representable in both -- and identical arithmetic downstream. The width is
/// the entire point, and the reason is an Xtensa detail visible only in the
/// disassembly of the shipped firmware.
///
/// The LX7 has no signed byte load. `l8ui` zero-extends, so reading an `i8`
/// costs a load *and* a `sext`, and the head's inner loop was doing both on
/// every one of 2.43M elements per token:
///
/// ```text
/// l8ui  a15, a3, 0
/// sext  a15, a15, 7      <-- one instruction per element, pure overhead
/// mull  a15, a15, a7
/// ```
///
/// `l16si` is a signed 16-bit load. Holding the activation as `i16` folds the
/// sign extension into the load and deletes that instruction. The cost is 96
/// extra bytes of scratch, read once per token; the saving is one instruction
/// per multiply-accumulate.
///
/// Bit-exact by construction, not by measurement: same clamp, same rounding
/// (`round_ties_even`, matching C's `lrintf`), same scale, and `i16 as i32`
/// equals `i8 as i32` for every value the quantizer can emit.
pub fn quantize_activations_i16(x: &[f32], iq: &mut [i16]) -> f32 {
    let n = iq.len();
    debug_assert!(x.len() >= n);
    let mut xmax = 1e-8f32;
    for &v in &x[..n] {
        let a = libm::fabsf(v);
        if a > xmax {
            xmax = a;
        }
    }
    let inv = 127.0f32 / xmax;
    for j in 0..n {
        let q = round_ties_even(x[j] * inv) as i32;
        iq[j] = q.clamp(-127, 127) as i16;
    }
    xmax / 127.0f32
}

/// `activation_sum` over `i16` activations. See `dot_int4_int16`.
#[inline]
pub fn activation_sum_i16(actq: &[i16]) -> i32 {
    let mut s = 0i32;
    for &a in actq {
        s += a as i32;
    }
    s
}

/// `dot_int4_int8` with the activation held as `i16`.
///
/// Same identity, same four lanes, same bias hoist -- only the activation's
/// storage width differs. See `quantize_activations_i16` for why: Xtensa's
/// `l8ui` zero-extends, so an `i8` activation costs a load plus a `sext`,
/// while `l16si` sign-extends in the load. One instruction per
/// multiply-accumulate, across 2.43M of them per token.
///
/// Bit-exact against `dot_int4_int8` over the same activation values, and
/// asserted so on every row of the shipped model by
/// `llm-host/tests/int16_act_equivalence.rs`. `i16 as i32` and `i8 as i32`
/// agree for everything `quantize_activations` can produce, and the
/// accumulation order is unchanged.
///
/// Bounds are unchanged too: the products still fit an `i32` accumulator with
/// room to spare (15 * 127 * 128 = 243,840 and 8 * 127 * 128 = 130,048).
#[inline]
pub fn dot_int4_int16(row_codes: &[u8], actq: &[i16], act_sum: i32) -> i32 {
    let cols = actq.len();
    let pairs = cols / 2;
    let blocks = pairs / DOT_LANES;
    let mut lanes = [0i32; DOT_LANES];
    let bytes = &row_codes[..blocks * DOT_LANES];
    let acts = &actq[..blocks * DOT_LANES * 2];
    for (bq, aq) in bytes
        .chunks_exact(DOT_LANES)
        .zip(acts.chunks_exact(DOT_LANES * 2))
    {
        for l in 0..DOT_LANES {
            let b = bq[l];
            lanes[l] += (b & 0xF) as i32 * aq[2 * l] as i32;
            lanes[l] += (b >> 4) as i32 * aq[2 * l + 1] as i32;
        }
    }
    let mut half = DOT_LANES;
    while half > 1 {
        half /= 2;
        for l in 0..half {
            lanes[l] += lanes[l + half];
        }
    }
    let mut acc = lanes[0];
    let done = blocks * DOT_LANES;
    for p in done..pairs {
        let b = row_codes[p];
        acc += (b & 0xF) as i32 * actq[2 * p] as i32;
        acc += (b >> 4) as i32 * actq[2 * p + 1] as i32;
    }
    if cols % 2 == 1 {
        acc += (row_codes[pairs] & 0xF) as i32 * actq[cols - 1] as i32;
    }
    acc - 8 * act_sum
}

/// Four rows at once, against one activation vector.
///
/// The same idea as `dot2_int4_int16` taken one step further: each activation
/// is loaded once and multiplied into four rows instead of two, so activation
/// loads fall from 0.5 per multiply-accumulate to 0.25. The head's loop is
/// issue-bound (8.13 cycles/MAC cache-resident against 9.01 streaming), so
/// instructions are the currency.
///
/// ```text
/// per 8 MACs      one row     two rows    four rows
///   weight loads      4           4           4
///   nibble extracts   8           8           8
///   activation loads  8           4           2
///   mull + add       16          16          16
/// ```
///
/// IT DID NOT PAY. **This function is not on the firmware's hot path, and
/// should not be put back on it without re-measuring.** The risk written here
/// before the measurement was that four rows need four lane sets, and at
/// `DOT_LANES = 4` that is sixteen live accumulators on a machine with sixteen
/// visible address registers — so the allocator would spill and the loads it
/// added would cost more than the loads this removes. That is what happened.
/// Wiring `head.rs` to this function moved the head **49.3 -> 52.8 ms** and the
/// token 93.7 -> 97.1, and the probe's cycles/MAC went the wrong way,
/// **8.13 -> 8.41**. Reverted at 9e0957c's two-row loop.
///
/// It is kept, tested and exported because the arithmetic is right and the
/// register budget is the only thing wrong with it: `DOT_LANES = 2` with four
/// rows is eight accumulators for the same 0.25 loads/MAC, and that variant is
/// worth a measurement someday. Deleting the function would delete the
/// evidence of where the wall is.
///
/// BIT-EXACT, for the same reason `dot2_int4_int16` is: the four rows share
/// loads and nothing else. Same terms, same order, same lanes, same hoisted
/// bias. `llm-host/tests/dot2_equivalence.rs` covers both.
#[inline]
pub fn dot4_int4_int16(rows: [&[u8]; 4], actq: &[i16], act_sum: i32) -> [i32; 4] {
    let cols = actq.len();
    let pairs = cols / 2;
    let blocks = pairs / DOT_LANES;
    let mut lanes = [[0i32; DOT_LANES]; 4];
    let acts = &actq[..blocks * DOT_LANES * 2];
    for (bi, aq) in acts.chunks_exact(DOT_LANES * 2).enumerate() {
        let base = bi * DOT_LANES;
        for l in 0..DOT_LANES {
            let a0 = aq[2 * l] as i32;
            let a1 = aq[2 * l + 1] as i32;
            for (k, lane) in lanes.iter_mut().enumerate() {
                let w = rows[k][base + l];
                lane[l] += (w & 0xF) as i32 * a0;
                lane[l] += (w >> 4) as i32 * a1;
            }
        }
    }
    let mut out = [0i32; 4];
    for (k, lane) in lanes.iter_mut().enumerate() {
        let mut half = DOT_LANES;
        while half > 1 {
            half /= 2;
            for l in 0..half {
                lane[l] += lane[l + half];
            }
        }
        let mut acc = lane[0];
        let done = blocks * DOT_LANES;
        for p in done..pairs {
            let w = rows[k][p];
            acc += (w & 0xF) as i32 * actq[2 * p] as i32;
            acc += (w >> 4) as i32 * actq[2 * p + 1] as i32;
        }
        if cols % 2 == 1 {
            acc += (rows[k][pairs] & 0xF) as i32 * actq[cols - 1] as i32;
        }
        out[k] = acc - 8 * act_sum;
    }
    out
}

/// Two rows at once, against one activation vector.
///
/// WHY. The head's inner loop was measured **issue-bound** on the ESP32-S3:
/// 8.96 cycles per multiply-accumulate with the weights resident in the data
/// cache, against 9.64 streaming from PSRAM — only 8% apart, so the four
/// accumulator chains already cover load-use latency and the only lever left
/// is fewer instructions per MAC.
///
/// One instruction is free for the taking. The activation vector is the *same*
/// for all 25,353 rows, and `dot_int4_int16` reloads it for every one of them
/// — 2.43M `l16si` per token to read 96 distinct values. Doing two rows in one
/// pass loads each activation once and uses it twice:
///
/// ```text
/// per 4 MACs, one row at a time      per 4 MACs, two rows at once
///   2x l8ui   (weight bytes)           2x l8ui   (one per row)
///   4x extract (nibbles)               4x extract
///   4x l16si  (activations)            2x l16si   <-- halved
///   4x mull                            4x mull
///   4x add                             4x add
/// ```
///
/// Roughly 18 instructions down to 16 per four MACs, on a loop that runs 2.43M
/// times per token.
///
/// BIT-EXACT. Each row's accumulation is untouched — same terms, same order,
/// same four-lane split, same hoisted bias. The two rows never interact; they
/// merely share a load. `llm-host/tests/dot2_equivalence.rs` asserts both
/// returned values against `dot_int4_int16` on every row pair of the shipped
/// model.
///
/// Returns `(dot for row_a, dot for row_b)`. Both rows must have the same
/// length, which for a `QT` they always do.
#[inline]
pub fn dot2_int4_int16(row_a: &[u8], row_b: &[u8], actq: &[i16], act_sum: i32) -> (i32, i32) {
    let cols = actq.len();
    let pairs = cols / 2;
    let blocks = pairs / DOT_LANES;
    let mut lanes_a = [0i32; DOT_LANES];
    let mut lanes_b = [0i32; DOT_LANES];
    let bytes_a = &row_a[..blocks * DOT_LANES];
    let bytes_b = &row_b[..blocks * DOT_LANES];
    let acts = &actq[..blocks * DOT_LANES * 2];
    for ((ba, bb), aq) in bytes_a
        .chunks_exact(DOT_LANES)
        .zip(bytes_b.chunks_exact(DOT_LANES))
        .zip(acts.chunks_exact(DOT_LANES * 2))
    {
        for l in 0..DOT_LANES {
            // Each activation is read into a register once and multiplied
            // twice. That is the entire point of this function.
            let a0 = aq[2 * l] as i32;
            let a1 = aq[2 * l + 1] as i32;
            let wa = ba[l];
            let wb = bb[l];
            lanes_a[l] += (wa & 0xF) as i32 * a0;
            lanes_a[l] += (wa >> 4) as i32 * a1;
            lanes_b[l] += (wb & 0xF) as i32 * a0;
            lanes_b[l] += (wb >> 4) as i32 * a1;
        }
    }
    let mut half = DOT_LANES;
    while half > 1 {
        half /= 2;
        for l in 0..half {
            lanes_a[l] += lanes_a[l + half];
            lanes_b[l] += lanes_b[l + half];
        }
    }
    let (mut acc_a, mut acc_b) = (lanes_a[0], lanes_b[0]);
    let done = blocks * DOT_LANES;
    for p in done..pairs {
        let (a0, a1) = (actq[2 * p] as i32, actq[2 * p + 1] as i32);
        acc_a += (row_a[p] & 0xF) as i32 * a0 + (row_a[p] >> 4) as i32 * a1;
        acc_b += (row_b[p] & 0xF) as i32 * a0 + (row_b[p] >> 4) as i32 * a1;
    }
    if cols % 2 == 1 {
        let a = actq[cols - 1] as i32;
        acc_a += (row_a[pairs] & 0xF) as i32 * a;
        acc_b += (row_b[pairs] & 0xF) as i32 * a;
    }
    (acc_a - 8 * act_sum, acc_b - 8 * act_sum)
}

/// int8-activation dot against a row of **already-unpacked int8 weights**.
///
/// The counterpart to `dot_int4_int8`, and the arm of an experiment rather
/// than a second-best fallback. `llm-firmware` stages the output head as
/// packed int4, which was chosen to halve what the stage streams -- 2.54 MB
/// down to 1.32 MB per token. That change moved the head by **0.0 ms**, and
/// the dual-core bandwidth probe later explained why: the head runs at 21% of
/// available PSRAM bandwidth, so it is compute-bound and always was.
///
/// Which makes the packing a cost with no benefit. Every element pays a mask
/// or a shift to get its nibble out, in a loop that runs 2.43M times per
/// token, and the C reference -- which stages int8 -- never pays it. That is a
/// candidate explanation for the head being the one stage still above C
/// (1.09x) while every other stage is at 0.89-0.90x.
///
/// Bit-exact against `dot_int4_int8` over the same row: unpacking is exact
/// (`nibble - 8` for a 4-bit code is always representable in `i8`), the
/// accumulation order is identical, and the same four-lane split applies.
/// Asserted on every row of the shipped model by
/// `llm-host/tests/int8_head_equivalence.rs`.
///
/// No `act_sum` parameter, unlike `dot_int4_int8`: that argument exists to
/// remove the stored `+8` bias once per row instead of once per element, and
/// unpacked weights carry no bias to remove.
#[inline]
pub fn dot_int8_int8(row: &[i8], actq: &[i8]) -> i32 {
    let n = actq.len();
    debug_assert!(row.len() >= n);
    let blocks = n / DOT_LANES;
    let mut lanes = [0i32; DOT_LANES];
    let ws = &row[..blocks * DOT_LANES];
    let acts = &actq[..blocks * DOT_LANES];
    // Four independent chains, same reason as `dot_int4_int8`: a single
    // accumulator makes every add wait on the previous one, and breaking that
    // dependency was worth 91.0 -> 81.3 ms on the int4 path.
    for (wq, aq) in ws.chunks_exact(DOT_LANES).zip(acts.chunks_exact(DOT_LANES)) {
        for l in 0..DOT_LANES {
            lanes[l] += wq[l] as i32 * aq[l] as i32;
        }
    }
    let mut half = DOT_LANES;
    while half > 1 {
        half /= 2;
        for l in 0..half {
            lanes[l] += lanes[l + half];
        }
    }
    let mut acc = lanes[0];
    for i in blocks * DOT_LANES..n {
        acc += row[i] as i32 * actq[i] as i32;
    }
    acc
}

/// int8-activation dot against a row of **packed int4 codes**, without
/// unpacking the row first. `act_sum` is `activation_sum(actq)`, the same value
/// for every row of a given token. Returns the raw `i32` accumulation; the
/// caller applies the row scale and the activation scale.
///
/// THE INNER LOOP HAS NO SUBTRACT AND NO TABLE. Codes are stored biased
/// (`nibble = value + 8`), so the obvious form is `(nibble - 8) * a`. But
///
/// ```text
/// sum_j (c_j - 8) * a_j  ==  sum_j c_j * a_j  -  8 * sum_j a_j
/// ```
///
/// and the right-hand correction does not depend on the row. So the loop
/// accumulates the biased product directly and the bias is removed once, at the
/// end, for all 96 elements at a time. That deletes one operation per element
/// across 2.43M elements per token.
///
/// It also deletes the `NIBBLE_I8` lookup this function used to do. That table
/// is worth its keep in the fp32 paths, where it replaces an integer-to-float
/// conversion with a load. Here it was replacing a subtract with a load, which
/// is a worse instruction on every machine -- the pattern was copied from
/// `matvec_range` without re-deriving whether it still applied. It did not.
///
/// What remains per element: extract a nibble, sign-extend an activation,
/// multiply, accumulate into one of `DOT_LANES` chains.
///
/// EXACT. Integer arithmetic throughout, and the identity is exact over the
/// integers, not an approximation of it. No overflow: codes are 0..=15,
/// activations -127..=127, `cols <= 128`, so the biased sum is bounded by
/// 15*127*128 = 243,840 and the correction by 8*127*128 = 130,048 -- both far
/// inside `i32`. `llm-host/tests/int4_head_equivalence.rs` asserts the result
/// against the unpacked-int8 dot over every row of the real model.
///
/// `row_codes` is one row of `QT::codes` (`row_bytes = ceil(cols/2)` bytes),
/// `actq` the quantized activations (`cols` of them). Assumes `n_groups == 1`
/// for the row, the same contract `unpack_row_int8` documents.
#[inline]
pub fn dot_int4_int8(row_codes: &[u8], actq: &[i8], act_sum: i32) -> i32 {
    let cols = actq.len();
    let pairs = cols / 2;

    let blocks = pairs / DOT_LANES;
    let mut lanes = [0i32; DOT_LANES];
    let bytes = &row_codes[..blocks * DOT_LANES];
    let acts = &actq[..blocks * DOT_LANES * 2];
    for (bq, aq) in bytes
        .chunks_exact(DOT_LANES)
        .zip(acts.chunks_exact(DOT_LANES * 2))
    {
        for l in 0..DOT_LANES {
            let b = bq[l];
            lanes[l] += (b & 0xF) as i32 * aq[2 * l] as i32;
            lanes[l] += (b >> 4) as i32 * aq[2 * l + 1] as i32;
        }
    }

    // Pairwise combine: same value as a running sum for integers, one less
    // serial step.
    let mut half = DOT_LANES;
    while half > 1 {
        half /= 2;
        for l in 0..half {
            lanes[l] += lanes[l + half];
        }
    }
    let mut acc = lanes[0];

    for (k, b) in row_codes[blocks * DOT_LANES..pairs].iter().enumerate() {
        let j = (blocks * DOT_LANES + k) * 2;
        acc += (b & 0xF) as i32 * actq[j] as i32;
        acc += (b >> 4) as i32 * actq[j + 1] as i32;
    }
    if cols & 1 == 1 {
        acc += (row_codes[pairs] & 0xF) as i32 * actq[cols - 1] as i32;
    }

    // The whole row's bias, removed once.
    acc - 8 * act_sum
}

/// A plain fp32 vector view (norm weights etc.), stored as raw little-endian
/// bytes in the file. Read element-by-element instead of reinterpret-cast
/// to avoid relying on the buffer being 4-byte aligned at this offset (the
/// C code casts a `const float *` here, which works on x86/Xtensa but isn't
/// guaranteed portable; this is the safe equivalent).
#[derive(Clone, Copy)]
pub struct FVec<'a> {
    bytes: &'a [u8],
    pub len: usize,
}

impl<'a> FVec<'a> {
    #[inline]
    pub fn get(&self, i: usize) -> f32 {
        let off = i * 4;
        f32::from_le_bytes([
            self.bytes[off],
            self.bytes[off + 1],
            self.bytes[off + 2],
            self.bytes[off + 3],
        ])
    }

    /// The first `n` elements, in order.
    ///
    /// Prefer this over `get(i)` in a loop. `get` indexes four separate
    /// bytes and bounds-checks each one, so a sequential walk pays four
    /// checks and a byte-assembly per element -- next to C's `const float
    /// *`, which is one load. Iterating `chunks_exact(4)` instead gives the
    /// backend a provable stride and one `from_le_bytes` per element, which
    /// lowers to a single (unaligned) 32-bit load on both x86-64 and
    /// Xtensa. Same values, and still no assumption that the model buffer
    /// is 4-byte aligned at this offset -- which is the reason `FVec` reads
    /// bytes rather than reinterpret-casting in the first place.
    #[inline]
    pub fn iter(&self, n: usize) -> impl Iterator<Item = f32> + 'a {
        self.bytes[..n * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
    }
}

/// Advances over the model.bin buffer, binding tensors in the exact order
/// `llm_load` does. Ports `bind_q` / `bind_f` as methods instead of
/// free functions threading a cursor pointer by hand.
pub(crate) struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Cursor { buf, pos: 0 }
    }

    pub fn at(buf: &'a [u8], pos: usize) -> Self {
        Cursor { buf, pos }
    }

    pub fn pos(&self) -> usize {
        self.pos
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], LoadError> {
        let end = self.pos.checked_add(n).ok_or(LoadError::TooShort)?;
        let s = self.buf.get(self.pos..end).ok_or(LoadError::TooShort)?;
        self.pos = end;
        Ok(s)
    }

    pub fn u32(&mut self) -> Result<u32, LoadError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    pub fn i32(&mut self) -> Result<i32, LoadError> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    pub fn f32(&mut self) -> Result<f32, LoadError> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    pub fn bind_q(&mut self, rows: usize, cols: usize) -> Result<QT<'a>, LoadError> {
        let group = self.i32()? as usize;
        let n_groups = (cols + group - 1) / group;
        let row_bytes = (cols + 1) / 2;
        let codes = self.take(rows * row_bytes)?;
        let scales = self.take(rows * n_groups * 2)?;
        Ok(QT {
            codes,
            scales,
            rows,
            cols,
            group,
            n_groups,
            row_bytes,
        })
    }

    pub fn bind_f(&mut self, n: usize) -> Result<FVec<'a>, LoadError> {
        let bytes = self.take(n * 4)?;
        Ok(FVec { bytes, len: n })
    }
}

pub(crate) const MAGIC_CHECK: u32 = MAGIC;
pub(crate) const MAGIC_V2_CHECK: u32 = MAGIC_V2;
pub(crate) const FLAG_TIED_HEAD_CHECK: u32 = FLAG_TIED_HEAD;
pub(crate) const MAX_FORMAT_VERSION_CHECK: u32 = MAX_FORMAT_VERSION;
