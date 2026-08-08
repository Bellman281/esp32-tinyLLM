//! Model config, quantized-tensor view, and the binary-cursor binder.
//!
//! Ports `struct Cfg`, `struct QT`, `bind_q`, `bind_f` from
//! reference-c/firmware/common/llm.h. Everything here borrows from the raw
//! model.bin byte buffer instead of copying it, same as the C code's
//! bind-in-place pointers -- just as `&'a [u8]` slices instead of raw
//! pointers, so out-of-range reads are caught instead of silently
//! corrupting memory.

const MAGIC: u32 = 0x504C_4531;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadError {
    TooShort,
    BadMagic,
    TooManyLayers,
}

/// Mirrors C's `Cfg`. All dims stored as `usize` (C uses `int`, always
/// non-negative in this file format).
#[derive(Debug, Clone, Copy)]
pub struct Cfg {
    pub vocab: usize,
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

    /// Signed code (value - 8, i.e. -8..=7) at logical column `j` of row `r`.
    #[inline]
    fn code(&self, r: usize, j: usize) -> i32 {
        let byte = self.codes[r * self.row_bytes + (j >> 1)];
        let nibble = if j & 1 == 1 { byte >> 4 } else { byte & 0xF };
        nibble as i32 - 8
    }

    /// Dequantize row `r` into `out` (len == self.cols). Ports `deq_row`.
    pub fn deq_row(&self, r: usize, out: &mut [f32]) {
        debug_assert!(out.len() >= self.cols);
        for gi in 0..self.n_groups {
            let begin = gi * self.group;
            let end = core::cmp::min(begin + self.group, self.cols);
            let scale = self.scale(r, gi);
            for j in begin..end {
                out[j] = self.code(r, j) as f32 * scale;
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
    pub fn matvec_range(&self, x: &[f32], y: &mut [f32], row_begin: usize, row_end: usize) {
        debug_assert!(x.len() >= self.cols);
        for r in row_begin..row_end {
            let row = &self.codes[r * self.row_bytes..(r + 1) * self.row_bytes];
            let mut acc = 0f32;
            for gi in 0..self.n_groups {
                let begin = gi * self.group;
                let end = core::cmp::min(begin + self.group, self.cols);
                let scale = self.scale(r, gi);
                let mut group_acc = 0f32;
                let mut j = begin;
                if j & 1 == 1 && j < end {
                    let byte = row[j >> 1];
                    group_acc += ((byte >> 4) as i32 - 8) as f32 * x[j];
                    j += 1;
                }
                while j + 1 < end {
                    let byte = row[j >> 1];
                    group_acc += ((byte & 0xF) as i32 - 8) as f32 * x[j];
                    group_acc += ((byte >> 4) as i32 - 8) as f32 * x[j + 1];
                    j += 2;
                }
                if j < end {
                    let byte = row[j >> 1];
                    let code = if j & 1 == 1 { (byte >> 4) as i32 } else { (byte & 0xF) as i32 };
                    group_acc += (code - 8) as f32 * x[j];
                }
                acc += group_acc * scale;
            }
            y[r] = acc;
        }
    }

    pub fn matvec(&self, x: &[f32], y: &mut [f32]) {
        let rows = self.rows;
        self.matvec_range(x, y, 0, rows);
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
        for j in 0..self.cols {
            out[j] = self.code(r, j) as i8;
        }
        self.scale(r, 0)
    }

    /// int8-activation path (SIMD-friendly form used on-device). Ports
    /// `matvec_q8_range` / `matvec_q8` / `quantize_act`. `iq` is caller-owned
    /// scratch (>= self.cols), replacing the C static buffer so this is
    /// reentrant/thread-safe instead of relying on a shared global.
    pub fn matvec_int8_activations(&self, x: &[f32], y: &mut [f32], iq: &mut [i8]) {
        let n = self.cols;
        let x_scale = quantize_activations(&x[..n], &mut iq[..n]);

        for r in 0..self.rows {
            let mut acc = 0f32;
            for gi in 0..self.n_groups {
                let begin = gi * self.group;
                let end = core::cmp::min(begin + self.group, self.cols);
                let mut g: i32 = 0;
                for j in begin..end {
                    g += self.code(r, j) * iq[j] as i32;
                }
                acc += g as f32 * self.scale(r, gi);
            }
            y[r] = acc * x_scale;
        }
    }
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
