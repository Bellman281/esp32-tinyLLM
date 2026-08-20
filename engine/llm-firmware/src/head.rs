//! Dual-core int8-staged output head -- ports the C reference's
//! `stage_head_int8`, `head_worker_main`, `head_matvec_int8`, `dot_i8`, and
//! `head_rows_range`, plus the task-handle/job plumbing around them
//! (`head_worker`, `inference_task`, `head_job_y`, `head_job_split`) from
//! `firmware/esp32_llm/esp32_llm.ino`.
//!
//! The head is a `[vocab, dim]` matvec run in full every token and
//! dominates per-token runtime, so the C reference stages it as int8 in
//! PSRAM once at boot (int4 nibbles unpacked once, no per-token float
//! conversion) and splits its rows across both LX7 cores: a worker task
//! pinned to core 0 computes the first half while whichever core calls
//! `Head::matvec` (the "inference" core, running the decode loop)
//! computes the second half itself, then waits for the worker. FreeRTOS
//! task notifications are the only synchronization primitive -- a single
//! producer/single consumer hand-off, same as the C reference, not a
//! mutex or channel.
//!
//! This module reaches for raw pointers (`*mut f32`, `UnsafeCell`) rather
//! than typed `&mut` slices for the buffer both cores write into, on
//! purpose: `Head::matvec` and `head_worker_main` write disjoint row
//! ranges of the *same* underlying allocation from two OS threads at the
//! same time. That's exactly what the C reference does with a plain
//! `float *y` and no aliasing rules to satisfy -- raw pointers are the
//! faithful translation, where a single `&mut [f32]` held across the
//! worker's concurrent write would not be (Rust's aliasing model doesn't
//! have a "disjoint sub-range, trust me" exception for `&mut`). The
//! invariant that makes this sound -- worker writes `[0, split)`, the
//! caller writes `[split, rows)`, never both at once for the same index --
//! is documented at each unsafe block below rather than enforced by the
//! type system, same as it's an unenforced (but true) invariant in the C
//! code.
//!
//! VERIFICATION STATUS: not compiled. The task API
//! (`esp_idf_svc::hal::task::{create, current, notify, wait_notification}`,
//! `esp_idf_svc::hal::cpu::Core`) was researched against esp-idf-hal's own
//! source (see llm-firmware/README.md for the confidence tier and
//! source link) but not run through a real build. The int8 numerics
//! (`llm_core::quantize_activations`, `QT::unpack_row_int8`) are the same
//! functions `llm-host`'s ppl parity tests already verified bit-for-bit
//! against the C reference -- the only genuinely untested-anywhere logic
//! in this file is the dual-core split and task plumbing itself, which no
//! host-side test can exercise (there's no FreeRTOS on the host).

use core::cell::{Cell, UnsafeCell};
use core::sync::atomic::{AtomicI32, AtomicPtr, AtomicU32, AtomicUsize, Ordering};
use esp_idf_svc::hal::cpu::Core;
use esp_idf_svc::hal::sys::{esp_timer_get_time, TaskHandle_t, TickType_t};
use esp_idf_svc::hal::task;
use llm_core::{activation_sum_i16, dot_int4_int16, quantize_activations_i16, QT};

use crate::psram;

/// Any nonzero value works as the notification token -- FreeRTOS task
/// notifications used this way (`xTaskNotifyGive`/`ulTaskNotifyTake`) are
/// a binary hand-off, not a counted semaphore or a message payload.
const NOTIFY: core::num::NonZeroU32 = core::num::NonZeroU32::new(1).unwrap();

/// C's `head_actq[128]`: "the max input dim across the model is 128" --
/// this head's actual cols (== model dim, 96 on the shipped models) is
/// always <= this, same fixed-size-scratch choice the C reference makes.
const MAX_HEAD_COLS: usize = 128;

/// What the worker should do when notified.
///
/// The worker exists to run the output head on core 0 while the inference core
/// runs its own half. But the head is now 61.5 ms of a 137.9 ms token, and the
/// other 76.4 ms -- input, attn, ffn, ple -- runs on ONE core with this task
/// parked in `wait_notification`. The fp32 matvecs inside those stages are
/// 556k multiply-accumulates per token at ~21 cycles each, and splitting them
/// by rows is bit-exact: `y[r]` depends only on row `r` and `x`, so which core
/// computes it cannot change the value (asserted at every split point by
/// `llm-host/tests/matvec_split.rs`).
///
/// So the worker takes a second kind of job.
///
/// And then a third. The matvec hook cannot reach attention, because attention
/// is not a matvec. Instrumenting the `attn` stage four ways gave
/// `qkv 7.8 | rope 0.14 | core 28.3 | proj 2.7` ms/token: `qkv` and `proj` are
/// matvecs and were already split, `rope` is nothing, and `core` -- 24% of a
/// token -- was the largest thing left still running on one LX7 with this task
/// parked in `wait_notification`.
///
/// Attention heads divide exactly as matvec rows do: each head reads the whole
/// (already stored) KV cache and writes only its own `head_dim` slice of the
/// output, so no dot product changes and nothing reorders.
/// `llm-host/tests/attn_split.rs` holds every split point of the shipped model
/// to bit-for-bit agreement across a 24-token generation.
const JOB_HEAD: u32 = 0;
const JOB_MATVEC: u32 = 1;
const JOB_ATTN: u32 = 2;

/// And a fourth, which computes nothing: stream a buffer and report how long it
/// took, so the same measurement can be taken with one core reading and with
/// two reading at once.
///
/// Splitting attention's heads across both cores took `core` from 28.3 to
/// 15.6 ms/token. A perfect halving plus the measured 0.1 ms of sync predicts
/// 14.25, so 1.35 ms -- 9.5% -- did not come back, and the single-core
/// bandwidth probe cannot say why. Two candidates:
///
///   * Contention. Unlike the output head, both cores now scan the same KV
///     cache at the same time. `probe_read_bandwidth` reports ~68 MB/s with
///     one core reading, and that number is equally consistent with a bus
///     that saturates at 68 and one that delivers 68 per core. Those imply
///     opposite next moves.
///   * Placement. The binary that introduced the split reads 68 MB/s where
///     its predecessor read 72, on an identical probe over an identical
///     buffer, before generation starts -- the layout tax, worth several
///     percent on this board.
///
/// `JOB_PROBE` distinguishes them. If two cores reading disjoint halves
/// finish in half the time one core takes over the whole buffer, the bus
/// scales and the shortfall was placement. If they take as long as one core
/// did, the bus is saturated, attention is memory-bound, and halving the KV
/// cache's working set is worth building.
#[cfg(feature = "bandwidth-probe")]
const JOB_PROBE: u32 = 3;

/// Ceiling on `seq_len` for the split attention path, because each core needs
/// its own `scores` scratch and these are fixed-size `[f32; N]` in `.bss`
/// rather than another PSRAM allocation -- 2 KB each, and `scores` is the
/// hottest small buffer in the stage, read and written once per position per
/// head. The shipped model is `seq_len = 512`.
///
/// Same shape of decision as `MAX_HEAD_COLS` above, and checked the same way:
/// `attn_parallel` refuses to split rather than truncating, so a larger model
/// gets the single-core path and correct output instead of a silent overrun.
pub const MAX_ATTN_SEQ: usize = 512;

/// Owns the staged int8 head tables, the dual-core job hand-off state, and
/// both task handles. Always accessed through `&Head` (interior
/// mutability for the parts genuinely written after construction) --
/// there is exactly one instance, leaked for `'static` lifetime by
/// `stage_and_spawn`, same as the C reference's file-scope globals.
pub struct Head {
    // [rows * row_bytes] of PACKED int4 codes in PSRAM -- NOT the C
    // reference's [rows * cols] int8. See `stage_and_spawn`.
    w4: &'static [u8],
    scale8: &'static [f32], // [rows], per-row dequant scale
    rows: usize,
    cols: usize,
    row_bytes: usize, // ceil(cols / 2)

    // Shared quantized-activation scratch: written once by `matvec` before
    // notifying the worker, read by both `matvec` (this core's half) and
    // `head_worker_main` (the worker's half) afterward, never written
    // again until the next `matvec` call -- which by construction can't
    // start until the previous cycle's `wait_notification` returns, i.e.
    // after the worker is done reading it. See the module doc comment.
    // i16, not i8, and the width is the whole point. Xtensa has no signed
    // byte load: `l8ui` zero-extends, so reading an `i8` activation costs a
    // load AND a `sext` -- visible in the shipped firmware's disassembly, on
    // every one of 2.43M elements per token. `l16si` sign-extends in the
    // load. Costs 128 extra bytes of scratch, written once per token. See
    // `llm_core::quantize_activations_i16`.
    actq: UnsafeCell<[i16; MAX_HEAD_COLS]>,
    acts: AtomicU32, // f32 bits (no AtomicF32 in core)
    // Sum of `actq`, written with it and read by both cores for every row --
    // see llm_core::dot_int4_int8 for why one number per token replaces one
    // subtract per element.
    act_sum: AtomicI32,

    // The current job's output pointer and row split, set by `matvec`
    // before notifying the worker.
    job_y: core::sync::atomic::AtomicPtr<f32>,
    job_split: AtomicUsize,

    worker: Cell<TaskHandle_t>, // set once, right after task::create succeeds
    inference_task: TaskHandle_t, // set once at construction, never changes

    // Sub-stage timing. The five-stage profile in llm-core reports the head as
    // one number, which was enough to see it dominate and not enough to see
    // WHY -- staging the weights as int4 halved what it streams and moved that
    // number by 0.0 ms, which no coarser instrument could explain. These split
    // it four ways.
    //
    // `Cell`, not an atomic: `own`/`quant`/`wait` are written only by the
    // inference core inside `matvec`, and `worker` only by the worker task
    // inside `head_worker_main`. Neither is read until generation has finished
    // and both tasks are quiescent. Same single-writer reasoning as the
    // `worker` handle above.
    prof_quant_us: Cell<u64>,  // quantizing the activation
    prof_own_us: Cell<u64>,    // this core's row range
    prof_wait_us: Cell<u64>,   // blocked on the worker after finishing ours
    prof_worker_us: Cell<u64>, // the worker's row range, timed on the worker
    prof_calls: Cell<u32>,

    // ---- greedy argmax, computed where the logits are produced -----------
    // The decode loop used to pick the next token by scanning all 25,353
    // logits back out of PSRAM -- 101 KB per token at roughly half the bus
    // rate, because a data-dependent branch per element leaves nothing to
    // pipeline. Measured at 2.74 ms/token, the largest single line item left
    // once the model math passed C.
    //
    // Both cores already hold each logit in a register the moment
    // `rows_range` computes it, so each tracks the max over its own rows and
    // the decode loop chooses between two candidates. Nothing is read back.
    //
    // f32 as bits: there is no AtomicF32 in core, same as `acts` above.
    lo_max: AtomicU32,
    lo_arg: AtomicUsize,
    hi_max: AtomicU32,
    hi_arg: AtomicUsize,

    // ---- generic parallel matvec ------------------------------------------
    // Which job the worker should run. Written by whichever core is dispatching
    // (always the inference core) before the notify that wakes the worker; the
    // Release on this store publishes every field below it.
    job_kind: AtomicU32,
    // The tensor, addressed as `QT<'static>`: llm-firmware's model is
    // `partition::map_model_partition() -> &'static [u8]`, so every QT bound
    // from it borrows for `'static`. Valid for the duration of the call
    // because the inference core holds the `&QT` across the notify/wait.
    mv_qt: AtomicPtr<QT<'static>>,
    mv_x: AtomicPtr<f32>,
    mv_x_len: AtomicUsize,
    mv_out: AtomicPtr<f32>,   // the WORKER's destination, not the whole output
    mv_out_len: AtomicUsize,
    mv_row_begin: AtomicUsize,
    // Sync cost of the split, which is the thing that could make it a loss:
    // ~43 matvecs per token, each paying a notify and a wake.
    prof_mv_wait_us: Cell<u64>,
    prof_mv_calls: Cell<u32>,

    // ---- parallel attention heads -----------------------------------------
    // The `HeadsJob` handed over for a JOB_ATTN, flattened into pointers and
    // lengths. Every one is valid only for the duration of one
    // `attn_parallel` call, which holds the borrows across the notify/wait --
    // same contract as the mv_* fields above.
    at_q: AtomicPtr<f32>,
    at_kc: AtomicPtr<f32>,
    at_vc: AtomicPtr<f32>,
    at_kv_len: AtomicUsize, // n_heads * seq_len * head_dim, both caches
    at_out: AtomicPtr<f32>, // the WORKER's slice of att_out, not the whole
    at_out_len: AtomicUsize,
    at_pos: AtomicUsize,
    at_head_dim: AtomicUsize,
    at_seq_len: AtomicUsize,
    at_h0: AtomicUsize,
    at_h1: AtomicUsize,

    // Per-head softmax scratch, one per core. This is the only part of the
    // head split that is not free: `scores` is reused per head, so two cores
    // running different head ranges at the same time need two buffers.
    //
    // `scores_a` is written only by the inference core inside `attn_parallel`,
    // `scores_b` only by the worker inside a JOB_ATTN. Neither is ever read by
    // the other core -- unlike `actq`, which is shared, these are genuinely
    // private to one task each and merely live in a shared struct because that
    // is where the worker can reach its own.
    scores_a: UnsafeCell<[f32; MAX_ATTN_SEQ]>,
    scores_b: UnsafeCell<[f32; MAX_ATTN_SEQ]>,

    prof_attn_wait_us: Cell<u64>,
    prof_attn_own_us: Cell<u64>,
    prof_attn_calls: Cell<u32>,

    // ---- dual-core bandwidth probe ----------------------------------------
    // The worker's half of a JOB_PROBE, and where it reports its elapsed time.
    // Atomic rather than `Cell` unlike the profile counters above: those are
    // read after generation ends with both tasks quiescent, this one is read
    // by the inference core immediately after the worker's notify. u32
    // microseconds, not u64 -- a half-buffer pass is ~9 ms, and AtomicU64 is
    // not lock-free on a 32-bit LX7.
    #[cfg(feature = "bandwidth-probe")]
    pr_ptr: AtomicPtr<u8>,
    #[cfg(feature = "bandwidth-probe")]
    pr_len: AtomicUsize,
    #[cfg(feature = "bandwidth-probe")]
    pr_us: AtomicU32,
}

impl Head {
    /// Ports the per-row body of `stage_head_int8` plus
    /// `head_rows_range`/`dot_i8`: computes `y[r0..r1]` from the staged
    /// int8 weights and the shared quantized activation. `y` is a raw
    /// pointer (not `&mut [f32]`) because both this core and the worker
    /// task call this concurrently on disjoint ranges of the same buffer
    /// -- see the module doc comment.
    ///
    /// # Safety
    /// `y` must point to a valid, initialized `[f32; self.rows]` (or
    /// longer) that the caller has the right to write `y[r0..r1]` of --
    /// i.e. no other writer is touching that sub-range concurrently.
    unsafe fn rows_range(&self, y: *mut f32, r0: usize, r1: usize) -> (f32, usize) {
        let mut best_val = f32::NEG_INFINITY;
        let mut best_idx = r0;
        let cols = self.cols;
        let rb = self.row_bytes;
        let w4 = self.w4;
        let scale8 = self.scale8;
        // Slice the fixed-size scratch down to `cols` ONCE, out here, rather
        // than indexing `actq[j]` in the inner loop. `actq` is a
        // `[i8; MAX_HEAD_COLS]` (128) but `cols` is a runtime field, so inside
        // the loop LLVM cannot prove `j < 128` and emits a bounds check per
        // multiply-accumulate. Slicing here gives `dot_int4_int8` a slice whose
        // length *is* `cols`, which is also how it learns how many codes to
        // read. Two steps, not `&(*self.actq.get())[..cols]` -- that form trips
        // rustc's deny-by-default `dangerous_implicit_autorefs` lint.
        // SAFETY: see the `actq` field doc comment -- by the time this runs
        // (either core), the write that populated it has already
        // happened-before, and nothing writes it again until both readers are
        // done.
        let actq_all: &[i16; MAX_HEAD_COLS] = unsafe { &*self.actq.get() };
        let actq: &[i16] = &actq_all[..cols];
        let acts = f32::from_bits(self.acts.load(Ordering::Acquire));
        let act_sum = self.act_sum.load(Ordering::Acquire);
        for r in r0..r1 {
            // Two codes per byte, unpacked in the loop instead of at boot.
            // `llm_core::dot_int4_int8` is asserted bit-for-bit equal to the
            // staged-int8 dot over every row of the shipped model by
            // llm-host/tests/int4_head_equivalence.rs -- integer addition is
            // associative and every term is the same term, so this is exact,
            // not approximate.
            let acc = dot_int4_int16(&w4[r * rb..r * rb + rb], actq, act_sum);
            let val = acc as f32 * scale8[r] * acts;
            // SAFETY: caller's contract (see this fn's doc comment) is that
            // `y[r0..r1]` is this call's alone to write.
            unsafe { core::ptr::write(y.add(r), val) };
            // Running argmax over this core's rows, on a value already in a
            // register. Strictly `>`, so the LOWEST index wins a tie -- the
            // same rule a `0..vocab_n` scan follows. See `last_argmax`.
            if val > best_val {
                best_val = val;
                best_idx = r;
            }
        }
        (best_val, best_idx)
    }

    /// Ports `head_matvec_int8`. This is the closure body
    /// `llm-firmware`'s `main` wires into
    /// `llm_core::llm_forward_with_head_override` in place of the default
    /// tied-embedding matvec. `x` is the pre-head activation
    /// (`s.x` post-RMSNorm, length >= `self.cols`); `y` is `s.logits`
    /// (length >= `self.rows` -- may be longer, since `self.rows` was
    /// capped to the trained vocab count at staging time while `s.logits`
    /// is sized for the full padded vocab; rows at/above `self.rows` are
    /// simply never written, same as the C reference's `s.logits[V]` with
    /// `head_rows < V`).
    pub fn matvec(&self, x: &[f32], y: &mut [f32]) {
        debug_assert!(x.len() >= self.cols);
        debug_assert!(y.len() >= self.rows);
        let t0 = unsafe { esp_timer_get_time() };

        // SAFETY: this function only ever runs on the inference core
        // (never concurrently with itself -- `llm_forward` is called in a
        // sequential decode loop), and the worker never writes `actq`, so
        // this is the sole writer.
        let actq: &mut [i16] = unsafe {
            let arr: &mut [i16; MAX_HEAD_COLS] = &mut *self.actq.get();
            &mut arr[..self.cols]
        };
        let scale = quantize_activations_i16(&x[..self.cols], actq);
        // Relaxed: the Release on `acts` immediately below publishes both, the
        // same way it already publishes `actq`.
        self.act_sum.store(activation_sum_i16(actq), Ordering::Relaxed);
        self.acts.store(scale.to_bits(), Ordering::Release);

        // Capture the raw pointer once; from here on both this core and
        // the worker index through raw pointers into the same allocation,
        // on disjoint ranges -- never through a typed `&mut [f32]` again
        // (see module doc comment).
        let t1 = unsafe { esp_timer_get_time() };

        let y_ptr = y.as_mut_ptr();
        let split = self.rows / 2;
        self.job_split.store(split, Ordering::Relaxed);
        self.job_y.store(y_ptr, Ordering::Relaxed);
        // Release: publishes job_y/job_split/actq/acts/act_sum to the worker.
        self.job_kind.store(JOB_HEAD, Ordering::Release);

        // SAFETY: `self.worker` was set exactly once in `stage_and_spawn`,
        // before this `Head` was ever returned/reachable, so it's always
        // valid by the time any caller can reach `matvec`.
        let _ = unsafe { task::notify(self.worker.get(), NOTIFY) };

        // SAFETY: rows [split, self.rows) belong exclusively to this call
        // until it returns -- the worker task, notified above, only ever
        // writes [0, split) (see `head_worker_main`). Disjoint index
        // ranges of the same buffer, concurrently, by construction.
        let (hi_val, hi_idx) = unsafe { self.rows_range(y_ptr, split, self.rows) };
        self.hi_max.store(hi_val.to_bits(), Ordering::Relaxed);
        self.hi_arg.store(hi_idx, Ordering::Relaxed);
        let t2 = unsafe { esp_timer_get_time() };

        // Blocks until `head_worker_main`'s `task::notify` (its half is
        // done). `TickType_t::MAX` is this port's equivalent of C's
        // `portMAX_DELAY` (wait forever) -- see this module's README
        // entry for the verification caveat on that mapping.
        let _ = task::wait_notification(TickType_t::MAX);
        let t3 = unsafe { esp_timer_get_time() };

        self.prof_quant_us.set(self.prof_quant_us.get() + (t1 - t0) as u64);
        self.prof_own_us.set(self.prof_own_us.get() + (t2 - t1) as u64);
        self.prof_wait_us.set(self.prof_wait_us.get() + (t3 - t2) as u64);
        self.prof_calls.set(self.prof_calls.get() + 1);
    }

    /// Zero the sub-stage counters. Called after prompt priming, for the same
    /// reason `llm_core::Profile::reset` is: priming decodes with a
    /// differently-shaped KV cache and would skew the averages.
    pub fn profile_reset(&self) {
        self.prof_quant_us.set(0);
        self.prof_own_us.set(0);
        self.prof_wait_us.set(0);
        self.prof_worker_us.set(0);
        self.prof_calls.set(0);
        self.prof_mv_wait_us.set(0);
        self.prof_mv_calls.set(0);
        self.prof_attn_wait_us.set(0);
        self.prof_attn_own_us.set(0);
        self.prof_attn_calls.set(0);
    }

    /// `[quantize, own half, wait for worker, worker half]` in ms/token, or
    /// `None` before any timed call.
    ///
    /// How to read it:
    /// - `own + wait` should account for essentially all of the head stage.
    /// - `wait` near zero means this core is the slow half and the split is
    ///   fine. `wait` large means the worker's half takes materially longer
    ///   than ours, and moving `split` off `rows / 2` is free throughput --
    ///   core 0 also services interrupts, so an even row split is not
    ///   necessarily an even time split.
    /// - `worker` is that half measured on the worker itself, so
    ///   `wait ~= worker - own` when the worker is slower. A `wait` much
    ///   larger than `worker - own` is scheduling latency, not compute.
    /// - `quant` should be negligible (96 elements); if it is not, something
    ///   is wrong with the activation path rather than the matvec.
    pub fn profile_ms_per_token(&self) -> Option<[f32; 4]> {
        let n = self.prof_calls.get();
        if n == 0 {
            return None;
        }
        let d = n as f32 * 1000.0;
        Some([
            self.prof_quant_us.get() as f32 / d,
            self.prof_own_us.get() as f32 / d,
            self.prof_wait_us.get() as f32 / d,
            self.prof_worker_us.get() as f32 / d,
        ])
    }

    /// The index of the largest logit from the most recent `matvec`.
    ///
    /// Exactly what `for v in 0..rows { if logits[v] > best { .. } }` would
    /// return, without reading the logits back. Each core tracked the maximum
    /// over its own rows with a strict `>`, so within a half the lowest index
    /// wins a tie; between halves `hi > lo` picks `hi` only on a strict win, so
    /// a tie again resolves to the lower index. That is the same token a
    /// front-to-back scan selects, in every case.
    ///
    /// Reads only what the worker published before its notify, and `matvec`
    /// has already waited on that notify by the time this can be called.
    pub fn last_argmax(&self) -> usize {
        // Acquire pairs with the worker's Release store of `lo_arg`.
        let lo_idx = self.lo_arg.load(Ordering::Acquire);
        let lo = f32::from_bits(self.lo_max.load(Ordering::Relaxed));
        let hi_idx = self.hi_arg.load(Ordering::Relaxed);
        let hi = f32::from_bits(self.hi_max.load(Ordering::Relaxed));
        if hi > lo {
            hi_idx
        } else {
            lo_idx
        }
    }

    /// Compute `y[..t.rows] = W * x` with the rows split across both cores.
    ///
    /// This is what `llm_forward_*_with_matvec_override` hands every non-head
    /// matvec to. The hook has existed since Phase 3 and never had a user; it
    /// was written for an esp-dsp SIMD path, and a dual-core split turns out to
    /// need exactly the same shape. `llm-core` is unchanged by this.
    ///
    /// Rows `[0, split)` go to the worker, `[split, rows)` stay here, and the
    /// caller blocks until the worker signals. Bit-exact: each output element
    /// is the same dot product either way, and `QT::matvec_rows_into`
    /// addresses the destination relative to the slice it is given so neither
    /// side needs to know about the other's rows.
    ///
    /// No size threshold. The smallest matvec in this model is `gate`/`up` at
    /// 66x96 = 6,336 MACs, about 0.55 ms on one core, against a notify/wake
    /// round trip in the tens of microseconds. If a future model has matvecs
    /// small enough for that to invert, this is where the threshold goes.
    ///
    /// # Panics
    /// Debug-only: `y` must be at least `t.rows` long and `x` at least
    /// `t.cols`, the same contract `QT::matvec` has.
    pub fn matvec_parallel(&self, t: &QT, x: &[f32], y: &mut [f32]) {
        let rows = t.rows;
        debug_assert!(y.len() >= rows);
        debug_assert!(x.len() >= t.cols);
        let split = rows / 2;

        // Raw pointer once, then two disjoint ranges -- the same structure
        // `matvec`/`rows_range` use below, and for the same reason: two OS
        // threads write disjoint sub-ranges of one allocation concurrently,
        // which `&mut [f32]` cannot express.
        let y_ptr = y.as_mut_ptr();

        self.mv_qt
            .store(t as *const QT as *mut QT<'static>, Ordering::Relaxed);
        self.mv_x.store(x.as_ptr() as *mut f32, Ordering::Relaxed);
        self.mv_x_len.store(x.len(), Ordering::Relaxed);
        self.mv_out.store(y_ptr, Ordering::Relaxed);
        self.mv_out_len.store(split, Ordering::Relaxed);
        self.mv_row_begin.store(0, Ordering::Relaxed);
        // Release: publishes every field above to the worker.
        self.job_kind.store(JOB_MATVEC, Ordering::Release);

        // SAFETY: `self.worker` was set once in `stage_and_spawn`, before this
        // `Head` was reachable.
        let _ = unsafe { task::notify(self.worker.get(), NOTIFY) };

        // SAFETY: rows [split, rows) are this call's alone until it returns --
        // the worker only ever writes [0, split) for a JOB_MATVEC (see
        // `head_worker_main`). Disjoint by construction.
        let hi = unsafe { core::slice::from_raw_parts_mut(y_ptr.add(split), rows - split) };
        t.matvec_rows_into(x, hi, split);

        let t0 = unsafe { esp_timer_get_time() };
        let _ = task::wait_notification(TickType_t::MAX);
        let t1 = unsafe { esp_timer_get_time() };
        self.prof_mv_wait_us
            .set(self.prof_mv_wait_us.get() + (t1 - t0) as u64);
        self.prof_mv_calls.set(self.prof_mv_calls.get() + 1);
    }

    /// Run one layer's attention heads with the heads split across both cores.
    ///
    /// This is what `llm_forward_profiled_with_overrides` hands each layer's
    /// `HeadsJob` to, after `store_kv` has already put this position's k/v in
    /// the cache. Heads `[0, mid)` stay here, `[mid, n_heads)` go to the
    /// worker; both read the same immutable cache and write disjoint slices of
    /// `att_out`.
    ///
    /// Bit-exact, and for a simpler reason than the matvec split: each head's
    /// softmax and weighted sum are computed by exactly one core, in exactly
    /// the same order, over exactly the same values. Nothing is combined
    /// across the boundary -- there is no reduction to reassociate. Asserted
    /// at every split point over a 24-token generation by
    /// `llm-host/tests/attn_split.rs`.
    ///
    /// Falls back to running every head here when the split cannot help or
    /// cannot be done safely: fewer than two heads to divide, or a `seq_len`
    /// past `MAX_ATTN_SEQ`. Correct either way -- only slower.
    ///
    /// One round trip per layer, so 6 per token on the shipped model, against
    /// ~43 for the matvecs. The sync cost is charged to `prof_attn_wait_us`.
    pub fn attn_parallel(&self, job: llm_core::HeadsJob) {
        let llm_core::HeadsJob {
            q,
            kcache_layer,
            vcache_layer,
            att_out,
            pos,
            n_heads,
            head_dim,
            seq_len,
        } = job;
        let t0 = unsafe { esp_timer_get_time() };

        // SAFETY: `scores_a` is written only here, on the inference core, and
        // `attn_parallel` never runs concurrently with itself -- the decode
        // loop is sequential. The worker never touches it.
        let scores_a: &mut [f32] = unsafe {
            let arr: &mut [f32; MAX_ATTN_SEQ] = &mut *self.scores_a.get();
            &mut arr[..]
        };

        let mid = n_heads / 2;
        if mid == 0 || seq_len > MAX_ATTN_SEQ {
            // Nothing to divide, or no room for the worker's scratch. See the
            // doc comment: slower, never wrong.
            llm_core::attention::heads(
                q,
                kcache_layer,
                vcache_layer,
                att_out,
                scores_a,
                pos,
                head_dim,
                seq_len,
                0,
                n_heads,
            );
            let t1 = unsafe { esp_timer_get_time() };
            self.prof_attn_own_us
                .set(self.prof_attn_own_us.get() + (t1 - t0) as u64);
            self.prof_attn_calls.set(self.prof_attn_calls.get() + 1);
            return;
        }

        // Split the output first, then immediately reduce the worker's half to
        // a raw pointer and stop naming it -- the same structure
        // `matvec_parallel` uses, and for the same reason: a `&mut [f32]` may
        // not be live on this stack while another task writes through it.
        let (lo, hi) = att_out.split_at_mut(mid * head_dim);
        let hi_ptr = hi.as_mut_ptr();
        let hi_len = hi.len();

        self.at_q.store(q.as_ptr() as *mut f32, Ordering::Relaxed);
        self.at_kc
            .store(kcache_layer.as_ptr() as *mut f32, Ordering::Relaxed);
        self.at_vc
            .store(vcache_layer.as_ptr() as *mut f32, Ordering::Relaxed);
        self.at_kv_len.store(kcache_layer.len(), Ordering::Relaxed);
        self.at_out.store(hi_ptr, Ordering::Relaxed);
        self.at_out_len.store(hi_len, Ordering::Relaxed);
        self.at_pos.store(pos, Ordering::Relaxed);
        self.at_head_dim.store(head_dim, Ordering::Relaxed);
        self.at_seq_len.store(seq_len, Ordering::Relaxed);
        self.at_h0.store(mid, Ordering::Relaxed);
        self.at_h1.store(n_heads, Ordering::Relaxed);
        // Release: publishes every field above to the worker.
        self.job_kind.store(JOB_ATTN, Ordering::Release);

        // SAFETY: `self.worker` was set once in `stage_and_spawn`, before this
        // `Head` was reachable.
        let _ = unsafe { task::notify(self.worker.get(), NOTIFY) };

        llm_core::attention::heads(
            q,
            kcache_layer,
            vcache_layer,
            lo,
            scores_a,
            pos,
            head_dim,
            seq_len,
            0,
            mid,
        );

        let t1 = unsafe { esp_timer_get_time() };
        let _ = task::wait_notification(TickType_t::MAX);
        let t2 = unsafe { esp_timer_get_time() };
        self.prof_attn_own_us
            .set(self.prof_attn_own_us.get() + (t1 - t0) as u64);
        self.prof_attn_wait_us
            .set(self.prof_attn_wait_us.get() + (t2 - t1) as u64);
        self.prof_attn_calls.set(self.prof_attn_calls.get() + 1);
    }

    /// `(attention calls per token, ms/token in this core's heads, ms/token
    /// waiting on the worker's)`, or `None` before any timed call.
    ///
    /// `wait` is what says whether the split paid. With an even head count the
    /// two ranges are the same arithmetic, so anything much above the ~12 us
    /// round trip the matvec split measured is imbalance from somewhere else
    /// -- core 0 also services interrupts -- rather than from the work.
    pub fn attn_profile(&self) -> Option<(f32, f32, f32)> {
        let n = self.prof_calls.get();
        if n == 0 {
            return None;
        }
        let d = n as f32 * 1000.0;
        Some((
            self.prof_attn_calls.get() as f32 / n as f32,
            self.prof_attn_own_us.get() as f32 / d,
            self.prof_attn_wait_us.get() as f32 / d,
        ))
    }

    /// `(matvec calls per token, ms/token spent waiting on the worker)`.
    ///
    /// The wait figure is what says whether the split paid: it is the sync cost
    /// plus whatever imbalance the row split leaves. Large means the halves are
    /// uneven or the wake latency is worse than assumed; near zero means the
    /// two cores finish together and the split is free.
    pub fn matvec_profile(&self) -> Option<(f32, f32)> {
        let n = self.prof_calls.get();
        if n == 0 {
            return None;
        }
        Some((
            self.prof_mv_calls.get() as f32 / n as f32,
            self.prof_mv_wait_us.get() as f32 / (n as f32 * 1000.0),
        ))
    }

    /// Measure sequential read bandwidth over this head's own staged weights,
    /// returning `(bytes, MB/s)`. Uses the real buffer rather than a synthetic
    /// one so the number is directly comparable to what the head streams per
    /// token -- same allocation, same size, same direction.
    pub fn probe_weight_bandwidth(&self) -> (usize, f64) {
        (self.w4.len(), psram::probe_read_bandwidth(self.w4))
    }

    /// Stream this head's staged weights with BOTH cores at once, over
    /// disjoint halves, and report `(bytes, aggregate MB/s, this core's us,
    /// the worker's us)`.
    ///
    /// Run next to `probe_weight_bandwidth`, which does the same work on one
    /// core. The ratio between the two is the answer:
    ///
    /// * **~2.0x** -- the bus scales with cores. Attention was not contending,
    ///   and the 1.35 ms the head split did not return is the layout tax. An
    ///   fp16 KV cache would buy nothing, and this closes the question.
    /// * **~1.0x** -- the bus is saturated by one core. Attention is
    ///   memory-bound, its two cores are queueing behind each other, and
    ///   halving the KV cache's working set is worth the tolerance it costs.
    /// * **in between** -- partial contention; the fraction is roughly how
    ///   much of the shortfall memory can account for.
    ///
    /// Disjoint halves rather than the same buffer twice, deliberately: two
    /// cores reading the *same* addresses could be served by the data cache
    /// and would understate contention. Disjoint is also what attention
    /// actually does -- each core scans its own heads' regions of the cache.
    ///
    /// The overlap is not perfect: the worker starts one notify latency
    /// (~12 us, measured) after this core does. Against a ~9 ms half-pass
    /// that is 0.13%, and both per-core times are returned so an imbalance
    /// large enough to matter is visible rather than folded into the average.
    #[cfg(feature = "bandwidth-probe")]
    pub fn probe_weight_bandwidth_dual(&self) -> (usize, f64, u32, u32) {
        let n = self.w4.len();
        let half = n / 2;
        let (lo, hi) = self.w4.split_at(half);

        self.pr_ptr
            .store(lo.as_ptr() as *mut u8, Ordering::Relaxed);
        self.pr_len.store(lo.len(), Ordering::Relaxed);
        // Release: publishes pr_ptr/pr_len to the worker.
        self.job_kind.store(JOB_PROBE, Ordering::Release);

        // SAFETY: `self.worker` was set once in `stage_and_spawn`, before this
        // `Head` was reachable.
        let _ = unsafe { task::notify(self.worker.get(), NOTIFY) };

        let own_us = psram::stream_read_us(hi).max(0) as u32;
        let _ = task::wait_notification(TickType_t::MAX);
        let worker_us = self.pr_us.load(Ordering::Relaxed);

        // Aggregate throughput is total bytes over the WALL time both cores
        // were busy, which is the slower of the two -- not the sum of the two
        // rates, which would report perfect scaling even if one core sat idle
        // while the other finished.
        let wall = own_us.max(worker_us);
        let mbs = if wall == 0 {
            f64::NAN
        } else {
            (half * 2) as f64 / wall as f64
        };
        (half * 2, mbs, own_us, worker_us)
    }

    /// Is the head's inner loop limited by instructions or by waiting for
    /// memory? Returns `(cache-resident cycles/MAC, streaming cycles/MAC)`.
    ///
    /// The head spends 41.9 ms/token on non-memory time for roughly 6.1M
    /// instructions -- about 1.65 cycles per instruction on a core that issues
    /// one. Either the loop is stalling on load-use latency, or the
    /// instruction count is higher than counting mnemonics suggests. Those
    /// call for opposite fixes: software pipelining versus a shorter loop.
    ///
    /// The distinguishing measurement is free of new machinery. Run the SAME
    /// `rows_range` over a row range small enough to stay in the 64 KB data
    /// cache and repeat it; every load hits. Compare against one pass over all
    /// rows, where the weights stream from PSRAM and nothing is reused. If the
    /// two rates match, the loop is issue-bound and the fix is fewer
    /// instructions. If the cache-resident rate is markedly faster, it is
    /// waiting for memory and the fix is to overlap the waiting.
    ///
    /// Single-core on purpose: the point is per-core issue rate, and running
    /// both would fold bus contention back in -- the thing being controlled
    /// for. Timing is data-independent (integer multiply-accumulate over a
    /// fixed shape), so the zeroed activation scratch at boot is fine.
    ///
    /// # Safety
    /// `y` must point to a valid `[f32; self.rows]` this call may write.
    #[cfg(feature = "bandwidth-probe")]
    pub unsafe fn probe_dot_cycles(&self, y: *mut f32) -> (f64, f64) {
        const CPU_HZ: f64 = 240_000_000.0;
        // 12 KB of weights against a 64 KB data cache: resident after the
        // first pass, and small enough that the scale array rides along.
        let warm_rows = 256.min(self.rows);
        let reps = 64;

        // Prime, so the first timed pass is not paying the misses.
        let _ = unsafe { self.rows_range(y, 0, warm_rows) };
        let t0 = unsafe { esp_timer_get_time() };
        for _ in 0..reps {
            let _ = unsafe { self.rows_range(y, 0, warm_rows) };
        }
        let t1 = unsafe { esp_timer_get_time() };
        let warm_macs = (reps * warm_rows * self.cols) as f64;
        let warm_cyc = (t1 - t0) as f64 * 1e-6 * CPU_HZ / warm_macs;

        let t2 = unsafe { esp_timer_get_time() };
        let _ = unsafe { self.rows_range(y, 0, self.rows) };
        let t3 = unsafe { esp_timer_get_time() };
        let cold_macs = (self.rows * self.cols) as f64;
        let cold_cyc = (t3 - t2) as f64 * 1e-6 * CPU_HZ / cold_macs;

        (warm_cyc, cold_cyc)
    }

    /// The staged row count (== `vocab_n` passed to `stage_and_spawn`).
    /// Not read anywhere in this crate today (`main` gets `vocab_n`
    /// straight from `vocab::vocab_n()`, the same value, before `Head`
    /// exists) -- kept as public API for a caller that only has a `Head`
    /// in hand (a future display/debug module, a test), since recomputing
    /// it from `vocab::vocab_n()` again elsewhere risks the two silently
    /// diverging if `stage_and_spawn` is ever called with a different cap.
    #[allow(dead_code)]
    pub fn rows(&self) -> usize {
        self.rows
    }
}

/// The worker task's entry point, pinned to core 0. Ports
/// `head_worker_main`'s `for (;;) { ulTaskNotifyTake(...); head_rows_range(...); xTaskNotifyGive(...); }`
/// loop. Takes `*mut c_void` (not a typed reference) because it's handed
/// to `esp_idf_hal::task::create` as a raw `extern "C" fn` -- the `Head`
/// pointer crosses into the new task entirely through that FFI boundary,
/// not through Rust's `Send`/`Sync`, which is why `Head` needs neither
/// bound: no safe code ever observes a `Head` reference on more than one
/// thread's stack at a time; both sides manufacture their own reference
/// from the same raw address under an explicit `unsafe` block.
extern "C" fn head_worker_main(arg: *mut core::ffi::c_void) {
    // SAFETY: `arg` is always the address of the single `Head` that
    // `stage_and_spawn` leaked and passed as `task_arg` when it spawned
    // this exact task -- never freed (leaked for `'static`), never a
    // different value.
    let head: &'static Head = unsafe { &*(arg as *const Head) };
    loop {
        let _ = task::wait_notification(TickType_t::MAX);
        // Acquire pairs with the Release store of `job_kind` by whichever
        // dispatcher set this job up, and publishes that job's fields.
        match head.job_kind.load(Ordering::Acquire) {
            JOB_MATVEC => {
                let qt = head.mv_qt.load(Ordering::Relaxed);
                let xp = head.mv_x.load(Ordering::Relaxed);
                let xn = head.mv_x_len.load(Ordering::Relaxed);
                let op = head.mv_out.load(Ordering::Relaxed);
                let on = head.mv_out_len.load(Ordering::Relaxed);
                let rb = head.mv_row_begin.load(Ordering::Relaxed);
                // SAFETY: the dispatcher holds the `&QT` and both slices alive
                // across the notify/wait, and writes only rows outside
                // [rb, rb+on) itself -- see `Head::matvec_parallel`. The QT
                // borrows the 'static flash mapping.
                unsafe {
                    let t: &QT<'static> = &*qt;
                    let x = core::slice::from_raw_parts(xp as *const f32, xn);
                    let out = core::slice::from_raw_parts_mut(op, on);
                    t.matvec_rows_into(x, out, rb);
                }
            }
            #[cfg(feature = "bandwidth-probe")]
            JOB_PROBE => {
                let p = head.pr_ptr.load(Ordering::Relaxed);
                let n = head.pr_len.load(Ordering::Relaxed);
                // SAFETY: the dispatcher holds `self.w4` (a 'static PSRAM
                // allocation) alive across the notify/wait and reads only the
                // other half. Read-only on both sides, so even overlap would
                // be sound; disjoint anyway, to match what attention does.
                let dt = unsafe { psram::stream_read_us(core::slice::from_raw_parts(p, n)) };
                head.pr_us.store(dt.max(0) as u32, Ordering::Relaxed);
            }
            JOB_ATTN => {
                let dh = head.at_head_dim.load(Ordering::Relaxed);
                let seq = head.at_seq_len.load(Ordering::Relaxed);
                let h0 = head.at_h0.load(Ordering::Relaxed);
                let h1 = head.at_h1.load(Ordering::Relaxed);
                let kv_len = head.at_kv_len.load(Ordering::Relaxed);
                let pos = head.at_pos.load(Ordering::Relaxed);
                let qp = head.at_q.load(Ordering::Relaxed);
                let kcp = head.at_kc.load(Ordering::Relaxed);
                let vcp = head.at_vc.load(Ordering::Relaxed);
                let op = head.at_out.load(Ordering::Relaxed);
                let on = head.at_out_len.load(Ordering::Relaxed);
                // SAFETY: `Head::attn_parallel` holds `q`, both caches and its
                // own half of `att_out` alive across the notify/wait, and
                // writes only heads [0, h0) -- a disjoint prefix of the same
                // output buffer. The caches are read-only for both cores here:
                // `store_kv` ran in `llm_forward_impl` before either was
                // notified. `scores_b` is this task's alone; see the field.
                unsafe {
                    let q = core::slice::from_raw_parts(qp as *const f32, h1 * dh);
                    let kc = core::slice::from_raw_parts(kcp as *const f32, kv_len);
                    let vc = core::slice::from_raw_parts(vcp as *const f32, kv_len);
                    let out = core::slice::from_raw_parts_mut(op, on);
                    let sc: &mut [f32; MAX_ATTN_SEQ] = &mut *head.scores_b.get();
                    llm_core::attention::heads(
                        q,
                        kc,
                        vc,
                        out,
                        &mut sc[..],
                        pos,
                        dh,
                        seq,
                        h0,
                        h1,
                    );
                }
            }
            _ => {
                let split = head.job_split.load(Ordering::Relaxed);
                let y_ptr = head.job_y.load(Ordering::Relaxed);
                // SAFETY: rows [0, split) belong exclusively to this call until
                // the notify below -- `Head::matvec`, which set up this job,
                // only ever writes [split, rows) itself.
                let w0 = unsafe { esp_timer_get_time() };
                let (lo_val, lo_idx) = unsafe { head.rows_range(y_ptr, 0, split) };
                let w1 = unsafe { esp_timer_get_time() };
                head.lo_max.store(lo_val.to_bits(), Ordering::Relaxed);
                // Release: publishes lo_max alongside lo_arg to whoever reads
                // them after this task's notify below.
                head.lo_arg.store(lo_idx, Ordering::Release);
                // Only this task writes this counter.
                head.prof_worker_us
                    .set(head.prof_worker_us.get() + (w1 - w0) as u64);
            }
        }
        let _ = unsafe { task::notify(head.inference_task, NOTIFY) };
    }
}

/// Ports the C reference's boot-time sequence: cap `tok_emb.rows` to the
/// trained vocab count, stage it as int8 in PSRAM
/// (`model.tok_emb.rows = VOCAB_N; stage_head_int8(&model.tok_emb);`),
/// record the calling task as `inference_task`
/// (`xTaskGetCurrentTaskHandle()`), and spawn the worker pinned to core 0
/// (`xTaskCreatePinnedToCore(head_worker_main, "head", 4096, NULL, 2,
/// &head_worker, 0)`). The C reference's final step,
/// `model.head_matvec = head_matvec_int8;`, has no direct equivalent here
/// -- `llm-firmware`'s `main` wires `Head::matvec` into
/// `llm_forward_with_head_override` directly instead of through a
/// function-pointer field on `Model`, since this port's `Model` has no
/// such field (see `llm_core::model`'s doc comment on why the override is
/// a closure parameter instead).
///
/// `tok_emb` should be `model.tok_emb()` (the tied embedding/output-head
/// tensor); `vocab_n` is the trained vocab count from `vocab::vocab_n()`
/// (NOT `model.cfg.vocab`, which includes padding rows above the trained
/// vocab that have no decode entry and are never scored -- same
/// distinction the C reference's `VOCAB_N` vs `Cfg.vocab` makes).
pub fn stage_and_spawn(tok_emb: QT<'static>, vocab_n: usize) -> Result<&'static Head, &'static str> {
    let mut t = tok_emb;
    t.rows = vocab_n;
    let rows = t.rows;
    let cols = t.cols;
    if cols > MAX_HEAD_COLS {
        // Mirrors the C reference's fixed `head_actq[128]` -- it has the
        // same implicit ceiling, just never hit it on the shipped models
        // (cols == dim == 96). Caught here instead of silently truncating.
        return Err("model dim exceeds MAX_HEAD_COLS (128) -- head_actq scratch too small");
    }

    let row_bytes = t.row_bytes;
    let (w4, scale8) = psram::alloc_head_tables(rows * row_bytes, rows);
    for r in 0..rows {
        let (codes, scale) = t.packed_row(r);
        w4[r * row_bytes..(r + 1) * row_bytes].copy_from_slice(codes);
        scale8[r] = scale;
    }
    log::info!(
        "head staged int4: {:.2} MB (int8 staging would be {:.2} MB)",
        (rows * row_bytes + rows * 4) as f64 / 1e6,
        (rows * cols + rows * 4) as f64 / 1e6
    );

    let inference_task =
        task::current().ok_or("no current FreeRTOS task handle -- call stage_and_spawn from main")?;

    let head: &'static Head = Box::leak(Box::new(Head {
        w4,
        scale8,
        rows,
        cols,
        row_bytes,
        actq: UnsafeCell::new([0i16; MAX_HEAD_COLS]),
        acts: AtomicU32::new(0),
        act_sum: AtomicI32::new(0),
        job_y: core::sync::atomic::AtomicPtr::new(core::ptr::null_mut()),
        job_split: AtomicUsize::new(0),
        worker: Cell::new(core::ptr::null_mut()),
        inference_task,
        prof_quant_us: Cell::new(0),
        prof_own_us: Cell::new(0),
        prof_wait_us: Cell::new(0),
        prof_worker_us: Cell::new(0),
        prof_calls: Cell::new(0),
        lo_max: AtomicU32::new(f32::NEG_INFINITY.to_bits()),
        lo_arg: AtomicUsize::new(0),
        hi_max: AtomicU32::new(f32::NEG_INFINITY.to_bits()),
        hi_arg: AtomicUsize::new(0),
        job_kind: AtomicU32::new(JOB_HEAD),
        mv_qt: AtomicPtr::new(core::ptr::null_mut()),
        mv_x: AtomicPtr::new(core::ptr::null_mut()),
        mv_x_len: AtomicUsize::new(0),
        mv_out: AtomicPtr::new(core::ptr::null_mut()),
        mv_out_len: AtomicUsize::new(0),
        mv_row_begin: AtomicUsize::new(0),
        prof_mv_wait_us: Cell::new(0),
        prof_mv_calls: Cell::new(0),
        at_q: AtomicPtr::new(core::ptr::null_mut()),
        at_kc: AtomicPtr::new(core::ptr::null_mut()),
        at_vc: AtomicPtr::new(core::ptr::null_mut()),
        at_kv_len: AtomicUsize::new(0),
        at_out: AtomicPtr::new(core::ptr::null_mut()),
        at_out_len: AtomicUsize::new(0),
        at_pos: AtomicUsize::new(0),
        at_head_dim: AtomicUsize::new(0),
        at_seq_len: AtomicUsize::new(0),
        at_h0: AtomicUsize::new(0),
        at_h1: AtomicUsize::new(0),
        scores_a: UnsafeCell::new([0f32; MAX_ATTN_SEQ]),
        scores_b: UnsafeCell::new([0f32; MAX_ATTN_SEQ]),
        prof_attn_wait_us: Cell::new(0),
        prof_attn_own_us: Cell::new(0),
        prof_attn_calls: Cell::new(0),
        #[cfg(feature = "bandwidth-probe")]
        pr_ptr: AtomicPtr::new(core::ptr::null_mut()),
        #[cfg(feature = "bandwidth-probe")]
        pr_len: AtomicUsize::new(0),
        #[cfg(feature = "bandwidth-probe")]
        pr_us: AtomicU32::new(0),
    }));

    // SAFETY: `head_worker_main` is a valid `extern "C" fn(*mut c_void)`;
    // the pointer we pass it (`head`'s own address) stays valid forever
    // (leaked, 'static) and is exactly what `head_worker_main` expects to
    // cast back. Stack size (4096) and priority (2) match the C
    // reference's `xTaskCreatePinnedToCore(..., 4096, NULL, 2, ..., 0)`
    // exactly; core 0 likewise (`Core::Core0`).
    let worker = unsafe {
        task::create(
            head_worker_main,
            c"head",
            4096,
            head as *const Head as *mut core::ffi::c_void,
            2,
            Some(Core::Core0),
        )
    }
    .map_err(|_| "head worker task creation failed")?;
    head.worker.set(worker);

    Ok(head)
}
