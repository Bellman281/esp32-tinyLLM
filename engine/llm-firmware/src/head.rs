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
use core::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use esp_idf_svc::hal::cpu::Core;
use esp_idf_svc::hal::sys::{TaskHandle_t, TickType_t};
use esp_idf_svc::hal::task;
use llm_core::{dot_int4_int8, quantize_activations, QT};

use crate::psram;

/// Any nonzero value works as the notification token -- FreeRTOS task
/// notifications used this way (`xTaskNotifyGive`/`ulTaskNotifyTake`) are
/// a binary hand-off, not a counted semaphore or a message payload.
const NOTIFY: core::num::NonZeroU32 = core::num::NonZeroU32::new(1).unwrap();

/// C's `head_actq[128]`: "the max input dim across the model is 128" --
/// this head's actual cols (== model dim, 96 on the shipped models) is
/// always <= this, same fixed-size-scratch choice the C reference makes.
const MAX_HEAD_COLS: usize = 128;

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
    actq: UnsafeCell<[i8; MAX_HEAD_COLS]>,
    acts: AtomicU32, // f32 bits (no AtomicF32 in core)

    // The current job's output pointer and row split, set by `matvec`
    // before notifying the worker.
    job_y: core::sync::atomic::AtomicPtr<f32>,
    job_split: AtomicUsize,

    worker: Cell<TaskHandle_t>, // set once, right after task::create succeeds
    inference_task: TaskHandle_t, // set once at construction, never changes
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
    unsafe fn rows_range(&self, y: *mut f32, r0: usize, r1: usize) {
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
        let actq_all: &[i8; MAX_HEAD_COLS] = unsafe { &*self.actq.get() };
        let actq: &[i8] = &actq_all[..cols];
        let acts = f32::from_bits(self.acts.load(Ordering::Acquire));
        for r in r0..r1 {
            // Two codes per byte, unpacked in the loop instead of at boot.
            // `llm_core::dot_int4_int8` is asserted bit-for-bit equal to the
            // staged-int8 dot over every row of the shipped model by
            // llm-host/tests/int4_head_equivalence.rs -- integer addition is
            // associative and every term is the same term, so this is exact,
            // not approximate.
            let acc = dot_int4_int8(&w4[r * rb..r * rb + rb], actq);
            let val = acc as f32 * scale8[r] * acts;
            // SAFETY: caller's contract (see this fn's doc comment) is that
            // `y[r0..r1]` is this call's alone to write.
            unsafe { core::ptr::write(y.add(r), val) };
        }
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

        // SAFETY: this function only ever runs on the inference core
        // (never concurrently with itself -- `llm_forward` is called in a
        // sequential decode loop), and the worker never writes `actq`, so
        // this is the sole writer.
        let actq: &mut [i8] = unsafe {
            let arr: &mut [i8; MAX_HEAD_COLS] = &mut *self.actq.get();
            &mut arr[..self.cols]
        };
        let scale = quantize_activations(&x[..self.cols], actq);
        self.acts.store(scale.to_bits(), Ordering::Release);

        // Capture the raw pointer once; from here on both this core and
        // the worker index through raw pointers into the same allocation,
        // on disjoint ranges -- never through a typed `&mut [f32]` again
        // (see module doc comment).
        let y_ptr = y.as_mut_ptr();
        let split = self.rows / 2;
        self.job_split.store(split, Ordering::Relaxed);
        self.job_y.store(y_ptr, Ordering::Release);

        // SAFETY: `self.worker` was set exactly once in `stage_and_spawn`,
        // before this `Head` was ever returned/reachable, so it's always
        // valid by the time any caller can reach `matvec`.
        let _ = unsafe { task::notify(self.worker.get(), NOTIFY) };

        // SAFETY: rows [split, self.rows) belong exclusively to this call
        // until it returns -- the worker task, notified above, only ever
        // writes [0, split) (see `head_worker_main`). Disjoint index
        // ranges of the same buffer, concurrently, by construction.
        unsafe { self.rows_range(y_ptr, split, self.rows) };

        // Blocks until `head_worker_main`'s `task::notify` (its half is
        // done). `TickType_t::MAX` is this port's equivalent of C's
        // `portMAX_DELAY` (wait forever) -- see this module's README
        // entry for the verification caveat on that mapping.
        let _ = task::wait_notification(TickType_t::MAX);
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
        let split = head.job_split.load(Ordering::Acquire);
        let y_ptr = head.job_y.load(Ordering::Acquire);
        // SAFETY: rows [0, split) belong exclusively to this call until
        // the notify below -- `Head::matvec`, which set up this job, only
        // ever writes [split, rows) itself (see its doc comment).
        unsafe { head.rows_range(y_ptr, 0, split) };
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
        actq: UnsafeCell::new([0i8; MAX_HEAD_COLS]),
        acts: AtomicU32::new(0),
        job_y: core::sync::atomic::AtomicPtr::new(core::ptr::null_mut()),
        job_split: AtomicUsize::new(0),
        worker: Cell::new(core::ptr::null_mut()),
        inference_task,
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
