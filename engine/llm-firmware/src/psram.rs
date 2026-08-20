//! PSRAM allocation for `llm-firmware`'s scratch buffers and the staged
//! int8 head -- ports the C reference's `ps()` helper and the block of
//! `s.x = (float *)ps(D * 4); ...` calls in `setup()`
//! (`firmware/esp32_llm/esp32_llm.ino`).
//!
//! VERIFICATION STATUS: not compiled. `heap_caps_malloc` and
//! `heap_caps_get_free_size` are confirmed to exist at
//! `esp_idf_svc::hal::sys::*` (bindgen-generated 1:1 from ESP-IDF's
//! `esp_heap_caps.h`) -- `heap_caps_malloc`'s exact signature was checked
//! directly; `heap_caps_get_free_size` was not independently checked but is
//! the same header, same calling convention (`(caps: u32) -> size_t`), so
//! it's the same confidence tier as the rest of this file's raw FFI calls.
//!
//! Unlike the C reference (`malloc`, uninitialized), every buffer here is
//! zero-filled once at alloc time. The C code gets away with uninitialized
//! scratch because every read happens after a matching write within the
//! same `llm_forward` call (kcache/vcache included -- attention only ever
//! reads `t <= pos`, which by construction has already been written this
//! decode). That invariant is real but implicit; zero-filling costs one
//! pass over a few MB at boot (once, not per-token) and removes any need
//! to reason about it here.

use esp_idf_svc::hal::sys::{
    esp_timer_get_time, heap_caps_get_free_size, heap_caps_malloc, MALLOC_CAP_SPIRAM,
};
use llm_core::{Cfg, Scratch};

/// Ports `ps()`: PSRAM-malloc `n` bytes or halt. The C reference prints and
/// spins forever on failure rather than returning an error (there's no
/// caller in `setup()` that could recover from a partial scratch
/// allocation), so this does the same -- `-> *mut u8` never actually
/// returns null.
fn ps(n_bytes: usize) -> *mut u8 {
    // SAFETY: `heap_caps_malloc` is ESP-IDF's own allocator; passing a
    // byte count and a capability mask is exactly its documented contract.
    let p = unsafe { heap_caps_malloc(n_bytes, MALLOC_CAP_SPIRAM) };
    if p.is_null() {
        log::error!("PSRAM alloc failed ({n_bytes} bytes)");
        loop {
            // Matches the C reference's `while (1) delay(1000);` -- halt
            // rather than run with a null buffer. `std::thread::sleep` is
            // available (esp-idf-svc's std target has a real scheduler),
            // so this actually yields instead of busy-spinning.
            std::thread::sleep(std::time::Duration::from_secs(1));
        }
    }
    p as *mut u8
}

/// Allocates `n` elements of `T` in PSRAM, zeroed, and leaks them as a
/// `'static` slice -- these buffers live for the entire process, same as
/// the C reference's `ps()`-allocated pointers, which are never freed.
///
/// # Safety
/// `T` must be safely representable by an all-zero bit pattern (true for
/// `f32`, `i8`, `u8` -- the only types this module instantiates it with).
unsafe fn ps_slice<T: Copy>(n: usize) -> &'static mut [T] {
    let n_bytes = n * core::mem::size_of::<T>();
    let p = ps(n_bytes) as *mut T;
    core::ptr::write_bytes(p as *mut u8, 0, n_bytes);
    core::slice::from_raw_parts_mut(p, n)
}

/// Allocates every `Scratch` buffer in PSRAM, ports the C reference's
/// `s.x = (float *)ps(D * 4); s.h = (float *)ps((F > D ? F : D) * 4); ...`
/// block verbatim (same sizes -- computed the same way `llm_core::scratch_sizes`
/// computes them, since that function ports the same C expressions). The
/// `'static` lifetime is honest: like the C globals `Model model; Scratch
/// s;`, these buffers are allocated once in `setup()`-equivalent code and
/// live until the device resets.
pub fn alloc_scratch_psram(cfg: &Cfg) -> Scratch<'static> {
    let sz = llm_core::scratch_sizes(cfg);
    // SAFETY: f32/i8 slices, zero is a valid bit pattern for both.
    unsafe {
        Scratch {
            x: ps_slice(sz.x),
            h: ps_slice(sz.h),
            qkv: ps_slice(sz.qkv),
            att: ps_slice(sz.att),
            g1: ps_slice(sz.g1),
            g2: ps_slice(sz.g2),
            ple: ps_slice(sz.ple),
            tmp_p: ps_slice(sz.tmp_p),
            trow: ps_slice(sz.trow),
            rope_cos: ps_slice(sz.rope_cos),
            rope_sin: ps_slice(sz.rope_sin),
            logits: ps_slice(sz.logits),
            scores: ps_slice(sz.scores),
            kcache: ps_slice(sz.kcache),
            vcache: ps_slice(sz.vcache),
            iq: ps_slice(sz.iq),
        }
    }
}

/// Allocates the staged head's two PSRAM tables (`head_w8`/`head_scale8` in
/// the C reference). Split out from `head::stage_and_spawn` so this module
/// stays the single place that knows how to talk to `heap_caps_malloc`.
///
/// `weight_bytes` is `rows * cols` for the int8 staging the C reference does,
/// or `rows * row_bytes` (half that) when the head is staged as packed int4 --
/// see `head::stage_and_spawn`. Taking bytes rather than (rows, cols) keeps
/// that choice in one place instead of two.
pub fn alloc_head_tables(weight_bytes: usize, rows: usize) -> (&'static mut [u8], &'static mut [f32]) {
    // SAFETY: u8/f32, zero is a valid bit pattern for both; every element is
    // overwritten by the caller immediately after this returns, before any of
    // it is read.
    unsafe { (ps_slice(weight_bytes), ps_slice(rows)) }
}

/// Ports `heap_caps_get_free_size(MALLOC_CAP_SPIRAM)` for the boot log line
/// `"PSRAM free after alloc: %u KB\n"`.
pub fn free_psram_bytes() -> usize {
    // SAFETY: no arguments beyond the capability mask, no preconditions
    // beyond the heap being initialized (true by the time `main` runs).
    unsafe { heap_caps_get_free_size(MALLOC_CAP_SPIRAM) as usize }
}

/// Sequential read bandwidth of `buf`, in MB/s, measured by streaming it once.
///
/// The minimum useful piece of `reference-c/firmware/bandwidth_bench/` (Phase 5,
/// still unported): enough to turn "the head is probably memory-bound" into a
/// number. Point it at the staged head weights and the answer is directly
/// comparable to what the head does per token -- same buffer, same direction,
/// same cold-stream access pattern, nothing cached between passes because it is
/// far larger than the data cache.
///
/// Read it as an upper bound on how much of the head's time is memory. If the
/// head streams `n` MB per token and this reports `b` MB/s, then at least
/// `head_ms - 1000*n/b` is arithmetic, and no amount of shrinking the weights
/// can touch it. That bound is what tells you whether to spend the next effort
/// on bytes or on instructions.
///
/// Accumulates rather than using `read_volatile` per word: a volatile load per
/// `u32` blocks the unrolling and pipelining the real head loop gets, and would
/// understate bandwidth. `black_box` on the result is what stops the loop being
/// deleted.
pub fn probe_read_bandwidth(buf: &[u8]) -> f64 {
    let dt = stream_read_us(buf);
    if dt <= 0 {
        return f64::NAN;
    }
    (buf.len() / 4 * 4) as f64 / dt as f64
}

/// Stream `buf` once and return the elapsed microseconds.
///
/// The timing half of `probe_read_bandwidth`, split out so two cores can each
/// time their own run and the caller can combine them. Bandwidth is a property
/// of the bus, not of one core: with the reads happening on one core at a time
/// there is no way to tell a bus that saturates at 68 MB/s from one that
/// delivers 68 MB/s per core, and those imply opposite things about what to
/// optimize next. See `Head::probe_weight_bandwidth_dual`.
pub fn stream_read_us(buf: &[u8]) -> i64 {
    let words = buf.len() / 4;
    let p = buf.as_ptr() as *const u32;
    let t0 = unsafe { esp_timer_get_time() };
    let mut acc: u32 = 0;
    for i in 0..words {
        // SAFETY: `i < words == buf.len()/4`, and `buf` is at least
        // `words * 4` bytes, so every read is inside it. u32 has no
        // alignment requirement this allocator can violate -- PSRAM
        // allocations are at least word-aligned.
        acc = acc.wrapping_add(unsafe { *p.add(i) });
    }
    let dt = unsafe { esp_timer_get_time() } - t0;
    core::hint::black_box(acc);
    dt
}
