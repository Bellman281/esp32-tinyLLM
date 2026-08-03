//! Maps the flash `model` partition into address space so `llm_core::Model::load`
//! can read it as an ordinary `&'static [u8]` -- no copy, same as the C
//! reference's `esp_partition_mmap`.
//!
//! VERIFICATION STATUS: not compiled. `esp_idf_svc::hal::sys::esp_partition_find_first`
//! and `esp_partition_mmap` themselves are bindgen-generated 1:1 from
//! ESP-IDF's C API (documented at
//! https://docs.espressif.com/projects/esp-idf/en/stable/esp32s3/api-reference/storage/partition.html)
//! and match the C reference's own calls closely enough that the function
//! names/argument order are the least likely thing to be wrong here. What
//! genuinely cannot be verified without a build is the exact
//! bindgen-mangled spelling of the C enum constants below
//! (`ESP_PARTITION_TYPE_DATA`, `ESP_PARTITION_MMAP_DATA`) -- esp-idf-sys
//! typically renders a C `typedef enum { FOO_BAR } foo_t;` constant as the
//! flat Rust item `foo_t_FOO_BAR`, which is what's used here, but if the
//! first `cargo build` errors on one of these two names specifically, grep
//! the generated bindings (`$IDF_PATH`/build output under
//! `target/.../bindings.rs`) for the real spelling and fix it here -- ESP-IDF's
//! own generated code is the authority, not this comment.

use esp_idf_svc::hal::sys::{
    esp_partition_find_first, esp_partition_mmap, esp_partition_mmap_memory_t_ESP_PARTITION_MMAP_DATA,
    esp_partition_subtype_t, esp_partition_type_t_ESP_PARTITION_TYPE_DATA, EspError,
};

/// Ports the C setup()'s partition lookup + mmap:
/// ```c
/// const esp_partition_t *part = esp_partition_find_first(
///     ESP_PARTITION_TYPE_DATA, (esp_partition_subtype_t)0x40, "model");
/// esp_partition_mmap(part, 0, part->size, ESP_PARTITION_MMAP_DATA, &base, &h);
/// ```
/// Returns the mapped region as a `&'static [u8]` -- 'static because the
/// mapping lives for the rest of the program, same as the C code (it's
/// never unmapped). The mmap handle itself is intentionally leaked, not
/// dropped: this firmware runs the model forever from `setup()`'s single
/// pass, same as the reference.
pub fn map_model_partition() -> Result<&'static [u8], &'static str> {
    unsafe {
        // subtype 0x40: the custom "model" data partition type this project's
        // partitions.csv defines (`model, data, 0x40, ...`) -- matches the C
        // reference's `(esp_partition_subtype_t)0x40` cast exactly.
        let label = c"model";
        let part = esp_partition_find_first(
            esp_partition_type_t_ESP_PARTITION_TYPE_DATA,
            0x40u32 as esp_partition_subtype_t,
            label.as_ptr(),
        );
        if part.is_null() {
            return Err("model partition not found (check partitions.csv was flashed)");
        }

        let size = (*part).size as usize;
        let mut mmap_base: *const core::ffi::c_void = core::ptr::null();
        let mut mmap_handle: esp_idf_svc::hal::sys::esp_partition_mmap_handle_t = 0;
        let err = esp_partition_mmap(
            part,
            0,
            size,
            esp_partition_mmap_memory_t_ESP_PARTITION_MMAP_DATA,
            &mut mmap_base,
            &mut mmap_handle,
        );
        if err != 0 {
            return Err("esp_partition_mmap failed");
        }
        // Leaked on purpose (see doc comment) -- never munmap'd for the
        // lifetime of the process, so treating it as 'static is sound.
        Ok(core::slice::from_raw_parts(mmap_base as *const u8, size))
    }
}

/// Trivial newtype so callers don't need to import EspError just to bubble
/// one up from code that otherwise only touches this module's plain
/// `Result<_, &str>` -- kept for parity with esp-idf-svc's own error type
/// even though this module doesn't currently produce one itself.
#[allow(dead_code)]
type _Unused = EspError;
