//! Guards the one mistake that already cost this project a wrong published
//! conclusion: the C firmware and the Rust firmware benchmarking at
//! *different settings*, and nobody noticing.
//!
//! `attn` is the only stage whose cost scales with sequence position (it
//! scans `0..=pos` of the KV cache every token), so a different
//! `N_GENERATE` or prompt length moves it while leaving every other stage
//! byte-identical. That is not hypothetical: the C sketch shipped with
//! `N_GENERATE = 200` and a 4-token prompt while the Rust firmware used
//! 400 and 18, and comparing the two produced an attention ratio of 2.10x
//! that was published in this repo's README as the port's worst stage. At
//! matched settings it is 1.25x — the *best* stage. The measurement was
//! never wrong; the two runs were simply not comparable.
//!
//! `model/models.toml` now declares `prompt_ids` and `n_generate` once.
//! This test asserts that both firmwares actually agree with it. Without
//! it the registry only *documents* those values while three files
//! independently hardcode them — which is documentation, not a single
//! source of truth.
//!
//! Why it parses source text: neither firmware can read a TOML file at
//! runtime (the prompt is compiled in, and on-device there is no
//! filesystem to read from), and `llm-firmware` is a standalone Cargo
//! project outside this workspace, so its constants can't simply be
//! imported. Scanning the two declarations is the cheapest thing that
//! actually catches drift. It assumes the literals contain no comments
//! between their delimiters, which is true today and fails loudly (a parse
//! error naming the file) rather than silently if that changes.
//!
//! Generating these constants at build time from the registry would remove
//! the duplication outright, rather than detecting it after the fact. That
//! is the better long-term fix; this test is what makes the current
//! arrangement safe in the meantime.

use llm_host::manifest;
use std::fs;
use std::path::Path;

const C_FIRMWARE: &str = "reference-c/firmware/esp32_llm/esp32_llm.ino";
const RUST_FIRMWARE: &str = "engine/llm-firmware/src/main.rs";

fn read_at_root(rel: &str) -> String {
    let p = Path::new(manifest::REPO_ROOT).join(rel);
    fs::read_to_string(&p)
        .unwrap_or_else(|e| panic!("read {}: {e} (run from engine/llm-host)", p.display()))
}

/// Everything between `marker` and the next `close`, parsed as a
/// comma-separated integer list. Trailing commas are tolerated (Rust's
/// array literal has one).
fn ints_after(haystack: &str, file: &str, marker: &str, close: char) -> Vec<usize> {
    let start = haystack
        .find(marker)
        .unwrap_or_else(|| panic!("{file}: no {marker:?} found"));
    let rest = &haystack[start + marker.len()..];
    let end = rest
        .find(close)
        .unwrap_or_else(|| panic!("{file}: {marker:?} is not closed by {close:?}"));
    rest[..end]
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse().unwrap_or_else(|e| {
                panic!("{file}: {marker:?} contains a non-integer element {s:?}: {e}")
            })
        })
        .collect()
}

/// The integer immediately following `marker`, up to `close`.
fn int_after(haystack: &str, file: &str, marker: &str, close: char) -> usize {
    let start = haystack
        .find(marker)
        .unwrap_or_else(|| panic!("{file}: no {marker:?} found"));
    let rest = &haystack[start + marker.len()..];
    let end = rest
        .find(close)
        .unwrap_or_else(|| panic!("{file}: {marker:?} is not terminated by {close:?}"));
    rest[..end].trim().parse().unwrap_or_else(|e| {
        panic!("{file}: {marker:?} is not followed by an integer: {e}")
    })
}

#[test]
fn both_firmwares_use_the_registry_benchmark_settings() {
    let entry = manifest::default_model();
    let want_ids = &entry.prompt_ids;
    let want_n = entry
        .n_generate
        .expect("default model needs `n_generate` in model/models.toml");
    assert!(
        !want_ids.is_empty(),
        "default model needs a non-empty `prompt_ids` in model/models.toml"
    );

    let c_src = read_at_root(C_FIRMWARE);
    let c_ids = ints_after(&c_src, C_FIRMWARE, "PROMPT_IDS[] = {", '}');
    let c_n = int_after(&c_src, C_FIRMWARE, "N_GENERATE = ", ';');

    let rust_src = read_at_root(RUST_FIRMWARE);
    let rust_ids = ints_after(&rust_src, RUST_FIRMWARE, "DEFAULT_PROMPT_IDS: [usize; 18] = [", ']');
    let rust_n = int_after(&rust_src, RUST_FIRMWARE, "const N_GENERATE: usize = ", ';');

    assert_eq!(
        &c_ids, want_ids,
        "\n{C_FIRMWARE}'s PROMPT_IDS disagrees with model/models.toml's prompt_ids.\n\
         Benchmarks taken from these two are NOT comparable -- see this file's doc comment."
    );
    assert_eq!(
        &rust_ids, want_ids,
        "\n{RUST_FIRMWARE}'s DEFAULT_PROMPT_IDS disagrees with model/models.toml's prompt_ids.\n\
         Benchmarks taken from these two are NOT comparable -- see this file's doc comment."
    );
    assert_eq!(
        c_n, want_n,
        "\n{C_FIRMWARE}'s N_GENERATE ({c_n}) disagrees with model/models.toml's n_generate ({want_n}).\n\
         This is the exact mismatch that produced a wrong 2.10x attention ratio -- see this file's doc comment."
    );
    assert_eq!(
        rust_n, want_n,
        "\n{RUST_FIRMWARE}'s N_GENERATE ({rust_n}) disagrees with model/models.toml's n_generate ({want_n}).\n\
         This is the exact mismatch that produced a wrong 2.10x attention ratio -- see this file's doc comment."
    );

    eprintln!(
        "benchmark settings agree across registry / C / Rust: {} prompt ids, n_generate = {want_n}",
        want_ids.len()
    );
}
