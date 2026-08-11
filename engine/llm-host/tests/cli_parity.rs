//! CLI parity gate: builds `host-generate` and diffs its actual stdout
//! against a byte-for-byte capture of the C reference binary
//! (`reference-c/esp32-llm-lab/host_generate.c`, compiled with plain `gcc
//! -O2` and run against the same model.bin/vocab.h). This is a coarser but
//! complementary check to `golden.rs`: golden.rs proves the raw logits
//! match PyTorch; this proves the full CLI (greedy decode loop, vocab
//! lookup, argmax-over-VOCAB_N bound) produces the exact same token
//! stream and text as the original C program end to end.
//!
//! Fixture captured once via (note the C original takes N as argv[2] and
//! has vocab.h #include'd at compile time -- NOT a runtime path; see the
//! `vocab.h is a runtime argument here` deviation note in host_generate.rs):
//!   gcc -O2 -I reference-c/common -o host_generate reference-c/host_tools/host_generate.c -lm
//!   ./host_generate reference-c/model/model_14p9mb.bin \
//!       > tests/fixtures/c_host_generate_stdout.txt   # default N=120

use std::process::Command;

const EXPECTED: &str = include_str!("fixtures/c_host_generate_stdout.txt");

/// Where `cargo build --bin host-generate` actually puts the binary.
///
/// BUG FIX: this used to be `CARGO_MANIFEST_DIR + "/target/release/..."`,
/// which is wrong for a workspace member -- `llm-host` is listed in
/// `engine/Cargo.toml`'s `[workspace] members`, so its build artifacts land
/// in the *workspace root's* shared `target/` dir (`engine/target/`), not
/// a `target/` folder next to `llm-host`'s own `Cargo.toml`. The old path
/// only ever resolved by accident, from a stale non-workspace-style
/// `llm-host/target/` directory left over from earlier in this project's
/// history (before llm-core/llm-host were combined under one workspace) --
/// on a genuinely fresh checkout it doesn't exist, which is exactly the
/// `Os { code: 2, kind: NotFound }` this crashed with. `CARGO_MANIFEST_DIR`
/// is still the right starting point; it just needs `/../target/...`
/// (workspace root == this crate's parent directory), not `/target/...`.
fn host_generate_exe() -> String {
    env!("CARGO_MANIFEST_DIR").to_string() + "/../target/release/host-generate"
}

#[test]
fn host_generate_cli_matches_c_reference_byte_for_byte() {
    // Force a build first so the run below isn't polluted by cargo's own
    // "Compiling.../Finished..." chatter landing on stderr mid-process --
    // we only care about this binary's own stdout.
    let build = Command::new(env!("CARGO"))
        .args(["build", "--release", "--bin", "host-generate"])
        .status()
        .expect("failed to invoke cargo build");
    assert!(build.success(), "cargo build --bin host-generate failed");

    let entry = llm_host::manifest::default_model();
    let output = Command::new(host_generate_exe())
        .args([entry.bin_path(), entry.vocab_path()])
        .output()
        .expect("failed to run host-generate");
    assert!(output.status.success(), "host-generate exited non-zero");

    let stdout = String::from_utf8(output.stdout).expect("stdout should be valid UTF-8");
    assert_eq!(
        stdout, EXPECTED,
        "host-generate's CLI output diverged from the captured C reference"
    );
}

/// int8-activations counterpart, added after noticing the existing parity
/// suite (golden.rs, ppl_parity.rs) only ever checks the int8 path
/// *numerically* (max-abs/rms diff, cross-entropy within a noise floor) --
/// nothing asserted that a full greedy-decode CLI run, token by token,
/// actually lands on the exact same argmax choice as the C reference at
/// every step under quantized activations. This closes that gap the same
/// way the fp32 test above does: real compiled C binary vs real compiled
/// Rust binary, byte-for-byte stdout diff, both built with the matching
/// quantized-activations path.
///
/// Fixture captured via:
///   gcc -O2 -DLLM_INT8_ACT -I reference-c/common \
///       -o host_generate_int8 reference-c/host_tools/host_generate.c -lm
///   ./host_generate_int8 reference-c/esp32-llm-lab/model.bin 120 \
///       > tests/fixtures/c_host_generate_stdout_int8.txt
///
/// NOTE for future readers: this fixture is currently byte-identical to
/// the fp32 one above. That's not a copy-paste mistake -- at this model
/// size and this short a run (120 tokens from a 4-token prompt), int8
/// activation quantization happened not to flip any single argmax
/// decision (consistent with ppl_parity.rs's own finding that int8 vs
/// fp32 divergence is a subtle, gradually-compounding effect that only
/// becomes numerically visible over much longer windows -- 2048
/// predictions there, vs 120 tokens here). If a future model/prompt/N
/// change ever makes these two fixture files diverge, that is expected
/// and should be re-captured, not treated as a regression by itself.
#[test]
#[cfg(feature = "int8-activations")]
fn host_generate_cli_matches_c_reference_byte_for_byte_int8_activations() {
    const EXPECTED_INT8: &str = include_str!("fixtures/c_host_generate_stdout_int8.txt");

    let build = Command::new(env!("CARGO"))
        .args([
            "build",
            "--release",
            "--bin",
            "host-generate",
            "--features",
            "int8-activations",
        ])
        .status()
        .expect("failed to invoke cargo build");
    assert!(
        build.success(),
        "cargo build --bin host-generate --features int8-activations failed"
    );

    let entry = llm_host::manifest::default_model();
    let output = Command::new(host_generate_exe())
        .args([entry.bin_path(), entry.vocab_path()])
        .output()
        .expect("failed to run host-generate");
    assert!(output.status.success(), "host-generate exited non-zero");

    let stdout = String::from_utf8(output.stdout).expect("stdout should be valid UTF-8");
    assert_eq!(
        stdout, EXPECTED_INT8,
        "host-generate's int8-activations CLI output diverged from the captured C reference"
    );
}
