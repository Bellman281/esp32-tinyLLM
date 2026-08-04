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

const MODEL_BIN: &str = "../../reference-c/esp32-llm-lab/model.bin";
const VOCAB_H: &str = "../../reference-c/esp32-llm-lab/vocab.h";
const EXPECTED: &str = include_str!("fixtures/c_host_generate_stdout.txt");

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

    let exe = env!("CARGO_BIN_EXE_host-generate");

    let output = Command::new(exe)
        .args([MODEL_BIN, VOCAB_H])
        .output()
        .expect("failed to run host-generate");
    assert!(output.status.success(), "host-generate exited non-zero");

    let stdout = String::from_utf8(output.stdout).expect("stdout should be valid UTF-8");
    assert_eq!(
        stdout, EXPECTED,
        "host-generate's CLI output diverged from the captured C reference"
    );
}
