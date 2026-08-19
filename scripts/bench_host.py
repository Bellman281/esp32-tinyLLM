#!/usr/bin/env python3
"""Benchmark the C and Rust engines against each other ON THE HOST.

This is the laptop-side counterpart to BENCHMARKING.md's device table -- it
times the same two engines the demo REPLs drive (chat.py -> C gen_prompt,
chat_rust.py -> Rust gen-prompt) without needing an ESP32.

    scripts/bench_host.py [--repeat K] [--short N] [--model NAME]

WHAT THIS IS NOT: a substitute for the device numbers. An x86-64 host has
caches, branch predictors, and memory bandwidth an LX7 does not, and the
device gap is dominated by PSRAM bandwidth in the output head -- which has
no analogue here. CONTRIBUTING.md is explicit that a host-only timing claim
does not belong in the README's numbers table. Use this to catch a host-
visible regression quickly; use real hardware to make a claim.

Three things it borrows from the device methodology, deliberately:

  1. MATCHED SETTINGS. Prompt and token count come from model/models.toml,
     the same registry benchmark_settings_match.rs gates the firmwares
     against. `attn` is the only stage that scales with sequence position,
     so comparing two runs at different token counts silently inflates it
     -- that is the artifact that produced this project's retracted "2.10x
     attention" figure. See BENCHMARKING.md, "Why matched settings matter".

  2. GREEDY ONLY (temp=0). Sampled mode uses a different PRNG in each
     engine, so it is neither comparable nor deterministic.

  3. PARITY BEFORE TIMING. The two engines must emit byte-identical output
     at these settings or this script refuses to print timings. A speed
     number for an engine computing something else is worse than no number.

MEASUREMENT: each engine is timed at two token counts and the per-token
cost is taken from the SLOPE between them, which cancels the fixed startup
cost -- reading a 14.9 MB model.bin dominates a short run and is not
per-token work. Best-of-K wall clock at each point (min, not mean: the
floor is the signal, everything above it is host noise).

Because the slope spans [short, long], the per-token figure is a MEAN over
that range, not the instantaneous cost at token 400 that the device table
reports -- `attn` grows across the range. Both engines are measured
identically, so the ratio is sound; the absolute ms/token is not directly
comparable to BENCHMARKING.md's table.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_SH = os.path.join(REPO, "scripts", "model.sh")
RUST_GEN = os.path.join(REPO, "engine", "target", "release", "gen-prompt")
LAB = os.path.join(REPO, "reference-c", "esp32-llm-lab")


def registry(field, model=None):
    """One field out of model/models.toml, via the C side's own reader."""
    argv = [MODEL_SH, field] + ([model] if model else [])
    try:
        return subprocess.run(argv, capture_output=True, text=True, check=True).stdout.strip()
    except subprocess.CalledProcessError as e:
        sys.exit(f"model.sh {field}: {e.stderr.strip()}")


def build_c(dest):
    """Compile the C reference exactly as demo/run_cli.sh does: cc -O3."""
    cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc")
    if not cc:
        sys.exit("no C compiler found (apt install build-essential)")
    cmd = [cc, "-O3", "-I", LAB, "-o", dest, os.path.join(LAB, "gen_prompt.c"), "-lm"]
    subprocess.run(cmd, check=True)
    return cc


def run(argv):
    """Wall-clock one full generation, returning (seconds, stdout bytes)."""
    t0 = time.perf_counter()
    out = subprocess.run(argv, capture_output=True)
    elapsed = time.perf_counter() - t0
    if out.returncode != 0:
        sys.exit(f"{argv[0]} failed:\n{out.stderr.decode(errors='replace')}")
    return elapsed, out.stdout


def best_of(argv, repeat):
    return min(run(argv)[0] for _ in range(repeat))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=5, help="runs per point, best taken (default 5)")
    ap.add_argument("--short", type=int, default=50, help="short-run token count (default 50)")
    ap.add_argument("--model", default=None, help="registry entry (default: models.toml's default)")
    args = ap.parse_args()

    model_bin, vocab = registry("bin", args.model), registry("vocab", args.model)
    prompt_ids = registry("prompt_ids", args.model)
    n_long = int(registry("n_generate", args.model))
    n_short = args.short
    if n_short >= n_long:
        sys.exit(f"--short ({n_short}) must be below the registry's n_generate ({n_long})")

    if not os.path.exists(RUST_GEN):
        sys.exit(f"missing {RUST_GEN}\nbuild it:  cd engine/llm-host && cargo build --release")

    tmp = tempfile.mkdtemp(prefix="bench_host.")
    c_gen = os.path.join(tmp, "gen_prompt")
    cc = build_c(c_gen)

    # argv differs: C bakes vocab.h in at compile time, Rust loads it at runtime.
    def c_argv(n):
        return [c_gen, model_bin, str(n), prompt_ids, "0"]

    def rust_argv(n):
        return [RUST_GEN, model_bin, vocab, str(n), prompt_ids, "0"]

    print(f"model    {model_bin}")
    print(f"prompt   {len(prompt_ids.split(','))} tokens, greedy (temp=0)")
    print(f"points   {n_short} and {n_long} tokens, best of {args.repeat}")
    print(f"C built  {cc} -O3\n")

    # (1) Parity gate -- refuse to time two engines that disagree.
    _, c_out = run(c_argv(n_long))
    _, rust_out = run(rust_argv(n_long))
    if c_out != rust_out:
        sys.exit("PARITY FAILED: engines emit different text at these settings; "
                 "not reporting timings. Investigate before trusting any number.")
    print(f"parity   OK -- byte-identical output, {len(c_out)} bytes, {n_long} tokens\n")

    # (2) Timing.
    rows = []
    for name, argv in (("C reference", c_argv), ("Rust", rust_argv)):
        short = best_of(argv(n_short), args.repeat)
        long_ = best_of(argv(n_long), args.repeat)
        per_token = (long_ - short) / (n_long - n_short)
        rows.append((name, short, long_, per_token * 1000, 1.0 / per_token, long_ - per_token * n_long))

    w = max(len(r[0]) for r in rows)
    print(f"{'':{w}}  {'total@'+str(n_long):>10}  {'ms/token':>9}  {'tok/s':>8}  {'startup':>8}")
    for name, _short, long_, ms, toks, startup in rows:
        print(f"{name:{w}}  {long_:9.2f}s  {ms:9.2f}  {toks:8.1f}  {startup:7.2f}s")

    c_ms, rust_ms = rows[0][3], rows[1][3]
    ratio = rust_ms / c_ms
    faster, slower = ("Rust", "C") if ratio < 1 else ("C", "Rust")
    print(f"\nratio    {ratio:.2f}x  ({faster} faster than {slower} per token, host x86-64)")
    print("\nHost numbers only -- not the device figures, and not for the README table.")
    print("See BENCHMARKING.md for the measured-on-hardware C-vs-Rust results.")

    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
