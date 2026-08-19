"""Quantization ablation for the TinyStories arms.

Loads each arm's checkpoint, measures fp32 and post-training-quantized
validation loss, and reports whether the PLE gain survives 4-bit PTQ. This is
the CLI behind the PTQ rows in RESULTS.md.

The reusable tensor math lives in src/quantize.py; everything here needs
checkpoints, the validation bins and the arm names, so it is reproduction code.

  uv run python -m research.tinystories.quantize_eval --tag cleandeploy
"""

import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch

from model import Config, TinyLM
from quantize import quantize_model

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "tinystories"
RUNS = str(ROOT / "runs")

def load(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = TinyLM(Config(**ck["cfg"])).to(device)
    m.load_state_dict(ck["state"])
    m.eval()
    return m


@torch.no_grad()
def val_loss(model, vocab, device, iters=60, bs=16, sl=256, seed=1234):
    if not 0 < vocab <= 65536:
        raise SystemExit(f"vocab {vocab} exceeds the uint16 token bins")
    data = np.memmap(DATA / f"vocab-{vocab}" / "val.bin", dtype=np.uint16, mode="r")
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(iters):
        ix = rng.integers(0, len(data) - sl - 1, bs)
        x = torch.from_numpy(np.stack([data[i:i+sl] for i in ix]).astype(np.int64)).to(device)
        y = torch.from_numpy(np.stack([data[i+1:i+1+sl] for i in ix]).astype(np.int64)).to(device)
        losses.append(model(x, y)[1].item())
    return sum(losses) / len(losses)


def format_retained(q_gap, fp_gap):
    """Percentage of the fp32 edge that survives quantization.

    Undefined when there was no fp32 edge to retain, which is a real outcome
    rather than a division to guard against."""
    if fp_gap == 0:
        return "undefined (no fp32 edge)"
    return f"{100 * q_gap / fp_gap:.0f}%"


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--tag", default="cleandeploy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fp16-scales", action="store_true",
                    help="round quantization scales to fp16 like the flash format")
    args = ap.parse_args()
    device = get_device()

    if not 2 <= args.bits <= 8:
        raise SystemExit(f"--bits must be 2..8, got {args.bits}")
    if args.group <= 0:
        raise SystemExit(f"--group must be positive, got {args.group}")

    arms = ["baseline", "ple", "fatembed"]
    paths = {a: os.path.join(RUNS, f"{a}-{args.tag}-s{args.seed}.pt") for a in arms}
    missing = [a for a, p in paths.items() if not os.path.exists(p)]
    if missing:
        # A partial comparison is not a comparison: the PLE-vs-baseline gap
        # below is meaningless if either side is absent.
        raise SystemExit(
            f"missing checkpoints for {', '.join(missing)} "
            f"(tag={args.tag} seed={args.seed}) in {RUNS}")

    # A filename is not an identity. Confirm each checkpoint is the arm its name
    # claims, and that all three were trained on the same data with the same
    # tokenizer, or the PTQ gap below compares unrelated models.
    meta, missing_identity = {}, set()
    for arm, path in paths.items():
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ck["cfg"]
        if cfg.get("arm") != arm:
            raise SystemExit(
                f"{os.path.basename(path)} contains arm={cfg.get('arm')!r}, "
                f"not {arm!r}")
        # Identity is recorded by newer runs; legacy checkpoints carry only the
        # filename, which is not evidence of anything.
        expect_name = f"{arm}{'-' + args.tag if args.tag else ''}-s{args.seed}"
        for field, want in (("seed", args.seed), ("tag", args.tag),
                            ("name", expect_name)):
            if field in ck and ck[field] != want:
                raise SystemExit(
                    f"{os.path.basename(path)} records {field}={ck[field]!r}, "
                    f"but was loaded as {field}={want!r}")
        missing_identity.update(
            f for f in ("seed", "tag", "name", "training", "tokenizer_sha256")
            if f not in ck)
        # The schedule decides the loss as much as the arm does, so two runs on
        # different schedules are not a cohort even when the filenames line up.
        sched = ck.get("training")
        meta[arm] = (cfg["vocab_size"], cfg["seq_len"], ck.get("tokenizer_sha256"),
                     tuple(sorted((k, v) for k, v in sched.items() if k != "seed"))
                     if sched else None)
    distinct = set(meta.values())
    if len(distinct) > 1:
        # Name the field that actually differs: "not comparable" without it
        # leaves the reader diffing checkpoints by hand.
        fields = ["vocab_size", "seq_len", "tokenizer_sha256"]
        for i, field in enumerate(fields):
            vals = {a: m[i] for a, m in meta.items()}
            if len(set(vals.values())) > 1:
                raise SystemExit(
                    f"checkpoints differ on {field}: "
                    + ", ".join(f"{a}={v!r}" for a, v in sorted(vals.items())))
        scheds = {a: dict(m[3]) if m[3] else None for a, m in meta.items()}
        keys = sorted({k for s in scheds.values() if s for k in s})
        for k in keys:
            vals = {a: (s or {}).get(k) for a, s in scheds.items()}
            if len(set(map(repr, vals.values()))) > 1:
                raise SystemExit(
                    f"checkpoints differ on training.{k}: "
                    + ", ".join(f"{a}={v!r}" for a, v in sorted(vals.items())))
        raise SystemExit(
            "checkpoints are not comparable: "
            + "; ".join(f"{a}=schedule {'recorded' if s else 'unrecorded'}"
                        for a, s in sorted(scheds.items())))
    vocab_shared, seq_shared, tok_shared, sched_shared = next(iter(distinct))
    # Two different kinds of gap. Identity can be inferred from the filename and
    # merely not confirmed; tokenizer and schedule cannot be recovered at all.
    from_name = sorted(missing_identity & {"name", "seed", "tag"})
    unrecoverable = sorted(missing_identity & {"tokenizer_sha256", "training"})
    if from_name:
        print(f"WARNING: checkpoints omit {', '.join(from_name)}; taken from the "
              f"filename and not verified")
    if unrecoverable:
        print(f"WARNING: checkpoints omit {', '.join(unrecoverable)}; the "
              f"tokenization and training schedule cannot be verified at all")

    rows = {}
    for arm in arms:
        path = paths[arm]
        cfg_vocab = torch.load(path, map_location="cpu", weights_only=False)["cfg"]["vocab_size"]

        m = load(path, device)
        fp = val_loss(m, cfg_vocab, device, sl=seq_shared)

        m = load(path, device)  # fresh copy
        nq = quantize_model(m, args.bits, args.group, quant_table=True,
                            fp16_scales=args.fp16_scales)
        q = val_loss(m, cfg_vocab, device, sl=seq_shared)

        rows[arm] = (fp, q, nq)
        print(f"{arm:9s} fp32 val {fp:.4f} (ppl {math.exp(fp):.2f}) | "
              f"{args.bits}-bit val {q:.4f} (ppl {math.exp(q):.2f}) | "
              f"deg {q-fp:+.4f} | quantized {nq/1e6:.1f}M params")

    if "baseline" in rows and "ple" in rows:
        print(f"\n--- does PLE's edge survive {args.bits}-bit? ---")
        fp_gap = rows["baseline"][0] - rows["ple"][0]
        q_gap = rows["baseline"][1] - rows["ple"][1]
        print(f"  PLE vs baseline, fp32   : {fp_gap:+.4f} nats")
        print(f"  PLE vs baseline, {args.bits}-bit  : {q_gap:+.4f} nats")
        print(f"  edge retained           : {format_retained(q_gap, fp_gap)}")


if __name__ == "__main__":
    main()
