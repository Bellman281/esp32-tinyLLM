"""Summarise one ablation cohort and answer the question it was built to ask.

A cohort is the set of runs that are actually comparable: same tag, same
tokenizer and same training configuration, differing only by arm and seed. The
token bins themselves are not hashed, so "same data" here means the same
tokenizer and the same recorded schedule, not a verified corpus.
Model shape is NOT part of that: arms vary width on purpose to hit the same
core budget, and bigcore is defined by being wider.

`runs/` accumulates several cohorts plus records from other models, so
selecting one is required rather than optional - averaging across cohorts
produces confident nonsense.

  python -m research.tinystories.analyze --tag clean \
      --expect-arms baseline,ple,ple_notable,fatembed,bigcore --expect-seeds 2

The table sweep is four separate cohorts, one per ple_dim, and must be read as
four comparisons: averaging them destroys the curve the sweep exists to show.
"""

import argparse
import glob
import json
import math
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS = str(ROOT / "runs")

LABEL = {
    "baseline": "baseline      (no table)",
    "ple": "ple           (proj + table)",
    "ple_notable": "ple_notable   (proj only)",
    "fatembed": "fatembed      (bottom-only table)",
    "bigcore": "bigcore       (2x core, no table)",
}
ORDER = ["baseline", "ple_notable", "fatembed", "ple", "bigcore"]

# Fields that must agree across every run in a cohort for the comparison to mean
# anything. Arm and seed are what the cohort varies. Paths are dotted because
# records nest the model config and the training config.
# Everything recorded that defines the conditions the arms are compared under:
# the data, the tokenizer that produced it, and the full training schedule.
# Seed is excluded because it is what the cohort varies. Shape is excluded
# because arms vary width on purpose to hit the same core budget.
COHORT_KEYS = [
    "config.vocab_size", "config.seq_len",
    "tokenizer_sha256",
    "training.batch_size", "training.steps", "training.lr",
    "training.warmup", "training.eval_every", "training.eval_iters",
    "training.target_core", "training.fixed_ffn",
]



# A recorded null is a value, not an absence: training.fixed_ffn is written as
# null whenever the arm solved its own ffn. Conflating the two would compare a
# recorded null against a missing field and call them equal.
_MISSING = object()


def dotted(r, path):
    """Fetch a dotted key, with fallbacks for records written before the
    training section existed. Returns _MISSING when absent."""
    cur = r
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            cur = _MISSING
            break
        cur = cur[part]
    if cur is not _MISSING:
        return cur
    # Legacy records: steps is top level, and batch size is recoverable from
    # tokens_seen = steps * batch_size * seq_len.
    if path == "training.steps" and "steps" in r:
        return r["steps"]
    if path == "training.batch_size":
        steps, sl = r.get("steps"), (r.get("config") or {}).get("seq_len")
        tok = r.get("tokens_seen")
        if steps and sl and tok:
            return tok // (steps * sl)
    return _MISSING


def record_tag(r):
    """Tag as recorded, or as embedded in the run name."""
    if "tag" in r:
        return r["tag"] or ""
    name = r.get("name", "")
    arm = r.get("arm", "")
    if name.startswith(arm + "-"):
        rest = name[len(arm) + 1:]
        return rest.rsplit("-s", 1)[0] if "-s" in rest else ""
    return ""


def load(runs_dir, tag):
    by_arm = defaultdict(list)
    skipped = 0
    for path in sorted(glob.glob(os.path.join(runs_dir, "*.json"))):
        try:
            with open(path) as f:
                r = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        # runs/ also collects records from other models and from device
        # benchmarks. Anything without an arm is not an ablation run.
        if not isinstance(r, dict) or "arm" not in r or "final_val" not in r:
            continue
        if record_tag(r) != tag:
            skipped += 1
            continue
        by_arm[r["arm"]].append(r)
    return by_arm, skipped


def run_name(r):
    tag = record_tag(r)
    return f"{r.get('arm', '?')}{'-' + tag if tag else ''}-s{r.get('seed', '?')}"


def check_comparable(by_arm):
    """Every run in the cohort must share the conditions under comparison.

    Returns (shared values, fields no record carried). Records written before a
    field existed cannot be checked, and saying so is the honest outcome:
    silently skipping them while claiming the field was verified is how a
    comparison looks safe and is not.
    """
    total = sum(len(rs) for rs in by_arm.values())
    seen, unverifiable = {}, []
    for k in COHORT_KEYS:
        values, present, absent_from = {}, 0, []
        for rs in by_arm.values():
            for r in rs:
                v = dotted(r, k)
                if v is _MISSING:
                    absent_from.append(run_name(r))
                    continue
                present += 1
                key = tuple(v) if isinstance(v, list) else v
                values.setdefault(key, run_name(r))
        if present == 0:
            unverifiable.append(k)
            continue
        if present != total:
            # Some runs carry the field and some do not, so the cohort cannot be
            # checked on it either way. Reporting it as verified from a subset
            # is the failure this whole check exists to prevent.
            raise SystemExit(
                f"{k} is recorded in {present} of {total} runs (absent from "
                f"{', '.join(sorted(absent_from))}), so the cohort cannot be "
                f"compared on it. Re-run the missing arms, or narrow --tag.")
        if len(values) > 1:
            (v1, n1), (v2, n2) = list(values.items())[:2]
            raise SystemExit(
                f"cohort is not comparable: {k}={v1} in {n1} but {k}={v2} in "
                f"{n2}. Narrow the selection with --tag.")
        seen[k] = next(iter(values))
    return seen, unverifiable


def check_complete(by_arm, expect_arms, expect_seeds):
    """A cohort must contain the arms the questions compare, each at the same
    set of distinct seeds. Counting records is not enough: two runs at seed 0
    are one seed, not two."""
    missing = expect_arms - set(by_arm)
    extra = set(by_arm) - expect_arms
    if missing:
        raise SystemExit(
            f"cohort is missing {', '.join(sorted(missing))}; expected "
            f"{', '.join(sorted(expect_arms))}. Use --expect-arms if this "
            f"cohort compares a different set.")
    if extra:
        raise SystemExit(
            f"cohort contains unexpected arms {', '.join(sorted(extra))}; "
            f"expected exactly {', '.join(sorted(expect_arms))}")

    # A record without a seed becomes seed=None, and two of those would look
    # like one seed rather than an unusable record.
    for arm, rs in by_arm.items():
        bad = [run_name(r) for r in rs if not isinstance(r.get("seed"), int)]
        if bad:
            raise SystemExit(
                f"{arm} has records with a missing or non-integer seed: "
                f"{', '.join(sorted(bad))}")

    seeds = {arm: sorted({r["seed"] for r in rs}) for arm, rs in by_arm.items()}
    for arm, rs in by_arm.items():
        if len(rs) != len(seeds[arm]):
            raise SystemExit(f"{arm} has {len(rs)} records but only "
                             f"{len(seeds[arm])} distinct seeds: {seeds[arm]}")

    distinct = {tuple(s) for s in seeds.values()}
    if len(distinct) > 1:
        detail = "; ".join(f"{a}={s}" for a, s in sorted(seeds.items()))
        raise SystemExit(f"arms were not run on the same seeds: {detail}")

    if expect_seeds is not None:
        n = len(next(iter(distinct)))
        if n != expect_seeds:
            raise SystemExit(f"expected {expect_seeds} seeds per arm, found {n}: "
                             f"{sorted(next(iter(distinct)))}")
    return sorted(next(iter(distinct)))


def mean(xs):
    return sum(xs) / len(xs)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default=DEFAULT_RUNS,
                    help="directory of run records to summarise")
    ap.add_argument("--tag", required=True,
                    help="exact run tag, e.g. clean or cleandeploy")
    ap.add_argument("--expect-arms", required=True,
                    help="comma-separated arms this cohort must contain, exactly")
    ap.add_argument("--expect-seeds", type=int, required=True,
                    help="require exactly this many distinct seeds per arm")
    args = ap.parse_args()

    by_arm, skipped = load(args.runs, args.tag)
    if not by_arm:
        raise SystemExit(f"no ablation records with tag {args.tag!r} in {args.runs}")

    shared, unverifiable = check_comparable(by_arm)
    expect_arms = {a.strip() for a in args.expect_arms.split(",") if a.strip()}
    seeds = check_complete(by_arm, expect_arms, args.expect_seeds)

    def short(k, v):
        return f"{k.split('.')[-1]}={v[:12] + '...' if isinstance(v, str) and len(v) > 12 else v}"
    cfg_desc = "  ".join(short(k, v) for k, v in sorted(shared.items()))
    print(f"cohort: {args.tag}   seeds={seeds}   {cfg_desc}")
    if unverifiable:
        print("not verifiable (absent from every record in this cohort): "
              + ", ".join(unverifiable))
    if skipped:
        print(f"({skipped} run records outside this cohort were excluded)")
    print()

    print(f"{'arm':38s} {'core':>10s} {'table':>10s} {'val loss':>10s} {'ppl':>9s} {'n':>3s}")
    print("-" * 84)
    vals = {}
    for arm in ORDER:
        if arm not in by_arm:
            continue
        rs = by_arm[arm]
        v = mean([r["final_val"] for r in rs])
        vals[arm] = v
        p = rs[0]["params"]
        spread = ""
        if len(rs) > 1:
            spread = f"  (+/-{(max(r['final_val'] for r in rs) - min(r['final_val'] for r in rs)) / 2:.4f})"
        print(
            f"{LABEL[arm]:38s} {p['core']:>10,} {p['table']:>10,} "
            f"{v:>10.4f} {math.exp(v):>9.2f} {len(rs):>3d}{spread}"
        )

    print("\n--- the comparisons that decide it " + "-" * 47)

    def delta(a, b, question):
        if a not in vals or b not in vals:
            return
        d = vals[b] - vals[a]
        verdict = "YES" if d > 0.01 else ("no" if d < -0.01 else "tie")
        print(f"\n{question}\n  {a} {vals[a]:.4f} vs {b} {vals[b]:.4f} "
              f"-> delta {d:+.4f} nats  [{verdict}]")

    delta("ple", "baseline",
          "Q1. Does PLE beat a plain tiny LM at the same core budget?")
    delta("ple", "ple_notable",
          "Q2. Is the gain from the flash TABLE, or just the per-layer plumbing?")
    delta("ple", "fatembed",
          "Q3. Does injecting PER-LAYER beat the same params injected at the bottom?")
    delta("bigcore", "ple",
          "Q4. How much of a 2x-core model does PLE recover? (positive = core still wins)")

    if all(k in vals for k in ("ple", "bigcore", "baseline")):
        gap = vals["baseline"] - vals["bigcore"]
        got = vals["baseline"] - vals["ple"]
        if gap > 0:
            print(f"\n  PLE recovers {100 * got / gap:.0f}% of the gap between the "
                  f"baseline core and a 2x core.")


if __name__ == "__main__":
    main()
