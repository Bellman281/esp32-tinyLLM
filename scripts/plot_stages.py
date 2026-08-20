#!/usr/bin/env python3
"""Render the per-stage C-vs-Rust device chart, and the table that goes with it.

    scripts/plot_stages.py                 # write png/benchmark-stages{,-dark}.svg
    scripts/plot_stages.py --table         # print BENCHMARKING.md's table to stdout
    scripts/plot_stages.py --runs c,rust-after

Both outputs come from `benchmarks/device.toml`, which is the only place a
measured figure is written down. Before this script the chart was a hand-made
PNG with no generator in the repo: it could not be regenerated, only redrawn,
so it silently aged every time someone reflashed. The table in BENCHMARKING.md
had the same problem from the other direction. Now a re-measurement is an edit
to one TOML section and a re-run of this.

NO DEPENDENCIES — stdlib only, same rule the rest of scripts/ follows, and the
reason the output is hand-written SVG rather than matplotlib. SVG also diffs as
text, so a number changing shows up in review as a number changing rather than
as a new binary blob.

TWO FILES, because GitHub renders README images against both themes. Embed as:

    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="./png/benchmark-stages-dark.svg">
      <img alt="Per-stage inference cost, C vs Rust" src="./png/benchmark-stages.svg">
    </picture>

COLOR: categorical slots 1–4 (blue/orange/aqua/yellow), stepped per mode,
validated as a set on the adjacent pairlist that grouped bars use — worst
adjacent CVD dE 9.1 light / 8.4 dark, normal-vision 22.9 / 19.8. Two light-mode
slots sit under 3:1 on the surface (aqua 2.74, yellow 2.11), so every bar
carries a visible value label and identity is never left to colour alone. Slot
4 is the last safe one in this theme: a fifth series must fold into "Other" or
facet, NOT take slot 5, because yellow-beside-orange already fails the
all-pairs floors and more slots make it worse. Re-run the validator, do not
re-pick by eye.
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "benchmarks", "device.toml")
OUT = os.path.join(REPO, "png")
STAGES = ["input", "attn", "ffn", "ple", "head"]

# The firmware's `attn detail:` line, when a run carries it. `attn` is the
# stage that scales with sequence position and the one where a five-number
# profile stopped being enough: `qkv` and `proj` are the same fp32 matvec that
# drives ffn and ple, `rope` is nothing, and `core` is attention proper. Those
# have different levers, and the head already demonstrated what guessing which
# one dominates costs. Optional -- the C reference has no equivalent
# instrumentation, and Rust runs before the four-way split do not carry it.
ATTN = [("attn_qkv", "qkv"), ("attn_rope", "rope"), ("attn_core", "core"),
        ("attn_proj", "proj")]

THEME = {
    "light": {
        "surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e",
        "grid": "#e4e3e0", "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
    },
    "dark": {
        "surface": "#1a1a19", "text": "#ffffff", "muted": "#c3c2b7",
        "grid": "#33322f", "series": ["#3987e5", "#d95926", "#199e70", "#c98500"],
    },
}


def load(path):
    """The model/models.toml subset: [section], key = value, # comments."""
    out, sec = {}, None
    for raw in open(path, encoding="utf-8"):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            sec = line[1:-1].strip()
            out[sec] = {}
            continue
        if "=" not in line or sec is None:
            raise SystemExit(f"plot_stages: cannot parse {line!r} in {path}")
        k, v = (p.strip() for p in line.split("=", 1))
        if v[:1] == '"' and v[-1:] == '"':
            out[sec][k] = v[1:-1]
        else:
            out[sec][k] = float(v) if "." in v else int(v)
    return out


def runs_of(doc, wanted):
    got = []
    for sec, body in doc.items():
        if not sec.startswith("run."):
            continue
        rid = sec[4:]
        if wanted and rid not in wanted:
            continue
        got.append((rid, body))
    if wanted:
        got.sort(key=lambda r: wanted.index(r[0]))
    return got


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def bar(x, y, w, h, fill, r=4, hatch=None):
    """Rounded data-end only; the baseline end stays square against the axis."""
    r = min(r, w, h / 2)
    d = (f"M{x},{y} H{x+w-r} A{r},{r} 0 0 1 {x+w},{y+r} V{y+h-r} "
         f"A{r},{r} 0 0 1 {x+w-r},{y+h} H{x} Z") if w > r else \
        f"M{x},{y} H{x+w} V{y+h} H{x} Z"
    out = f'<path d="{d}" fill="{fill}"/>'
    if hatch:
        out += f'<path d="{d}" fill="url(#{hatch})"/>'
    return out


def svg(doc, runs, mode):
    t = THEME[mode]
    if len(runs) > len(t["series"]):
        # Never cycle categorical hues, and never take a fifth slot from this
        # theme: yellow already sits beside orange, and a fifth makes the
        # colourblind floors unreachable. Select runs, or facet.
        sys.exit(f"plot_stages: {len(runs)} runs but only {len(t['series'])} "
                 f"validated colour slots -- pass --runs to choose, or facet")
    s = doc["settings"]
    n = len(runs)
    BH, GAP, PAD = 18, 2, 26          # bar height, 2px surface gap, group padding
    L, R, TOP = 74, 150, 96
    W = 860
    gh = n * BH + (n - 1) * GAP
    H = TOP + len(STAGES) * (gh + PAD) + 78

    peak = max(r[1][st] for _, r in [(0, x) for x in runs] for st in STAGES) if runs else 1
    peak = max(rb[st] for _, rb in runs for st in STAGES)
    step = 20 if peak > 60 else 10
    top = (int(peak / step) + 1) * step
    plot_w = W - L - R

    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" font-family="ui-sans-serif,-apple-system,Segoe UI,Roboto,sans-serif">']
    o.append('<defs><pattern id="prov" width="6" height="6" patternUnits="userSpaceOnUse" '
             'patternTransform="rotate(45)"><rect width="6" height="6" fill="none"/>'
             f'<line x1="0" y1="0" x2="0" y2="6" stroke="{t["surface"]}" stroke-width="2" '
             'opacity="0.55"/></pattern></defs>')
    o.append(f'<rect width="{W}" height="{H}" fill="{t["surface"]}"/>')

    o.append(f'<text x="{L}" y="34" font-size="17" font-weight="600" fill="{t["text"]}">'
             'Per-stage inference cost, C vs Rust</text>')
    o.append(f'<text x="{L}" y="55" font-size="12" fill="{t["muted"]}">'
             f'ESP32-S3 @ 240 MHz · {int(s["n_generate"])} tokens, {int(s["prompt_len"])}-token prompt · '
             'ms per token, lower is better</text>')
    if any(r[1].get("provisional_stages") for r in runs):
        o.append(f'<text x="{W-24}" y="55" font-size="11" fill="{t["muted"]}" '
                 f'text-anchor="end">hatched = provisional</text>')

    lx = L
    for i, (_, rb) in enumerate(runs):
        o.append(f'<rect x="{lx}" y="66" width="10" height="10" rx="2" fill="{t["series"][i]}"/>')
        o.append(f'<text x="{lx+15}" y="75" font-size="12" fill="{t["muted"]}">{esc(rb["label"])}</text>')
        lx += 15 + 7.2 * len(str(rb["label"])) + 26

    for g in range(0, top + 1, step):
        x = L + plot_w * g / top
        o.append(f'<line x1="{x:.1f}" y1="{TOP-10}" x2="{x:.1f}" y2="{H-62}" '
                 f'stroke="{t["grid"]}" stroke-width="1"/>')
        o.append(f'<text x="{x:.1f}" y="{H-44}" font-size="11" fill="{t["muted"]}" '
                 f'text-anchor="middle">{g}</text>')
    o.append(f'<text x="{L+plot_w/2:.1f}" y="{H-24}" font-size="11.5" fill="{t["muted"]}" '
             f'text-anchor="middle">ms / token</text>')

    y = TOP
    for st in STAGES:
        o.append(f'<text x="{L-12}" y="{y+gh/2+4:.1f}" font-size="12.5" font-weight="500" '
                 f'fill="{t["text"]}" text-anchor="end">{st}</text>')
        for i, (_, rb) in enumerate(runs):
            v = rb[st]
            w = plot_w * v / top
            by = y + i * (BH + GAP)
            prov = st in str(rb.get("provisional_stages", "")).split()
            o.append(bar(L, by, max(w, 1.5), BH, t["series"][i],
                         hatch="prov" if prov else None))
            o.append(f'<text x="{L+w+7:.1f}" y="{by+BH-5}" font-size="11.5" '
                     f'fill="{t["muted"]}">{v:.1f}</text>')
        y += gh + PAD

    return "\n".join(o) + "\n</svg>\n"


def table(doc, runs):
    """BENCHMARKING.md's own orientation: runs as rows, stages as columns.

    Matches the table that was already there rather than imposing a new shape,
    so the diff when this first lands is the numbers moving and nothing else.
    `total` is the sum of the profiled stages; `tok/s` comes from wall clock,
    which runs ~2-3 ms/token above that sum on both engines (sampling, argmax
    and serial output sit outside the profiled stages). That distinction was
    already implicit in the hand-written table -- it is stated here.
    """
    ref = runs[0][1]
    tot = lambda r: sum(r[st] for st in STAGES)
    mark = {}
    for _, rb in runs:
        for st in str(rb.get("provisional_stages", "")).split():
            mark[(id(rb), st)] = " †"

    # A run flagged `experimental` is shown but does NOT become "now".
    #
    # Without this, adding a row measured on a board config the repo does not
    # ship -- 120 MHz MSPI, which needs CONFIG_IDF_EXPERIMENTAL_FEATURES and is
    # temperature-sensitive -- would silently rewrite the headline percentage
    # and the "ratio vs C" row to describe a firmware nobody gets by cloning.
    # The measurement is real and belongs in the table; the claim built on top
    # of it is not the project's claim.
    shipping = [rb for _, rb in runs if not rb.get("experimental")]

    # `stages` is the profiled sum; `wall` is the number a user experiences.
    # They diverge by whatever sits outside the profiled stages -- 2.2 ms for
    # the C reference, and 0.03 for Rust since the argmax moved into the head.
    # Reporting only the sum hid a 2.74 ms win once; it does not any more.
    o = ["| ms/token | " + " | ".join(STAGES) + " | stages | **wall** | tok/s |",
         "|---" * (len(STAGES) + 4) + "|"]
    for _, rb in runs:
        cells = [f"{rb[st]:.1f}{mark.get((id(rb), st), '')}" for st in STAGES]
        star = " ‡" if rb.get("experimental") else ""
        o.append(f"| **{rb['label']}**{star} | " + " | ".join(cells) +
                 f" | {tot(rb):.1f} | **{rb['wall_ms']:.1f}** | {rb['tok_s']:.2f} |")

    last = shipping[-1] if shipping else runs[-1][1]
    if len(runs) > 1:
        o.append("| ratio vs " + str(ref["label"]) + " | " +
                 " | ".join(f"{last[st]/ref[st]:.2f}x" for st in STAGES) +
                 f" | {tot(last)/tot(ref):.2f}x | **{last['wall_ms']/ref['wall_ms']:.2f}x** | |")
        o.append("| absolute gap | " +
                 " | ".join(f"{last[st]-ref[st]:+.1f}" for st in STAGES) +
                 f" | {tot(last)-tot(ref):+.1f} | **{last['wall_ms']-ref['wall_ms']:+.1f}** | |")
    if len(shipping) > 2:
        prev = shipping[-2]
        o.append("| **change this brought** | " +
                 " | ".join(f"{last[st]-prev[st]:+.1f}" for st in STAGES) +
                 f" | {tot(last)-tot(prev):+.1f} | **{last['wall_ms']-prev['wall_ms']:+.1f}** | |")

    exp_runs = [rb for _, rb in runs if rb.get("experimental")]
    if exp_runs:
        o.append("")
        for rb in exp_runs:
            o.append(f"‡ **{rb['label']} is not the shipping configuration** and is "
                     f"excluded from the ratios and the summary below. "
                     f"{rb.get('experimental_note', '')}")

    notes = [rb for _, rb in runs if rb.get("provisional_stages")]
    if notes:
        o.append("")
        for rb in notes:
            stg = ", ".join(f"`{x}`" for x in str(rb["provisional_stages"]).split())
            o.append(f"† **{rb['label']}: {stg} is provisional.** {rb.get('note', '')}")

    sub = attn_table(runs)
    if sub:
        o += ["", sub]

    # The headline sentence, generated rather than written. It had been prose
    # under the table quoting a tok/s and a percentage by hand, which is a
    # third copy of a figure that lives in one file -- and it was already one
    # run stale by the time anyone noticed. Same rule as the table: measured
    # numbers are emitted, never typed.
    if len(runs) > 1 and ref["engine"] == "c":
        pct = (ref["wall_ms"] / last["wall_ms"] - 1.0) * 100.0
        verb = "faster" if pct >= 0 else "slower"
        o += ["", f"**Rust is {abs(pct):.1f}% {verb} per token than the C reference "
                  f"it was ported from** — {last['wall_ms']:.1f} ms against "
                  f"{ref['wall_ms']:.1f}, {last['tok_s']:.2f} tok/s against "
                  f"{ref['tok_s']:.2f}, on the same board with byte-identical "
                  f"output. It beats C on "
                  f"{sum(1 for st in STAGES if last[st] < ref[st])} of the "
                  f"{len(STAGES)} stages."]
    return "\n".join(o)


def attn_table(runs):
    """The `attn detail:` breakdown, for the runs that carry it.

    Emitted from the same file as the main table for the same reason: the
    four-way split is what identified `core` as the last single-core stage, and
    a figure that only exists in a commit message is a figure that goes stale.
    Rows are omitted rather than zero-filled where a run predates the
    instrumentation -- a missing measurement and a measured zero are different
    things, and `rope` really is 0.1.
    """
    rows = [rb for _, rb in runs if all(k in rb for k, _ in ATTN)]
    if not rows:
        return ""
    o = ["**`attn` broken down** — the sub-stages the five-stage profile hides. "
         "`qkv`/`proj` are the fp32 matvec that also drives `ffn` and `ple`; "
         "`core` is attention proper, the only part that grows with sequence "
         "position. The C reference has no equivalent instrumentation, so it is "
         "absent rather than zero.",
         "",
         "| attn ms/token | " + " | ".join(n for _, n in ATTN) +
         " | sum | both cores |",
         "|---" * (len(ATTN) + 3) + "|"]
    for rb in rows:
        cells = [f"{rb[k]:.2f}" if k == "attn_rope" else f"{rb[k]:.1f}"
                 for k, _ in ATTN]
        s = sum(rb[k] for k, _ in ATTN)
        dual = str(rb.get("attn_dual", "")).split()
        tag = ", ".join(f"`{d}`" for d in dual) if dual else "—"
        o.append(f"| **{rb['label']}** | " + " | ".join(cells) +
                 f" | {s:.1f} | {tag} |")
    return "\n".join(o)


BEGIN = "<!-- BEGIN device-table (generated: scripts/plot_stages.py --inject) -->"
END = "<!-- END device-table -->"


def inject(paths, body):
    """Replace the marked region in each file. No markers -> hard error.

    The point of the markers is that README.md and BENCHMARKING.md stop being
    two hand-maintained copies of the same figures. Both were stale within a
    day of the last re-measurement.
    """
    for rel in paths:
        f = os.path.join(REPO, rel)
        src = open(f, encoding="utf-8").read()
        if BEGIN not in src or END not in src:
            sys.exit(f"plot_stages: {rel} has no {BEGIN} / {END} markers")
        head, rest = src.split(BEGIN, 1)
        _, tail = rest.split(END, 1)
        open(f, "w", encoding="utf-8").write(
            head + BEGIN + "\n\n" + body + "\n\n" + END + tail)
        print(f"updated {rel}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="", help="comma-separated run ids for the TABLE; default is every run in the file, in file order")
    ap.add_argument("--chart-runs", default="",
                    help="run ids for the CHART, which has only 4 validated colour slots; "
                         "default is the first two and the last -- where it started, the "
                         "reference, and where it is now")
    ap.add_argument("--table", action="store_true", help="print the markdown table to stdout")
    ap.add_argument("--inject", nargs="*", metavar="FILE",
                    default=None, help="rewrite the marked table region in these files "
                                       "(default: README.md BENCHMARKING.md)")
    a = ap.parse_args()

    doc = load(DATA)
    wanted = [r.strip() for r in a.runs.split(",") if r.strip()]
    runs = runs_of(doc, wanted)
    missing = [w for w in wanted if w not in [r[0] for r in runs]]
    if missing:
        sys.exit(f"plot_stages: no [run.{missing[0]}] in {DATA}")

    if a.table:
        print(table(doc, runs))
        return

    if a.inject is not None:
        inject(a.inject or ["README.md", "BENCHMARKING.md"], table(doc, runs))
        return

    # The table can hold every run; the chart cannot, and a chart with one bar
    # per commit stops being readable long before it stops fitting.
    chart = runs
    if a.chart_runs:
        want = [r.strip() for r in a.chart_runs.split(",") if r.strip()]
        chart = runs_of(doc, want)
    elif len(runs) > 3:
        # Same rule as the table's headline: an experimental run is shown in
        # the table but must not become the "now" bar in the README's chart,
        # which is the one figure most people will read and none of them will
        # read the footnote for.
        ship = [r for r in runs if not r[1].get("experimental")]
        chart = [runs[0], runs[1], (ship or runs)[-1]]

    os.makedirs(OUT, exist_ok=True)
    for mode, name in (("light", "benchmark-stages.svg"), ("dark", "benchmark-stages-dark.svg")):
        p = os.path.join(OUT, name)
        with open(p, "w", encoding="utf-8") as f:
            f.write(svg(doc, chart, mode))
        print(f"wrote {os.path.relpath(p, REPO)}")


if __name__ == "__main__":
    main()
