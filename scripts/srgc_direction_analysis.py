#!/usr/bin/env python3
"""Direction-versus-magnitude analysis of the SR-GC contrast from recorded histories (no GPU).

Reads ``seed-N/<arm>-progress.json`` histories of a run root and, for every
selection refresh, tabulates the contrast D, its two inner products and - where
the training child recorded them (``srgc_direction_records``) - the norm and
cosine of each mean gradient, plus statistics of the 40 candidate cosine
scores (spread, top-4 gap). Optionally overlays the recorded seed 3/4 D series
from ``selector-pair-srgc-all-d-*.txt``. Writes a CSV, a per-step summary
across seeds, and a three-panel figure when matplotlib is installed.

    python scripts/srgc_direction_analysis.py --plan <group-storage plan>          # or --root <run root>
    python scripts/srgc_direction_analysis.py --root runs/additional-seeds --legacy-d v7/evidence/.../selector-pair-srgc-all-d-1119.txt

The mechanism this tests: D = ||v|| (||g_on|| cos_on - ||g_sr|| cos_sr). If the
early On-policy advantage is direction and the late SR advantage is magnitude,
cos_on - cos_sr should shrink toward zero over training while ||g_sr|| stays
above ||g_on||, and the candidate cosine gaps (ranking_gap4, std) should shrink.
"""

import argparse
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

COLUMNS = ["source", "seed", "arm", "step", "d", "d_source", "on_dot", "sr_dot", "validation_norm", "on_mean_norm", "sr_mean_norm",
           "on_mean_cos", "sr_mean_cos", "on_top4_dot", "on_random4_expected_dot", "ranking_cos_mean",
           "ranking_cos_std", "ranking_cos_top4", "on_cos_top4_minus_mean", "ranking_gap4", "selector"]
ARM_PATTERN = re.compile(r"^(on_policy|switch|switch_repeat|switch_single|switch_consecutive|switch_fixed\d+|direction_(removed|magnitude|replaced))$")


def refresh_rows(root, arms=None):
    """One row per selection refresh from every ``seed-N/<arm>-progress.json`` under ``root``."""
    rows = []
    # Replicates (seed-N/replicate-<k>/<arm>-progress.json) are separate series, labelled replicate<k>-<arm>.
    progress_files = [*Path(root).glob("seed-*/*-progress.json"), *Path(root).glob("seed-*/replicate-*/*-progress.json")]
    for progress in sorted(progress_files):
        arm = progress.name[: -len("-progress.json")]
        if not ARM_PATTERN.match(arm):
            continue
        seed_folder = progress.parent
        if seed_folder.name.startswith("replicate-"):
            arm = f"{seed_folder.name.replace('-', '', 1)}-{arm}"
            seed_folder = seed_folder.parent
        if arms and arm not in arms:
            continue
        seed = int(seed_folder.name.split("-", 1)[1])
        try:
            history = json.loads(progress.read_text()).get("history", [])
        except (OSError, ValueError):
            continue
        for record in history:
            if not record.get("selection_refreshed"):
                continue
            scores = record.get("ranking_scores")
            row = {"source": "recorded", "seed": seed, "arm": arm, "step": record.get("checkpoint"),
                   "d": record.get("d"), "on_dot": record.get("on_mean_validation_dot"),
                   "sr_dot": record.get("sr_mean_validation_dot"), "selector": record.get("selector")}
            for key in COLUMNS:
                if key in record and key not in row:
                    row[key] = record[key]
            row["d_source"] = "decision" if row["d"] is not None else None
            fields = ("validation_norm", "on_mean_norm", "sr_mean_norm", "on_mean_cos", "sr_mean_cos")
            if row["d"] is None and all(isinstance(row.get(k), (int, float)) and math.isfinite(row[k]) for k in fields):
                on_dot = row["validation_norm"] * row["on_mean_norm"] * row["on_mean_cos"]
                sr_dot = row["validation_norm"] * row["sr_mean_norm"] * row["sr_mean_cos"]
                if all(math.isfinite(v) for v in (on_dot, sr_dot, on_dot - sr_dot)):
                    row.update(d=on_dot - sr_dot, on_dot=on_dot, sr_dot=sr_dot,
                               d_source="reconstructed_diagnostic")
            if scores and "ranking_cos_mean" not in row:
                ordered = sorted(scores, reverse=True)
                k = min(4, len(ordered))
                row.update(ranking_cos_mean=statistics.fmean(scores), ranking_cos_std=statistics.pstdev(scores),
                           ranking_cos_top4=statistics.fmean(ordered[:k]),
                           on_cos_top4_minus_mean=statistics.fmean(ordered[:k]) - statistics.fmean(scores),
                           ranking_gap4=(ordered[k - 1] - ordered[k]) if len(ordered) > k else None)
            rows.append({key: row.get(key) for key in COLUMNS})
    return rows


def legacy_d_rows(path):
    """Seed 3/4 D series from a ``selector-pair-srgc-all-d`` export (state,step,d_a,d_b,d,...)."""
    rows = []
    for line in Path(path).read_text().splitlines():
        match = re.match(r"^s(\d+)-t(\d+),(\d+),([^,]*),([^,]*),([^,]*),", line)
        if not match or not match.group(6):
            continue
        rows.append({**{key: None for key in COLUMNS}, "source": Path(path).name, "seed": int(match.group(1)),
                     "arm": "recorded-on-policy-path", "step": int(match.group(3)), "d": float(match.group(6)),
                     "d_source": "legacy_recorded"})
    return rows


def summarize(rows):
    """Per (arm, step): count, mean and sd across seeds of D, cos_on - cos_sr, norm ratio, gap; sign counts."""
    groups = {}
    for row in rows:
        groups.setdefault((row["arm"], row["step"]), []).append(row)
    lines = ["arm                    step  n   D mean(sd)          D<0   cos_on-cos_sr      ||g_sr||/||g_on||   gap4 mean"]
    for (arm, step), items in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
        def stat(values):
            values = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
            if not values:
                return "-"
            sd = statistics.stdev(values) if len(values) > 1 else 0.0
            return f"{statistics.fmean(values):+.3f}({sd:.3f})"
        d_values = [r["d"] for r in items if r["d"] is not None]
        negatives = sum(1 for v in d_values if v < 0)
        cos_diff = [r["on_mean_cos"] - r["sr_mean_cos"] for r in items
                    if r.get("on_mean_cos") is not None and r.get("sr_mean_cos") is not None]
        ratio = [r["sr_mean_norm"] / r["on_mean_norm"] for r in items
                 if r.get("sr_mean_norm") is not None and r.get("on_mean_norm")]
        lines.append(f"{arm:<22} {str(step):>5} {len(items):>2}   {stat(d_values):<19} {negatives:>2}/{len(d_values):<2}  "
                     f"{stat(cos_diff):<18} {stat(ratio):<19} {stat([r.get('ranking_gap4') for r in items])}")
    return "\n".join(lines)


def plot(rows, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    by_arm = {}
    for row in rows:
        by_arm.setdefault(row["arm"], []).append(row)
    for arm, items in sorted(by_arm.items()):
        for seed in sorted({r["seed"] for r in items}):
            series = sorted((r for r in items if r["seed"] == seed and r["step"] is not None), key=lambda r: r["step"])
            label = f"{arm} s{seed}"
            steps = [r["step"] for r in series if r["d"] is not None]
            axes[0].plot(steps, [r["d"] for r in series if r["d"] is not None], marker="o", ms=3, label=label)
            cos = [(r["step"], r["on_mean_cos"] - r["sr_mean_cos"]) for r in series
                   if r.get("on_mean_cos") is not None and r.get("sr_mean_cos") is not None]
            if cos:
                axes[1].plot([s for s, _ in cos], [c for _, c in cos], marker="o", ms=3, label=label)
            gap = [(r["step"], r["ranking_gap4"]) for r in series if r.get("ranking_gap4") is not None]
            if gap:
                axes[2].plot([s for s, _ in gap], [g for _, g in gap], marker="o", ms=3, label=label)
    axes[0].axhline(0, color="k", lw=0.6)
    axes[0].set_title("SR-GC contrast D per check")
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].set_title("direction: cos(on) - cos(sr)")
    axes[2].set_title("ranking gap: 4th - 5th candidate cosine")
    for ax in axes:
        ax.set_xlabel("update")
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", type=Path, help="plan whose run root is analysed (group-storage path)")
    parser.add_argument("--root", type=Path, help="run root containing seed-N/ folders (overrides --plan)")
    parser.add_argument("--arms", nargs="*",
                        help="arms to include (default: on_policy, switch, switch_repeat, switch_fixed*, direction_*, "
                             "and their replicate<k>- series)")
    parser.add_argument("--legacy-d", nargs="*", type=Path, default=[],
                        help="selector-pair-srgc-all-d exports with the recorded seed 3/4 D series")
    parser.add_argument("--out", type=Path, help="output directory (default: <root>/analysis/direction)")
    args = parser.parse_args(argv)
    if args.root is None:
        if args.plan is None:
            parser.error("--plan or --root is required")
        from srgc_rebuttal.plan import load_plan
        from srgc_rebuttal.runtime import run_root
        args.root = run_root(args.plan.resolve(), load_plan(args.plan))
    rows = refresh_rows(args.root, args.arms)
    for path in args.legacy_d:
        rows.extend(legacy_d_rows(path))
    out = args.out or (args.root / "analysis" / "direction")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "direction.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    (out / "summary.txt").write_text(summary + "\n")
    figure = plot(rows, out / "direction.png")
    recorded = sum(1 for r in rows if r["source"] == "recorded")
    decomposed = sum(1 for r in rows if r.get("on_mean_cos") is not None)
    print(f"{len(rows)} refresh rows ({recorded} recorded, {decomposed} with norm/cosine decomposition) -> {out}")
    print(summary)
    if figure:
        print(f"figure: {figure}")
    elif rows:
        print("matplotlib not installed: CSV and summary only")
    return 0


if __name__ == "__main__":
    sys.exit(main())
