#!/usr/bin/env python3
"""Do divergence diagnostics predict utility retention? (extension E6, 2026-09-07)

Hypothesis 5 of the plan: token KL, trajectory ESS, and clipping frequency are
diagnostics, not stale-only safety certificates. For every completed run and
stale selector this module pairs the point retention (full pool, no
bootstrap) with the run's diagnostics and reports Spearman rank correlations
over runs. No registered label is recomputed.

    PYTHONPATH=src python3 src/diagnostics_vs_retention.py <run>... --output-dir results/diagnostics
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from regime_map import analyze_run

SELECTORS = ("stale_g00", "stale_g10", "stale_g01", "stale_g11")
DIAGNOSTICS = ("token_kl_beta_pi", "traj_ess_frac_g11", "clipfrac_g11", "clipfrac_selector")


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        mean_rank = (i + j) / 2.0 + 1.0
        for pos in range(i, j + 1):
            ranks[order[pos]] = mean_rank
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys, strict=True)
             if x is not None and y is not None and math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return None
    rx = _rank([p[0] for p in pairs])
    ry = _rank([p[1] for p in pairs])
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0 or vy == 0:
        return None
    return cov / math.sqrt(vx * vy)


def _selector_clipfrac(run: Path, selector: str) -> float | None:
    path = run / "divergence_stats.json"
    if not path.exists():
        return None
    doc = json.loads(path.read_text())
    value = doc.get(f"clipfrac_{selector.removeprefix('stale_')}")
    return float(value) if value is not None else None


def collect(runs: list[Path], frac: float = 0.10) -> list[dict]:
    rows = []
    for run in runs:
        try:
            analysed = analyze_run(run, frac=frac, first_bootstrap=0)
        except (OSError, ValueError, KeyError) as exc:
            print(f"[skip] {run}: {exc}")
            continue
        for row in analysed:
            if row.get("stratum") != "all" or row.get("policy") not in SELECTORS:
                continue
            rows.append({
                "run": run.name, "dataset": row.get("dataset"), "drift": row.get("drift"),
                "seed": row.get("seed"), "selector": row["policy"],
                "utility_retention": row.get("utility_retention"),
                "topk_precision": row.get("topk_precision"),
                "floor": row.get("floor"),
                "fresh_gain": row.get("fresh_gain"),
                "token_kl_beta_pi": row.get("token_kl_beta_pi"),
                "traj_ess_frac_g11": row.get("traj_ess_frac_g11"),
                "clipfrac_g11": row.get("clipfrac_g11"),
                "clipfrac_selector": _selector_clipfrac(run, row["policy"]),
            })
    return rows


def correlations(rows: list[dict]) -> list[dict]:
    out = []
    for selector in SELECTORS:
        subset = [r for r in rows if r["selector"] == selector]
        for diagnostic in DIAGNOSTICS:
            for target in ("utility_retention", "topk_precision"):
                xs = [r[diagnostic] for r in subset]
                ys = [r[target] for r in subset]
                usable = sum(1 for x, y in zip(xs, ys, strict=True)
                             if x is not None and y is not None and math.isfinite(x) and math.isfinite(y))
                out.append({"selector": selector, "diagnostic": diagnostic, "target": target,
                            "runs": usable, "spearman": spearman(xs, ys)})
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frac", type=float, default=0.10)
    args = parser.parse_args(argv)
    rows = collect(args.runs, frac=args.frac)
    if not rows:
        print("no analysable runs")
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "diagnostics_vs_retention.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    corr = correlations(rows)
    with (args.output_dir / "diagnostics_spearman.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(corr[0]))
        writer.writeheader()
        writer.writerows(corr)
    for row in corr:
        value = "-" if row["spearman"] is None else f"{row['spearman']:+.3f}"
        print(f"{row['selector']:>10} {row['diagnostic']:>18} -> {row['target']:<18} rho={value} (runs={row['runs']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
