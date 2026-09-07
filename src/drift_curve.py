#!/usr/bin/env python3
"""Retention against cumulative updates (extension E1, 2026-09-07).

Collects point metrics from registered runs (``--registered``) and from the
separate fine-grid chain produced by ``scripts/run_drift_curve.sh``
(``--curve``). Each run contributes, for the full pool and every stale
selector, the split-half reliability, top-k precision, utility gain, fresh
gain and retention of ``regime_map``'s selection metrics (no bootstrap, no
label). Output: a CSV with one row per run and selector, a per-(source,
dataset, drift) summary, and pgfplots tables per source.

    PYTHONPATH=src python3 src/drift_curve.py --registered <run>... --curve <run>... --output-dir results/drift_curve
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from regime_map import _behavior_rates, _selection_metrics, _strata, _subset
from score_artifacts import ScoreArtifactError, load_complete_score_artifacts

ESTIMATORS = ("g00", "g10", "g01", "g11")
METRICS = ("floor", "topk_precision", "utility_gain", "fresh_gain", "utility_retention")


def point_metrics(run: Path, frac: float = 0.10, source: str = "registered") -> list[dict]:
    artifacts = load_complete_score_artifacts(run)
    config = json.loads((run / "run_config.json").read_text())
    if any("r" not in halves for halves in artifacts.splithalf.values()):
        raise ScoreArtifactError(f"{run.name}: scores_splithalf.json lacks the matched R split")
    fresh = {i: h["r"] for i, h in artifacts.splithalf.items()}
    fresh_high = {i: h.get("r_high_budget", h["r"]) for i, h in artifacts.splithalf.items()}
    half_a = {i: h["a"] for i, h in artifacts.splithalf.items()}
    half_b = {i: h["b"] for i, h in artifacts.splithalf.items()}
    truth = {i: (half_a[i] + half_b[i]) / 2.0 for i in artifacts.splithalf}
    ids = _strata(_behavior_rates(run, set(truth)))["all"]
    seed = int(config.get("seed", 0)) + 1_000
    rows = []
    for est in ESTIMATORS:
        metrics = _selection_metrics(
            _subset(artifacts.offpolicy[est], ids), _subset(fresh, ids), _subset(fresh_high, ids),
            _subset(truth, ids), _subset(half_a, ids), _subset(half_b, ids), frac=frac, seed=seed,
        )
        rows.append({
            "run": run.name, "dataset": config.get("dataset"), "drift": int(config.get("drift", 0)),
            "seed": int(config.get("seed", 0)), "source": source,
            "selector": est, "n": metrics["n"], "k": metrics["k"], **{m: metrics[m] for m in METRICS},
        })
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["source"], row["dataset"], row["drift"], row["selector"]), []).append(row)
    out = []
    for (source, dataset, drift, selector), items in sorted(
            grouped.items(), key=lambda kv: (kv[0][0], str(kv[0][1]), kv[0][2], kv[0][3])):
        summary = {"source": source, "dataset": dataset, "drift": drift, "selector": selector,
                   "seeds": len(items)}
        for metric in METRICS:
            values = [it[metric] for it in items if isinstance(it[metric], (int, float)) and it[metric] == it[metric]]
            summary[f"{metric}_mean"] = statistics.fmean(values) if values else None
            summary[f"{metric}_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
        out.append(summary)
    return out


def write_dat(path: Path, summary: list[dict], source: str, dataset: str, metric: str) -> None:
    rows = [r for r in summary if r["source"] == source and r["dataset"] == dataset]
    drifts = sorted({row["drift"] for row in rows})
    lines = ["drift " + " ".join(f"{e}_mean {e}_sd" for e in ESTIMATORS)]
    for drift in drifts:
        cells = [str(drift)]
        for est in ESTIMATORS:
            match = [r for r in rows if r["drift"] == drift and r["selector"] == est]
            if match and match[0][f"{metric}_mean"] is not None:
                cells.append(f"{match[0][f'{metric}_mean']:.5f} {match[0][f'{metric}_sd']:.5f}")
            else:
                cells.append("nan nan")
        lines.append(" ".join(cells))
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--registered", nargs="*", type=Path, default=[])
    parser.add_argument("--curve", nargs="*", type=Path, default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frac", type=float, default=0.10)
    args = parser.parse_args(argv)
    if not args.registered and not args.curve:
        parser.error("give at least one run via --registered or --curve")
    rows: list[dict] = []
    for source, runs in (("registered", args.registered), ("curve", args.curve)):
        for run in runs:
            try:
                rows.extend(point_metrics(run, frac=args.frac, source=source))
            except (ScoreArtifactError, OSError, ValueError, KeyError) as exc:
                print(f"[skip] {run}: {exc}")
    if not rows:
        print("no analysable runs")
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "drift_curve_rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    with (args.output_dir / "drift_curve_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    for source in sorted({row["source"] for row in summary}):
        for dataset in sorted({row["dataset"] for row in summary if row["source"] == source}):
            for metric in ("utility_retention", "topk_precision", "floor"):
                write_dat(args.output_dir / f"{source}_{dataset}_{metric}_vs_drift.dat", summary, source, dataset, metric)
    for row in summary:
        ret = row["utility_retention_mean"]
        print(f"{row['source']:>10} {row['dataset']:>8} d={row['drift']:<4} {row['selector']:>4} retention="
              f"{'nan' if ret is None else f'{ret:.3f}'} precision={row['topk_precision_mean']:.3f} "
              f"floor={row['floor_mean']:.3f} seeds={row['seeds']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
