#!/usr/bin/env python3
"""Sensitivity of retention and precision to fixed budgets (extension E6).

Three descriptive sweeps over completed runs, none of which recomputes a
registered label:

* selection fraction: top 5%, 10% (registered), 20% from the stored scores;
* pool composition: the ``all``, ``mixed_reward`` and ``identical_reward``
  strata that ``regime_map.analyze_run`` already evaluates;
* re-scored variants: every ``scores_offpolicy.variant-<name>.json`` written by
  ``rescore_variants.py`` (behavior budget ``bk2/bk4/bk8``, clipping
  ``clip3/clip30``) is scored with the same selection metrics as the
  registered file.

    PYTHONPATH=src python3 src/sensitivity_tables.py <run>... --output-dir results/sensitivity
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from regime_map import _behavior_rates, _selection_metrics, _strata, _subset, analyze_run
from score_artifacts import ScoreArtifactError, _score_map, load_complete_score_artifacts

ESTIMATORS = ("g00", "g10", "g01", "g11")
SELECTORS = tuple(f"stale_{e}" for e in ESTIMATORS) + ("passrate_beta",)
METRICS = ("utility_retention", "topk_precision", "utility_gain", "fresh_gain", "floor")


def fraction_rows(run: Path, fracs: list[float]) -> list[dict]:
    rows = []
    for frac in fracs:
        for row in analyze_run(run, frac=frac, first_bootstrap=0):
            if row.get("policy") not in SELECTORS:
                continue
            rows.append({
                "run": run.name, "dataset": row.get("dataset"), "drift": row.get("drift"),
                "seed": row.get("seed"), "variant": f"frac{frac:g}", "stratum": row["stratum"],
                "selector": row["policy"], **{m: row.get(m) for m in METRICS},
            })
    return rows


def variant_rows(run: Path, frac: float = 0.10) -> list[dict]:
    artifacts = load_complete_score_artifacts(run)
    config = json.loads((run / "run_config.json").read_text())
    if any("r" not in halves for halves in artifacts.splithalf.values()):
        raise ScoreArtifactError(f"{run.name}: scores_splithalf.json lacks the matched R split")
    fresh = {i: h["r"] for i, h in artifacts.splithalf.items()}
    fresh_high = {i: h.get("r_high_budget", h["r"]) for i, h in artifacts.splithalf.items()}
    half_a = {i: h["a"] for i, h in artifacts.splithalf.items()}
    half_b = {i: h["b"] for i, h in artifacts.splithalf.items()}
    truth = {i: (half_a[i] + half_b[i]) / 2.0 for i in artifacts.splithalf}
    ids = sorted(truth)
    rates = _behavior_rates(run, set(ids))
    seed = int(config.get("seed", 0)) + 1_000
    rows = []
    for path in sorted(run.glob("scores_offpolicy.variant-*.json")):
        variant = path.name[len("scores_offpolicy.variant-"):-len(".json")]
        raw = json.loads(path.read_text())
        for est in ESTIMATORS:
            if est not in raw:
                continue
            scores = _score_map(raw[est], f"{path.name}[{est}]")
            if set(scores) != set(ids):
                raise ScoreArtifactError(f"{path.name}[{est}] ID coverage differs from the registered scores")
            for stratum, stratum_ids in _strata(rates).items():
                if len(stratum_ids) < 20:
                    continue
                metrics = _selection_metrics(
                    _subset(scores, stratum_ids), _subset(fresh, stratum_ids),
                    _subset(fresh_high, stratum_ids), _subset(truth, stratum_ids),
                    _subset(half_a, stratum_ids), _subset(half_b, stratum_ids),
                    frac=frac, seed=seed,
                )
                rows.append({
                    "run": run.name, "dataset": config.get("dataset"), "drift": int(config.get("drift", 0)),
                    "seed": int(config.get("seed", 0)), "variant": variant, "stratum": stratum,
                    "selector": f"stale_{est}", **{m: metrics.get(m) for m in METRICS},
                })
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["dataset"], row["drift"], row["variant"], row["stratum"], row["selector"]), []).append(row)
    out = []
    for key, items in sorted(grouped.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        summary = dict(zip(("dataset", "drift", "variant", "stratum", "selector"), key))
        summary["seeds"] = len(items)
        for metric in METRICS:
            values = [it[metric] for it in items if isinstance(it[metric], (int, float)) and it[metric] == it[metric]]
            summary[f"{metric}_mean"] = statistics.fmean(values) if values else None
            summary[f"{metric}_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
        out.append(summary)
    return out


def render_markdown(summary: list[dict]) -> str:
    lines = ["| dataset | drift | variant | stratum | selector | retention | precision | seeds |",
             "|---|---|---|---|---|---|---|---|"]
    for row in summary:
        def fmt(metric: str) -> str:
            value = row[f"{metric}_mean"]
            return "-" if value is None else f"{value:.3f} +- {row[f'{metric}_sd']:.3f}"
        lines.append(f"| {row['dataset']} | {row['drift']} | {row['variant']} | {row['stratum']} | "
                     f"{row['selector']} | {fmt('utility_retention')} | {fmt('topk_precision')} | {row['seeds']} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fracs", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    args = parser.parse_args(argv)
    rows: list[dict] = []
    for run in args.runs:
        try:
            rows.extend(fraction_rows(run, args.fracs))
            rows.extend(variant_rows(run))
        except (ScoreArtifactError, OSError, ValueError, KeyError) as exc:
            print(f"[skip] {run}: {exc}")
    if not rows:
        print("no analysable runs")
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "sensitivity_rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = aggregate(rows)
    with (args.output_dir / "sensitivity_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (args.output_dir / "sensitivity_summary.md").write_text(render_markdown(summary))
    print(render_markdown(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
