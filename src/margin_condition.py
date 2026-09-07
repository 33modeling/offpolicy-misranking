#!/usr/bin/env python3
"""Empirical check of the top-k margin condition (extension E3, 2026-09-07).

Proposition 1 of the manuscript: if every score error is smaller than half the
selection margin, the stale ranking recovers the exact top-k set. Per run and
selector this module computes

* ``margin``: gap between the k-th and (k+1)-th fresh reference score
  (``scores_splithalf.json`` field ``r``) after both score vectors are scaled
  to unit maximum absolute value;
* ``error``: maximum absolute difference between the scaled selector score and
  the scaled fresh score over the pool;
* ``ratio = margin / (2 * error)``;
* ``recovered``: whether the selector's top-k set equals the fresh top-k set
  under the fixed tie stream of ``select_rules.jittered_topk``.

A cell with ``ratio > 1`` and ``recovered == False`` contradicts the
proposition and is reported as a violation; the exit status is nonzero in that
case so the numerical freeze cannot proceed silently.

    PYTHONPATH=src python3 src/margin_condition.py <run>... --output-dir results/margin
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from score_artifacts import ScoreArtifactError, load_complete_score_artifacts
from select_rules import jittered_topk, topk_count

ESTIMATORS = ("g00", "g10", "g01", "g11")


def _unit_scale(scores: dict[int, float]) -> dict[int, float]:
    peak = max((abs(v) for v in scores.values()), default=0.0)
    if peak == 0.0:
        return {i: 0.0 for i in scores}
    return {i: v / peak for i, v in scores.items()}


def margin_of(scores: dict[int, float], k: int) -> float:
    ordered = sorted(scores.values())
    if k >= len(ordered):
        return float("inf")
    return ordered[-k] - ordered[-k - 1]


def analyze_run(run: Path, frac: float = 0.10, tie_seed: int = 1_000) -> list[dict]:
    artifacts = load_complete_score_artifacts(run)
    config = json.loads((run / "run_config.json").read_text())
    if any("r" not in halves for halves in artifacts.splithalf.values()):
        raise ScoreArtifactError(f"{run.name}: scores_splithalf.json lacks the matched R split")
    fresh = _unit_scale({idx: halves["r"] for idx, halves in artifacts.splithalf.items()})
    n = len(fresh)
    k = topk_count(n, frac)
    margin = margin_of(fresh, k)
    seed = int(config.get("seed", 0)) + tie_seed
    fresh_top = jittered_topk(fresh, k, seed)
    rows = []
    for est in ESTIMATORS:
        scaled = _unit_scale(artifacts.offpolicy[est])
        error = max(abs(scaled[idx] - fresh[idx]) for idx in fresh)
        ratio = float("inf") if error == 0.0 else margin / (2.0 * error)
        recovered = jittered_topk(scaled, k, seed) == fresh_top
        rows.append({
            "run": run.name, "dataset": config.get("dataset"), "drift": int(config.get("drift", 0)),
            "seed": int(config.get("seed", 0)), "selector": est, "n": n, "k": k,
            "margin": margin, "error": error, "ratio": ratio,
            "condition_holds": ratio > 1.0, "recovered": recovered,
            "violation": ratio > 1.0 and not recovered,
        })
    return rows


def summarize(rows: list[dict]) -> dict:
    holds = [r for r in rows if r["condition_holds"]]
    fails = [r for r in rows if not r["condition_holds"]]
    return {
        "cells": len(rows),
        "condition_holds": len(holds),
        "recovered_when_condition_holds": sum(r["recovered"] for r in holds),
        "condition_fails": len(fails),
        "recovered_when_condition_fails": sum(r["recovered"] for r in fails),
        "violations": [f"{r['run']}:{r['selector']}" for r in rows if r["violation"]],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frac", type=float, default=0.10)
    args = parser.parse_args(argv)
    rows: list[dict] = []
    for run in args.runs:
        try:
            rows.extend(analyze_run(run, frac=args.frac))
        except (ScoreArtifactError, OSError, ValueError, KeyError) as exc:
            print(f"[skip] {run}: {exc}")
    if not rows:
        print("no analysable runs")
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "margin_condition.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    (args.output_dir / "margin_condition_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 2 if summary["violations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
