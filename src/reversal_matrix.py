#!/usr/bin/env python3
"""Sign-reversal frequency across the completed matrix (extension E2, 2026-09-07).

For every completed run, a selector score ``s_c(x)`` is compared with the fresh
ranking reference ``s_R(x)`` (the matched eight-response R split stored in
``scores_splithalf.json``). Over prompts with both scores nonzero we count

* the overall reversal rate ``P[sign s_c != sign s_R]``;
* the boundary-band reversal rate, restricted to prompts whose fresh rank lies
  in ``k +- round(k/2)``, where a reversal can change the selection;
* the anchor rate, the same quantity between the two independent reference
  halves A and B, which bounds what any selector can achieve.

Rates are aggregated per (dataset, drift) as mean and standard deviation over
seeds. Only the excess over the anchor is selector error.

    PYTHONPATH=src python3 src/reversal_matrix.py <run>... --output-dir results/reversal
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from pathlib import Path

from score_artifacts import ScoreArtifactError, load_complete_score_artifacts
from select_rules import topk_count

ESTIMATORS = ("g00", "g10", "g01", "g11")


def _ranks(scores: dict[int, float], seed: int) -> dict[int, int]:
    rng = random.Random(seed)
    jitter = {i: rng.random() for i in scores}
    order = sorted(scores, key=lambda i: (-scores[i], jitter[i]))
    return {i: rank + 1 for rank, i in enumerate(order)}


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def analyze_run(run: Path, frac: float = 0.10, seed: int = 0) -> dict:
    artifacts = load_complete_score_artifacts(run)
    config = json.loads((run / "run_config.json").read_text())
    if any("r" not in halves for halves in artifacts.splithalf.values()):
        raise ScoreArtifactError(f"{run.name}: scores_splithalf.json lacks the matched R split")
    fresh = {idx: halves["r"] for idx, halves in artifacts.splithalf.items()}
    half_a = {idx: halves["a"] for idx, halves in artifacts.splithalf.items()}
    half_b = {idx: halves["b"] for idx, halves in artifacts.splithalf.items()}
    n = len(fresh)
    k = topk_count(n, frac)
    width = max(1, round(k / 2))
    ranks = _ranks(fresh, seed)
    band = {idx for idx, rank in ranks.items() if k - width + 1 <= rank <= k + width}

    def reversal(scores: dict[int, float], reference: dict[int, float]) -> dict:
        both = [idx for idx in reference if reference[idx] != 0.0 and scores.get(idx, 0.0) != 0.0]
        flipped = [idx for idx in both if scores[idx] * reference[idx] < 0]
        band_both = [idx for idx in both if idx in band]
        band_flipped = [idx for idx in band_both if scores[idx] * reference[idx] < 0]
        return {
            "nonzero": len(both), "flipped": len(flipped), "rate": _rate(len(flipped), len(both)),
            "band_nonzero": len(band_both), "band_flipped": len(band_flipped),
            "band_rate": _rate(len(band_flipped), len(band_both)),
        }

    return {
        "run": run.name,
        "dataset": config.get("dataset"),
        "drift": int(config.get("drift", 0)),
        "seed": int(config.get("seed", 0)),
        "n": n, "k": k, "band_width": width,
        "anchor": reversal(half_a, half_b),
        "selectors": {est: reversal(artifacts.offpolicy[est], fresh) for est in ESTIMATORS},
    }


def aggregate(results: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for res in results:
        grouped.setdefault((res["dataset"], res["drift"]), []).append(res)
    rows = []
    for (dataset, drift), items in sorted(grouped.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        for name in ("anchor",) + ESTIMATORS:
            def pick(item: dict) -> dict:
                return item["anchor"] if name == "anchor" else item["selectors"][name]

            for key in ("rate", "band_rate"):
                values = [pick(it)[key] for it in items if pick(it)[key] is not None]
                rows.append({
                    "dataset": dataset, "drift": drift, "selector": name, "quantity": key,
                    "seeds": len(values),
                    "mean": statistics.fmean(values) if values else None,
                    "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "min": min(values) if values else None, "max": max(values) if values else None,
                })
    return rows


def render_markdown(rows: list[dict]) -> str:
    lines = ["| dataset | drift | selector | overall reversal | boundary-band reversal | seeds |",
             "|---|---|---|---|---|---|"]
    cells: dict[tuple, dict[str, str]] = {}
    for row in rows:
        key = (row["dataset"], row["drift"], row["selector"])
        text = "-" if row["mean"] is None else f"{row['mean']:.3f} +- {row['sd']:.3f}"
        cells.setdefault(key, {"seeds": str(row["seeds"])})[row["quantity"]] = text
    for (dataset, drift, selector), values in cells.items():
        lines.append(f"| {dataset} | {drift} | {selector} | {values.get('rate', '-')} | "
                     f"{values.get('band_rate', '-')} | {values['seeds']} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frac", type=float, default=0.10)
    args = parser.parse_args(argv)
    results = []
    for run in args.runs:
        try:
            results.append(analyze_run(run, frac=args.frac))
        except (ScoreArtifactError, OSError, ValueError, KeyError) as exc:
            print(f"[skip] {run}: {exc}")
    if not results:
        print("no analysable runs")
        return 1
    rows = aggregate(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "reversal_runs.json").write_text(json.dumps(results, indent=1))
    with (args.output_dir / "reversal_matrix.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "reversal_matrix.md").write_text(render_markdown(rows))
    print(render_markdown(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
