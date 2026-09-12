"""Score gain of noisy top-k selection against split-half reliability, on the
completed matrix points (CPU).

For every completed point the two independent half scores a, b of a signal
are standardized within the point; rho_h = corr(a, b); the top k = frac*n
prompts are selected on one half and their mean score on the other half,
above the pool mean and in units of that half's standard deviation, is the
cross-half gain. Both directions (a->b and b->a) are averaged. Under the
Gaussian score-noise model the expected cross-half gain is rho_h * c_{k,n},
where c_{k,n} is the expected mean of the top k of n standard normals: the
half used for evaluation carries its own noise, so this cross-half version
attenuates by rho, while the latent-score statement of Proposition gain
attenuates by sqrt(rho). Signals:

    fresh       scores_splithalf.json a/b (validation-alignment, current policy)
    difficulty  -|p-1/2| on the two halves of the behavior responses
    g00..g11    scores_stale_splithalf.json when present (src/stale_splithalf.py)

    python src/gain_vs_reliability.py --root MATRIX_ROOT [--frac 0.1] [--out PREFIX]

Writes PREFIX.csv (one row per point and signal), PREFIX.dat (pgfplots) and
prints the table with the pooled ratio of observed to predicted gain.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

from gate_decision import signal_halves

SIGNALS = ("fresh", "difficulty", "g00", "g10", "g01", "g11")


def topk_constant(n: int, k: int, draws: int = 20000, seed: int = 0) -> float:
    """Expected mean of the largest k of n standard normals (Monte Carlo)."""
    if not 1 <= k <= n:
        raise ValueError("need 1 <= k <= n")
    rng = np.random.default_rng(seed)
    total = 0.0
    for start in range(0, draws, 500):
        block = min(500, draws - start)
        z = rng.standard_normal((block, n))
        part = np.partition(z, n - k, axis=1)[:, n - k:]
        total += float(part.mean(axis=1).sum())
    return total / draws


def standardize(values: np.ndarray) -> np.ndarray:
    sd = values.std(ddof=1)
    if not sd > 0:
        raise ValueError("zero variance")
    return (values - values.mean()) / sd


def cross_half_gain(a: np.ndarray, b: np.ndarray, k: int) -> float:
    """Mean of b over the top-k of a (standardized b), averaged with the reverse."""
    a, b = standardize(a), standardize(b)
    gains = []
    for select, evaluate in ((a, b), (b, a)):
        top = np.argsort(-select, kind="stable")[:k]
        gains.append(float(evaluate[top].mean()))
    return float(np.mean(gains))


def point_rows(run: Path, frac: float, constant_cache: dict) -> list[dict]:
    config = json.loads((run / "run_config.json").read_text())
    rows = []
    for signal in SIGNALS:
        try:
            halves = signal_halves(run, signal)
        except FileNotFoundError:
            continue
        ids = sorted(halves)
        a = np.array([halves[i][0] for i in ids], dtype=float)
        b = np.array([halves[i][1] for i in ids], dtype=float)
        n = len(ids)
        k = max(1, int(n * frac))
        if n < 4 or a.std() == 0 or b.std() == 0:
            rows.append({"dataset": config.get("dataset"), "seed": config.get("seed"), "drift": config.get("drift"),
                         "signal": signal, "n": n, "k": k, "rho_half": None, "gain": None, "predicted": None,
                         "ratio": None, "note": "degenerate halves"})
            continue
        rho = float(np.corrcoef(a, b)[0, 1])
        c = constant_cache.setdefault((n, k), topk_constant(n, k))
        gain = cross_half_gain(a, b, k)
        predicted = rho * c
        rows.append({"dataset": config.get("dataset"), "seed": config.get("seed"), "drift": config.get("drift"),
                     "signal": signal, "n": n, "k": k, "rho_half": rho, "gain": gain, "predicted": predicted,
                     "ratio": gain / predicted if predicted else None, "note": ""})
    return rows


def completed_points(root: Path) -> list[Path]:
    return sorted(p for p in root.glob("family-*/*-d*") if (p / "DONE").is_file() and (p / "scores_splithalf.json").is_file())


def collect(root: Path, frac: float) -> list[dict]:
    cache: dict = {}
    rows = []
    for run in completed_points(root):
        try:
            rows.extend(point_rows(run, frac, cache))
        except (OSError, ValueError, KeyError) as exc:
            rows.append({"dataset": None, "seed": None, "drift": None, "signal": run.name, "n": None, "k": None,
                         "rho_half": None, "gain": None, "predicted": None, "ratio": None, "note": f"skipped: {exc}"})
    return rows


def pooled(rows: list[dict]) -> dict[str, dict]:
    out = {}
    for signal in SIGNALS:
        valid = [r for r in rows if r["signal"] == signal and r["gain"] is not None]
        if not valid:
            continue
        gains = np.array([r["gain"] for r in valid])
        preds = np.array([r["predicted"] for r in valid])
        out[signal] = {"points": len(valid), "mean_rho": float(np.mean([r["rho_half"] for r in valid])),
                       "mean_gain": float(gains.mean()), "mean_predicted": float(preds.mean()),
                       "ratio_of_means": float(gains.mean() / preds.mean()) if preds.mean() else None,
                       "slope_through_origin": float((gains * preds).sum() / (preds ** 2).sum()) if (preds ** 2).sum() else None,
                       "rmse": float(np.sqrt(np.mean((gains - preds) ** 2)))}
    return out


def write_outputs(rows: list[dict], prefix: Path) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["signal"])
        writer.writeheader()
        writer.writerows(rows)
    with prefix.with_suffix(".dat").open("w", encoding="utf-8") as handle:
        handle.write("signal dataset seed drift rho gain predicted\n")
        for r in rows:
            if r["gain"] is None:
                continue
            handle.write(f"{r['signal']} {r['dataset']} {r['seed']} {r['drift']} {r['rho_half']:.4f} {r['gain']:.4f} {r['predicted']:.4f}\n")


def render(rows: list[dict], summary: dict) -> str:
    f = lambda v: "-".rjust(7) if v is None else f"{v:+.3f}".rjust(7)  # noqa: E731
    lines = ["point-level cross-half gain (sd units) against rho_h * c_{k,n}",
             "  dataset  seed drift signal      n   k   rho_h    gain    pred   ratio"]
    for r in rows:
        if r["gain"] is None:
            lines.append(f"  {str(r['dataset']):8s} {str(r['seed']):>4s} {str(r['drift']):>5s} {r['signal']:10s} {r['note']}")
            continue
        lines.append(f"  {r['dataset']:8s} {r['seed']:4d} {r['drift']:5d} {r['signal']:10s} {r['n']:3d} {r['k']:3d} "
                     f"{f(r['rho_half'])} {f(r['gain'])} {f(r['predicted'])} {f(r['ratio'])}")
    lines.append("pooled by signal: points, mean rho, mean gain, mean predicted, ratio of means, slope through origin, rmse")
    for signal, s in summary.items():
        lines.append(f"  {signal:10s} {s['points']:3d} {f(s['mean_rho'])} {f(s['mean_gain'])} {f(s['mean_predicted'])} "
                     f"{f(s['ratio_of_means'])} {f(s['slope_through_origin'])} {f(s['rmse'])}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="matrix root with family-*/ point directories")
    parser.add_argument("--frac", type=float, default=0.1)
    parser.add_argument("--out", type=Path, help="output prefix (default: <root>/gain_vs_reliability)")
    args = parser.parse_args(argv)
    try:
        rows = collect(args.root.resolve(), args.frac)
        if not rows:
            raise ValueError(f"no completed points with split-half scores under {args.root}")
        summary = pooled(rows)
        prefix = args.out or (args.root.resolve() / "gain_vs_reliability")
        write_outputs(rows, prefix)
        print(render(rows, summary))
        print(f"[gain-law] written: {prefix.with_suffix('.csv')} and .dat")
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"[abort] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
