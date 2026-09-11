"""Split-half reliability of selection signals along a training run.

Reads the per-rank rows written by ``train_policy_grpo.py --reliability-log``
(one row per prompt per step: two half-group pass rates, two half-group
gradient cosines against the other prompts of the batch) and reports, in
sliding step windows, how repeatable each signal is across prompts.

    python src/reliability_trajectory.py --policy OUT/random/policy [--window 20]

For each window the Pearson correlation between the two halves is the
split-half reliability of a signal measured with K/2 responses; the
Spearman--Brown formula 2r/(1+r) gives the full-group value. Also reported:
the fraction of prompts with mixed rewards (nonzero GRPO advantage), the mean
within-prompt cosine of the two half gradients, and a percentile bootstrap
interval over the rows of the window. CPU only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

SIGNALS = {"pass": ("pass_a", "pass_b"), "grad": ("cos_a_others", "cos_b_others")}


def load_rows(policy: Path) -> list[dict]:
    rows = []
    for path in sorted(policy.glob("reliability_log.rank*.jsonl")):
        with path.open(encoding="utf-8") as stream:
            rows.extend(json.loads(line) for line in stream if line.strip())
    if not rows:
        raise ValueError(f"no reliability_log.rank*.jsonl under {policy}")
    rows.sort(key=lambda r: (r["step"], r["rank"]))
    return rows


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman_brown(r: float, factor: float = 2.0) -> float:
    if not math.isfinite(r) or r <= -1:
        return float("nan")
    return factor * r / (1 + (factor - 1) * r)


def window_summary(rows: list[dict], signal: str, seed: int, reps: int = 1000) -> dict:
    key_a, key_b = SIGNALS[signal]
    pairs = [(r[key_a], r[key_b]) for r in rows if r.get(key_a) is not None and r.get(key_b) is not None]
    if len(pairs) < 3:
        return {"n": len(pairs), "r_half": float("nan"), "r_full": float("nan"),
                "r_half_lo": float("nan"), "r_half_hi": float("nan"),
                "sd_true": float("nan"), "sd_noise": float("nan")}
    a = np.array([p[0] for p in pairs], dtype=float)
    b = np.array([p[1] for p in pairs], dtype=float)
    r = pearson(a, b)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(reps):
        idx = rng.integers(len(a), size=len(a))
        draws.append(pearson(a[idx], b[idx]))
    draws = np.array([d for d in draws if math.isfinite(d)])
    lo, hi = (np.quantile(draws, [0.025, 0.975]) if len(draws) else (float("nan"), float("nan")))
    # Variance decomposition of one half-measurement: true-score variance is
    # the covariance of the halves; noise is the remainder of the mean variance.
    cov = float(np.cov(a, b)[0, 1])
    var = float((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return {"n": len(a), "r_half": r, "r_full": spearman_brown(r),
            "r_half_lo": float(lo), "r_half_hi": float(hi),
            "sd_true": math.sqrt(max(cov, 0.0)), "sd_noise": math.sqrt(max(var - cov, 0.0))}


def trajectory(rows: list[dict], window: int, seed: int = 0, reps: int = 1000) -> list[dict]:
    steps = sorted({r["step"] for r in rows})
    if not steps:
        return []
    out = []
    for start in range(steps[0], steps[-1] + 1, max(1, window // 2)):
        end = start + window - 1
        block = [r for r in rows if start <= r["step"] <= end]
        if not block:
            continue
        record = {"step_start": start, "step_end": min(end, steps[-1]), "rows": len(block),
                  "mixed_fraction": float(np.mean([bool(r.get("mixed")) for r in block])),
                  "mean_pass": float(np.mean([r["pass"] for r in block])),
                  "mean_cos_ab": float(np.mean([r["cos_ab"] for r in block]))}
        for signal in SIGNALS:
            summary = window_summary(block, signal, seed + start, reps)
            record.update({f"{signal}_{k}": v for k, v in summary.items()})
        out.append(record)
        if end >= steps[-1]:
            break
    return out


def write_outputs(records: list[dict], policy: Path) -> Path:
    target = policy / "reliability_trajectory.csv"
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    dat = policy / "reliability_trajectory.dat"  # pgfplots-friendly
    with dat.open("w", encoding="utf-8") as handle:
        handle.write("step pass_r_full grad_r_full pass_r_half_lo pass_r_half_hi grad_r_half_lo grad_r_half_hi mixed_fraction mean_pass\n")
        for r in records:
            mid = (r["step_start"] + r["step_end"]) / 2
            handle.write(f"{mid:.1f} {r['pass_r_full']:.4f} {r['grad_r_full']:.4f} {r['pass_r_half_lo']:.4f} "
                         f"{r['pass_r_half_hi']:.4f} {r['grad_r_half_lo']:.4f} {r['grad_r_half_hi']:.4f} "
                         f"{r['mixed_fraction']:.4f} {r['mean_pass']:.4f}\n")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True, help="trainer output directory with reliability_log.rank*.jsonl")
    parser.add_argument("--window", type=int, default=20, help="steps per window (stride is half the window)")
    parser.add_argument("--reps", type=int, default=1000)
    args = parser.parse_args()
    try:
        rows = load_rows(args.policy.resolve())
        records = trajectory(rows, args.window, reps=args.reps)
        if not records:
            raise ValueError("no complete window")
        target = write_outputs(records, args.policy.resolve())
    except (OSError, ValueError, KeyError) as exc:
        print(f"[abort] {exc}")
        return 2
    print(f"[reliability] {len(rows)} rows, {len(records)} windows -> {target}")
    for r in records:
        print(f"  steps {r['step_start']:>4}-{r['step_end']:<4} n={r['rows']:<4} "
              f"pass r_full={r['pass_r_full']:.2f} [{r['pass_r_half_lo']:.2f},{r['pass_r_half_hi']:.2f}]  "
              f"grad r_full={r['grad_r_full']:.2f} [{r['grad_r_half_lo']:.2f},{r['grad_r_half_hi']:.2f}]  "
              f"mixed={r['mixed_fraction']:.2f} cos_ab={r['mean_cos_ab']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
