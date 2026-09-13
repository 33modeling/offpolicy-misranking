"""Synthetic calibration of the reliability law (CPU, local).

For a pool of n prompts with latent score theta and two noisy measurements
(halves) whose split-half correlation is rho, the simulation reports

  latent_ratio   E[latent gain of the noisy top-k] / E[latent gain of the
                 oracle top-k], both expectations over pools (the ratio of
                 expected gains of the proposition): sqrt(rho) under the
                 Gaussian model; not distribution-free
  pool_ratio     mean over pools of the per-pool ratio (a different, smaller
                 statistic; reported for comparison only)
  cross_gain     mean of one half over the top-k of the other, averaged over
                 both directions, in population-sd units: rho * c_{k,n}
                 under the Gaussian model

for three score families:

  gaussian    theta ~ N(0,1), halves = theta + N(0, s^2)
  bernoulli   pass-rate difficulty: p ~ Beta(alpha, alpha) latent, halves are
              -|binomial(K/2, p)/(K/2) - 1/2|; rho is measured, not set
  student3    theta ~ t_3 (heavy tails), Gaussian noise

    python src/gain_law_simulation.py [--n 400] [--frac 0.1] [--reps 300] [--out PREFIX]

Writes PREFIX.dat (family rho latent_ratio pool_ratio sqrt_rho cross_gain predicted)
and prints the table.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

from gain_vs_reliability import topk_constant

RHOS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def draw(family: str, rho: float, n: int, rng: np.random.Generator, group: int = 8):
    if family in ("gaussian", "student3"):
        theta = rng.standard_normal(n) if family == "gaussian" else rng.standard_t(3, n) / math.sqrt(3)
        noise_sd = math.sqrt((1 - rho) / rho)
        a = theta + noise_sd * rng.standard_normal(n)
        b = theta + noise_sd * rng.standard_normal(n)
        return theta, a, b
    if family == "bernoulli":
        # rho is used as a spread parameter: concentrated Beta -> low reliability
        alpha = 0.5 / max(rho, 1e-3)
        p = rng.beta(alpha, alpha, n)
        half = group // 2
        pa = rng.binomial(half, p) / half
        pb = rng.binomial(half, p) / half
        theta = -np.abs(p - 0.5)
        return theta, -np.abs(pa - 0.5), -np.abs(pb - 0.5)
    raise ValueError(family)


def simulate(family: str, rho: float, n: int, k: int, reps: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    noisy_gains, oracle_gains, pool_ratios, cross_gains, rhos = [], [], [], [], []
    c = topk_constant(n, k, seed=seed)
    for _ in range(reps):
        theta, a, b = draw(family, rho, n, rng)
        if a.std() == 0 or b.std() == 0 or theta.std() == 0:
            continue
        rhos.append(float(np.corrcoef(a, b)[0, 1]))
        top_a = np.argsort(-a, kind="stable")[:k]
        top_b = np.argsort(-b, kind="stable")[:k]
        top_oracle = np.argsort(-theta, kind="stable")[:k]
        oracle_gain = theta[top_oracle].mean() - theta.mean()
        noisy_gain = theta[top_a].mean() - theta.mean()
        oracle_gains.append(oracle_gain)
        noisy_gains.append(noisy_gain)
        pool_ratios.append(noisy_gain / oracle_gain if oracle_gain > 0 else float("nan"))
        az = (a - a.mean()) / a.std()
        bz = (b - b.mean()) / b.std()
        cross_gains.append(0.5 * (float(bz[top_a].mean()) + float(az[top_b].mean())))
    measured = float(np.mean(rhos))
    mean_oracle = float(np.mean(oracle_gains))
    return {"family": family, "rho_set": rho, "rho_measured": measured,
            "latent_ratio": float(np.mean(noisy_gains)) / mean_oracle if mean_oracle > 0 else float("nan"),
            "pool_ratio": float(np.nanmean(pool_ratios)), "sqrt_rho": math.sqrt(max(measured, 0.0)),
            "cross_gain": float(np.mean(cross_gains)), "predicted": measured * c, "c": c, "reps": len(rhos)}


def run(n: int, frac: float, reps: int, seed: int, families=("gaussian", "bernoulli", "student3")) -> list[dict]:
    k = max(1, int(n * frac))
    return [simulate(f, rho, n, k, reps, seed + i) for i, f in enumerate(families) for rho in RHOS]


def write_dat(rows: list[dict], prefix: Path) -> Path:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    target = prefix.with_suffix(".dat")
    with target.open("w", encoding="utf-8") as handle:
        handle.write("family rho latent_ratio pool_ratio sqrt_rho cross_gain predicted\n")
        for r in rows:
            handle.write(f"{r['family']} {r['rho_measured']:.4f} {r['latent_ratio']:.4f} {r['pool_ratio']:.4f} {r['sqrt_rho']:.4f} "
                         f"{r['cross_gain']:.4f} {r['predicted']:.4f}\n")
    return target


def render(rows: list[dict]) -> str:
    lines = ["  family     rho_set rho_meas latent_ratio pool_ratio sqrt_rho  cross_gain  rho*c   (latent_ratio = ratio of expected gains)"]
    for r in rows:
        lines.append(f"  {r['family']:10s} {r['rho_set']:7.2f} {r['rho_measured']:8.3f} {r['latent_ratio']:12.3f} {r['pool_ratio']:10.3f} "
                     f"{r['sqrt_rho']:8.3f} {r['cross_gain']:11.3f} {r['predicted']:7.3f}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--frac", type=float, default=0.1)
    parser.add_argument("--reps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("gain_law_simulation"))
    args = parser.parse_args(argv)
    rows = run(args.n, args.frac, args.reps, args.seed)
    print(render(rows))
    target = write_dat(rows, args.out)
    print(f"[simulation] written: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
