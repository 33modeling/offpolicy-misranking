"""CPU-only checks of finite-budget cosine scores; never changes registered labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from first_interval import _bootstrap_scores
from reliability_budget import registered_half_scores


def cosine(a, b):
    a, b = np.asarray(a), np.asarray(b)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return np.divide((a * b).sum(axis=-1), den, out=np.zeros_like(den), where=den > 0)


def exact_score(mu, amplitude, groups, val_noise=0.0, val_count=25):
    """Exact expectation for independent Rademacher candidate/validation noise."""
    mu = np.asarray(mu, dtype=float)
    result = np.zeros(mu.shape[0])
    for j in range(groups + 1):
        g = mu.copy()
        g[:, 1] += amplitude * (2 * j / groups - 1)
        for h in range(val_count + 1):
            v = np.array([1., val_noise * (2 * h / val_count - 1)])
            weight = math.comb(groups, j) / 2**groups * math.comb(val_count, h) / 2**val_count
            result += weight * cosine(g, v)
    return result


def examples():
    budgets = (1, 2, 4, 8, 16, 32, 64)
    curve = [{"groups": m, "a_expected_cosine": float(exact_score(
        [[1., 0.]], 1., m, val_count=1)[0]), "b_cosine": .8} for m in budgets]
    return {
        "cosine_counterexample": {"population_A": 1., "finite_A": 1 / math.sqrt(2),
                                  "population_B": .8, "finite_B": .8,
                                  "population_gain_select_B": -.1,
                                  "finite_gain_select_B": (.8 - 1 / math.sqrt(2)) / 2},
        "budget_curve": curve,
        "tied_pool": {"n": 400, "k": 40, "equally_best": 300,
                      "expected_overlap": 40 / 300, "registered_point_threshold": .2,
                      "gain_every_best_subset": .2},
        "scope": "Exact synthetic examples, not measured OLMo bias or a refutation of the Gaussian model.",
    }


def calibration(trials=20, draws=200, seed=0):
    """Stress-test fixed-set gain only using the production half-score bootstrap.

    Candidate identities and the selected set remain fixed. Outer trials generate
    new observations; inner draws resample each observed A/B half. This does not
    calibrate the top-k gate, Gaussian lookup, or method-choice winner.
    """
    if trials < 2 or draws < 100:
        raise ValueError("calibration needs trials >= 2 and draws >= 100")
    rng = np.random.default_rng(seed)
    theta = np.linspace(0., 1.4, 40)
    mu = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    selected = np.arange(4)
    gain = lambda s: s[..., selected].mean(axis=-1) - s.mean(axis=-1)
    population = float(gain(cosine(mu, [1., 0.])))
    rows = []
    for name, amplitude, val_noise in (("no_noise", np.zeros(40), 0.),
                                      ("heteroskedastic", np.linspace(2., .1, 40), .5)):
        finite = float(gain(exact_score(mu, amplitude, 2, val_noise)))
        for trial in range(trials):
            scores, boots = [], []
            gen = torch.Generator().manual_seed(seed + trial + 10000)
            for _ in range(2):
                candidates = np.repeat(mu[:, None, :], 2, axis=1)
                candidates[:, :, 1] += amplitude[:, None] * rng.choice([-1., 1.], (40, 2))
                validation = np.column_stack([np.ones(25), val_noise * rng.choice([-1., 1.], 25)])
                scores.append(cosine(candidates.mean(axis=1), validation.mean(axis=0)))
                boots.append(_bootstrap_scores(torch.tensor(candidates, dtype=torch.float32),
                                               torch.tensor(validation, dtype=torch.float32), draws, gen).numpy())
            point = float(gain((scores[0] + scores[1]) / 2))
            lo, hi = np.quantile(gain((boots[0] + boots[1]) / 2), [.025, .975])
            rows.append({"scenario": name, "trial": trial, "population_gain": population,
                         "finite_budget_gain": finite, "estimate": point, "lower": float(lo),
                         "upper": float(hi), "covers_population": bool(lo - 1e-7 <= population <= hi + 1e-7),
                         "covers_finite_budget": bool(lo - 1e-7 <= finite <= hi + 1e-7)})
        print(f"[audit] {name}: {trials} independent trials completed", flush=True)
    summaries = []
    for name in sorted({r["scenario"] for r in rows}):
        subset = [r for r in rows if r["scenario"] == name]
        summary = {"scenario": name, "trials": trials, "bootstrap_draws": draws}
        for target in ("population", "finite_budget"):
            p = sum(r[f"covers_{target}"] for r in subset) / trials
            z = 1.959963984540054
            center = (p + z*z/(2*trials)) / (1 + z*z/trials)
            half = z * math.sqrt(p*(1-p)/trials + z*z/(4*trials*trials)) / (1 + z*z/trials)
            summary[target] = {"coverage": p, "wilson_interval": [center-half, center+half]}
        summaries.append(summary)
    return {"scope": "Fixed-set gain only; default is a smoke study, not coverage certification.",
            "scenarios": summaries, "trials": rows}


def artifact_audit(run: Path):
    files = [run / name for name in ("oracle_micro_groups.pt", "val_groups.pt", "scores_splithalf.json")]
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    raw = torch.load(files[0], weights_only=True, map_location="cpu")
    ids = sorted(int(i) for i in raw)
    if not ids or len(set(ids)) != len(raw):
        raise ValueError("invalid/duplicate prompt IDs")
    normalized = {int(i): v for i, v in raw.items()}
    stack = torch.stack([normalized[i] for i in ids]).float()
    val = torch.load(files[1], weights_only=True, map_location="cpu").float()
    if stack.ndim != 3 or val.ndim != 2 or stack.shape[-1] != val.shape[-1]:
        raise ValueError("incompatible gradient dimensions")
    if not torch.isfinite(stack).all() or not torch.isfinite(val).all():
        raise ValueError("non-finite gradients")
    a, b = registered_half_scores(stack, val)
    saved = json.loads(files[2].read_text())
    if set(saved) != {str(i) for i in ids}:
        raise ValueError("stored score IDs do not match gradient IDs")
    for pos, idx in enumerate(ids):
        if not all(math.isfinite(saved[str(idx)][h]) for h in ("a", "b")):
            raise ValueError("non-finite stored scores")
        if max(abs(float(a[pos]) - saved[str(idx)]["a"]), abs(float(b[pos]) - saved[str(idx)]["b"])) > 1e-3:
            raise ValueError("stored A/B scores do not reproduce")
    pooled = cosine(stack[:, stack.shape[1]//2:].mean(dim=1).numpy(), val[val.shape[0]//2:].mean(dim=0).numpy())
    rows = [{"prompt_idx": i, "a": float(a[j]), "b": float(b[j]),
             "mean_half_cosine": float((a[j]+b[j])/2), "pooled_cosine": float(pooled[j])}
            for j, i in enumerate(ids)]
    if before != {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}:
        raise ValueError("input artifacts changed during audit")
    return {"run": str(run.resolve()), "sha256": before, "prompts": rows,
            "scope": "Descriptive same-pool sensitivity; pooling doubles each-half budget. Not population bias or independent budget evidence."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--run", type=Path)
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--draws", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.run and (args.out.resolve() == args.run.resolve() or args.run.resolve() in args.out.resolve().parents):
        p.error("output must be outside the source run")
    torch.set_num_threads(1)
    report = examples()
    if args.calibrate:
        report["calibration"] = calibration(args.trials, args.draws, args.seed)
    if args.run:
        report["artifacts"] = artifact_audit(args.run)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    tables = {"budget_curve": report["budget_curve"]}
    if args.run:
        tables["prompt_scores"] = report["artifacts"]["prompts"]
    if args.calibrate:
        tables["coverage_trials"] = report["calibration"]["trials"]
    for name, rows in tables.items():
        with (args.out / f"{name}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"[audit] CPU only; report: {args.out / 'audit.json'}", flush=True)


if __name__ == "__main__":
    main()
