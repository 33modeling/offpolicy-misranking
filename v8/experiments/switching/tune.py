"""Tuning on development data only, and LOSO-CV over the rebuttal Branch Bank.

LOSO (rebuttal plan v2 §6.2): tau_hat(-s) = argmax_tau mean_{s' != s} reward(tau, s'),
ties -> earlier tau; tuned_fixed(s) = reward(tau_hat(-s), s). Folds share tuning seeds,
so per-seed contrasts are NOT independent: no IID interval is reported as primary.
"""

from __future__ import annotations

import math

from .common import sha256_json, utc_now


def loso(rewards: dict, seeds, *, tie="earlier_tau"):
    """rewards: {tau: {seed: reward_fraction or None}}. Seeds with any missing tau are reported, not dropped silently."""
    taus = sorted(rewards)
    complete = [s for s in seeds if all(rewards[t].get(s) is not None for t in taus)]
    missing = [s for s in seeds if s not in complete]
    out = {}
    for s in complete:
        others = [x for x in complete if x != s]
        if not others:
            out[s] = {"tau_hat": None, "reason": "no other seeds"}
            continue
        means = {t: sum(rewards[t][x] for x in others) / len(others) for t in taus}
        best = max(means.values())
        ties = [t for t in taus if math.isclose(means[t], best, rel_tol=0, abs_tol=1e-12)]
        tau_hat = min(ties) if tie == "earlier_tau" else max(ties)
        out[s] = {"tau_hat": tau_hat, "tuned_reward": rewards[tau_hat][s], "fold_means": means, "tied": ties}
    hats = [v["tau_hat"] for v in out.values() if v.get("tau_hat") is not None]
    return {"per_seed": out, "complete_seeds": complete, "missing_seeds": missing, "taus": taus,
            "tau_hat_values": sorted(set(hats)), "tau_hat_unstable": len(set(hats)) > 1,
            "dependence_note": "folds share tuning seeds; per-seed deltas are not IID",
            "grid_best_same_seeds": grid_best(rewards, complete), "oracle": oracle(rewards, complete)}


def grid_best(rewards, seeds):
    if not seeds:
        return None
    means = {t: sum(rewards[t][s] for s in seeds) / len(seeds) for t in rewards}
    best = max(means.values())
    tau = min(t for t, v in means.items() if math.isclose(v, best, abs_tol=1e-12))
    return {"tau": tau, "means": means, "label": "same-seed hindsight (optimistic)"}


def oracle(rewards, seeds):
    out = {}
    for s in seeds:
        best = max(rewards[t][s] for t in rewards)
        out[s] = {"tau": min(t for t in rewards if math.isclose(rewards[t][s], best, abs_tol=1e-12)), "reward": best,
                  "label": "per-seed hindsight oracle (reference only)"}
    return out


def select_dev(rows, *, candidates, key, value_key="reward_fraction", cost_key="continuation_gpu_seconds",
               tie_tolerance_pp=None):
    """Development tuning with the frozen tie rule: mean reward; within tolerance -> lower mean cost; then smaller value."""
    if tie_tolerance_pp is None:
        raise ValueError("tie tolerance (pp) is not set: tuning cannot be completed (plan Code §8 E1)")
    bad = [r for r in rows if r.get("role") != "development"]
    if bad:
        raise ValueError(f"tuning received non-development rows: {sorted({r.get('role') for r in bad})}")
    stats = {}
    for cand in candidates:
        rs = [r for r in rows if r[key] == cand]
        vals = [r[value_key] for r in rs]
        if not rs or any(v is None for v in vals):
            raise ValueError(f"candidate {cand!r} has missing development outcomes")
        costs = [r.get(cost_key) for r in rs]
        stats[cand] = {"mean_reward": sum(vals) / len(vals), "n": len(vals),
                       "mean_cost": None if any(c is None for c in costs) else sum(costs) / len(costs)}
    best = max(v["mean_reward"] for v in stats.values())
    near = [c for c, v in stats.items() if 100 * (best - v["mean_reward"]) <= tie_tolerance_pp]
    if len(near) > 1 and all(stats[c]["mean_cost"] is not None for c in near):
        low = min(stats[c]["mean_cost"] for c in near)
        near = [c for c in near if stats[c]["mean_cost"] == low]
    chosen = min(near, key=lambda c: (c is None, c if c is not None else 0))
    return {"selected": chosen, "stats": stats, "tie_tolerance_pp": tie_tolerance_pp,
            "tie_rule": ["mean_reward", "lower_mean_cost_within_tolerance", "smaller_value"]}


def frozen_tuning(*, cohort, dev_seeds, dev_eval_hash, fixed, plateau=None, refresh=None, budget=None,
                  tau_distribution=None, dev_run_ids=(), code_identity=None, config_hash=None, test_runs_seen=()):
    body = {"cohort": cohort, "development_seed_ids": list(dev_seeds), "development_eval_split_hash": dev_eval_hash,
            "fixed": fixed, "fixed_sr_tuned_tau": fixed["selected"] if fixed else None,
            "plateau": plateau, "plateau_epsilon": plateau["selected"] if plateau else None,
            "refresh": refresh, "refresh_interval": refresh["selected"] if refresh else None,
            "primary_budget": budget, "tau_distribution": tau_distribution,
            "tau_distribution_hash": sha256_json(tau_distribution) if tau_distribution else None,
            "development_run_ids": list(dev_run_ids), "code_identity": code_identity, "config_hash": config_hash,
            "test_access_status": "clean" if not test_runs_seen else "contaminated",
            "test_runs_seen": list(test_runs_seen), "frozen_at": utc_now()}
    body["tuning_hash"] = sha256_json({k: v for k, v in body.items() if k != "frozen_at"})
    return body
