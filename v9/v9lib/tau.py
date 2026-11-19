"""P0 freeze-tau: the switch time tau of Staged, selected on the OLMo MATH Branch Bank only.

Rule (fixed before any other cohort's V9 aggregation): among the Staged candidates
tau in {50, 75, ..., 250} (the V8 bank grid without the prefix boundary 25, whose branch IS the SR
arm), take the largest seed-mean reward m*; choose the EARLIEST tau whose mean is within 0.5 pp of m*.

Basis, in this order (the first one that is complete for every candidate tau and every seed):
  1. equal_budget: reward at the budget endpoint under the frozen V8 budget of olmo_math
     (one common cost-only budget for every tau; "Staged cost basis" when available),
  2. matched_update: reward at the matched horizon 275.
Only olmo_math files under $V8_ROOT/olmo_math are read (no other cohort's evaluation file).
The result is write-once: $V9_ROOT/frozen/tau.json, with the rule, the per-tau means, the
per-seed values and the sha256 of every input file.
"""

from __future__ import annotations

from pathlib import Path

from . import env
from .common import Blocked, cfg, eval_records, read_json, tau_path, v8c, v9c

RULE = ("earliest tau whose seed-mean reward is within 0.5 pp of the best seed-mean among Staged candidates "
        "tau in the OLMo MATH Branch Bank grid without 25 (the SR arm); basis = equal budget at the frozen V8 "
        "budget when complete for every tau and seed, else matched update 275")


def candidate_taus(c):
    grid = sorted(set(c["bank"]["core_grid"]) | set(c["bank"]["extended_grid"]))
    return [t for t in grid if t > c["training"]["prefix_updates"]]


def choose(means, tolerance_pp=env.TAU_TOLERANCE_PP):
    """means: {tau: mean reward fraction}. Earliest tau within tolerance of the best mean."""
    best = max(means.values())
    for tau in sorted(means):
        if 100.0 * (best - means[tau]) <= tolerance_pp + 1e-12:
            return tau, best
    raise AssertionError("unreachable")


def aggregation_outputs():
    """V9 aggregation files that would exist if any cohort were aggregated before tau was frozen."""
    root = env.v9_root()
    found = [str(p) for p in (root / "paper").glob("*")] if (root / "paper").is_dir() else []
    found += [str(p) for p in root.glob("*/analysis/*")]
    return sorted(found)


def compute():
    from switching.bank import SeedBank
    from switching.common import sha256_file
    cohort = env.DEV_COHORT
    c = cfg(cohort)
    root = v8c(cohort)
    seeds = c["seeds"]["training"]
    taus = candidate_taus(c)
    recs = eval_records(root)
    evals = {k: v[0] for k, v in recs.items()}
    budget_path = root / "frozen" / "budget.json"
    budget = read_json(budget_path)["value"] if budget_path.is_file() else None
    table = {b: {t: {} for t in taus} for b in ("equal_budget", "matched_update")}
    used_files, missing = set(), {"equal_budget": [], "matched_update": []}
    costs = {t: {} for t in taus}
    for seed in seeds:
        rows = SeedBank(root, c, cohort, seed, "rebuttal_bank", evals=evals).policy_rows(budget=budget)
        by = {r["policy_id"]: r for r in rows}
        for t in taus:
            r = by.get(f"fixed_sr_t{t:03d}")
            if r is None or r.get("status") != "triggered":
                for b in missing:
                    missing[b].append(f"s{seed} tau={t}: V8 branch bank_sr_t{t:03d} not completed")
                continue
            costs[t][seed] = r.get("continuation_gpu_seconds")
            if r.get("reward_fraction") is None:
                missing["matched_update"].append(f"s{seed} tau={t}: waiting for V8 eval-final")
            else:
                table["matched_update"][t][seed] = r["reward_fraction"]
                used_files.add(recs[(r["run_id"], c["training"]["matched_update_endpoint"], "final")][1])
            if budget is None:
                missing["equal_budget"].append("V8 budget not frozen")
            elif r.get("budget_status") in ("selected",) and r.get("budget_reward_fraction") is not None:
                table["equal_budget"][t][seed] = r["budget_reward_fraction"]
                used_files.add(_endpoint_file(recs, root, c, seed, r))
            else:
                missing["equal_budget"].append(f"s{seed} tau={t}: budget endpoint {r.get('budget_status')} "
                                               f"({'unmeasured' if r.get('budget_status') == 'selected' else 'not observed'})")
    basis = next((b for b in ("equal_budget", "matched_update") if not missing[b]), None)
    if basis is None:
        raise Blocked("olmo_math bank incomplete: " + "; ".join(sorted(set(missing["matched_update"]))[:6]))
    means = {t: sum(v.values()) / len(v) for t, v in table[basis].items()}
    tau, best = choose(means)
    other = "matched_update" if basis == "equal_budget" else "equal_budget"
    other_means = ({t: sum(v.values()) / len(v) for t, v in table[other].items()} if not missing[other] else None)
    inputs = {str(p): sha256_file(p) for p in sorted(used_files, key=str)}
    for extra in (budget_path, root / "locator.json"):
        if Path(extra).is_file():
            inputs[str(extra)] = sha256_file(extra)
    return {
        "tau": tau, "rule": RULE, "tolerance_pp": env.TAU_TOLERANCE_PP, "basis": basis,
        "basis_note": ("equal_budget = reward at the budget endpoint selected from cost metadata under the frozen "
                       "V8 olmo_math budget (cost-only, srgc_current mean); matched_update = reward at update 275"),
        "development_cohort": cohort, "seeds": seeds, "candidate_taus": taus,
        "best_mean_reward_fraction": best,
        "mean_reward_fraction_by_tau": {str(t): means[t] for t in taus},
        "per_seed_reward_fraction": {str(t): {str(s): v for s, v in table[basis][t].items()} for t in taus},
        "other_basis": other, "other_basis_means": None if other_means is None else
        {str(t): v for t, v in other_means.items()},
        "other_basis_choice": None if other_means is None else choose(other_means)[0],
        "v8_budget_value_gpu_seconds": budget,
        "equal_budget_definition": ("one common cost-only budget for every tau: the frozen V8 olmo_math budget "
                                    "(written before any evaluation was read); it is the only common budget whose "
                                    "endpoints the V8 queue evaluated, so it is the computable equal-budget basis"),
        "mean_staged_continuation_gpu_seconds_by_tau": {
            str(t): (sum(v.values()) / len(v) if v and all(x is not None for x in v.values()) else None)
            for t, v in costs.items()},
        "basis_missing": {b: sorted(set(m))[:20] for b, m in missing.items()},
        "input_files_sha256": inputs,
        "reads": f"only {root} (olmo_math); no other cohort's evaluation file is opened",
    }


def _endpoint_file(recs, root, c, seed, row):
    """Evaluation file of a budget endpoint: it lies on the trajectory, the branch or its extension."""
    from switching.bank import SeedBank
    b = SeedBank(root, c, c["cohort_id"], seed, "rebuttal_bank", evals={})
    rids = [row["run_id"], b.traj["run_id"] if b.traj else None]
    rids += [m["run_id"] for k, m in b.extensions.items() if b.branches.get(k, {}).get("run_id") == row["run_id"]]
    for rid in rids:
        hit = recs.get((rid, row["budget_endpoint_update"], "final"))
        if hit:
            return hit[1]
    raise Blocked(f"s{seed}: evaluation file of the budget endpoint {row['budget_endpoint_update']} not found")


def freeze():
    """Write-once. Returns (record, 'frozen'|'reused')."""
    from switching.common import utc_now, write_once_json
    p = tau_path()
    if p.is_file():
        return read_json(p), "reused"
    before = aggregation_outputs()
    if before:
        raise Blocked(f"V9 aggregation outputs exist before tau was frozen ({before[:3]}); tau can no longer "
                      "be certified as chosen before other cohorts were aggregated")
    rec = compute()
    rec["frozen_at"] = utc_now()
    rec["v9_outputs_before_freeze"] = before
    write_once_json(p, rec)
    return rec, "frozen"


def status_items():
    from .common import item
    p = tau_path()
    if p.is_file():
        r = read_json(p)
        return [item("freeze-tau", env.DEV_COHORT, "tau", "done", f"tau={r['tau']} basis={r['basis']}")]
    try:
        compute()
        return [item("freeze-tau", env.DEV_COHORT, "tau", "ready", "olmo_math bank complete; run freeze-tau")]
    except Blocked as exc:
        return [item("freeze-tau", env.DEV_COHORT, "tau", "blocked", str(exc))]
    except Exception as exc:  # noqa: BLE001
        return [item("freeze-tau", env.DEV_COHORT, "tau", "blocked", f"{type(exc).__name__}: {exc}")]


_ = v9c
