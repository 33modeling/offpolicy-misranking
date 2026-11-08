"""Analysis (plan Code §13, rebuttal plan v2 §6): training seed is the independent unit.

All effects are reward differences in percentage points (100 x fraction difference).
Missing/censored/unavailable outcomes are counted and shown, never dropped silently.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        12: 2.179, 15: 2.131, 20: 2.086, 30: 2.042}


def t_quantile(df, confidence=0.95):
    try:
        from scipy import stats
        return float(stats.t.ppf(0.5 + confidence / 2, df))
    except Exception:  # noqa: BLE001
        if confidence != 0.95:
            raise
        keys = sorted(T975)
        return T975[max(k for k in keys if k <= df)] if df <= 30 else 1.96


def paired(a: dict, b: dict, seeds, *, confidence=0.95, label=None, interval=True):
    deltas, missing = {}, []
    for s in seeds:
        if a.get(s) is None or b.get(s) is None:
            missing.append(s)
        else:
            deltas[s] = 100.0 * (a[s] - b[s])
    vals = list(deltas.values())
    n = len(vals)
    mean = sum(vals) / n if n else None
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1)) if n > 1 else None
    out = {"label": label, "n_training_seeds": n, "mean_delta_pp": mean, "sample_sd_delta_pp": sd,
           "per_seed_delta_pp": deltas, "missing_seeds": missing,
           "wins": sum(v > 1e-12 for v in vals), "ties": sum(abs(v) <= 1e-12 for v in vals),
           "losses": sum(v < -1e-12 for v in vals), "interval_method": None, "ci_low": None, "ci_high": None,
           "p_two_sided": None}
    if interval and n > 1 and sd is not None:
        q = t_quantile(n - 1, confidence)
        half = q * sd / math.sqrt(n)
        out.update(interval_method=f"paired_seed_t (df={n - 1}, normal-error assumption, n={n})",
                   ci_low=mean - half, ci_high=mean + half, confidence=confidence)
        if sd > 0:
            try:
                from scipy import stats
                out["p_two_sided"] = float(2 * stats.t.sf(abs(mean / (sd / math.sqrt(n))), n - 1))
            except Exception:  # noqa: BLE001
                pass
    return out


def bootstrap_ci(deltas, *, resamples=10000, seed=20261108, confidence=0.95):
    vals = np.asarray(list(deltas), dtype=float)
    if len(vals) < 2:
        return None
    rng = np.random.default_rng(seed)
    means = rng.choice(vals, size=(resamples, len(vals)), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [(1 - confidence) / 2, 0.5 + confidence / 2])
    return {"method": f"seed bootstrap ({resamples} resamples; n={len(vals)} is small)", "low": float(lo), "high": float(hi)}


def binom_two_sided(k, n):
    if n == 0:
        return None
    probs = [math.comb(n, i) * 0.5 ** n for i in range(n + 1)]
    return min(1.0, sum(p for p in probs if p <= probs[k] * (1 + 1e-12)))


def stratified_sign_test(per_cohort_deltas: dict):
    """Pool wins/losses over cohorts (each delta a seed pair within one cohort); ties excluded."""
    wins = sum(sum(v > 1e-12 for v in d.values()) for d in per_cohort_deltas.values())
    losses = sum(sum(v < -1e-12 for v in d.values()) for d in per_cohort_deltas.values())
    ties = sum(sum(abs(v) <= 1e-12 for v in d.values()) for d in per_cohort_deltas.values())
    signs = {c: (sum(d.values()) / len(d) if d else None) for c, d in per_cohort_deltas.items()}
    return {"wins": wins, "losses": losses, "ties_excluded": ties, "n": wins + losses,
            "p_two_sided": binom_two_sided(wins, wins + losses),
            "cohort_mean_positive": sum(1 for v in signs.values() if v is not None and v > 0), "n_cohorts": len(signs)}


def by_policy(rows, key="reward_fraction", *, status_ok=("triggered", "not_triggered", "ran")):
    out = {}
    for r in rows:
        if r.get("status") in status_ok and r.get(key) is not None:
            out.setdefault(r["policy_id"], {})[r["seed"]] = r[key]
    return out


def status_counts(rows):
    out = {}
    for r in rows:
        d = out.setdefault(r["policy_id"], {})
        d[r.get("status")] = d.get(r.get("status"), 0) + 1
    return out


def loso_contrast(rows, seeds, *, taus=None, adaptive="srgc_current", key="reward_fraction"):
    from .tune import loso
    table = by_policy(rows, key)
    fixed = {int(p[len("fixed_sr_t"):]): v for p, v in table.items() if p.startswith("fixed_sr_t")}
    if taus:
        fixed = {t: v for t, v in fixed.items() if t in taus}
    if not fixed:
        return {"status": "no_fixed_branches"}
    rewards = {t: {s: fixed[t].get(s) for s in seeds} for t in fixed}
    res = loso(rewards, seeds)
    tuned = {s: v["tuned_reward"] for s, v in res["per_seed"].items() if v.get("tau_hat") is not None}
    contrast = paired(table.get(adaptive, {}), tuned, seeds, label=f"{adaptive} - LOSO tuned fixed", interval=False)
    contrast.update(interval_method="withheld: crossfit folds share tuning seeds (not IID)",
                    naive_iid_t_not_valid=paired(table.get(adaptive, {}), tuned, seeds, interval=True)[
                        "ci_low"] is not None)
    oracle = {s: v["reward"] for s, v in res["oracle"].items()}
    gb = res["grid_best_same_seeds"]
    return {"loso": res, "primary_contrast": contrast,
            "minus_oracle": paired(table.get(adaptive, {}), oracle, seeds, label="minus per-seed oracle", interval=False),
            "minus_grid_best": paired(table.get(adaptive, {}), fixed.get(gb["tau"], {}) if gb else {}, seeds,
                                      label="minus same-seed grid best (optimistic)", interval=False)}


def secondary(rows, seeds, *, confidence=0.95):
    t = by_policy(rows)
    pairs = [("srgc_current", "on_continued"), ("srgc_current", "fixed_sr_t050"), ("srgc_current", "srgc_two_negative"),
             ("srgc_two_negative", "srgc_one_negative"), ("srgc_current", "plateau_sr_eps0005"),
             ("srgc_current", "plateau_sr_eps0000"), ("srgc_current", "plateau_sr_eps0010"),
             ("srgc_current", "srgc_three_negative"), ("srgc_current", "srgc_current_c50"),
             ("srgc_current", "srgc_uniform_destination"), ("srgc_current", "sampled_time_sr.d0")]
    out = []
    for a, b in pairs:
        if a in t and b in t:
            r = paired(t[a], t[b], seeds, confidence=confidence, label=f"{a} - {b}")
            r["exploratory"] = True
            out.append(r)
    for p in t:
        if p.startswith("fixed_uniform_t"):
            tau = p[-3:]
            out.append(dict(paired(t.get(f"fixed_sr_t{tau}", {}), t[p], seeds, confidence=confidence,
                                   label=f"E3 same-parent SR - Uniform at tau_s ({tau})"), exploratory=True))
    return out


def e3_destination(rows, seeds):
    """SR vs Uniform from the SAME current-rule trigger state (per seed tau_s)."""
    out = []
    for s in seeds:
        sr = [r for r in rows if r["seed"] == s and r["policy_id"] == "srgc_current"]
        un = [r for r in rows if r["seed"] == s and r["policy_id"] == "srgc_uniform_destination"]
        if not sr or sr[0]["status"] != "triggered":
            out.append({"seed": s, "status": sr[0]["status"] if sr else "missing"})
            continue
        u = un[0] if un else {}
        a, b = sr[0].get("reward_fraction"), u.get("reward_fraction")
        ok = u.get("status") == "triggered" and a is not None and b is not None
        out.append({"seed": s, "tau_s": sr[0]["switch_update"], "sr": a, "uniform": b,
                    "delta_pp": 100 * (a - b) if ok else None,
                    "status": "paired" if ok else (u.get("status") or "missing_uniform") + "_uniform"})
    return out


def e4_rules(rows):
    out = []
    for r in rows:
        if r["policy_id"].startswith("srgc_") and r.get("replay_source"):
            out.append({k: r.get(k) for k in ("seed", "policy_id", "status", "switch_update", "observed_through",
                                              "checks_used", "reward_fraction", "continuation_gpu_seconds")})
    return out


def e5_decision_utility(signals_by_seed, curve, parents, horizons, H):
    """signals_by_seed: {seed: {update: D}}; curve: {(seed, policy, update): reward} on the dev split.

    Per parent tau: SR branch (fixed_sr_t{tau}) vs On continuation, short horizons and H.
    The decision implied by D<0 is 'switch'. A seed's parents are NOT independent seeds.
    """
    rows = []
    for s, sig in signals_by_seed.items():
        for tau in parents:
            d = sig.get(tau)
            for h in [*horizons, H - tau]:
                u = tau + h
                on, sr = curve.get((s, "on_continued", u)), curve.get((s, f"fixed_sr_t{tau:03d}", u))
                diff = None if on is None or sr is None else 100 * (sr - on)
                chosen = None if d is None else ("sr" if d < 0 else "on")
                rows.append({"seed": s, "parent": tau, "horizon": h, "eval_update": u, "D": d, "sr_minus_on_pp": diff,
                             "implied_choice": chosen,
                             "sign_agree": None if diff is None or d is None else ((d < 0) == (diff > 0)),
                             "decision_loss_pp": None if diff is None or d is None else
                             (max(diff, 0) - (diff if chosen == "sr" else 0)),
                             "status": "measured" if diff is not None else "unmeasured"})
    agree = [r["sign_agree"] for r in rows if r["sign_agree"] is not None]
    return {"rows": rows, "sign_agreement": (sum(agree) / len(agree)) if agree else None, "n_pairs": len(agree),
            "note": "seed x checkpoint pairs are clustered by training seed; the observed better arm is a noisy reference"}


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys or ["empty"])
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 10) if isinstance(v, float) else v) for k, v in r.items()})


def pareto_plot(rows, path, *, title, cost_key="continuation_gpu_seconds"):
    """Reward vs measured algorithm GPU-hours per seed; mean per policy. Skipped without matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return {"status": "skipped_no_matplotlib"}
    pts = [r for r in rows if r.get(cost_key) is not None and r.get("reward_fraction") is not None
           and r.get("status") in ("triggered", "not_triggered", "ran")]
    if not pts:
        return {"status": "no_points"}
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    policies = sorted({r["policy_id"] for r in pts})
    for p in policies:
        xs = [r[cost_key] / 3600 for r in pts if r["policy_id"] == p]
        ys = [100 * r["reward_fraction"] for r in pts if r["policy_id"] == p]
        ax.scatter(xs, ys, s=12, alpha=0.5)
        ax.scatter([sum(xs) / len(xs)], [sum(ys) / len(ys)], s=50, marker="D", label=p)
    ax.set_xlabel(f"{cost_key.replace('_gpu_seconds', '')} cost (GPU-hours)")
    ax.set_ylabel("reward (%)")
    ax.set_title(title)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"status": "written", "path": str(path), "points": len(pts)}
