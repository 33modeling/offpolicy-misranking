"""Planning GPU-hour estimates per cohort and stage from recorded v7 cost bases (not v8 measurements)."""

from __future__ import annotations

from .common import read_json
from .spec import CONFIGS

# Recorded transition boundaries of the original Switch arms (seeds 5-9), used only to size E3.
TAU_S = {"olmo_math": [50, 50, 150, 50, 50],   # v7/evidence/2026-10-01/math-selection-components.md
         "qwen_mbpp": [200, 250, 125, 75, 100]}  # v7/evidence/2026-10-19/qwen-mbpp-seeds5-9.csv


def estimate(cohort, *, seeds=5, prefix=25, H=275, interval=25, core=(50, 100, 150, 200),
             extended=(25, 75, 125, 175, 225, 250), dev_seeds=3, test_seeds=5):
    b = read_json(CONFIGS / "cohorts" / f"{cohort}.json")["cost_basis"]
    need = ("sr_update", "on_training_update", "on_refresh", "srgc_check_extra")
    if any(b.get(k) is None for k in need):
        return {"cohort": cohort, "status": "no_cost_basis", "sources": b.get("sources")}
    refreshes = (H - prefix) // interval
    traj = (H - prefix) * b["on_training_update"] + refreshes * (b["on_refresh"] + b["srgc_check_extra"])
    on_only = (H - prefix) * b["on_training_update"] + refreshes * b["on_refresh"]
    branch = lambda taus: sum(H - t for t in taus) * b["sr_update"]  # noqa: E731
    tau_s = TAU_S.get(cohort, [50] * seeds)[:seeds]
    rows = {
        "bank-trajectory (On+D monitor, regenerate with retained snapshots)": seeds * traj,
        "bank-branches core {50,100,150,200}": seeds * branch(core),
        "bank-extended {25,75,125,175,225,250}": seeds * branch(extended),
        "destination (E3 Uniform at recorded tau_s)": branch(tau_s),
        "budget-extension upper bound (SR runs to hard cap 400)": seeds * (len(core) + 1) * 125 * b["sr_update"],
    }
    per_run = {"on_continued": on_only, "sr_early": (H - prefix) * b["sr_update"],
               "srgc_current": (H - prefix) * b["on_training_update"] + 4 * (b["on_refresh"] + b["srgc_check_extra"]) +
               0.0}
    fixed_grid = [25, 50, 100, 150, 200]
    dev = dev_seeds * (sum((t - prefix) * b["on_training_update"] + (t - prefix) // interval * b["on_refresh"] +
                           (H - t) * b["sr_update"] for t in fixed_grid) + 3 * per_run["srgc_current"] +
                       2 * (H - prefix) * b["sr_update"])
    test = test_seeds * (per_run["on_continued"] + per_run["sr_early"] + (H - prefix) * b["sr_update"] +
                         6 * per_run["srgc_current"] + 2 * (H - prefix) * b["sr_update"])
    rows["prospective development (fixed grid 5, plateau 3 eps, refresh 2 K, srgc tau dist; 3 seeds)"] = dev
    rows["prospective test (11 arms; 5 fresh seeds)"] = test
    n_final = seeds * (1 + len(core) + len(extended) + 1)
    n_dev = seeds * (refreshes + sum((H - t) // interval for t in [*core, *extended]))
    ev = b.get("evaluation_final")
    return {"cohort": cohort, "unit": "GPU-hours (training/selection only; planning estimate)", "stages": rows,
            "rebuttal_core_total": rows["bank-trajectory (On+D monitor, regenerate with retained snapshots)"]
            + rows["bank-branches core {50,100,150,200}"] + rows["destination (E3 Uniform at recorded tau_s)"],
            "final_evaluations": n_final, "dev_evaluations": n_dev,
            "evaluation_gpu_hours": None if ev is None else n_final * ev + n_dev * ev / 6.0,
            "evaluation_basis": "measured per-endpoint evaluation GPU-h (dev split = 50/300 of final)" if ev else
            "unknown for this cohort: run `profile` on the cluster (endpoint receipts record evaluation_gpu_seconds)",
            "assumptions": ["SR-GC srgc_current sized with 4 checks (median recorded boundary <= 100)",
                            "Uniform update cost taken equal to SR update cost"],
            "sources": b["sources"]}
