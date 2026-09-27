"""Summarize complete extra-seed runs, preserving every paired seed result."""

import argparse
import json
import math
from pathlib import Path
import statistics

from .plan import DEFAULT_PLAN, digest, load_plan
from .runtime import matches
from .cost_report import compare


def summarize(plan_path: Path) -> dict:
    plan = load_plan(plan_path)
    root = (plan_path.parent / plan["output_root"]).resolve()
    rows = []
    for seed in plan["seeds"]:
        folder = root / f"seed-{seed}"
        marker = json.loads((folder / "run.json").read_text())
        if marker["status"] != "complete" or marker["plan_sha256"] != digest(plan_path):
            raise ValueError(f"seed {seed} is incomplete or belongs to another plan")
        prefix = json.loads((folder / "prefix-ready.json").read_text())
        expected = {k: marker[k] for k in ("seed", "plan_sha256", "input_sha256", "implementation_sha256")}
        if (marker["seed"] != seed or not matches(prefix, expected) or
                prefix["completed_updates"] != plan["shared_prefix_updates"]):
            raise ValueError(f"seed {seed}: incomparable shared prefix")
        arms = {}
        for arm in plan["arms"]:
            run = json.loads((folder / f"{arm}-endpoint.json").read_text())
            if (run["seed"] != seed or run["arm"] != arm or
                    run["total_updates"] != plan["total_updates"] or
                    run["shared_prefix_updates"] != plan["shared_prefix_updates"] or
                    run["plan_sha256"] != marker["plan_sha256"] or
                    run["input_sha256"] != marker["input_sha256"] or
                    run["implementation_sha256"] != marker["implementation_sha256"] or
                    run["prefix_checkpoint_sha256"] != prefix["checkpoint_sha256"]):
                raise ValueError(f"seed {seed}/{arm}: incomparable run")
            if len(run["per_question_reward"]) != 300:
                raise ValueError("endpoint must contain all 300 question means")
            if any(not math.isfinite(v) or not 0 <= v <= 1 for v in run["per_question_reward"].values()):
                raise ValueError("invalid per-question reward")
            for key in ("selection_gpu_seconds", "training_gpu_seconds", "sr_preparation_gpu_seconds"):
                if not math.isfinite(run["costs"][key]) or run["costs"][key] < 0:
                    raise ValueError("invalid cost")
            actual = statistics.mean(run["per_question_reward"].values())
            if not 0 <= actual <= 1 or abs(actual - run["reward"]) > 1e-10:
                raise ValueError("endpoint reward is inconsistent")
            arms[arm] = run
        questions = [set(run["per_question_reward"]) for run in arms.values()]
        if any(q != questions[0] for q in questions[1:]):
            raise ValueError("paired arms have different evaluation questions")
        rows.append({"seed": seed, "reward_percent": {a: 100 * r["reward"] for a, r in arms.items()},
            "switch_minus_on_policy_pp": 100 * (arms["switch"]["reward"] - arms["on_policy"]["reward"]),
            "switch_minus_sr_pp": 100 * (arms["switch"]["reward"] - arms["sr"]["reward"]),
            "switched_at": arms["switch"]["switched_at"],
            "selection_and_training_gpu_hours": {a: sum(r["costs"][key] for key in
                ("selection_gpu_seconds", "training_gpu_seconds",
                 "preparation_gpu_seconds" if "preparation_gpu_seconds" in r["costs"] else "sr_preparation_gpu_seconds")) / 3600
                if r.get("cost_measurement_complete", False) else None for a, r in arms.items()},
            "incomplete_cost_arms": [a for a, r in arms.items() if not r.get("cost_measurement_complete", False)]})
    stats = {}
    for contrast in ("switch_minus_on_policy_pp", "switch_minus_sr_pp"):
        values = [r[contrast] for r in rows]
        stats[contrast] = {"mean": statistics.mean(values),
                           "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
                           "positive_seeds": sum(v > 0 for v in values), "seed_count": len(values)}
    return {"cohort": "additional held-out seeds only", "per_seed": rows, "paired_seed_statistics": stats,
            "detailed_cost_comparison": compare(plan_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    args = parser.parse_args()
    print(json.dumps(summarize(args.plan), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
