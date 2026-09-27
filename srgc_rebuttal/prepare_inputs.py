"""Prepare all planned seeds on CPU; cache and training remain GPU queue tasks."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

from .build_inputs import MBPP_DATASET, MBPP_REVISION, build, load_rows
from .plan import DEFAULT_PLAN, digest, input_path, load_plan, validate_inputs
from .runtime import atomic_json, lease


def prepare(plan_path, rows_path=None):
    plan = load_plan(plan_path)
    if plan["dataset"] == "math_train":
        from .prepare_rebuttal import prepare as prepare_math
        return prepare_math(plan_path, rows_path)
    if plan["dataset"] != "mbpp" or plan["verifier"] != "srgc_rebuttal.verifiers:code_reward":
        raise ValueError("prepare supports the MATH and MBPP plans with their matching verifiers")
    if plan.get("dataset_revision") != MBPP_REVISION:
        raise ValueError("MBPP plan and loader revisions differ")
    rows = load_rows("mbpp", rows_path)
    for row in rows:
        statements = ast.parse(row["answer"]).body
        if not statements or any(not isinstance(s, ast.Assert) for s in statements):
            raise ValueError("MBPP verifier answers must be Python assert statements")
    source_hash = hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    entries = []
    for seed in plan["seeds"]:
        data = build("mbpp", rows, split_seed=plan["split_seed"], kind="code",
                     ranking_validation=plan["ranking_validation_prompts"], cache=None,
                     provenance={"source": f"huggingface:{MBPP_DATASET}", "dataset_revision": MBPP_REVISION,
                        "source_split": "all published splits, full config", "source_rows_sha256": source_hash,
                        "experiment_seed": seed, "verifier": plan["verifier"],
                        **({"local_rows": str(rows_path), "local_rows_sha256": digest(rows_path)} if rows_path else {})})
        path = input_path(plan_path, plan, seed)
        with lease(path.with_suffix(".prepare.lock")):
            if path.exists():
                prior = json.loads(path.read_text())
                for key in ("records", "candidate_ids", "validation_pool_ids", "ranking_validation_ids", "evaluation_ids"):
                    if prior[key] != data[key]:
                        raise ValueError(f"refusing to replace different existing inputs: {path}")
                data = prior
            else:
                atomic_json(path, data)
        validate_inputs(data, require_cache=False)
        entries.append({"seed": seed, "input": str(path), "input_sha256": digest(path),
                        "cached_candidate_count": len(data["cached_rewards"])})
    return {"dataset": "mbpp", "plan_sha256": digest(plan_path), "dataset_revision": MBPP_REVISION,
            "source_rows": len(rows), "source_rows_sha256": source_hash, "split_seed": plan["split_seed"],
            "jobs": entries, "status": "inputs prepared; no GPU experiments launched"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--rows", type=Path, help="optional local question/answer JSONL; source hash is retained")
    args = parser.parse_args()
    report = prepare(args.plan, args.rows)
    plan = load_plan(args.plan)
    root = input_path(args.plan, plan, plan["seeds"][0]).parent
    atomic_json(root / f"{plan['dataset']}-preparation.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
