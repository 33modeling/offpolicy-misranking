"""Validate the extra-seed plan and emit commands without launching training."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import unicodedata


DEFAULT_PLAN = Path(__file__).parent / "experiments/additional_seeds.json"


def load_plan(path: Path) -> dict:
    plan = json.loads(path.read_text())
    expected = {"schema": "srgc-additional-seeds-v1", "shared_prefix_updates": 25,
                "total_updates": 275, "world_size": 4, "responses": 8,
                "scoring_prompts_per_set": 40, "training_prompts": 4,
                "selection_interval": 25, "check_interval": 25, "first_check": 25, "projection_dim": 4096}
    for key, value in expected.items():
        if plan.get(key) != value:
            raise ValueError(f"frozen experiment setting changed: {key}")
    if not re.fullmatch(r"[0-9a-f]{40}", plan.get("model_revision", "")):
        raise ValueError("model_revision must pin the exact model/tokenizer commit")
    if set(plan["arms"]) != {"random", "sr", "on_policy", "switch"} or len(plan["arms"]) != 4:
        raise ValueError("plan needs all four arms exactly once")
    seeds = plan["seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 5 for s in seeds):
        raise ValueError("additional held-out seeds must be distinct integers >= 5")
    return plan


def input_path(plan_path: Path, plan: dict, seed: int) -> Path:
    return (plan_path.parent / plan["input_pattern"].format(seed=seed)).resolve()


def validate_inputs(data: dict, *, require_cache: bool = True) -> None:
    if data.get("schema") != "srgc-inputs-v1":
        raise ValueError("expected srgc-inputs-v1 data")
    records = data["records"]
    groups = [data[k] for k in ("candidate_ids", "validation_pool_ids", "evaluation_ids")]
    for group, size in zip(groups, (400, 100, 300)):
        if len(group) != size or len(set(group)) != size:
            raise ValueError("need distinct 400 candidate, 100 validation and 300 evaluation IDs")
    if any(set(a) & set(b) for i, a in enumerate(groups) for b in groups[i + 1:]):
        raise ValueError("input splits overlap by ID")
    normalized = []
    for group in groups:
        texts = set()
        for i in group:
            r = records[i]
            if not isinstance(r["prompt"], str) or not r["prompt"].strip() or not r.get("answer"):
                raise ValueError("each record needs a formatted prompt and verifier answer")
            if not isinstance(r.get("question"), str) or not r["question"].strip():
                raise ValueError("each record needs raw question text for overlap checks")
            texts.add(" ".join(unicodedata.normalize("NFKC", r["question"]).split()))
        if len(texts) != len(group):
            raise ValueError("duplicate normalized question text within a split")
        normalized.append(texts)
    if any(a & b for i, a in enumerate(normalized) for b in normalized[i + 1:]):
        raise ValueError("input splits overlap by normalized question text")
    val = data["ranking_validation_ids"]
    if not val or len(set(val)) != len(val) or not set(val) <= set(groups[1]):
        raise ValueError("explicit ranking-validation IDs must be a nonempty subset of validation")
    cache = data.get("cached_rewards", {})
    if set(cache) - set(groups[0]):
        raise ValueError("cache includes IDs outside the candidate split")
    if require_cache and set(cache) != set(groups[0]):
        raise ValueError("candidate cache is incomplete; generate the missing rewards first")
    for i, rewards in cache.items():
        if len(rewards) != 8 or any(r not in (0, 1) for r in rewards):
            raise ValueError("each candidate needs eight existing binary cache rewards")
    if not data.get("provenance"):
        raise ValueError("record prompt formatting, split, cache and verifier provenance")
    prior = data["provenance"].get("cache", {}) if isinstance(data["provenance"], dict) else {}
    if cache and isinstance(prior, dict):
        from .verifiers import verifier_protocol
        expected = verifier_protocol(prior.get("verifier"))
        if any(prior.get(key) != value for key, value in expected.items()):
            raise ValueError("cached code rewards use an unverified verifier version; regenerate in a new run")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--check-inputs", action="store_true")
    parser.add_argument("--allow-pending-cache", action="store_true", help="validate data splits before GPU cache generation")
    args = parser.parse_args()
    plan = load_plan(args.plan)
    output = []
    for seed in plan["seeds"]:
        path = input_path(args.plan, plan, seed)
        if args.check_inputs:
            validate_inputs(json.loads(path.read_text()), require_cache=not args.allow_pending_cache)
        cmd = ["torchrun", "--standalone", "--nproc_per_node=4", "-m",
               "srgc_rebuttal.run_experiment", "--plan", str(args.plan), "--seed", str(seed)]
        output.append({"seed": seed, "input": str(path), "input_present": path.exists(),
                       "command": shlex.join(cmd)})
    print(json.dumps({"status": "prepared; not launched", "plan_sha256": digest(args.plan),
                      "jobs": output}, indent=2))


if __name__ == "__main__":
    main()
