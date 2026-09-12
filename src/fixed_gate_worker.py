"""GPU-only scoring worker for fixed_gate.py; never updates a policy."""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import torch

import evidence_downstream as ed
from fixed_gate import GPUS, bind, expected_ids
from grads import ProjectionSpec, cosine, grad_params, loo_advantages, prompt_gradient, sequence_logprobs_batch, token_weights


def exact_groups(groups: dict, ids: list[int], k: int) -> dict:
    if set(groups) != set(ids):
        raise ValueError("response prompt coverage differs from assigned prompts")
    result = {}
    for i in ids:
        rows = sorted(groups[i], key=lambda r: int(r["rollout_idx"]))
        if len(rows) != k or [int(r["rollout_idx"]) for r in rows] != list(range(k)):
            raise ValueError("response group has missing or duplicate responses")
        if any(not math.isfinite(float(r["reward"])) or not 0 <= float(r["reward"]) <= 1 for r in rows):
            raise ValueError("invalid response reward")
        result[i] = rows
    return result


def score_group(model, params, rows, beta_logps, spec, direction, clip_cap, micro_batch):
    advantages = loo_advantages(torch.tensor([float(r["reward"]) for r in rows]))
    logps = sequence_logprobs_batch(model, rows, micro_batch=micro_batch)
    weights = [token_weights(lp, lb, float(a), "g11", clip_cap=clip_cap)
               for lp, lb, a in zip(logps, beta_logps, advantages, strict=True)]
    gradient = prompt_gradient(model, params, rows, weights, spec, micro_batch=micro_batch)
    return cosine(gradient, direction)


def compute(out: Path, phase: str, shard: int, *, loader=None, collector=None) -> dict:
    from artifact_contract import validate_generation_contract
    from experiment import read_rollouts, split_validation_directions
    from rollout import collect_rollouts, load_policy
    from rollout_contract import rollout_seed_base

    if shard not in range(GPUS):
        raise ValueError("invalid shard")
    c = ed.read(out / "contract.json")
    config, rule, run = c["config"], c["rule"], Path(c["run"])
    target = out / phase / f"shard-{shard}.json"
    if target.exists():
        raise ValueError("fixed diagnostic/scoring attempt is not repeated; use its frozen result")
    ids = expected_ids(c, phase, shard)
    loader, collector = loader or load_policy, collector or collect_rollouts
    timing = {"model_load_seconds": 0.0, "primary_seconds": 0.0, "replica_generation_seconds": 0.0}
    rows_out = []
    if not ids:
        payload = {"contract_sha256": ed.digest(out / "contract.json"), "phase": phase, "shard": shard,
                   "rows": [], "timing": timing, "training_updates": 0}
        bind(target, payload)
        return payload
    validate_generation_contract(run, ("rollouts_behavior_train",), require_rng_binding=True)
    groups = read_rollouts(run / "rollouts_behavior_train.jsonl")
    primary = exact_groups({i: groups[i] for i in ids}, ids, 8)
    del groups
    micro_batch = int(config.get("gradient_micro_batch", config.get("micro_batch", 1)))
    if micro_batch < 1:
        raise ValueError("gradient micro-batch must be positive")
    start = time.monotonic()
    beta, tokenizer = loader(config["model"], None)
    timing["model_load_seconds"] += time.monotonic() - start
    replica = {}
    if phase == "pilot":
        replica_path = out / phase / f"replica-{shard}.jsonl"
        prompts = ed.read(run / "prompts.json")["train"]
        seed = 1_300_000_003 + rule["pilot_seed"] + config["seed"] * 1_000_003 + config["drift"] * 7919 + shard * 104729
        if seed == rollout_seed_base(config["seed"], 0, "rollouts_behavior_train"):
            raise ValueError("replica generation must use an independent random stream")
        start = time.monotonic()
        collector(beta, tokenizer, [prompts[i] for i in ids], 8, config["max_new_tokens"],
                  float(config["temperature"]), replica_path, sampling_seed_base=seed)
        timing["replica_generation_seconds"] = time.monotonic() - start
        local = exact_groups(read_rollouts(replica_path), list(range(len(ids))), 8)
        replica = {i: local[j] for j, i in enumerate(ids)}
        bind(out / phase / f"replica-{shard}.binding.json", {"contract_sha256": ed.digest(out / "contract.json"),
             "local_to_source_prompt_ids": ids, "sampling_seed_base": seed, "policy": "behavior base model",
             "sha256": ed.digest(replica_path), "responses_per_prompt": 8})
    beta_logps, replica_logps = {}, {}
    for i in ids:
        start = time.monotonic()
        beta_logps[i] = sequence_logprobs_batch(beta, primary[i], micro_batch=micro_batch)
        timing["primary_seconds"] += time.monotonic() - start
        if replica:
            replica_logps[i] = sequence_logprobs_batch(beta, replica[i], micro_batch=micro_batch)
    del tokenizer
    if config["drift"] == 0:
        pi = beta
    else:
        del beta
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        start = time.monotonic()
        pi, tokenizer = loader(config["model"], run / f"policy_step_{config['drift']}")
        timing["model_load_seconds"] += time.monotonic() - start
    params = grad_params(pi, int(config.get("grad_layers", 4)))
    direction, _, _ = split_validation_directions(torch.load(run / "val_groups.pt", weights_only=True))
    spec = ProjectionSpec(dim=int(config.get("proj_dim", 4096)))
    clip_cap = float(config.get("clip_cap", 10.))
    for done, i in enumerate(ids, 1):
        start = time.monotonic()
        score = score_group(pi, params, primary[i], beta_logps[i], spec, direction, clip_cap, micro_batch)
        timing["primary_seconds"] += time.monotonic() - start
        row = {"prompt_idx": i, "primary": score}
        if replica:
            row["replica"] = score_group(pi, params, replica[i], replica_logps[i], spec, direction, clip_cap, micro_batch)
        rows_out.append(row)
        print(f"[fixed-gate] {phase} shard={shard} prompts={done}/{len(ids)}", flush=True)
    payload = {"contract_sha256": ed.digest(out / "contract.json"), "phase": phase, "shard": shard,
               "rows": rows_out, "timing": timing, "training_updates": 0,
               "code_sha256": ed.digest(Path(__file__)), "replica_sha256": ed.digest(replica_path) if replica else None}
    bind(target, payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase", choices=("pilot", "assess", "remaining", "baseline"), required=True)
    parser.add_argument("--shard", type=int, required=True)
    args = parser.parse_args()
    if args.phase == "assess":
        from fixed_gate import assess, projected_remaining_cost, read_phase, verify
        from selection_gate import fingerprint
        c = verify(args.out)
        rows, parts = read_phase(args.out, "pilot")
        result = assess(c, rows, projected_remaining_cost(c, parts))
        result["record_sha256"] = fingerprint(result)
        bind(args.out / "assessment.json", result)
        return
    compute(args.out, args.phase, args.shard)


if __name__ == "__main__":
    main()
