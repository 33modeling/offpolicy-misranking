#!/usr/bin/env python3
"""Evaluate exhausted MBPP branches without training or certifying budget compliance.

The queue calls recover() while holding the original branch task lease. All new
artifacts and reporting charges live in budget-recovery/, outside canonical
results, convergence labels and the original cost ledger.
"""

import argparse
import statistics
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import selection_switch_gpu as switch

base, core = switch.base, switch.core
HERE = Path(__file__).resolve()
SCHEMA = "mbpp-budget-recovery/v1"
FILES = ("adapter_config.json", "adapter_model.safetensors", "optimizer.pt", "grpo_stats.jsonl")


def required(p, directory):
    if p.get("dataset") != "mbpp" or (directory / "result.json").exists():
        return False
    if not (directory / "decision.json").is_file():
        return False
    cap = core.number(core.read(directory / "decision.json")["budget_gpu_seconds"], "budget", 0)
    used = base.spent(directory)
    # Exactly-at-cap, normally finalized policies still use canonical publication.
    finalized = any((directory / "policy" / name).is_file()
                    for name in ("policy_train.json", "budget_stop.json"))
    return used > cap or (used == cap and not finalized)


def checkpoint_contract(out, c, arm):
    import evidence_downstream as ed
    from train_policy_grpo import CHECKPOINT_SCHEMA, GrpoConfig

    cfg = c["config"]
    parent = (Path(c["source_run"]) / f"policy_step_{cfg['drift']}").resolve()
    config = asdict(GrpoConfig(**dict(
        {field.removeprefix("grpo_"): cfg[field] for field in ed.TRAIN_FLAGS.values()
         if field != "grpo_logprob_micro_batch"}, checkpoint_every=5)))
    return {"schema": CHECKPOINT_SCHEMA, "training_objective": "grpo",
            "base_model": str(Path(cfg["model"]).resolve()), "seed": cfg["seed"],
            "world_size": 4, "start_step": cfg["drift"],
            "target_steps": cfg["drift"] + c["max_steps"],
            "prompts_sha256": base.digest(out / "subsets" / f"subset-{arm}.json"),
            "max_new_tokens": cfg["max_new_tokens"], "prompt_format": cfg["prompt_format"],
            "resume_adapter": str(parent), "resume_optimizer": str(parent / "optimizer.pt"),
            "config": config}


def validate_candidate(path, out, c, arm, expected):
    import train_policy_grpo as trainer

    args = dict(world_size=4, training_objective="grpo", expected_start_step=expected["start_step"],
                expected_parent=Path(expected["resume_adapter"]), expected_model=Path(expected["base_model"]),
                expected_seed=expected["seed"], expected_max_new_tokens=expected["max_new_tokens"],
                expected_prompt_format=expected["prompt_format"], expected_config=expected["config"],
                expected_prompts=out / "subsets" / f"subset-{arm}.json", require_complete_hashes=True)
    if (path / "policy_train.json").exists():
        manifest = core.read(path / "policy_train.json")
        completed = manifest["completed_steps"]
        trainer.validate_policy_lineage(path, target_steps=completed, **args)
        evidence = "policy_train.json"
    else:
        completed = trainer._checkpoint_step(path, expected)
        if completed is None:
            raise ValueError(f"checkpoint contract or hashes invalid: {path}")
        state = core.read(path / "checkpoint_state.json")
        if type(state["completed_steps"]) is not int:
            raise ValueError("checkpoint step is not an integer")
        parent = Path(expected["resume_adapter"])
        # A disposable manifest reuses the trainer's full stats/lineage validator.
        # It is never published as an original final policy or budget stop.
        manifest = {**state, "schema": trainer.POLICY_SCHEMA,
                    "policy_update": trainer.policy_update_for_objective("grpo"),
                    "reward_source": "verifier", "reference_kl_beta": 0.,
                    "supervised_loss": False, "positive_only_filter": False, "parameterization": "lora",
                    "samples_per_step": 4 * expected["config"]["group_size"],
                    "advantage_normalization": trainer.advantage_normalization_for_objective("grpo"),
                    "token_normalization": trainer.token_normalization_for_objective("grpo"),
                    "parent_policy": str(parent),
                    "parent_policy_manifest_sha256": base.digest(parent / "policy_train.json"),
                    "parent_adapter_sha256": base.digest(parent / "adapter_model.safetensors"),
                    "parent_optimizer_sha256": base.digest(parent / "optimizer.pt")}
        with tempfile.TemporaryDirectory(prefix="mbpp-policy-check-") as temp:
            view = Path(temp)
            for name in FILES:
                (view / name).symlink_to((path / name).resolve())
            core.atomic_json(view / "policy_train.json", manifest)
            trainer.validate_policy_lineage(view, target_steps=completed, **args)
        evidence = "checkpoint_state.json"
    if type(completed) is not int or not expected["start_step"] < completed <= expected["target_steps"]:
        raise ValueError("saved policy is outside the frozen continuation interval")
    return completed, evidence


def point(adapter, step, k, seed, evidence, *, final=False):
    return {"adapter": str(adapter.resolve()), "step": step, "k": k, "seed": seed, "final": final,
            "hashes": {name: base.digest(adapter / name)
                       for name in ("adapter_model.safetensors", "adapter_config.json", evidence)}}


def prepare(p, directory):
    out, arm = directory.parent, directory.name
    c = switch.verify(out)
    protocol = switch.protocol(out.parent.parent)
    choice = core.read(directory / "decision.json")
    if choice["binding"] != {"protocol_sha256": core.fingerprint(protocol),
                             "contract_sha256": base.digest(out / "contract.json")}:
        raise ValueError("frozen decision binding changed")
    cap = core.number(choice["budget_gpu_seconds"], "budget", 0)
    measured = core.number(choice["measurement_gpu_seconds"], "diagnosis cost", 0)
    if cap != c["budget_gpu_seconds"] - measured:
        raise ValueError("frozen allocation changed")
    for path in (directory / "execution.json", out / "subsets" / f"subset-{arm}.json"):
        if core.read(path.with_suffix(".sha256.json")) != {"sha256": base.digest(path)}:
            raise ValueError(f"saved input hash changed: {path}")
    expected = checkpoint_contract(out, c, arm)
    policy = directory / "policy"
    if (policy / "budget_stop.json").is_file() and core.read(policy / "budget_stop.json").get("use_parent_policy"):
        raise ValueError("parent-only stop is not a saved continuation; manual review required")
    plan_path = directory / "budget-recovery/plan.json"
    if plan_path.exists():
        selected = Path(core.read(plan_path)["points"][-1]["adapter"])
        if selected != policy.resolve() and selected.parent != policy.resolve():
            raise ValueError("recovery candidate is outside the original policy")
        completed, evidence = validate_candidate(selected, out, c, arm, expected)
    elif (policy / "policy_train.json").exists():
        selected = policy
        completed, evidence = validate_candidate(selected, out, c, arm, expected)
    else:
        # Like the original resume path, choose the latest fully valid checkpoint.
        candidates = []
        for path in policy.glob("checkpoint-*"):
            try:
                step, source = validate_candidate(path, out, c, arm, expected)
                candidates.append((step, path, source))
            except (OSError, ValueError, KeyError, TypeError):
                continue
        if not candidates:
            raise ValueError("no valid saved continuation checkpoint; no parent restart or new training")
        completed, selected, evidence = max(candidates, key=lambda row: (row[0], str(row[1])))
    points = []
    curve = switch.curve_config(p)
    if curve:
        saved = {}
        for path in (policy / "curve-checkpoints").glob("step-*"):
            state = core.read(path / "checkpoint_state.json")
            step = state.get("completed_steps")
            if (type(step) is not int or path.name != f"step-{step}"
                    or any(state.get(key) != value for key, value in expected.items())
                    or base.digest(path / "adapter_model.safetensors") != state.get("adapter_sha256")):
                raise ValueError(f"archived curve checkpoint changed: {path}")
            saved[step] = path
        start = expected["start_step"]
        steps = switch.curve_steps(start, completed, switch.curve_fractions(curve["points"]), saved)
        for step in [start, *steps]:
            adapter = Path(expected["resume_adapter"]) if step == start else saved[step]
            source = "policy_train.json" if step == start else "checkpoint_state.json"
            points.append(point(adapter, step, curve["k"], c["eval_seed"] + 7919 * (step + 1), source))
    points.append(point(selected, completed, c["eval_k"], c["eval_seed"], evidence, final=True))
    inputs = [out / "contract.json", directory / "decision.json", directory / "execution.json",
              out / "subsets" / f"subset-{arm}.json", directory / "cost.jsonl"]
    plan = {"schema": SCHEMA, "canonical_complete": False, "purpose": "posthoc_saved_policy_evaluation",
            "runner_sha256": base.digest(HERE), "switch_protocol_sha256": core.fingerprint(p),
            "budget_gpu_seconds": cap, "used_gpu_seconds": base.spent(directory),
            "over_budget_gpu_seconds": max(0., base.spent(directory) - cap),
            "original_cost": base.cost(directory), "arm": arm, "start_step": expected["start_step"],
            "completed_steps": completed, "points": points,
            "inputs": {str(path.resolve()): base.digest(path) for path in inputs}}
    base.bind(plan_path, plan)
    return c, plan


def shard_info(directory, c, plan, index, shard):
    item = plan["points"][index]
    for name, digest in item["hashes"].items():
        if base.digest(Path(item["adapter"]) / name) != digest:
            raise ValueError("recovery evaluation policy changed")
    n = len(c["evaluation"]["val"])
    indices = range(n * shard // 4, n * (shard + 1) // 4)
    target = directory / "budget-recovery" / f"point-{index}"
    binding = {"plan_sha256": base.digest(directory / "budget-recovery/plan.json"),
               "point": index, "shard": shard}
    return item, indices, target, binding


def shard_rows(directory, c, plan, index, shard):
    import evidence_downstream as ed
    item, indices, target, binding = shard_info(directory, c, plan, index, shard)
    path = target / f"shard-{shard}.jsonl"
    if core.read(path.with_suffix(".done.json")) != {"binding": binding, "sha256": base.digest(path)}:
        raise ValueError("recovery evaluation completion changed")
    return ed.reward_rows(path, indices, item["k"])


def evaluate(directory, index, shard):
    import evidence_downstream as ed
    plan = core.read(directory / "budget-recovery/plan.json")
    if plan["runner_sha256"] != base.digest(HERE):
        raise ValueError("recovery runtime changed")
    for name, digest in plan["inputs"].items():
        if base.digest(Path(name)) != digest:
            raise ValueError(f"original recovery input changed: {name}")
    c = switch.verify(directory.parent)
    item, indices, target, binding = shard_info(directory, c, plan, index, shard)
    with base.lease(target / f"shard-{shard}.lock"):
        base.bind(target / f"shard-{shard}.contract.json", binding)
        path = target / f"shard-{shard}.jsonl"
        if path.with_suffix(".done.json").exists():
            shard_rows(directory, c, plan, index, shard)
            return
        from rollout import collect_rollouts, load_policy
        model, tokenizer = load_policy(c["config"]["model"], Path(item["adapter"]))
        collect_rollouts(model, tokenizer, c["evaluation"]["val"][indices.start:indices.stop], item["k"],
                         c["config"]["max_new_tokens"], float(c["config"]["temperature"]), path,
                         idx_offset=indices.start, sampling_seed_base=item["seed"])
        ed.reward_rows(path, indices, item["k"])
        base.bind(path.with_suffix(".done.json"), {"binding": binding, "sha256": base.digest(path)})


def reuse_shard(directory, c, plan, index, shard):
    """Reuse an already sealed canonical measurement with identical sampling."""
    item, indices, target, binding = shard_info(directory, c, plan, index, shard)
    if (target / f"shard-{shard}.done.json").exists():
        return
    adapter = Path(item["adapter"])
    if item["final"]:
        if adapter != (directory / "policy").resolve():
            return
        source = directory / "evaluation"
        original_binding = {"experiment_sha256": base.digest(directory.parent / "contract.json"),
                            "adapter_sha256": item["hashes"]["adapter_model.safetensors"],
                            "policy_manifest_sha256": item["hashes"]["policy_train.json"],
                            "arm": directory.name, "shard": shard}
    else:
        source = switch.curve_point_dir(directory.parent, directory.name, item["step"], plan["start_step"])
        original_binding = {"experiment_sha256": base.digest(directory.parent / "contract.json"),
                            "adapter_sha256": item["hashes"]["adapter_model.safetensors"],
                            "arm": "parent" if item["step"] == plan["start_step"] else directory.name,
                            "step": item["step"], "shard": shard, "k": item["k"]}
    path = source / f"shard-{shard}.jsonl"
    done = path.with_suffix(".done.json")
    if not done.exists():
        return
    import evidence_downstream as ed
    if core.read(done) != {"binding": original_binding, "sha256": base.digest(path)}:
        raise ValueError("saved canonical evaluation changed; cannot reuse it")
    ed.reward_rows(path, indices, item["k"])
    copy = target / path.name
    target.mkdir(parents=True, exist_ok=True)
    if copy.exists() or copy.is_symlink():
        if base.digest(copy) != base.digest(path):
            raise ValueError("partial recovery evaluation differs from saved canonical evaluation")
    else:
        copy.symlink_to(path.resolve())
    base.bind(target / f"shard-{shard}.contract.json", binding)
    base.bind(copy.with_suffix(".done.json"), {"binding": binding, "sha256": base.digest(path)})


def recover(p, directory, devices, env, *, prepared=None):
    c, plan = prepare(p, directory) if prepared is None else prepared
    target = directory / "budget-recovery"
    base.spent(target)
    suite = core.read(directory.parent.parent.parent / "suite.json")
    points = []
    for index, item in enumerate(plan["points"]):
        commands = []
        for shard in range(4):
            reuse_shard(directory, c, plan, index, shard)
            if (target / f"point-{index}/shard-{shard}.done.json").exists():
                shard_rows(directory, c, plan, index, shard)
            else:
                commands.append(([sys.executable, str(HERE), "--directory", str(directory),
                                  "--point", str(index), "--shard", str(shard)], devices[shard]))
        if commands:
            if (target / "result.json").exists():
                raise ValueError("published recovery evaluation lost a shard; preserve the result for review")
            base.meter(target, "evaluate", c["scope"]["gpu_type"], commands=commands, env=env,
                       timeout=suite["eval_timeout"], ledger="reporting")
        values = {str(i): [] for i in range(len(c["evaluation"]["val"]))}
        for shard in range(4):
            for row in shard_rows(directory, c, plan, index, shard):
                values[str(row["prompt_idx"])].append(row["reward"])
        rewards = {key: statistics.fmean(value) for key, value in values.items()}
        points.append({"step": item["step"], "k": item["k"], "final": item["final"], "rewards": rewards})
    base.spent(target)
    result = {"schema": SCHEMA, "canonical_complete": False, "evaluation_complete": True,
              "plan_sha256": base.digest(target / "plan.json"), "points": points,
              "original_cost": plan["original_cost"], "recovery_cost": base.cost(target),
              "budget_gpu_seconds": plan["budget_gpu_seconds"], "used_gpu_seconds": plan["used_gpu_seconds"],
              "over_budget_gpu_seconds": plan["over_budget_gpu_seconds"]}
    base.bind(target / "result.json", result)
    base.bind(target / "result.sha256.json", {"sha256": base.digest(target / "result.json")})
    print(f"[WAIT] {directory}: saved-policy evaluation recovered; original budget exceeded by "
          f"{plan['over_budget_gpu_seconds']:.3f} GPU-s; not canonical DONE", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--point", type=int, required=True)
    parser.add_argument("--shard", type=int, choices=range(4), required=True)
    args = parser.parse_args()
    if args.point < 0:
        parser.error("point must be nonnegative")
    switch.install_runtime()
    evaluate(args.directory, args.point, args.shard)
