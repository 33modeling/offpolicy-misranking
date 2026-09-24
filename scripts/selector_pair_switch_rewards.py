"""Execute the measured SR-GC switch from existing Pair checkpoints.

Missing trigger optimizers are recovered by replaying from the full step-25
parent before training the SR suffix. Shared-filesystem workers claim independent
training/evaluation jobs; source Pair artifacts are read-only. Stored D fixes
the first two-negative trigger before any new reward is evaluated.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import io
import json
import math
import os
import shutil
import signal
import statistics
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import selector_pair_srgc as srgc
import selector_pair_srgc_repeat as repeat
from selector_pair_resume_artifacts import (
    ResumeArtifactsUnavailable,
    artifact_bytes,
    artifact_digest,
    find_resume_artifacts,
)

import selection_gate as core
import selection_gate_gpu as base

SCHEMA = "offpolicy-selector-pair/executed-switch-rewards-v1"
START = 25
INTERVAL = 25
ARMS = ("random", "on_policy", "cached", "switch")
LABELS = {"random": "Random", "on_policy": "On-policy continued",
          "cached": "SR from step 25", "switch": "On-policy -> SR"}
FILES = {"adapter_sha256": "adapter_model.safetensors",
         "optimizer_sha256": "optimizer.pt", "grpo_stats_sha256": "grpo_stats.jsonl"}


@contextlib.contextmanager
def task_lease(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield handle.fileno()


@contextlib.contextmanager
def inherited_task_lock(fd):
    # A killed controller must not release a task still running on its GPUs.
    # Metered children inherit this descriptor and retain the shared file lock.
    original = base.subprocess
    base.subprocess = SimpleNamespace(
        Popen=lambda *a, **kw: original.Popen(*a, **kw, pass_fds=(fd,)), STDOUT=original.STDOUT)
    try:
        yield
    finally:
        base.subprocess = original


def check_paths(root, output):
    if root == output or root in output.parents or output in root.parents:
        raise ValueError("switch output must be separate from, not inside, the source Pair root")
    if not (root / "pair.json").is_file():
        raise FileNotFoundError(f"Pair protocol not found: {root / 'pair.json'}")


def source_arm(root, seed, arm):
    branch = "cached" if arm == "cached" else "on_policy"
    name = "random_full" if arm == "random" else "selection_full"
    return root / f"branches/{branch}/states/s{seed}-t25/points/view-25" / name


def saved_checkpoints(policy):
    found = {}
    # Full checkpoints have optimizer state. Never select an adapter-only archive
    # as a training resume point just because it was discovered first.
    for directory, pattern, prefix in ((policy, "checkpoint-*", "checkpoint-"),
                                       (policy / "curve-checkpoints", "step-*", "step-")):
        for path in sorted(directory.glob(pattern)):
            suffix = path.name.removeprefix(prefix)
            if suffix.isdigit() and (path / "checkpoint_state.json").is_file():
                found.setdefault(int(suffix), path)
    return found


def checked_checkpoint(path, *, seed, step, start, subset, parent, config, full=False):
    import evidence_downstream as ed
    state = core.read(path / "checkpoint_state.json")
    expected = {"schema": "offpolicy-grpo-checkpoint/v2", "training_objective": "grpo",
                "seed": seed, "completed_steps": step, "start_step": start, "world_size": 4,
                "prompts_sha256": base.digest(subset), "base_model": str(Path(config["model"]).resolve()),
                "max_new_tokens": config["max_new_tokens"], "prompt_format": config["prompt_format"],
                "config": ed._expected_config(config), "resume_adapter": str(parent.resolve()),
                "resume_optimizer": str((parent / "optimizer.pt").resolve())}
    if any(state.get(key) != value for key, value in expected.items()):
        raise ValueError(f"checkpoint identity/lineage/configuration mismatch: {path}")
    for key, name in FILES.items() if full else [("adapter_sha256", FILES["adapter_sha256"])]:
        if not (path / name).is_file() or state.get(key) != base.digest(path / name):
            raise ValueError(f"checkpoint missing or changed {name}: {path}")
    if not (path / "adapter_config.json").is_file():
        raise ValueError(f"checkpoint missing adapter_config.json: {path}")
    return state


def initial_choice(root, seed):
    """Validate stored D without repairing or depending on historical cost ledgers."""
    directory = root / f"sr-gc/s{seed}-t25"
    initial = core.read(directory / "decision.json")
    reference = core.read(directory / "reference.json")
    if (initial["seed"] != seed or initial["step"] != START
            or initial["protocol_id"] != core.read(root / "pair.json")["protocol_id"]
            or initial["reference_sha256"] != base.digest(directory / "reference.json")
            or initial["state_id"] != reference["state_id"] or initial["sets"] != reference["sets"]
            or initial["reference_shards"] != {f"{stage}-{i}.json": base.digest(directory / f"{stage}-{i}.json")
                                              for stage in srgc.score.STAGES for i in range(4)}):
        raise ValueError("stored initial D identity or projection binding changed")
    for path, digest in ((Path(reference["parent"]) / "adapter_model.safetensors", reference["adapter_sha256"]),
                         (Path(reference["prompts"]), reference["prompts_sha256"]),
                         (Path(reference["parent"]).parent / "rollouts_behavior_train.jsonl", initial["cache_sha256"])):
        if base.digest(path) != digest:
            raise ValueError(f"initial D source changed: {path}")
    recomputed = srgc.reference_contrast(directory, initial["sets"])
    if any(not math.isclose(recomputed[key], initial[key], rel_tol=1e-9, abs_tol=1e-9)
           for key in ("d", "d_a", "d_b")) or initial["selector"] != recomputed["selector"]:
        raise ValueError("initial D differs from the measured projections")
    return initial


def decision(root, seed):
    initial = initial_choice(root, seed)
    checks = [{"step": START, "d": initial["d"], "source_sha256":
               base.digest(root / f"sr-gc/s{seed}-t25/decision.json")}]
    checkpoints = repeat.inventory(root, seed, START)
    for step in range(START + INTERVAL, max(checkpoints, default=START) + 1, INTERVAL):
        if step not in checkpoints:
            raise ValueError(f"seed={seed} missing On-policy checkpoint at step={step}; cannot skip a D check")
        reference, _ = repeat.checkpoint_reference(root, seed, START, step, checkpoints[step], initial, INTERVAL)
        directory = repeat.output_dir(root, seed, START, INTERVAL) / f"step-{step}"
        value = repeat.check_projections(directory, root, reference)
        d = value["d"]
        if not math.isfinite(d):
            raise ValueError(f"seed={seed} non-finite D at step={step}")
        checks.append({"step": step, "d": d, "source_sha256": value["reference_sha256"]})
        if checks[-2]["d"] < 0 and d < 0:
            return step, checks, initial
    raise ValueError(f"seed={seed}: no confirmed two-consecutive-negative switch in saved D")


def common_horizon(inventories, switch):
    common = set.intersection(*(set(values) for values in inventories.values()))
    # A saved common checkpoint, never a reward-selected endpoint or target35.
    candidates = [step for step in common if step > switch]
    if not candidates:
        raise ValueError("no saved common control checkpoint after the switch")
    return max(candidates)


def estimate_hours(stats_path, updates, phase="SR suffix"):
    rows = [json.loads(line) for line in stats_path.read_text().splitlines() if line.strip()]
    seconds = sorted(float(row["step_seconds"]) for row in rows
                     if isinstance(row.get("step_seconds"), (int, float))
                     and math.isfinite(row["step_seconds"]) and row["step_seconds"] > 0)
    if not seconds:
        return None
    return {"median_hours": statistics.median(seconds) * updates / 3600,
            "p90_hours": seconds[math.ceil(.9 * len(seconds)) - 1] * updates / 3600,
            "scope": f"{phase} training only; excludes evaluation, startup and interruptions"}


def make_plan(root, seed):
    switch, checks, initial = decision(root, seed)
    on = source_arm(root, seed, "on_policy")
    contract = core.read(on.parent / "contract.json")
    config = contract["config"]
    parent = Path(contract["source_run"]) / "policy_step_25"
    reference = core.read(root / f"sr-gc/s{seed}-t25/reference.json")
    if parent.resolve() != Path(reference["parent"]).resolve():
        raise ValueError("Pair parent and measured D parent differ")
    prompts = core.read(Path(reference["prompts"]))
    controls, inventories, bindings = {}, {}, {}
    for arm in ARMS[:-1]:
        directory = source_arm(root, seed, arm)
        source = core.read(directory.parent / "contract.json")
        for key in ("config", "source_hashes", "evaluation", "eval_k", "eval_seed"):
            if source[key] != contract[key]:
                raise ValueError(f"{arm} has a different common-parent/evaluation contract: {key}")
        subset = directory.parent / "subsets" / f"subset-{directory.name}.json"
        if arm in ("on_policy", "cached") and core.read(subset)["train"] != [
                prompts["train"][i] for i in initial["sets"][arm]]:
            raise ValueError(f"{arm} subset differs from the subset used to measure D")
        controls[arm] = str(directory)
        inventories[arm] = saved_checkpoints(directory / "policy")
        for path in (directory.parent / "contract.json", subset):
            bindings[str(path)] = base.digest(path)
    horizon = common_horizon(inventories, switch)
    checkpoint = inventories["on_policy"].get(switch)
    if checkpoint is None:
        raise ValueError(f"seed={seed} step={switch}: On-policy checkpoint not found")
    state = checked_checkpoint(checkpoint, seed=seed, step=switch, start=START,
                               subset=on.parent / "subsets/subset-selection_full.json",
                               parent=parent, config=config)
    resume_mode = "exact_checkpoint"
    try:
        artifacts, discovery = find_resume_artifacts(root, checkpoint, on / "policy", state)
    except ResumeArtifactsUnavailable as exc:
        if exc.report["missing"] != ["optimizer.pt"]:
            raise
        from train_policy_grpo import validate_policy_lineage, validate_policy_manifest
        validate_policy_manifest(parent, target_steps=START, world_size=4,
                                 training_objective="grpo", require_complete_hashes=True)
        final = core.read(on / "policy/policy_train.json")
        validate_policy_lineage(on / "policy", target_steps=final["completed_steps"], world_size=4,
                                training_objective="grpo", expected_start_step=START,
                                expected_parent=parent, expected_seed=seed,
                                expected_model=Path(config["model"]), expected_config=state["config"],
                                expected_prompts=on.parent / "subsets/subset-selection_full.json",
                                expected_max_new_tokens=state["max_new_tokens"],
                                expected_prompt_format=state["prompt_format"], require_complete_hashes=True)
        artifacts, discovery = exc.report["artifacts"], exc.report
        resume_mode = "replay_on_policy_from_25"
        print(f"[switch] seed={seed} original trigger optimizer unavailable; "
              f"replay On-policy {START} -> {switch} from saved model AND optimizer; originals retained", flush=True)
    endpoints = {}
    for arm, inventory in inventories.items():
        directory = Path(controls[arm])
        selected = inventory[horizon]
        arm_contract = core.read(directory.parent / "contract.json")
        checked_checkpoint(selected, seed=seed, step=horizon, start=START,
                           parent=Path(arm_contract["source_run"]) / "policy_step_25",
                           config=config, subset=directory.parent / "subsets" / f"subset-{directory.name}.json")
        endpoints[arm] = str(selected)
    for path in (checkpoint / "checkpoint_state.json", parent / "policy_train.json",
                 parent / "adapter_model.safetensors", parent / "optimizer.pt", parent / "adapter_config.json",
                 parent / "grpo_stats.jsonl", on / "policy/policy_train.json",
                 Path(config["model"]) / "config.json"):
        bindings[str(path)] = base.digest(path)
    return {"schema": SCHEMA, "seed": seed, "pair_root": str(root), "start_step": START,
            "switch_step": switch, "end_step": horizon, "interval": INTERVAL,
            "decision_rule": "two consecutive D < 0 checks, 25 steps apart; SR thereafter",
            "d_checks_through_switch": checks, "config": config, "contract": contract,
            "source_bindings": bindings, "controls": controls, "endpoints": endpoints,
            "checkpoint": str(checkpoint), "checkpoint_state": state,
            "resume_artifacts": artifacts, "resume_discovery": discovery, "resume_mode": resume_mode,
            "replay_training_estimate": estimate_hours(on / "policy/grpo_stats.jsonl", switch-START,
                                                        "On-policy prefix replay")
                if resume_mode == "replay_on_policy_from_25" else None,
            "source_policy": str(on / "policy"),
            "sr_subset": str(Path(controls["cached"]).parent / "subsets/subset-selection_full.json"),
            "training_estimate": estimate_hours(Path(controls["cached"]) / "policy/grpo_stats.jsonl", horizon-switch),
            "comparison": "Common On-policy prefix 0-25; controls differ from step 25. "
                          "Switch uses the saved trigger checkpoint or a separately measured On-policy replay "
                          "from step 25 when its optimizer is missing. Final rewards share one saved step.",
            "execution_scope": "Retrospective replay of a prefix-only decision rule, with a genuinely "
                               "trained switched suffix; not an untouched prospective experiment. "
                               "A regenerated prefix uses the previously frozen trigger, not newly measured D; "
                               "original/replayed checkpoint hash agreement is reported separately."}


def needs_replay(plan):
    return plan.get("resume_mode") == "replay_on_policy_from_25"


def verify_plan(plan):
    for filename, digest in plan["source_bindings"].items():
        if base.digest(Path(filename)) != digest:
            raise ValueError(f"frozen source changed: {filename}")
    checkpoint = Path(plan["checkpoint"])
    for key, filename in FILES.items():
        if filename == "optimizer.pt" and needs_replay(plan):
            continue  # Never substitute a later optimizer for the missing one.
        descriptor = plan.get("resume_artifacts", {}).get(filename, {"path": str(checkpoint / filename)})
        if artifact_digest(descriptor) != plan["checkpoint_state"][key]:
            raise ValueError(f"resume checkpoint changed: {descriptor['path']}")


def ensure_plan(root, output, seed):
    path = output / f"s{seed}/plan.json"
    with base.lease(path.parent / ".plan.lock", blocking=True):
        if path.exists():
            plan = core.read(path)
            if plan.get("schema") != SCHEMA or plan.get("seed") != seed or plan.get("pair_root") != str(root):
                raise ValueError(f"incompatible saved switch plan: {path}")
            verify_plan(plan)
        else:
            try:
                plan = make_plan(root, seed)
            except ResumeArtifactsUnavailable as exc:
                core.atomic_json(path.parent / "resume-sources.json", exc.report)
                print(f"[switch] resume search report: {path.parent / 'resume-sources.json'}", flush=True)
                raise
            base.bind(path, plan)
    return plan


def inspect_checkpoints(root, output, seeds):
    reports = []
    for seed in seeds:
        try:
            plan = make_plan(root, seed)
            reports.append({"status": "replay_required" if needs_replay(plan) else "ready",
                            **plan["resume_discovery"], "resume_mode": plan["resume_mode"],
                            "replay_from_step": START if needs_replay(plan) else None,
                            "replay_training_estimate": plan["replay_training_estimate"]})
        except ResumeArtifactsUnavailable as exc:
            reports.append({"status": "missing_resume_artifacts", **exc.report})
        except (OSError, ValueError, KeyError) as exc:
            reports.append({"seed": seed, "status": "invalid_source", "error": str(exc)})
        current = reports[-1]
        print(f"[checkpoints] seed={seed} status={current['status']} "
              f"step={current.get('step')} missing={current.get('missing', [])}", flush=True)
    core.atomic_json(output / "checkpoint-search.json", {"seeds": reports})
    path = Path.home() / "selector-pair-switch-checkpoints.txt"
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    temporary.write_text("SR-GC SWITCH RESUME CHECKPOINT SEARCH\n" + json.dumps(reports, indent=2) + "\n")
    temporary.replace(path)
    print(f"[checkpoints] report: {path}", flush=True)
    return all(row["status"] in ("ready", "replay_required") for row in reports)


def materialize_parent(directory, plan):
    """Publish a validated manifest view of the immutable full source checkpoint."""
    from train_policy_grpo import validate_policy_lineage, validate_policy_manifest
    target = directory / "parent"
    checkpoint, state = Path(plan["checkpoint"]), plan["checkpoint_state"]
    replay = needs_replay(plan)
    if replay and not training_complete(directory, plan, replay=True):
        raise ValueError("On-policy replay must finish before SR continuation")
    source_policy = directory / "replay/policy" if replay else Path(plan.get("source_policy", checkpoint.parent))
    final = validate_policy_manifest(source_policy, world_size=4, training_objective="grpo",
                                     require_complete_hashes=True)
    if final["completed_steps"] < plan["switch_step"] or final["start_step"] != START:
        raise ValueError("source policy manifest does not cover the switch checkpoint")
    if replay:
        manifest = {**final, "materialized_from_replay": str(source_policy)}
        comparison = {key: {"original": state[key], "replayed": final[key],
                            "matches": state[key] == final[key]} for key in FILES}
        audit = {"seed": plan["seed"], "start_step": START, "switch_step": plan["switch_step"],
                 "source_manifest_sha256": base.digest(source_policy / "policy_train.json"),
                 "original_checkpoint": str(checkpoint), "replayed_policy": str(source_policy),
                 "hash_comparison": comparison,
                 "model_optimizer_byte_identical": all(comparison[key]["matches"]
                     for key in ("adapter_sha256", "optimizer_sha256")),
                 "decision_scope": "Previously frozen original-trajectory trigger; D not recomputed on replay."}
        base.bind(directory / "replay-audit.json", audit)
        descriptors = {name: {"path": str(source_policy / name)}
                       for name in (*FILES.values(), "adapter_config.json")}
    else:
        manifest = {**final, "completed_steps": plan["switch_step"],
                    **{key: state[key] for key in FILES},
                    "materialized_from_checkpoint": str(checkpoint),
                    "checkpoint_state_sha256": base.digest(checkpoint / "checkpoint_state.json")}
        descriptors = plan.get("resume_artifacts", {})
    manifest.pop("training_budget", None)
    if not target.exists():
        temporary = directory / f".parent-{uuid.uuid4().hex}"
        temporary.mkdir(parents=True)
        for filename in (*FILES.values(), "adapter_config.json"):
            descriptor = descriptors.get(filename, {"path": str(checkpoint / filename)})
            if "through_step" in descriptor:
                (temporary / filename).write_bytes(artifact_bytes(descriptor))
            else:
                shutil.copy2(descriptor["path"], temporary / filename)
        core.atomic_json(temporary / "policy_train.json", manifest)
        temporary.rename(target)
    else:
        base.bind(target / "policy_train.json", manifest)
    validate_policy_lineage(target, target_steps=plan["switch_step"], world_size=4,
                            training_objective="grpo", expected_start_step=START,
                            expected_parent=Path(state["resume_adapter"]),
                            expected_model=Path(plan["config"]["model"]), expected_seed=plan["seed"],
                            expected_config=state["config"], expected_max_new_tokens=state["max_new_tokens"],
                            expected_prompt_format=state["prompt_format"],
                            expected_prompts=Path(plan["controls"]["on_policy"]).parent /
                            "subsets/subset-selection_full.json", require_complete_hashes=True)
    return target


def training_spec(directory, plan, replay=False):
    if replay:
        return (directory / "replay/policy", Path(plan["contract"]["source_run"]) / "policy_step_25",
                START, plan["switch_step"], Path(plan["controls"]["on_policy"]).parent /
                "subsets/subset-selection_full.json")
    return directory / "policy", directory / "parent", plan["switch_step"], plan["end_step"], Path(plan["sr_subset"])


def train_command(directory, plan, replay=False):
    import evidence_downstream as ed
    policy, parent, start, end, subset = training_spec(directory, plan, replay)
    config = {**plan["config"], "drift": start}
    args = ed.train_args(config, directory, directory, "switch", end-start)
    args[args.index(str(REPO / "src/train_policy_grpo.py"))] = str(REPO / "src/selection_switch_curve_train.py")
    for flag, value in (("--prompts", subset), ("--output", policy),
                        ("--resume-adapter", parent), ("--resume-optimizer", parent / "optimizer.pt")):
        args[args.index(flag)+1] = str(value)
    return [sys.executable, *args]


def training_complete(directory, plan, replay=False):
    policy, parent, start, end, subset = training_spec(directory, plan, replay)
    path = policy / "policy_train.json"
    if not path.is_file():
        return False
    from train_policy_grpo import validate_policy_lineage
    try:
        validate_policy_lineage(path.parent, target_steps=end, world_size=4,
                                training_objective="grpo", expected_start_step=start,
                                expected_parent=parent, expected_seed=plan["seed"],
                                expected_model=Path(plan["config"]["model"]),
                                expected_config=plan["checkpoint_state"]["config"],
                                expected_prompt_format=plan["config"]["prompt_format"],
                                expected_max_new_tokens=plan["config"]["max_new_tokens"],
                                expected_prompts=subset, require_complete_hashes=True)
    except (OSError, ValueError, KeyError):
        return False  # The existing trainer repairs interrupted final publication.
    return True


def preserve_early_interruption(directory, plan, replay=False):
    policy, _, start, _, _ = training_spec(directory, plan, replay)
    if not policy.exists() or list(policy.glob("checkpoint-*")) or (policy / "policy_train.json").exists():
        return
    # Before the first durable checkpoint, preserve the interrupted attempt and
    # resume the unchanged parent. Never truncate or delete its partial history.
    stats = policy / "grpo_stats.jsonl"
    if not stats.exists() and not any(policy.iterdir()):
        return
    if any((policy / name).exists() for name in ("optimizer.pt", "adapter_model.safetensors")):
        raise ValueError(f"unpublished weights without a full checkpoint; preserve and inspect {policy}")
    if stats.exists():
        complete = stats.read_text().splitlines()
        rows = []
        for line in complete:
            try:
                rows.append(json.loads(line))
            except ValueError:
                if line != complete[-1]:
                    raise
        first_save = (start // 5 + 1) * 5
        if any(row.get("step", 0) > first_save for row in rows):
            raise ValueError("training passed a checkpoint boundary but no checkpoint remains")
    preserved = policy.parent / "interrupted-attempts" / uuid.uuid4().hex
    preserved.parent.mkdir(parents=True, exist_ok=True)
    policy.rename(preserved)
    print(f"[switch] seed={plan['seed']} preserved interrupted pre-checkpoint attempt: {preserved}", flush=True)


def point_steps(plan):
    return sorted({0, START, plan["switch_step"], plan["end_step"],
                   *range(START, plan["end_step"]+1, INTERVAL)})


def point_adapter(directory, plan, arm, step):
    if step == 0:
        return None
    if step == START:
        return Path(plan["contract"]["source_run"]) / "policy_step_25"
    if arm == "switch" and step <= plan["switch_step"] and not needs_replay(plan):
        arm = "on_policy"
    if arm == "switch":
        policy, parent, start, _, subset = training_spec(directory, plan, step <= plan["switch_step"])
        path = policy / f"checkpoint-{step:06d}"
        if not (path / "checkpoint_state.json").is_file():
            return False
        checked_checkpoint(path, seed=plan["seed"], step=step, start=start,
                           subset=subset, parent=parent, config=plan["config"])
        return path
    path = Path(plan["endpoints"][arm]) if step == plan["end_step"] else saved_checkpoints(
        Path(plan["controls"][arm]) / "policy").get(step)
    if path is None:
        raise ValueError(f"seed={plan['seed']} {arm} missing saved checkpoint at step={step}")
    source = Path(plan["controls"][arm])
    arm_contract = core.read(source.parent / "contract.json")
    checked_checkpoint(path, seed=plan["seed"], step=step, start=START, config=plan["config"],
                       parent=Path(arm_contract["source_run"]) / "policy_step_25",
                       subset=source.parent / "subsets" / f"subset-{source.name}.json")
    return path


def canonical_arm(plan, arm, step):
    return "on_policy" if step <= START or (
        arm == "switch" and step <= plan["switch_step"] and not needs_replay(plan)) else arm


def evaluation_binding(directory, plan, arm, step, shard):
    adapter = point_adapter(directory, plan, arm, step)
    if adapter is False:
        raise FileNotFoundError(f"switch checkpoint not yet published: seed={plan['seed']} step={step}")
    n = len(plan["contract"]["evaluation"]["val"])
    return adapter, range(n*shard//4, n*(shard+1)//4), {
        "plan_sha256": base.digest(directory / "plan.json"), "arm": arm, "step": step, "shard": shard,
        "adapter_sha256": base.digest(adapter / "adapter_model.safetensors") if adapter else None,
        "base_config_sha256": base.digest(Path(plan["config"]["model"]) / "config.json"),
        "k": plan["contract"]["eval_k"]}


def evaluate_shard(directory, arm, step, shard):
    import evidence_downstream as ed
    plan = core.read(directory / "plan.json")
    adapter, indices, binding = evaluation_binding(directory, plan, arm, step, shard)
    target = directory / f"evaluations/{arm}/step-{step}"
    with base.lease(target / f"shard-{shard}.lock"):
        base.bind(target / f"shard-{shard}.contract.json", binding)
        path = target / f"shard-{shard}.jsonl"
        done = path.with_suffix(".done.json")
        if not done.exists():
            from rollout import collect_rollouts, load_policy
            model, tokenizer = load_policy(plan["config"]["model"], adapter)
            collect_rollouts(model, tokenizer, plan["contract"]["evaluation"]["val"][indices.start:indices.stop],
                             binding["k"], plan["config"]["max_new_tokens"], float(plan["config"]["temperature"]),
                             path, idx_offset=indices.start,
                             sampling_seed_base=plan["contract"]["eval_seed"] + 7919*(step+1))
            ed.reward_rows(path, indices, binding["k"])
            base.bind(done, {"binding": binding, "sha256": base.digest(path)})
        if core.read(done) != {"binding": binding, "sha256": base.digest(path)}:
            raise ValueError(f"evaluation receipt changed: {done}")
        ed.reward_rows(path, indices, binding["k"])


def measured_point(directory, plan, arm, step):
    import evidence_downstream as ed
    target = directory / f"evaluations/{arm}/step-{step}"
    if not all((target / f"shard-{shard}.done.json").exists() for shard in range(4)):
        return None
    rewards = []
    for shard in range(4):
        _, indices, binding = evaluation_binding(directory, plan, arm, step, shard)
        path = target / f"shard-{shard}.jsonl"
        if core.read(path.with_suffix(".done.json")) != {"binding": binding, "sha256": base.digest(path)}:
            raise ValueError(f"evaluation hash mismatch: {path}")
        rewards.extend(row["reward"] for row in ed.reward_rows(path, indices, binding["k"]))
    return {"step": step, "reward": statistics.fmean(rewards), "k": plan["contract"]["eval_k"],
            "question_count": len(plan["contract"]["evaluation"]["val"]), "source": str(target),
            "source_kind": "checkpoint_evaluation"}


def reused_points(plan, arm):
    """Reuse actual curve shards, not a summary or another arm's endpoint."""
    import selection_switch_gpu as switch
    source = Path(plan["controls"][arm])
    path = source / "curve.json"
    if not path.exists():
        return []
    curve = core.read(path)
    if curve["result_sha256"] != base.digest(source / "result.json"):
        raise ValueError(f"source curve result binding changed: {path}")
    k = curve["k"]
    points = []
    for text, point in curve["points"].items():
        step = int(text)
        if step > plan["end_step"] or point.get("final") or k != plan["contract"]["eval_k"]:
            continue
        reward = switch.curve_reward(source.parent, plan["contract"] if arm != "cached" else
                                     core.read(source.parent / "contract.json"), source.name, step, k)
        if not math.isclose(reward, point["reward"], abs_tol=1e-12):
            raise ValueError(f"source curve and raw rewards differ: {path} step={step}")
        points.append({"step": step, "reward": reward, "k": k,
                       "question_count": len(plan["contract"]["evaluation"]["val"]),
                       "source": str(path), "source_sha256": base.digest(path), "source_kind": "reused_curve"})
    return points


def reuse_cache(directory, plan):
    path = directory / "reused-curves.json"
    with base.lease(directory / ".reuse.lock", blocking=True):
        if not path.exists():
            base.bind(path, {arm: reused_points(plan, arm) for arm in ARMS[:-1]})
    cached = core.read(path)
    for arm, points in cached.items():
        if arm not in ARMS[:-1]:
            raise ValueError("invalid reused curve arm")
        for point in points:
            source = Path(point["source"])
            if base.digest(source) != point["source_sha256"]:
                raise ValueError(f"reused curve source changed: {source}")
            if core.read(source)["points"][str(point["step"])]["reward"] != point["reward"]:
                raise ValueError("reused curve reward differs from its frozen source")
    return cached


def tasks(directory, plan, reused):
    """Final rewards first, then sparse Switch checkpoints; controls are reused."""
    requested = set()
    for arm in ARMS:
        if arm == "switch":
            steps = point_steps(plan)
        else:
            source = Path(plan["controls"][arm]) / "curve.json"
            curve = core.read(source) if source.exists() else {"points": {}}
            existing_steps = [int(step) for step in curve["points"] if int(step) <= plan["end_step"]]
            steps = {0, START, plan["end_step"], *(existing_steps or point_steps(plan))}
        for step in steps:
            owner = canonical_arm(plan, arm, step)
            if any(point["step"] == step for point in reused.get(owner, [])):
                continue
            requested.add((owner, step))
    return sorted(requested, key=lambda item: (item[1] != plan["end_step"], item[1], item[0]))


def export_seed(directory, plan, reused):
    points = {arm: {p["step"]: p for p in reused.get(arm, [])} for arm in ARMS}
    missing = []
    for arm, step in tasks(directory, plan, reused):
        point = measured_point(directory, plan, arm, step)
        if point is None:
            missing.append({"arm": arm, "step": step})
        else:
            points[arm][step] = point
    for arm in ARMS:
        for step, point in points["on_policy"].items():
            if step <= START or (arm == "switch" and canonical_arm(plan, arm, step) == "on_policy"):
                points[arm][step] = {**point, "shared_on_policy_prefix": True}
    complete = training_complete(directory, plan)
    curves = {arm: [{**p, "segment": "on_policy" if arm == "switch" and step <= plan["switch_step"]
                    else "cached" if arm == "switch" else arm} for step, p in sorted(values.items())]
              for arm, values in points.items()}
    final = {arm: points[arm].get(plan["end_step"], {}).get("reward") for arm in ARMS}
    costs = []
    for path in (directory / "attempts").glob("*"):
        if not (path / "cost.jsonl").exists():
            continue
        try:
            costs.append({"source": str(path), "phase": path.name.split("-", 1)[0], **base.cost(path)})
        except (OSError, ValueError) as exc:
            costs.append({"source": str(path), "phase": path.name.split("-", 1)[0],
                          "complete": False, "total_gpu_seconds": None, "error": str(exc)})
    saved_step = max(saved_checkpoints(directory / "policy"), default=None)
    executed = (saved_step is not None and saved_step > plan["switch_step"]
                and point_adapter(directory, plan, "switch", saved_step) is not False)
    return {"seed": plan["seed"], "switch_step": plan["switch_step"], "end_step": plan["end_step"],
            "executed_switch": executed,
            "training_complete": complete, "complete": complete and not missing,
            "last_saved_switch_step": saved_step,
            "missing_evaluations": missing, "curves": curves, "final_rewards": final,
            "switch_minus_on_policy": (final["switch"]-final["on_policy"])
                if final["switch"] is not None and final["on_policy"] is not None else None,
            "d_checks_through_switch": plan["d_checks_through_switch"],
            "comparison": plan["comparison"], "execution_scope": plan["execution_scope"],
            "resume_mode": plan.get("resume_mode", "exact_checkpoint"),
            "replay_training_complete": training_complete(directory, plan, replay=True) if needs_replay(plan) else None,
            "replay_audit": core.read(directory / "replay-audit.json")
                if (directory / "replay-audit.json").exists() else None,
            "replay_training_estimate": plan.get("replay_training_estimate"),
            "training_estimate": plan["training_estimate"], "new_work_cost": costs,
            "new_work_cost_complete": all(cost["complete"] for cost in costs)}


def report(output, seeds, *, tolerate_errors=False):
    rows, pending, errors = [], [], []
    for seed in seeds:
        directory = output / f"s{seed}"
        if not (directory / "plan.json").exists():
            pending.append(seed)
            continue
        try:
            plan = core.read(directory / "plan.json")
            reused = core.read(directory / "reused-curves.json") if (directory / "reused-curves.json").exists() else {}
            rows.append(export_seed(directory, plan, reused))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            if not tolerate_errors:
                raise
            errors.append({"seed": seed, "error": str(exc)})
    result = {"schema": SCHEMA, "complete": not pending and not errors and all(row["complete"] for row in rows),
              "seeds": rows, "pending_seeds": pending, "errors": errors}
    return result


def write_report(output, seeds, out=None):
    from selector_pair_switch_status import atomic_text, duration, snapshot
    data = report(output, seeds, tolerate_errors=True)
    data["status"] = snapshot(output, seeds, tasks)
    stream = io.StringIO()
    stream.write("EXECUTED SR-GC SWITCH REWARDS\n")
    stream.write(f"Root: {output}\n")
    stream.write(f"Overall: {'COMPLETE' if data['complete'] else 'PARTIAL'}\n")
    if data["pending_seeds"]:
        stream.write(f"Plan not yet saved: seeds {data['pending_seeds']}\n")
    for error in data["errors"]:
        stream.write(f"Seed {error['seed']} ERROR: {error['error']}\n")
    stream.write("Rewards below are percentages at the common final step; missing is not zero.\n")
    writer = csv.writer(stream)
    writer.writerow(("seed", "switch_step", "last_saved_switch_step", "common_final_step",
                     *(LABELS[arm] + " (%)" for arm in ARMS), "complete"))
    for row in data["seeds"]:
        writer.writerow((row["seed"], row["switch_step"], row["last_saved_switch_step"], row["end_step"],
                         *(f"{100*row['final_rewards'][arm]:.3f}" if row["final_rewards"][arm] is not None
                           else "pending" for arm in ARMS), row["complete"]))
    for row in data["seeds"]:
        stream.write(f"Seed {row['seed']}: training_complete={row['training_complete']}; "
                     f"missing evaluations={len(row['missing_evaluations'])}\n")
        for arm in ARMS:
            points = row["curves"][arm]
            if points:
                latest = max(points, key=lambda point: point["step"])
                stream.write(f"  {LABELS[arm]} latest measured: step {latest['step']}, "
                             f"reward {100*latest['reward']:.3f}%\n")
        for arm in ARMS:
            missing = [str(point["step"]) for point in row["missing_evaluations"] if point["arm"] == arm]
            if missing:
                stream.write(f"  Pending {LABELS[arm]} steps: {', '.join(missing)}\n")
        for phase, label in (("replay", "On-policy replay"), ("train", "SR continuation"), ("eval", "Evaluation")):
            costs = [cost for cost in row["new_work_cost"] if cost["phase"] == phase]
            if not costs:
                continue
            closed = [cost for cost in costs if cost["complete"]]
            wall = sum(item["wall_seconds"] for cost in closed for item in cost["ledgers"].values())
            gpu = sum(item["gpu_seconds"] for cost in closed for item in cost["ledgers"].values())
            stream.write(f"  {label} recorded closed-attempt time: "
                         f"{duration(wall) if closed else 'unknown'} wall, "
                         f"{f'{gpu/3600:.3f}' if closed else 'unknown'} GPU-hours; "
                         f"open/unknown attempts={len(costs)-len(closed)}\n")
    stream.write("Phase durations are summed work, not parallel elapsed completion time. "
                 "Open/interrupted costs are not treated as zero.\n")
    stream.write("\nJSON\n" + json.dumps(data, allow_nan=False, separators=(",", ":")) + "\n")
    target = Path(out) if out else output / "switch-rewards.txt"
    atomic_text(target, stream.getvalue())
    if out is None:
        home_copy = Path.home() / "selector-pair-switch-results.txt"
        atomic_text(home_copy, stream.getvalue())
        print(f"[results] copy for export: {home_copy}", flush=True)
    with base.lease(output / ".report.lock", blocking=True):
        core.atomic_json(output / "switch-rewards.json", data)
    print(stream.getvalue().split("\nJSON\n")[0], flush=True)
    print(f"[results] {target}", flush=True)
    return data


def plot_report(output, data):
    if not data["complete"]:
        print("[plot] waiting for measured switch checkpoints; no fabricated curve", flush=True)
        return
    try:
        import matplotlib
    except ImportError:
        print("[plot] rewards exported; PDF/PNG needs optional matplotlib. "
              "Install requirements-switch-plots.txt and rerun results; no GPU work repeats.", flush=True)
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"random": "#777777", "on_policy": "#a7c5e5", "cached": "#a9d4ae"}
    rows = sorted(data["seeds"], key=lambda row: -row["seed"])
    fig, axes = plt.subplots(1, len(rows), figsize=(6.0*len(rows), 4.6), squeeze=False)
    for ax, row in zip(axes[0], rows):
        for arm in ARMS[:-1]:
            points = row["curves"][arm]
            ax.plot([p["step"] for p in points], [100*p["reward"] for p in points],
                    color=colors[arm], lw=.8, marker="o", ms=2.8,
                    ls="--" if arm == "random" else "-",
                    label=f"{LABELS[arm]}: {100*row['final_rewards'][arm]:.2f}%")
        points = row["curves"]["switch"]
        for before, color in ((True, "#1565b0"), (False, "#17843c")):
            segment = [p for p in points if (p["step"] <= row["switch_step"] if before
                                             else p["step"] >= row["switch_step"])]
            ax.plot([p["step"] for p in segment], [100*p["reward"] for p in segment],
                    color=color, lw=1.0, marker="o", ms=3.2,
                    label=None if before else f"On-policy -> SR: {100*row['final_rewards']['switch']:.2f}%")
        ax.axvline(row["switch_step"], color="#b87919", lw=.8, ls=":")
        ax.set_title(f"Seed {row['seed']} | Switch at step {row['switch_step']}", fontsize=11)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Evaluation reward (%)")
        ax.set_xticks(sorted({0, 75, row["switch_step"], row["end_step"], *range(50, row["end_step"], 50)}))
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=.16)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="best", fontsize=8, frameon=False)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"switch-rewards.{suffix}", dpi=200)
    plt.close(fig)


def attempt(directory, name, plan, commands, env, seconds, lock_fd):
    location = directory / "attempts" / f"{name}-{uuid.uuid4().hex}"
    with inherited_task_lock(lock_fd):
        base.meter(location, f"s{plan['seed']}-{name}", plan["contract"]["scope"]["gpu_type"],
                   commands=commands, env=env, timeout=seconds,
                   ledger="deployment" if name == "train" else "research" if name == "replay" else "reporting")


def worker(root, output, seeds, devices, hours, idle_minutes):
    import selector_pair_gpu as pair
    if len(devices) != 4 or len(set(devices)) != 4:
        raise ValueError("worker needs four distinct allocated GPUs; training world size remains four")
    deadline, idle_since = time.monotonic() + hours*3600, time.monotonic()
    # Reuse Pair's bounded four-rank probe and only its verified overrides.
    # Receipts belong to this new root; no source experiment is changed.
    overrides = pair.admission_probe(output)
    nccl_env = {"NCCL_DEBUG": os.environ.get("NCCL_DEBUG", "WARN"), **overrides}
    plans, blocked = [], []
    for seed in seeds:
        try:
            plans.append((output / f"s{seed}", ensure_plan(root, output, seed)))
        except ResumeArtifactsUnavailable as exc:
            blocked.append(seed)
            print(f"[switch] {exc}; checking other seeds", flush=True)
    if not plans:
        print("[switch] no resumable seed. Run: bash scripts/run_selector_pair_switch_rewards.sh checkpoints", flush=True)
        return False
    active_seeds = [plan["seed"] for _, plan in plans]
    caches = {plan["seed"]: reuse_cache(directory, plan) for directory, plan in plans}
    finished_jobs = set()
    finished_training = set()
    for _, plan in plans:
        print(f"[switch] seed={plan['seed']} switch={plan['switch_step']} final={plan['end_step']} "
              f"suffix_updates={plan['end_step']-plan['switch_step']} suffix_ETA={plan['training_estimate']} "
              f"replay_ETA={plan.get('replay_training_estimate')}", flush=True)
    while time.monotonic() < deadline:
        worked = False
        # Two independent training jobs at most. Extra nodes immediately claim
        # evaluations, including newly saved checkpoints while training continues.
        for directory, plan in plans:
            if plan["seed"] in finished_training:
                continue
            if training_complete(directory, plan):
                finished_training.add(plan["seed"])
                continue
            with contextlib.ExitStack() as stack:
                try:
                    lock_fd = stack.enter_context(task_lease(directory / ".train.lock"))
                except BlockingIOError:
                    continue
                if training_complete(directory, plan):
                    continue
                verify_plan(plan)
                if needs_replay(plan) and not training_complete(directory, plan, replay=True):
                    preserve_early_interruption(directory, plan, replay=True)
                    print(f"[switch] seed={plan['seed']} recover On-policy {START} -> {plan['switch_step']} "
                          "with saved optimizer; keeping EVERY full checkpoint", flush=True)
                    attempt(directory, "replay", plan, [(train_command(directory, plan, replay=True), ",".join(devices))],
                            {**pair.environment(plan["contract"]), **nccl_env},
                            max(1, deadline-time.monotonic()), lock_fd)
                    if not training_complete(directory, plan, replay=True):
                        raise ValueError("trainer exited without a validated On-policy replay")
                    if time.monotonic() >= deadline:
                        return False
                materialize_parent(directory, plan)
                preserve_early_interruption(directory, plan)
                print(f"[switch] seed={plan['seed']} train SR suffix {plan['switch_step']} -> {plan['end_step']}", flush=True)
                attempt(directory, "train", plan, [(train_command(directory, plan), ",".join(devices))],
                        {**pair.environment(plan["contract"]), **nccl_env},
                        max(1, deadline-time.monotonic()), lock_fd)
                if not training_complete(directory, plan):
                    raise ValueError("trainer exited without a validated final switch policy")
                finished_training.add(plan["seed"])
                worked = True
                break
        if not worked:
            jobs = [(directory, plan, arm, step) for directory, plan in plans
                    for arm, step in tasks(directory, plan, caches[plan["seed"]])]
            jobs.sort(key=lambda job: (job[3] != job[1]["end_step"], job[3], job[1]["seed"]))
            for directory, plan, arm, step in jobs:
                job_key = (plan["seed"], arm, step)
                if job_key in finished_jobs:
                    continue
                if measured_point(directory, plan, arm, step) is not None:
                    finished_jobs.add(job_key)
                    continue
                if arm == "switch" and point_adapter(directory, plan, arm, step) is False:
                    continue
                target = directory / f"evaluations/{arm}/step-{step}"
                with contextlib.ExitStack() as stack:
                    try:
                        lock_fd = stack.enter_context(task_lease(target / ".task.lock"))
                    except BlockingIOError:
                        continue
                    if measured_point(directory, plan, arm, step) is not None:
                        continue
                    commands = [([sys.executable, str(Path(__file__).resolve()), "evaluate-shard",
                                  "--output", str(output), "--seed", str(plan["seed"]),
                                  "--arm", arm, "--step", str(step), "--shard", str(shard)], devices[shard])
                                for shard in range(4) if not (target / f"shard-{shard}.done.json").exists()]
                    print(f"[switch] seed={plan['seed']} evaluate {arm} step={step} ({len(commands)} shards)", flush=True)
                    attempt(directory, f"eval-{arm}-{step}", plan, commands,
                            {**pair.environment(plan["contract"]), **nccl_env},
                            max(1, deadline-time.monotonic()), lock_fd)
                    measured_point(directory, plan, arm, step)
                    finished_jobs.add(job_key)
                    worked = True
                    break
        if worked:
            idle_since = time.monotonic()
            continue
        data = report(output, active_seeds)
        if data["complete"]:
            with base.lease(output / ".publication.lock", blocking=True):
                data = write_report(output, seeds)
                plot_report(output, data)
            return not blocked
        if time.monotonic()-idle_since >= idle_minutes*60:
            print("[switch] no unclaimed ready work; worker exits without changing peer jobs. Rerun the same command.", flush=True)
            return
        print("[switch] ready work held by peers or waiting for a saved checkpoint; rechecking in 30s", flush=True)
        time.sleep(min(30, max(0, deadline-time.monotonic())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "run", "status", "results", "checkpoints", "evaluate-shard"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(3, 4), action="append")
    parser.add_argument("--hours", type=float, default=24)
    parser.add_argument("--idle-minutes", type=float, default=120)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--watch", type=float, nargs="?", const=15)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--step", type=int)
    parser.add_argument("--shard", type=int, choices=range(4))
    args = parser.parse_args()
    seeds = args.seed or [3, 4]
    output = args.output.resolve()
    if args.watch is not None and (args.mode != "status" or not math.isfinite(args.watch) or args.watch <= 0):
        parser.error("--watch requires status and a positive finite interval")
    if args.json and args.mode != "status":
        parser.error("--json is only available for status")
    for value in (args.hours, args.idle_minutes):
        if not math.isfinite(value) or value <= 0:
            parser.error("worker hours and idle minutes must be positive and finite")
    if args.mode == "evaluate-shard":
        if len(seeds) != 1 or args.arm is None or args.step is None or args.shard is None:
            parser.error("evaluate-shard needs one seed, arm, step and shard")
        evaluate_shard(output / f"s{seeds[0]}", args.arm, args.step, args.shard)
        return
    if args.mode == "status":
        from selector_pair_switch_status import write_status
        try:
            while True:
                write_status(output, seeds, tasks, out=args.out, json_output=args.json)
                if args.watch is None:
                    break
                time.sleep(args.watch)
        except KeyboardInterrupt:
            pass
        return
    if args.mode == "results":
        with base.lease(output / ".publication.lock", blocking=True):
            data = write_report(output, seeds, args.out)
            plot_report(output, data)
        return
    if args.root is None:
        parser.error("--root is required")
    root = args.root.resolve()
    check_paths(root, output)
    if args.mode == "checkpoints":
        if not inspect_checkpoints(root, output, seeds):
            raise SystemExit(2)
        return
    if args.mode == "plan":
        for seed in seeds:
            plan = make_plan(root, seed)
            print(json.dumps({key: plan[key] for key in ("seed", "switch_step", "end_step", "checkpoint",
                              "training_estimate", "d_checks_through_switch")}, indent=2))
        return
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}; saved checkpoints and shard receipts are retained")
    signal.signal(signal.SIGTERM, interrupted)
    if worker(root, output, seeds, os.environ.get("CUDA_VISIBLE_DEVICES", "").split(","),
              args.hours, args.idle_minutes) is False:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
