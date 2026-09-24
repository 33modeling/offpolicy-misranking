#!/usr/bin/env python3
"""Finish evaluation of two saved Pair policies in a separate output root.

Never trains, rewrites a recovery plan, publishes canonical Pair completion,
or writes anywhere in the source experiment. All checkpoints remain read-only.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import shutil
import statistics
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
REPO = Path(os.environ.get("PAIR_FINISH_REPO", Path(__file__).resolve().parents[1])).resolve()
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "src")]
import mbpp_budget_recovery as saved
import evidence_downstream as ed

base, core, switch = saved.base, saved.core, saved.switch
HERE = Path(__file__).resolve()
SCHEMA = "selector-pair-saved-final-evaluation/v1"
TARGETS = {1: (50, "selection_reduced"), 4: (100, "random_full")}


def source_directory(root, seed):
    step, arm = TARGETS[seed]
    return root / f"branches/on_policy/states/s{seed}-t{step}/points/view-{step}" / arm


def checked_output(root, output):
    root, output = root.resolve(), output.absolute()
    # Reject aliases as well as lexical nesting before creating anything.
    if any(p.is_symlink() for p in [output, *output.parents]):
        raise ValueError("output path must not contain symlinks")
    output = output.resolve()
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("output must be separate from the original Pair root")
    if output.exists() and any(p.is_symlink() for p in output.rglob("*")):
        raise ValueError("refusing symlinks in the new evaluation output")
    return output


@contextlib.contextmanager
def source_guard(directory):
    # Shared leases use existing lock files opened read-only, without creating,
    # truncating, deleting, or rewriting them. Active writers remain excluded.
    with contextlib.ExitStack() as stack:
        fds = []
        for name in (".task.lock", ".cost.lock"):
            path = directory / name
            if path.is_file():
                handle = stack.enter_context(path.open("rb"))
                fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
                fds.append(handle.fileno())
        yield fds


def input_hashes(directory, points):
    out = directory.parent
    paths = [out / "contract.json", out / "evaluation.json", out.parent.parent / "net_protocol.json",
             directory / "decision.json", directory / "execution.json",
             out / "subsets" / f"subset-{directory.name}.json"]
    for name in ("cost.jsonl", "curve/cost.jsonl", "budget-recovery/cost.jsonl",
                 "budget-recovery/plan.json", "budget-recovery/result.json",
                 "budget-recovery/result.sha256.json"):
        if (directory / name).is_file():
            paths.append(directory / name)
    for point in points:
        paths.extend(Path(point["adapter"]) / name for name in point["hashes"])
    paths.extend(directory / "policy" / name for name in saved.FILES)
    return {str(path.resolve()): base.digest(path) for path in paths}


def runtime_hashes():
    names = ("scripts/mbpp_budget_recovery.py", "src/train_policy_grpo.py",
             "src/selection_gate_gpu.py", "src/selection_gate.py", "src/selection_switch_gpu.py",
             "src/additive_experiment.py", "src/evidence_downstream.py", "src/rollout.py",
             "src/rollout_contract.py", "src/data.py", "src/artifact_contract.py")
    return {name: base.digest(REPO / name) for name in names}


def make_plan(root, seed):
    directory = source_directory(root, seed)
    out, arm = directory.parent, directory.name
    c = base.verify(out)  # read-only; do not call runtime/manifest migrations
    start = TARGETS[seed][0]
    if (c["config"]["seed"], c["config"]["drift"]) != (seed, start):
        raise ValueError("source is not the requested frozen Pair branch")
    protocol = core.read(out.parent.parent / "net_protocol.json")
    decision = core.read(directory / "decision.json")
    if decision["binding"] != {"protocol_sha256": core.fingerprint(protocol),
                               "contract_sha256": base.digest(out / "contract.json")}:
        raise ValueError("original decision binding changed")
    for path in (directory / "execution.json", out / "subsets" / f"subset-{arm}.json"):
        if core.read(path.with_suffix(".sha256.json")) != {"sha256": base.digest(path)}:
            raise ValueError(f"original sealed input changed: {path}")
    expected = saved.checkpoint_contract(out, c, arm)
    policy = directory / "policy"
    if not (policy / "policy_train.json").is_file():
        raise ValueError("no saved final policy; no parent restart or new training")
    completed, evidence = saved.validate_candidate(policy, out, c, arm, expected)
    branch = out.parents[3]
    curve = core.read(branch / "switch.json").get("curve")
    points = []
    if curve:
        count = core.integer(curve["points"], "curve points", 1)
        k = core.integer(curve["k"], "curve responses", 1)
        archived = {}
        for path in (policy / "curve-checkpoints").glob("step-*"):
            state = core.read(path / "checkpoint_state.json")
            step = state.get("completed_steps")
            if (type(step) is not int or path.name != f"step-{step}"
                    or any(state.get(key) != value for key, value in expected.items())
                    or base.digest(path / "adapter_model.safetensors") != state.get("adapter_sha256")):
                raise ValueError(f"archived curve checkpoint changed: {path}")
            archived[step] = path
        selected = switch.curve_steps(start, completed, switch.curve_fractions(count), archived)
        if not selected:
            raise ValueError("saved curve checkpoints unavailable; no invented curve")
        for step in [start, *selected]:
            adapter = Path(expected["resume_adapter"]) if step == start else archived[step]
            manifest = "policy_train.json" if step == start else "checkpoint_state.json"
            points.append(saved.point(adapter, step, k, c["eval_seed"] + 7919 * (step + 1), manifest))
    points.append(saved.point(policy, completed, c["eval_k"], c["eval_seed"], evidence, final=True))
    original_costs = {name: base.cost(directory / name) for name in (".", "curve", "budget-recovery")
                      if (directory / name / "cost.jsonl").is_file()}
    return {"schema": SCHEMA, "source_root": str(root), "directory": str(directory),
            "seed": seed, "start_step": start, "completed_steps": completed, "arm": arm,
            "canonical_complete": False, "purpose": "evaluate_existing_final_policy_without_training",
            "runner_sha256": base.digest(HERE), "runtime_hashes": runtime_hashes(),
            "inputs": input_hashes(directory, points), "points": points,
            "original_costs": original_costs,
            "eval_timeout": core.read(out.parent.parent / "suite.json")["eval_timeout"]}


def verify_plan(plan):
    if plan["schema"] != SCHEMA or plan["runner_sha256"] != base.digest(HERE):
        raise ValueError("new evaluation runtime changed; existing files preserved")
    if plan["runtime_hashes"] != runtime_hashes():
        raise ValueError("evaluation dependencies changed")
    for name, digest in plan["inputs"].items():
        if base.digest(Path(name)) != digest:
            raise ValueError(f"source input changed: {name}")


def shard_info(target, plan, index, shard):
    item = plan["points"][index]
    c = core.read(Path(plan["directory"]).parent / "contract.json")
    n = len(c["evaluation"]["val"])
    indices = range(n * shard // 4, n * (shard + 1) // 4)
    dest = target / f"point-{index}"
    binding = {"plan_sha256": base.digest(target / "plan.json"), "point": index, "shard": shard}
    return c, item, indices, dest, binding


def checked_shard(target, plan, index, shard):
    _, item, indices, dest, binding = shard_info(target, plan, index, shard)
    path = dest / f"shard-{shard}.jsonl"
    if core.read(path.with_suffix(".done.json")) != {"binding": binding, "sha256": base.digest(path)}:
        raise ValueError("new evaluation shard binding changed")
    return ed.reward_rows(path, indices, item["k"])


def reuse_shard(target, plan, index, shard):
    _, item, indices, dest, binding = shard_info(target, plan, index, shard)
    output = dest / f"shard-{shard}.jsonl"
    if output.with_suffix(".done.json").exists():
        checked_shard(target, plan, index, shard)
        return True
    directory = Path(plan["directory"])
    out = directory.parent
    common = {"experiment_sha256": base.digest(out / "contract.json"),
              "adapter_sha256": item["hashes"]["adapter_model.safetensors"], "shard": shard}
    if item["final"]:
        source = directory / "evaluation"
        expected = {**common, "arm": directory.name,
                    "policy_manifest_sha256": item["hashes"]["policy_train.json"]}
    else:
        source = switch.curve_point_dir(out, directory.name, item["step"], plan["start_step"])
        expected = {**common, "arm": "parent" if item["step"] == plan["start_step"] else directory.name,
                    "step": item["step"], "k": item["k"]}
    candidates = [(source, expected)]
    old_path = directory / "budget-recovery/plan.json"
    if old_path.is_file():
        old = core.read(old_path)
        if old.get("inputs", {}).get(str((out / "contract.json").resolve())) == base.digest(out / "contract.json"):
            for old_index, point in enumerate(old.get("points", [])):
                if all(point.get(key) == item[key] for key in ("step", "k", "seed", "hashes")):
                    candidates.append((old_path.parent / f"point-{old_index}",
                                       {"plan_sha256": base.digest(old_path), "point": old_index, "shard": shard}))
    for source, expected in candidates:
        path = source / f"shard-{shard}.jsonl"
        done = path.with_suffix(".done.json")
        if not done.is_file():
            continue
        seal = core.read(done)
        if seal.get("binding") != expected:
            continue  # an evaluation of a different policy/step is not reused
        if seal != {"binding": expected, "sha256": base.digest(path)}:
            raise ValueError(f"sealed source evaluation changed: {path}")
        ed.reward_rows(path, indices, item["k"])
        dest.mkdir(parents=True, exist_ok=True)
        if output.exists():
            if base.digest(output) != seal["sha256"]:
                raise ValueError("partial new evaluation differs from saved measurement")
        else:
            shutil.copyfile(path, output)
        if base.digest(output) != seal["sha256"]:
            raise ValueError("source evaluation changed while copied")
        base.bind(output.with_suffix(".done.json"), {"binding": binding, "sha256": seal["sha256"]})
        return True
    return False


def evaluate(root, output, seed, index, shard):
    output = checked_output(root, output)
    target = output / f"seed-{seed}"
    plan = core.read(target / "plan.json")
    verify_plan(plan)
    c, item, indices, dest, binding = shard_info(target, plan, index, shard)
    with base.lease(dest / f"shard-{shard}.lock"):
        path = dest / f"shard-{shard}.jsonl"
        if path.with_suffix(".done.json").is_file():
            checked_shard(target, plan, index, shard)
            return
        base.bind(dest / f"shard-{shard}.contract.json", binding)
        from rollout import collect_rollouts, load_policy
        model, tokenizer = load_policy(c["config"]["model"], Path(item["adapter"]))
        collect_rollouts(model, tokenizer, c["evaluation"]["val"][indices.start:indices.stop],
                         item["k"], c["config"]["max_new_tokens"], float(c["config"]["temperature"]),
                         path, idx_offset=indices.start, sampling_seed_base=item["seed"])
        ed.reward_rows(path, indices, item["k"])
        verify_plan(plan)
        base.bind(path.with_suffix(".done.json"), {"binding": binding, "sha256": base.digest(path)})


@contextlib.contextmanager
def retain_locks(fds):
    original = base.subprocess
    inherited = list(fds)
    for fd in (7, 8):
        try:
            os.fstat(fd)
            inherited.append(fd)
        except OSError:
            pass
    base.subprocess = SimpleNamespace(Popen=lambda *a, **kw: original.Popen(
        *a, **kw, pass_fds=tuple(set(inherited))), STDOUT=original.STDOUT)
    try:
        yield
    finally:
        base.subprocess = original


def finish(root, output, seed, devices):
    output = checked_output(root, output)
    target = output / f"seed-{seed}"
    directory = source_directory(root, seed)
    if target.exists() and any(target.iterdir()) and not (target / "plan.json").is_file():
        if {p.name for p in target.iterdir()} != {".finish.lock"}:
            raise ValueError("output directory contains unrelated or incomplete files")
    with source_guard(directory) as source_fds:
        target.mkdir(parents=True, exist_ok=True)
        with (target / ".finish.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return finish_locked(root, target, seed, devices, source_fds + [lock.fileno()])


def finish_locked(root, target, seed, devices, fds):
    plan = make_plan(root, seed)
    base.bind(target / "plan.json", plan)
    verify_plan(plan)
    directory = Path(plan["directory"])
    c = core.read(directory.parent / "contract.json")
    import additive_experiment as ae
    env = {**ae.model_environment(c["config"]), "PAIR_FINISH_REPO": str(REPO),
           "PYTHONDONTWRITEBYTECODE": "1"}
    result_path = target / "result.json"
    if result_path.is_file():
        result = core.read(result_path)
        if (core.read(result_path.with_suffix(".sha256.json")) != {"sha256": base.digest(result_path)}
                or result.get("plan_sha256") != base.digest(target / "plan.json")):
            raise ValueError("saved final-evaluation result changed")
        return result
    points = []
    for index, item in enumerate(plan["points"]):
        commands = []
        for shard in range(4):
            if not reuse_shard(target, plan, index, shard):
                commands.append(([sys.executable, "-B", str(HERE), "shard", "--root", str(root),
                                  "--output", str(target.parent), "--seed", str(seed),
                                  "--point", str(index), "--shard", str(shard)], devices[shard]))
        if commands:
            print(f"[evaluate] seed={seed} step={item['step']} missing_shards={len(commands)}", flush=True)
            with retain_locks(fds):
                base.meter(target, f"evaluate-{index}", c["scope"]["gpu_type"], commands=commands,
                           env=env, timeout=plan["eval_timeout"], ledger="reporting")
        rewards = {str(i): [] for i in range(len(c["evaluation"]["val"]))}
        for shard in range(4):
            for row in checked_shard(target, plan, index, shard):
                rewards[str(row["prompt_idx"])].append(row["reward"])
        points.append({"step": item["step"], "k": item["k"], "final": item["final"],
                       "rewards": {k: statistics.fmean(v) for k, v in rewards.items()}})
    verify_plan(plan)
    result = {"schema": SCHEMA, "plan_sha256": base.digest(target / "plan.json"),
              "evaluation_complete": True, "canonical_complete": False, "training_performed": False,
              "seed": seed, "start_step": plan["start_step"], "completed_steps": plan["completed_steps"],
              "points": points, "original_costs": plan["original_costs"], "new_evaluation_cost": base.cost(target)}
    base.bind(result_path, result)
    base.bind(result_path.with_suffix(".sha256.json"), {"sha256": base.digest(result_path)})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "run", "shard", "results"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, choices=TARGETS)
    parser.add_argument("--point", type=int)
    parser.add_argument("--shard", type=int, choices=range(4))
    args = parser.parse_args()
    root = args.root.resolve()
    output = checked_output(root, args.output)
    if args.mode == "shard":
        if args.seed is None or args.point is None or args.point < 0 or args.shard is None:
            parser.error("shard needs --seed, --point and --shard")
        evaluate(root, output, args.seed, args.point, args.shard)
        return
    failed = False
    for seed in ([args.seed] if args.seed else TARGETS):
        try:
            if args.mode == "plan":
                with source_guard(source_directory(root, seed)):
                    result = make_plan(root, seed)
            elif args.mode == "results":
                path = output / f"seed-{seed}/result.json"
                result = core.read(path)
                if core.read(path.with_suffix(".sha256.json")) != {"sha256": base.digest(path)}:
                    raise ValueError("result seal changed")
            else:
                devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
                if len(devices) != 4 or len(set(devices)) != 4 or any(not d for d in devices):
                    raise ValueError("four distinct allocated GPUs required")
                result = finish(root, output, seed, devices)
            print(json.dumps(result, indent=2), flush=True)
        except BlockingIOError:
            print(f"[busy] seed {seed}: another worker owns this task; nothing stopped", flush=True)
            failed = True
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            print(f"[failed] seed {seed}: {exc}", file=sys.stderr, flush=True)
            failed = True
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
