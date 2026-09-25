"""Evaluate the three Pair t=25 controls at the common step 275 (added 2026-09-25).

Figure 2 of the manuscript ends at step 275, where only the Switch arm has a
measurement; the Random, On-policy and SR controls were last evaluated at
steps 255-270. This job evaluates each control's saved step-275 checkpoint on
the same 300 held-out questions, eight responses each, with the same sampling
recipe the Switch experiment used at step 275. Everything is read from the
existing Switch root (plans, checkpoints, receipts) and written only under a
new output root. Nothing is trained and no source file is modified.

    python scripts/selector_pair_step275_eval.py plan|run|status|results \
        --switch-root SWITCH_ROOT --output NEW_ROOT
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import signal
import statistics
import sys
import time
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
import selection_gate as core  # noqa: E402
import selection_gate_gpu as base  # noqa: E402
import selector_pair_switch_rewards as sw  # noqa: E402

SCHEMA = "offpolicy-selector-pair/common-step-control-evaluation-v1"
STEP = 275
CONTROLS = ("random", "on_policy", "cached")
SEEDS = (3, 4)


def switch_dir(switch_root, seed):
    directory = switch_root / f"s{seed}"
    if not (directory / "plan.json").is_file():
        raise ValueError(f"no Switch plan for seed {seed}: {directory}")
    return directory


def load_plan(switch_root, seed):
    plan = core.read(switch_dir(switch_root, seed) / "plan.json")
    if plan["seed"] != seed or plan["switch_step"] >= STEP or plan["end_step"] < STEP:
        raise ValueError(f"seed {seed}: step {STEP} is not inside the Switch comparison window")
    return plan


def control_checkpoint(switch_root, plan, arm):
    """The control's saved full checkpoint at STEP, validated exactly as the Switch export validates points."""
    path = sw.point_adapter(switch_dir(switch_root, plan["seed"]), plan, arm, STEP)
    if not path:
        raise ValueError(f"seed {plan['seed']} {arm}: no saved checkpoint at step {STEP}")
    return path


def binding(switch_root, plan, arm, shard):
    adapter, indices, value = sw.evaluation_binding(switch_dir(switch_root, plan["seed"]), plan, arm, STEP, shard)
    return adapter, indices, {**value, "schema": SCHEMA, "switch_plan": str(switch_dir(switch_root, plan["seed"]) / "plan.json")}


def target(output, seed, arm):
    return output / f"s{seed}/evaluations/{arm}/step-{STEP}"


def evaluate_shard(switch_root, output, seed, arm, shard):
    import evidence_downstream as ed
    plan = load_plan(switch_root, seed)
    adapter, indices, value = binding(switch_root, plan, arm, shard)
    directory = target(output, seed, arm)
    with base.lease(directory / f"shard-{shard}.lock"):
        base.bind(directory / f"shard-{shard}.contract.json", value)
        path = directory / f"shard-{shard}.jsonl"
        done = path.with_suffix(".done.json")
        if not done.exists():
            from rollout import collect_rollouts, load_policy
            model, tokenizer = load_policy(plan["config"]["model"], adapter)
            # Same held-out prompts, response count and per-step sampling seed as the Switch step-275 points.
            collect_rollouts(model, tokenizer, plan["contract"]["evaluation"]["val"][indices.start:indices.stop],
                             value["k"], plan["config"]["max_new_tokens"], float(plan["config"]["temperature"]),
                             path, idx_offset=indices.start,
                             sampling_seed_base=plan["contract"]["eval_seed"] + 7919 * (STEP + 1))
            ed.reward_rows(path, indices, value["k"])
            base.bind(done, {"binding": value, "sha256": base.digest(path)})
        if core.read(done) != {"binding": value, "sha256": base.digest(path)}:
            raise ValueError(f"evaluation receipt changed: {done}")
        ed.reward_rows(path, indices, value["k"])


def measured(switch_root, output, plan, arm):
    import evidence_downstream as ed
    directory = target(output, plan["seed"], arm)
    if not all((directory / f"shard-{shard}.done.json").is_file() for shard in range(4)):
        return None
    rewards = []
    for shard in range(4):
        _, indices, value = binding(switch_root, plan, arm, shard)
        path = directory / f"shard-{shard}.jsonl"
        if core.read(path.with_suffix(".done.json")) != {"binding": value, "sha256": base.digest(path)}:
            raise ValueError(f"evaluation hash mismatch: {path}")
        rewards.extend(row["reward"] for row in ed.reward_rows(path, indices, value["k"]))
    return {"step": STEP, "reward": statistics.fmean(rewards), "k": plan["contract"]["eval_k"],
            "question_count": len(plan["contract"]["evaluation"]["val"]), "source": str(directory)}


def switch_point(switch_root, plan):
    """The Switch arm's own step-275 measurement, read from the Switch root."""
    return sw.measured_point(switch_dir(switch_root, plan["seed"]), plan, "switch", STEP)


def report(switch_root, output, seeds):
    rows, missing = [], []
    for seed in seeds:
        plan = load_plan(switch_root, seed)
        for arm in CONTROLS:
            point = measured(switch_root, output, plan, arm)
            (rows if point else missing).append({"seed": seed, "arm": arm, **(point or {})})
        point = switch_point(switch_root, plan)
        (rows if point else missing).append({"seed": seed, "arm": "switch", **(point or {}), "source_kind": "switch_root"})
    return {"schema": SCHEMA, "step": STEP, "complete": not missing, "rows": rows, "missing": missing,
            "scope": "Fresh step-275 evaluations of the saved control checkpoints on the held-out questions; "
                     "the Switch value is the existing step-275 measurement. Reward is a fraction; missing is not zero.",
            "switch_root": str(switch_root), "output": str(output)}


def write_report(switch_root, output, seeds, out=None):
    data = report(switch_root, output, seeds)
    lines = [f"COMMON-STEP CONTROL EVALUATION AT STEP {STEP}", f"Switch root: {switch_root}", f"Output: {output}",
             "Overall: " + ("COMPLETE" if data["complete"] else "INCOMPLETE"),
             "seed,arm,step,reward_percent,k,questions"]
    for row in sorted(data["rows"], key=lambda r: (r["seed"], CONTROLS.index(r["arm"]) if r["arm"] in CONTROLS else 9)):
        lines.append(f"{row['seed']},{row['arm']},{row['step']},{100 * row['reward']:.3f},{row['k']},{row['question_count']}")
    for row in data["missing"]:
        lines.append(f"{row['seed']},{row['arm']},{STEP},missing,,")
    text = "\n".join(lines) + "\n\nJSON\n" + json.dumps(data, ensure_ascii=True, allow_nan=False) + "\n"
    output.mkdir(parents=True, exist_ok=True)
    (output / f"step{STEP}-controls.txt").write_text(text)
    (output / f"step{STEP}-controls.json").write_text(json.dumps(data, indent=1, allow_nan=False) + "\n")
    destination = out or Path.home() / f"selector-pair-step{STEP}-results.txt"
    destination.write_text(text)
    print(text, end="")
    print(f"[saved] {destination}")
    return data


def plan_text(switch_root, seeds):
    for seed in seeds:
        plan = load_plan(switch_root, seed)
        for arm in CONTROLS:
            print(f"seed {seed} {arm}: checkpoint {control_checkpoint(switch_root, plan, arm)}")
        point = switch_point(switch_root, plan)
        print(f"seed {seed} switch: step-{STEP} measurement {'present' if point else 'MISSING'}")


def worker(switch_root, output, seeds, devices, hours, idle_minutes):
    import selector_pair_gpu as pair
    if len(devices) != 4 or len(set(devices)) != 4:
        raise ValueError("worker needs four distinct allocated GPUs")
    deadline, idle_since = time.monotonic() + hours * 3600, time.monotonic()
    overrides = pair.admission_probe(output)
    nccl_env = {"NCCL_DEBUG": os.environ.get("NCCL_DEBUG", "WARN"), **overrides}
    plans = {seed: load_plan(switch_root, seed) for seed in seeds}
    for seed, plan in plans.items():
        for arm in CONTROLS:
            control_checkpoint(switch_root, plan, arm)  # refuse early if a checkpoint is missing
    while time.monotonic() < deadline:
        worked = False
        for seed, plan in plans.items():
            for arm in CONTROLS:
                if measured(switch_root, output, plan, arm) is not None:
                    continue
                directory = target(output, seed, arm)
                with contextlib.ExitStack() as stack:
                    try:
                        lock_fd = stack.enter_context(sw.task_lease(directory / ".task.lock"))
                    except BlockingIOError:
                        continue
                    if measured(switch_root, output, plan, arm) is not None:
                        continue
                    commands = [([sys.executable, str(Path(__file__).resolve()), "evaluate-shard",
                                  "--switch-root", str(switch_root), "--output", str(output), "--seed", str(seed),
                                  "--arm", arm, "--shard", str(shard)], devices[shard])
                                for shard in range(4) if not (directory / f"shard-{shard}.done.json").exists()]
                    print(f"[step{STEP}] seed={seed} evaluate {arm} at step {STEP} ({len(commands)} shards)", flush=True)
                    location = output / f"s{seed}/attempts/eval-{arm}-{STEP}-{uuid.uuid4().hex}"
                    with sw.inherited_task_lock(lock_fd):
                        base.meter(location, f"s{seed}-eval-{arm}-{STEP}", plan["contract"]["scope"]["gpu_type"],
                                   commands=commands, env={**pair.environment(plan["contract"]), **nccl_env},
                                   timeout=max(1, deadline - time.monotonic()), devices=4, ledger="reporting")
                    measured(switch_root, output, plan, arm)
                    worked = True
                    break
            if worked:
                break
        if worked:
            idle_since = time.monotonic()
            continue
        data = report(switch_root, output, seeds)
        if data["complete"]:
            with base.lease(output / ".publication.lock", blocking=True):
                write_report(switch_root, output, seeds)
            return True
        if time.monotonic() - idle_since >= idle_minutes * 60:
            print(f"[step{STEP}] no unclaimed work; worker exits without changing peer jobs.", flush=True)
            return
        print(f"[step{STEP}] remaining work held by peers; rechecking in 30s", flush=True)
        time.sleep(min(30, max(0, deadline - time.monotonic())))
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "run", "status", "results", "evaluate-shard"))
    parser.add_argument("--switch-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, action="append")
    parser.add_argument("--arm", choices=CONTROLS)
    parser.add_argument("--shard", type=int, choices=range(4))
    parser.add_argument("--hours", type=float, default=24)
    parser.add_argument("--idle-minutes", type=float, default=60)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    switch_root, output = args.switch_root.resolve(), args.output.resolve()
    if output == switch_root or output in switch_root.parents or switch_root in output.parents:
        raise SystemExit("[abort] the output root must be separate from the Switch root")
    seeds = args.seed or list(SEEDS)
    if args.mode == "evaluate-shard":
        if len(seeds) != 1 or args.arm is None or args.shard is None:
            parser.error("evaluate-shard needs one seed, an arm and a shard")
        evaluate_shard(switch_root, output, seeds[0], args.arm, args.shard)
        return
    if args.mode == "plan":
        plan_text(switch_root, seeds)
        return
    if args.mode in ("status", "results"):
        if args.mode == "results":
            with base.lease(output / ".publication.lock", blocking=True):
                write_report(switch_root, output, seeds, args.out)
        else:
            data = report(switch_root, output, seeds)
            print(json.dumps({"complete": data["complete"], "measured": [(r["seed"], r["arm"]) for r in data["rows"]],
                              "missing": [(r["seed"], r["arm"]) for r in data["missing"]]}, indent=1))
        return

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}; finished shard receipts are retained")
    signal.signal(signal.SIGTERM, interrupted)
    if worker(switch_root, output, seeds, os.environ.get("CUDA_VISIBLE_DEVICES", "").split(","),
              args.hours, args.idle_minutes) is False:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
