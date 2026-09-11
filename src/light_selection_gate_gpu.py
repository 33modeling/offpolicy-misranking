"""Two-arm, equal-allocation test of the v2 one-shot screening hypothesis.

Uses the existing isolated budgeted trainer; original E5/Qwen workers are untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

import light_selection_gate as light
import selection_gate as core
import selection_gate_gpu as base

HERE = Path(__file__).resolve()
ARMS = ("random_full", "gated")


def install_signal_handlers():
    def interrupted(signum, frame):
        # Unwind meter's finally block so child process groups and costs close.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)


def prepare(args):
    rule = light.validate_rule(core.read(args.rule)) if args.rule else light.default_rule()
    if args.role == "test" and rule["status"] != "development_frozen":
        raise ValueError("held-out testing requires a development-frozen rule; default runs are exploratory")
    args.mode, args.model, args.selector = "study", None, "passrate_beta"
    base.prepare(args)
    for out in base.entries(args.root):
        c = core.read(out / "contract.json")
        trajectory = f"{c['scope']['model']}:seed-{c['config']['seed']}"
        if args.role == "test" and trajectory in rule["development_trajectory_ids"]:
            raise ValueError("test continuation shares a development trajectory")
    with base.lease(args.root / ".light-prepare.lock", blocking=True):
        base.bind(args.root / "light_protocol.json", {
            "schema": light.SCHEMA, "rule": rule, "arms": list(ARMS),
            "role": args.role, "schedule": "once_before_training",
            "historical_cache_cost": "shared existing input; unknown historical cost is not zero",
            "evaluation_cost": "separate reporting allocation, equal question IDs and response count",
            "learner": "fresh-response GRPO on one fixed 10% prompt subset",
        })


def protocol(root):
    value = core.read(root / "light_protocol.json")
    if value["schema"] != light.SCHEMA or value["arms"] != list(ARMS):
        raise ValueError("not a v2 light-gate suite")
    light.validate_rule(value["rule"])
    return value


def measurement_worker(out):
    from artifact_contract import validate_generation_contract
    from select_rules import topk_count
    c = core.read(out / "contract.json")
    run = Path(c["source_run"])
    validate_generation_contract(run, ("rollouts_behavior_train",))
    report = light.measure(run / "rollouts_behavior_train.jsonl", prompts=c["n"], responses=8,
                           k=topk_count(c["n"], c["config"]["topk_frac"]),
                           seed=c["config"]["seed"], cache_step=0, target_step=c["config"]["drift"])
    base.bind(out / "gated/measurement.json", report)


def initial(out, suite, rule, devices, env):
    directory = out / "gated"
    path = directory / "decision.json"
    if path.exists():
        value = core.read(path)
        if value["rule_sha256"] != core.fingerprint(rule):
            raise ValueError("frozen decision uses a different rule")
        return value
    c = core.read(out / "contract.json")
    base.spent(directory)
    marker = directory / "measurement-attempt.json"
    report_path = directory / "measurement.json"
    if not marker.exists():
        base.bind(marker, {"schedule": "once_before_training", "rule_sha256": core.fingerprint(rule)})
        remaining = c["budget_gpu_seconds"]-base.spent(directory)
        cap = min(suite["measurement_wall_seconds"],
                  c["budget_gpu_seconds"]*rule["max_measurement_fraction"]/base.GPUS,
                  remaining/base.GPUS)
        try:
            if cap <= 0:
                raise TimeoutError("no measurement budget remains")
            command = [sys.executable, str(HERE), "worker", "--root", str(out), "--phase", "measure"]
            base.meter(directory, "measurement", c["scope"]["gpu_type"],
                       commands=[(command, "")], env=env, timeout=cap, ledger="deployment")
        except Exception as exc:
            base.bind(directory / "measurement-failure.json", {"error": str(exc)})
    # On restart, a completed payload is reused. A failed attempt never remeasures.
    cost_path = directory / "cost.jsonl"
    measured = sum(row["allocated_gpu_seconds"] for row in
                   (json.loads(v) for v in (cost_path.read_text().splitlines() if cost_path.exists() else []))
                   if row["state"] == "finished" and row["phase"] == "measurement")
    base.spent(directory)
    if report_path.exists() and not (directory / "measurement-failure.json").exists():
        report = core.read(report_path)
        decision = light.choose(report, rule, measured_gpu_seconds=measured,
                                budget_gpu_seconds=c["budget_gpu_seconds"])
        decision["report_sha256"] = base.digest(report_path)
        decision["selected_indices"] = report["selected_indices"] if decision["action"] == "select" else None
    else:
        decision = {"action": "random", "reasons": ["measurement_failed_or_interrupted"],
                    "measurement_gpu_seconds": measured, "selected_indices": None,
                    "rule_sha256": core.fingerprint(rule), "schedule": "once_before_training",
                    "claim": rule["claim"]}
    base.bind(path, decision)
    return decision


def run_arm(out, suite, rule, arm, devices, env):
    directory = out / arm
    c = core.read(out / "contract.json")
    result_path = directory / "result.json"
    if result_path.exists():
        validate_result(out, arm)
        return
    base.spent(directory)
    base.meter(directory, "verify-inputs", c["scope"]["gpu_type"], action=lambda: base.verify(out), ledger="deployment")
    decision = (initial(out, suite, rule, devices, env) if arm == "gated" else
                {"action": "random", "selected_indices": None, "measurement_gpu_seconds": 0.,
                 "rule_sha256": core.fingerprint(rule), "schedule": "once_before_training"})
    if arm == "random_full":
        base.bind(directory / "decision.json", decision)
    subset = out / "subsets" / f"subset-{arm}.json"
    if not subset.exists():
        base.meter(directory, "freeze-subset", c["scope"]["gpu_type"],
                   action=lambda: base.freeze_subset(out, c, arm, decision["selected_indices"]), ledger="deployment")
    if core.read(subset.with_suffix(".sha256.json")) != {"sha256": base.digest(subset)}:
        raise ValueError("frozen training subset changed")
    stop_path = directory / "policy/budget_stop.json"
    if not stop_path.exists():
        remaining = c["budget_gpu_seconds"]-base.spent(directory)
        if remaining/base.GPUS <= 30:
            base.bind(stop_path, {"completed_steps": c["config"]["drift"], "stop_reason": "no_block_fits",
                                  "use_parent_policy": True, "requested_target_steps": c["config"]["drift"]+c["max_steps"]})
        else:
            base.meter(directory, "train", c["scope"]["gpu_type"],
                       commands=[(base.train_command(out, c, arm, remaining), ",".join(devices))],
                       env=env, timeout=remaining/base.GPUS, ledger="deployment")
    base.policy(out, c, arm)
    commands = [([sys.executable, str(HERE), "worker", "--root", str(out), "--phase", "evaluate",
                  "--arm", arm, "--shard", str(i)], devices[i]) for i in range(base.GPUS)
                if not (directory / "evaluation" / f"shard-{i}.done.json").exists()]
    if commands:
        base.meter(directory, "evaluate", c["scope"]["gpu_type"], commands=commands,
                   env=env, timeout=suite["eval_timeout"], ledger="reporting")
    used = base.spent(directory)
    stop = core.read(stop_path)
    result = {"schema": light.SCHEMA, "contract_sha256": base.digest(out / "contract.json"),
              "decision_sha256": base.digest(directory / "decision.json"),
              "subset_sha256": base.digest(subset), "rewards": base.rewards(out, c, arm),
              "used_gpu_seconds": used, "budget_gpu_seconds": c["budget_gpu_seconds"],
              "matched_budget": used <= c["budget_gpu_seconds"], "complete": True,
              "action": decision["action"], "completed_steps": stop["completed_steps"],
              "stop_reason": stop["stop_reason"], "cost": base.cost(directory)}
    base.bind(result_path, result)
    base.bind(directory / "result.sha256.json", {"sha256": base.digest(result_path)})
    (directory / "failure.json").unlink(missing_ok=True)


def validate_result(out, arm):
    directory = out / arm
    value = core.read(directory / "result.json")
    if core.read(directory / "result.sha256.json") != {"sha256": base.digest(directory / "result.json")}:
        raise ValueError("result hash changed")
    for field, path in (("contract_sha256", out / "contract.json"),
                         ("decision_sha256", directory / "decision.json"),
                         ("subset_sha256", out / "subsets" / f"subset-{arm}.json")):
        if value[field] != base.digest(path):
            raise ValueError(f"result binding changed: {field}")
    if value["used_gpu_seconds"] != base.spent(directory):
        raise ValueError("cost ledger changed after result")
    c = core.read(out / "contract.json")
    if value["budget_gpu_seconds"] != c["budget_gpu_seconds"] or value["matched_budget"] != (value["used_gpu_seconds"] <= c["budget_gpu_seconds"]):
        raise ValueError("reported budget differs from frozen allocation")
    if value["complete"] is not True or not value["rewards"]:
        raise ValueError("incomplete result")
    for reward in value["rewards"].values():
        core.number(reward, "test reward", 0, 1)
    return value


def work(root):
    import additive_experiment as ae
    p, suite = protocol(root), core.read(root / "suite.json")
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(devices) != 4 or len(set(devices)) != 4 or not all(devices) or os.environ.get("OM_NODE_LOCK_HELD") != "1":
        raise ValueError("an admitted four-GPU node is required")
    hardware = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader",
                                        "-i", ",".join(devices)], text=True, timeout=20).strip().splitlines()
    failures = 0
    for out in base.entries(root):
        c = core.read(out / "contract.json")
        if len(hardware) != 4 or set(map(str.strip, hardware)) != {c["scope"]["gpu_type"]}:
            raise ValueError("allocated hardware differs from frozen contract")
        # Alternate execution order by seed to reduce systematic time/order effects.
        for arm in ARMS if c["config"]["seed"] % 2 == 0 else ARMS[::-1]:
            try:
                with base.lease(out / arm / ".task.lock"):
                    run_arm(out, suite, p["rule"], arm, devices, ae.model_environment(c["config"]))
            except BlockingIOError:
                continue
            except Exception as exc:
                failures += 1
                traceback.print_exc()
                core.atomic_json(out / arm / "failure.json", {"error": str(exc), "host": socket.gethostname(), "time": time.time()})
                print(f"[failed] {out.name}/{arm}: {exc}; continuing other tasks", flush=True)
    status(root)
    return bool(failures)


def status(root):
    if not (root / "light_protocol.json").exists():
        print("[light gate] not prepared")
        return
    protocol(root)
    print("POINT                       ARM          STATE        ACTION   UPDATES   GPU-s / B")
    for out in base.entries(root):
        for arm in ARMS:
            directory = out / arm
            try:
                if (directory / "result.json").exists():
                    r = validate_result(out, arm)
                    state = "DONE" if r["matched_budget"] else "OVER-BUDGET"
                    c = core.read(out / "contract.json")
                    print(f"{out.name:27} {arm:12} {state:12} {r['action']:8} {r['completed_steps']-c['config']['drift']:7}   {r['used_gpu_seconds']:.0f}/{r['budget_gpu_seconds']:.0f}")
                elif (directory / "failure.json").exists():
                    print(f"{out.name:27} {arm:12} FAILED: {core.read(directory / 'failure.json')['error']}")
                elif (directory / "progress.json").exists():
                    r = core.read(directory / "progress.json")
                    age = time.time()-r["updated"]
                    state = "STALE" if age > 60 else r["state"]
                    print(f"{out.name:27} {arm:12} {state}: {r['phase']} {r['seconds']:.0f}s; heartbeat {age:.0f}s ago")
                else:
                    print(f"{out.name:27} {arm:12} QUEUED")
            except (ValueError, KeyError, OSError) as exc:
                print(f"{out.name:27} {arm:12} INVALID: {exc}")


def summarize(root):
    p = protocol(root)
    points, excluded = [], []
    for out in base.entries(root):
        try:
            c = base.verify(out)
            r, g = [validate_result(out, arm) for arm in ARMS]
            if not all(v["matched_budget"] for v in (r, g)):
                raise ValueError("budget overrun; cannot claim equal-allocation result")
            if set(r["rewards"]) != set(g["rewards"]) or len(r["rewards"]) != len(c["evaluation"]["val"]):
                raise ValueError("independent test question coverage differs")
            # Validate adapters, manifests and response hashes before reporting outcomes.
            for arm, value in zip(ARMS, (r, g)):
                if core.read(out / arm / "decision.json")["rule_sha256"] != core.fingerprint(p["rule"]):
                    raise ValueError("decision rule differs from the frozen suite")
                if base.rewards(out, c, arm) != value["rewards"]:
                    raise ValueError("evaluation rewards changed")
            points.append({"point": out.name, "seed": c["config"]["seed"], "drift": c["config"]["drift"],
                           "trajectory_id": f"{c['scope']['model']}:seed-{c['config']['seed']}",
                           "paired_reward_difference": statistics.fmean(g["rewards"].values())-statistics.fmean(r["rewards"].values()),
                           "gate_action": g["action"], "random_gpu_seconds": r["used_gpu_seconds"],
                           "gated_gpu_seconds": g["used_gpu_seconds"], "budget_gpu_seconds": r["budget_gpu_seconds"]})
        except (OSError, ValueError, KeyError) as exc:
            excluded.append({"point": out.name, "reason": str(exc)})
    ids = [row["trajectory_id"] for row in points]
    result = {"schema": light.SCHEMA, "role": p["role"], "rule": p["rule"], "points": points,
              "excluded": excluded, "independent_trajectories": len(set(ids)),
              "mean_paired_reward_difference": statistics.fmean(v["paired_reward_difference"] for v in points) if points else None,
              "uncertainty": "report seed-paired outcomes; shared trajectories are not independent replicates",
              "certified_improvement": False}
    core.atomic_json(root / "light_results.json", result)
    print(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("prepare", "run", "worker", "status", "summarize"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--matrix", type=Path)
    p.add_argument("--runs", type=Path, nargs="+")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    p.add_argument("--drift", type=int, default=100)
    p.add_argument("--rule", type=Path)
    p.add_argument("--role", choices=("development", "test"), default="development")
    p.add_argument("--gpu-type", default="NVIDIA H100 80GB HBM3")
    p.add_argument("--budget-gpu-seconds", type=float)
    p.add_argument("--equivalent-steps", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=100000)
    p.add_argument("--measurement-wall-seconds", type=float, default=30.)
    p.add_argument("--eval-prompts", type=Path)
    p.add_argument("--pool", type=Path)
    p.add_argument("--pool-manifest", type=Path)
    p.add_argument("--eval-k", type=int, default=8)
    p.add_argument("--test-count", type=int, default=300)
    p.add_argument("--eval-timeout", type=float, default=14400.)
    p.add_argument("--phase", choices=("measure", "evaluate"))
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--shard", type=int, choices=range(4))
    args = p.parse_args()
    args.root = args.root.resolve()
    if args.command == "prepare": prepare(args)
    elif args.command == "run": return work(args.root)
    elif args.command == "status": status(args.root)
    elif args.command == "summarize": summarize(args.root)
    elif args.phase == "measure": measurement_worker(args.root)
    elif (args.phase == "evaluate" and args.arm and args.shard is not None
          and os.environ.get("OM_NODE_LOCK_HELD") == "1"):
        base.evaluate(args.root, args.arm, args.shard)
    else: p.error("worker needs --phase; evaluation needs --arm/--shard and admitted GPUs")
    return 0


if __name__ == "__main__":
    install_signal_handlers()
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[light gate] interrupted; active child processes reaped", file=sys.stderr)
        raise SystemExit(130)
