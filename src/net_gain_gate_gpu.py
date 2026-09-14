"""Isolated v3 fixed-budget study and actual held-out three-arm gate test."""

from __future__ import annotations

import argparse
import copy
import json
import os
import socket
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

import net_gain_gate as net
import selection_gate as core
import selection_gate_gpu as base
import selection_gate_study as study
from light_selection_gate_gpu import install_signal_handlers

HERE = Path(__file__).resolve()
TEST_ARMS = ("random_full", "selection_full", "gated")
SELECTORS = ("difficulty", "low_order", "pair_u2")
CODE_FILES = ("src/net_gain_gate.py", "src/net_gain_gate_gpu.py", "src/selection_gate.py",
              "src/selection_gate_study.py", "src/selection_gate_gpu.py", "src/selection_gate_budget.py",
              "src/train_selection_gate_grpo.py", "src/train_policy_grpo.py", "src/low_order_experiment.py",
              "src/low_order_backend.py", "src/low_order_reuse.py", "src/evidence_downstream.py")


def identity(c):
    prefix = f"policy_step_{c['config']['drift']}"
    return (f"{c['scope']['model']}:seed-{c['config']['seed']}",
            (c["source_hashes"][f"{prefix}/adapter_model.safetensors"],
             c["source_hashes"][f"{prefix}/optimizer.pt"]))


def check_model(model, c):
    trajectory, parent = identity(c)
    net.check_scope(model, c["scope"], c["budget_gpu_seconds"], trajectory=trajectory,
                    parent=parent, role=c["role"], observed=True)


def protocol(root):
    value = core.read(root / "net_protocol.json")
    if value.get("schema") != net.SCHEMA or value.get("schedule") != net.SCHEDULE:
        raise ValueError("not a v3 net-gain suite")
    arms = list(study.BRANCHES) if value["mode"] == "study" else list(TEST_ARMS)
    if value["mode"] not in {"study", "test"} or value["arms"] != arms or value["selector"] not in SELECTORS:
        raise ValueError("invalid v3 experimental design")
    if value["mode"] == "test":
        net.validate_model(value["model"])
        if value["role"] != "test" or value["model"]["data_kind"] != "observed":
            raise ValueError("actual gate testing needs an observed frozen model and held-out trajectories")
    elif value["model"] is not None or value["role"] != "development":
        raise ValueError("study collects development labels; it does not run a fitted gate")
    core.number(value["max_measurement_fraction"], "measurement fraction", 1e-12, .1)
    core.integer(value["recent_window"], "recent window", 1)
    for name, sha in value.get("code_hashes", {}).items():
        if name not in CODE_FILES or base.digest(base.ROOT / name) != sha:
            raise ValueError(f"frozen experiment code changed: {name}; preserve the original code for this suite")
    return value


def prepare(args):
    mode = args.mode
    model = net.validate_model(core.read(args.model)) if args.model else None
    if (mode == "test") != (model is not None):
        raise ValueError("--mode test requires --model; study must not use a fitted model")
    role = "test" if mode == "test" else "development"
    if model:
        if model["data_kind"] != "observed":
            raise ValueError("synthetic models cannot control GPU training")
        if args.budget_gpu_seconds is None:
            args.budget_gpu_seconds = model["budget_gpu_seconds"]
        if model["scope"]["selector"] != args.selector:
            raise ValueError("specify the same --selector as the frozen model")
    core.number(args.max_measurement_fraction, "measurement fraction", 1e-12, .1)
    core.integer(args.recent_window, "recent window", 1)
    config = net.measurement_config({"recent_window": args.recent_window,
        "max_measurement_fraction": args.max_measurement_fraction, "measurement_wall_seconds": args.measurement_wall_seconds})
    if model and model["measurement_config"] != config:
        raise ValueError("measurement settings must match the frozen model")
    delegated = copy.copy(args)
    delegated.mode, delegated.model, delegated.role = "study", None, role
    if args.drifts:
        if args.runs or args.matrix is None or len(set(args.drifts)) != len(args.drifts):
            raise ValueError("--drifts requires a matrix, unique steps, and no --runs")
        import additive_experiment as ae
        delegated.runs = [run for step in args.drifts for run in ae.resolve_runs(args.matrix, args.seeds, step)]
    base.prepare(delegated)
    with base.lease(args.root / ".net-prepare.lock", blocking=True):
        for out in base.entries(args.root):
            c = core.read(out / "contract.json")
            if model:
                check_model(model, c)
            run = Path(c["source_run"])
            inputs = {name: base.digest(run / name) for name in (
                "rollouts_behavior_train.jsonl", f"policy_step_{c['config']['drift']}/grpo_stats.jsonl")}
            base.bind(out / "net_inputs.json", inputs)
        base.bind(args.root / "net_protocol.json", {
            "schema": net.SCHEMA, "mode": mode, "model": model, "role": role,
            "selector": args.selector, "schedule": net.SCHEDULE,
            "arms": list(study.BRANCHES if mode == "study" else TEST_ARMS),
            "recent_window": args.recent_window, "max_measurement_fraction": args.max_measurement_fraction,
            "code_hashes": {name: base.digest(base.ROOT / name) for name in CODE_FILES},
            "historical_cost": "existing cache and parent are shared inputs; historical costs are unknown, not zero",
            "preparation_cost": "source hashing and suite preparation are offline research, outside branch caps",
            "evaluation_cost": "independent identical test questions, separate reporting allocation",
            "horizon": "one fixed-budget continuation; no optimal switch step or periodic remeasurement",
        })
    protocol(args.root)


def measurement_worker(out, arm, *, window, wall_cap, scoring_only=False):
    from artifact_contract import validate_generation_contract
    c = core.read(out / "contract.json")
    run, step = Path(c["source_run"]), c["config"]["drift"]
    validate_generation_contract(run, ("rollouts_behavior_train",))
    report = net.measure(run / "rollouts_behavior_train.jsonl",
                         stats=None if scoring_only else run / f"policy_step_{step}/grpo_stats.jsonl",
                         step=0 if scoring_only else step,
                         prompts=c["n"], responses=8, seed=c["config"]["seed"],
                         window=window, wall_cap=wall_cap)
    expected = core.read(out / "net_inputs.json")
    if report["source_sha256"] != expected["rollouts_behavior_train.jsonl"] or (not scoring_only and report["stats_sha256"] != expected[f"policy_step_{step}/grpo_stats.jsonl"]):
        raise ValueError("pre-decision cache or training statistics changed")
    if arm == "gate_measurement":
        p = protocol(out.parent.parent)
        check_model(p["model"], c)
        # The fitted rule is evaluated within the charged diagnostic subprocess.
        report["choice"] = net.choose(p["model"], report["features"])
    base.bind(out / arm / "measurement.json", report)


def measure_once(out, suite, p, directory, env):
    """A failed attempt is charged and never retried. A lock never occupies a waiting node."""
    c = core.read(out / "contract.json")
    binding = {"contract_sha256": base.digest(out / "contract.json"),
               "inputs_sha256": base.digest(out / "net_inputs.json"), "protocol_sha256": core.fingerprint(p)}
    with base.lease(directory / ".measurement.lock"):
        base.spent(directory)
        path = directory / "initial.json"
        if path.exists():
            value = core.read(path)
            if value["binding"] != binding or value["gpu_seconds"] != base.spent(directory):
                raise ValueError("measurement binding or ledger changed")
            if value["report_sha256"] is not None and value["report_sha256"] != base.digest(directory / "measurement.json"):
                raise ValueError("measurement payload changed")
            return value
        marker = directory / "attempt.json"
        if not marker.exists():
            base.bind(marker, binding)
            cap = min(suite["measurement_wall_seconds"], c["budget_gpu_seconds"]*p["max_measurement_fraction"]/4)
            command = [sys.executable, str(HERE), "worker", "--root", str(out), "--arm", directory.name,
                       "--phase", "measure", "--recent-window", str(p["recent_window"]),
                       "--measurement-wall-seconds", str(cap)]
            try:
                base.meter(directory, "diagnose", c["scope"]["gpu_type"], commands=[(command, "")],
                           env=env, timeout=cap, ledger="research" if p["mode"] == "study" else "deployment")
            except Exception as exc:
                base.bind(directory / "measurement-failure.json", {"error": str(exc)})
        elif core.read(marker) != binding:
            raise ValueError("measurement attempt belongs to another protocol")
        measured = base.spent(directory)
        payload = directory / "measurement.json"
        events_path = directory / "cost.jsonl"
        finished = [json.loads(line) for line in events_path.read_text().splitlines()
                    if line.strip() and json.loads(line).get("state") == "finished"] if events_path.exists() else []
        ok = (payload.exists() and not (directory / "measurement-failure.json").exists()
              and len(finished) == 1 and finished[0]["exit_code"] == 0)
        value = {"binding": binding, "gpu_seconds": measured,
                 "report_sha256": base.digest(payload) if ok else None,
                 "status": "complete" if ok else "failed_no_retry"}
        base.bind(path, value)
        return value


def decision(out, suite, p, arm, env):
    c = core.read(out / "contract.json")
    directory = out / arm
    path = directory / "decision.json"
    binding = {"protocol_sha256": core.fingerprint(p), "contract_sha256": base.digest(out / "contract.json")}
    if path.exists():
        value = core.read(path)
        if value["binding"] != binding:
            raise ValueError("frozen decision binding changed")
        if arm in {"random_reduced", "selection_reduced", "gated"}:
            measured_dir = out / ("measurement" if p["mode"] == "study" else "gate_measurement")
            first = measure_once(out, suite, p, measured_dir, env)
            if value["measurement_gpu_seconds"] != first["gpu_seconds"] or value["profile_sha256"] != first["report_sha256"]:
                raise ValueError("frozen decision measurement changed")
        return value
    action = "select" if arm.startswith("selection_") else "random"
    value = {"binding": binding, "action": action, "reason": "control_arm", "profile_sha256": None,
             "measurement_gpu_seconds": 0., "budget_gpu_seconds": c["budget_gpu_seconds"],
             "start_step": c["config"]["drift"], "schedule": net.SCHEDULE}
    if arm in {"random_reduced", "selection_reduced", "gated"}:
        measured_dir = out / ("measurement" if p["mode"] == "study" else "gate_measurement")
        first = measure_once(out, suite, p, measured_dir, env)
        value.update(measurement_gpu_seconds=first["gpu_seconds"], profile_sha256=first["report_sha256"])
        # Charge shared measurement to each counterfactual, but execute it only once.
        value["budget_gpu_seconds"] -= first["gpu_seconds"]
        if arm == "gated":
            check_model(p["model"], c)
            value.update(core.read(measured_dir / "measurement.json")["choice"]
                         if first["status"] == "complete" else
                         {"action": "random", "reason": "measurement_failed_no_retry", "prediction": None})
        elif first["status"] != "complete":
            raise ValueError("failed measurement cannot become a development label")
    if value["budget_gpu_seconds"] <= 0:
        raise ValueError("diagnosis exhausted the branch allocation")
    base.bind(path, value)
    return value


def select_once(out, c, p, arm, choice, env, devices):
    directory = out / arm
    if p["selector"] == "difficulty":
        if choice["profile_sha256"]:
            measured_dir = out / ("measurement" if p["mode"] == "study" else "gate_measurement")
            path = measured_dir / "measurement.json"
        else:
            path = directory / "measurement.json"
            if not path.exists():
                command = [sys.executable, str(HERE), "worker", "--root", str(out), "--arm", arm,
                           "--phase", "score", "--recent-window", str(p["recent_window"])]
                base.meter(directory, "difficulty-score", c["scope"]["gpu_type"], commands=[(command, "")],
                           env=env, timeout=min(30., (choice["budget_gpu_seconds"]-base.spent(directory))/4),
                           ledger="deployment")
        if choice["profile_sha256"] and base.digest(path) != choice["profile_sha256"]:
            raise ValueError("diagnostic-selected indices changed")
        return core.read(path)["difficulty_indices"]
    # Do not share another arm's paid gradient scores as a free cache.
    private = out / "selector-work" / arm
    base.bind(private / "inputs/test.json", {"test": c["evaluation"]["val"], "provenance": c["evaluation"]["provenance"]})
    return base.select_once(private, c, directory, choice["budget_gpu_seconds"], env, devices, ledger="deployment")


def run_arm(out, suite, p, arm, devices, env):
    directory = out / arm
    c = core.read(out / "contract.json")
    if (directory / "result.json").exists():
        validate_result(out, p, arm)
        return
    base.spent(directory)
    base.meter(directory, "verify-inputs", c["scope"]["gpu_type"], action=lambda: base.verify(out), ledger="deployment")
    choice = decision(out, suite, p, arm, env)
    cap = choice["budget_gpu_seconds"]
    subset = out / "subsets" / f"subset-{arm}.json"
    execution = directory / "execution.json"
    if execution.exists():
        if core.read(directory / "execution.sha256.json") != {"sha256": base.digest(execution)}:
            raise ValueError("frozen execution changed")
        actual = core.read(execution)
    else:
        indices, actual = None, {"action": choice["action"], "reason": choice["reason"]}
        if choice["action"] == "select":
            try:
                indices = select_once(out, c, p, arm, choice, env, devices)
            except Exception as exc:
                if arm != "gated":
                    raise
                base.spent(directory)
                actual = {"action": "random", "reason": "selector_failed", "error": str(exc)}
        actual["indices"] = indices
        base.bind(execution, actual)
        base.bind(directory / "execution.sha256.json", {"sha256": base.digest(execution)})
    if not subset.exists():
        base.meter(directory, "freeze-subset", c["scope"]["gpu_type"],
                   action=lambda: base.freeze_subset(out, c, arm, actual["indices"]), ledger="deployment")
    if core.read(subset.with_suffix(".sha256.json")) != {"sha256": base.digest(subset)}:
        raise ValueError("fixed training subset changed")
    stop_path = directory / "policy/budget_stop.json"
    if not stop_path.exists():
        remaining = cap-base.spent(directory)
        if remaining <= 0:
            raise ValueError("branch allocation exhausted before a valid checkpoint")
        if remaining/4 <= 30:
            base.bind(stop_path, {"completed_steps": c["config"]["drift"], "stop_reason": "no_block_fits",
                                  "use_parent_policy": True, "requested_target_steps": c["config"]["drift"]+c["max_steps"]})
        else:
            base.meter(directory, "train", c["scope"]["gpu_type"],
                       commands=[(base.train_command(out, c, arm, remaining), ",".join(devices))],
                       env=env, timeout=remaining/4, ledger="deployment")
    stop = core.read(stop_path)
    if stop["stop_reason"] not in {"budget_exhausted", "no_block_fits"}:
        raise ValueError("update-count termination is not a fixed-budget result")
    base.policy(out, c, arm)
    commands = [([sys.executable, str(HERE), "worker", "--root", str(out), "--phase", "evaluate",
                  "--arm", arm, "--shard", str(i)], devices[i]) for i in range(4)
                if not (directory / "evaluation" / f"shard-{i}.done.json").exists()]
    if commands:
        base.meter(directory, "evaluate", c["scope"]["gpu_type"], commands=commands, env=env,
                   timeout=suite["eval_timeout"], ledger="reporting")
    used = base.spent(directory)
    if used > cap:
        raise ValueError("branch exceeded its fixed allocation; do not count as DONE")
    result = {"schema": net.SCHEMA, "binding": choice["binding"], "complete": True,
              "rewards": base.rewards(out, c, arm), "budget_gpu_seconds": cap, "used_gpu_seconds": used,
              "measurement_gpu_seconds": choice["measurement_gpu_seconds"], "cost": base.cost(directory),
              "action": actual["action"], "completed_steps": stop["completed_steps"], "stop_reason": stop["stop_reason"],
              "artifact_hashes": {str(path.relative_to(out)): base.digest(path) for path in
                  (out / "contract.json", directory / "decision.json", execution, subset, stop_path)}}
    base.bind(directory / "result.json", result)
    base.bind(directory / "result.sha256.json", {"sha256": base.digest(directory / "result.json")})
    (directory / "failure.json").unlink(missing_ok=True)


def validate_result(out, p, arm):
    directory = out / arm
    result = core.read(directory / "result.json")
    if core.read(directory / "result.sha256.json") != {"sha256": base.digest(directory / "result.json")}:
        raise ValueError("result hash changed")
    c = core.read(out / "contract.json")
    if result["binding"] != {"protocol_sha256": core.fingerprint(p), "contract_sha256": base.digest(out / "contract.json")}:
        raise ValueError("result protocol changed")
    expected_paths = {str(path.relative_to(out)) for path in
                      (out / "contract.json", directory / "decision.json", directory / "execution.json",
                       out / "subsets" / f"subset-{arm}.json", directory / "policy/budget_stop.json")}
    if set(result["artifact_hashes"]) != expected_paths:
        raise ValueError("missing result artifact binding")
    for name, sha in result["artifact_hashes"].items():
        if base.digest(out / name) != sha:
            raise ValueError(f"result artifact changed: {name}")
    choice = core.read(directory / "decision.json")
    if arm in {"random_reduced", "selection_reduced", "gated"}:
        measured_dir = out / ("measurement" if p["mode"] == "study" else "gate_measurement")
        first = core.read(measured_dir / "initial.json")
        if first["gpu_seconds"] != base.spent(measured_dir) or first["gpu_seconds"] != choice["measurement_gpu_seconds"]:
            raise ValueError("diagnostic ledger changed")
        if first["report_sha256"] != choice["profile_sha256"] or (first["report_sha256"] and first["report_sha256"] != base.digest(measured_dir / "measurement.json")):
            raise ValueError("diagnostic evidence changed")
    if result["measurement_gpu_seconds"] != choice["measurement_gpu_seconds"] or result["budget_gpu_seconds"] != c["budget_gpu_seconds"]-choice["measurement_gpu_seconds"]:
        raise ValueError("diagnostic cost or remaining budget changed")
    if result["used_gpu_seconds"] != base.spent(directory) or result["cost"] != base.cost(directory):
        raise ValueError("cost ledger changed")
    core.number(result["used_gpu_seconds"], "branch cost", 0, result["budget_gpu_seconds"])
    if not result["complete"] or result["stop_reason"] not in {"budget_exhausted", "no_block_fits"}:
        raise ValueError("not a complete fixed-budget continuation")
    study.reward_mean(result)
    return result


def work(root):
    import additive_experiment as ae
    p, suite = protocol(root), core.read(root / "suite.json")
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(devices) != 4 or len(set(devices)) != 4 or not all(devices) or os.environ.get("OM_NODE_LOCK_HELD") != "1":
        raise ValueError("an admitted four-GPU node is required")
    hardware = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", ",".join(devices)],
                                       text=True, timeout=20).strip().splitlines()
    failures = 0
    for out in base.entries(root):
        c = core.read(out / "contract.json")
        if len(hardware) != 4 or set(map(str.strip, hardware)) != {c["scope"]["gpu_type"]}:
            raise ValueError("allocated hardware differs from frozen contract")
        for arm in p["arms"] if c["config"]["seed"] % 2 == 0 else p["arms"][::-1]:
            try:
                with base.lease(out / arm / ".task.lock"):
                    run_arm(out, suite, p, arm, devices, ae.model_environment(c["config"]))
            except BlockingIOError:
                continue
            except Exception as exc:
                failures += 1
                traceback.print_exc()
                core.atomic_json(out / arm / "failure.json", {"error": str(exc), "host": socket.gethostname(), "time": time.time()})
                print(f"[failed] {out.name}/{arm}: {exc}; continuing other tasks", flush=True)
    status(root)
    return int(bool(failures))


def status(root):
    if not (root / "net_protocol.json").exists():
        print("[v3 net gate] not prepared")
        return
    p = protocol(root)
    done, total = 0, 0
    print("POINT        ARM                  STATE        ACTION   UPDATES   GPU-s / CAP    REWARD")
    for out in base.entries(root):
        c = core.read(out / "contract.json")
        for arm in p["arms"]:
            total += 1
            directory = out / arm
            label = f"s{c['config']['seed']}/d{c['config']['drift']}"
            try:
                if (directory / "result.json").exists():
                    r = validate_result(out, p, arm)
                    done += 1
                    print(f"{label:12} {arm:20} DONE         {r['action']:8} {r['completed_steps']-c['config']['drift']:7}   "
                          f"{r['used_gpu_seconds']:.0f}/{r['budget_gpu_seconds']:.0f}       {study.reward_mean(r):.4f}")
                else:
                    progress = core.read(directory / "progress.json") if (directory / "progress.json").exists() else {}
                    measured_dir = out / ("measurement" if p["mode"] == "study" else "gate_measurement")
                    if arm in {"random_reduced", "selection_reduced", "gated"} and not (directory / "decision.json").exists() and (measured_dir / "progress.json").exists():
                        progress = core.read(measured_dir / "progress.json")
                    live = progress.get("state") == "running" and time.time()-progress.get("updated", 0) < 60
                    failure = core.read(directory / "failure.json")["error"] if (directory / "failure.json").exists() else ""
                    state = "RUNNING" if live else "FAILED" if failure else "INCOMPLETE" if progress else "QUEUED"
                    print(f"{label:12} {arm:20} {state:12} {progress.get('phase', '')} {failure}")
            except (ValueError, KeyError, OSError, TypeError) as exc:
                print(f"{label:12} {arm:20} INVALID: {exc}")
    print(f"[v3 net gate] {done}/{total} DONE; diagnosis charged separately against each applicable cap")


def summarize(root):
    p, rows, excluded = protocol(root), [], []
    for out in base.entries(root):
        try:
            c = base.verify(out)
            trajectory, parent = identity(c)
            results = {arm: validate_result(out, p, arm) for arm in p["arms"]}
            for arm, result in results.items():
                if result["rewards"] != base.rewards(out, c, arm):
                    raise ValueError("evaluation artifacts differ from published rewards")
            if len({tuple(sorted(r["rewards"])) for r in results.values()}) != 1:
                raise ValueError("test question identities differ across arms")
            if p["mode"] == "study":
                first = core.read(out / "measurement/initial.json")
                path = out / "measurement/measurement.json"
                if first["status"] != "complete" or first["gpu_seconds"] != base.spent(out / "measurement") or first["report_sha256"] != base.digest(path):
                    raise ValueError("measurement evidence changed")
                profile = core.read(path)
                lineage = {"parent_sha256": parent[0], "optimizer_sha256": parent[1],
                           "pool_sha256": c["scope"]["pool_sha256"], "evaluation_sha256": core.fingerprint(c["evaluation"]),
                           "learner_config_sha256": core.fingerprint(c["config"]), "gpu_type": c["scope"]["gpu_type"]}
                rows.append({"id": out.name, "trajectory_id": trajectory, "role": c["role"], "step": c["config"]["drift"],
                             "feature_step": profile["feature_step"], "scope": c["scope"], "full_pool_coverage": True,
                             "decision_schedule": "once_before_training", "features": profile["features"],
                             "budget_gpu_seconds": c["budget_gpu_seconds"], "measurement_gpu_seconds": first["gpu_seconds"],
                             "branches": {a: {**r, **lineage} for a, r in results.items()}})
            else:
                check_model(p["model"], c)
                means = {arm: study.reward_mean(r) for arm, r in results.items()}
                rows.append({"id": out.name, "trajectory_id": trajectory, "step": c["config"]["drift"], "means": means,
                             "gate_minus_random": means["gated"]-means["random_full"],
                             "gate_minus_always_select": means["gated"]-means["selection_full"],
                             "selection_minus_random": means["selection_full"]-means["random_full"],
                             "action": results["gated"]["action"], "branches": results})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            excluded.append({"point": out.name, "reason": str(exc)})
    result = {"schema": net.SCHEMA, "data_kind": "observed", "excluded": excluded,
              "complete": not excluded and bool(rows), "protocol_sha256": core.fingerprint(p)}
    if p["mode"] == "study":
        suite = core.read(root / "suite.json")
        result.update(label_protocol="v3_measured", points=rows, measurement_config={
            "recent_window": p["recent_window"], "max_measurement_fraction": p["max_measurement_fraction"],
            "measurement_wall_seconds": suite["measurement_wall_seconds"]})
        if rows:
            net.validate_study(result)
        name = "study.json"
    else:
        ids = sorted({r["trajectory_id"] for r in rows})
        result.update(rows=rows, trajectories=len(ids), model_id=p["model"]["model_id"],
                      evaluation_kind="actual held-out gated training", certified=False,
                      mean_gate_minus_random=statistics.fmean(statistics.fmean(r["gate_minus_random"] for r in rows
                          if r["trajectory_id"] == name) for name in ids) if ids else None,
                      optimal_switch_step=None)
        name = "net_results.json"
    core.atomic_json(root / name, result)
    print(f"[v3] {root/name}: {len(rows)} complete points, {len(excluded)} excluded")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "worker", "status", "summarize"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--matrix", type=Path)
    parser.add_argument("--runs", type=Path, nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--drift", type=int, default=100)
    parser.add_argument("--drifts", type=int, nargs="+", help="joint development checkpoints with a common compute cap")
    parser.add_argument("--mode", choices=("study", "test"), default="study")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--selector", choices=SELECTORS, default="low_order")
    parser.add_argument("--gpu-type", default="NVIDIA H100 80GB HBM3")
    parser.add_argument("--budget-gpu-seconds", type=float)
    parser.add_argument("--equivalent-steps", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--measurement-wall-seconds", type=float, default=30.)
    parser.add_argument("--max-measurement-fraction", type=float, default=.01)
    parser.add_argument("--recent-window", type=int, default=20)
    parser.add_argument("--eval-prompts", type=Path)
    parser.add_argument("--pool", type=Path)
    parser.add_argument("--pool-manifest", type=Path)
    parser.add_argument("--eval-k", type=int, default=8)
    parser.add_argument("--test-count", type=int, default=300)
    parser.add_argument("--eval-timeout", type=float, default=14400.)
    parser.add_argument("--phase", choices=("measure", "score", "evaluate"))
    parser.add_argument("--arm", choices=(*study.BRANCHES, *TEST_ARMS, "measurement", "gate_measurement"))
    parser.add_argument("--shard", type=int, choices=range(4))
    args = parser.parse_args()
    if args.command == "prepare":
        if not args.runs and args.matrix is None:
            parser.error("prepare requires --runs or --matrix")
        prepare(args)
    elif args.command == "run":
        return work(args.root)
    elif args.command == "worker":
        if not args.arm or not args.phase or os.environ.get("OM_NODE_LOCK_HELD") != "1":
            parser.error("worker requires an admitted node, --arm and --phase")
        if args.phase in {"measure", "score"}:
            measurement_worker(args.root, args.arm, window=args.recent_window, wall_cap=args.measurement_wall_seconds,
                               scoring_only=args.phase == "score")
        else:
            if args.shard is None:
                parser.error("evaluation requires --shard")
            base.evaluate(args.root, args.arm, args.shard)
    elif args.command == "status":
        status(args.root)
    else:
        summarize(args.root)
    return 0


if __name__ == "__main__":
    install_signal_handlers()
    raise SystemExit(main())
