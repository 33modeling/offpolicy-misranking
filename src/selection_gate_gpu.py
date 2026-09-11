"""One-shot gate continuations, isolated from the original experiment launchers.

Study mode collects three matched-parent counterfactuals. Deployment runs only
the initially chosen arm. Neither mode remeasures the gate during training.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import random
import signal
import socket
import statistics
import subprocess
import sys
import time
import traceback
import uuid
from dataclasses import asdict
from pathlib import Path

import selection_gate as gate
import selection_gate_study as study

SCHEMA = "offpolicy-selection-gate-gpu/one-shot-v1"
HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
GPUS = 4


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


@contextlib.contextmanager
def lease(path, *, blocking=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield


def bind(path, value):
    if path.exists():
        if gate.read(path) != value:
            raise ValueError(f"frozen contract changed: {path}")
    else:
        gate.atomic_json(path, value)


def journal(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def cost(directory):
    path = directory / "cost.jsonl"
    events = [json.loads(v) for v in path.read_text().splitlines() if v.strip()] if path.exists() else []
    return gate.cost_summary(events)


def spent(directory):
    result = cost(directory)
    if not result["complete"]:
        raise ValueError(f"unclosed cost event at {directory}; unknown cost cannot be treated as zero")
    return sum(v["gpu_seconds"] for k, v in result["ledgers"].items() if k != "reporting")


def terminate(processes):
    for p in processes:
        if p.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(p.pid, signal.SIGTERM)
    deadline = time.monotonic() + 5
    for p in processes:
        try:
            p.wait(timeout=max(.01, deadline-time.monotonic()))
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(p.pid, signal.SIGKILL)
            p.wait()


def meter(directory, name, gpu_type, *, action=None, commands=None, env=None,
          timeout=None, ledger="research", devices=GPUS):
    """One allocation interval, including idle GPUs while a CPU phase runs."""
    directory.mkdir(parents=True, exist_ok=True)
    base = {"event_id": uuid.uuid4().hex, "phase": name, "ledger": ledger,
            "gpus": devices, "gpu_type": gpu_type, "host": socket.gethostname()}
    path = directory / "cost.jsonl"
    journal(path, {**base, "state": "started", "time": time.time()})
    started, rc, processes = time.monotonic(), 1, []
    def progress(elapsed, state):
        gate.atomic_json(directory / "progress.json", {**base, "state": state, "pid": os.getpid(),
                         "updated": time.time(), "seconds": elapsed, "timeout": timeout})
    progress(0., "running")
    try:
        if action is not None:
            result = action()
        else:
            if not commands or timeout is None or timeout <= 0:
                raise ValueError("GPU phases need commands and a positive finite timeout")
            gate.number(timeout, "phase timeout", 1e-12)
            with contextlib.ExitStack() as stack:
                for i, (command, visible) in enumerate(commands):
                    log = stack.enter_context((directory / f"{name}-{i}.log").open("a"))
                    worker_env = {**os.environ, **(env or {}), "CUDA_VISIBLE_DEVICES": visible}
                    p = subprocess.Popen(command, env=worker_env, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
                    processes.append(p)
                heartbeat = 0.
                while True:
                    codes = [p.poll() for p in processes]
                    elapsed = time.monotonic()-started
                    if any(v is not None and v != 0 for v in codes):
                        raise RuntimeError(f"{name} worker failed: {codes}; see {directory}/{name}-*.log")
                    if all(v == 0 for v in codes):
                        break
                    if elapsed >= timeout:
                        raise TimeoutError(f"{name} exceeded {timeout:.0f}s allocation limit")
                    if elapsed >= heartbeat:
                        progress(elapsed, "running")
                        print(f"[gate] {directory.name} {name}: {elapsed:.0f}s / {timeout:.0f}s", flush=True)
                        heartbeat = elapsed + 15
                    time.sleep(min(.25, max(.01, timeout-elapsed)))
                result = None
        rc = 0
        return result
    finally:
        terminate(processes)
        seconds = time.monotonic()-started
        journal(path, {**base, "state": "finished", "time": time.time(), "seconds": seconds,
                       "allocated_gpu_seconds": seconds*devices, "exit_code": rc})
        progress(seconds, "finished" if rc == 0 else "failed")


def source_contract(run, evaluation, *, budget, gpu_type, role, selector, eval_k, max_steps):
    import evidence_downstream as ed
    from train_policy_grpo import validate_policy_manifest

    c = gate.read(run / "run_config.json")
    if (c.get("dataset") != "math500" or c.get("prompt_format") != "olmo_rlzero_math"
            or c.get("grpo_world_size") != 4 or c.get("grpo_epochs_per_batch") != 1
            or c.get("behavior_k") != 8 or c.get("grpo_group_size") != 8
            or c.get("topk_frac") != .1 or c.get("temperature") != 1.
            or c.get("top_p", 1.) != 1. or c.get("drift", 0) <= 0):
        raise ValueError("initial gate protocol requires positive-drift OLMo MATH, four ranks, K=G=8, top 10%, one epoch")
    parent = run / f"policy_step_{c['drift']}"
    manifest = validate_policy_manifest(parent, target_steps=c["drift"], world_size=4,
                                       training_objective="grpo", require_complete_hashes=True)
    if manifest["seed"] != c["seed"] or manifest["prompt_format"] != c["prompt_format"]:
        raise ValueError("parent policy differs from source seed/prompt format")
    prompts = gate.read(run / "prompts.json")
    ed.questions(prompts["train"])
    test = ed.independent_test(prompts, evaluation)
    if len(test) < 4:
        raise ValueError("independent test needs at least four questions")
    ed.train_args(c, run, Path("unused"), "random_full", max_steps)
    hashes = {name: digest(run / name) for name in ["run_config.json", "prompts.json"]
              + [f"policy_step_{c['drift']}/{f}" for f in ed.POLICY_FILES]}
    model_hash = digest(Path(c["model"]) / "config.json")
    scope = {"model": model_hash, "dataset": "math500", "selector": selector,
             "verifier": "math_verify", "pool_sha256": hashes["prompts.json"], "gpu_type": gpu_type}
    return {"schema": SCHEMA, "source_run": str(run), "config": c, "source_hashes": hashes,
            "scope": scope, "role": role, "budget_gpu_seconds": budget, "max_steps": max_steps,
            "evaluation": {"val": test, "provenance": evaluation["provenance"]}, "eval_k": eval_k,
            "eval_seed": 701_000_003 + c["seed"]*1_000_003,
            "n": len(prompts["train"]), "decision_schedule": "once_before_training"}


def prepare(args):
    import additive_experiment as ae
    import evidence_downstream as ed

    runs = [p.resolve() for p in (args.runs or ae.resolve_runs(args.matrix, args.seeds, args.drift))]
    root = ed.require_separate_output(args.root, runs)
    if not runs or len({p.name for p in runs}) != len(runs) or any(root in p.parents for p in runs):
        raise ValueError("need unique source points, separate from the output tree")
    model = gate.validate_model(gate.read(args.model)) if args.model else None
    if model and (args.mode != "deploy" or model["data_kind"] != "observed"):
        raise ValueError("only an observed-data gate may control GPU deployment")
    durations = []
    if args.budget_gpu_seconds is None and model is None:
        for run in runs:
            c = gate.read(run / "run_config.json")
            for line in (run / f"policy_step_{c['drift']}" / "grpo_stats.jsonl").read_text().splitlines():
                if line.strip():
                    durations.append(gate.number(json.loads(line)["step_seconds"], "source step duration", 1e-12))
        if not durations:
            raise ValueError("no source timing; provide --budget-gpu-seconds")
    budget = (args.budget_gpu_seconds if args.budget_gpu_seconds is not None else
              model["budget_gpu_seconds"] if model else
              math.ceil(statistics.median(durations)*GPUS*args.equivalent_steps/60)*60)
    gate.number(budget, "total GPU seconds", 1e-12)
    for v, name in ((args.max_steps, "max steps"), (args.equivalent_steps, "equivalent steps"), (args.eval_k, "evaluation K")):
        gate.integer(v, name, 1)
    gate.number(args.measurement_wall_seconds, "measurement wall cap", 1e-12)
    gate.number(args.eval_timeout, "evaluation timeout", 1e-12)
    if args.eval_prompts:
        evaluation = gate.read(args.eval_prompts)
    else:
        if not args.pool or not args.pool_manifest:
            raise ValueError("need independent test file or math pool and manifest")
        evaluation = ed.prepare_test(args.pool, runs, root / "inputs/test.json", args.test_count,
                                     20260912, "EleutherAI/hendrycks_math",
                                     gate.read(args.pool_manifest)["source_revision"], "train")
    contracts = [source_contract(run, evaluation, budget=budget, gpu_type=args.gpu_type,
                                role=args.role, selector=args.selector, eval_k=args.eval_k,
                                max_steps=args.max_steps) for run in runs]
    for c in contracts:
        if model and (not gate.compatible_scope(model["scope"], c["scope"])
                      or model["budget_gpu_seconds"] != budget):
            raise ValueError("gate model scope or budget differs from deployment")
    with lease(root / ".prepare.lock", blocking=True):
        entries = []
        for run, c in zip(runs, contracts, strict=True):
            out = root / "points" / run.name
            bind(out / "contract.json", c)
            bind(out / "evaluation.json", c["evaluation"])
            entries.append({"name": run.name, "sha256": digest(out / "contract.json")})
        bind(root / "suite.json", {"schema": SCHEMA, "points": entries, "mode": args.mode,
                                   "model": model, "budget_gpu_seconds": budget,
                                   "measurement_wall_seconds": args.measurement_wall_seconds,
                                   "eval_timeout": args.eval_timeout,
                                   "cost_scope": "incremental branch allocations; shared source cache and offline fit reported separately"})
    print(f"[prepared] {len(entries)} points, {budget:.0f} GPU-s per branch; mode={args.mode}")


def entries(root):
    s = gate.read(root / "suite.json")
    if s.get("schema") != SCHEMA:
        raise ValueError("unsupported GPU gate suite")
    for item in s["points"]:
        if Path(item["name"]).name != item["name"]:
            raise ValueError("invalid point path")
        out = root / "points" / item["name"]
        if digest(out / "contract.json") != item["sha256"]:
            raise ValueError("point contract changed")
        yield out


def verify(out):
    c = gate.read(out / "contract.json")
    for name, value in c["source_hashes"].items():
        if digest(Path(c["source_run"]) / name) != value:
            raise ValueError(f"source changed: {name}")
    if digest(Path(c["config"]["model"]) / "config.json") != c["scope"]["model"]:
        raise ValueError("base model configuration changed")
    if gate.read(out / "evaluation.json") != c["evaluation"]:
        raise ValueError("independent evaluation changed")
    return c


def initial(out, suite, c):
    with lease(out / ".initial.lock", blocking=True):
        path = out / "initial.json"
        if path.exists():
            return gate.read(path)
        directory = out / "measurement"
        spent(directory)
        payload_path = directory / "payload.json"
        if payload_path.exists():
            result = {"payload": gate.read(payload_path), "gpu_seconds": spent(directory),
                      "decision_schedule": "once_before_training"}
            bind(path, result)
            return result
        if (directory / "cost.jsonl").exists():
            raise ValueError("previous initial measurement did not publish a decision; refusing a second measurement")
        model = suite["model"]
        def action():
            if suite["mode"] == "study":
                from artifact_contract import validate_generation_contract
                validate_generation_contract(Path(c["source_run"]), ("rollouts_behavior_train",))
                return study.cached_features(Path(c["source_run"]) / "rollouts_behavior_train.jsonl",
                                             step=c["config"]["drift"], expected_prompts=c["n"],
                                             expected_responses=8, allocated_gpus=GPUS,
                                             max_wall_seconds=suite["measurement_wall_seconds"])
            config = gate.GateConfig(scope=c["scope"], total_gpu_seconds=c["budget_gpu_seconds"],
                                     measurement_gpu_seconds=min(c["budget_gpu_seconds"], GPUS*suite["measurement_wall_seconds"]),
                                     measurement_wall_seconds=suite["measurement_wall_seconds"],
                                     start_step=c["config"]["drift"], model_id=model["model_id"] if model else None)
            if model:
                from artifact_contract import validate_generation_contract
                validate_generation_contract(Path(c["source_run"]), ("rollouts_behavior_train",))
            return study.initialize(out / "gate.json", config, model=model,
                                    rollouts=Path(c["source_run"]) / "rollouts_behavior_train.jsonl",
                                    prompts=c["n"], responses=8, allocated_gpus=GPUS)
        def save_payload():
            value = action()
            bind(payload_path, value)
            return value
        try:
            payload = meter(directory, "initial-distribution", c["scope"]["gpu_type"], action=save_payload,
                            ledger="research" if suite["mode"] == "study" else "deployment")
        except Exception as exc:
            if suite["mode"] != "deploy":
                raise
            spent(directory)
            payload = {"action": "random", "reason": "invalid_initial_measurement", "error": str(exc)}
            bind(payload_path, payload)
        result = {"payload": payload, "gpu_seconds": spent(directory),
                  "decision_schedule": "once_before_training"}
        bind(path, result)
        return result


def freeze_subset(out, c, arm, indices=None):
    path = out / "subsets" / f"subset-{arm}.json"
    from select_rules import topk_count
    k = topk_count(c["n"], c["config"]["topk_frac"])
    if indices is None:
        indices = sorted(random.Random(c["config"]["seed"]+907).sample(range(c["n"]), k))
    if len(indices) != k or len(set(indices)) != k or any(type(i) is not int or i not in range(c["n"]) for i in indices):
        raise ValueError("invalid fixed subset")
    source = gate.read(Path(c["source_run"]) / "prompts.json")
    bind(path, {**source, "train": [source["train"][i] for i in indices],
                "selector": arm, "selected_idx": indices, "k": k})
    bind(path.with_suffix(".sha256.json"), {"sha256": digest(path)})
    return path


def select_once(out, c, arm_dir, cap, env, devices, *, ledger="research"):
    path = out / "selected.json"
    if path.exists():
        if gate.read(path.with_suffix(".sha256.json")) != {"sha256": digest(path)}:
            raise ValueError("frozen selection changed")
        return gate.read(path)["indices"]
    import low_order_experiment as low
    scoring = out / "scoring"
    meter(arm_dir, "score-prepare", c["scope"]["gpu_type"], action=lambda: low.prepare(
        scoring, [Path(c["source_run"])], evaluation=out / "inputs/test.json", steps=1,
        eval_k=c["eval_k"], arms=(c["scope"]["selector"],)), ledger=ledger)
    point = next(low.entries(scoring))
    for stage in ("validation", "score"):
        if stage == "validation" and (point / "direction.pt").exists():
            continue
        seconds = (cap-spent(arm_dir))/GPUS
        commands = [([sys.executable, str(ROOT / "src/low_order_experiment.py"), "worker",
                      "--out", str(point), "--stage", stage, "--shard", str(i)], devices[i]) for i in range(GPUS)]
        meter(arm_dir, stage, c["scope"]["gpu_type"], commands=commands, env=env, timeout=seconds, ledger=ledger)
        meter(arm_dir, f"merge-{stage}", c["scope"]["gpu_type"],
              action=lambda stage=stage: low.merge_direction(point) if stage == "validation" else low.merge(point), ledger=ledger)
    selected = gate.read(point / "selection.json")["selected"][c["scope"]["selector"]]
    bind(path, {"indices": selected, "scoring_sha256": digest(point / "selection.json")})
    bind(path.with_suffix(".sha256.json"), {"sha256": digest(path)})
    return selected


def train_command(out, c, arm, remaining):
    import evidence_downstream as ed
    args = ed.train_args(c["config"], Path(c["source_run"]), out, arm, c["max_steps"])
    # E5's optional online diagnostic is outside this frozen one-shot protocol.
    args = [arg for arg in args if arg != "--reliability-log"]
    args[args.index(str(ROOT / "src/train_policy_grpo.py"))] = str(ROOT / "src/train_selection_gate_grpo.py")
    return [sys.executable, *args, "--wall-budget-deadline", str(time.monotonic()+remaining/GPUS),
            "--budget-save-reserve", "30"]


def policy(out, c, arm):
    import evidence_downstream as ed
    from train_policy_grpo import GrpoConfig, validate_policy_lineage
    path = out / arm / "policy"
    stop = gate.read(path / "budget_stop.json")
    parent = Path(c["source_run"]) / f"policy_step_{c['config']['drift']}"
    if stop["use_parent_policy"]:
        if stop["completed_steps"] != c["config"]["drift"]:
            raise ValueError("parent-only branch reports completed updates")
        return parent
    cfg = c["config"]
    validate_policy_lineage(path, target_steps=stop["completed_steps"], world_size=4,
                            training_objective="grpo", expected_start_step=cfg["drift"],
                            expected_parent=parent, expected_model=Path(cfg["model"]), expected_seed=cfg["seed"],
                            expected_max_new_tokens=cfg["max_new_tokens"], expected_prompt_format=cfg["prompt_format"],
                            expected_config=asdict(GrpoConfig(**dict({field.removeprefix("grpo_"): cfg[field]
                                                 for field in ed.TRAIN_FLAGS.values() if field != "grpo_logprob_micro_batch"},
                                                 checkpoint_every=5))),
                            expected_prompts=out / "subsets" / f"subset-{arm}.json", require_complete_hashes=True)
    return path


def eval_contract(out, c, arm, shard):
    p = policy(out, c, arm)
    n = len(c["evaluation"]["val"])
    indices = range(n*shard//GPUS, n*(shard+1)//GPUS)
    binding = {"experiment_sha256": digest(out / "contract.json"), "adapter_sha256": digest(p / "adapter_model.safetensors"),
               "policy_manifest_sha256": digest(p / "policy_train.json"), "arm": arm, "shard": shard}
    return p, indices, binding


def evaluate(out, arm, shard):
    import evidence_downstream as ed
    c = verify(out)
    p, indices, binding = eval_contract(out, c, arm, shard)
    target = out / arm / "evaluation"
    with lease(target / f"shard-{shard}.lock"):
        bind(target / f"shard-{shard}.contract.json", binding)
        path = target / f"shard-{shard}.jsonl"
        done = target / f"shard-{shard}.done.json"
        if done.exists():
            if gate.read(done) != {"binding": binding, "sha256": digest(path)}:
                raise ValueError("evaluation artifact changed")
            ed.reward_rows(path, indices, c["eval_k"])
            return
        from rollout import collect_rollouts, load_policy
        model, tokenizer = load_policy(c["config"]["model"], p)
        collect_rollouts(model, tokenizer, c["evaluation"]["val"][indices.start:indices.stop], c["eval_k"],
                         c["config"]["max_new_tokens"], float(c["config"]["temperature"]), path,
                         idx_offset=indices.start, sampling_seed_base=c["eval_seed"])
        ed.reward_rows(path, indices, c["eval_k"])
        bind(done, {"binding": binding, "sha256": digest(path)})


def rewards(out, c, arm):
    import evidence_downstream as ed
    values = {str(i): [] for i in range(len(c["evaluation"]["val"]))}
    for shard in range(GPUS):
        _, indices, binding = eval_contract(out, c, arm, shard)
        path = out / arm / "evaluation" / f"shard-{shard}.jsonl"
        if gate.read(path.with_suffix(".done.json")) != {"binding": binding, "sha256": digest(path)}:
            raise ValueError("evaluation completion hash changed")
        for row in ed.reward_rows(path, indices, c["eval_k"]):
            values[str(row["prompt_idx"])].append(row["reward"])
    return {key: statistics.fmean(value) for key, value in values.items()}


def run_arm(out, suite, arm, devices, env):
    directory = out / arm
    c = gate.read(out / "contract.json")
    gpu_type = c["scope"]["gpu_type"]
    ledger = "research" if suite["mode"] == "study" else "deployment"
    if (directory / "result.json").exists():
        return
    spent(directory)
    meter(directory, "verify-inputs", gpu_type, action=lambda: verify(out), ledger=ledger)
    if arm == "random_full":
        measured, action = 0., "random"
    else:
        first = initial(out, suite, c)
        measured = first["gpu_seconds"]
        action = ("select" if arm == "selection_reduced" else "random") if suite["mode"] == "study" else first["payload"]["action"]
    cap = c["budget_gpu_seconds"]-measured
    bind(directory / "decision.json", {"action": action, "measurement_gpu_seconds": measured,
                                        "budget_gpu_seconds": cap, "decision_schedule": "once_before_training"})
    if cap <= 0:
        raise ValueError("initial measurement exhausted the training allocation")
    subset = out / "subsets" / f"subset-{arm}.json"
    if subset.exists():
        if gate.read(subset.with_suffix(".sha256.json")) != {"sha256": digest(subset)}:
            raise ValueError("frozen training subset changed")
        execution = directory / "execution.json"
        if execution.exists():
            action = gate.read(execution)["action"]
    else:
        indices = None
        execution = directory / "execution.json"
        if execution.exists():
            action = gate.read(execution)["action"]
        if action == "select":
            try:
                indices = select_once(out, c, directory, cap, env, devices, ledger=ledger)
            except Exception as exc:
                if suite["mode"] != "deploy":
                    raise
                # This is a failed-selector fallback, not another gate measurement.
                action = "random"
                bind(execution, {"action": action, "reason": "selector_failed", "error": str(exc)})
        if not execution.exists():
            bind(execution, {"action": action, "reason": "initial_choice"})
        meter(directory, "freeze-subset", gpu_type, action=lambda: freeze_subset(out, c, arm, indices), ledger=ledger)
    stop_path = directory / "policy/budget_stop.json"
    if not stop_path.exists():
        remaining = cap-spent(directory)
        if remaining <= 0:
            raise ValueError("scoring or previous failed attempts exhausted the branch budget")
        if remaining/GPUS <= 30:
            bind(stop_path, {"completed_steps": c["config"]["drift"], "stop_reason": "no_block_fits",
                             "use_parent_policy": True, "requested_target_steps": c["config"]["drift"]+c["max_steps"]})
        else:
            meter(directory, "train", gpu_type, commands=[(train_command(out, c, arm, remaining), ",".join(devices))],
                  env=env, timeout=remaining/GPUS, ledger="research" if suite["mode"] == "study" else "deployment")
    policy(out, c, arm)
    commands = [([sys.executable, str(HERE), "worker", "--root", str(out), "--arm", arm,
                  "--shard", str(i)], devices[i]) for i in range(GPUS)
                if not (directory / "evaluation" / f"shard-{i}.done.json").exists()]
    if commands:
        meter(directory, "evaluate", gpu_type, commands=commands, env=env,
              timeout=suite["eval_timeout"], ledger="reporting")
    stop = gate.read(stop_path)
    used = spent(directory)
    bind(directory / "result.json", {"rewards": rewards(out, c, arm), "used_gpu_seconds": used,
                                     "budget_gpu_seconds": cap, "stop_reason": stop["stop_reason"],
                                     "completed_steps": stop["completed_steps"], "action": action,
                                     "complete": True, "matched_budget": used <= cap,
                                     "cost": cost(directory)})
    (directory / "failure.json").unlink(missing_ok=True)


def work(root):
    import additive_experiment as ae
    suite = gate.read(root / "suite.json")
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(devices) != GPUS or any(not v for v in devices) or len(set(devices)) != GPUS or os.environ.get("OM_NODE_LOCK_HELD") != "1":
        raise ValueError("runner requires an admitted four-GPU allocation")
    hardware = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader",
                                        "-i", ",".join(devices)], text=True, timeout=20).strip().splitlines()
    failures = 0
    for out in entries(root):
        c = gate.read(out / "contract.json")
        if len(hardware) != GPUS or set(map(str.strip, hardware)) != {c["scope"]["gpu_type"]}:
            raise ValueError(f"allocated hardware differs from contract: {hardware}")
        bind(out / "inputs/test.json", {"test": c["evaluation"]["val"], "provenance": c["evaluation"]["provenance"]})
        for arm in study.BRANCHES if suite["mode"] == "study" else ("deployment",):
            try:
                with lease(out / arm / ".task.lock"):
                    run_arm(out, suite, arm, devices, ae.model_environment(c["config"]))
            except BlockingIOError:
                continue
            except Exception as exc:  # noqa: BLE001 - isolate and report task failures
                failures += 1
                traceback.print_exc()
                gate.atomic_json(out / arm / "failure.json", {"error": str(exc), "host": socket.gethostname(), "time": time.time()})
                print(f"[failed] {out.name}/{arm}: {exc}; continuing to other arms", flush=True)
    print_status(status(root))
    return 1 if failures else 0


def status(root):
    if not (root / "suite.json").exists():
        return {"state": "not prepared", "root": str(root)}
    suite = gate.read(root / "suite.json")
    rows = []
    for out in entries(root):
        for arm in study.BRANCHES if suite["mode"] == "study" else ("deployment",):
            directory = out / arm
            result = directory / "result.json"
            failure = directory / "failure.json"
            progress_path = directory / "progress.json"
            current = gate.read(progress_path) if progress_path.exists() else {}
            active = current.get("state") == "running" and time.time()-current.get("updated", 0) < 45
            cfg = gate.read(out / "contract.json")["config"]
            rows.append({"point": out.name, "arm": arm,
                         "seed": cfg["seed"], "drift": cfg["drift"], "progress": current,
                         "state": "complete" if result.exists() else "running" if active else "failed" if failure.exists() else "pending",
                         "result": gate.read(result) if result.exists() else None,
                         "error": gate.read(failure)["error"] if failure.exists() else None,
                         "cost": cost(directory)})
    return {"schema": SCHEMA, "mode": suite["mode"], "rows": rows,
            "complete": sum(r["state"] == "complete" for r in rows), "total": len(rows)}


def print_status(result):
    if "rows" not in result:
        print(json.dumps(result, indent=2))
        return
    print(f"[gate] {result['mode']}: {result['complete']}/{result['total']} complete")
    print("POINT       ARM                 STATE      GPU-MIN   REWARD / PHASE")
    for row in result["rows"]:
        allocation = sum(v["gpu_seconds"] for v in row["cost"]["ledgers"].values())/60
        reward = row["result"]["rewards"] if row["result"] else None
        detail = f"{statistics.fmean(reward.values()):.4f}" if reward else row["progress"].get("phase", "-")
        label = f"s{row['seed']}/d{row['drift']}"
        print(f"{label:<11} {row['arm']:<19} {row['state']:<10} {allocation:>7.1f}   {detail}")
        if row["error"]:
            print(f"  ERROR: {row['error']}")
        if not row["cost"]["complete"]:
            print("  COST: allocation still open or interrupted; not counted as zero")


def summarize(root):
    suite = gate.read(root / "suite.json")
    if suite["mode"] != "study":
        return status(root)
    points, excluded = [], []
    for out in entries(root):
        try:
            c = verify(out)
            first = gate.read(out / "initial.json")
            cfg = c["config"]
            prefix = f"policy_step_{cfg['drift']}"
            lineage = {"parent_sha256": c["source_hashes"][f"{prefix}/adapter_model.safetensors"],
                       "optimizer_sha256": c["source_hashes"][f"{prefix}/optimizer.pt"],
                       "pool_sha256": c["scope"]["pool_sha256"], "evaluation_sha256": gate.fingerprint(c["evaluation"]),
                       "learner_config_sha256": gate.fingerprint(cfg), "gpu_type": c["scope"]["gpu_type"]}
            branches = {}
            for arm in study.BRANCHES:
                result = gate.read(out / arm / "result.json")
                if result["rewards"] != rewards(out, c, arm) or result["used_gpu_seconds"] != spent(out / arm):
                    raise ValueError("reward or cost record changed after publication")
                branches[arm] = {**result, **lineage}
            point = {"id": out.name, "trajectory_id": f"{cfg['model']}:seed-{cfg['seed']}", "role": c["role"],
                     "step": cfg["drift"], "feature_step": first["payload"]["feature_step"], "scope": c["scope"],
                     "full_pool_coverage": first["payload"]["full_pool_coverage"], "decision_schedule": "once_before_training",
                     "features": first["payload"]["features"], "budget_gpu_seconds": c["budget_gpu_seconds"],
                     "measurement_gpu_seconds": first["gpu_seconds"], "branches": branches}
            study.validate_point(point)
            points.append(point)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            excluded.append({"point": out.name, "reason": str(exc)})
    result = {"schema": study.STUDY_SCHEMA, "data_kind": "observed", "points": points, "excluded": excluded}
    if points:
        study.validate_study(result)
    gate.atomic_json(root / "study.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("prepare", "run", "worker", "status", "summarize"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--matrix", type=Path)
    p.add_argument("--runs", type=Path, nargs="+")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    p.add_argument("--drift", type=int, default=100)
    p.add_argument("--mode", choices=("study", "deploy"), default="study")
    p.add_argument("--model", type=Path)
    p.add_argument("--selector", choices=("low_order", "pair_u2"), default="low_order")
    p.add_argument("--role", choices=("development", "calibration", "test"), default="development")
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
    p.add_argument("--arm", choices=(*study.BRANCHES, "deployment"))
    p.add_argument("--shard", type=int, choices=range(4))
    p.add_argument("--json", action="store_true", help="JSON status instead of the compact table")
    args = p.parse_args()
    try:
        if args.command == "prepare":
            if not args.runs and args.matrix is None:
                p.error("prepare requires --runs or --matrix")
            prepare(args)
        elif args.command == "run":
            return work(args.root)
        elif args.command == "worker":
            if args.arm is None or args.shard is None or os.environ.get("OM_NODE_LOCK_HELD") != "1":
                p.error("worker requires --arm, --shard, and allocated node admission")
            evaluate(args.root, args.arm, args.shard)
        else:
            result = status(args.root) if args.command == "status" else summarize(args.root)
            if args.command == "status" and not args.json:
                print_status(result)
            else:
                print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError) as exc:
        print(f"[gate] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    raise SystemExit(main())
