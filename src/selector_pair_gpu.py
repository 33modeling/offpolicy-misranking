"""Matched-state G/D curves and an independently executed, held-out decision.

Four private switch roots reuse certified prefixes, never continuation results.
The legacy random gate is not fitted or used. Its frozen learner, selection,
evaluation, atomic publication, and failed-attempt ledgers remain unchanged.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import selection_gate as core
import selection_gate_gpu as base
import selection_switch_gpu as switch
import selector_pair as pair

BRANCHES = {"on_policy": "fresh_r", "cached": "difficulty",
            "adaptive-on_policy": "fresh_r", "adaptive-cached": "difficulty"}
TRAINER = "src/selector_pair_train.py"
EXTRA_CODE = ("src/selector_pair.py", "src/selector_pair_gpu.py", TRAINER,
              "src/selection_switch_curve_train.py", "scripts/run_selector_pair.sh")
BOOTSTRAP_SCHEMA = "offpolicy-selector-pair/setup-v1"
CONFIG_KEYS = ("matrix", "prefix_source", "target_reward", "budget_gpu_seconds", "curve_points",
               "eval_k", "dataset", "gpu_type", "eval_timeout")
# c78ca17: only startup/CLI handling changes in this patch. Preserve existing
# manifests, protocol IDs, labels and decisions, and reject any scientific change.
PRE_BOOTSTRAP_CODE = "3ad11e06bc7670012c91898b6d4e09802eab195422f4a62919ffaf4e95725f9b"
PRE_DEFAULTS_CODE = "b589d47572403fe0c217ac3e2f925e69906db54822f1f29c6dca9dd1fba3b98b"
STARTUP_FILES = {"src/selector_pair_gpu.py", "scripts/run_selector_pair.sh"}
RUN_DEFAULTS = {"target_reward": .35, "budget_gpu_seconds": 87120.}


def code_hashes():
    return {**switch.code_hashes(), **{name: base.digest(base.ROOT / name) for name in EXTRA_CODE}}


def compatible_code(recorded):
    current = code_hashes()
    return recorded == current or (
        isinstance(recorded, dict) and core.fingerprint(recorded) in {PRE_BOOTSTRAP_CODE, PRE_DEFAULTS_CODE}
        and set(recorded) == set(current)
        and all(recorded[name] == value for name, value in current.items() if name not in STARTUP_FILES))


def bind_startup_runtime(root, recorded):
    if recorded != code_hashes():
        # Pin the first reviewed upgrade too. Later edits to these entry points
        # must not acquire an unlimited exemption on a predecessor's run.
        path = root / "startup-runtime.json"
        receipt = {
            "schema": "offpolicy-selector-pair/startup-runtime-v1",
            "frozen_code_hashes": recorded, "runtime_code_hashes": code_hashes(),
            "change": "setup placeholder, actionable first launch and interrupted prepare recovery only"}
        if path.exists():
            previous = core.read(path)
            runtime = previous.get("runtime_code_hashes", {})
            if (core.fingerprint(runtime) == PRE_DEFAULTS_CODE and compatible_code(runtime)
                    and previous == {**receipt, "runtime_code_hashes": runtime}):
                # Preserve the first upgrade receipt; pin this reviewed second
                # startup-only upgrade separately, without changing the study.
                base.bind(root / "startup-defaults-runtime.json", {
                    **receipt, "previous_receipt_sha256": base.digest(path),
                    "change": "no-argument launch and defaults for unfrozen setup only"})
                return
        base.bind(path, receipt)


def setup_config():
    work = Path(os.environ.get("OM_WORK", f"/group-volume/{os.environ.get('OM_USER', 'minsoo3.kim')}/offpolicy-misranking"))
    return {"matrix": os.environ.get("OM_OLMO3_ROOT", str(work / "runs" /
                os.environ.get("OM_OLMO3_MODEL_TAG", "olmo3-1025-7b-base-rlzero-grpo-h100-v2"))),
            "prefix_source": os.environ.get("SWITCH_PREFIX_SOURCE", str(work / "runs/selection-switch-v1")),
            **RUN_DEFAULTS, "curve_points": 9, "eval_k": 8,
            "dataset": "math500", "gpu_type": "NVIDIA H100 80GB HBM3", "eval_timeout": 14400.}


def request_config(request):
    return {key: request["training_cap_gpu_seconds" if key == "budget_gpu_seconds" else key] for key in CONFIG_KEYS}


def initialize(root, configuration=None):
    """Publish a clearly non-runnable setup file; never invent a frozen study."""
    root = root.resolve()
    if root in {base.ROOT, Path.home(), Path(root.anchor)}:
        raise ValueError("use a dedicated pair output directory")
    with base.lease(root / ".pair.lock"):
        path = root / "pair.json"
        if path.exists():
            value = core.read(path)
            if value.get("schema") not in {BOOTSTRAP_SCHEMA, pair.SCHEMA}:
                raise ValueError(f"unrecognized pair.json; preserved without overwriting: {path}")
            if value["schema"] == BOOTSTRAP_SCHEMA and not (root / "request.json").exists():
                config = value.get("configuration", {})
                missing = {key: default for key, default in RUN_DEFAULTS.items()
                           if key in config and config[key] in (None, "")}
                if missing:
                    value = {**value, "configuration": {**config, **missing},
                             "status": "ready_to_prepare",
                             "note": "Setup only. The launcher validates inputs and freezes the experiment before training."}
                    core.atomic_json(path, value)
                    print(f"[defaults] filled unset setup values: {', '.join(missing)}", flush=True)
            return value
        config = setup_config() if configuration is None else configuration
        request = root / "request.json"
        status = "ready_to_prepare"
        if request.exists():
            saved = core.read(request)
            if saved.get("schema") != pair.SCHEMA or not compatible_code(saved.get("code_hashes")):
                raise ValueError("partial preparation has an incompatible frozen request; preserved")
            config, status = request_config(saved), "preparation_incomplete"
        value = {"schema": BOOTSTRAP_SCHEMA, "status": status, "configuration": config,
                 "note": "Setup only. The launcher validates inputs and freezes the experiment before training."}
        base.bind(path, value)
        print(f"[initialized] {path}", flush=True)
        return value


def preparation_options(root, overrides=None):
    value = initialize(root)
    config = value["configuration"] if value["schema"] == BOOTSTRAP_SCHEMA else request_config(value)
    if set(config) != set(CONFIG_KEYS):
        raise ValueError("setup configuration fields changed; expected " + ", ".join(CONFIG_KEYS))
    config = {**config, **{k: v for k, v in (overrides or {}).items() if v is not None}}
    missing = [key for key in CONFIG_KEYS if config[key] is None or config[key] == ""]
    if missing:
        raise ValueError(f"incomplete setup configuration: {root / 'pair.json'}; "
                         f"missing {', '.join(missing)}. No GPU work started.")
    config["matrix"], config["prefix_source"] = Path(config["matrix"]), Path(config["prefix_source"])
    return SimpleNamespace(root=root, **config)


def ensure_prepared(root):
    value = initialize(root)
    if value["schema"] == BOOTSTRAP_SCHEMA:
        prepare(preparation_options(root))
    return manifest(root)


def manifest(root):
    if not (root / "pair.json").exists():
        raise ValueError(f"pair is not prepared: {root}; run the launcher prepare command first")
    p = core.read(root / "pair.json")
    if p.get("schema") == BOOTSTRAP_SCHEMA:
        raise ValueError(f"pair.json is a setup template, not a runnable experiment: {root}; run prepare")
    if p.get("schema") != pair.SCHEMA or not compatible_code(p.get("code_hashes")):
        raise ValueError("pair runtime changed; preserve this frozen run and use its original code")
    if p.get("protocol_id") != core.fingerprint({k: v for k, v in p.items() if k != "protocol_id"}):
        raise ValueError("pair manifest changed")
    for name, digest in p["branch_manifests"].items():
        if base.digest(root / "branches" / name / "switch.json") != digest:
            raise ValueError("frozen branch manifest changed")
    bind_startup_runtime(root, p["code_hashes"])
    return p


def prepare(args):
    root = args.root.resolve()
    prefix = args.prefix_source.resolve()
    matrix = args.matrix.resolve()
    if (root == base.ROOT or root == Path.home() or root == Path(root.anchor)
            or root == prefix or root in prefix.parents or prefix in root.parents
            or root == matrix or root in matrix.parents or matrix in root.parents):
        raise ValueError("use a new output root disjoint from the prefix source and matrix")
    core.number(args.target_reward, "preregistered target reward", 1e-12, 1.)
    core.number(args.budget_gpu_seconds, "training GPU-second cap", 120.)
    core.integer(args.curve_points, "curve points", 1)
    core.integer(args.eval_k, "evaluation responses", 1)
    config = {key: str(getattr(args, key)) if key in {"matrix", "prefix_source"} else getattr(args, key)
              for key in CONFIG_KEYS}
    initialize(root, config)
    if not (prefix / "switch.json").is_file():
        raise ValueError(f"certified prefix source is missing: {prefix / 'switch.json'}. "
                         "Run on the node with the original experiment storage mounted. No GPU work started.")
    request = {"schema": pair.SCHEMA, "matrix": str(matrix), "prefix_source": str(prefix),
               "prefix_sha256": base.digest(prefix / "switch.json"),
               "target_reward": args.target_reward, "training_cap_gpu_seconds": args.budget_gpu_seconds,
               "curve_points": args.curve_points, "eval_k": args.eval_k, "gpu_type": args.gpu_type,
               "dataset": args.dataset, "eval_timeout": args.eval_timeout,
               "code_hashes": code_hashes()}
    with base.lease(root / ".pair.lock"):
        if (root / "request.json").exists():
            previous = core.read(root / "request.json")
            if compatible_code(previous.get("code_hashes")):
                request["code_hashes"] = previous["code_hashes"]
        base.bind(root / "request.json", request)
        bind_startup_runtime(root, request["code_hashes"])
        existing = core.read(root / "pair.json")
        if existing.get("schema") == pair.SCHEMA:
            manifest(root)
            print(f"[prepared] unchanged pair protocol: {root}")
            return
        if existing.get("schema") != BOOTSTRAP_SCHEMA:
            raise ValueError("unrecognized pair.json; refusing to overwrite")
        core.atomic_json(root / "pair.json", {**existing, "configuration": request_config(request),
                                               "status": "preparation_incomplete"})
        for name, selector in BRANCHES.items():
            options = SimpleNamespace(root=root / "branches" / name, matrix=matrix,
                prefix_source=prefix, selector=selector, gate="convergence", accounting="matched",
                curve_points=args.curve_points, curve_k=args.eval_k, eval_k=args.eval_k,
                budget_gpu_seconds=args.budget_gpu_seconds, dataset=args.dataset, gpu_type=args.gpu_type,
                eval_timeout=args.eval_timeout, prefix_timeout=args.eval_timeout,
                eval_prompts=None, pool=None, pool_manifest=None, test_count=300)
            switch.prepare(options)
        p = {**request, "development_seeds": list(pair.DEV_SEEDS), "test_seeds": list(pair.TEST_SEEDS),
             "steps": list(pair.STEPS), "features": list(pair.FEATURES),
             "target_policy": "fixed absolute reward, frozen before all continuations",
             "crossing": "first evaluated checkpoint; no interpolation or endpoint-derived target",
             "fit": "ridge alpha=1, margin=0; all nine uncensored development states required",
             "schedule": "one shot at each separately branched certified prefix; not repeated online decisions",
             "cost_scope": "allocated GPU seconds through checkpoint, including scoring, startup, retries; "
                           "adaptive diagnosis and inference charged once; offline evaluation separate",
             "trainer_override": TRAINER,
             "branch_manifests": {name: base.digest(root / "branches" / name / "switch.json") for name in BRANCHES}}
        p["protocol_id"] = core.fingerprint(p)
        # The setup placeholder is replaced only after all real inputs and four
        # branch manifests have been validated, under the same preparation lock.
        core.atomic_json(root / "pair.json", p)
    print(f"[prepared] 18 development + 24 held-out continuations; target={args.target_reward}; {root}")


def state(root, name, seed, step):
    branch = root / "branches" / name
    child = switch.child_root(branch, seed, step)
    if not (child / "net_protocol.json").exists():
        switch.publish_state(branch, seed, step)
    out = next(base.entries(child))
    return branch, out, core.read(out / "contract.json"), switch.protocol(child), core.read(child / "suite.json")


def verify_pair(root, seed, step):
    entries = {name: state(root, name, seed, step) for name in BRANCHES}
    identity = pair.matched_state([item[2] for item in entries.values()])
    return identity, entries


def install_runtime():
    switch.install_runtime()
    def train_command(out, c, arm, remaining):
        command = switch.train_command(out, c, arm, remaining)
        previous = str(base.ROOT / switch.CURVE_TRAINER)
        if previous not in command:
            raise ValueError("pair runner requires the curve trainer")
        command[command.index(previous)] = str(base.ROOT / TRAINER)
        return command
    base.train_command = train_command


def training_artifacts(out):
    """Also reject partly completed work, not just published final rewards."""
    return [str(path) for arm in switch.rule.TEST_ARMS
            for path in (out / arm / "execution.json", out / arm / "result.json",
                         out / arm / "policy", out / "selector-work" / arm,
                         out / arm / "cached-select", out / arm / "cost.jsonl") if path.exists()]


def diagnostic(entry, env):
    _, out, _, protocol, suite = entry
    directory = out / ("measurement" if protocol["mode"] == "study" else "gate_measurement")
    record = switch.runtime.measure_once(out, suite, protocol, directory, env)
    if record["status"] != "complete":
        raise ValueError("diagnosis failed: no unmeasured prediction or silent default action")
    return core.read(directory / "measurement.json")["features"], record, directory


def environment(c):
    import additive_experiment as ae
    return ae.model_environment(c["config"])


def execute(entry, arm, devices):
    branch, out, c, protocol, suite = entry
    root = branch.parent.parent
    manifest(root)  # Recheck before starting a new process from on-disk code.
    env = {**environment(c), "PAIR_PROTOCOL_ROOT": str(root)}
    with base.lease(out / arm / ".task.lock"):
        switch.runtime.run_arm(out, suite, protocol, arm, devices, env)
        switch.curve_once(branch, switch.manifest(branch), out, c, arm, suite, devices, env)
        if not switch.branch_finished(switch.manifest(branch), out / arm):
            raise ValueError(f"curve publication is pending: {out / arm}")


def final_receipt(directory, events, completed, adapter):
    path = directory / "policy/curve-cost/final.json"
    if not path.exists():
        # The trainer can die after policy/budget_stop publication but before its
        # final timestamp. A validated policy plus a closed allocation gives a
        # conservative, exact allocation-end cost; no training is repeated.
        trains = [(a, b) for a, b in pair.finished_events(events) if b["phase"] == "train"]
        if not trains:
            raise ValueError("published policy has no training allocation")
        finish = trains[-1][1]
        base.bind(path, {"step": completed, "adapter_sha256": base.digest(adapter),
                         "event_id": finish["event_id"], "time": finish["time"],
                         "recovered_from": "closed allocation after verified policy publication"})
    return core.read(path)


def measured_curve(entry, arm):
    _, out, c, protocol, _ = entry
    directory = out / arm
    result = switch.runtime.validate_result(out, protocol, arm)
    base.policy(out, c, arm)  # Full optimizer/policy lineage before trusting receipts.
    curve = core.read(directory / "curve.json")
    if curve["result_sha256"] != base.digest(directory / "result.json"):
        raise ValueError("curve/result binding changed")
    _, events = base.read_cost_events(directory)
    closed = pair.finished_events(events)
    observed = sum(finish["allocated_gpu_seconds"] for _, finish in closed
                   if finish["ledger"] != "reporting" or finish["phase"] in switch.SCORING_PHASES)
    start = c["config"]["drift"]
    stop = result["completed_steps"]
    # The legacy curve map overwrites the parent when zero updates fit. Obtain
    # the genuine parent evaluation, never invent a successful target crossing.
    baseline = switch.curve_reward(out, c, arm, start, c["eval_k"])
    points = [{"updates": 0, "reward": baseline, "gpu_seconds": 0.,
               "training_gpu_seconds": 0., "scoring_gpu_seconds": 0., "other_gpu_seconds": 0.}]
    hashes = {"result": base.digest(directory / "result.json"),
              "curve": base.digest(directory / "curve.json"), "cost": base.digest(directory / "cost.jsonl")}
    for key, point in sorted(curve["points"].items(), key=lambda item: int(item[0])):
        step = int(key)
        if step <= start:
            continue
        is_final = bool(point.get("final"))
        if is_final:
            if step != stop:
                raise ValueError("final curve checkpoint differs from the policy")
            reward = statistics.fmean(result["rewards"].values())
            adapter = directory / "policy/adapter_model.safetensors"
            receipt = final_receipt(directory, events, step, adapter)
            receipt_path = directory / "policy/curve-cost/final.json"
        else:
            reward = switch.curve_reward(out, c, arm, step, c["eval_k"])
            adapter = switch.curve_adapter(out, c, arm, step) / "adapter_model.safetensors"
            receipt_path = adapter.parent / "cost-receipt.json"
            receipt = core.read(receipt_path)
            if receipt.get("checkpoint_state_id") != core.fingerprint(core.read(adapter.parent / "checkpoint_state.json")):
                raise ValueError("checkpoint cost receipt/state binding changed")
        if (receipt["step"] != step or receipt["adapter_sha256"] != base.digest(adapter)
                or point["updates"] != step-start or not math_close(reward, point["reward"])):
            raise ValueError("checkpoint reward/cost evidence changed")
        points.append({"updates": step-start, "reward": reward,
                       **pair.cost_at_checkpoint(events, receipt, final=is_final)})
        hashes[str(receipt_path.relative_to(directory))] = base.digest(receipt_path)
    pair.crossing(points, 1.)  # Validate ordering even if the actual target is low.
    return {"points": points, "artifact_hashes": hashes, "path": str(directory),
            "observed_gpu_seconds": observed,
            "allocated_cost": result["cost"],
            "curve_evaluation_cost": base.cost(directory / "curve"),
            "parent_evaluation_cost": base.cost(out / "curve-parent")}


def math_close(a, b):
    return abs(a-b) <= 1e-12


def development_row(root, p, seed, step):
    identity, entries = verify_pair(root, seed, step)
    curves = {name: measured_curve(entries[name], "selection_reduced") for name in pair.SELECTORS}
    if curves["on_policy"]["points"][0]["reward"] != curves["cached"]["points"][0]["reward"]:
        raise ValueError("paired parent evaluations differ; investigate before fitting")
    features, record, directory = diagnostic(entries["on_policy"], environment(entries["on_policy"][2]))
    other, _, _ = diagnostic(entries["cached"], environment(entries["cached"][2]))
    if features != other:
        raise ValueError("paired pre-continuation features differ")
    crossings = {name: pair.crossing(curve["points"], p["target_reward"],
                    observed_gpu_seconds=curve.get("observed_gpu_seconds")) for name, curve in curves.items()}
    return {"seed": seed, "step": step, "role": "development", "protocol_id": p["protocol_id"],
            "state_id": identity, "features": features, "diagnostic": record,
            "diagnostic_path": str(directory), "curves": curves, "crossings": crossings,
            "contrast": pair.contrast(crossings["on_policy"], crossings["cached"])}


def develop(root, p, devices):
    for seed in pair.DEV_SEEDS:
        for step in pair.STEPS:
            identity, entries = verify_pair(root, seed, step)
            folder = root / "development" / f"s{seed}-t{step}"
            base.bind(folder / "state.json", {"state_id": identity, "protocol_id": p["protocol_id"]})
            for name in (tuple(pair.SELECTORS) if seed % 2 == 0 else tuple(pair.SELECTORS)[::-1]):
                print(f"[pair] development s{seed}/t{step}/{name}", flush=True)
                execute(entries[name], "selection_reduced", devices)
            base.bind(folder / "result.json", development_row(root, p, seed, step))


def fit(root, p):
    rows = [development_row(root, p, seed, step) for seed in pair.DEV_SEEDS for step in pair.STEPS]
    for row in rows:
        base.bind(root / "development" / f"s{row['seed']}-t{row['step']}" / "result.json", row)
    started = time.monotonic()
    model = pair.fit(rows, p["protocol_id"])
    path = root / "model.json"
    base.bind(path, model)
    if not (root / "fit-cost.json").exists():
        elapsed = time.monotonic()-started
        allocated = 4 if os.environ.get("OM_NODE_LOCK_HELD") == "1" and os.environ.get("CUDA_VISIBLE_DEVICES") else 0
        base.bind(root / "fit-cost.json", {"cpu_wall_seconds": elapsed,
                  "ledger": "offline research", "gpu_seconds": elapsed*allocated,
                  "allocated_gpus": allocated, "model_sha256": base.digest(path)})
    print(f"[frozen] development-only H predictor: {path}")


def freeze(root, p):
    path = root / "test-decisions.json"
    if path.exists():
        return decisions(root, p)
    model = pair.validate_model(core.read(root / "model.json"))
    # Global barrier: every held-out decision must precede *any* held-out
    # scoring/training/evaluation. Diagnose only cached inputs/prefix logs.
    entries = {}
    for seed in pair.TEST_SEEDS:
        for step in pair.STEPS:
            identity, states = verify_pair(root, seed, step)
            if any(training_artifacts(entry[1]) for entry in states.values()):
                raise ValueError("held-out continuation precedes the frozen decision barrier")
            entries[(seed, step)] = (identity, states)
    for (seed, step), (identity, states) in entries.items():
        dest = root / "decisions" / f"s{seed}-t{step}"
        if (dest / "decision.json").exists():
            continue
        features, initial, measured = diagnostic(states["on_policy"], environment(states["on_policy"][2]))
        def predict():
            choice = pair.choose(model, features, seed=seed, state_id=identity, protocol_id=p["protocol_id"])
            base.bind(dest / "decision.json", {**choice, "seed": seed, "step": step,
                "state_id": identity, "protocol_id": p["protocol_id"], "features": features,
                "model_sha256": base.digest(root / "model.json"), "diagnostic": initial,
                "diagnostic_path": str(measured)})
        base.meter(dest, "predict", p["gpu_type"], action=predict, ledger="deployment")
    barrier = {"protocol_id": p["protocol_id"], "model_sha256": base.digest(root / "model.json"),
               "decisions": {f"s{s}-t{t}": base.digest(root / "decisions" / f"s{s}-t{t}" / "decision.json")
                             for s in pair.TEST_SEEDS for t in pair.STEPS}}
    # A crash after decision publication but before receipt completion is not
    # permission to ignore inference cost.
    for name in barrier["decisions"]:
        base.spent(root / "decisions" / name)
    base.bind(path, barrier)
    return decisions(root, p)


def decisions(root, p):
    barrier = core.read(root / "test-decisions.json")
    if (barrier["protocol_id"] != p["protocol_id"]
            or barrier["model_sha256"] != base.digest(root / "model.json")
            or set(barrier["decisions"]) != {f"s{s}-t{t}" for s in pair.TEST_SEEDS for t in pair.STEPS}):
        raise ValueError("test decision barrier changed")
    model, choices = pair.validate_model(core.read(root / "model.json")), {}
    for name, digest in barrier["decisions"].items():
        directory = root / "decisions" / name
        if base.digest(directory / "decision.json") != digest:
            raise ValueError("frozen test decision changed")
        choice = core.read(directory / "decision.json")
        if choice["model_sha256"] != barrier["model_sha256"]:
            raise ValueError("decision model binding changed")
        expected = pair.choose(model, choice["features"], seed=choice["seed"],
                               state_id=choice["state_id"], protocol_id=p["protocol_id"])
        if any(choice[k] != v for k, v in expected.items()):
            raise ValueError("decision does not match the frozen predictor")
        measured = Path(choice["diagnostic_path"])
        initial = core.read(measured / "initial.json")
        if (initial != choice["diagnostic"] or initial["gpu_seconds"] != base.spent(measured)
                or initial["report_sha256"] != base.digest(measured / "measurement.json")
                or choice["features"] != core.read(measured / "measurement.json")["features"]):
            raise ValueError("pre-continuation diagnostic evidence changed")
        choices[name] = {**choice, "diagnosis_gpu_seconds": initial["gpu_seconds"]+base.spent(directory)}
    return choices


def test(root, p, devices):
    choices = decisions(root, p)
    for seed in pair.TEST_SEEDS:
        for step in pair.STEPS:
            name = f"s{seed}-t{step}"
            decision = choices[name]
            identity, entries = verify_pair(root, seed, step)
            if identity != decision["state_id"]:
                raise ValueError("test parent state differs from the frozen decision")
            adaptive = entries[f"adaptive-{decision['selector']}"]
            tasks = [(entries["on_policy"], "selection_full"), (entries["cached"], "selection_full"),
                     (adaptive, "selection_full"), (entries["on_policy"], "random_full")]
            for entry, arm in tasks if seed % 2 == 0 else tasks[::-1]:
                print(f"[pair] test {name}/{entry[0].name}/{arm}", flush=True)
                execute(entry, arm, devices)
            base.bind(root / "test" / name / "result.json", test_row(root, p, seed, step, choices))


def test_row(root, p, seed, step, choices):
    decision = choices[f"s{seed}-t{step}"]
    identity, entries = verify_pair(root, seed, step)
    if identity != decision["state_id"]:
        raise ValueError("test parent state differs from the frozen decision")
    adaptive = entries[f"adaptive-{decision['selector']}"]
    curves = {"on_policy": measured_curve(entries["on_policy"], "selection_full"),
              "cached": measured_curve(entries["cached"], "selection_full"),
              "adaptive": measured_curve(adaptive, "selection_full"),
              "random": measured_curve(entries["on_policy"], "random_full")}
    if len({item["points"][0]["reward"] for item in curves.values()}) != 1:
        raise ValueError("held-out parent reward differs across matched branches")
    crossings = {key: pair.crossing(value["points"], p["target_reward"],
        diagnosis=decision["diagnosis_gpu_seconds"] if key == "adaptive" else 0.,
        observed_gpu_seconds=value.get("observed_gpu_seconds"))
        for key, value in curves.items()}
    row = {"seed": seed, "step": step, "role": "test", "state_id": identity,
           "protocol_id": p["protocol_id"], "decision": decision, "curves": curves, "crossings": crossings}
    row["audit"] = pair.audit(row)
    return row


def report(root, p):
    rows, missing = [], []
    choices = decisions(root, p) if (root / "test-decisions.json").exists() else None
    for seed in pair.TEST_SEEDS:
        for step in pair.STEPS:
            path = root / "test" / f"s{seed}-t{step}" / "result.json"
            if not path.exists():
                missing.append(f"s{seed}-t{step}")
                continue
            if choices is None:
                raise ValueError("test result without a frozen decision barrier")
            row = test_row(root, p, seed, step, choices)
            base.bind(path, row)  # Recompute from sealed evidence; reject tampered summaries.
            rows.append(row)
    value = {"schema": pair.SCHEMA, "protocol_id": p["protocol_id"], "target_reward": p["target_reward"],
             "summary": pair.summarize(rows), "rows": rows, "missing_states": missing}
    development, development_missing = [], []
    for seed in pair.DEV_SEEDS:
        for step in pair.STEPS:
            path = root / "development" / f"s{seed}-t{step}" / "result.json"
            if not path.exists():
                development_missing.append(f"s{seed}-t{step}")
                continue
            row = development_row(root, p, seed, step)
            base.bind(path, row)
            development.append(row)
    value.update(development_rows=development, missing_development_states=development_missing,
                 offline_fit_cost=core.read(root / "fit-cost.json") if (root / "fit-cost.json").exists() else None,
                 shared_prefix_cost="reused certified source; historical cost is not zero and remains in the source ledgers")
    core.atomic_json(root / "report.json", value)
    # JSON is the authoritative resumable report; CSV is a reproducible view.
    text = io.StringIO()
    writer = csv.writer(text)
    writer.writerow(["role", "seed", "prefix_updates", "arm", "updates", "reward", "gpu_seconds",
                     "training_gpu_seconds", "scoring_gpu_seconds", "other_gpu_seconds", "diagnostic_gpu_seconds"])
    for row in (*development, *rows):
        for arm, curve in row["curves"].items():
            diagnosis = row["decision"]["diagnosis_gpu_seconds"] if arm == "adaptive" else 0.
            for point in curve["points"]:
                writer.writerow([row["role"], row["seed"], row["step"], arm, point["updates"], point["reward"],
                                 point["gpu_seconds"]+(diagnosis if point["updates"] else 0.),
                                 point["training_gpu_seconds"], point["scoring_gpu_seconds"],
                                 point["other_gpu_seconds"], diagnosis if point["updates"] else 0.])
    temporary = root / "curves.csv.tmp"
    temporary.write_text(text.getvalue())
    temporary.replace(root / "curves.csv")
    print(json.dumps(value["summary"], indent=2))


def status(root, p):
    states = {}
    for role, seeds in (("development", pair.DEV_SEEDS), ("test", pair.TEST_SEEDS)):
        states[role] = [f"s{s}-t{t}" for s in seeds for t in pair.STEPS
                        if (root / role / f"s{s}-t{t}" / "result.json").exists()]
    print(json.dumps({"root": str(root), "target_reward": p["target_reward"],
                     "completed": states, "model_frozen": (root / "model.json").exists(),
                     "test_decisions_frozen": (root / "test-decisions.json").exists()}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "prepare", "ensure-prepared", "run", "develop", "fit", "freeze", "test", "report", "status", "check-code"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--matrix", type=Path)
    parser.add_argument("--prefix-source", type=Path)
    parser.add_argument("--target-reward", type=float)
    parser.add_argument("--budget-gpu-seconds", type=float)
    parser.add_argument("--curve-points", type=int)
    parser.add_argument("--eval-k", type=int)
    parser.add_argument("--dataset", choices=switch.DATASETS)
    parser.add_argument("--gpu-type")
    parser.add_argument("--eval-timeout", type=float)
    args = parser.parse_args()
    args.root = args.root.resolve()
    install_runtime()
    if args.command == "prepare":
        prepare(preparation_options(args.root, {key: getattr(args, key) for key in CONFIG_KEYS}))
        return
    if any(v is not None for k, v in vars(args).items() if k not in {"command", "root"}):
        parser.error("preparation options cannot change a frozen run; prepare a new root")
    if args.command in {"init", "status"}:
        value = initialize(args.root)
        if value["schema"] == BOOTSTRAP_SCHEMA:
            print(json.dumps({"root": str(args.root), **value}, indent=2))
            return
    p = ensure_prepared(args.root) if args.command in {"run", "develop", "ensure-prepared"} else manifest(args.root)
    if args.command == "init":
        print(f"[prepared] existing experiment preserved: {args.root / 'pair.json'}")
        return
    if args.command == "ensure-prepared":
        print(f"[ready] {args.root / 'pair.json'}")
        return
    if args.command == "check-code":
        print("[verified] pair and legacy code compatible (including reviewed startup-only migration)")
        return
    if args.command == "status":
        status(args.root, p)
        return
    if args.command == "report":
        with base.lease(args.root / ".pair.lock"):
            report(args.root, p)
        return
    with base.lease(args.root / ".pair.lock"):
        devices = switch.admitted_devices(p) if args.command in ("run", "develop", "freeze", "test") else None
        if args.command in ("run", "develop"):
            develop(args.root, p, devices)
        if args.command in ("run", "fit"):
            fit(args.root, p)
        if args.command in ("run", "freeze"):
            freeze(args.root, p)
        if args.command in ("run", "test"):
            test(args.root, p, devices)
            report(args.root, p)


if __name__ == "__main__":
    from light_selection_gate_gpu import install_signal_handlers
    install_signal_handlers()
    try:
        main()
    except (ValueError, FileNotFoundError, BlockingIOError) as exc:
        raise SystemExit(f"[pair] {exc}") from None
