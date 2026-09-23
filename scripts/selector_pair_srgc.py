"""Prospective, zero-threshold SR-GC decisions for the existing Pair queue.

Only parent-policy inputs enter decisions. Continuation rewards and target
crossings remain report outputs, never features or fitting labels.
"""
from __future__ import annotations

import contextlib
import math
import os
from pathlib import Path
import sys
import time

import selector_pair_gpu as worker
import selector_pair_srgc_score as score

SCHEMA = "offpolicy-selector-pair/sr-gc-v1"
RECEIPT = "pair-sr-gc-runtime.json"
PRE_FAILURE_HANDLING_HASHES = {
    "selector_pair_srgc.py": "8feaa80610e51f55d960d056d1f3b327846e0c3290b75ee68047bbe384758924",
    "selector_pair_srgc_score.py": "29b30516d0678ef041bd873c9d3ca56e4800c6a4c23d2e211940ebbfe463b8f6",
}
PRE_BUDGET_RECOVERY_HASHES = {
    **PRE_FAILURE_HANDLING_HASHES,
    "selector_pair_srgc.py": "07c5bb63c0d2361b155d1c0ec87abed001ad95757c7dd482fc4ccb8e7e21bcb8",
}
PRE_SRGC_COST_RECOVERY_HASHES = {
    **PRE_FAILURE_HANDLING_HASHES,
    "selector_pair_srgc.py": "b17c2b804106e493937374b9b7c5225da78d3f91451c65acc559c4d3a8067dfa",
}
PRE_FREEZE_RETRY_HASHES = {
    **PRE_FAILURE_HANDLING_HASHES,
    "selector_pair_srgc.py": "ec7a7539366e897af9cb3831029e81f66fb33bd12ed25a2bc19675bd184cf3d3",
}
# Recomputing a frozen contrast on another node may differ in the last bits
# (BLAS kernels depend on the CPU). The selector must still match exactly.
CONTRAST_REL_TOL, CONTRAST_ABS_TOL = 1e-9, 1e-9
FREEZE_POLL_SECONDS = 60.
FREEZE_WAIT_SECONDS = 14400.


class FreezePending(worker.IncompletePairRun):
    """Only peer-owned states remain; no local measurement failed."""
RULE = {"name": "SR-GC", "threshold": 0., "negative": "cached", "nonnegative": "on_policy",
        "statistic": "mean_A_B(dot(validation_h, mean(on_h)-mean(cached_h)))",
        "references": "independent eight-response LOO4 candidate groups; disjoint A/B validation prompts",
        "scope": "one current-parent decision per Pair state, not a within-continuation switch trajectory"}
HERE = Path(__file__).resolve().parent


def choose(a, b):
    for value in (a, b):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("SR-GC requires two finite independent reference contrasts")
    d = a / 2 + b / 2
    return {"method": "SR-GC", "d_a": a, "d_b": b, "d": d,
            "selector": "cached" if d < 0 else "on_policy"}


def receipt(root, p):
    return {"schema": SCHEMA, "protocol_id": p["protocol_id"], "root": str(root),
            "pair_manifest_sha256": worker.base.digest(root / "pair.json"), "rule": RULE,
            "supersedes": "legacy Adaptive H regression and development-label barrier; fixed Pair arms unchanged",
            "code_sha256": {name: worker.base.digest(HERE / name) for name in (
                "selector_pair_srgc.py", "selector_pair_srgc_score.py")},
            "cost_policy": "meter new references; charge reused ranking cost once to adaptive; no budget or cost reset"}


def validate(root, p):
    path = root / RECEIPT
    expected = receipt(root, p)
    # These predecessors differ only in queue recovery, not the statistic.
    previous = {**expected, "code_sha256": PRE_FAILURE_HANDLING_HASHES}
    before_recovery = {**expected, "code_sha256": PRE_BUDGET_RECOVERY_HASHES}
    before_srgc_cost = {**expected, "code_sha256": PRE_SRGC_COST_RECOVERY_HASHES}
    before_freeze_retry = {**expected, "code_sha256": PRE_FREEZE_RETRY_HASHES}
    if path.is_symlink() or not path.is_file() or worker.core.read(path) not in (
            expected, previous, before_recovery, before_srgc_cost, before_freeze_retry):
        raise ValueError("SR-GC runtime receipt missing or changed")


def activate(root, p):
    from selector_pair_parallel import checked
    for path in (root / RECEIPT, root / "sr-gc", root / "test-decisions.json", root / "decisions"):
        checked(path)
    with worker.queue_lease(root / ".pair-barrier.lock"):
        if (root / RECEIPT).exists():
            validate(root, p)
            return
        barrier = root / "test-decisions.json"
        if barrier.exists() or barrier.is_symlink():
            raise ValueError("legacy Adaptive decisions already exist; preserved, not relabeled as SR-GC")
        if list((root / "decisions").glob("*/decision.json")):
            raise ValueError("partial legacy Adaptive decisions exist; preserved without replacement")
        for name in ("adaptive-on_policy", "adaptive-cached"):
            for out in (root / "branches" / name / "states").glob("*/points/*"):
                if worker.training_artifacts(out):
                    raise ValueError("legacy Adaptive artifacts exist; cannot retroactively certify SR-GC")
        worker.base.bind(root / RECEIPT, receipt(root, p))


def sets_and_ranking(directory, entry, devices, p):
    """Reuse only sealed, same-parent R scores; otherwise score R once here."""
    _, out, c, protocol, _ = entry
    base, core = worker.base, worker.core
    run = Path(c["source_run"])
    parent = run / f"policy_step_{c['config']['drift']}"
    contract = {"config": c["config"], "parent": str(parent),
                "adapter_sha256": base.digest(parent / "adapter_model.safetensors"),
                "prompts": str(run / "prompts.json"), "prompts_sha256": base.digest(run / "prompts.json"),
                "sampling_seed": 701000003 + c["config"]["seed"] * 1000003 + c["config"]["drift"] * 7919,
                "contract_sha256": base.digest(out / "contract.json"), "protocol_sha256": core.fingerprint(protocol)}
    selected = directory / "ranking-source.json"
    if not selected.exists():
        candidates = [out / arm / "fresh-r" for arm in ("selection_full", "selection_reduced")]
        source = next((path for path in candidates if (path / "selected.sha256.json").is_file()
                       and core.read(path / "scoring.json") == contract), directory / "ranking")
        # Record reuse before new work. A restart cannot choose a later source.
        reused = source != directory / "ranking"
        inherited_cost = ranking_cost(source.parent) if reused else 0.
        base.bind(selected, {"path": str(source), "reused": reused, "reused_gpu_seconds": inherited_cost})
    source_record = core.read(selected)
    source = Path(source_record["path"])
    allowed = {directory / "ranking", *(out / arm / "fresh-r" for arm in ("selection_full", "selection_reduced"))}
    if source not in allowed or source.is_symlink():
        raise ValueError("SR-GC ranking escaped its parent-state namespace")
    base.bind(source / "scoring.json", contract)
    for stage in ("validation", "candidate"):
        commands = [([sys.executable, str(base.ROOT / "src/selection_switch_score.py"),
                      "--root", str(source), "--stage", stage, "--shard", str(i)], devices[i])
                    for i in range(4) if not (source / f"{stage}-{i}.done.json").exists()]
        if commands:
            if source_record["reused"]:
                raise ValueError("sealed SR-GC ranking is incomplete; refusing to repair another arm's scores")
            paid(directory, p, "sr-gc-r-" + stage, commands, worker.environment(c))
        worker.switch.scoring.merge(source, stage)
    if core.read(source / "selected.sha256.json") != {"sha256": base.digest(source / "selected.json")}:
        raise ValueError("SR-GC ranking receipt changed")
    on = core.read(source / "selected.json")["indices"]
    cached = worker.switch.cached_selection(run / "rollouts_behavior_train.jsonl", prompts=c["n"],
                                            responses=8, seed=c["config"]["seed"], selector="difficulty")
    if len(on) != cached["k"] or len(set(on)) != len(on):
        raise ValueError("SR-GC subset sizes differ")
    return {"on_policy": on, "cached": cached["indices"]}, contract, source_record, cached


def ranking_cost(directory):
    """Only closed R-scoring events, never a future training outcome or free reuse."""
    import json
    events = [json.loads(line) for line in (directory / "cost.jsonl").read_text().splitlines() if line.strip()]
    events = [row for row in events if row.get("phase", "").startswith("fresh-r-")]
    closed = worker.pair.finished_events(events)
    if not closed or not {"fresh-r-validation", "fresh-r-candidate"}.issubset({b["phase"] for _, b in closed}):
        raise ValueError("reused SR-GC ranking has no complete scoring-cost evidence")
    return sum(b["allocated_gpu_seconds"] for _, b in closed)


def paid(directory, p, phase, commands, env):
    remaining = worker.core.number(p["training_cap_gpu_seconds"], "SR-GC measurement cap", 0.) - worker.base.spent(directory)
    if remaining <= 0:
        raise ValueError("SR-GC measurement allocation exhausted; no automatic budget increase")
    worker.base.meter(directory, phase, p["gpu_type"], commands=commands, env=env,
                      timeout=remaining / 4, ledger="deployment")


def reference_contrast(directory, sets):
    import numpy as np
    halves = []
    for half in ("a", "b"):
        candidates = score.projections(directory, "candidate-" + half)
        direction = np.mean(list(score.projections(directory, "validation-" + half).values()), axis=0)
        delta = (np.mean([candidates[i] for i in sets["on_policy"]], axis=0)
                 - np.mean([candidates[i] for i in sets["cached"]], axis=0))
        halves.append(float(delta @ direction))
    return choose(*halves)


def measure(directory, identity, entry, devices, p):
    base, core = worker.base, worker.core
    _, out, c, _, _ = entry
    if devices is None or len(devices) != 4:
        raise ValueError("SR-GC missing measurements require the existing four-GPU allocation")
    sets, ranking, source, cached = sets_and_ranking(directory, entry, devices, p)
    reference = {**ranking, "sets": sets, "state_id": identity, "rule": RULE,
                 "cache_sha256": cached["cache_sha256"]}
    base.bind(directory / "reference.json", reference)
    for stage in score.STAGES:
        commands = [([sys.executable, str(HERE / "selector_pair_srgc_score.py"), "--root", str(directory),
                      "--stage", stage, "--shard", str(i)], devices[i])
                    for i in range(4) if not (directory / f"{stage}-{i}.done.json").exists()]
        if commands:
            paid(directory, p, "sr-gc-" + stage, commands, worker.environment(c))
    # Fail closed before appending: a new row after an open event would make the
    # ledger unrecoverable, whereas a trailing open event is closed by recovery.
    base.spent(directory)
    contrast = base.meter(directory, "sr-gc-aggregate", p["gpu_type"],
                          action=lambda: reference_contrast(directory, sets), ledger="deployment")
    value = {**contrast, "sets": sets, "state_id": identity,
             "seed": c["config"]["seed"], "step": c["config"]["drift"], "protocol_id": p["protocol_id"],
             "reference_sha256": base.digest(directory / "reference.json"),
             "ranking_selected_sha256": base.digest(Path(source["path"]) / "selected.json"),
             "cache_sha256": cached["cache_sha256"],
             "reference_shards": {f"{stage}-{i}.json": base.digest(directory / f"{stage}-{i}.json")
                                  for stage in score.STAGES for i in range(4)},
             "new_measurement_gpu_seconds": base.spent(directory),
             "reused_ranking_gpu_seconds": source["reused_gpu_seconds"],
             "diagnosis_gpu_seconds": base.spent(directory) + source["reused_gpu_seconds"]}
    base.bind(directory / "decision.json", value)
    return value


def validate_choice(root, p, seed, step):
    base, core = worker.base, worker.core
    directory = root / "sr-gc" / f"s{seed}-t{step}"
    value = core.read(directory / "decision.json")
    if (value.get("protocol_id") != p["protocol_id"] or value.get("seed") != seed or value.get("step") != step
            or any(value.get(key) != item for key, item in choose(value.get("d_a"), value.get("d_b")).items())):
        raise ValueError("SR-GC decision differs from its fixed rule or state")
    if (value["reference_sha256"] != base.digest(directory / "reference.json")
            or value["new_measurement_gpu_seconds"] != base.spent(directory)):
        raise ValueError("SR-GC reference or measurement costs changed")
    reference = core.read(directory / "reference.json")
    if (reference["state_id"] != value["state_id"] or reference["sets"] != value["sets"]
            or reference["rule"] != RULE
            or base.digest(Path(reference["parent"]) / "adapter_model.safetensors") != reference["adapter_sha256"]
            or base.digest(Path(reference["prompts"])) != reference["prompts_sha256"]
            or base.digest(Path(reference["parent"]).parent / "rollouts_behavior_train.jsonl") != value["cache_sha256"]):
        raise ValueError("SR-GC parent-policy inputs changed")
    source = core.read(directory / "ranking-source.json")
    if (value["ranking_selected_sha256"] != base.digest(Path(source["path"]) / "selected.json")
            or source["reused_gpu_seconds"] != value["reused_ranking_gpu_seconds"]
            or value["diagnosis_gpu_seconds"] != value["new_measurement_gpu_seconds"] + source["reused_gpu_seconds"]):
        raise ValueError("SR-GC ranking or total cost binding changed")
    # The source arm may append its own later scoring events; the ranking SR-GC used
    # is pinned by selected.json above. Only a smaller ledger indicates tampering.
    if source["reused"] and ranking_cost(Path(source["path"]).parent) < source["reused_gpu_seconds"] - 1e-6:
        raise ValueError("reused SR-GC scoring costs decreased")
    expected = {f"{stage}-{i}.json" for stage in score.STAGES for i in range(4)}
    if set(value["reference_shards"]) != expected or any(base.digest(directory / name) != digest
                                                       for name, digest in value["reference_shards"].items()):
        raise ValueError("SR-GC reference projections changed")
    recomputed = reference_contrast(directory, value["sets"])
    if (recomputed["method"] != value["method"] or recomputed["selector"] != value["selector"]
            or any(not math.isclose(recomputed[key], value[key], rel_tol=CONTRAST_REL_TOL,
                                    abs_tol=CONTRAST_ABS_TOL) for key in ("d_a", "d_b", "d"))):
        raise ValueError("SR-GC decision does not match its independent projections")
    return value


def decisions(root, p):
    validate(root, p)
    barrier = worker.core.read(root / "test-decisions.json")
    keys = {f"s{s}-t{t}" for s in worker.pair.TEST_SEEDS for t in worker.pair.STEPS}
    if (barrier.get("schema") != SCHEMA or barrier.get("protocol_id") != p["protocol_id"]
            or barrier.get("runtime_sha256") != worker.base.digest(root / RECEIPT)
            or set(barrier.get("decisions", {})) != keys):
        raise ValueError("SR-GC decision barrier changed")
    values = {}
    for seed in worker.pair.TEST_SEEDS:
        for step in worker.pair.STEPS:
            name = f"s{seed}-t{step}"
            if worker.base.digest(root / "sr-gc" / name / "decision.json") != barrier["decisions"][name]:
                raise ValueError("frozen SR-GC decision changed")
            values[name] = validate_choice(root, p, seed, step)
    return values


def freeze(root, p, devices):
    validate(root, p)
    import selector_pair_srgc_cost_recovery as recovery
    recovery.recover(root, p)
    if (root / "test-decisions.json").exists():
        return decisions(root, p)
    pending = []
    failures = []
    for seed in worker.pair.TEST_SEEDS:
        for step in worker.pair.STEPS:
            name = f"s{seed}-t{step}"
            directory = root / "sr-gc" / name
            from selector_pair_parallel import checked
            checked(directory)
            try:
                with worker.pair_lease(directory / ".decision.lock"):
                    with worker.pair_lease(root / "test" / name / ".prepare-state.lock"):
                        identity, entries = worker.verify_pair(root, seed, step)
                    if not (directory / "decision.json").exists():
                        worker.core.atomic_json(directory / "measurement-status.json", {
                            "protocol_id": p["protocol_id"], "state": "RUN", "updated": time.time()})
                        measure(directory, identity, entries["on_policy"], devices, p)
                    value = validate_choice(root, p, seed, step)
                    if value["state_id"] != identity:
                        raise ValueError("SR-GC decision belongs to another parent policy")
                    worker.core.atomic_json(directory / "measurement-status.json", {
                        "protocol_id": p["protocol_id"], "state": "DONE", "updated": time.time()})
                    print(f"[SR-GC] {name}: D={value['d']:.6g}, selector={value['selector']}", flush=True)
            except worker.PairLockBusy:
                pending.append(name)
            except (ValueError, OSError, RuntimeError) as exc:
                failures.append(f"{name}: {exc}")
                worker.core.atomic_json(directory / "measurement-status.json", {
                    "protocol_id": p["protocol_id"], "state": "BLOCKED", "updated": time.time(), "error": str(exc)})
                print(f"[SR-GC unavailable] {name}: {exc}; checking other states", flush=True)
    if pending or failures:
        kind = worker.IncompletePairRun if failures else FreezePending
        raise kind("SR-GC pending measurements: " + "; ".join([
            *(f"{name} owned by peer" for name in pending), *failures]))
    with worker.queue_lease(root / ".pair-barrier.lock"):
        worker.base.bind(root / "test-decisions.json", {"schema": SCHEMA, "protocol_id": p["protocol_id"],
            "runtime_sha256": worker.base.digest(root / RECEIPT),
            "decisions": {f"s{s}-t{t}": worker.base.digest(root / "sr-gc" / f"s{s}-t{t}" / "decision.json")
                          for s in worker.pair.TEST_SEEDS for t in worker.pair.STEPS}})
    return decisions(root, p)


@contextlib.contextmanager
def activated(root, p, devices, *, initialize=True):
    if initialize:
        activate(root, p)
    else:
        validate(root, p)
    old_decisions, old_select = worker.decisions, worker.switch.runtime.select_once

    def select(out, c, protocol, arm, choice, env, allocated):
        branch = worker.switch.switch_root(out)
        if branch not in {root / "branches/adaptive-on_policy", root / "branches/adaptive-cached"}:
            return old_select(out, c, protocol, arm, choice, env, allocated)
        value = decisions(root, p)[f"s{c['config']['seed']}-t{c['config']['drift']}"]
        if branch.name != "adaptive-" + value["selector"] or arm != "selection_full":
            raise ValueError("Adaptive execution differs from frozen SR-GC choice")
        return value["sets"][value["selector"]]

    worker.decisions, worker.switch.runtime.select_once = decisions, select
    try:
        import selector_pair_budget_recovery as recovery
        with recovery.activated(root):
            yield
    finally:
        worker.decisions, worker.switch.runtime.select_once = old_decisions, old_select


def attempt_freeze(root, p, devices):
    try:
        freeze(root, p, devices)
    except (worker.IncompletePairRun, worker.PairWaitTimeout, ValueError, OSError) as exc:
        return exc
    return None


def wait_for_freeze(root, p, devices, *, limit=None, poll=None, sleep=None, clock=None):
    """Retry while peers own the remaining states, so the barrier does not depend on
    one worker seeing all six states free in a single pass. Real failures stop at once."""
    if limit is None:
        limit = float(os.environ.get("PAIR_SRGC_FREEZE_WAIT_SECONDS", FREEZE_WAIT_SECONDS))
    poll = FREEZE_POLL_SECONDS if poll is None else poll
    sleep, clock = sleep or time.sleep, clock or time.monotonic
    deadline = clock() + limit
    while True:
        failure = attempt_freeze(root, p, devices)
        if not isinstance(failure, FreezePending) or clock() >= deadline:
            return failure
        print(f"[SR-GC] waiting for peer-owned measurements: {failure}", flush=True)
        sleep(poll)


def run_stages(root, p, devices, command):
    """No fit, development-label barrier, or target-attainment prerequisite."""
    if command == "develop":
        return worker.distributed_stage(root, p, devices, "development")
    failure = attempt_freeze(root, p, devices)
    if command == "freeze":
        if isinstance(failure, FreezePending):
            failure = wait_for_freeze(root, p, devices)
        if failure:
            raise failure
        return
    if failure:
        # A reference failure must not strand independent fixed experiments.
        import selector_pair_parallel as parallel
        controls = parallel.FixedQueue(root, p, devices)
        while controls.step():
            pass
        if isinstance(failure, FreezePending):
            failure = wait_for_freeze(root, p, devices)
    if not failure and command in {"run", "test"}:
        try:
            worker.distributed_stage(root, p, devices, "test")
        except (worker.IncompletePairRun, worker.PairWaitTimeout) as exc:
            failure = exc
    if command == "run":
        worker.distributed_stage(root, p, devices, "development")
    if failure:
        raise failure
    with worker.queue_lease(root / ".pair-barrier.lock"):
        with worker.completed_state_leases(root, ("development", "test"), require_complete=False):
            report(root, p)


def report(root, p):
    worker.report(root, p)
    path = root / "report.json"
    value = worker.core.read(path)
    value.update(adaptive_method="SR-GC", adaptive_rule=RULE,
                 adaptive_runtime_sha256=worker.base.digest(root / RECEIPT),
                 legacy_offline_fit_cost=value.pop("offline_fit_cost", None), offline_fit_cost=None)
    worker.core.atomic_json(path, value)
