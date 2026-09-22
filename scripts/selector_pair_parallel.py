"""Run preregistered held-out controls while development peers own their work."""
from __future__ import annotations

import contextlib
from pathlib import Path
import time
import uuid

import selector_pair_gpu as worker

RECEIPT = "pair-parallel-controls-runtime.json"
PRE_NONATTAINMENT_HASHES = {
    "selector_pair_parallel.py": "cdc531e6a7210ac0cbe920324d6551d8e3cc0c383bcb680c3a0f792e1758b3fc",
    "queue_selector_pair_gpu.py": "7c833839ccb31e77a259831c77b31d4de52480bc64ee59fa509a07f0b487da3c",
}
FIXED_CONTROLS = (("on_policy", "selection_full"), ("cached", "selection_full"),
                  ("on_policy", "random_full"))


def checked(path):
    path = Path(path)
    try:
        if path.resolve() != path:
            raise ValueError(f"parallel Pair scheduling refuses a symlinked path: {path}")
    except RuntimeError as exc:
        raise ValueError(f"parallel Pair scheduling refuses a cyclic path: {path}") from exc
    return path


def fixed_tasks():
    return [{"seed": seed, "step": step, "branch": branch, "arm": arm}
            for seed in worker.pair.TEST_SEEDS for step in worker.pair.STEPS
            for branch, arm in FIXED_CONTROLS]


def receipt_value(root, protocol):
    root = Path(root).resolve()
    scripts = Path(__file__).resolve().parent
    return {
        "schema": "offpolicy-selector-pair/parallel-fixed-controls-v1",
        "root": str(root), "protocol_id": protocol["protocol_id"],
        "pair_manifest_sha256": worker.base.digest(root / "pair.json"),
        "frozen_code_hashes": protocol["code_hashes"],
        "runtime_code_hashes": {name: worker.base.digest(scripts / name) for name in (
            "selector_pair_parallel.py", "queue_selector_pair_gpu.py")},
        "fixed_controls": fixed_tasks(),
        "schedule": "development first; independent fixed held-out controls during development waits; adaptive only after frozen decisions",
        "fit_policy": "fit and choose from development labels and prefix diagnostics only; never inspect held-out outcomes",
        "cost_policy": "preserve every budget, cost, checkpoint, result and active lease; no repeated completed training",
    }


def validate_protocol(root, protocol):
    if any(name not in worker.BRANCHES for name in protocol.get("branch_manifests", {})):
        raise ValueError("parallel Pair protocol has an unknown branch")
    for path in (root / "pair.json", *(root / "branches" / name / "switch.json"
                                      for name in protocol.get("branch_manifests", {}))):
        if not checked(path).is_file():
            raise ValueError(f"parallel Pair metadata is not a regular file: {path}")
    if worker.manifest(root, bind_runtime=False) != protocol:
        raise ValueError("parallel Pair protocol changed")


def validate_receipt(root, protocol):
    """Read-only schedule authorization for workers, handoff and status."""
    root = Path(root).resolve()
    validate_protocol(root, protocol)
    for name in worker.BRANCHES:
        checked(root / "branches" / name)
    path = checked(root / RECEIPT)
    expected = receipt_value(root, protocol)
    previous = {**expected, "runtime_code_hashes": PRE_NONATTAINMENT_HASHES}
    if not path.is_file() or worker.core.read(path) not in (expected, previous):
        raise ValueError(f"parallel Pair schedule receipt missing or changed: {path}")


def validate_activation(root, protocol):
    """Never retroactively authorize previously unregistered held-out work."""
    root = Path(root).resolve()
    validate_protocol(root, protocol)
    path = root / RECEIPT
    if path.exists() or path.is_symlink():
        validate_receipt(root, protocol)
        return
    for name in worker.BRANCHES:
        checked(root / "branches" / name)
        for seed in worker.pair.TEST_SEEDS:
            for step in worker.pair.STEPS:
                validate_artifact_paths(root / "branches" / name / "states" / f"s{seed}-t{step}" / "points" / f"view-{step}")
    barrier = root / "test-decisions.json"
    if barrier.exists() or barrier.is_symlink():
        if not checked(barrier).is_file() or not checked(root / "model.json").is_file():
            raise ValueError("parallel Pair decision barrier metadata is not regular")
        for seed in worker.pair.TEST_SEEDS:
            for step in worker.pair.STEPS:
                if not checked(root / "decisions" / f"s{seed}-t{step}" / "decision.json").is_file():
                    raise ValueError("parallel Pair decision metadata is not regular")
        worker.decisions(root, protocol)
        return
    for name in worker.BRANCHES:
        checked(root / "branches" / name)
        for seed in worker.pair.TEST_SEEDS:
            for step in worker.pair.STEPS:
                out = checked(root / "branches" / name / "states" / f"s{seed}-t{step}" / "points" / f"view-{step}")
                if worker.training_artifacts(out):
                    raise ValueError("held-out work predates the parallel Pair schedule receipt; preserved without authorization")


def validate_artifact_paths(out):
    checked(out)
    checked(out / "selector-work")
    for arm in worker.switch.rule.TEST_ARMS:
        checked(out / arm)
        checked(out / "selector-work" / arm)
        for name in ("execution.json", "result.json", "policy", "cached-select", "cost.jsonl"):
            checked(out / arm / name)


def fixed_arms(root, out):
    out = Path(out)
    checked(out)
    for task in fixed_tasks():
        expected = root / "branches" / task["branch"] / "states" / f"s{task['seed']}-t{task['step']}" / "points" / f"view-{task['step']}"
        if out == expected:
            return {arm for branch, arm in FIXED_CONTROLS if branch == task["branch"]}
    return set()


@contextlib.contextmanager
def allow_fixed_before_decisions(root, protocol):
    """Relax only the registered fixed-arm pre-freeze artifact check."""
    root = Path(root).resolve()
    validate_receipt(root, protocol)
    original = worker.training_artifacts

    def artifacts(out):
        validate_receipt(root, protocol)
        validate_artifact_paths(Path(out))
        allowed = fixed_arms(root, out)
        ignored = {str(Path(out) / arm / name) for arm in allowed for name in (
            "execution.json", "result.json", "policy", "cached-select", "cost.jsonl")}
        ignored.update(str(Path(out) / "selector-work" / arm) for arm in allowed)
        return [path for path in original(out) if path not in ignored]

    worker.training_artifacts = artifacts
    try:
        yield
    finally:
        worker.training_artifacts = original


def signature(paths):
    result = []
    for path in paths:
        try:
            stat = path.stat()
            result.append((stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
        except FileNotFoundError:
            result.append(None)
    return tuple(result)


class FixedQueue:
    """One controller's bounded fixed-arm attempts, using the original leases."""

    def __init__(self, root, protocol, devices):
        self.root, self.protocol, self.devices = Path(root).resolve(), protocol, devices
        self.failed = {}
        self.identity = uuid.uuid4().hex
        self.record_path = self.root / "queue-workers" / f"{self.identity}.json"

    def record(self, state, task=None):
        import os
        worker.core.atomic_json(self.record_path, {
            "worker": self.identity, "host": worker.base.node_id(), "pid": os.getpid(),
            "protocol_id": self.protocol["protocol_id"], "stage": "test", "state": state,
            "task": task, "updated": time.time(), "schedule": "parallel-fixed-controls",
            "total_states": 6, "total_branches": 24, "fixed_control_branches": 18,
            "failures": list(self.failed.values()),
        })

    def step(self):
        """Try at most one executable control before checking development again."""
        validate_receipt(self.root, self.protocol)
        try:
            for seed in worker.pair.TEST_SEEDS:
                for step in worker.pair.STEPS:
                    folder = self.root / "test" / f"s{seed}-t{step}"
                    try:
                        with worker.pair_lease(checked(folder / ".state.lock"), shared=True):
                            if (folder / "result.json").exists():
                                continue
                            with worker.pair_lease(checked(folder / ".prepare-state.lock")):
                                identity, entries = worker.verify_pair(self.root, seed, step)
                            for name, arm in FIXED_CONTROLS:
                                entry = entries[name]
                                out = Path(entry[1])
                                if arm not in fixed_arms(self.root, out) or Path(entry[0]) != self.root / "branches" / name:
                                    raise ValueError("fixed Pair task escaped its registered namespace")
                                validate_artifact_paths(out)
                                directory = checked(out / arm)
                                receipt = checked(folder / "queue-branches" / f"{name}--{arm}.json")
                                task = f"test/s{seed}-t{step}/{name}/{arm}"
                                payloads = (receipt, directory / "result.json", directory / "curve.json")
                                if task in self.failed and not all(path.is_file() for path in payloads[1:]):
                                    continue
                                attempted = False
                                try:
                                    with worker.pair_lease(checked(receipt.with_suffix(".lock"))):
                                        if receipt.exists() or all(path.is_file() for path in payloads[1:]):
                                            worker.base.bind(receipt, worker.branch_receipt(self.protocol, identity, name, arm, entry))
                                            self.failed.pop(task, None)
                                            continue
                                        checked(directory / ".task.lock")
                                        self.record("RUN", task)
                                        print(f"[RUN] host={worker.base.node_id()} {task} (fixed control; development pending)", flush=True)
                                        progress_paths = (*payloads, directory / "cost.jsonl", directory / "curve/cost.jsonl",
                                                          out / "curve-parent/cost.jsonl")
                                        before = signature(progress_paths)
                                        try:
                                            attempted = True
                                            failure = worker.attempt_branch(self.root, self.protocol, entry, arm, self.devices)
                                        except worker.PairWorkPending:
                                            if signature(progress_paths) != before:
                                                return True
                                            continue
                                        if failure:
                                            self.failed[task] = failure
                                            return True
                                        worker.base.bind(receipt, worker.branch_receipt(self.protocol, identity, name, arm, entry))
                                        return True
                                except worker.PairLockBusy as exc:
                                    if exc.path != receipt.with_suffix(".lock"):
                                        raise
                                except worker.NodeAdmissionError:
                                    raise
                                except (OSError, ValueError, RuntimeError) as exc:
                                    self.failed[task] = {"task": task, "error": f"{type(exc).__name__}: {exc}"}
                                    print(f"[WAIT] {task}: {exc}; trying other fixed controls", flush=True)
                                    if attempted:
                                        return True
                    except worker.PairLockBusy as exc:
                        if exc.path not in {folder / ".state.lock", folder / ".prepare-state.lock"}:
                            raise
                    except worker.NodeAdmissionError:
                        raise
                    except (OSError, ValueError, RuntimeError) as exc:
                        print(f"[WAIT] test/s{seed}-t{step}: {exc}; trying other fixed states", flush=True)
            return False
        finally:
            self.record("WAIT", "fixed-control pass yielded; no task owned")


class _ControlProgress(BaseException):
    pass


class AdaptiveUnavailable(ValueError):
    """A terminal model dependency failure, not a live worker's lock."""


def run_distributed(root, protocol, devices, command, original):
    """Keep the frozen scheduler, inserting fixed work only at its idle point."""
    root = Path(root).resolve()
    validate_receipt(root, protocol)
    original_stage, original_progress, original_attempt = worker.distributed_stage, worker.pair_progress, worker.attempt_branch
    original_freeze, original_verify, original_atomic = worker.freeze, worker.verify_pair, worker.core.atomic_json
    original_fit = worker.fit
    controls = FixedQueue(root, protocol, devices)
    development_failures = {}
    observation = {}

    def fit(candidate, p):
        try:
            return original_fit(candidate, p)
        except ValueError as exc:
            if str(exc) != "unreached or ineligible target: no point label; do not fit on successful states only":
                raise
            raise AdaptiveUnavailable(
                "Adaptive BLOCKED: a development target was not reached or was already met at the parent; "
                "the legacy H model cannot be fitted. Fixed Pair controls remain independent. "
                "No target, result, cost or active lock was changed."
            ) from exc

    def atomic(path, value):
        original_atomic(path, value)
        if (Path(path).parent == root / "queue-workers" and isinstance(value, dict)
                and value.get("stage") == "development" and value.get("protocol_id") == protocol["protocol_id"]):
            observation["path"], observation["value"] = Path(path), value

    def fixed_pass():
        if observation.get("value", {}).get("state") == "RUN":
            value = {**observation["value"], "state": "WAIT", "task": "development pass yielded; no task owned", "updated": time.time()}
            atomic(observation["path"], value)
        return controls.step()

    def freeze(candidate, p):
        def prepare(observed_root, seed, step):
            if Path(observed_root).resolve() == root and seed in worker.pair.TEST_SEEDS:
                lock = checked(root / "test" / f"s{seed}-t{step}" / ".prepare-state.lock")
                with worker.queue_lease(lock):
                    return original_verify(observed_root, seed, step)
            return original_verify(observed_root, seed, step)

        worker.verify_pair = prepare
        try:
            return original_freeze(candidate, p)
        finally:
            worker.verify_pair = original_verify

    def attempt(candidate, p, entry, arm, allocated):
        key = (str(entry[1]), arm)
        development = Path(candidate).resolve() == root and entry[2]["config"]["seed"] in worker.pair.DEV_SEEDS
        if development and key in development_failures:
            return development_failures[key]
        failure = original_attempt(candidate, p, entry, arm, allocated)
        if development and failure:
            development_failures[key] = failure
        return failure

    def stage(candidate, p, allocated, role, **kwargs):
        if command != "run" or role != "development" or Path(candidate).resolve() != root:
            return original_stage(candidate, p, allocated, role, **kwargs)

        def progress(observed_root, states=None):
            if states is not None and Path(observed_root).resolve() == root and fixed_pass():
                raise _ControlProgress()
            return original_progress(observed_root, states)

        worker.pair_progress = progress
        try:
            while True:
                try:
                    return original_stage(candidate, p, allocated, role, **kwargs)
                except _ControlProgress:
                    continue
                except (worker.IncompletePairRun, worker.PairWaitTimeout):
                    if not fixed_pass():
                        raise
        finally:
            worker.pair_progress = original_progress

    worker.distributed_stage, worker.attempt_branch = stage, attempt
    worker.freeze, worker.core.atomic_json, worker.fit = freeze, atomic, fit
    try:
        with allow_fixed_before_decisions(root, protocol):
            try:
                return original(root, protocol, devices, command)
            except AdaptiveUnavailable as exc:
                if command != "run":
                    raise
                print(f"[BLOCKED] {exc}; continuing all available fixed controls", flush=True)
                # The original fit barrier has unwound. Fixed GPU work must
                # never run while holding the global model/decision lock.
                while fixed_pass():
                    pass
                controls.record("BLOCKED", str(exc))
                raise
    finally:
        worker.distributed_stage, worker.pair_progress, worker.attempt_branch = original_stage, original_progress, original_attempt
        worker.freeze, worker.verify_pair, worker.core.atomic_json = original_freeze, original_verify, original_atomic
        worker.fit = original_fit
