"""Matched-state G/D curves and an independently executed, held-out decision.

Four private switch roots reuse certified prefixes, never continuation results.
The legacy random gate is not fitted or used. Its frozen learner, selection,
evaluation, atomic publication, and failed-attempt ledgers remain unchanged.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import io
import json
import os
from pathlib import Path
import statistics
import sys
import time
import uuid
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
# Reviewed operational predecessors only. Preserve manifests, protocol IDs,
# labels and decisions; CPU scheduling changes get a separate runtime receipt.
PRE_BOOTSTRAP_CODE = "3ad11e06bc7670012c91898b6d4e09802eab195422f4a62919ffaf4e95725f9b"
PRE_DEFAULTS_CODE = "b589d47572403fe0c217ac3e2f925e69906db54822f1f29c6dca9dd1fba3b98b"
PRE_RESOURCES_CODE = "d5d354a91ec95ae5d941c619a5a50da10e1606072a36dbe06e83d974aed41ec5"
# Exact released resource/shared runtimes (045dcc1, 6345433, 2444e51, cff7832).
# Their shared-file changes require Switch's independent reviewed hash pins;
# pair selectors, pair trainer, curve trainer and branch protocols stay frozen.
PRE_BUDGET_STOP_EVALUATION_CODE = "bfe7b00d57a365d9ddd8936422a723c86938e2feb1f82ae5c72d648a42ce7c30"
PRE_PAIR_OPERATIONS_CODE = "ad310eefce6b1d6e9d7f9ed634b0a121b491fb496899870363e2affb7f42adb7"
PRE_PAIR_LOCK_OBSERVATION_CODE = "89983f762d8fe1ea0af35bda9088e48c6ff2335d6d2c22192471377b196d0798"
# Exact released 1aebf1d code map, before distributed task leases.
PRE_PAIR_DISTRIBUTED_CODE = "76dd34fc37746ad2a1be05a8e29c7c13c620b2d281919acbea0b28119656b201"
# Exact released 5fd2410 code map, before bounded waits and owner diagnostics.
PRE_PAIR_WAIT_GUARD_CODE = "2a6c4dcd2fb062159f3212efb7d19f5898774e3f0d18a90953e76b6e3cf309a6"
# Exact 9be50a8 pair map before the shared MBPP quarantine compatibility patch.
PRE_SHARED_MBPP_QUARANTINE_CODE = "0894fdfe1edb03163abc02589bfd941dc8ffc5e41f000b5ef6981be74c791b5d"
PRE_PAIR_STATUS_CODE = "ad4d1718999848103a577e2efc2ce6b1352a9c5d7f76fccdc875924c15cd57f3"
PRE_PAIR_BRANCH_QUEUE_CODE = "456af840a1bd6f184078f9cee6b30a2c7554523fa611b1156e53ce6f49400c28"
PRE_SHARED_RUNTIME_CODES = {
    PRE_PAIR_BRANCH_QUEUE_CODE,
    PRE_PAIR_STATUS_CODE,
    PRE_BUDGET_STOP_EVALUATION_CODE,
    PRE_PAIR_OPERATIONS_CODE,
    PRE_PAIR_LOCK_OBSERVATION_CODE,
    PRE_PAIR_DISTRIBUTED_CODE,
    PRE_PAIR_WAIT_GUARD_CODE,
    PRE_SHARED_MBPP_QUARANTINE_CODE,
    "9eab1b016f5f897a4bd3b85998a25b1bc724b8bf3383ef9e6cfe2ba43f9a6d67",
    "cec86006408b80d7901f3f44a3b113d702c860e6c4a84a40cd2422e6438ef27a",
    "b5dfeae35bc95922893636bc5ad1c6d801e68648f1d60907353a016e7c0c1738",
    "1cf6c9ff347ec6db413c5603170a2e64914a85daa3a06b55bf4705d8367e53d7",
}
STARTUP_FILES = {"src/selector_pair_gpu.py", "scripts/run_selector_pair.sh"}
RUN_DEFAULTS = {"target_reward": .35, "budget_gpu_seconds": 87120.}
CPU_ENV = {**dict.fromkeys(("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "OPENBLAS_DEFAULT_NUM_THREADS", "GOTO_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS", "NUMEXPR_MAX_THREADS", "OMP_THREAD_LIMIT", "RAYON_NUM_THREADS"), "1"),
    "TOKENIZERS_PARALLELISM": "false"}


class NodeAdmissionError(RuntimeError):
    """Do not dispatch more work on a node without successful GPU admission."""


class IncompletePairRun(ValueError):
    """A pass kept independent work moving but cannot advance the fit barrier."""


class PairWaitTimeout(ValueError):
    """Stop this waiting invocation, never an existing worker or its lease."""


class PairWorkPending(Exception):
    """A peer owns shared curve evaluation; retry publication without GPU admission."""


class PairLockBusy(ValueError):
    def __init__(self, path):
        self.path = Path(path)
        super().__init__(f"pair lock busy: {path}; another controller/worker holds it. "
                         "Do not delete the lock or reset results.")


def admission_probe(root):
    # The same tested four-rank NCCL/DDP probe used by Switch/MBPP. Import here
    # so read-only status and CPU analysis never initialize CUDA.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "pair_nccl_preflight", base.ROOT / "scripts/selection_nccl_preflight.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.preflight(root)


def admit_node(root, p):
    try:
        devices = switch.admitted_devices(p)
        overrides = admission_probe(root)
    except (OSError, ValueError, RuntimeError) as exc:
        raise NodeAdmissionError(f"pair node admission failed; no further work dispatched: {exc}") from exc
    os.environ.update(overrides)
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    return devices


def configure_cpu_runtime():
    # Also cover direct Python invocation, before importing torch/numpy/tokenizers.
    # Do not change GPU concurrency, generation batches, samples or seeds.
    os.environ.update(CPU_ENV)


def resource_diagnostics():
    """Best-effort, read-only limits; never dump the user's full environment."""
    import resource
    value = {"cpu_environment": {key: os.environ.get(key) for key in CPU_ENV},
             "max_user_processes": resource.getrlimit(resource.RLIMIT_NPROC)}
    try:
        value["controller_threads"] = next(line.split(":", 1)[1].strip()
            for line in Path("/proc/self/status").read_text().splitlines() if line.startswith("Threads:"))
    except (OSError, StopIteration):
        pass
    # Common cgroup v2 and v1 layouts, including namespace-rooted containers.
    roots = [Path("/sys/fs/cgroup"), Path("/sys/fs/cgroup/pids")]
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            _, controllers, relative = line.split(":", 2)
            if not controllers or "pids" in controllers.split(","):
                mount = Path("/sys/fs/cgroup") / ("pids" if controllers else "")
                candidate = (mount / relative.lstrip("/")).resolve()
                if candidate.is_relative_to(mount):
                    roots.extend([candidate, *[p for p in candidate.parents if p.is_relative_to(mount)]])
    except (OSError, ValueError):
        pass
    value["pids_limits"] = {}
    for root in dict.fromkeys(roots):
        try:
            value["pids_limits"][str(root)] = {
                name: (root / f"pids.{name}").read_text().strip() for name in ("current", "max")}
        except OSError:
            pass
    print("[pair-resources] " + json.dumps(value, sort_keys=True), file=sys.stderr, flush=True)


@contextlib.contextmanager
def pair_lease(path, *, shared=False):
    # Catch acquisition only: EAGAIN from a subprocess inside the lease is not
    # evidence of lock contention and must retain its original traceback.
    with contextlib.ExitStack() as stack:
        try:
            if shared:
                path.parent.mkdir(parents=True, exist_ok=True)
                handle = stack.enter_context(path.open("a+"))
                fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
            else:
                stack.enter_context(base.lease(path))
        except BlockingIOError as exc:
            raise PairLockBusy(path) from exc
        yield


def pair_progress(root, states=None):
    """Read bounded, relevant metadata; no ledger scans or GPU queries."""
    names = states if states is not None else [f"s{s}-t{t}" for s in (*pair.DEV_SEEDS, *pair.TEST_SEEDS) for t in pair.STEPS]
    pending = [(root / "branches" / branch / "states" / name, 0) for branch in BRANCHES for name in names]
    if states is None:
        pending.append((root / "node-preflight", 0))
    observations, examined = [], 0
    deadline = time.monotonic() + 2.
    resolved = root.resolve()
    while pending and examined < 2048 and len(observations) < 256 and time.monotonic() < deadline:
        directory, depth = pending.pop()
        try:
            if not directory.resolve().is_relative_to(resolved):
                continue
            with os.scandir(directory) as entries:
                for entry in entries:
                    examined += 1
                    if examined > 2048 or time.monotonic() >= deadline:
                        break
                    if entry.is_symlink():
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        if depth < 7 and entry.name not in {"policy", "curve-checkpoints", "selector-work"}:
                            pending.append((Path(entry.path), depth + 1))
                    elif entry.name == "progress.json":
                        path = Path(entry.path)
                        try:
                            with path.open("rb") as handle:
                                raw = handle.read(65537)
                            if len(raw) > 65536:
                                continue
                            value = json.loads(raw)
                            if not isinstance(value, dict):
                                continue
                            updated = core.number(value.get("updated", 0.), "progress timestamp", 0.)
                            observations.append((updated, path, value))
                        except (OSError, ValueError, TypeError):
                            continue
        except (OSError, RuntimeError):
            continue
    return sorted(observations, key=lambda row: row[0], reverse=True)


def wait_diagnostics(path):
    # Diagnostics must never replace the terminal timeout with another error.
    try:
        _wait_diagnostics(path)
    except Exception as exc:
        print(f"[pair-wait] diagnostics unavailable ({type(exc).__name__}); lock={path}", flush=True)


def _wait_diagnostics(path):
    """Do not confuse local lock ownership with an observed remote heartbeat."""
    print(f"[pair-wait] lock={path}; existing processes, lock and saved work preserved", flush=True)
    try:
        stat = path.stat()
        with Path("/proc/locks").open() as handle:
            rows = handle.read(1_048_576).splitlines()
        owners = []
        for row in rows:
            fields = row.split()
            if len(fields) < 8 or "->" in fields or fields[3] != "WRITE":
                continue
            device = fields[5].split(":")
            if len(device) == 3 and (int(device[0], 16), int(device[1], 16), int(device[2])) == (
                    os.major(stat.st_dev), os.minor(stat.st_dev), stat.st_ino) and int(fields[4]) > 0:
                owners.append(fields[4])
        for pid in owners[:4]:
            print(f"[pair-lock-owner] local host={base.node_id()} pid={pid}", flush=True)
    except (OSError, ValueError):
        owners = []
    if not owners:
        print("[pair-lock-owner] not visible locally; this is not proof that the owner died", flush=True)
    root = path.parent
    if path.name == ".state.lock":
        root = path.parents[2]
    observations = pair_progress(root)
    now = time.time()
    for updated, item, value in observations[:4]:
        fresh = value.get("state") == "running" and -5 <= now - updated < 60
        print(f"[pair-observed] host={value.get('host', '?')} phase={value.get('phase', '?')} "
              f"age={max(0., now-updated):.0f}s task={item.parent.relative_to(root)}; "
              + ("recent heartbeat, not proof of lock ownership" if fresh else "historical record, not confirmed running"), flush=True)


@contextlib.contextmanager
def queue_lease(path, *, shared=False, wait_seconds=15., max_wait_seconds=180.):
    """Shared worker lifetime vs legacy exclusive controller; never break locks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    core.number(wait_seconds, "wait interval", 1e-6)
    core.number(max_wait_seconds, "lock wait limit", 0.)
    deadline = time.monotonic() + max_wait_seconds
    with path.open("a+") as handle:
        while True:
            try:
                fcntl.flock(handle, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
            except BlockingIOError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    wait_diagnostics(path)
                    raise PairWaitTimeout(f"lock wait exceeded {max_wait_seconds:g}s: {path}; "
                                          "this waiting worker stopped; no lock was bypassed or deleted") from None
                detail = ("previous single-controller run or preparation must finish before queue handoff"
                          if shared else "another node is publishing the shared preparation/decision barrier")
                delay = min(wait_seconds, remaining)
                print(f"[WAIT] host={base.node_id()} lock={path} {detail}; "
                      f"retry in {delay:g}s; remaining wait={remaining:.0f}s", flush=True)
                time.sleep(delay)
            else:
                break
        yield


def show_pair_activity(root):
    """Bounded metadata observation only; never claim a stale owner is live."""
    print(f"[already running] pair root is locked: {root}; existing work left untouched", flush=True)
    print("[pair] One controller per root; this invocation starts no additional GPU work.", flush=True)
    observations = []
    now = time.time()
    for path in root.glob("branches/*/states/*/points/*/**/progress.json"):
        try:
            if not path.resolve().is_relative_to(root.resolve()):
                continue
            with path.open("rb") as handle:
                raw = handle.read(65537)
            if len(raw) > 65536:
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                continue
            updated = core.number(value.get("updated", 0.), "progress timestamp", 0.)
            observations.append((updated, path, value))
        except (OSError, ValueError, TypeError):
            continue
    observations.sort(key=lambda row: row[0], reverse=True)
    active = [row for row in observations if row[2].get("state") == "running" and -5 <= now-row[0] < 60]
    for updated, path, value in (active or observations[:1])[:4]:
        marker = "pair-active" if active else "pair-last"
        print(f"[{marker}] host={value.get('host', '?')} phase={value.get('phase', '?')} "
              f"age={max(0., now-updated):.0f}s task={path.parent.relative_to(root)}"
              + ("" if active else " (historical evidence; not confirmed running)"), flush=True)
    if not observations:
        print("[pair] Owner/phase metadata unavailable; preparation or CPU work may hold the lock.", flush=True)


def check_running(root):
    """Probe an existing root lease before setup or GPU admission, without writes."""
    try:
        handle = (root / ".pair.lock").open("rb")
    except FileNotFoundError:
        return False
    with handle:
        try:
            # Controllers hold EX; a shared probe detects them without needing
            # a writable descriptor (required for EX on NFS).
            fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            show_pair_activity(root)
            return True
        else:
            fcntl.flock(handle, fcntl.LOCK_UN)
    return False


def code_hashes():
    return {**switch.code_hashes(), **{name: base.digest(base.ROOT / name) for name in EXTRA_CODE}}


def compatible_code(recorded):
    current = code_hashes()
    if recorded == current:
        return True
    if (not isinstance(recorded, dict) or core.fingerprint(recorded) not in {
            PRE_BOOTSTRAP_CODE, PRE_DEFAULTS_CODE, PRE_RESOURCES_CODE, *PRE_SHARED_RUNTIME_CODES}
            or set(recorded) != set(current)
            or any(recorded[name] != value for name, value in current.items()
                   if name not in STARTUP_FILES and name not in switch.CODE)):
        return False
    shared_recorded = {name: recorded[name] for name in switch.CODE}
    shared_current = {name: current[name] for name in switch.CODE}
    if shared_recorded == shared_current:
        return True
    try:
        return switch.validate_code_hashes(shared_recorded) == shared_current
    except (ValueError, TypeError, KeyError):
        return False


def bind_startup_runtime(root, recorded):
    def reviewed_receipt(path, receipt, predecessors):
        switch.bind_reviewed_runtime_receipt(path, receipt,
                                            {*predecessors, PRE_PAIR_STATUS_CODE, PRE_PAIR_BRANCH_QUEUE_CODE})

    if recorded != code_hashes():
        # Preserve and validate the exact historical upgrade chain. The resource
        # patch has its own pinned receipt, never rewrites a scientific manifest,
        # and does not grant future entrypoint edits an unlimited exemption.
        path = root / "startup-runtime.json"
        receipt = {
            "schema": "offpolicy-selector-pair/startup-runtime-v1",
            "frozen_code_hashes": recorded,
            "change": "setup placeholder, actionable first launch and interrupted prepare recovery only"}
        prior = {}
        if path.exists():
            previous = core.read(path)
            runtime = previous.get("runtime_code_hashes", {})
            if (core.fingerprint(runtime) not in {PRE_DEFAULTS_CODE, PRE_RESOURCES_CODE}
                    or not compatible_code(runtime)
                    or previous != {**receipt, "runtime_code_hashes": runtime}):
                raise ValueError(f"frozen contract changed: {path}")
            prior[path.name] = base.digest(path)
        defaults = root / "startup-defaults-runtime.json"
        if defaults.exists():
            previous_defaults = core.read(defaults)
            runtime_defaults = previous_defaults.get("runtime_code_hashes", {})
            if (not path.exists() or core.fingerprint(runtime) != PRE_DEFAULTS_CODE
                    or core.fingerprint(runtime_defaults) != PRE_RESOURCES_CODE
                    or not compatible_code(runtime_defaults)
                    or previous_defaults != {**receipt, "runtime_code_hashes": runtime_defaults,
                        "previous_receipt_sha256": base.digest(path),
                        "change": "no-argument launch and defaults for unfrozen setup only"}):
                raise ValueError(f"frozen contract changed: {defaults}")
            prior[defaults.name] = base.digest(defaults)
        resources_path = root / "startup-resources-runtime.json"
        resources_receipt = {
            "schema": "offpolicy-selector-pair/resources-runtime-v1",
            "frozen_code_hashes": recorded, "runtime_code_hashes": code_hashes(),
            "previous_receipt_sha256": prior, "cpu_environment": CPU_ENV,
            "change": "bound CPU thread pools and distinguish lock contention; actual elapsed costs retained"}
        if resources_path.exists():
            previous_resources = core.read(resources_path)
            previous_code = previous_resources.get("runtime_code_hashes", {})
            if (previous_resources != resources_receipt and
                    (core.fingerprint(previous_code) not in PRE_SHARED_RUNTIME_CODES
                     or not compatible_code(previous_code)
                     or previous_resources != {**resources_receipt, "runtime_code_hashes": previous_code})):
                raise ValueError(f"frozen contract changed: {resources_path}")
        else:
            base.bind(resources_path, resources_receipt)
        recovery_path = root / "shared-checkpoint-recovery-runtime.json"
        reviewed_receipt(recovery_path, {
            "schema": "offpolicy-selector-pair/shared-checkpoint-recovery-runtime-v1",
            "frozen_code_hashes": recorded, "runtime_code_hashes": code_hashes(),
            "resources_runtime_sha256": base.digest(resources_path),
            "change": "shared Switch runtime recovery and validated checkpoint retention only; pair design, selectors and trainer unchanged",
            "cost_policy": "preserve all protocols, receipts, policies, costs, choices and budgets; no refunds or parent restart",
        }, {PRE_BUDGET_STOP_EVALUATION_CODE, PRE_PAIR_OPERATIONS_CODE,
            PRE_PAIR_LOCK_OBSERVATION_CODE, PRE_PAIR_DISTRIBUTED_CODE, PRE_PAIR_WAIT_GUARD_CODE, PRE_SHARED_MBPP_QUARANTINE_CODE})
        if core.fingerprint(code_hashes()) != PRE_BUDGET_STOP_EVALUATION_CODE:
            evaluation_path = root / "budget-stop-evaluation-runtime.json"
            reviewed_receipt(evaluation_path, {
                "schema": "offpolicy-selector-pair/budget-stop-evaluation-runtime-v1",
                "frozen_code_hashes": recorded, "runtime_code_hashes": code_hashes(),
                "shared_checkpoint_recovery_runtime_sha256": base.digest(recovery_path),
                "change": "shared Switch completed-policy evaluation resume only; pair design, selectors and trainer unchanged",
                "cost_policy": "preserve all protocols, receipts, policies, costs, choices and budgets; no refunds or retraining",
            }, {PRE_PAIR_OPERATIONS_CODE, PRE_PAIR_LOCK_OBSERVATION_CODE,
                PRE_PAIR_DISTRIBUTED_CODE, PRE_PAIR_WAIT_GUARD_CODE, PRE_SHARED_MBPP_QUARANTINE_CODE})
            operations_path = root / "pair-operations-runtime.json"
            reviewed_receipt(operations_path, {
                "schema": "offpolicy-selector-pair/operations-runtime-v1",
                "frozen_code_hashes": recorded, "runtime_code_hashes": code_hashes(),
                "evaluation_runtime_sha256": base.digest(evaluation_path),
                "change": "read-only status, bounded branch retries with NCCL admission, exhausted-allocation guard",
                "cost_policy": "preserve selectors, trainer, target, caps, checkpoints, decisions and all prior costs",
            }, {PRE_PAIR_LOCK_OBSERVATION_CODE, PRE_PAIR_DISTRIBUTED_CODE, PRE_PAIR_WAIT_GUARD_CODE, PRE_SHARED_MBPP_QUARANTINE_CODE})
            observation_path = root / "pair-lock-observation-runtime.json"
            reviewed_receipt(observation_path, {
                "schema": "offpolicy-selector-pair/lock-observation-runtime-v1",
                "frozen_code_hashes": recorded, "runtime_code_hashes": code_hashes(),
                "operations_runtime_sha256": base.digest(operations_path),
                "change": "observe an existing controller before setup/GPU admission; no duplicate worker",
                "cost_policy": "preserve targets, allocations, costs, results, decisions and all previous receipts",
            }, {PRE_PAIR_DISTRIBUTED_CODE, PRE_PAIR_WAIT_GUARD_CODE, PRE_SHARED_MBPP_QUARANTINE_CODE})
            distributed_path = root / "pair-distributed-runtime.json"
            reviewed_receipt(distributed_path, {
                "schema": "offpolicy-selector-pair/distributed-runtime-v1",
                "frozen_code_hashes": recorded, "runtime_code_hashes": code_hashes(),
                "lock_observation_runtime_sha256": base.digest(observation_path),
                "change": "matched-state leases across nodes with shared preparation and fit/freeze barriers",
                "cost_policy": "preserve selectors, trainer, targets, caps, all saved work, costs, decisions and previous receipts; no refunds",
            }, {PRE_PAIR_WAIT_GUARD_CODE, PRE_SHARED_MBPP_QUARANTINE_CODE})
            wait_path = root / "pair-wait-guard-runtime.json"
            reviewed_receipt(wait_path, {
                "schema": "offpolicy-selector-pair/wait-guard-runtime-v1",
                "frozen_code_hashes": recorded, "runtime_code_hashes": code_hashes(),
                "distributed_runtime_sha256": base.digest(distributed_path),
                "change": "bounded queue waits and owner diagnostics without breaking peer leases or restarting saved work",
                "cost_policy": "preserve targets, caps, protocols, decisions, results, checkpoints, costs and all previous receipts; no refunds",
            }, {PRE_SHARED_MBPP_QUARANTINE_CODE})
            if core.fingerprint(code_hashes()) not in PRE_SHARED_RUNTIME_CODES:
                reviewed_receipt(root / "shared-mbpp-quarantine-runtime.json", {
                    "schema": "offpolicy-selector-pair/shared-mbpp-quarantine-runtime-v1",
                    "frozen_code_hashes": recorded, "runtime_code_hashes": code_hashes(),
                    "wait_guard_runtime_sha256": base.digest(wait_path),
                    "change": "shared Switch MBPP branch quarantine compatibility only; pair scheduling and science unchanged",
                    "cost_policy": "preserve all protocols, receipts, checkpoints, results, costs, targets and budgets; no refunds or restart",
                }, set())
                reviewed_receipt(root / "pair-status-runtime.json", {
                    "schema": "offpolicy-selector-pair/status-runtime-v1",
                    "frozen_code_hashes": recorded, "runtime_code_hashes": code_hashes(),
                    "quarantine_runtime_sha256": base.digest(root / "shared-mbpp-quarantine-runtime.json"),
                    "change": "read-only dashboard dispatch; training and accounting unchanged",
                }, {PRE_PAIR_BRANCH_QUEUE_CODE})
                base.bind(root / "pair-branch-queue-runtime.json", {
                    "schema": "offpolicy-selector-pair/branch-queue-runtime-v1",
                    "frozen_code_hashes": recorded, "runtime_code_hashes": code_hashes(),
                    "status_runtime_sha256": base.digest(root / "pair-status-runtime.json"),
                    "change": "independent branch leases, serialized state preparation/publication, existing fit/freeze barriers",
                    "cost_policy": "preserve selectors, targets, budgets, checkpoints, decisions, all costs and prior receipts",
                })


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
    with pair_lease(root / ".pair.lock"):
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
    path = root / "pair.json"
    if path.exists() and core.read(path).get("schema") == pair.SCHEMA:
        # New controllers may join a running queue. An old exclusive controller
        # must release its lease first: it does not know the new state locks.
        with queue_lease(root / ".pair.lock", shared=True):
            with queue_lease(root / ".pair-runtime.lock"):
                return manifest(root)
    value = initialize(root)
    if value["schema"] == BOOTSTRAP_SCHEMA:
        prepare(preparation_options(root))
    return manifest(root)


def manifest(root, *, bind_runtime=True):
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
    if bind_runtime:
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
    with pair_lease(root / ".pair.lock"):
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
    configure_cpu_runtime()
    import additive_experiment as ae
    return {**ae.model_environment(c["config"]), **CPU_ENV}


def execute(entry, arm, devices):
    branch, out, c, protocol, suite = entry
    root = branch.parent.parent
    manifest(root)  # Recheck before starting a new process from on-disk code.
    env = {**environment(c), "PAIR_PROTOCOL_ROOT": str(root)}
    with pair_lease(out / arm / ".task.lock"):
        directory = out / arm
        # Exhausted attempts must not meter verification/scoring again. Let the
        # original validated final-policy/result recovery finish reporting work.
        if not any((directory / name).exists() for name in (
                "result.json", "policy/budget_stop.json", "policy/policy_train.json")):
            choice_path = directory / "decision.json"
            choice = core.read(choice_path) if choice_path.exists() else {
                "budget_gpu_seconds": c["budget_gpu_seconds"]}
            switch.remaining_allocation(directory, choice)
        switch.runtime.run_arm(out, suite, protocol, arm, devices, env)
        switch.curve_once(branch, switch.manifest(branch), out, c, arm, suite, devices, env)
        if not switch.branch_finished(switch.manifest(branch), out / arm):
            raise PairWorkPending(f"curve publication is pending: {out / arm}")


def attempt_branch(root, p, entry, arm, devices):
    """Keep unrelated branches moving; at most one runtime retry per pass.

    Cost/contract failures are not retried. Runtime failures require a fresh
    admission probe before either retrying or proceeding to another branch.
    Signals are never swallowed and previous work/costs are never reset.
    """
    directory = entry[1] / arm
    path = directory / "pair-attempt.json"
    task = str(directory.relative_to(root))
    for attempt in (1, 2):
        core.atomic_json(path, {"state": "RUN", "task": task, "attempt": attempt,
                               "host": base.node_id(), "pid": os.getpid(), "updated": time.time()})
        try:
            execute(entry, arm, devices)
        except PairWorkPending:
            core.atomic_json(path, {"state": "WAIT", "task": task, "attempt": attempt,
                                   "host": base.node_id(), "updated": time.time(),
                                   "reason": "shared curve evaluation is held by a peer"})
            raise
        except (ValueError, OSError, RuntimeError) as exc:
            failure = {"state": "WAIT", "task": task, "attempt": attempt,
                       "host": base.node_id(), "updated": time.time(),
                       "error": f"{type(exc).__name__}: {exc}"}
            core.atomic_json(path, failure)
            print(f"[pair-task] {task}: {failure['error']}", file=sys.stderr, flush=True)
            if isinstance(exc, NodeAdmissionError):
                raise
            if isinstance(exc, (OSError, RuntimeError)):
                admitted = admit_node(root, p)
                if admitted != devices:
                    raise NodeAdmissionError("allocated GPUs changed during the pair pass")
                if attempt == 1:
                    print(f"[pair-retry] {task}: resume saved work after successful node admission", flush=True)
                    continue
            return failure
        else:
            core.atomic_json(path, {"state": "DONE", "task": task, "attempt": attempt,
                                   "host": base.node_id(), "updated": time.time()})
            return None


def finish_pass(root, stage, failures):
    core.atomic_json(root / f"{stage}-pass.json", {"stage": stage, "updated": time.time(),
                     "state": "WAIT" if failures else "DONE", "failures": failures})
    if failures:
        raise IncompletePairRun(f"{stage}: {len(failures)} task(s) remain; other available work was attempted; "
                                f"saved work and costs preserved; see {root / (stage + '-pass.json')}")


def queue_branches(entries, seed, stage, choice=None):
    names = [(name, "selection_reduced") for name in pair.SELECTORS] if stage == "development" else [
        ("on_policy", "selection_full"), ("cached", "selection_full"),
        (f"adaptive-{choice['selector']}", "selection_full"), ("on_policy", "random_full")]
    return [(name, arm, entries[name]) for name, arm in (names[::-1] if seed % 2 else names)]


def branch_receipt(p, identity, name, arm, entry):
    return {"protocol_id": p["protocol_id"], "state_id": identity, "branch": name, "arm": arm,
            "curve": measured_curve(entry, arm)}


def distributed_stage(root, p, devices, stage, *, wait_seconds=15., idle_timeout=180.):
    """Share a state among independent branches; publish only after all finish.

    Shared state leases exclude older exclusive state workers and the final
    fit/report validators. Preparation is serialized separately, and each
    branch claim remains exclusive through execution and its validated receipt.
    """
    if stage not in {"development", "test"}:
        raise ValueError("unknown pair queue stage")
    core.number(wait_seconds, "queue wait interval", 1e-6)
    core.number(idle_timeout, "queue idle limit", 0.)
    choices = decisions(root, p) if stage == "test" else None
    seeds = pair.DEV_SEEDS if stage == "development" else pair.TEST_SEEDS
    states = [(seed, step) for seed in seeds for step in pair.STEPS]
    verified, completed, failed = set(), set(), {}
    worker = uuid.uuid4().hex
    worker_path = root / "queue-workers" / f"{worker}.json"
    observation = {"host": base.node_id(), "pid": os.getpid(), "stage": stage,
                   "protocol_id": p["protocol_id"], "worker": worker}
    waited = 0.
    last_activity = time.monotonic()
    seen_progress = {}

    def record(state, task=None):
        core.atomic_json(worker_path, {**observation, "state": state, "task": task,
                         "updated": time.time(), "verified_states": len(verified),
                         "total_states": len(states), "verified_branches": len(completed),
                         "total_branches": len(states) * (2 if stage == "development" else 4),
                         "queue_wait_wall_seconds": waited,
                         "failures": list(failed.values())})

    def result_row(seed, step):
        return (development_row(root, p, seed, step) if stage == "development"
                else test_row(root, p, seed, step, choices))

    try:
        while len(verified) < len(states):
            busy, busy_directories = set(), set()
            previous_verified = (len(verified), len(completed))
            for seed, step in states:
                key = (seed, step)
                if key in verified:
                    continue
                folder = root / stage / f"s{seed}-t{step}"
                path = folder / "result.json"
                # Invalid published summaries must never restart their branches.
                if key in failed and not path.exists():
                    continue
                entered_state = False
                try:
                    if path.exists():
                        with pair_lease(folder / ".state.lock"):
                            base.bind(path, result_row(seed, step))
                            verified.add(key)
                            failed.pop(key, None)
                        continue
                    with pair_lease(folder / ".state.lock", shared=True):
                        entered_state = True
                        with pair_lease(folder / ".prepare-state.lock"):
                            identity, entries = verify_pair(root, seed, step)
                            if stage == "development":
                                base.bind(folder / "state.json", {"state_id": identity, "protocol_id": p["protocol_id"]})
                            choice = choices[f"s{seed}-t{step}"] if choices is not None else None
                            if choice is not None and identity != choice["state_id"]:
                                raise ValueError("test parent state differs from the frozen decision")
                        tasks = queue_branches(entries, seed, stage, choice)
                        for name, arm, entry in tasks:
                            branch_key = (seed, step, name, arm)
                            receipt = folder / "queue-branches" / f"{name}--{arm}.json"
                            if branch_key in completed or branch_key in failed and not receipt.exists():
                                continue
                            directory = entry[1] / arm
                            try:
                                with pair_lease(receipt.with_suffix(".lock")):
                                    if not receipt.exists() and not all((directory / item).is_file()
                                                                      for item in ("result.json", "curve.json")):
                                        record("RUN", f"{stage}/s{seed}-t{step}/{name}/{arm}")
                                        print(f"[RUN] host={base.node_id()} {stage}/s{seed}-t{step}/{name}/{arm}", flush=True)
                                        failure = attempt_branch(root, p, entry, arm, devices)
                                        if failure:
                                            failed[branch_key] = failure
                                            continue
                                    base.bind(receipt, branch_receipt(p, identity, name, arm, entry))
                                    completed.add(branch_key)
                                    failed.pop(branch_key, None)
                            except PairLockBusy as exc:
                                if exc.path != receipt.with_suffix(".lock"):
                                    raise
                                busy.add(f"s{seed}-t{step}")
                                busy_directories.add(directory)
                            except PairWorkPending:
                                busy.add(f"s{seed}-t{step}")
                                busy_directories.update((directory, entry[1] / "curve-parent"))
                            except NodeAdmissionError:
                                raise
                            except (ValueError, OSError, RuntimeError) as exc:
                                failed[branch_key] = {"task": str(directory.relative_to(root)),
                                                      "error": f"{type(exc).__name__}: {exc}"}
                                print(f"[WAIT] {directory}: {exc}; trying other branches", flush=True)
                    # EX cannot be acquired while any new or old peer still
                    # owns this state. Validate the complete curves again here.
                    with pair_lease(folder / ".state.lock"):
                        if all((seed, step, name, arm) in completed for name, arm, _ in tasks):
                            base.bind(path, result_row(seed, step))
                            verified.add(key)
                except PairLockBusy as exc:
                    if exc.path not in {folder / ".state.lock", folder / ".prepare-state.lock"}:
                        raise
                    busy.add(f"s{seed}-t{step}")
                    if not entered_state:
                        busy_directories.update(root / "branches" / branch / "states" / f"s{seed}-t{step}"
                                                for branch in BRANCHES)
                except NodeAdmissionError:
                    raise
                except (ValueError, OSError, RuntimeError) as exc:
                    failed[key] = {"state": f"s{seed}-t{step}", "error": f"{type(exc).__name__}: {exc}"}
                    print(f"[WAIT] {stage}/s{seed}-t{step}: {exc}; trying other states", flush=True)
            if len(verified) == len(states):
                record("DONE")
                return
            if not busy:
                record("WAIT")
                raise IncompletePairRun(f"{stage}: {len(states)-len(verified)} state(s) remain; "
                                        f"saved work preserved; see {worker_path}")
            if (len(verified), len(completed)) != previous_verified:
                last_activity = time.monotonic()
            for updated, path, value in pair_progress(root, busy):
                if (value.get("state") == "running" and -5 <= time.time()-updated < 60
                        and updated > seen_progress.get(path, 0.)
                        and any(directory in path.parents for directory in busy_directories)):
                    last_activity = time.monotonic()
                    seen_progress[path] = updated
            remaining = idle_timeout - (time.monotonic()-last_activity)
            if remaining <= 0:
                for name in sorted(busy)[:4]:
                    wait_diagnostics(root / stage / name / ".state.lock")
                raise PairWaitTimeout(f"{stage}: no new peer heartbeat or completed state for {idle_timeout:g}s; "
                                      f"busy states={','.join(sorted(busy))}; this idle worker stopped, saved work preserved")
            record("WAIT", "peer branches: " + ", ".join(sorted(busy)))
            delay = min(wait_seconds, remaining)
            print(f"[WAIT] host={base.node_id()} {stage}: verified {len(verified)}/{len(states)}; "
                  f"peer states={','.join(sorted(busy))}; checking for the next task in {delay:g}s; "
                  f"no-progress limit remaining={remaining:.0f}s", flush=True)
            started = time.monotonic()
            try:
                time.sleep(delay)
            finally:
                waited += time.monotonic() - started
    except BaseException:
        record("WAIT", "worker stopped; existing results/checkpoints preserved")
        raise


@contextlib.contextmanager
def completed_state_leases(root, stages=("development",), *, require_complete=True):
    """Use the same locks for barrier validation as for peer state validation."""
    folders = []
    for stage in stages:
        seeds = pair.DEV_SEEDS if stage == "development" else pair.TEST_SEEDS
        for seed in seeds:
            for step in pair.STEPS:
                folder = root / stage / f"s{seed}-t{step}"
                if (folder / "result.json").exists():
                    folders.append(folder)
                elif require_complete:
                    raise IncompletePairRun(f"{stage}/s{seed}-t{step}: validated result is not published")
    with contextlib.ExitStack() as stack:
        for folder in folders:
            stack.enter_context(queue_lease(folder / ".state.lock"))
        yield


def admission_required(root, command):
    """A publication hint may skip GPU admission, never scientific validation."""
    if command == "freeze":
        return not ((root / "model.json").is_file() and (root / "test-decisions.json").is_file())
    if command not in {"run", "develop", "test"}:
        return False
    stages = ("development", "test") if command == "run" else ("development" if command == "develop" else "test",)
    if command in {"run", "test"} and not all((root / name).is_file() for name in ("model.json", "test-decisions.json")):
        return True
    return any(not (root / stage / f"s{seed}-t{step}" / "result.json").is_file()
               for stage in stages for seed in (pair.DEV_SEEDS if stage == "development" else pair.TEST_SEEDS)
               for step in pair.STEPS)


def run_distributed(root, p, devices, command):
    if command in {"run", "develop"}:
        distributed_stage(root, p, devices, "development")
        if command == "develop":
            return
    # Only this short barrier is exclusive across queue workers. No held-out
    # execution starts until every development label and all six decisions
    # have been validated and frozen. Long GPU continuations are outside it.
    with queue_lease(root / ".pair-barrier.lock"):
        if command in {"run", "fit"}:
            with completed_state_leases(root):
                fit(root, p)
            finish_pass(root, "development", [])
        if command in {"run", "freeze"}:
            freeze(root, p)
    if command in {"run", "test"}:
        distributed_stage(root, p, devices, "test")
        with queue_lease(root / ".pair-barrier.lock"):
            with completed_state_leases(root, ("development", "test"), require_complete=False):
                report(root, p)
            finish_pass(root, "test", [])


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
    failures = []
    for seed in pair.DEV_SEEDS:
        for step in pair.STEPS:
            identity, entries = verify_pair(root, seed, step)
            folder = root / "development" / f"s{seed}-t{step}"
            base.bind(folder / "state.json", {"state_id": identity, "protocol_id": p["protocol_id"]})
            state_failures = []
            for name in (tuple(pair.SELECTORS) if seed % 2 == 0 else tuple(pair.SELECTORS)[::-1]):
                print(f"[pair] development s{seed}/t{step}/{name}", flush=True)
                failure = attempt_branch(root, p, entries[name], "selection_reduced", devices)
                if failure:
                    state_failures.append(failure)
            failures.extend(state_failures)
            if not state_failures:
                base.bind(folder / "result.json", development_row(root, p, seed, step))
    finish_pass(root, "development", failures)


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
    failures = []
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
            state_failures = []
            for entry, arm in tasks if seed % 2 == 0 else tasks[::-1]:
                print(f"[pair] test {name}/{entry[0].name}/{arm}", flush=True)
                failure = attempt_branch(root, p, entry, arm, devices)
                if failure:
                    state_failures.append(failure)
            failures.extend(state_failures)
            if not state_failures:
                base.bind(root / "test" / name / "result.json", test_row(root, p, seed, step, choices))
    finish_pass(root, "test", failures)


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


def read_status(root):
    """Observation only: no initialization, exclusive lock, migration or GPU work."""
    path = root / "pair.json"
    if not path.exists():
        print(json.dumps({"root": str(root), "state": "WAIT", "reason": "pair is not prepared"}))
        return
    p = core.read(path)
    if p.get("schema") == BOOTSTRAP_SCHEMA:
        print(json.dumps({"root": str(root), **p}, indent=2))
        return
    status(root, manifest(root, bind_runtime=False))


def main():
    configure_cpu_runtime()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "prepare", "ensure-prepared", "run", "develop", "fit", "freeze", "test", "report", "status", "check-code", "check-running"))
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
    if args.command == "check-running":
        raise SystemExit(75 if check_running(args.root) else 0)
    install_runtime()
    if args.command == "prepare":
        prepare(preparation_options(args.root, {key: getattr(args, key) for key in CONFIG_KEYS}))
        return
    if any(v is not None for k, v in vars(args).items() if k not in {"command", "root"}):
        parser.error("preparation options cannot change a frozen run; prepare a new root")
    if args.command == "status":
        read_status(args.root)
        return
    if args.command == "init":
        value = initialize(args.root)
        if value["schema"] == BOOTSTRAP_SCHEMA:
            print(json.dumps({"root": str(args.root), **value}, indent=2))
            return
    if args.command in {"run", "develop", "ensure-prepared"}:
        p = ensure_prepared(args.root)
    else:
        with queue_lease(args.root / ".pair.lock", shared=True):
            with queue_lease(args.root / ".pair-runtime.lock"):
                p = manifest(args.root)
    if args.command == "init":
        print(f"[prepared] existing experiment preserved: {args.root / 'pair.json'}")
        return
    if args.command == "ensure-prepared":
        print(f"[ready] {args.root / 'pair.json'}")
        return
    if args.command == "check-code":
        print("[verified] pair and legacy code compatible (including reviewed operational migrations)")
        return
    with queue_lease(args.root / ".pair.lock", shared=True):
        if args.command == "report":
            with queue_lease(args.root / ".pair-barrier.lock"):
                with completed_state_leases(args.root, ("development", "test"), require_complete=False):
                    report(args.root, p)
            return
        resource_diagnostics()
        devices = admit_node(args.root, p) if admission_required(args.root, args.command) else None
        run_distributed(args.root, p, devices, args.command)


if __name__ == "__main__":
    from light_selection_gate_gpu import install_signal_handlers
    install_signal_handlers()
    try:
        main()
    except PairWaitTimeout as exc:
        print(f"[pair-wait-timeout] {exc}", file=sys.stderr, flush=True)
        raise SystemExit(76) from None
    except PairLockBusy as exc:
        if exc.path.name == ".pair.lock":
            show_pair_activity(exc.path.parent)
            raise SystemExit(75) from None
        raise SystemExit(f"[pair] {exc}") from None
    except NodeAdmissionError as exc:
        print(f"[blocked] {exc}", file=sys.stderr, flush=True)
        raise SystemExit(78) from None
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"[pair] {exc}") from None
    except (OSError, RuntimeError):
        resource_diagnostics()
        raise
