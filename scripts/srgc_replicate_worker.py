#!/usr/bin/env python3
"""Distribute planned extra experiments without modifying the completed P0 queue."""

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from srgc_rebuttal.plan import input_path, load_plan  # noqa: E402
from srgc_rebuttal.runtime import Busy, atomic_json, lease, run_root  # noqa: E402
from scripts.srgc_extra_plan import select_plan  # noqa: E402
from scripts.srgc_pair_inputs import default_plan  # noqa: E402
from scripts.srgc_shared_storage import route_plan  # noqa: E402

SUPPORT_ARMS = ("direction_removed", "sr_hold", "sr_refresh_matched")
TIMING_STEPS = (50, 100, 150, 200, 250)
TIMING_ARMS = tuple(f"switch_fixed{step}" for step in TIMING_STEPS)
RULE_ARMS = ("switch_single", "switch_consecutive")
SCOPES = ("all", "support", "mechanism", "switch_validation", "timing", "rules", "switch_fixed200", "replicate", "candidates", "switch_repeat",
          "sr_hold", "pool", "direction", "direction_removed", "direction_magnitude",
          "direction_replaced")


def conditions(scope):
    if scope == "mechanism":
        return [(0, "stage_mechanism")]
    if scope == "support":
        return [(0, arm) for arm in SUPPORT_ARMS]
    if scope in {"switch_validation", "timing", "rules"}:
        arms = {"switch_validation": (*TIMING_ARMS, *RULE_ARMS),
                "timing": TIMING_ARMS, "rules": RULE_ARMS}[scope]
        return [(0, arm) for arm in arms]
    groups = {
        "switch_fixed200": [(0, "switch_fixed200")],
        "replicate": [(k, arm) for k in (1, 2) for arm in ("sr", "switch")],
        "candidates": [(0, "sr_refresh")], "switch_repeat": [(0, "switch_repeat")],
        "sr_hold": [(0, "sr_hold")], "pool": [(0, "sr_refresh-pool")],
        "direction": [(0, f"direction_{mode}") for mode in ("removed", "magnitude", "replaced")],
    }
    if scope == "all":
        return [condition for group in groups.values() for condition in group]
    if scope.startswith("direction_") and scope in SCOPES:
        return [(0, scope)]
    return groups[scope]


@dataclass(frozen=True)
class Task:
    dataset: str
    seed: int
    repeat: int
    arm: str
    plan: Path

    @property
    def name(self):
        return f"replicate{self.repeat}-{self.arm}" if self.repeat else self.arm

    @property
    def key(self):
        return f"{self.dataset}.seed-{self.seed}.{self.name}"

    @property
    def folder(self):
        return run_root(self.plan, load_plan(self.plan)) / f"seed-{self.seed}"

    @property
    def out(self):
        return self.folder / f"replicate-{self.repeat}" if self.repeat else self.folder

    @property
    def receipt(self):
        return self.out / "launches" / self.arm / "queue.json"


def tasks_for(dataset, scope="replicate"):
    datasets = ("math", "mbpp") if dataset == "all" else (dataset,)
    plans = {}
    for name in datasets:
        source = default_plan(REPO, name, os.environ, writing=False)
        active = route_plan(source, writing=False)
        for seed in range(5, 10):
            plans[name, seed] = select_plan(active, seed)
    return [Task(name, seed, repeat, arm, plans[name, seed])
            for repeat, arm in conditions(scope) for name in datasets
            for seed in range(5, 10)]


def signature(task):
    """Invalidate a successful validation when any of its saved inputs changes."""
    paths = [task.plan, input_path(task.plan, load_plan(task.plan), task.seed),
             task.folder / "prefix-ready.json", task.folder / "prefix.pt", task.folder / "run.json",
             task.out / f"{task.arm}-endpoint.json"]
    if task.arm == "stage_mechanism":
        paths.append(REPO / "scripts/srgc_stage_mechanism.py")
    if task.repeat:
        paths.insert(-1, task.out / "replicate.json")
    result = []
    for path in paths:
        try:
            stat = path.stat()
        except FileNotFoundError:
            return None
        result.append([str(path), stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns])
    return result


def python_for(dataset):
    explicit = os.environ.get("PAIR_PYTHON" if dataset == "math" else "SWITCH_PYTHON")
    if explicit:
        if not shutil.which(explicit):
            raise ValueError(f"Python not found: {explicit}")
        return explicit
    return sys.executable


def run_task(task, handle):
    command = [python_for(task.dataset), str(REPO / "scripts/srgc_extra_worker.py"),
               "--plan", str(task.plan), "--seed", str(task.seed), "--arm", task.name]
    child = None
    stopped = []

    def stop(sig, _frame):
        stopped.append(sig)
        if child is None:
            return
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        child = subprocess.Popen(command, cwd=REPO, start_new_session=True,
                                 pass_fds=(handle.fileno(),))
        if stopped:
            stop(stopped[0], None)
        code = child.wait()
        return 128 + stopped[0] if stopped else (code if code >= 0 else 128 - code)
    finally:
        if child is not None and child.poll() is None:
            stop(signal.SIGTERM, None)
            child.wait()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def node_available():
    from srgc_rebuttal import cluster
    from scripts.srgc_process_guard import canonical_lock_root, reap_orphans
    reap_orphans()
    try:
        _, uuids = cluster.gpu_identity()
        root = canonical_lock_root()
        if root is None:
            raise ValueError("extra experiment queue requires group storage")
        with cluster.device_leases(root, uuids):
            pass
    except Busy:
        return False
    return True


def sweep(tasks, *, max_attempts=3, retry_delay=120, runner=run_task, now=time.time):
    counts = dict(complete=0, busy=0, waiting=0, failed=0)
    for task in tasks:
        if not all((task.folder / name).is_file() for name in ("prefix-ready.json", "prefix.pt")):
            counts["waiting"] += 1
            continue
        with ExitStack() as locks:
            try:
                handle = locks.enter_context(lease(task.out / f".{task.arm}.dispatch.lock"))
                # Manual launchers keep using their existing locks. A concurrent
                # manual start can still win the race; its launch lock remains authoritative.
                with lease(task.out / f".{task.arm}.launch.lock"), lease(task.out / f".{task.arm}.execution.lock"):
                    pass
            except Busy:
                counts["busy"] += 1
                continue
            try:
                previous = json.loads(task.receipt.read_text()) if task.receipt.exists() else {}
                if (not isinstance(previous, dict) or type(previous.get("attempt", 0)) is not int
                        or previous.get("attempt", 0) < 0):
                    raise ValueError("invalid queue receipt or attempt count")
                for field in ("started", "finished"):
                    if field in previous and (type(previous[field]) not in (int, float)
                            or not math.isfinite(previous[field]) or previous[field] < 0):
                        raise ValueError(f"invalid queue {field} timestamp")
            except (OSError, ValueError, TypeError) as exc:
                # Preserve the damaged record for inspection; never reset its
                # retry budget or prevent other independent tasks from running.
                print(f"INVALID {task.key} receipt={task.receipt}: {exc}", file=sys.stderr, flush=True)
                counts["failed"] += 1
                continue
            fingerprint = signature(task)
            if (previous.get("status") == "complete" and fingerprint is not None
                    and previous.get("verified_files") == fingerprint):
                counts["complete"] += 1
                continue
            attempts = previous.get("attempt", 0)
            endpoint_exists = (task.out / f"{task.arm}-endpoint.json").exists()
            changed_endpoint = endpoint_exists and fingerprint != previous.get("failed_files")
            if attempts >= max_attempts and not changed_endpoint:
                counts["failed"] += 1
                continue
            if (previous.get("status") == "failed" and not changed_endpoint
                    and now() < previous.get("finished", 0) + retry_delay):
                counts["waiting"] += 1
                continue
            record = dict(task=task.key, plan=str(task.plan), host=socket.gethostname(),
                          pid=os.getpid(), status="running", attempt=attempts + 1, started=now())
            atomic_json(task.receipt, record)
            print(f"TASK {task.key} attempt={record['attempt']} plan={task.plan}", flush=True)
            code = runner(task, handle)
            if code == 0:
                fingerprint = signature(task)
                if fingerprint is None:
                    code = 2
            if code == 75:
                # GPU/tuple contention is not a failed training attempt.
                atomic_json(task.receipt, {**previous, "status": "busy", "attempt": attempts})
                counts["busy"] += 1
                return counts, code
            if code in (130, 143):
                atomic_json(task.receipt, {**record, "status": "interrupted", "attempt": attempts,
                                          "exit_code": code, "finished": now()})
                return counts, code
            atomic_json(task.receipt, {**record, "status": "complete" if code == 0 else "failed",
                                      "exit_code": code, "finished": now(),
                                      "verified_files": fingerprint if code == 0 else None,
                                      "failed_files": signature(task) if code != 0 else None})
            print(f"TASK {task.key} {'complete' if code == 0 else 'failed'} exit={code}", flush=True)
            return counts, code
    return counts, None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("all", "math", "mbpp"), required=True)
    parser.add_argument("--scope", choices=SCOPES, default="replicate")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--poll", type=float, default=10)
    args = parser.parse_args(argv)
    if args.max_attempts < 1 or args.poll <= 0:
        parser.error("max-attempts and poll must be positive")
    tasks = tasks_for(args.dataset) if args.scope == "replicate" else tasks_for(args.dataset, args.scope)
    print(f"EXTRAS {args.dataset} scope={args.scope}: {len(tasks)} tasks, seeds 5-9 automatic; P0 unchanged", flush=True)
    while True:
        if not node_available():
            print("NODE idle: this allocation already has a GPU owner; no task claimed", flush=True)
            time.sleep(args.poll)
            continue
        counts, code = sweep(tasks, max_attempts=args.max_attempts)
        if code in (130, 143):
            return code
        if code is not None and code != 75:
            continue
        if counts["complete"] == len(tasks):
            print("PASS: all requested extra experiments complete", flush=True)
            return 0
        if code is None and counts["complete"] + counts["failed"] == len(tasks):
            print(f"FAILED: {counts['failed']} task(s) reached the retry limit; inspect launches/<arm>/queue.json", flush=True)
            return 1
        print(f"NODE idle complete={counts['complete']} busy={counts['busy']} "
              f"waiting={counts['waiting']} failed={counts['failed']}", flush=True)
        time.sleep(args.poll)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"INVALID: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2) from None
