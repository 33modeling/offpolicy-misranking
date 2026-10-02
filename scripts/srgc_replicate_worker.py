#!/usr/bin/env python3
"""Distribute the planned SR/Switch replicates without modifying the P0 queue."""

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
import json
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


@dataclass(frozen=True)
class Task:
    dataset: str
    seed: int
    repeat: int
    arm: str
    plan: Path

    @property
    def name(self):
        return f"replicate{self.repeat}-{self.arm}"

    @property
    def key(self):
        return f"{self.dataset}.seed-{self.seed}.{self.name}"

    @property
    def folder(self):
        return run_root(self.plan, load_plan(self.plan)) / f"seed-{self.seed}"

    @property
    def out(self):
        return self.folder / f"replicate-{self.repeat}"

    @property
    def receipt(self):
        return self.out / "launches" / self.arm / "queue.json"


def tasks_for(dataset):
    datasets = ("math", "mbpp") if dataset == "all" else (dataset,)
    plans = {}
    for name in datasets:
        source = default_plan(REPO, name, os.environ, writing=False)
        active = route_plan(source, writing=False)
        for seed in range(5, 10):
            plans[name, seed] = select_plan(active, seed)
    return [Task(name, seed, repeat, arm, plans[name, seed])
            for repeat in (1, 2) for name in datasets for arm in ("sr", "switch")
            for seed in range(5, 10)]


def signature(task):
    """Invalidate a successful validation when any of its saved inputs changes."""
    paths = [task.plan, input_path(task.plan, load_plan(task.plan), task.seed),
             task.folder / "prefix-ready.json", task.folder / "prefix.pt", task.folder / "run.json",
             task.out / "replicate.json",
             task.out / f"{task.arm}-endpoint.json"]
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
            raise ValueError("replicate queue requires group storage")
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
            previous = json.loads(task.receipt.read_text()) if task.receipt.exists() else {}
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
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--poll", type=float, default=10)
    args = parser.parse_args(argv)
    if args.max_attempts < 1 or args.poll <= 0:
        parser.error("max-attempts and poll must be positive")
    tasks = tasks_for(args.dataset)
    print(f"REPLICATE {args.dataset}: {len(tasks)} tasks, seeds 5-9, k=1,2, SR/Switch only; P0 unchanged", flush=True)
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
            print("PASS: all planned replicates complete", flush=True)
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
