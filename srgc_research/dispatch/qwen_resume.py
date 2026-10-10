"""Finish abandoned Qwen work across both datasets before claiming fresh tasks."""

import copy
import json
import math
import os
import socket
import time
import uuid
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import MethodType
from unittest.mock import patch

from srgc_rebuttal.plan import input_path
from srgc_rebuttal.runtime import Busy, atomic_json, lease


class ResumeFirst:
    def __init__(self, queues):
        self.queues = {str(q.plan_path.resolve()): q for q in queues}
        parent = Path(os.path.commonpath([q.plan_path.parent for q in queues]))
        self.root = parent.parent if parent.name == "experiments" else parent
        # A math-only node must also respect an MBPP resume backlog, and vice versa.
        if parent.name == "experiments":
            for name in ("math", "mbpp"):
                path = parent / f"qwen35-9b-{name}.json"
                if path.exists() and str(path.resolve()) not in self.queues:
                    self.queues[str(path.resolve())] = type(queues[0])(path)
        self.tasks = {(key, task.key): (queue, task) for key, queue in self.queues.items() for task in queue.tasks}
        self.marker = self.root / ".dispatch/resume-first.json"

    def receipt(self, queue, task):
        path = queue.receipt(task)
        row = json.loads(path.read_text()) if path.exists() else {}
        if not isinstance(row, dict):
            raise TypeError(f"invalid task receipt: {path}")
        if row and row.get("task") != task.key:
            raise ValueError(f"task receipt identity differs: {path}")
        if row and row.get("status") not in {"ready", "running", "complete", "failed", "interrupted"}:
            raise ValueError(f"invalid task status: {path}")
        for field in ("attempt", "retry_attempt"):
            if field in row and (type(row[field]) is not int or row[field] < 0):
                raise ValueError(f"invalid {field}: {path}")
        for field in ("started", "finished"):
            if field in row and (type(row[field]) not in (int, float) or
                                 not math.isfinite(row[field]) or row[field] < 0):
                raise ValueError(f"invalid {field}: {path}")
        return row

    def execution_lock(self, queue, task):
        if task.arm == "cache":
            return input_path(queue.plan_path, queue.plan, task.seed).with_suffix(".cache") / "execution.lock"
        return queue.root / f"seed-{task.seed}" / f".{task.arm}.execution.lock"

    def owned(self, queue, task):
        if queue.locked(task):
            return True
        path = self.execution_lock(queue, task)
        if path.exists():
            try:
                with lease(path):
                    pass
            except Busy:
                return True
        return False

    def started(self, queue, task, row):
        if row.get("status") in {"running", "interrupted", "failed"}:
            return True
        if task.arm == "cache":
            cache = input_path(queue.plan_path, queue.plan, task.seed).with_suffix(".cache")
            return any(cache.glob("*.json"))
        return (queue.root / f"seed-{task.seed}" / f"{task.arm}-latest.pt").is_file()

    def failure_counts(self):
        """Count actual failed attempts, preserving launch IDs and abandoned history."""
        failures = {}
        for queue in self.queues.values():
            for path in (queue.directory / "attempts").glob("*.json"):
                row = json.loads(path.read_text())
                if not isinstance(row, dict):
                    raise TypeError(f"invalid attempt receipt: {path}")
                if row.get("status") == "failed":
                    key = (str(queue.plan_path.resolve()), row.get("task"))
                    failures.setdefault(key, set()).add(row.get("attempt_id", path.stem))
        return {key: len(value) for key, value in failures.items()}

    def pending(self):
        plans = {key: {field: queue.protocol[field] for field in ("plan_sha256", "implementation_sha256")}
                 for key, queue in self.queues.items()}
        for queue in self.queues.values():
            queue.verify()
        saved = json.loads(self.marker.read_text()) if self.marker.exists() else {}
        if not isinstance(saved, dict):
            raise TypeError("invalid shared resume backlog")
        if saved and (saved.get("schema") != "qwen-resume-first-v1" or saved.get("plans") != plans):
            raise ValueError("resume backlog belongs to different queues, code or plans")
        entries = saved.get("tasks", [])
        if not isinstance(entries, list) or any(not isinstance(key, list) or len(key) != 2 or
                                               any(not isinstance(value, str) for value in key) for key in entries):
            raise ValueError("invalid task list in resume backlog")
        pending = {tuple(key) for key in entries}
        if not pending <= self.tasks.keys():
            raise ValueError("resume backlog contains an unknown task")
        pending = {key for key in pending if not (self.tasks[key][0].complete(self.tasks[key][1]) and
                                                 not self.owned(*self.tasks[key]))}
        started, abandoned = set(), set()
        for key, (queue, task) in self.tasks.items():
            if queue.complete(task):
                if self.owned(queue, task):
                    started.add(key)
                continue
            row = self.receipt(queue, task)
            if row.get("status") == "complete":
                raise ValueError(f"completed task lost its completion evidence: {queue.plan_path}:{task.key}")
            if self.started(queue, task, row):
                started.add(key)
                if not self.owned(queue, task):
                    abandoned.add(key)
        if pending or abandoned:
            pending.update(started)
            # Restore prerequisites of an existing task without opening another seed.
            for key in list(pending):
                queue, task = self.tasks[key]
                parent = queue.dependency(task)
                while parent is not None:
                    if not queue.complete(parent):
                        pending.add((key[0], parent.key))
                    parent = queue.dependency(parent)
        record = {"schema": "qwen-resume-first-v1", "plans": plans, "tasks": [list(key) for key in sorted(pending)],
                  "status": "resuming" if pending else "clear"}
        if record != saved:
            atomic_json(self.marker, record)
        return pending

    @contextmanager
    def claim(self, queue, *, retry_failed=False, max_attempts=3, retry_delay=60, worker_id=None):
        selected, handle = None, None
        with ExitStack() as locks:
            # Serialize only dispatch decisions; task execution remains parallel.
            with lease(self.marker.with_suffix(".lock"), wait=True):
                pending = self.pending()
                failures = self.failure_counts()
                active, possible = False, False
                for key in pending:
                    candidate_queue, candidate = self.tasks[key]
                    row = self.receipt(candidate_queue, candidate)
                    spent = failures.get(key, row.get("retry_attempt", row.get("attempt", 0))
                                         if row.get("status") == "failed" else 0)
                    active = active or self.owned(candidate_queue, candidate)
                    possible = possible or (spent < max_attempts and candidate_queue.ready(candidate))
                if pending and not active and not possible:
                    raise RuntimeError("unfinished Qwen tasks reached the real failure limit; fresh tasks remain blocked")
                for task in queue.tasks:
                    key = (str(queue.plan_path.resolve()), task.key)
                    if pending and key not in pending:
                        continue
                    if queue.complete(task) or not queue.ready(task) or self.owned(queue, task):
                        continue
                    try:
                        handle = locks.enter_context(lease(queue.directory / "leases" / f"{task.key}.lock"))
                    except Busy:
                        continue
                    execution = self.execution_lock(queue, task)
                    try:
                        if execution.exists():
                            with lease(execution):
                                pass
                    except Busy:
                        continue
                    queue.verify()
                    if queue.complete(task) or not queue.ready(task):
                        continue
                    previous = self.receipt(queue, task)
                    spent = failures.get(key, previous.get("retry_attempt", previous.get("attempt", 0))
                                         if previous.get("status") == "failed" else 0)
                    if spent >= max_attempts:
                        continue
                    if previous.get("status") == "failed" and (
                            not retry_failed or time.time() < previous.get("finished", 0) + retry_delay):
                        continue
                    if previous.get("status") == "running":
                        attempt_id = previous.get("attempt_id") or f"legacy-{uuid.uuid4().hex}"
                        if not isinstance(attempt_id, str) or Path(attempt_id).name != attempt_id or attempt_id in {".", ".."}:
                            raise ValueError("invalid abandoned attempt ID")
                        atomic_json(queue.directory / "attempts" / f"{attempt_id}.json",
                                    {**previous, "status": "abandoned", "recovered": time.time()})
                    record = {"task": task.key, "status": "running", "host": socket.gethostname(),
                        "pid": os.getpid(), "worker_id": worker_id, "attempt": previous.get("attempt", 0) + 1,
                        "retry_attempt": spent + 1, "attempt_id": uuid.uuid4().hex, "started": time.time(),
                        "resume_first": key in pending}
                    atomic_json(queue.receipt(task), record)
                    selected = task
                    print(f"{'RESUME' if key in pending else 'NEW'} {queue.plan['dataset']}:{task.key} "
                          f"failures={spent}/{max_attempts}", flush=True)
                    break
            previous_fds = queue.claim_fds
            queue.claim_fds = (handle.fileno(),) if selected is not None else ()
            try:
                yield selected
            finally:
                queue.claim_fds = previous_fds


@contextmanager
def resume_first_worker():
    # Match the imports used by the existing launcher and its runtime adapter.
    import srgc_qwen35_worker as worker

    from scripts import srgc_process_guard as guard

    original = worker.drain

    def drain(queues, args, environment, gpu_fds, worker_id, update):
        manager = ResumeFirst(queues)
        options = copy.copy(args)
        options.retry_failed = True
        with ExitStack() as stack:
            for queue in queues:
                def claim(instance, _manager=manager, **kwargs):
                    return _manager.claim(instance, **kwargs)
                stack.enter_context(patch.object(queue, "claim", MethodType(claim, queue)))
            return original(queues, options, environment, gpu_fds, worker_id, update)

    with patch.object(worker, "drain", drain), patch.object(guard, "OWNER_MARKERS", (
            *guard.OWNER_MARKERS, "srgc_research.dispatch.qwen_run")):
        yield
