"""Fill idle model workers without changing the frozen experiment adapters."""

import hashlib
import json
import os
import socket
import time
import uuid
from contextlib import ExitStack, contextmanager
from pathlib import Path

from srgc_rebuttal.runtime import Busy, atomic_json, lease

from .model_resume import GemmaResumeFirst as FrozenGemmaResumeFirst
from .model_resume import LlamaResumeFirst as FrozenLlamaResumeFirst
from .model_resume import ModelResumeFirst as FrozenBudgetedResumeFirst
from .qwen_resume import ResumeFirst as FrozenResumeFirst


class ReadyClaims:
    """Retry existing errors first, then resume other work and claim fresh tasks."""

    # Keep the historical ledger name for compatibility; dispatch no longer
    # reads, resets or enforces its failure budget.
    retry_revision = "failed-first-resume-budget-20261017-v1"

    def failure_counts(self):
        return FrozenResumeFirst.failure_counts(self)

    def dataset_allowed(self, queue):
        phase = os.environ.get("SRGC_DATASET_PHASE")
        if phase is not None and phase not in {"math", "mbpp"}:
            raise ValueError("SRGC_DATASET_PHASE must be math or mbpp")
        if phase and queue.plan["dataset"] != phase:
            return False
        if queue.plan["dataset"] == "mbpp":
            # A completed endpoint can still be owned while its writer exits.
            # Do not cross the dataset barrier until those leases are released.
            return all(q.complete(task) and not self.owned(q, task)
                       for q, task in self.tasks.values() if q.plan["dataset"] == "math")
        return True

    def priorities(self, pending):
        priorities = {key: 1 if key in pending else 2 for key in self.tasks}
        for key in pending:
            queue, task = self.tasks[key]
            if self.receipt(queue, task).get("status") != "failed":
                continue
            priorities[key] = 0
            # Recover missing cache/prefix evidence for the failed seed before
            # unrelated interrupted work; the ERROR itself may not be ready yet.
            parent = queue.dependency(task)
            while parent is not None:
                if not queue.complete(parent):
                    priorities[(key[0], parent.key)] = 0
                parent = queue.dependency(parent)
        return priorities

    def spent(self, key, failures):
        row = self.receipt(*self.tasks[key])
        return failures.get(key, row.get("retry_attempt", row.get("attempt", 0))
                            if row.get("status") == "failed" else 0)

    @contextmanager
    def claim(self, queue, *, retry_failed=False, max_attempts=None, retry_delay=60, worker_id=None):
        from srgc_rebuttal.cluster import stop_requested

        selected, handle = None, None
        with ExitStack() as locks:
            with lease(self.marker.with_suffix(".lock"), wait=True):
                pending = self.pending()
                priorities = self.priorities(pending)
                failures = self.failure_counts()
                allowed = {key for key, candidate_queue in self.queues.items()
                           if self.dataset_allowed(candidate_queue)}
                backlog = {key for key in pending if key[0] in allowed
                           and not stop_requested(self.tasks[key][0])
                           and not self.tasks[key][0].complete(self.tasks[key][1])
                           and not self.owned(*self.tasks[key])}
                runnable_priority = 1 if backlog else 2
                for key in pending:
                    candidate_queue, candidate = self.tasks[key]
                    if stop_requested(candidate_queue) or key[0] not in allowed:
                        continue
                    row = self.receipt(candidate_queue, candidate)
                    eligible = (not candidate_queue.complete(candidate) and
                                not self.owned(candidate_queue, candidate) and candidate_queue.ready(candidate))
                    if eligible and (row.get("status") != "failed" or
                            (retry_failed and time.time() >= row.get("finished", 0) + retry_delay)):
                        runnable_priority = min(runnable_priority, priorities[key])
                # An unowned unfinished task remains ahead of fresh work even
                # while its retry delay is running. Failure counts are history.
                tasks = (sorted(queue.tasks, key=lambda task:
                                priorities[(str(queue.plan_path.resolve()), task.key)])
                         if str(queue.plan_path.resolve()) in allowed else [])
                for task in tasks:
                    key = (str(queue.plan_path.resolve()), task.key)
                    if os.environ.get("SRGC_RESUME_ONLY") == "1" and key not in pending:
                        continue
                    if priorities[key] > runnable_priority:
                        continue
                    if stop_requested(queue):
                        break
                    if queue.complete(task) or not queue.ready(task) or self.owned(queue, task):
                        continue
                    with ExitStack() as candidate_locks:
                        try:
                            handle = candidate_locks.enter_context(lease(queue.directory / "leases" / f"{task.key}.lock"))
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
                        spent = self.spent(key, failures)
                        if previous.get("status") == "failed" and (
                                not retry_failed or time.time() < previous.get("finished", 0) + retry_delay):
                            continue
                        if previous.get("status") in {"running", "failed"}:
                            attempt_id = previous.get("attempt_id") or (
                                "legacy-failed-" + hashlib.sha256(json.dumps(previous, sort_keys=True).encode()).hexdigest()
                                if previous["status"] == "failed" else f"legacy-{uuid.uuid4().hex}")
                            if not isinstance(attempt_id, str) or Path(attempt_id).name != attempt_id or attempt_id in {".", ".."}:
                                raise ValueError("invalid historical attempt ID")
                            archive = queue.directory / "attempts" / f"{attempt_id}.json"
                            if not archive.exists():
                                atomic_json(archive, {**previous, "status": "abandoned", "recovered": time.time()}
                                            if previous["status"] == "running" else previous)
                        record = {"task": task.key, "status": "running", "host": socket.gethostname(),
                            "pid": os.getpid(), "worker_id": worker_id, "attempt": previous.get("attempt", 0) + 1,
                            "retry_attempt": spent + 1, "attempt_id": uuid.uuid4().hex, "started": time.time(),
                            "resume_first": key in pending}
                        atomic_json(queue.receipt(task), record)
                        selected = task
                        # Skipped candidates release immediately. Only the chosen
                        # task's lease lives through the child process.
                        locks.enter_context(candidate_locks.pop_all())
                        print(f"{'RESUME' if key in pending else 'NEW'} {queue.plan['dataset']}:{task.key} "
                              f"recorded_failures={spent} retry_limit=none", flush=True)
                        break
            previous_fds = queue.claim_fds
            queue.claim_fds = (handle.fileno(),) if selected is not None else ()
            try:
                yield selected
            finally:
                queue.claim_fds = previous_fds


class ResumeFirst(ReadyClaims, FrozenBudgetedResumeFirst):
    model_label = "Qwen3.5"


class LlamaResumeFirst(ReadyClaims, FrozenLlamaResumeFirst):
    pass


class GemmaResumeFirst(ReadyClaims, FrozenGemmaResumeFirst):
    pass
