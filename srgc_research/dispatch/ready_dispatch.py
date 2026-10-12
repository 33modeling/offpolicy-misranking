"""Fill idle model workers without changing the frozen experiment adapters."""

import os
import socket
import time
import uuid
from contextlib import ExitStack, contextmanager
from pathlib import Path

from srgc_rebuttal.runtime import Busy, atomic_json, lease

from .model_resume import GemmaResumeFirst as FrozenGemmaResumeFirst
from .model_resume import LlamaResumeFirst as FrozenLlamaResumeFirst
from .qwen_resume import ResumeFirst as FrozenResumeFirst


class ReadyClaims:
    """Resume runnable backlog first, then claim independent ready tasks."""

    def spent(self, key, failures):
        row = self.receipt(*self.tasks[key])
        return failures.get(key, row.get("retry_attempt", row.get("attempt", 0))
                            if row.get("status") == "failed" else 0)

    @contextmanager
    def claim(self, queue, *, retry_failed=False, max_attempts=3, retry_delay=60, worker_id=None):
        from srgc_rebuttal.cluster import stop_requested

        selected, handle = None, None
        with ExitStack() as locks:
            with lease(self.marker.with_suffix(".lock"), wait=True):
                pending = self.pending()
                failures = self.failure_counts()
                runnable = False
                for key in pending:
                    candidate_queue, candidate = self.tasks[key]
                    if stop_requested(candidate_queue):
                        continue
                    row = self.receipt(candidate_queue, candidate)
                    eligible = (not candidate_queue.complete(candidate) and
                                not self.owned(candidate_queue, candidate) and
                                self.spent(key, failures) < max_attempts and candidate_queue.ready(candidate))
                    runnable = runnable or (eligible and (row.get("status") != "failed" or
                        (retry_failed and time.time() >= row.get("finished", 0) + retry_delay)))
                # A running, cooling or exhausted resume is not a completion
                # barrier. Preserve resume priority whenever it can be claimed.
                tasks = sorted(queue.tasks, key=lambda task: (str(queue.plan_path.resolve()), task.key) not in pending)
                for task in tasks:
                    key = (str(queue.plan_path.resolve()), task.key)
                    if runnable and key not in pending:
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
                        # Skipped candidates release immediately. Only the chosen
                        # task's lease lives through the child process.
                        locks.enter_context(candidate_locks.pop_all())
                        print(f"{'RESUME' if key in pending else 'NEW'} {queue.plan['dataset']}:{task.key} "
                              f"failures={spent}/{max_attempts}", flush=True)
                        break
            previous_fds = queue.claim_fds
            queue.claim_fds = (handle.fileno(),) if selected is not None else ()
            try:
                yield selected
            finally:
                queue.claim_fds = previous_fds


class ResumeFirst(ReadyClaims, FrozenResumeFirst):
    pass


class LlamaResumeFirst(ReadyClaims, FrozenLlamaResumeFirst):
    pass


class GemmaResumeFirst(ReadyClaims, FrozenGemmaResumeFirst):
    pass
