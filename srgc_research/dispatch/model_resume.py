"""Bounded retries after an operational fix without resetting scientific runs."""

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

from scripts.srgc_log_tail import tail_lines
from srgc_rebuttal.runtime import atomic_json
from srgc_research.dispatch.qwen_resume import ResumeFirst


class ModelResumeFirst(ResumeFirst):
    """Preserve all failed attempts; grant one new retry budget per dispatcher fix."""

    model_label = "MODEL"
    retry_revision = "model-retry-budget-v1"

    def failure_counts(self):
        counts = super().failure_counts()
        path = self.root / ".dispatch" / f"{self.retry_revision}.json"
        plans = {key: {name: queue.protocol[name] for name in ("plan_sha256", "implementation_sha256")}
                 for key, queue in self.queues.items()}
        saved = json.loads(path.read_text()) if path.exists() else None
        if saved is None:
            # Old versions may have only the latest failed receipt, without
            # an attempt archive. Preserve that real evidence before retrying.
            for queue, task in self.tasks.values():
                row = self.receipt(queue, task)
                if row.get("status") != "failed":
                    continue
                attempt_id = row.get("attempt_id") or "legacy-failed-" + hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
                if not isinstance(attempt_id, str) or Path(attempt_id).name != attempt_id or attempt_id in {".", ".."}:
                    raise ValueError("invalid historical failed attempt ID")
                archive = queue.directory / "attempts" / f"{attempt_id}.json"
                if not archive.exists():
                    atomic_json(archive, row)
            counts = super().failure_counts()
            baselines = []
            for key, (queue, task) in sorted(self.tasks.items()):
                baselines.append([*key, counts.get(key, 0)])
            saved = {"schema": "srgc-operational-retry-budget-v1", "revision": self.retry_revision,
                     "plans": plans, "baselines": baselines}
            # Called only by claim while the shared resume dispatch lease is held.
            atomic_json(path, saved)
            if any(row[2] for row in baselines):
                print(f"{self.model_label} RESUME: renewed bounded retries after dispatcher fix; historical attempts/checkpoints preserved", flush=True)
        if (not isinstance(saved, dict) or saved.get("schema") != "srgc-operational-retry-budget-v1"
                or saved.get("revision") != self.retry_revision or saved.get("plans") != plans):
            raise ValueError(f"{self.model_label} retry budget belongs to different plans/code")
        rows = saved.get("baselines")
        if not isinstance(rows, list) or any(not isinstance(row, list) or len(row) != 3 or
                any(not isinstance(value, str) for value in row[:2]) or type(row[2]) is not int or row[2] < 0 for row in rows):
            raise ValueError(f"invalid {self.model_label} retry baselines")
        baselines = {(row[0], row[1]): row[2] for row in rows}
        if len(baselines) != len(rows) or baselines.keys() != self.tasks.keys():
            raise ValueError(f"{self.model_label} retry baseline task set changed")
        remaining = {}
        for key, (queue, task) in self.tasks.items():
            actual = counts.get(key, 0)
            if actual < baselines[key]:
                raise ValueError(f"{self.model_label} failed-attempt history shrank: {task.key}")
            remaining[key] = actual - baselines[key]
            row = self.receipt(queue, task)
            if row.get("status") == "failed" and remaining[key] == 0 and row.get("retry_attempt", row.get("attempt", 0)):
                # The frozen worker also inspects receipt counters while a
                # retry is cooling down. Preserve launch IDs/history but make
                # that operational counter reflect the renewed budget.
                atomic_json(queue.receipt(task), {**row, "retry_attempt": 0,
                            "retry_budget_revision": self.retry_revision})
        return remaining

    @contextmanager
    def claim(self, queue, **kwargs):
        try:
            with super().claim(queue, **kwargs) as task:
                yield task
        except RuntimeError as error:
            if str(error) == "unfinished Qwen tasks reached the real failure limit; fresh tasks remain blocked":
                raise RuntimeError(f"unfinished {self.model_label} tasks exhausted this dispatcher's bounded retries; failure details are printed below") from error
            raise


class LlamaResumeFirst(ModelResumeFirst):
    model_label = "Llama-3.1"
    retry_revision = "llama31-resume-budget-v1"


class GemmaResumeFirst(ModelResumeFirst):
    model_label = "Gemma-4-12B-PT"
    retry_revision = "gemma4-resume-budget-v1"


def failure_footer(root, plan_paths, label):
    """Read actual task receipts, so traceback details follow backup shutdown logs."""
    from srgc_rebuttal.plan import load_plan
    from srgc_rebuttal.runtime import run_root

    failures = []
    for plan_path in plan_paths:
        try:
            plan_path = Path(plan_path)
            plan = load_plan(plan_path)
            queue = run_root(plan_path, plan) / ".queue"
            for receipt in sorted((queue / "tasks").glob("*.json")):
                row = json.loads(receipt.read_text())
                key = row.get("task", "")
                if row.get("status") != "failed" or not isinstance(key, str) or Path(key).name != key:
                    continue
                failures.append((plan["dataset"], key, queue / "logs" / f"{key}.log"))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    if failures:
        print(f"\n{label} FAILURE DETAILS", flush=True)
    for dataset, key, log in failures:
        print(f"{dataset}:{key}", flush=True)
        try:
            for line in tail_lines(log, lines=25):
                print(line, flush=True)
        except OSError as error:
            print(f"log unavailable: {error}", flush=True)
