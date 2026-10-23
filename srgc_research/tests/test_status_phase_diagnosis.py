"""Distinguish display errors from learner faults using unchanged run records."""

import json
import os
import time

import pytest

from srgc_rebuttal.runtime import atomic_json, lease
from srgc_research.dispatch import qwen_status as status
from srgc_research.tests.test_model_results import finish, make_cohort, saved
from srgc_research.tests.test_status_current import inspect


@pytest.fixture(params=["gemma4_it", "qwen35", "llama31", "gemma4"])
def cohort(request, tmp_path):
    return make_cohort(request.param, tmp_path)


def active(cohort, *, endpoint=False):
    folder = finish(cohort, "math")
    if not endpoint:
        (folder / "on_policy-endpoint.json").unlink()
    queue = folder.parent / ".queue"
    atomic_json(queue / "tasks/seed-5.on_policy.json", {
        "task": "seed-5.on_policy", "status": "running", "host": "current-node",
        "worker_id": "current-worker", "started": time.time() - 10, "attempt": 1})
    lock = queue / "leases/seed-5.on_policy.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.touch()
    return folder, queue


@pytest.mark.parametrize("clock_order", ["tied", "eval_newer"])
def test_old_eval_does_not_hide_next_optimizer_update(cohort, capsys, clock_order):
    folder, queue = active(cohort)
    training = folder / "cost-receipts/on_policy/train.json"
    evaluation = folder / "reward-curves/on_policy/cost-receipts/step-50/eval.json"
    atomic_json(training, {"phase": "training", "state": "started", "checkpoint": 50})
    atomic_json(evaluation, {"phase": "evaluation", "state": "started", "checkpoint": 50})
    stamp = time.time_ns()
    os.utime(training, ns=(stamp, stamp))
    eval_stamp = stamp + (1000000 if clock_order == "eval_newer" else 0)
    os.utime(evaluation, ns=(eval_stamp, eval_stamp))
    before = saved(cohort)
    with lease(queue / "leases/seed-5.on_policy.lock"):
        value = inspect(cohort, capsys)
        row = next(r for r in value["datasets"][0]["tasks"] if r["task"] == "seed-5.on_policy")
        assert row["phase_stage"] == "training"
        assert row["completed_steps"] == 50 and row["current_step"] == 51
        assert "evaluation_step" not in row
        assert status.cell(row) == "RUN 50/275"
    assert saved(cohort) == before


def test_saved_valid_endpoint_is_finalizing_while_writer_still_holds_lease(cohort, capsys):
    folder, queue = active(cohort, endpoint=True)
    atomic_json(folder / "cost-receipts/on_policy/eval.json", {
        "phase": "evaluation", "state": "started", "checkpoint": 275})
    before = saved(cohort)
    with lease(queue / "leases/seed-5.on_policy.lock"):
        value = inspect(cohort, capsys)
        row = next(r for r in value["datasets"][0]["tasks"] if r["task"] == "seed-5.on_policy")
        assert row["status"] == "running"  # The writer still owns the task.
        assert row["phase_stage"] == "finalizing" and row["endpoint_ready"]
        assert row["saved_reward_percent"] == 50.
        assert row["completed_steps"] == 275 and row["current_step"] is None
        assert "final_evaluation" not in row
        assert status.cell(row) == "FINALIZING"
        assert "result saved 50.0%" in status.progress(row)
        assert "lease release" in status.progress(row)
        from srgc_research.dispatch.status_current import render_results
        results = "\n".join(render_results(value["datasets"][0]))
        result_line = next(line for line in results.splitlines() if "on_policy" in line)
        assert result_line.split()[2] == "50.0"
    assert saved(cohort) == before
    value = inspect(cohort, capsys)
    row = next(r for r in value["datasets"][0]["tasks"] if r["task"] == "seed-5.on_policy")
    assert status.cell(row) == "DONE 50.0%"


def test_invalid_endpoint_with_live_writer_is_not_finalizing(cohort, capsys):
    folder, queue = active(cohort, endpoint=True)
    path = folder / "on_policy-endpoint.json"
    endpoint = json.loads(path.read_text())
    endpoint["reward"] = .875  # Deliberately contradict its per-question rewards.
    atomic_json(path, endpoint)
    before = saved(cohort)
    from srgc_research.dispatch.model_status import load_reports
    with lease(queue / "leases/seed-5.on_policy.lock"):
        reports = load_reports(cohort.model, "math", cohort.root, cohort.env)
    row = next(r for r in reports[0]["tasks"] if r["task"] == "seed-5.on_policy")
    assert not row.get("endpoint_ready")
    assert reports[0]["errors"] and not reports[0]["complete"]
    assert status.cell(row) != "FINALIZING" and not status.cell(row).startswith("DONE")
    assert saved(cohort) == before


def test_model_queue_footer_does_not_mix_other_roots_and_model_protocols(tmp_path):
    root = tmp_path / "gemma-it"
    base = {"status": "running", "model": "gemma4", "status_model": "gemma4_it",
            "root": str(root), "heartbeat_age_seconds": 0, "active_dataset": "math",
            "order": ["gemma4", "qwen35", "llama31"], "started": time.time()}
    coordinators = [
        {**base, "host": "current-gemma-node"},
        {**base, "host": "other-root-node", "root": str(tmp_path / "old-run")},
        {**base, "host": "old-pt-node", "status_model": "gemma4_text"},
        {**base, "host": "qwen-node", "model": "qwen35", "status_model": "qwen35"},
    ]
    text = status.render([], root, status_model="gemma4_it", coordinators=coordinators)
    assert "current-gemma-node" in text
    assert "other-root-node" not in text and "old-pt-node" not in text and "qwen-node" not in text
