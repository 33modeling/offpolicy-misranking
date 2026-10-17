"""Expired worker rows are a view concern; saved evidence and leases survive."""

import hashlib
import json
import time

import pytest

from srgc_rebuttal.runtime import atomic_json, lease
from srgc_research.dispatch.qwen_status import node_rows, render
from srgc_research.dispatch.worker_history import prune_workers
from srgc_research.tests.test_model_results import (
    cohort as cohort,  # noqa: PLC0414 -- Register all three prepared model fixtures.
)
from srgc_research.tests.test_model_results import finish, saved
from srgc_research.tests.test_model_status import read_status


def worker(name, age, *, state="heartbeat_stale", **details):
    return {"worker_id": name, "host": name, "status": state,
            "heartbeat_age_seconds": age, **details}


def report(root, workers, tasks=()):
    return {"output_root": str(root), "label": "MATH", "workers": workers, "tasks": list(tasks)}


def test_retention_boundary_and_unknown_ages_are_not_silently_removed(tmp_path):
    rows = [worker("recent", 120), worker("boundary", 900), worker("expired", 901),
            worker("finished", 3600, state="complete"), worker("unknown", None),
            worker("nan", float("nan")), worker("fresh-running", 3, state="running")]
    value = report(tmp_path, rows)
    prune_workers([value])
    assert {w["worker_id"] for w in value["workers"]} == {"recent", "boundary", "unknown", "nan", "fresh-running"}
    assert value["worker_history"] == {"retention_seconds": 900,
                                      "hidden_worker_ids": ["expired", "finished"], "hidden_count": 2}
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("lock_kind", ("task", "execution"))
def test_live_child_keeps_stale_session_in_both_datasets_even_with_invalid_inputs(tmp_path, lock_kind):
    root = tmp_path / "math"
    task = {"task": "seed-5.sr", "seed": 5, "arm": "sr", "status": "invalid", "worker_id": "owner"}
    values = [report(root, [worker("owner", 3600)], [task]),
              report(tmp_path / "mbpp", [worker("owner", 3600)])]
    lock = (root / ".queue/leases/seed-5.sr.lock" if lock_kind == "task"
            else root / "seed-5/.sr.execution.lock")
    with lease(lock):
        prune_workers(values)
        assert all(r["workers"][0]["worker_id"] == "owner" for r in values)
    prune_workers(values)
    assert all(not r["workers"] for r in values)
    assert lock.is_file()


def test_gpu_admission_lock_keeps_stale_preflight_without_writing_any_files(tmp_path):
    locks = tmp_path / "gpu-locks"
    path = locks / f"{hashlib.sha256(b'GPU-test').hexdigest()}.lock"
    value = report(tmp_path / "runs", [worker("admission", 3600, gpu_uuids=["GPU-test"])])
    with lease(path):
        before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
        prune_workers([value], gpu_lock_roots=[locks])
        assert len(value["workers"]) == 1
        assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
    prune_workers([value], gpu_lock_roots=[locks])
    assert not value["workers"]
    assert path.is_file()


def test_heartbeat_stale_child_with_owned_task_wins_over_new_idle_session(tmp_path):
    task = {"task": "seed-5.sr", "seed": 5, "arm": "sr", "status": "running", "worker_id": "child"}
    values = [report(tmp_path, [worker("child", 3600, host="h100", task=task["task"]),
                                worker("idle", 0, host="h100", state="idle")], [task])]
    prune_workers(values)
    nodes = node_rows(values)
    assert len(nodes) == 1 and nodes[0]["worker_id"] == "child"


def test_reallocated_gpu_does_not_keep_old_unowned_session_forever(tmp_path):
    locks = tmp_path / "gpu-locks"
    path = locks / f"{hashlib.sha256(b'GPU-test').hexdigest()}.lock"
    value = report(tmp_path / "runs", [worker("old-host", 86400, gpu_uuids=["GPU-test"]),
                                         worker("new-host", 1, state="preflight", gpu_uuids=["GPU-test"])])
    with lease(path):
        prune_workers([value], gpu_lock_roots=[locks])
        assert [w["worker_id"] for w in value["workers"]] == ["new-host"]


def test_shared_preflight_receipts_remain_visible_despite_heartbeat_write_skew(tmp_path):
    locks = tmp_path / "gpu-locks"
    path = locks / f"{hashlib.sha256(b'GPU-test').hexdigest()}.lock"
    values = [report(tmp_path / "math", [worker("shared", 3600.1, gpu_uuids=["GPU-test"])]),
              report(tmp_path / "mbpp", [worker("shared", 3600.0, gpu_uuids=["GPU-test"])])]
    with lease(path):
        prune_workers(values, gpu_lock_roots=[locks])
        assert all([w["worker_id"] for w in value["workers"]] == ["shared"] for value in values)
        assert all(value["worker_history"]["hidden_count"] == 0 for value in values)


def test_all_public_models_hide_expired_workers_keep_owned_work_and_preserve_files(cohort, capsys):
    folder = finish(cohort, "math")
    queue = folder.parent / ".queue"
    for name, seconds in (("expired", 3600), ("recent", 120), ("owner", 3600)):
        atomic_json(queue / "workers" / f"{name}.json", {
            "worker_id": name, "host": name, "status": "running",
            "heartbeat": time.time() - seconds, "task": "seed-5.sr" if name == "owner" else None})
    atomic_json(queue / "tasks/seed-5.sr.json", {
        "task": "seed-5.sr", "status": "running", "worker_id": "owner", "host": "owner", "attempt": 1})
    atomic_json(queue / "tasks/seed-5.switch.json", {
        "task": "seed-5.switch", "status": "running", "worker_id": "expired", "host": "expired", "attempt": 1})
    checkpoint = folder / "switch-latest.pt"
    checkpoint.write_bytes(b"checkpoint evidence must survive status")
    with lease(queue / "leases/seed-5.sr.lock"):
        before = saved(cohort)
        code, value = read_status(cohort, capsys)
        assert code == 0
        result = value["datasets"][0]
        assert {w["worker_id"] for w in result["workers"]} == {"recent", "owner"}
        assert result["worker_history"]["hidden_worker_ids"] == ["expired"]
        recoverable = next(r for r in result["tasks"] if r["task"] == "seed-5.switch")
        assert recoverable["status"] == "recoverable"
        text = render(value["datasets"], cohort.root)
        assert "HISTORY  1 expired unowned worker sessions hidden" in text
        assert "recent" in text and "owner" in text and "RESUME" in text
        assert saved(cohort) == before
    assert json.loads((queue / "workers/expired.json").read_text())["worker_id"] == "expired"
    assert checkpoint.read_bytes() == b"checkpoint evidence must survive status"
