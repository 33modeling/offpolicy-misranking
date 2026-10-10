"""Real shared-file claims, independent processes, node loss and evidence audits."""

import json
import multiprocessing
import os
import signal
import sys
import time
from pathlib import Path

import pytest

from srgc_rebuttal.runtime import Busy, atomic_json, lease
from srgc_research.dispatch import information_queue as queue


def finish(task):
    task.output.mkdir(parents=True, exist_ok=True)
    atomic_json(task.output / "manifest.json", {"identity": task.identity})
    (task.output / "inputs.json").write_bytes(task.inputs.read_bytes())
    (task.output / "plan.json").write_bytes(task.plan.read_bytes())
    for phase in queue.PHASES:
        tensor = task.output / f"{phase}.pt"
        tensor.write_bytes(f"{task.key}:{phase}".encode())
        atomic_json(task.output / f"{phase}.json", {"artifacts": [
            {"file": tensor.name, "sha256": queue.digest(tensor)}]})
    atomic_json(task.output / "endpoint.json", {"identity": task.identity,
        "phases": {phase: f"{phase}.json" for phase in queue.PHASES}})


def audit(folder):
    endpoint = queue.read_object(folder / "endpoint.json")
    for name in endpoint["phases"].values():
        for artifact in queue.read_object(folder / name)["artifacts"]:
            if queue.digest(folder / artifact["file"]) != artifact["sha256"]:
                raise ValueError("changed measured tensors")
    return endpoint, {}


@pytest.fixture
def tasks(tmp_path, monkeypatch):
    plan, inputs = tmp_path / "source-plan.json", tmp_path / "source-inputs.json"
    plan.write_text('{"model": "test"}')
    inputs.write_text('{"cache": "immutable"}')
    monkeypatch.setattr(queue, "read_measurement", audit)
    return [queue.Task(dataset, seed, plan, inputs, tmp_path / "output",
                       queue.digest(plan), queue.digest(inputs))
            for seed in queue.SEEDS for dataset in ("math", "mbpp")]


def completed_runner(task, handle):
    finish(task)
    return 0


def test_all_tasks_complete_and_restart_launches_none(tasks):
    for _ in tasks:
        assert queue.sweep(tasks, runner=completed_runner, available=lambda: True)[1] == 0
    counts, code = queue.sweep(tasks, runner=lambda *_: pytest.fail("duplicate launch"))
    assert code is None and counts["complete"] == 10
    assert not list(tasks[0].root.rglob("*.html"))


@pytest.mark.parametrize("lock", [".dispatch.lock", ".execution.lock", "queue"])
def test_existing_manual_or_queue_owner_is_skipped(tasks, lock):
    path = tasks[0].receipt.with_suffix(".lock") if lock == "queue" else tasks[0].output / lock
    selected = []
    with lease(path):
        counts, code = queue.sweep(tasks, runner=lambda task, _: selected.append(task.key) or 143,
                                   available=lambda: True)
    assert code == 143 and counts["busy"] == 1
    assert selected == [tasks[1].key]
    assert not tasks[0].receipt.exists()


def concurrent_worker(tasks, event, results):
    event.wait(10)
    def runner(task, handle):
        results.put(task.key)
        time.sleep(.15)
        finish(task)
        return 0
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        counts, _code = queue.sweep(tasks, runner=runner, available=lambda: True)
        if counts["complete"] == len(tasks):
            return
        time.sleep(.01)
    raise AssertionError("queue did not drain")


def test_six_processes_share_math_and_all_without_duplicate_claims(tasks):
    ctx = multiprocessing.get_context("fork")
    event, results = ctx.Event(), ctx.Queue()
    # Dataset-specific workers must share the same claims with `all` workers.
    workers = [ctx.Process(target=concurrent_worker, args=(
        tasks if i % 2 else [t for t in tasks if t.dataset == "math"], event, results))
        for i in range(6)]
    try:
        for worker in workers:
            worker.start()
        event.set()
        for worker in workers:
            worker.join(15)
            assert worker.exitcode == 0
        keys = [results.get(timeout=2) for _ in tasks]
        assert sorted(keys) == sorted(t.key for t in tasks)
        assert results.empty()
        assert queue.sweep(tasks)[0]["complete"] == len(tasks)
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.kill()
                worker.join()


def test_node_loss_resumes_partial_data_without_using_failure_retry(tasks):
    task = tasks[0]
    atomic_json(task.receipt, {"task": task.key, "identity": task.identity,
                             "status": "running", "attempt": 3, "started": 10})
    atomic_json(task.output / "manifest.json", {"identity": task.identity})
    saved = task.output / "score-A.json"
    saved.write_text('{"saved": "phase must survive"}')
    before = saved.read_bytes()
    def runner(current, handle):
        assert current == task and saved.read_bytes() == before
        return 143
    assert queue.sweep([task], runner=runner, available=lambda: True)[1] == 143
    assert queue.read_object(task.receipt)["attempt"] == 2
    assert saved.read_bytes() == before


def test_claim_survives_killed_parent_until_child_exits(tasks, monkeypatch, tmp_path):
    task = tasks[0]
    ready = tmp_path / "child-ready"
    child_script = tmp_path / "child.py"
    child_script.write_text("import os, sys, time\nfrom pathlib import Path\n"
                            "Path(sys.argv[1]).write_text(str(os.getpid()))\n"
                            "time.sleep(20)\n")
    monkeypatch.setattr(queue, "command_for", lambda _: [sys.executable, str(child_script), str(ready)])
    def worker():
        queue.sweep([task], available=lambda: True)
    ctx = multiprocessing.get_context("fork")
    parent = ctx.Process(target=worker)
    parent.start()
    child_pid = None
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert ready.exists()
        child_pid = int(ready.read_text())
        parent.kill()
        parent.join(5)
        counts, code = queue.sweep([task], runner=lambda *_: pytest.fail("claim was lost"))
        assert code is None and counts["busy"] == 1
        os.kill(child_pid, signal.SIGTERM)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                with lease(task.receipt.with_suffix(".lock")):
                    break
            except Busy:
                time.sleep(.01)
        else:
            pytest.fail("claim was not released after child exit")
        assert queue.sweep([task], runner=completed_runner, available=lambda: True)[1] == 0
    finally:
        if parent.is_alive():
            parent.kill()
            parent.join()
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_sigterm_waits_for_owned_child_and_refunds_attempt(tasks, monkeypatch, tmp_path):
    task = tasks[0]
    ready, stopped = tmp_path / "ready", tmp_path / "stopped"
    script = tmp_path / "child.py"
    script.write_text("import signal, sys, time\nfrom pathlib import Path\n"
                      "def stop(*args):\n    Path(sys.argv[2]).touch()\n    sys.exit(0)\n"
                      "signal.signal(signal.SIGTERM, stop)\nPath(sys.argv[1]).touch()\n"
                      "time.sleep(20)\n")
    monkeypatch.setattr(queue, "command_for", lambda _: [sys.executable, str(script), str(ready), str(stopped)])
    ctx = multiprocessing.get_context("fork")
    def worker():
        _, code = queue.sweep([task], available=lambda: True)
        sys.exit(code)
    parent = ctx.Process(target=worker)
    parent.start()
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert ready.exists()
        os.kill(parent.pid, signal.SIGTERM)
        parent.join(5)
        assert parent.exitcode == 143 and stopped.exists()
        assert queue.read_object(task.receipt)["attempt"] == 0
        with lease(task.receipt.with_suffix(".lock")):
            pass
    finally:
        if parent.is_alive():
            parent.kill()
            parent.join()


@pytest.mark.parametrize("code", [75, 130, 143])
def test_busy_and_interruptions_do_not_exhaust_retries(tasks, code):
    task = tasks[0]
    for _ in range(5):
        assert queue.sweep([task], runner=lambda *_: code, available=lambda: True)[1] == code
    assert queue.read_object(task.receipt)["attempt"] == 0


def test_busy_gpu_does_not_claim_or_charge_any_task(tasks):
    assert queue.sweep(tasks, runner=lambda *_: pytest.fail("busy node launched"),
                       available=lambda: False)[1] == 75
    assert not any(t.receipt.exists() for t in tasks)


@pytest.mark.parametrize("busy_at", [None, "identity", "lease"])
def test_node_admission_uses_canonical_gpu_leases_and_restores_guard_markers(tmp_path, monkeypatch, busy_at):
    from contextlib import contextmanager

    from scripts import srgc_process_guard as guard
    from srgc_rebuttal import cluster
    original = guard.TARGET_MARKERS, guard.OWNER_MARKERS
    seen = []
    def reap():
        assert "srgc_research/information_rank.py" in guard.TARGET_MARKERS
        assert "srgc_research.information_cli" in guard.OWNER_MARKERS
        assert "srgc_research.dispatch.information_queue" in guard.OWNER_MARKERS
        seen.append("reap")
    def gpu_identity():
        if busy_at == "identity":
            raise Busy("occupied GPU memory")
        return "0,1,2,3", ("u0", "u1", "u2", "u3")
    @contextmanager
    def leases(root, uuids):
        assert root == tmp_path and len(uuids) == 4
        seen.append("lease")
        if busy_at == "lease":
            raise Busy("another launcher owns the GPU")
        yield ()
    monkeypatch.setattr(guard, "reap_orphans", reap)
    monkeypatch.setattr(guard, "canonical_lock_root", lambda: tmp_path)
    monkeypatch.setattr(cluster, "gpu_identity", gpu_identity)
    monkeypatch.setattr(cluster, "device_leases", leases)
    assert queue.node_available() is (busy_at is None)
    assert seen[0] == "reap" and (guard.TARGET_MARKERS, guard.OWNER_MARKERS) == original


def test_unsupported_gpu_is_reported_without_spending_a_task_retry(tasks, monkeypatch):
    from scripts import srgc_process_guard as guard
    from srgc_rebuttal import cluster
    original = guard.TARGET_MARKERS, guard.OWNER_MARKERS
    monkeypatch.setattr(guard, "reap_orphans", lambda: None)
    def invalid():
        raise ValueError("four full H100 GPUs required")
    monkeypatch.setattr(cluster, "gpu_identity", invalid)
    with pytest.raises(ValueError, match="H100"):
        queue.sweep(tasks)
    assert not any(t.receipt.exists() for t in tasks)
    assert (guard.TARGET_MARKERS, guard.OWNER_MARKERS) == original


def test_changing_evidence_during_audit_is_rejected(tasks, monkeypatch):
    task = tasks[0]
    finish(task)
    def racing_audit(folder):
        result = audit(folder)
        (folder / "score-A.pt").write_bytes(b"changed after hash check")
        return result
    monkeypatch.setattr(queue, "read_measurement", racing_audit)
    assert queue.sweep([task])[0]["failed"] == 1
    assert not task.receipt.exists()


def test_failed_task_cools_down_and_other_tasks_continue(tasks):
    first = tasks[0]
    assert queue.sweep(tasks, runner=lambda *_: 9, available=lambda: True, now=lambda: 10)[1] == 9
    selected = []
    counts, code = queue.sweep(tasks, runner=lambda t, _: selected.append(t.key) or 143,
                              available=lambda: True, now=lambda: 11)
    assert counts["waiting"] == 1 and code == 143 and selected == [tasks[1].key]
    for attempt in (2, 3):
        assert queue.sweep([first], runner=lambda *_: 9, available=lambda: True,
                           retry_delay=0)[1] == 9
        assert queue.read_object(first.receipt)["attempt"] == attempt
    assert queue.sweep([first], runner=lambda *_: pytest.fail("failed retry budget ignored"))[0]["failed"] == 1


def test_zero_exit_without_real_endpoint_is_failure(tasks):
    assert queue.sweep(tasks, runner=lambda *_: 0, available=lambda: True)[1] == 1
    assert queue.read_object(tasks[0].receipt)["status"] == "failed"


def test_completed_evidence_is_reaudited_after_tensor_changes(tasks):
    task = tasks[0]
    assert queue.sweep([task], runner=completed_runner, available=lambda: True)[1] == 0
    tensor = task.output / "score-A.pt"
    tensor.write_bytes(b"corrupt")
    counts, code = queue.sweep([task], runner=lambda *_: pytest.fail("corrupt output overwritten"))
    assert code is None and counts["failed"] == 1
    assert tensor.read_bytes() == b"corrupt"


def test_deleted_completed_endpoint_is_not_silently_rebuilt(tasks):
    task = tasks[0]
    queue.sweep([task], runner=completed_runner, available=lambda: True)
    (task.output / "endpoint.json").unlink()
    assert queue.sweep([task], runner=lambda *_: pytest.fail("completed output rerun"))[0]["failed"] == 1


@pytest.mark.parametrize("change", ["plan", "inputs", "manifest", "receipt"])
def test_changed_identities_are_preserved_and_stop_only_their_task(tasks, change):
    task = tasks[0]
    if change in {"plan", "inputs"}:
        getattr(task, change).write_text("changed")
        tasks = [task]  # The fixture's tasks deliberately share these two sources.
    elif change == "manifest":
        atomic_json(task.output / "manifest.json", {"identity": {**task.identity, "seed": 99}})
    else:
        atomic_json(task.receipt, {"task": task.key, "identity": {"wrong": True}})
    assert queue.sweep(tasks, runner=lambda *_: 143, available=lambda: True)[0]["failed"] == 1


@pytest.mark.parametrize("bad", [{"attempt": True}, {"attempt": -1}, {"finished": float("inf")},
                                 {"status": "failed"}, {"status": "wrong"}])
def test_invalid_receipt_is_never_reset(tasks, bad):
    task = tasks[0]
    row = {"task": task.key, "identity": task.identity, "status": "running", "attempt": 1, **bad}
    task.receipt.parent.mkdir(parents=True, exist_ok=True)
    task.receipt.write_text(json.dumps(row))
    before = task.receipt.read_bytes()
    assert queue.sweep([task], runner=lambda *_: pytest.fail("invalid receipt reset"))[0]["failed"] == 1
    assert task.receipt.read_bytes() == before


def test_complete_manual_output_is_discovered_without_gpu_admission(tasks):
    task = tasks[0]
    finish(task)
    assert queue.sweep([task], available=lambda: pytest.fail("completed work needs GPUs"))[0]["complete"] == 1
    assert queue.read_object(task.receipt)["attempt"] == 0


def test_dispatch_changes_preserve_existing_measurement_runtime_hash():
    from srgc_research.storage import runtime_files
    assert not any("srgc_research/dispatch" in name for name in runtime_files())
    script = Path(queue.REPO / "scripts/run_srgc_information.sh")
    assert str(script.relative_to(queue.REPO)) not in runtime_files()


def test_main_drains_queue_and_returns_failure_for_failed_tasks(tasks, monkeypatch):
    monkeypatch.setattr(queue, "tasks_for", lambda _: tasks)
    monkeypatch.setattr(queue, "run_task", completed_runner)
    monkeypatch.setattr(queue, "node_available", lambda: True)
    assert queue.main(["all"]) == 0
    tasks[0].inputs.write_text("changed source")
    assert queue.main(["all"]) == 1


def test_each_dataset_uses_its_existing_interpreter(tasks, monkeypatch):
    monkeypatch.setenv("PAIR_PYTHON", sys.executable)
    monkeypatch.setenv("SWITCH_PYTHON", "/missing/switch-python")
    assert queue.command_for(tasks[0])[0] == os.path.abspath(sys.executable)
    with pytest.raises(ValueError, match="SWITCH_PYTHON"):
        queue.command_for(tasks[1])
