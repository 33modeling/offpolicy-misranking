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


def old_failure(task, **extra):
    row = {"task": task.key, "identity": task.identity, "status": "failed",
           "attempt": 3, "started": 10, "finished": 20, "exit_code": 1, **extra}
    atomic_json(task.receipt, row)
    return row


def test_updated_dispatcher_resumes_exhausted_failure_and_preserves_phases(tasks):
    task = tasks[0]
    previous = old_failure(task)
    atomic_json(task.output / "manifest.json", {"identity": task.identity})
    phase = task.output / "score-A.pt"
    phase.write_bytes(b"already measured; do not overwrite")
    def runner(current, handle):
        assert current == task and phase.read_bytes() == b"already measured; do not overwrite"
        return 143
    assert queue.sweep([task], runner=runner, available=lambda: True)[1] == 143
    row = queue.read_receipt(task)
    assert row["attempt"] == 0 and row["dispatch_revision"] == queue.DISPATCH_REVISION
    assert queue.read_object(Path(row["previous_receipt"])) == previous
    assert phase.read_bytes() == b"already measured; do not overwrite"


def test_updated_dispatcher_has_only_one_shared_new_budget(tasks):
    task = tasks[0]
    old_failure(task, dispatch_revision="older code")
    for attempt in (1, 2, 3):
        assert queue.sweep([task], runner=lambda *_: 9, available=lambda: True, retry_delay=0)[1] == 9
        assert queue.read_receipt(task)["attempt"] == attempt
    assert queue.sweep([task], runner=lambda *_: pytest.fail("budget was reset twice"))[0]["failed"] == 1
    assert len(list((task.root / ".queue/history" / task.key).glob("*.json"))) == 1


def exhausted_worker(task, event, results):
    event.wait(10)
    def runner(current, _handle):
        results.put(current.key)
        time.sleep(.03)
        return 9
    for _ in range(5):
        queue.sweep([task], runner=runner, available=lambda: True, retry_delay=0)
        time.sleep(.03)


def test_concurrent_nodes_do_not_each_reset_the_retry_budget(tasks):
    task = tasks[0]
    old_failure(task)
    ctx = multiprocessing.get_context("fork")
    event, results = ctx.Event(), ctx.Queue()
    workers = [ctx.Process(target=exhausted_worker, args=(task, event, results)) for _ in range(6)]
    try:
        for worker in workers:
            worker.start()
        event.set()
        for worker in workers:
            worker.join(10)
            assert worker.exitcode == 0
        assert [results.get(timeout=2) for _ in range(3)] == [task.key] * 3
        assert results.empty()
        assert queue.read_receipt(task)["attempt"] == 3
        assert len(list((task.root / ".queue/history" / task.key).glob("*.json"))) == 1
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.kill()
                worker.join()


def test_changed_identity_is_never_reset_even_after_dispatcher_update(tasks):
    task = tasks[0]
    before = old_failure(task)
    atomic_json(task.output / "manifest.json", {"identity": {**task.identity, "seed": 99}})
    counts, code = queue.sweep([task], runner=lambda *_: pytest.fail("changed measurement restarted"))
    assert counts["failed"] == 1 and code is None
    assert queue.read_receipt(task) == before
    assert not (task.root / ".queue/history").exists()


def test_startup_stderr_is_logged_outside_measurement_and_saved_in_receipt(tasks, monkeypatch, capsys):
    task = tasks[0]
    monkeypatch.setattr(queue, "command_for", lambda _: [sys.executable, "-c",
        "import sys; print('starting'); print('ERROR: missing local model', file=sys.stderr); sys.exit(1)"])
    assert queue.sweep([task], available=lambda: True)[1] == 1
    row = queue.read_receipt(task)
    path = Path(row["attempt_log"])
    assert not path.is_relative_to(task.output)
    assert "starting" in path.read_text() and "ERROR: missing local model" in path.read_text()
    assert row["error"] == "ERROR: missing local model"
    printed = capsys.readouterr()
    assert "starting" in printed.out and "missing local model" in printed.err
    assert str(path) in printed.err


def test_real_collector_startup_error_is_visible_and_does_not_pollute_measurement(tmp_path, monkeypatch):
    plan = queue.REPO / "srgc_rebuttal/experiments/additional_seeds.json"
    # The prepared inputs intentionally have no completed SR cache. Exercise
    # the actual CLI and its validation, without a fake collector or GPUs.
    inputs = queue.REPO / "srgc_rebuttal/inputs/seed-7.json"
    task = queue.Task("math", 7, plan, inputs, tmp_path / "selection-information",
                      queue.digest(plan), queue.digest(inputs))
    monkeypatch.setenv("GROUP_VOLUME", str(tmp_path))
    monkeypatch.setenv("PAIR_PYTHON", sys.executable)
    monkeypatch.setenv("OM_WORK", str(tmp_path / "work"))
    assert queue.sweep([task], available=lambda: True)[1] == 1
    row = queue.read_receipt(task)
    assert "candidate cache is incomplete" in row["error"]
    assert "candidate cache is incomplete" in Path(row["attempt_log"]).read_text()
    assert {p.name for p in task.output.iterdir()} == {".dispatch.lock", ".execution.lock"}


def test_launch_oserror_is_a_recorded_failure_not_an_abandoned_running_task(tasks, monkeypatch):
    task = tasks[0]
    monkeypatch.setattr(queue, "command_for", lambda _: ["/nonexistent/information-python"])
    assert queue.sweep([task], available=lambda: True)[1] == 1
    row = queue.read_receipt(task)
    assert row["status"] == "failed" and row["attempt"] == 1
    assert "cannot start collector" in row["error"]


def test_old_rank_log_reports_original_error_without_requiring_gpu_imports(tasks, capsys):
    task = tasks[0]
    old_failure(task, dispatch_revision=queue.DISPATCH_REVISION)
    task.output.mkdir(parents=True, exist_ok=True)
    (task.output / "task.log").write_text("[rank0]: torch.OutOfMemoryError: CUDA out of memory\nChildFailedError: ranks failed\n")
    assert queue.sweep([task])[0]["failed"] == 1
    assert "torch.OutOfMemoryError" in capsys.readouterr().err


def test_node_cleanup_leaves_other_experiment_ranks_alone(monkeypatch):
    from scripts import srgc_process_guard as guard
    from srgc_rebuttal import cluster
    def reap():
        assert not guard.is_target("python -m srgc_rebuttal.run_experiment --plan other.json")
        assert not guard.is_target("python scripts/srgc_qwen35_rank.py")
        assert guard.is_target("python /frozen/srgc_research/information_rank.py --output measurement")
    monkeypatch.setattr(guard, "reap_orphans", reap)
    monkeypatch.setattr(cluster, "gpu_identity", lambda: (_ for _ in ()).throw(Busy("occupied")))
    assert not queue.node_available()


def test_information_child_scopes_cleanup_and_restores_markers(monkeypatch):
    from scripts import srgc_process_guard as guard
    from srgc_research.dispatch import information_run
    original = guard.TARGET_MARKERS, guard.OWNER_MARKERS
    def collect(argv):
        assert argv == ["math", "collect"]
        assert guard.TARGET_MARKERS == information_run.TARGETS
        assert "srgc_research.dispatch.information_run" in guard.OWNER_MARKERS
        assert not guard.is_target("python scripts/srgc_qwen35_rank.py")
        return 9
    monkeypatch.setattr(information_run.information_cli, "main", collect)
    assert information_run.main(["math", "collect"]) == 9
    assert (guard.TARGET_MARKERS, guard.OWNER_MARKERS) == original


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


def test_footer_shows_existing_errors_last_without_relaunching_current_failures(tasks, monkeypatch, capsys):
    before = {}
    for task in tasks:
        old_failure(task, dispatch_revision=queue.DISPATCH_REVISION,
                    error=f"ERROR: actual startup failure for {task.key}")
        before[task.receipt] = task.receipt.read_bytes()
    monkeypatch.setattr(queue, "tasks_for", lambda _: tasks)
    monkeypatch.setattr(queue, "run_task", lambda *_: pytest.fail("display change relaunched failed GPU work"))
    monkeypatch.setattr(queue, "node_available", lambda: pytest.fail("displaying errors admitted GPUs"))
    assert queue.main(["all"]) == 1
    stdout = capsys.readouterr().out
    footer = stdout.split("INFORMATION FAILURE DETAILS\n")[-1]
    assert "FAILED: 10 measurement(s)" in stdout
    assert len(footer.strip().splitlines()) == 10
    assert footer.strip().endswith(f"{tasks[-1].key}: ERROR: actual startup failure for {tasks[-1].key}")
    assert ".queue" not in footer and "log:" not in footer
    assert {path: path.read_bytes() for path in before} == before


def test_footer_includes_validation_failures_without_existing_receipts(tasks, monkeypatch, capsys):
    tasks[0].inputs.write_text("changed source data")
    monkeypatch.setattr(queue, "tasks_for", lambda _: tasks)
    monkeypatch.setattr(queue, "node_available", lambda: pytest.fail("invalid inputs admitted GPUs"))
    assert queue.main(["all"]) == 1
    footer = capsys.readouterr().out.split("INFORMATION FAILURE DETAILS\n")[-1]
    assert len(footer.strip().splitlines()) == 10
    assert "source plan or cached inputs changed" in footer
    assert not any(task.receipt.exists() for task in tasks)


def test_successful_retry_is_removed_from_failure_footer(tasks, capsys):
    task = tasks[0]
    failures = {}
    assert queue.sweep([task], runner=lambda *_: 9, available=lambda: True, failures=failures)[1] == 9
    assert task.key in failures
    assert queue.sweep([task], runner=completed_runner, available=lambda: True,
                       failures=failures, retry_delay=0)[1] == 0
    assert not failures
    capsys.readouterr()
    queue.failure_summary([task], failures)
    assert task.key not in capsys.readouterr().out


def test_each_dataset_uses_its_existing_interpreter(tasks, monkeypatch):
    monkeypatch.setenv("PAIR_PYTHON", sys.executable)
    monkeypatch.setenv("SWITCH_PYTHON", "/missing/switch-python")
    assert queue.command_for(tasks[0])[0] == os.path.abspath(sys.executable)
    with pytest.raises(ValueError, match="SWITCH_PYTHON"):
        queue.command_for(tasks[1])
