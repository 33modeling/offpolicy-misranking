"""One failed RLOO branch must not prevent independent work on healthy GPUs."""
from contextlib import contextmanager
import errno
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import queue_rloo as queue


@pytest.fixture
def work(tmp_path, monkeypatch):
    e = queue.experiment
    completed, calls = set(), []
    monkeypatch.setattr(e, "validate", lambda out: ({}, {}))
    monkeypatch.setattr(e, "complete", lambda out, arm: (out, arm) in completed)
    monkeypatch.setattr(e, "report", lambda out: None)

    def run(out, arm, seconds):
        calls.append((out, arm))
        completed.add((out, arm))

    monkeypatch.setattr(e, "run_arm", run)
    return e, completed, calls, run


@pytest.mark.parametrize("error", [ValueError("invalid branch"), RuntimeError("GPU worker failed"),
                                   BlockingIOError(errno.EAGAIN, "worker resource failure")])
def test_failure_isolated_and_runtime_error_requires_readmission(tmp_path, monkeypatch, work, error):
    e, completed, calls, original = work
    attempts, probes = [], []

    def run(out, arm, seconds):
        attempts.append((out, arm))
        if out == tmp_path / "math500-d0/s0" and arm == "random":
            raise error
        original(out, arm, seconds)

    monkeypatch.setattr(e, "run_arm", run)
    monkeypatch.setattr(queue, "admit_after_failure", lambda root: probes.append(root))
    assert queue.run(tmp_path, 60) == 1
    assert len(attempts) == 24 and len(completed) == 23
    assert len(probes) == (0 if isinstance(error, ValueError) else 1)
    assert e.ed.read(tmp_path / "math500-d0/s0/random/queue-attempt.json")["state"] == "FAILED"


def test_broken_completion_receipt_does_not_stop_other_points(tmp_path, monkeypatch, work):
    e, completed, calls, _ = work
    complete = e.complete

    def check(out, arm):
        if out == tmp_path / "math500-d0/s0" and arm == "before":
            raise ValueError("evaluation seal mismatch")
        return complete(out, arm)

    monkeypatch.setattr(e, "complete", check)
    assert queue.run(tmp_path, 60) == 1
    assert len(calls) == 23


def test_bad_node_stops_before_next_gpu_assignment(tmp_path, monkeypatch, work):
    e, _, calls, _ = work
    attempted = []

    def fail(*args):
        attempted.append(args)
        raise RuntimeError("CUDA failed")

    monkeypatch.setattr(e, "run_arm", fail)
    monkeypatch.setattr(queue, "admit_after_failure", lambda root: (_ for _ in ()).throw(RuntimeError("NCCL failed")))
    assert queue.run(tmp_path, 60) == 78
    assert len(attempted) == 1 and not calls


def test_busy_task_yields_without_waiting_or_touching_owner(tmp_path, monkeypatch, work):
    e, _, calls, _ = work
    lock = e.lock

    @contextmanager
    def claim(path, **kwargs):
        if path == tmp_path / "math500-d0/s0/before/.worker.lock":
            raise BlockingIOError("peer owns task")
        with lock(path, **kwargs):
            yield

    monkeypatch.setattr(e, "lock", claim)
    assert queue.run(tmp_path, 60) == 75
    assert len(calls) == 23


def test_completed_work_never_restarts_and_failure_receipt_clears(tmp_path, work):
    e, completed, calls, _ = work
    path = tmp_path / "math500-d0/s0/random"
    queue.record(path, "FAILED", "previous failure")
    assert queue.run(tmp_path, 60) == 0
    assert len(calls) == 24
    assert queue.run(tmp_path, 60) == 0
    assert len(calls) == 24
    assert e.ed.read(path / "queue-attempt.json")["state"] == "DONE"


def test_signal_unwinds_task_lease_and_is_not_swallowed(tmp_path, monkeypatch, work):
    e, _, calls, _ = work
    monkeypatch.setattr(e, "run_arm", lambda *a: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        queue.run(tmp_path, 60)
    with e.lock(tmp_path / "math500-d0/s0/before/.worker.lock"):
        pass
    assert not calls


def test_invalid_root_contract_is_rejected_before_any_gpu_work(tmp_path, monkeypatch, work):
    e, _, calls, _ = work
    monkeypatch.setattr(e, "validate", lambda *a: (_ for _ in ()).throw(ValueError("frozen code changed")))
    with pytest.raises(ValueError):
        queue.run(tmp_path, 60)
    assert not calls


def test_launcher_sigterm_reaps_worker_and_closes_cost(tmp_path):
    from test_selection_worker_shutdown import alive, make_worker, wait_until
    import selection_gate_gpu as base
    command = make_worker(tmp_path, "")
    probe = tmp_path / "queue-probe.py"
    probe.write_text('''
import sys
from pathlib import Path
import selection_gate_gpu as base
sys.path.insert(0, str(base.ROOT / "scripts"))
import queue_rloo as queue
root, child = Path(sys.argv[1]), sys.argv[2]
queue.experiment.POINTS = ((0, 0),)
queue.experiment.validate = lambda out: ({}, {})
queue.experiment.complete = lambda out, arm: False
def run(out, arm, seconds):
    base.meter(out / arm, "train", "cpu-process-test", devices=1, timeout=seconds,
               commands=[([sys.executable, child, str(root / "worker.pid")], "")])
queue.experiment.run_arm = run
sys.argv = ["queue", "--root", str(root), "--max-phase-seconds", "120"]
raise SystemExit(queue.main())
''')
    launcher = subprocess.Popen([
        "bash", "-c", 'source "$1"; shift; selection_run_worker "$@"', "rloo-test",
        str(base.ROOT / "scripts/_selection_worker.sh"), sys.executable, str(probe),
        str(tmp_path), command[1]], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=True)
    try:
        wait_until(lambda: (tmp_path / "worker.pid").exists() or launcher.poll() is not None, timeout=10)
        assert launcher.poll() is None, launcher.communicate(timeout=3)
        pid = int((tmp_path / "worker.pid").read_text())
        launcher.send_signal(signal.SIGTERM)
        output = launcher.communicate(timeout=15)
        assert launcher.returncode == 143, output
        assert not alive(pid)
        directory = tmp_path / "math500-d0/s0/before"
        assert base.cost(directory)["complete"]
        with queue.experiment.lock(directory / ".worker.lock"):
            pass
        assert not (directory.parent / "random").exists()
    finally:
        if launcher.poll() is None:
            os.killpg(launcher.pid, signal.SIGKILL)
        launcher.communicate(timeout=5)
        marker = tmp_path / "worker.pid"
        if marker.exists() and alive(int(marker.read_text())):
            os.killpg(int(marker.read_text()), signal.SIGKILL)
