"""Queue barriers share real state leases and completed reruns spend no GPU cost."""

import contextlib
import multiprocessing
import sys
import time

import pytest

import selection_gate as core
import selection_gate_gpu as base
import selector_pair_gpu as gpu
from test_selector_pair_gpu import fake_study


@contextlib.contextmanager
def peer_locks(paths):
    """A separately forked process owns these real flocks until released."""
    context = multiprocessing.get_context("fork")
    ready, release = context.Event(), context.Event()

    def peer():
        with contextlib.ExitStack() as stack:
            for path in paths:
                stack.enter_context(base.lease(path))
            ready.set()
            if not release.wait(15):
                raise AssertionError("test did not release its peer state owner")

    process = context.Process(target=peer)
    process.start()
    try:
        assert ready.wait(5), "peer did not acquire state/measurement leases"
        yield process, release
    finally:
        release.set()
        process.join(5)
        if process.is_alive():
            process.terminate()  # Only the test-created process, never a real worker.
            process.join(5)
        assert process.exitcode == 0


def complete_study(root, protocol):
    gpu.develop(root, protocol, [])
    gpu.fit(root, protocol)
    gpu.freeze(root, protocol)
    gpu.test(root, protocol, [])


def cpu_main(monkeypatch, root, protocol, command):
    monkeypatch.setattr(sys, "argv", ["pair", command, "--root", str(root)])
    monkeypatch.setattr(gpu, "install_runtime", lambda: None)
    monkeypatch.setattr(gpu, "ensure_prepared", lambda path: protocol)
    monkeypatch.setattr(gpu, "manifest", lambda *args, **kwargs: protocol)
    monkeypatch.setattr(gpu, "resource_diagnostics", lambda: None)


def assert_exclusively_held(path):
    with pytest.raises(BlockingIOError):
        with base.lease(path):
            pytest.fail("barrier did not hold the published state's lease")


def test_fit_waits_for_peer_state_before_taking_measurement_lock(tmp_path, fake_study, monkeypatch):
    protocol, calls, states = fake_study
    gpu.develop(tmp_path, protocol, [])
    _, entries = states(tmp_path, 0, 25)
    state_lock = tmp_path / "development/s0-t25/.state.lock"
    measurement_lock = entries["on_policy"][1] / "measurement/.measurement.lock"
    diagnose = gpu.diagnostic
    fit = gpu.fit
    polls = []

    def locked_diagnostic(entry, env):
        with base.lease(entry[1] / "measurement/.measurement.lock"):
            return diagnose(entry, env)

    def checked_fit(root, p):
        assert_exclusively_held(state_lock)
        return fit(root, p)

    monkeypatch.setattr(gpu, "diagnostic", locked_diagnostic)
    monkeypatch.setattr(gpu, "fit", checked_fit)
    original_sleep = time.sleep
    with peer_locks([state_lock, measurement_lock]) as (peer, release):
        def poll(seconds):
            assert not (tmp_path / "model.json").exists(), "fit started before peer state was released"
            polls.append(seconds)
            release.set()
            peer.join(5)
            assert peer.exitcode == 0
            original_sleep(.001)

        monkeypatch.setattr(gpu.time, "sleep", poll)
        gpu.run_distributed(tmp_path, protocol, None, "fit")

    assert polls and len(calls) == 18
    assert (tmp_path / "model.json").exists()
    assert core.read(tmp_path / "development-pass.json")["state"] == "DONE"
    assert not (tmp_path / "test-decisions.json").exists()


def test_manual_report_waits_for_published_test_state_and_locks_both_stages(tmp_path, fake_study, monkeypatch):
    protocol, calls, _ = fake_study
    complete_study(tmp_path, protocol)
    test_lock = tmp_path / "test/s3-t25/.state.lock"
    development_lock = tmp_path / "development/s0-t25/.state.lock"
    report = gpu.report
    polls = []

    def checked_report(root, p):
        assert_exclusively_held(development_lock)
        assert_exclusively_held(test_lock)
        return report(root, p)

    cpu_main(monkeypatch, tmp_path, protocol, "report")
    monkeypatch.setattr(gpu, "report", checked_report)
    monkeypatch.setattr(gpu, "admit_node", lambda *args: pytest.fail("report must not admit GPUs"))
    original_sleep = time.sleep
    with peer_locks([test_lock]) as (peer, release):
        def poll(seconds):
            assert not (tmp_path / "report.json").exists()
            polls.append(seconds)
            release.set()
            peer.join(5)
            assert peer.exitcode == 0
            original_sleep(.001)

        monkeypatch.setattr(gpu.time, "sleep", poll)
        gpu.main()

    assert polls and len(calls) == 42
    result = core.read(tmp_path / "report.json")
    assert not result["missing_states"] and not result["missing_development_states"]


def test_fit_barrier_requires_all_development_summaries_before_fitting(tmp_path, fake_study, monkeypatch):
    protocol, calls, _ = fake_study
    monkeypatch.setattr(gpu, "fit", lambda *args: pytest.fail("missing development states reached fit"))
    with pytest.raises(gpu.IncompletePairRun):
        gpu.run_distributed(tmp_path, protocol, None, "fit")
    assert not calls and not (tmp_path / "model.json").exists()


@pytest.mark.parametrize("command", ["run", "develop", "freeze", "test"])
def test_completed_command_skips_gpu_admission_and_preserves_costs(tmp_path, fake_study, monkeypatch, command):
    protocol, calls, _ = fake_study
    complete_study(tmp_path, protocol)
    preserved_paths = [tmp_path / "model.json", tmp_path / "test-decisions.json",
                       *tmp_path.glob("decisions/*/decision.json"), *tmp_path.rglob("cost.jsonl")]
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in preserved_paths}
    cpu_main(monkeypatch, tmp_path, protocol, command)
    monkeypatch.setattr(gpu, "admit_node", lambda *args: pytest.fail("completed rerun charged NCCL admission"))
    monkeypatch.setattr(gpu, "execute", lambda *args: pytest.fail("completed rerun dispatched a branch"))
    assert gpu.admission_required(tmp_path, command) is False
    gpu.main()
    assert len(calls) == 42
    assert before == {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in preserved_paths}


def test_admission_skip_is_not_a_completion_certificate(tmp_path, fake_study, monkeypatch):
    protocol, calls, _ = fake_study
    complete_study(tmp_path, protocol)
    path = tmp_path / "development/s0-t25/result.json"
    core.atomic_json(path, {**core.read(path), "state_id": "tampered"})
    before = path.read_bytes()
    cpu_main(monkeypatch, tmp_path, protocol, "run")
    monkeypatch.setattr(gpu, "admit_node", lambda *args: pytest.fail("saved result verification needs no GPU admission"))
    monkeypatch.setattr(gpu, "execute", lambda *args: pytest.fail("invalid saved summary must not restart training"))
    assert gpu.admission_required(tmp_path, "run") is False
    with pytest.raises((gpu.IncompletePairRun, ValueError)):
        gpu.main()
    assert path.read_bytes() == before and len(calls) == 42


def test_missing_heldout_result_still_requires_admission(tmp_path, fake_study, monkeypatch):
    protocol, _, _ = fake_study
    gpu.develop(tmp_path, protocol, [])
    gpu.fit(tmp_path, protocol)
    gpu.freeze(tmp_path, protocol)
    cpu_main(monkeypatch, tmp_path, protocol, "run")
    admitted = []
    monkeypatch.setattr(gpu, "admit_node", lambda *args: admitted.append(True) or list("0123"))
    monkeypatch.setattr(gpu, "run_distributed", lambda *args: None)
    assert gpu.admission_required(tmp_path, "run") is True
    gpu.main()
    assert admitted == [True]
