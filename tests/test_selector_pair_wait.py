"""Bound all lock-only waiting without stealing locks or restarting GPU work."""

import contextlib
import fcntl
import os
import subprocess
import sys

import pytest

import selection_gate as core
import selection_gate_gpu as base
import selector_pair as pair
import selector_pair_gpu as gpu
from test_selector_pair_gpu import fake_study


class Clock:
    def __init__(self):
        self.elapsed = 0.
        self.sleeps = []
        self.after_sleep = lambda: None

    def monotonic(self):
        return self.elapsed

    def time(self):
        return 1900000000. + self.elapsed

    def sleep(self, seconds):
        assert seconds > 0, "waiting must not become a busy-spin"
        self.sleeps.append(seconds)
        assert len(self.sleeps) < 100, "lock waiting did not terminate"
        self.elapsed += seconds
        self.after_sleep()


@pytest.fixture
def clock(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(gpu, "time", clock)
    return clock


@contextlib.contextmanager
def held(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield handle


def unchanged(root):
    return {path: (path.read_bytes(), path.stat().st_mtime_ns)
            for path in root.rglob("*") if path.is_file()}


def metadata(root, clock, *, state="s0-t25", status="running", age=0):
    path = root / "branches/on_policy/states" / state / "points/view-25/selection_reduced/curve/step-50/progress.json"
    core.atomic_json(path, {"state": status, "host": "actual-peer-node", "phase": "curve",
                            "updated": clock.time() - age, "pid": 1234})
    return path


def prepare_other_states(root, fake_study, excluded=((0, 25),)):
    p, _, states = fake_study
    for seed in pair.DEV_SEEDS:
        for step in pair.STEPS:
            if (seed, step) in excluded:
                continue
            _, entries = states(root, seed, step)
            for name in pair.SELECTORS:
                gpu.execute(entries[name], "selection_reduced", [])
            base.bind(root / "development" / f"s{seed}-t{step}" / "result.json",
                      gpu.development_row(root, p, seed, step))
    return p


@pytest.mark.parametrize("shared,name", [(True, ".pair.lock"), (False, ".pair-runtime.lock"),
                                         (False, ".pair-barrier.lock")])
def test_live_root_or_barrier_lock_has_a_finite_total_wait(tmp_path, clock, shared, name):
    path = tmp_path / name
    path.write_text("preserve the actual lock file")
    with held(path):
        before, inode = unchanged(tmp_path), path.stat().st_ino
        with pytest.raises(gpu.PairWaitTimeout):
            with gpu.queue_lease(path, shared=shared, wait_seconds=15., max_wait_seconds=180.):
                pytest.fail("an exclusively held lock was bypassed")
        assert 180 <= clock.elapsed <= 195
        assert path.stat().st_ino == inode and unchanged(tmp_path) == before
        with path.open("a+") as probe:
            with pytest.raises(BlockingIOError):
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_root_lease_enters_when_owner_releases_before_deadline(tmp_path, clock):
    path = tmp_path / ".pair.lock"
    with held(path) as owner:
        before, inode = unchanged(tmp_path), path.stat().st_ino
        clock.after_sleep = lambda: fcntl.flock(owner, fcntl.LOCK_UN) if clock.elapsed >= 30 else None
        with gpu.queue_lease(path, shared=True, wait_seconds=15., max_wait_seconds=180.):
            assert clock.elapsed == 30
            assert unchanged(tmp_path) == before and path.stat().st_ino == inode


def test_root_total_deadline_is_not_refreshed_by_other_worker_logs(tmp_path, clock):
    path = tmp_path / ".pair.lock"
    with held(path):
        clock.after_sleep = lambda: metadata(tmp_path, clock)
        with pytest.raises(gpu.PairWaitTimeout):
            with gpu.queue_lease(path, shared=True, wait_seconds=30., max_wait_seconds=180.):
                pytest.fail("progress metadata cannot override a held root lock")
        assert 180 <= clock.elapsed <= 210


def test_diagnostic_failure_cannot_hide_terminal_timeout(tmp_path, clock, monkeypatch):
    monkeypatch.setattr(gpu, "_wait_diagnostics", lambda path: (_ for _ in ()).throw(OSError("storage unavailable")))
    with held(tmp_path / ".pair.lock"):
        with pytest.raises(gpu.PairWaitTimeout):
            with gpu.queue_lease(tmp_path / ".pair.lock", shared=True, max_wait_seconds=30.):
                pytest.fail("held lock bypassed")
    assert clock.elapsed == 30


@pytest.mark.parametrize("evidence", ["none", "stale", "unrelated", "finished", "queue-wait", "future"])
def test_idle_peer_wait_cannot_be_kept_alive_by_irrelevant_or_old_metadata(tmp_path, fake_study, monkeypatch, clock, evidence):
    p = prepare_other_states(tmp_path, fake_study)
    progress = None
    if evidence == "stale":
        progress = metadata(tmp_path, clock, age=1000)
    elif evidence == "unrelated":
        progress = metadata(tmp_path, clock, state="s1-t25")
        clock.after_sleep = lambda: metadata(tmp_path, clock, state="s1-t25")
    elif evidence == "finished":
        progress = metadata(tmp_path, clock, status="finished")
        clock.after_sleep = lambda: metadata(tmp_path, clock, status="finished")
    elif evidence == "future":
        progress = metadata(tmp_path, clock, age=-1000)
    elif evidence == "queue-wait":
        progress = tmp_path / "queue-workers/peer.json"
        def heartbeat():
            core.atomic_json(progress, {"state": "WAIT", "host": "idle-peer-node", "stage": "development",
                                        "task": "development/s0-t25", "updated": clock.time()})
        heartbeat()
        clock.after_sleep = heartbeat
    monkeypatch.setattr(gpu, "execute", lambda *a: pytest.fail("waiting node started GPU work"))
    lock = tmp_path / "development/s0-t25/.state.lock"
    with held(lock):
        inode = lock.stat().st_ino
        saved = {path: path.read_bytes() for path in tmp_path.glob("branches/*/states/*/point/*/result.json")}
        with pytest.raises(gpu.PairWaitTimeout):
            gpu.distributed_stage(tmp_path, p, [], "development", wait_seconds=30., idle_timeout=180.)
        assert 180 <= clock.elapsed <= 210
        assert lock.stat().st_ino == inode and not (lock.parent / "result.json").exists()
        assert all(path.read_bytes() == value for path, value in saved.items())
        if progress is not None:
            assert progress.exists()


@pytest.mark.parametrize('age', [0, -3600, 3600])
def test_fresh_relevant_nested_progress_allows_long_training_then_lock_release(tmp_path, fake_study, clock, age):
    p = prepare_other_states(tmp_path, fake_study)
    _, calls, _ = fake_study
    metadata(tmp_path, clock, age=age)
    lock = tmp_path / "development/s0-t25/.state.lock"
    with held(lock) as owner:
        def heartbeat():
            metadata(tmp_path, clock, age=age)
            if clock.elapsed >= 240:
                fcntl.flock(owner, fcntl.LOCK_UN)
        clock.after_sleep = heartbeat
        gpu.distributed_stage(tmp_path, p, [], "development", wait_seconds=30., idle_timeout=180.)
    assert clock.elapsed == 240 and len(calls) == 18
    assert len(list((tmp_path / "development").glob("*/result.json"))) == 9


def test_repeated_reads_of_the_same_heartbeat_do_not_extend_idle_deadline(tmp_path, fake_study, monkeypatch, clock):
    p = prepare_other_states(tmp_path, fake_study)
    progress = metadata(tmp_path, clock)
    before = progress.read_bytes()
    monkeypatch.setattr(gpu, "execute", lambda *a: pytest.fail("waiting node started GPU work"))
    with held(tmp_path / "development/s0-t25/.state.lock"):
        with pytest.raises(gpu.PairWaitTimeout):
            gpu.distributed_stage(tmp_path, p, [], "development", wait_seconds=15., idle_timeout=180.)
    assert clock.elapsed == 180
    assert progress.read_bytes() == before


def test_partial_barrier_state_leases_release_on_later_lock_timeout(tmp_path, fake_study, clock):
    prepare_other_states(tmp_path, fake_study, excluded=())
    first = tmp_path / "development/s0-t25/.state.lock"
    second = tmp_path / "development/s0-t50/.state.lock"
    saved = {path: (path.read_bytes(), path.stat().st_mtime_ns)
             for path in tmp_path.glob("development/*/result.json")}
    with held(second):
        with pytest.raises(gpu.PairWaitTimeout):
            with gpu.completed_state_leases(tmp_path):
                pytest.fail("barrier entered despite peer-held completed state")
        assert clock.elapsed == 180
        with first.open("a+") as probe:
            # A timed-out multi-lock acquisition must not keep earlier locks.
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with second.open("a+") as probe:
            with pytest.raises(BlockingIOError):
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert saved == {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in saved}


def test_newly_validated_state_resets_the_idle_wait_deadline(tmp_path, fake_study, clock):
    p = prepare_other_states(tmp_path, fake_study, excluded=((0, 25), (0, 50)))
    _, calls, _ = fake_study
    with held(tmp_path / "development/s0-t25/.state.lock") as first:
        with held(tmp_path / "development/s0-t50/.state.lock") as second:
            def release():
                if clock.elapsed >= 120:
                    fcntl.flock(first, fcntl.LOCK_UN)
                if clock.elapsed >= 240:
                    fcntl.flock(second, fcntl.LOCK_UN)
            clock.after_sleep = release
            gpu.distributed_stage(tmp_path, p, [], "development", wait_seconds=30., idle_timeout=180.)
    assert clock.elapsed == 240 and len(calls) == 18
    assert len(list((tmp_path / "development").glob("*/result.json"))) == 9


def test_cli_reports_wait_timeout_as_exit_76_without_touching_saved_files(tmp_path):
    core.atomic_json(tmp_path / "pair.json", {"schema": pair.SCHEMA})
    script = str(base.ROOT / "src/selector_pair_gpu.py")
    code = (
        "import runpy, sys, time\n"
        "clock = [0.]\n"
        "time.monotonic = lambda: clock[0]\n"
        "time.sleep = lambda seconds: clock.__setitem__(0, clock[0] + seconds)\n"
        f"sys.argv = [{script!r}, 'ensure-prepared', '--root', {str(tmp_path)!r}]\n"
        f"runpy.run_path({script!r}, run_name='__main__')\n"
    )
    with held(tmp_path / ".pair.lock"):
        before = unchanged(tmp_path)
        process = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                                 cwd=base.ROOT, timeout=15,
                                 env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        assert process.returncode == 76, process.stdout + process.stderr
        assert "Traceback" not in process.stderr
        assert unchanged(tmp_path) == before


@pytest.mark.parametrize("mode", ["run", "develop", "freeze", "test"])
def test_shell_caps_legacy_busy_retries_and_keeps_timeout_nonzero(tmp_path, mode):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "python-calls.txt"
    python = bin_dir / "fake-python"
    python.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$PAIR_TEST_CALLS"\nexit 75\n')
    sleep = bin_dir / "sleep"
    sleep.write_text("#!/bin/sh\nexit 0\n")
    forbidden = bin_dir / "nvidia-smi"
    forbidden.write_text('#!/bin/sh\nprintf gpu > "$PAIR_TEST_GPU"\nexit 99\n')
    for path in (python, sleep, forbidden):
        path.chmod(0o755)
    root, work = tmp_path / "missing-pair", tmp_path / "missing-work"
    process = subprocess.run(["bash", "scripts/run_selector_pair.sh", mode], cwd=base.ROOT,
        env={**os.environ, "PAIR_PYTHON": str(python), "PAIR_ROOT": str(root), "OM_WORK": str(work),
             "PAIR_TEST_CALLS": str(calls), "PAIR_TEST_GPU": str(tmp_path / "gpu-queried"),
             "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"], "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True, text=True, timeout=10)
    assert process.returncode == 76, process.stdout + process.stderr
    assert len(calls.read_text().splitlines()) == 13
    assert not root.exists() and not work.exists() and not (tmp_path / "gpu-queried").exists()
