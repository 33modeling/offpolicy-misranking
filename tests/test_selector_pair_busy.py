"""Duplicate pair launches observe their owner without touching GPU work."""

import errno
import fcntl
import os
import subprocess
import sys
import time

import pytest

import selection_gate as core
import selection_gate_gpu as base
import selector_pair_gpu as gpu


def contents(root):
    return {path.relative_to(root): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in root.rglob("*") if path.is_file()}


def progress(root, *, age=1, suffix="selection_reduced", host="run284000-wts-10-g1234", phase="evaluate"):
    path = root / "branches/on_policy/states/s0-t25/points/view-25" / suffix / "progress.json"
    core.atomic_json(path, {"host": host, "phase": phase, "state": "running", "pid": 123,
                            "updated": time.time() - age, "seconds": 12, "timeout": 3600})
    return path


@pytest.fixture(autouse=True)
def no_gpu_or_initialization(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("check-running must not initialize or admit GPU work")

    for name in ("initialize", "ensure_prepared", "manifest", "admit_node", "admission_probe"):
        monkeypatch.setattr(gpu, name, forbidden)


@pytest.mark.parametrize("kind", ["missing-root", "missing-lock", "free-lock"])
def test_free_or_missing_lock_check_is_readonly_and_does_not_initialize(tmp_path, kind, capsys):
    root = tmp_path / "pair"
    if kind != "missing-root":
        root.mkdir()
        (root / "preserved.txt").write_text("existing saved work")
    if kind == "free-lock":
        (root / ".pair.lock").write_text("preserve lock inode contents")
    before = contents(tmp_path)
    assert gpu.check_running(root) is False
    assert contents(tmp_path) == before
    assert "[already running]" not in capsys.readouterr().out
    if kind == "missing-root":
        assert not root.exists()


@pytest.mark.parametrize("bootstrap", [False, True])
@pytest.mark.parametrize("suffix", ["selection_reduced", "selection_reduced/curve/step-50"])
def test_held_lock_reports_observed_work_without_prepared_pair(tmp_path, bootstrap, suffix, capsys):
    root = tmp_path / "pair"
    if bootstrap:
        core.atomic_json(root / "pair.json", {"schema": gpu.BOOTSTRAP_SCHEMA,
                                              "status": "preparation_incomplete"})
    progress(root, suffix=suffix)
    with base.lease(root / ".pair.lock"):
        before = contents(root)
        assert gpu.check_running(root) is True
        output = capsys.readouterr().out
        assert "[already running]" in output
        assert "run284000-wts-10-g1234" in output and "evaluate" in output
        assert contents(root) == before
        assert (root / "pair.json").exists() is bootstrap


def test_held_lock_without_progress_still_reports_existing_controller(tmp_path, capsys):
    with base.lease(tmp_path / ".pair.lock"):
        before = contents(tmp_path)
        assert gpu.check_running(tmp_path) is True
        assert "[already running]" in capsys.readouterr().out
        assert contents(tmp_path) == before


def test_free_probe_uses_shared_lock_on_readonly_descriptor(tmp_path, monkeypatch):
    (tmp_path / ".pair.lock").touch()
    original = fcntl.flock
    operations = []

    def nfs_flock(handle, operation):
        access = fcntl.fcntl(handle, fcntl.F_GETFL) & os.O_ACCMODE
        assert access == os.O_RDONLY
        if operation & fcntl.LOCK_EX:
            raise OSError(errno.EBADF, "NFS exclusive locks require write access")
        operations.append(operation)
        return original(handle, operation)

    monkeypatch.setattr(fcntl, "flock", nfs_flock)
    before = contents(tmp_path)
    assert gpu.check_running(tmp_path) is False
    assert operations == [fcntl.LOCK_SH | fcntl.LOCK_NB, fcntl.LOCK_UN]
    assert contents(tmp_path) == before


def test_old_progress_is_not_advertised_as_current_work(tmp_path, capsys):
    progress(tmp_path, age=900, host="old-owner-node", phase="old-evaluate")
    with base.lease(tmp_path / ".pair.lock"):
        before = contents(tmp_path)
        assert gpu.check_running(tmp_path) is True
        output = capsys.readouterr().out
        assert "[already running]" in output
        assert "old-owner-node" not in output or any(
            word in output.lower() for word in ("stale", "historical", "last observed", "오래된", "이전 기록"))
        assert contents(tmp_path) == before


@pytest.mark.parametrize("held", [False, True])
def test_check_running_cli_exit_code_and_files_are_observation_only(tmp_path, held):
    root = tmp_path / "pair"
    progress(root)
    command = [sys.executable, str(base.ROOT / "src/selector_pair_gpu.py"), "check-running", "--root", str(root)]
    with base.lease(root / ".pair.lock"):
        if not held:
            # Use a separate untouched root to exercise a missing lock through
            # the CLI while keeping the held-lock test setup identical.
            root = tmp_path / "missing"
            command[-1] = str(root)
        before = contents(tmp_path)
        process = subprocess.run(command, capture_output=True, text=True, timeout=15,
                                 env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        assert process.returncode == (75 if held else 0), process.stderr
        assert ("[already running]" in process.stdout) is held
        assert "Traceback" not in process.stderr
        assert contents(tmp_path) == before
        if not held:
            assert not root.exists()


@pytest.mark.parametrize("mode", ["run", "develop", "freeze", "test"])
@pytest.mark.parametrize("bootstrap", [False, True])
def test_shell_duplicate_launch_succeeds_before_source_checks_or_gpu_admission(tmp_path, mode, bootstrap):
    root = tmp_path / "pair"
    if bootstrap:
        core.atomic_json(root / "pair.json", {"schema": gpu.BOOTSTRAP_SCHEMA,
                                              "status": "preparation_incomplete"})
    progress(root)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gpu_marker = tmp_path / "gpu-queried"
    trap = bin_dir / "nvidia-smi"
    trap.write_text('#!/bin/sh\nprintf forbidden > "$PAIR_GPU_TEST_MARKER"\nexit 99\n')
    trap.chmod(0o755)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PAIR_ROOT": str(root),
           "OM_WORK": str(tmp_path / "missing-storage"), "PAIR_PYTHON": sys.executable,
           "SWITCH_PREFIX_SOURCE": str(tmp_path / "missing-source"),
           "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
           "PAIR_GPU_TEST_MARKER": str(gpu_marker)}
    with base.lease(root / ".pair.lock"):
        before = contents(tmp_path)
        process = subprocess.run(["bash", "scripts/run_selector_pair.sh", mode], cwd=base.ROOT,
                                 env=env, capture_output=True, text=True, timeout=20)
        assert process.returncode == 0, process.stdout + process.stderr
        assert "[already running]" in process.stdout
        assert "run284000-wts-10-g1234" in process.stdout and "evaluate" in process.stdout
        assert "[node]" not in process.stdout and "certified prefix source is missing" not in process.stderr
        assert "pair lock busy" not in process.stderr and "Traceback" not in process.stderr
        assert not gpu_marker.exists() and not (tmp_path / "missing-storage").exists()
        assert contents(tmp_path) == before
        assert (root / "pair.json").exists() is bootstrap


def test_busy_probe_keeps_lease_body_eagain_distinct_from_lock_contention(tmp_path):
    failure = BlockingIOError(errno.EAGAIN, "Resource temporarily unavailable")
    with pytest.raises(BlockingIOError) as exc:
        with gpu.pair_lease(tmp_path / ".pair.lock"):
            raise failure
    assert exc.value is failure
    with gpu.pair_lease(tmp_path / ".pair.lock"):
        pass


@pytest.mark.parametrize("command", ["ensure-prepared", "run", "develop"])
def test_lock_acquired_after_precheck_is_reported_without_traceback(tmp_path, command):
    # The real lease still protects a race after the shell's optimistic probe.
    progress(tmp_path)
    with base.lease(tmp_path / ".pair.lock"):
        before = contents(tmp_path)
        process = subprocess.run([sys.executable, str(base.ROOT / "src/selector_pair_gpu.py"),
                                  command, "--root", str(tmp_path)],
                                 env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                                 capture_output=True, text=True, timeout=15)
        assert process.returncode == 75
        assert "[already running]" in process.stdout and "run284000-wts-10-g1234" in process.stdout
        assert "pair lock busy" not in process.stderr and "Traceback" not in process.stderr
        assert contents(tmp_path) == before
