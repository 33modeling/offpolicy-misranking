"""Orphan helper recovery must not mistake a live compiler pool for stale work."""

import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

import cleanup_run_processes as cleanup

ROOT = Path(__file__).resolve().parents[1]
COMPILER = ("/venv/bin/python", "/venv/site-packages/torch/_inductor/compile_worker/__main__.py",
            "--pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler")


def process(pid, ppid, argv, lock="/test/primary.lock", env=None):
    return cleanup.Process(pid, ppid, " ".join(argv), env or {}, frozenset([lock]), argv=tuple(argv))


@pytest.mark.parametrize("argv", [COMPILER, ("/bin/sleep", "15"), ("/bin/tee", "session.log")])
def test_orphans_require_the_exact_lock_not_just_a_scope_marker(monkeypatch, argv):
    rows = {101: process(101, 1, argv), 102: process(102, 1, argv, "/other/primary.lock", {"OUT_ROOT": "/our-run"})}
    monkeypatch.setattr(cleanup, "_snapshot", lambda: rows)
    targets = cleanup.matching_processes("/our-run", open_files=("/test/primary.lock",), orphan_lock_helpers_only=True)
    assert set(targets) == {101}


@pytest.mark.parametrize("argv", [COMPILER, ("/bin/sleep", "15")])
def test_live_parent_or_unreadable_ancestry_is_not_authorized(monkeypatch, argv):
    rows = {101: process(101, 200, argv), 102: process(102, 99999, argv),
            200: process(200, 1, ("python", "src/train_policy_grpo.py"))}
    monkeypatch.setattr(cleanup, "_snapshot", lambda: rows)
    assert not cleanup.matching_processes("/none", open_files=("/test/primary.lock",), orphan_lock_helpers_only=True)


def test_live_declared_torch_parent_is_preserved_even_after_reparenting(monkeypatch):
    rows = {101: process(101, 1, (*COMPILER, f"--parent={os.getpid()}"))}
    monkeypatch.setattr(cleanup, "_snapshot", lambda: rows)
    assert not cleanup.matching_processes("/none", open_files=("/test/primary.lock",), orphan_lock_helpers_only=True)


def test_init_itself_is_never_a_helper_cleanup_target(monkeypatch):
    rows = {1: process(1, 0, ("python", "src/train_policy_grpo.py"))}
    monkeypatch.setattr(cleanup, "_snapshot", lambda: rows)
    assert not cleanup.matching_processes("/none", open_files=("/test/primary.lock",), orphan_lock_helpers_only=True)


def test_inaccessible_declared_training_parent_is_not_treated_as_dead(monkeypatch):
    rows = {101: process(101, 1, (*COMPILER, "--parent=99999"))}
    monkeypatch.setattr(cleanup, "_snapshot", lambda: rows)
    def denied(pid, sig):
        raise PermissionError("parent belongs to another user")
    monkeypatch.setattr(cleanup.os, "kill", denied)
    assert not cleanup.matching_processes("/none", open_files=("/test/primary.lock",), orphan_lock_helpers_only=True)


def test_orphan_pool_descendants_and_sleep_are_grouped_under_actual_holder(monkeypatch):
    rows = {100: process(100, 1, COMPILER)}
    rows.update({pid: process(pid, 100, COMPILER) for pid in range(101, 201)})
    monkeypatch.setattr(cleanup, "_snapshot", lambda: rows)
    assert len(cleanup.matching_processes("/none", open_files=("/test/primary.lock",), orphan_lock_helpers_only=True)) == 101
    lines = cleanup.describe_lock_owners(("/test/primary.lock",))
    assert len(lines) == 2
    assert "101 processes" in lines[0] and "1 owner groups" in lines[0]
    assert "pid=100 ppid=1 openers=101 orphan-helper" in lines[1]


def test_diagnostics_are_bounded_for_many_orphans(monkeypatch):
    rows = {pid: process(pid, 1, COMPILER) for pid in range(100, 200)}
    monkeypatch.setattr(cleanup, "_snapshot", lambda: rows)
    lines = cleanup.describe_lock_owners(("/test/primary.lock",))
    assert len(lines) == 10
    assert "92 more owner groups" in lines[-1]


@pytest.mark.parametrize("helper", ["sleep", "compiler"])
def test_e5_default_reclaims_a_real_orphan_without_force(tmp_path, helper):
    # The short-lived parent leaves only an unlabelled helper holding the lock.
    script = tmp_path / "torch/_inductor/compile_worker/__main__.py"
    script.parent.mkdir(parents=True)
    script.write_text("import time\ntime.sleep(120)\n")
    env = {key: value for key, value in os.environ.items()
           if key not in {"OUT_ROOT", "RUN_BASE", "REGIME_ROOT", "E5_FORCE"}}
    env.update(OM_LOCAL_LOCK_DIR=str(tmp_path), PY=sys.executable, HELPER=helper, HELPER_SCRIPT=str(script))
    parent = subprocess.Popen(["bash", "-c", '''
exec 8>"$OM_LOCAL_LOCK_DIR/primary.lock"
flock 8
if [ "$HELPER" = compiler ]; then
  "$PY" "$HELPER_SCRIPT" --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --parent="$$" &
else
  /bin/sleep 120 &
fi
echo $! > "$OM_LOCAL_LOCK_DIR/helper.pid"
'''], env=env, start_new_session=True)
    child_pid = None
    try:
        assert parent.wait(timeout=5) == 0
        child_pid = int((tmp_path / "helper.pid").read_text())
        deadline = time.monotonic() + 5
        while True:
            child = cleanup._read_process(child_pid)
            if child and child.ppid == 1 and child.argv[0] in {sys.executable, "/bin/sleep"}:
                break
            assert time.monotonic() < deadline, child
            time.sleep(0.02)
        # Old E5 root-only cleanup cannot discover this orphan.
        assert child_pid not in cleanup.matching_processes(str(tmp_path / "runs/e5-reduced"))
        result = subprocess.run(["bash", "-c", "source scripts/_e5_node.sh; e5_acquire_node"],
                                cwd=ROOT, env=env, capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, result.stdout + result.stderr
        assert f"pid={child_pid}" in result.stdout
        assert "node ownership acquired" in result.stdout
        assert cleanup._read_process(child_pid) is None
    finally:
        try:
            os.killpg(parent.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        parent.wait(timeout=5)
