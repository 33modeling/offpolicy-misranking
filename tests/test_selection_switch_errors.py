import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest

import selection_gate as core


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("switch_errors", ROOT / "scripts/selection_switch_errors.py")
errors = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(errors)


def failed_prefix(root):
    directory = root / "prefixes/seed-0/segment-25"
    core.atomic_json(directory / "failure.json", {"error": "prefix-train worker failed: [1]"})
    core.atomic_json(directory / "progress.json", {"phase": "prefix-train", "state": "failed"})
    (directory / "prefix-train-0.log").write_text("Traceback (most recent call last):\nRuntimeError: actual worker cause\n")
    return directory


def test_errors_prints_actual_worker_exception_without_mutating_run(tmp_path):
    failed_prefix(tmp_path)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    result = subprocess.run(["bash", "scripts/run_selection_switch.sh", "errors", "--phase", "prefix-train"],
                            cwd=ROOT, env={**os.environ, "SWITCH_ROOT": str(tmp_path), "SWITCH_PYTHON": sys.executable},
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert "prefix-train worker failed" in result.stdout
    assert "RuntimeError: actual worker cause" in result.stdout
    assert "prefixes/seed-0/segment-25/prefix-train-0.log" in result.stdout
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


def test_missing_worker_log_is_not_reported_as_success(tmp_path, capsys):
    directory = failed_prefix(tmp_path)
    (directory / "prefix-train-0.log").unlink()
    assert errors.show_errors(tmp_path) == 1
    assert "[no worker log]" in capsys.readouterr().out


def test_error_log_tail_is_bounded_and_tolerates_non_utf8(tmp_path):
    path = tmp_path / "worker.log"
    path.write_bytes(b"stale line\n" * 100000 + b"\xff\nRuntimeError: current failure\n")
    tail = errors.log_tail(path, 2)
    assert len(tail.splitlines()) == 2
    assert "RuntimeError: current failure" in tail and "stale" not in tail


def test_phase_filter_does_not_show_an_unrelated_worker(tmp_path, capsys):
    failed_prefix(tmp_path)
    assert errors.show_errors(tmp_path, phase="fresh-r-validation") == 0
    assert "actual worker cause" not in capsys.readouterr().out


def test_worker_log_symlink_cannot_escape_run(tmp_path):
    directory = failed_prefix(tmp_path / "run")
    target = tmp_path / "outside.log"
    target.write_text("outside data")
    log = directory / "prefix-train-0.log"
    log.unlink()
    log.symlink_to(target)
    with pytest.raises(ValueError, match="outside the switch root"):
        errors.show_errors(tmp_path / "run")


def test_controller_traceback_is_shown_without_task_failure_record(tmp_path, capsys):
    path = tmp_path / "logs/launcher.node-1.log"
    path.parent.mkdir(parents=True)
    path.write_text("[launcher-start] pid=1\nTraceback (most recent call last):\n"
                    "ValueError: switch protocol or scientific code changed; preserve the frozen run\n"
                    "[launcher-exit] pid=1 rc=1\n")
    before = path.read_bytes()
    errors.show_errors(tmp_path, limit=1)
    output = capsys.readouterr().out
    assert "[launcher-log]" in output and "scientific code changed" in output
    assert "rc=1" in output
    assert path.read_bytes() == before


def test_launcher_tail_drops_previous_invocation_when_start_marker_is_present(tmp_path, capsys):
    path = tmp_path / "logs/launcher.node-1.log"
    path.parent.mkdir(parents=True)
    path.write_text("[launcher-start] pid=1\nold CUDA failure\n[launcher-exit] rc=1\n"
                    "[launcher-start] pid=2\n[launcher-exit] pid=2 rc=0\n")
    errors.show_errors(tmp_path)
    output = capsys.readouterr().out
    assert "old CUDA failure" not in output
    assert "pid=2 rc=0" in output


def test_launcher_symlink_cannot_escape_run(tmp_path):
    root = tmp_path / "run"
    (root / "logs").mkdir(parents=True)
    target = tmp_path / "outside.log"
    target.write_text("outside\n")
    (root / "logs/launcher.node.log").symlink_to(target)
    with pytest.raises(ValueError, match="outside"):
        errors.show_errors(root)


def test_historical_failure_is_identified_separately_from_a_new_owner(tmp_path, capsys):
    directory = failed_prefix(tmp_path)
    core.atomic_json(directory / "failure.json", {"error": "old CUDA 802", "host": "failed-node", "time": 100.})
    core.atomic_json(directory / "progress.json", {"host": "new-node", "pid": 42,
        "phase": "prefix-train", "state": "running", "updated": 200.})
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    errors.show_errors(tmp_path)
    output = capsys.readouterr().out
    assert "[recorded failure] host=failed-node utc=1970-01-01T00:01:40+00:00" in output
    assert "this display does not establish a new failure" in output
    assert "[last progress] host=new-node pid=42 state=running" in output
    assert "old CUDA 802" in output
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


def test_node_checks_show_only_the_latest_record_per_host_and_never_claim_training_success(tmp_path, capsys):
    for name, host, state, revision, when in [("old", "node-a", "failed", "old-code", 100),
            ("new", "node-a", "passed", "new-code", 200), ("other", "node-b", "failed", "other-code", 150)]:
        path = tmp_path / "node-preflight" / name / "admission.json"
        core.atomic_json(path, {"host": host, "state": state, "runtime_commit": revision})
        os.utime(path, (when, when))
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    errors.show_errors(tmp_path)
    output = capsys.readouterr().out
    assert "host=node-a state=passed runtime=new-code" in output
    assert "host=node-b state=failed runtime=other-code" in output
    assert "runtime=old-code" not in output
    assert "probe outcome only, not current training status" in output
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("value", [None, "unknown", float("nan"), float("inf"), True, 1e100])
def test_invalid_recorded_timestamp_is_not_made_up(value):
    assert errors.utc_time(value) == "unknown"


def test_admission_symlink_cannot_escape_run(tmp_path):
    root = tmp_path / "run"
    path = root / "node-preflight/node-a/admission.json"
    path.parent.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    core.atomic_json(outside, {"host": "outside", "state": "passed"})
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="outside"):
        errors.show_errors(root)
