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
