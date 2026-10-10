"""Read-only total completion, live leases and actual measured tensor audits."""

import json
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from srgc_rebuttal.runtime import atomic_json, lease
from srgc_research.dispatch import information_status as show
from srgc_research.tests import test_information as fixtures
from srgc_research.tests import test_information_wrapper as wrappers

backend, measured, wrapper = fixtures.backend, fixtures.measured, wrappers.wrapper


def test_empty_status_reports_all_ten_without_creating_files(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("OM_WORK", str(tmp_path))
    before = list(tmp_path.rglob("*"))
    assert show.main(["all", "status"]) == 0
    text = capsys.readouterr().out
    assert "0/10 complete" in text and "INCOMPLETE" in text
    assert "math.seed-7.t0" in text and "mbpp.seed-9.t0" in text
    assert list(tmp_path.rglob("*")) == before


@pytest.mark.parametrize("lock", ("queue", ".dispatch.lock", ".execution.lock"))
def test_existing_owner_is_reported_without_creating_another_lock(tmp_path, lock):
    output = tmp_path / "math/seed-5/t0"
    receipt = tmp_path / ".queue/math.seed-5.t0.json"
    path = receipt.with_suffix(".lock") if lock == "queue" else output / lock
    with lease(path):
        assert show.inspect(tmp_path, "math", 5)["state"] == "running"
        before = sorted(tmp_path.rglob("*"))
        show.inspect(tmp_path, "math", 5)
        assert sorted(tmp_path.rglob("*")) == before


def test_stale_claim_is_resume_and_failures_keep_actual_error(tmp_path):
    receipt = tmp_path / ".queue/math.seed-5.t0.json"
    row = {"task": "math.seed-5.t0", "status": "running", "host": "node-1"}
    atomic_json(receipt, row)
    assert show.inspect(tmp_path, "math", 5)["state"] == "resume"
    atomic_json(receipt, {**row, "status": "failed", "error": "FloatingPointError: actual cause"})
    result = show.inspect(tmp_path, "math", 5)
    assert result["state"] == "failed" and result["error"] == "FloatingPointError: actual cause"
    assert result["host"] == "node-1"


def test_completed_receipt_alone_is_never_counted_as_complete(tmp_path):
    atomic_json(tmp_path / ".queue/math.seed-5.t0.json", {"task": "math.seed-5.t0", "status": "complete"})
    assert show.inspect(tmp_path, "math", 5)["state"] == "invalid"


def test_real_measurement_is_audited_and_tensor_corruption_is_rejected(tmp_path, measured):
    folder = tmp_path / "math/seed-5/t0"
    shutil.copytree(measured, folder)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    row = show.inspect(tmp_path, "math", 5)
    assert row["state"] == "complete" and row["phases"] == len(show.PHASES)
    assert before == {path: path.read_bytes() for path in before}
    with (folder / "sr.pt").open("ab") as handle:
        handle.write(b"damaged tensor")
    assert show.inspect(tmp_path, "math", 5)["state"] == "invalid"


def test_wrong_seed_or_stage_cannot_be_counted_as_a_different_measurement(tmp_path, measured):
    folder = tmp_path / "math/seed-7/t0"
    shutil.copytree(measured, folder)
    assert show.inspect(tmp_path, "math", 7)["state"] == "invalid"


def test_aggregate_completion_requires_every_requested_endpoint(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("OM_WORK", str(tmp_path))
    with patch.object(show, "inspect", side_effect=lambda root, name, seed: {
        "task": f"{name}.seed-{seed}.t0", "state": "complete", "phases": len(show.PHASES), "host": "-", "error": None}):
        assert show.main(["all", "status", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["complete"] and result["counts"] == {"complete": 10}


def test_public_shell_status_needs_no_folder_or_gpu_python(wrapper):
    run, work, _ = wrapper
    result, calls = run("all", "status", overrides={"PAIR_PYTHON": "/missing", "SWITCH_PYTHON": "/missing"})
    assert result.returncode == 0, result.stderr
    assert "0/10 complete" in result.stdout and calls == []
    assert not (work / "selection-information").exists()


def test_public_shell_status_json_and_explicit_output_remain_supported(wrapper):
    run, work, _ = wrapper
    result, calls = run("mbpp", "status", "--json")
    assert result.returncode == 0 and calls == []
    data = json.loads(result.stdout)
    assert len(data["rows"]) == 5 and not data["complete"]
    result, calls = run("all", "status", "--output", str(work / "single"))
    assert result.returncode == 0 and calls[-1]["args"][1] == "srgc_research.information_cli"


def test_status_subprocess_does_not_import_torch(tmp_path):
    root = Path(__file__).resolve().parents[2]
    code = """import sys
sys.modules['torch'] = None
from srgc_research.dispatch.information_status import main
raise SystemExit(main(['all', 'status']))
"""
    import os
    import sys
    result = subprocess.run([sys.executable, "-c", code], cwd=root,
        env={**os.environ, "OM_WORK": str(tmp_path)}, capture_output=True, text=True, timeout=15, check=False)
    assert result.returncode == 0, result.stderr
    assert "0/10 complete" in result.stdout
