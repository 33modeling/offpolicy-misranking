import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def exporter(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("mbpp_paper_results")


def prepared(root):
    root.mkdir()
    (root / "switch.json").write_text("{}")
    return root


def export_data(path):
    return json.loads(path.read_text().split("DATA_JSON\n", 1)[1])


def test_multiple_suites_export_one_txt_with_duplicate_roots_deduplicated(exporter, tmp_path, monkeypatch):
    first = prepared(tmp_path / "quality")
    second = prepared(tmp_path / "difficulty")
    output = tmp_path / "mbpp-results.txt"
    calls = []

    def report(root):
        calls.append(root)
        return f"saved measurements {root.name}"

    monkeypatch.setattr(exporter.switch_results, "report", report)
    monkeypatch.setattr(sys, "argv", ["mbpp_paper_results.py", "--root", str(first),
                                    "--root", str(second), "--root", str(first), "--out", str(output)])
    exporter.main()
    assert calls == [first, second]
    assert list(tmp_path.glob("*.txt")) == [output]
    assert not list(tmp_path.glob("*.tmp.*"))
    assert "saved measurements quality" in output.read_text()
    assert "saved measurements difficulty" in output.read_text()
    assert [suite["status"] for suite in export_data(output)["suites"]] == ["exported", "exported"]
    assert output.stat().st_size < 1_900_000


def test_missing_suite_is_labeled_without_hiding_saved_results(exporter, tmp_path, monkeypatch):
    valid = prepared(tmp_path / "quality")
    missing = tmp_path / "missing"
    output = tmp_path / "mbpp-results.txt"
    calls = []

    def report(root):
        calls.append(root)
        return "partial saved measurements"

    monkeypatch.setattr(exporter.switch_results, "report", report)
    monkeypatch.setattr(sys, "argv", ["mbpp_paper_results.py", "--root", str(missing),
                                    "--root", str(valid), "--out", str(output)])
    exporter.main()
    assert calls == [valid]
    assert not missing.exists()
    assert "partial saved measurements" in output.read_text()
    assert [suite["status"] for suite in export_data(output)["suites"]] == ["unprepared", "exported"]


def test_invalid_suite_still_writes_valid_results_and_error(exporter, tmp_path, monkeypatch):
    invalid = prepared(tmp_path / "invalid")
    valid = prepared(tmp_path / "quality")
    output = tmp_path / "mbpp-results.txt"

    def report(root):
        if root == invalid:
            raise ValueError("invalid saved result")
        return "valid measurements retained"

    monkeypatch.setattr(exporter.switch_results, "report", report)
    monkeypatch.setattr(sys, "argv", ["mbpp_paper_results.py", "--root", str(invalid),
                                    "--root", str(valid), "--out", str(output)])
    with pytest.raises(SystemExit) as error:
        exporter.main()
    assert error.value.code == 1
    assert "valid measurements retained" in output.read_text()
    suites = export_data(output)["suites"]
    assert suites[0]["status"] == "error"
    assert suites[0]["error"] == "invalid saved result"
    assert suites[1]["status"] == "exported"
    assert list(tmp_path.glob("*.txt")) == [output]


def test_mbpp_wrapper_includes_all_existing_suites_in_one_txt(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    work = tmp_path / "work"
    runs = work / "runs"
    runs.mkdir(parents=True)
    names = ["selection-switch-mbpp-quality-v1", "selection-switch-mbpp-v1",
             "selection-switch-mbpp-difficulty-v1", "selection-switch-mbpp-long-v1"]
    for name in names:
        prepared(runs / name)
    output = tmp_path / "mbpp-results.txt"
    env = dict(os.environ, OM_WORK=str(work), SWITCH_PYTHON=sys.executable,
               PYTHONDONTWRITEBYTECODE="1")
    for variable in ("SWITCH_MBPP_ROOT", "SWITCH_MBPP_QUALITY_ROOT",
                     "SWITCH_MBPP_DIFFICULTY_ROOT", "SWITCH_MBPP_LONG_ROOT"):
        env.pop(variable, None)
    result = subprocess.run(["bash", "scripts/run_paper_results.sh", "results", "mbpp",
                             "--out", str(output)], cwd=repository, env=env,
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    suites = export_data(output)["suites"]
    assert [Path(suite["root"]).name for suite in suites] == names
    assert all(suite["status"] == "exported" for suite in suites)
    assert list(tmp_path.rglob("*.txt")) == [output]
