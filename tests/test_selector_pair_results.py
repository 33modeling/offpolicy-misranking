"""Pair paper exports regenerate validated reports before packaging partial data."""

import json
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import selector_pair_results as results


@pytest.mark.parametrize("missing,development,complete", [
    (["s4-t25"], [], False),
    ([], ["s1-t25"], False),
    ([], [], True),
])
def test_exports_current_partial_report_and_curves(tmp_path, monkeypatch, missing, development, complete):
    root = tmp_path / "run"
    root.mkdir()
    target = tmp_path / "selector-pair-results.txt"
    report = {"missing_states": missing, "missing_development_states": development,
              "rows": [{"state": "s3-t25", "score": 0.5}]}
    curves = "state,step,score\ns3-t25,10,0.5\n"
    calls = []

    def regenerate(command, **kwargs):
        calls.append((command, kwargs))
        (root / "report.json").write_text(json.dumps(report))
        (root / "curves.csv").write_text(curves)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(results.subprocess, "run", regenerate)
    monkeypatch.setattr(sys, "argv", ["selector_pair_results", "--root", str(root), "--out", str(target)])
    results.main()

    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command == ["bash", "scripts/run_selector_pair.sh", "report"]
    assert kwargs["env"]["PAIR_ROOT"] == str(root.resolve())
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
    content = target.read_text()
    assert curves in content
    data = json.loads(content.split("DATA_JSON\n", 1)[1])
    assert data == {**report, "source_root": str(root.resolve()), "complete": complete,
                    "branch_measurements": [], "branch_measurement_errors": [],
                    "branch_measurement_scope": results.BRANCH_SCOPE}
    assert list(tmp_path.glob("*.txt")) == [target]


def branch_fixture(root, *, selector="on_policy", seed=0, step=25, arm="selection_reduced"):
    directory = root / f"branches/{selector}/states/s{seed}-t{step}/points/view-{step}/{arm}"
    directory.mkdir(parents=True)
    result = {"complete": True, "completed_steps": step+10, "rewards": {"q0": .25, "q1": .75},
              "used_gpu_seconds": 80, "cost": {"train": 80}}
    (directory / "result.json").write_text(json.dumps(result))
    digest = hashlib.sha256((directory / "result.json").read_bytes()).hexdigest()
    (directory / "result.sha256.json").write_text(json.dumps({"sha256": digest}))
    curve = {"result_sha256": digest, "points": {
        str(step): {"updates": 0, "reward": .25},
        str(step+5): {"updates": 5, "reward": .375},
        str(step+10): {"updates": 10, "reward": .5, "final": True}}}
    (directory / "curve.json").write_text(json.dumps(curve))
    return directory, result, curve


def test_incomplete_pair_exports_independent_measured_arm_without_h(tmp_path, monkeypatch):
    root = tmp_path / "run"
    directory, endpoint, curve = branch_fixture(root)
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.iterdir()}
    report = {"rows": [], "development_rows": [], "missing_states": ["s3-t25"],
              "missing_development_states": ["s0-t25"], "summary": {"complete_test_states": 0}}
    target = tmp_path / "results.txt"
    def regenerate(*args, **kwargs):
        (root / "report.json").write_text(json.dumps(report))
        (root / "curves.csv").write_text("role,seed,arm,reward\n")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(results.subprocess, "run", regenerate)
    monkeypatch.setattr(sys, "argv", ["selector_pair_results", "--root", str(root), "--out", str(target)])
    results.main()
    text = target.read_text()
    data = json.loads(text.split("DATA_JSON\n")[1])
    assert data["rows"] == data["development_rows"] == []
    assert data["summary"] == report["summary"]
    assert data["complete"] is False
    row, = data["branch_measurements"]
    assert row["source_result"] == endpoint and row["source_curve"] == curve
    assert row["mean_reward"] == .5 and row["updates"] == 10
    assert row["question_count"] == 2 and row["issues"] == []
    assert row["eligible_for_paired_comparison"] is False
    assert row["independently_certified"] is False
    assert [p["reward"] for p in row["curve_points"]] == [.25, .375, .5]
    assert all("gpu_seconds" not in p for p in row["curve_points"])
    assert "crossing" not in row and "H" not in row and "cost_to_target" not in row
    assert "INDEPENDENT BRANCH MEASUREMENTS" in text
    assert "selection_reduced,saved_branch_measurement,10,0.5,2" in text
    assert "selection_reduced,5,0.375," in text
    assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.iterdir()} == before
    assert list(tmp_path.glob("*.txt")) == [target]


def test_endpoint_export_does_not_require_curve_or_other_selector(tmp_path):
    directory, _, _ = branch_fixture(tmp_path)
    (directory / "curve.json").unlink()
    rows, errors = results.saved_branch_measurements(tmp_path)
    assert not errors
    assert rows[0]["mean_reward"] == .5
    assert rows[0]["curve_points"] == []


@pytest.mark.parametrize("damage", ["seal_missing", "seal_changed", "incomplete", "negative", "nan", "stop_before_prefix"])
def test_invalid_endpoint_never_becomes_measured_row(tmp_path, damage):
    directory, endpoint, _ = branch_fixture(tmp_path)
    if damage == "seal_missing":
        (directory / "result.sha256.json").unlink()
    elif damage == "seal_changed":
        (directory / "result.sha256.json").write_text('{"sha256":"wrong"}')
    else:
        if damage == "incomplete":
            endpoint["complete"] = False
        elif damage == "negative":
            endpoint["rewards"]["q0"] = -.2
        elif damage == "nan":
            endpoint["rewards"]["q0"] = float("nan")
        else:
            endpoint["completed_steps"] = 24
        (directory / "result.json").write_text(json.dumps(endpoint))
        digest = hashlib.sha256((directory / "result.json").read_bytes()).hexdigest()
        (directory / "result.sha256.json").write_text(json.dumps({"sha256": digest}))
    rows, errors = results.saved_branch_measurements(tmp_path)
    if damage == "nan":
        assert not rows and errors
    else:
        assert not errors and rows[0]["issues"]
        assert rows[0]["mean_reward"] is None and rows[0]["updates"] is None
        assert rows[0]["curve_points"] == []
        assert rows[0]["status"] == "unverified_source_only"


@pytest.mark.parametrize("damage", ["binding", "updates", "reward", "final", "json"])
def test_bad_curve_keeps_separate_endpoint_but_no_curve_numbers(tmp_path, damage):
    directory, _, curve = branch_fixture(tmp_path)
    if damage == "binding":
        curve["result_sha256"] = "wrong"
    elif damage == "updates":
        curve["points"]["30"]["updates"] = 100
    elif damage == "reward":
        curve["points"]["30"]["reward"] = 2
    elif damage == "final":
        curve["points"]["35"]["reward"] = .25
    (directory / "curve.json").write_text("{" if damage == "json" else json.dumps(curve))
    rows, errors = results.saved_branch_measurements(tmp_path)
    assert not errors and rows[0]["mean_reward"] == .5
    assert rows[0]["curve_points"] == [] and rows[0]["issues"]


def test_only_known_current_layout_arms_are_included(tmp_path):
    branch_fixture(tmp_path, selector="on_policy", seed=3, arm="random_full")
    branch_fixture(tmp_path, selector="cached", seed=3, arm="selection_full")
    branch_fixture(tmp_path, selector="adaptive-cached", seed=3, arm="selection_full")
    branch_fixture(tmp_path, selector="cached", seed=3, arm="random_full")
    branch_fixture(tmp_path, selector="adaptive-cached", seed=0)
    branch_fixture(tmp_path, selector="discarded")
    branch_fixture(tmp_path / "archive")
    rows, errors = results.saved_branch_measurements(tmp_path)
    assert not errors and len(rows) == 3
    assert {row["selector_branch"] for row in rows} == {"on_policy", "cached", "adaptive-cached"}
    assert all(row["role"] == "test" for row in rows)
    assert all(row["eligible_for_paired_comparison"] is False for row in rows)


def test_symlink_outside_root_is_not_exported(tmp_path):
    root = tmp_path / "run"
    outside, _, _ = branch_fixture(tmp_path / "outside")
    link = root / "branches/on_policy/states/s0-t25/points/view-25/selection_reduced"
    link.parent.mkdir(parents=True)
    link.symlink_to(outside, target_is_directory=True)
    rows, errors = results.saved_branch_measurements(root)
    assert not rows and errors
    assert "escapes experiment root" in errors[0]["error"]


def test_failed_report_never_exports_stale_data(tmp_path, monkeypatch):
    root = tmp_path / "run"
    root.mkdir()
    (root / "report.json").write_text(json.dumps({"missing_states": [], "missing_development_states": []}))
    (root / "curves.csv").write_text("stale curves")
    target = tmp_path / "selector-pair-results.txt"
    target.write_text("previous valid export")
    monkeypatch.setattr(results.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=80))
    monkeypatch.setattr(sys, "argv", ["selector_pair_results", "--root", str(root), "--out", str(target)])

    with pytest.raises(SystemExit) as failure:
        results.main()

    assert failure.value.code == 80
    assert target.read_text() == "previous valid export"
    assert list(tmp_path.glob("*.txt")) == [target]
