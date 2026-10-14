"""GPU status separates missing costs, partial receipts, and complete measurements."""

import copy
import json
import os
import subprocess
import sys

import pytest

from scripts.srgc_time_status import attach_scope, from_arm, from_ledger, render, timing
from srgc_research.tests.test_model_results import (
    cohort as cohort,  # noqa: PLC0414 -- re-export pytest fixture
)
from srgc_research.tests.test_model_results import finish, saved
from srgc_research.tests.test_model_status import read_status


def test_complete_phase_costs_are_converted_once_and_do_not_include_sessions():
    ledger = {"recorded_phases": 3, "complete": True, "unfinished_phases": [],
              "known_gpu_seconds": {"selection_gpu_seconds": 7200., "training_gpu_seconds": 14400.,
                                    "evaluation_gpu_seconds": 3600.}, "total_gpu_seconds": 25200.}
    value = from_ledger(ledger, completed=True)
    assert value["record"] == "COMPLETE"
    assert value["gpu_seconds"]["primary"] == 21600.
    assert value["gpu_hours"]["primary"] == 6.
    assert value["gpu_hours"]["phases"] == 7.
    assert value["gpu_hours"]["preparation"] == 0.


def test_unfinished_phase_is_a_lower_bound_not_an_estimated_live_timer():
    cost = {"complete": False, "known_completed_phase_gpu_seconds": {
                "selection_gpu_seconds": 3600., "training_gpu_seconds": 0.},
            "unfinished_phases": [{"id": "pending", "phase": "training", "checkpoint": 50}]}
    original = copy.deepcopy(cost)
    value = from_arm(cost, completed=False)
    assert value["record"] == "PARTIAL" and value["lower_bound"]
    assert value["gpu_hours"]["selection"] == 1.
    assert value["gpu_hours"]["training"] is None
    assert value["gpu_hours"]["phases"] == 1.
    text = "\n".join(render({"tasks": [{"seed": 5, "arm": "switch", "status": "running", "timing": value}]}))
    assert ">=1.000" in text and "mean +/-" not in text
    assert cost == original


def test_missing_measurements_are_never_zero_and_invalid_rows_are_hidden():
    assert timing()["gpu_hours"]["phases"] is None
    measured = timing("COMPLETE", {key: 3600 for key in ("primary", "phases")})
    text = "\n".join(render({"tasks": [{"seed": 5, "arm": "sr", "status": "invalid", "timing": measured}]}))
    assert "INVALID" in text and "1.000" not in text
    with pytest.raises(ValueError):
        timing("COMPLETE", {"primary": float("nan")})
    with pytest.raises(ValueError):
        timing("COMPLETE", {"training": -1})


def test_completed_zero_cost_is_different_from_a_missing_scope():
    ledger = {"recorded_phases": 1, "complete": True, "unfinished_phases": [],
              "known_gpu_seconds": {"training_gpu_seconds": 0., "selection_gpu_seconds": 0.},
              "total_gpu_seconds": 0.}
    assert from_ledger(ledger, completed=True)["gpu_hours"]["phases"] == 0.
    assert from_ledger({**ledger, "recorded_phases": 0}, completed=True)["record"] == "MISSING"


def test_public_model_status_exposes_both_datasets_without_writes(cohort, capsys):
    for name in ("math", "mbpp"):
        finish(cohort, name)
    before = saved(cohort)
    code, data = read_status(cohort, capsys)
    assert code == 0
    for report in data["datasets"]:
        assert "GPU-seconds" in report["timing_units"]
        assert all("timing" in row for row in report["tasks"])
        row = next(row for row in report["tasks"] if row["arm"] == "on_policy" and row["seed"] == 5)
        assert row["timing"]["record"] == "MISSING"
        assert row["timing"]["gpu_hours"]["phases"] is None
    assert saved(cohort) == before


def test_public_model_status_reports_complete_measured_components(cohort, capsys):
    folder = finish(cohort, "math")
    endpoint_path = folder / "on_policy-endpoint.json"
    endpoint = json.loads(endpoint_path.read_text())
    endpoint["cost_measurement_complete"] = True
    endpoint_path.write_text(json.dumps(endpoint))
    for phase, seconds in (("selection", 7200.), ("training", 14400.), ("evaluation", 3600.)):
        path = folder / "cost-receipts/on_policy" / (phase + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"id": phase, "phase": phase, "state": "finished",
            "gpu_count": 4, "gpu_seconds": seconds, "wall_seconds": seconds / 4, "checkpoint": 275}))
    session = folder / "invocations/on_policy/session.json"
    session.parent.mkdir(parents=True, exist_ok=True)
    session.write_text(json.dumps({"id": "session", "phase": "session", "state": "finished",
        "gpu_count": 4, "gpu_seconds": 28800., "wall_seconds": 7200., "checkpoint": 275}))
    before = saved(cohort)
    code, value = read_status(cohort, capsys)
    assert code == 0
    row = next(row for row in value["datasets"][0]["tasks"] if row["task"] == "seed-5.on_policy")
    assert row["timing"]["record"] == "COMPLETE"
    assert row["timing"]["gpu_hours"]["selection"] == 2.
    assert row["timing"]["gpu_hours"]["training"] == 4.
    assert row["timing"]["gpu_hours"]["evaluation"] == 1.
    assert row["timing"]["gpu_hours"]["primary"] == 6.
    assert row["timing"]["gpu_hours"]["phases"] == 7.
    assert saved(cohort) == before


def test_extra_scope_reads_retry_costs_once_and_preserves_files(tmp_path):
    receipts = tmp_path / "cost-receipts/switch_fixed50"
    receipts.mkdir(parents=True)
    for name, value in (("first", 3600.), ("retry", 7200.)):
        (receipts / (name + ".json")).write_text(json.dumps({"id": name, "phase": "training",
            "state": "finished", "gpu_seconds": value, "checkpoint": 26}))
    endpoint = tmp_path / "switch_fixed50-endpoint.json"
    endpoint.write_text(json.dumps({"cost_measurement_complete": True}))
    before = {str(path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    row = attach_scope({"state": "complete", "seed": 5, "arm": "switch_fixed50"}, tmp_path, arm="switch_fixed50")
    assert row["timing"]["gpu_hours"]["training"] == 3.
    assert {str(path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


def test_invalid_extra_cost_does_not_invent_a_total(tmp_path):
    directory = tmp_path / "cost-receipts"
    directory.mkdir()
    (directory / "bad.json").write_text(json.dumps({"id": "bad", "phase": "training",
        "state": "finished", "gpu_seconds": -1, "checkpoint": 1}))
    row = attach_scope({"state": "running", "arm": "information-t0"}, tmp_path)
    assert row["timing"]["record"] == "INVALID"
    assert row["timing"]["gpu_hours"]["phases"] is None


def test_research_status_shell_reports_both_datasets_without_preparing_files(tmp_path):
    from srgc_research.tests.test_model_status import REPO

    root = tmp_path / "research"
    environment = {**os.environ, "GROUP_VOLUME": str(tmp_path), "OM_WORK": str(tmp_path / "work"),
                   "SRGC_RESEARCH_ROOT": str(root), "PAIR_PYTHON": sys.executable}
    before = sorted(str(path) for path in tmp_path.rglob("*"))
    result = subprocess.run(["sh", str(REPO / "scripts/run_srgc_research.sh"), "all", "n01", "status", "--json"],
                            cwd="/tmp", env=environment, capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert {row["dataset"] for row in data["rows"]} == {"math", "mbpp"}
    assert all(row["timing"]["record"] == "MISSING" for row in data["rows"])
    assert before == sorted(str(path) for path in tmp_path.rglob("*"))


def test_research_status_lock_observation_never_creates_a_missing_lock(tmp_path):
    from scripts.srgc_research_status import observed_lease

    path = tmp_path / "absent/.dispatch.lock"
    with observed_lease(path):
        pass
    assert not path.parent.exists()
