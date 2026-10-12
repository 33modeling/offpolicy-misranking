"""Switch timing is visible for all models without touching experiment state."""

import json
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts import srgc_extra_status as extras
from scripts import srgc_live_status as live
from srgc_rebuttal.plan import load_plan
from srgc_rebuttal.runtime import atomic_json, lease
from srgc_research.dispatch import model_status
from srgc_research.dispatch.qwen_status import render
from srgc_research.tests.test_model_results import REPO, finish, saved
from srgc_research.tests.test_model_results import (
    cohort as cohort,  # noqa: PLC0414 -- Parameterized fixture.
)


def test_final_switch_boundary_is_visible_in_public_text_and_json_for_each_model(cohort):
    for name, step in (("math", 75), ("mbpp", 125)):
        folder = finish(cohort, name, arm="switch")
        path = folder / "switch-endpoint.json"
        atomic_json(path, {**json.loads(path.read_text()), "switched_at": step})
    before = saved(cohort)
    script = REPO / "scripts" / f"run_srgc_{cohort.model}.sh"
    text = subprocess.run(["sh", str(script), "status"], cwd="/tmp", env=cohort.env,
                          capture_output=True, text=True, timeout=30, check=False)
    assert text.returncode == 0, text.stdout + text.stderr
    assert "SW@75" in text.stdout and "SW@125" in text.stdout
    assert "MATH SWITCH DECISIONS" in text.stdout and "MBPP SWITCH DECISIONS" in text.stdout
    assert "Switch after update" in text.stdout and "First SR update" in text.stdout
    for name, step in (("math", 75), ("mbpp", 125)):
        result = subprocess.run(["sh", str(script), name, "status", "--json"], cwd="/tmp", env=cohort.env,
                                capture_output=True, text=True, timeout=30, check=False)
        assert result.returncode == 0, result.stdout + result.stderr
        data = json.loads(result.stdout)
        task = next(row for row in data["datasets"][0]["tasks"] if row["task"] == "seed-5.switch")
        assert task["status"] == "complete" and task["switched_at"] == step
        assert live.switch_info(task)["first_sr_update"] == step + 1
    assert saved(cohort) == before


def test_live_switch_and_no_switch_are_read_from_saved_progress_on_all_models(cohort):
    folder = finish(cohort, "math")
    queue = folder.parent / ".queue"
    atomic_json(folder / "switch-progress.json", {"seed": 5, "arm": "switch", "step": 100, "switched_at": 75})
    with lease(queue / "leases/seed-5.switch.lock"):
        before = saved(cohort)
        reports = model_status.load_reports(cohort.model, "math", cohort.root, cohort.env)
        task = next(row for row in reports[0]["tasks"] if row["task"] == "seed-5.switch")
        assert task["status"] == "running" and task["completed_steps"] == 100
        assert "switch after update 75 | first SR update 76 | selector sr" in render(reports, cohort.root)
        assert saved(cohort) == before
    atomic_json(folder / "switch-progress.json", {"seed": 5, "arm": "switch", "step": 100, "switched_at": None})
    reports = model_status.load_reports(cohort.model, "math", cohort.root, cohort.env)
    task = next(row for row in reports[0]["tasks"] if row["task"] == "seed-5.switch")
    assert live.switch_info(task)["state"] == "NOT YET"
    assert live.switch_info(task)["selector"] == "on_policy"


def test_validated_final_no_switch_is_distinct_from_missing_live_progress(cohort):
    finish(cohort, "math", arm="switch")
    reports = model_status.load_reports(cohort.model, "math", cohort.root, cohort.env)
    task = next(row for row in reports[0]["tasks"] if row["task"] == "seed-5.switch")
    assert live.switch_info(task)["state"] == "NEVER SWITCHED"
    assert "NEVER SWITCHED" in render(reports, cohort.root)
    task.update(status="running", step=None, completed_steps=None, switch_recorded=False)
    assert live.switch_info(task)["state"] == "NOT RECORDED"


@pytest.mark.parametrize("value", [True, False, -1, 276, "75", 75.0])
def test_invalid_switch_values_cannot_be_displayed_as_a_confirmed_switch(value):
    row = {"arm": "switch", "status": "running", "completed_steps": 100,
           "total_steps": 275, "switched_at": value}
    assert live.switch_info(row)["state"] == "INVALID RECORD"
    assert "SW@" not in live.switch_cell(row, "RUN 100/275")


def test_stale_switch_after_current_resume_checkpoint_is_not_reported_as_confirmed():
    row = {"arm": "switch", "status": "running", "completed_steps": 50,
           "total_steps": 275, "switched_at": 75}
    assert live.switch_info(row)["state"] == "INVALID RECORD"
    assert "switch after update" not in live.switch_detail(row)


def test_fixed_schedule_does_not_imply_an_actual_switch():
    row = {"arm": "replicate1-switch_fixed200", "state": "waiting", "step": None, "total": 275}
    assert live.switch_info(row)["state"] == "PENDING"
    assert live.switch_info(row)["step"] is None
    row.update(state="checkpoint_saved", step=225, switched_at=200, switch_recorded=True)
    assert "switch after update 200 | first SR update 201" in live.switch_detail(row)


def test_repeated_switches_display_actual_transitions_and_current_selector_without_mutation():
    row = {"arm": "switch_repeat", "state": "checkpoint_saved", "step": 150, "total": 275,
           "switched_at": 75, "transitions": [{"step": 75, "to": "sr"}, {"step": 125, "to": "on"}]}
    before = json.dumps(row, sort_keys=True)
    assert live.switch_info(row)["selector"] == "on_policy"
    assert "transitions after 75 -> sr; after 125 -> on_policy" in live.switch_detail(row)
    assert json.dumps(row, sort_keys=True) == before
    row["transitions"].reverse()
    assert live.switch_info(row)["state"] == "INVALID RECORD"


@pytest.mark.parametrize("transitions", ["invalid", [{"step": 75, "to": []}],
    [{"step": True, "to": "sr"}], [{"step": 75, "to": "unknown"}], [None]])
def test_bad_transition_records_are_visible_without_crashing_status(transitions):
    row = {"arm": "switch_repeat", "state": "checkpoint_saved", "step": 150, "total": 275,
           "switched_at": 75, "transitions": transitions}
    assert live.switch_detail(row) == "switch invalid record"


def test_generic_status_shows_switch_decisions_and_keeps_other_arms_unchanged():
    tasks = [{"seed": 5, "arm": arm, "task": f"seed-5.{arm}", "status": "complete", "step": 275,
              "total_steps": 275, "reward_percent": 50., "switched_at": 75 if arm == "switch" else None}
             for arm in ("on_policy", "switch", "sr", "random")]
    report = {"dataset": "math_train", "output_root": "/recorded", "generated": 0,
              "tasks": tasks, "workers": [], "warnings": [], "errors": [], "counts": {"complete": 4}}
    with patch.object(live, "gpu_memory_summary", return_value=None):
        text = live.render(report)
    assert "done 50.0% SW@75" in text
    assert "switch after update 75 | first SR update 76" in text
    assert live.cell(tasks[0]) == "done 50.0%" and live.switch_info(tasks[2]) is None


def test_extra_status_reads_progress_and_validated_final_switch_history(cohort, tmp_path):
    out = tmp_path / "extra"
    task = SimpleNamespace(key="math.seed-5.switch_repeat", dataset="math", seed=5,
        name="switch_repeat", arm="switch_repeat", plan=cohort.plans["math"], out=out,
        folder=out, receipt=out / "queue.json")
    total = load_plan(task.plan)["total_updates"]
    progress = {"seed": 5, "arm": "switch_repeat", "step": 150, "switched_at": 75,
                "transitions": [{"step": 75, "to": "sr"}, {"step": 125, "to": "on"}]}
    atomic_json(out / "switch_repeat-progress.json", progress)
    before = {p: p.read_bytes() for p in out.rglob("*") if p.is_file()}
    row = extras.inspect(task)
    assert live.switch_info(row)["selector"] == "on_policy"
    assert {p: p.read_bytes() for p in out.rglob("*") if p.is_file()} == before
    atomic_json(task.receipt, {"status": "complete", "verified_files": {"final": "recorded"}})
    atomic_json(out / "switch_repeat-endpoint.json", {**progress, "step": total,
        "switched_at": 75, "transitions": [{"step": 75, "to": "sr"}, {"step": 125, "to": "on"},
                                          {"step": 175, "to": "sr"}]})
    with patch.object(extras, "signature", return_value={"final": "recorded"}):
        row = extras.inspect(task)
    assert row["state"] == "complete" and live.switch_info(row)["selector"] == "sr"
    with patch.object(extras, "signature", side_effect=[{"final": "recorded"}, {"final": "replaced"}]):
        row = extras.inspect(task)
    assert row["state"] == "invalid" and "endpoint changed" in row["error"]
