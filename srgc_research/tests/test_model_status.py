"""The three public runners expose the same read-only status and results data."""

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from srgc_rebuttal.plan import input_path, load_plan
from srgc_rebuttal.runtime import atomic_json
from srgc_research.tests.test_model_results import (
    cohort as cohort,  # noqa: PLC0414 -- Re-export a pytest fixture.
)
from srgc_research.tests.test_model_results import finish, read_result, saved

RUNNERS = {"qwen35": "qwen_run", "llama31": "llama_run", "gemma4": "gemma_run"}
REPO = Path(__file__).resolve().parents[2]


def read_status(cohort, capsys):
    runner = importlib.import_module("srgc_research.dispatch." + RUNNERS[cohort.model])
    with patch.dict(os.environ, cohort.env, clear=True), \
            patch.dict(sys.modules, {"torch": None, "transformers": None, "peft": None}):
        code = runner.main(["all", "status", "--root", str(cohort.root), "--json"])
    return code, json.loads(capsys.readouterr().out)


def test_public_status_and_results_share_validated_tasks_and_initial_cache_statistics(cohort, capsys):
    from srgc_research.dispatch.model_results import export

    for name in ("math", "mbpp"):
        finish(cohort, name, reward=.3)
    before = saved(cohort)
    code, value = read_status(cohort, capsys)
    assert code == 0 and value["model"] == cohort.adapter.MODEL
    assert len(value["datasets"]) == 2
    assert export(cohort.model, environment=cohort.env) == 0
    result = read_result(cohort)
    for name, status in zip(("math", "mbpp"), value["datasets"], strict=True):
        assert status["initial_candidate_accuracy"] == [
            {"seed": 5, "responses": 3200, "success_percent": 50.0}]
        assert result["datasets"][name]["initial_candidate_accuracy"] == status["initial_candidate_accuracy"]
        assert result["datasets"][name]["counts"] == status["counts"]
        assert next(row for row in status["tasks"] if row["task"] == "seed-5.on_policy")["reward_percent"] == 30.0
        assert [(r["task"], r["status"], r["reward_percent"]) for r in status["tasks"]] == [
            (r["task"], r["status"], r["reward_percent"]) for r in result["datasets"][name]["tasks"]]
    assert saved(cohort) == before


def test_one_corrupt_cached_seed_keeps_other_seeds_and_dataset_visible(cohort, capsys):
    from srgc_research.dispatch.model_results import export

    finish(cohort, "math", seed=5)
    finish(cohort, "math", seed=6)
    path = input_path(cohort.plans["math"], load_plan(cohort.plans["math"]), 5)
    data = json.loads(path.read_text())
    first = data["candidate_ids"][0]
    data["cached_rewards"][first] = [2] * 8
    atomic_json(path, data)
    before = saved(cohort)
    code, value = read_status(cohort, capsys)
    assert code == 1
    report = value["datasets"][0]
    assert report["prepared"] and len(report["tasks"]) == 30
    assert report["errors"] and not report["complete"]
    assert all(row["status"] == "invalid" and row["reward_percent"] is None
               for row in report["tasks"] if row["seed"] == 5)
    assert next(row for row in report["tasks"] if row["task"] == "seed-6.on_policy")["status"] == "complete"
    assert report["initial_candidate_accuracy"] == [
        {"seed": 6, "responses": 3200, "success_percent": 50.0}]
    assert value["datasets"][1]["prepared"]
    assert export(cohort.model, environment=cohort.env) == 1
    assert len(read_result(cohort)["datasets"]["math"]["endpoints"]) == 1
    assert saved(cohort) == before


def test_public_shell_status_shortcut_and_dataset_filter_show_the_same_data(cohort):
    script = REPO / "scripts" / f"run_srgc_{cohort.model}.sh"
    finish(cohort, "math")
    before = saved(cohort)
    for arguments, expected in ((("status",), 2), (("math", "status"), 1)):
        result = subprocess.run(["sh", str(script), *arguments, "--json"],
                                cwd="/tmp", env=cohort.env, capture_output=True, text=True,
                                timeout=30, check=False)
        assert result.returncode == 0, result.stdout + result.stderr
        value = json.loads(result.stdout)
        assert value["model"] == cohort.adapter.MODEL and len(value["datasets"]) == expected
        assert value["datasets"][0]["initial_candidate_accuracy"][0]["responses"] == 3200
    assert saved(cohort) == before


def test_valid_partial_cache_is_pending_and_has_no_premature_success_rate(cohort, capsys):
    plan = load_plan(cohort.plans["math"])
    path = input_path(cohort.plans["math"], plan, 5)
    data = json.loads(path.read_text())
    data["cached_rewards"] = {data["candidate_ids"][0]: [0, 1] * 4}
    data["provenance"]["cache"] = {
        "model": cohort.adapter.MODEL, "model_revision": cohort.adapter.REVISION,
        "responses": 8, "max_new_tokens": plan["max_new_tokens"], "cache_seed": 5,
        "verifier": plan["verifier"], "attention": "eager"}
    atomic_json(path, data)
    before = saved(cohort)
    code, value = read_status(cohort, capsys)
    report = value["datasets"][0]
    assert code == 0 and report["prepared"] and len(report["tasks"]) == 30
    assert report["initial_candidate_accuracy"] == []
    assert report["errors"] == []
    assert saved(cohort) == before


def test_public_results_failure_uses_the_same_exit_code_for_all_models(cohort, capsys):
    from srgc_research.dispatch import model_results

    runner = importlib.import_module("srgc_research.dispatch." + RUNNERS[cohort.model])
    with patch.object(model_results, "export", side_effect=ValueError("invalid source root")):
        assert runner.main(["results"]) == 1
    assert "ERROR: invalid source root" in capsys.readouterr().out
