"""Read-only IT status stays visible during preparation and fixed-panel checks."""

import hashlib
import json
import os
import sys
from unittest.mock import patch

import pytest

from srgc_rebuttal.plan import input_path, load_plan
from srgc_rebuttal.runtime import atomic_json, lease
from srgc_research.dispatch import model_status
from srgc_research.dispatch.gemma4_it import preflight
from srgc_research.tests.test_model_results import make_cohort, saved


def run_status(cohort, capsys, *options):
    with patch.dict(os.environ, cohort.env, clear=True), \
            patch.dict(sys.modules, {"torch": None, "transformers": None, "peft": None}):
        code = model_status.main(["gemma4_it", "--root", str(cohort.root), *options])
    return code, capsys.readouterr()


def test_status_skips_rollout_tree_and_announces_root_before_loading(tmp_path, capsys):
    cohort = make_cohort("gemma4_it", tmp_path)
    # A rollout archive is unrelated to status metadata and can be huge on NFS.
    archive = cohort.root / "rollout-archive"
    archive.symlink_to(tmp_path / "other-storage", target_is_directory=True)
    before = saved(cohort)
    original = model_status.load_reports

    def loading(*args):
        assert "Reading Gemma-4-12B-IT" in capsys.readouterr().err
        return original(*args)

    with patch.object(model_status, "load_reports", side_effect=loading), \
            patch("os.walk", side_effect=AssertionError("status walked the experiment tree")):
        code, output = run_status(cohort, capsys)
    assert code == 0
    assert "MATH" in output.out and "MBPP" in output.out
    assert "PREFLIGHT NOT STARTED" in output.out
    assert saved(cohort) == before


def test_missing_it_plans_show_preparing_and_separate_historical_pt_path(tmp_path, capsys):
    cohort = make_cohort("gemma4_it", tmp_path)
    for path in cohort.plans.values():
        path.unlink()
    old_root = cohort.work / "srgc-rebuttal/gemma4-12b-pt-v1"
    old = old_root / "experiments/gemma4-12b-pt-math.json"
    atomic_json(old, {"model": "google/gemma-4-12B", "historical_rewards": [0, 1]})
    old_bytes = old.read_bytes()
    with lease(cohort.root / ".start-prepare.lock"):
        code, output = run_status(cohort, capsys)
    assert code == 0  # Initial preparation is a state, not a corrupt experiment.
    assert "MATH PREPARING" in output.out and "MBPP PREPARING" in output.out
    assert "IT experiment plan has not been created" in output.out
    assert "No such file" not in output.out and "READ ERRORS" not in output.out
    assert f"Historical PT records: {old_root}" in output.out
    assert "model_status gemma4 --root" in output.out
    assert "initial candidate success" not in output.out
    assert old.read_bytes() == old_bytes


def test_public_status_on_a_fully_empty_it_root_identifies_required_preparation(tmp_path):
    import subprocess
    from pathlib import Path
    root = tmp_path / "work/srgc-rebuttal/gemma4-12b-it-v1"
    environment = {**os.environ, "GROUP_VOLUME": str(tmp_path), "OM_WORK": str(tmp_path / "work"),
                   "SRGC_GEMMA_IT_ROOT": str(root), "GEMMA_PYTHON": "/missing",
                   "SRGC_GEMMA_LEGACY_PROTOCOL": "0", "SRGC_GEMMA_IT_PROTOCOL": "1"}
    repo = Path(__file__).resolve().parents[2]
    result = subprocess.run(["sh", str(repo / "scripts/run_srgc_gemma4.sh"), "status"],
                            cwd="/tmp", env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "MATH  NOT INITIALIZED" in result.stdout and "MBPP  NOT INITIALIZED" in result.stdout
    assert "all prepare --root" in result.stdout
    assert "No such file" not in result.stdout and "READ ERRORS" not in result.stdout
    assert not root.exists()  # Status must not invent a plan or fabricate a run.


def test_missing_it_plan_with_saved_queue_work_is_still_an_error(tmp_path, capsys):
    cohort = make_cohort("gemma4_it", tmp_path)
    cohort.plans["math"].unlink()
    receipt = cohort.root / "runs/math/.queue/tasks/seed-5.cache.json"
    atomic_json(receipt, {"status": "interrupted", "attempt": 1})
    before = saved(cohort)
    code, output = run_status(cohort, capsys)
    assert code == 1 and "restore the original plan" in output.out
    assert "MBPP PREFLIGHT NOT STARTED" in output.out
    assert saved(cohort) == before


@pytest.mark.parametrize("correct_per_prompt,state", [(0, "PAUSED"), (2, "PASS")])
def test_status_displays_partial_then_validated_preflight_without_a_model(tmp_path, capsys,
                                                                       correct_per_prompt, state):
    cohort = make_cohort("gemma4_it", tmp_path)
    path = cohort.plans["math"]
    plan = load_plan(path)
    data = json.loads(input_path(path, plan, 5).read_text())
    protocol = preflight.identity(path, plan, data)
    directory = preflight.folder(path, plan)
    rows = []
    for candidate in protocol["candidate_ids"]:
        row = {"protocol": protocol, "id": candidate,
               "rewards": [1] * correct_per_prompt + [0] * (8 - correct_per_prompt),
               "responses": ["synthetic status fixture"] * 8, "token_ids": [[9, 106]] * 8}
        rows.append(row)
        atomic_json(directory / f"{hashlib.sha256(candidate.encode()).hexdigest()}.json", row)
        if len(rows) == 1:
            code, output = run_status(cohort, capsys)
            assert code == 0 and "MATH PREFLIGHT PARTIAL | 1/8 prompts recorded" in output.out
            assert "correct (" not in output.out
    summary = preflight.summarize(protocol, rows)
    atomic_json(directory / "summary.json", summary)
    before = saved(cohort)
    code, output = run_status(cohort, capsys)
    assert code == 0 and f"MATH PREFLIGHT {state} | 8/8 prompts recorded" in output.out
    assert f"{8 * correct_per_prompt}/64 correct" in output.out
    assert "full initial-cache accuracy is separate" in output.out
    assert saved(cohort) == before
    code, output = run_status(cohort, capsys, "--json")
    assert code == 0 and output.err == ""
    report = json.loads(output.out)["datasets"][0]
    assert report["preflight"]["state"] == state
    assert report["initial_candidate_accuracy"] == []
    atomic_json(directory / "summary.json", {**summary, "correct": 64})
    code, output = run_status(cohort, capsys)
    assert code == 1 and "PREFLIGHT INVALID" in output.out
    assert "summary differs from raw receipts" in output.out


def test_status_rejects_escaped_plan_metadata(tmp_path, capsys):
    cohort = make_cohort("gemma4_it", tmp_path)
    experiments = cohort.root / "experiments"
    moved = tmp_path / "escaped-plans"
    experiments.rename(moved)
    experiments.symlink_to(moved, target_is_directory=True)
    code, output = run_status(cohort, capsys)
    assert code == 2 and "path must be below group storage" in output.err
