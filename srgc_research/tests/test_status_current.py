"""Current status keeps progress visible and replaces one time/result snapshot."""

import json
import os
import sys
import time
from unittest.mock import patch

import pytest

from scripts.srgc_resumable_rollouts import ResumableRolloutMixin, _key
from srgc_rebuttal.plan import input_path, load_plan
from srgc_rebuttal.runtime import atomic_json, lease
from srgc_rebuttal.srgc import stream_seed
from srgc_research.dispatch import model_status, qwen_status
from srgc_research.tests.test_model_results import make_cohort, finish, saved
from srgc_research.tests.test_submission_time import progress


@pytest.fixture(params=("gemma4_it", "qwen35", "llama31", "gemma4"))
def cohort(request, tmp_path):
    return make_cohort(request.param, tmp_path)


def inspect(cohort, capsys, *, as_json=True, dataset="all"):
    with patch.dict(os.environ, cohort.env, clear=True), \
            patch.dict(sys.modules, {"torch": None, "transformers": None, "peft": None}):
        code = model_status.main([cohort.model, dataset, "--root", str(cohort.root),
                                  *(["--json"] if as_json else [])])
    captured = capsys.readouterr()
    assert code == 0, captured.out + captured.err
    return json.loads(captured.out) if as_json else captured.out


def test_status_replaces_latest_time_and_results_without_adding_previous_values(cohort, capsys):
    folder = finish(cohort, "math", reward=.3)
    timer = progress(folder, "on_policy")
    full_result = cohort.work / "results" / f"{cohort.model}-results.json"
    atomic_json(full_result, {"reward_curves": ["retained scientific evidence"]})
    full_bytes = full_result.read_bytes()
    before = saved(cohort)
    value = inspect(cohort, capsys)
    destination = cohort.work / "status" / f"{cohort.model}-latest.json"
    assert value["saved_results"] == str(destination)
    initial = json.loads(destination.read_text())
    task = next(row for row in initial["datasets"]["math"]["tasks"] if row["task"] == "seed-5.on_policy")
    assert task["reward_percent"] == 30.
    assert task["gpu_time"]["gpu_seconds"]["primary"] == 1400.
    assert saved(cohort) == before
    # An old snapshot must never be an input to the next cost/result snapshot.
    destination.write_text(json.dumps({"history": [initial] * 5, "gpu_seconds": 999999.}))
    inspect(cohort, capsys)
    repeated = json.loads(destination.read_text())
    assert repeated["datasets"] == initial["datasets"]
    assert "history" not in repeated
    assert list(destination.parent.iterdir()) == [destination]
    assert saved(cohort) == before

    document = json.loads(timer.read_text())
    for update in document["history"]:
        update["training_gpu_seconds"] = 8.
    atomic_json(timer, document)
    finish(cohort, "math", reward=.6)
    updated_source = saved(cohort)
    inspect(cohort, capsys)
    updated = json.loads(destination.read_text())
    task = next(row for row in updated["datasets"]["math"]["tasks"] if row["task"] == "seed-5.on_policy")
    assert task["reward_percent"] == 60.
    assert task["gpu_time"]["gpu_seconds"]["primary"] == 2400.
    assert saved(cohort) == updated_source
    assert full_result.read_bytes() == full_bytes


def test_dataset_filter_replaces_scope_without_retaining_other_dataset(cohort, capsys):
    inspect(cohort, capsys)
    value = inspect(cohort, capsys, dataset="math")
    current = json.loads((cohort.work / "status" / f"{cohort.model}-latest.json").read_text())
    assert len(value["datasets"]) == 1
    assert current["dataset_scope"] == "math" and set(current["datasets"]) == {"math"}


def test_terminal_shows_current_state_without_cost_history_tables(cohort, capsys):
    folder = finish(cohort, "math", reward=.3)
    progress(folder, "on_policy")
    text = inspect(cohort, capsys, as_json=False)
    assert "Tasks" in text and "DONE" in text
    assert "Saved GPU time/results:" in text
    for historical in ("GPU TIMES", "GPU STAGES", "RECORDED WORK", "SUBMITTED OLMO TIME BASIS",
                       "RESTART", "HISTORY", "all recorded attempts", "30.0%"):
        assert historical not in text


def test_training_progress_is_measured_updates_with_percent(cohort, capsys):
    queue = cohort.root / "runs/math/.queue"
    atomic_json(queue / "tasks/seed-5.prefix.json", {
        "task": "seed-5.prefix", "status": "running", "started": time.time() - 10,
        "host": "training-node", "worker_id": "training", "attempt": 1})
    atomic_json(queue.parent / "seed-5/cost-receipts/shared-prefix/current.json", {
        "phase": "training", "state": "started", "checkpoint": 3})
    with lease(queue / "leases/seed-5.prefix.lock"):
        text = inspect(cohort, capsys, as_json=False)
    assert "TRAIN 3/25 completed" in text and "12.0%" in text
    assert "update 4/25 in progress" in text
    assert "TRAIN 4/25 completed" not in text


@pytest.mark.parametrize("count", [None, 0, 37, 300])
def test_eval_progress_never_confuses_training_completion_with_saved_eval(count):
    row = {"status": "running", "arm": "on_policy", "phase_stage": "evaluation",
           "completed_steps": 275, "total_steps": 275, "progress_age_seconds": 1,
           "final_evaluation": {"saved_prompts": count, "total_prompts": 300,
                                "responses_per_prompt": 8}}
    text = qwen_status.progress(row)
    assert "EVAL" in text
    if count is None:
        assert "?/300 prompts" in text and "awaiting saved progress" in text
        assert "0.0%" not in text and "100.0%" not in text
    else:
        assert f"{count}/300 prompts" in text and f"{100 * count / 300:.1f}%" in text
    if count == 300:
        assert "awaiting endpoint" in text


def test_unknown_training_progress_is_not_reported_as_zero_percent():
    text = qwen_status.progress({"status": "running", "arm": "on_policy", "total_steps": 275,
                                 "completed_steps": None, "phase_stage": "training"})
    assert "TRAIN ?/275 completed" in text and "0.0%" not in text


def test_current_checkpoint_eval_uses_its_step_and_ignores_past_eval_after_training_resumes(cohort, capsys):
    folder = finish(cohort, "math")
    (folder / "on_policy-endpoint.json").unlink()
    plan = load_plan(cohort.plans["math"])
    data = json.loads(input_path(cohort.plans["math"], plan, 5).read_text())
    queue = folder.parent / ".queue"
    atomic_json(queue / "tasks/seed-5.on_policy.json", {
        "task": "seed-5.on_policy", "status": "running", "started": time.time() - 10,
        "host": "eval-node", "worker_id": "evaluation", "attempt": 1})
    training = folder / "cost-receipts/on_policy/training.json"
    atomic_json(training, {"phase": "training", "state": "finished", "checkpoint": 49})
    evaluation = folder / "reward-curves/on_policy/cost-receipts/step-50/eval.json"
    atomic_json(evaluation, {"phase": "evaluation", "state": "started", "checkpoint": 50})
    cache = folder / "rollout-cache/on_policy" / f"eval-{stream_seed(5, 50, 'reporting-evaluation')}"
    writer = ResumableRolloutMixin()
    for prompt in data["evaluation_ids"][:17]:
        writer._save_rollout(cache / _key(prompt, 8), [[1, 2]] * 8, [0, 1] * 4, 1)
    before = saved(cohort)
    with lease(queue / "leases/seed-5.on_policy.lock"):
        value = inspect(cohort, capsys)
        task = next(row for row in value["datasets"][0]["tasks"] if row["task"] == "seed-5.on_policy")
        assert task["evaluation_step"] == 50 and task["phase_stage"] == "evaluation"
        assert task["final_evaluation"]["saved_prompts"] == 17
        text = inspect(cohort, capsys, as_json=False)
        assert "eval at update 50" in text and "EVAL 17/300 prompts" in text and "5.7%" in text
        # New training receipts supersede an old checkpoint-evaluation journal.
        atomic_json(training, {"phase": "training", "state": "started", "checkpoint": 50})
        modified = evaluation.stat().st_mtime + 1
        os.utime(training, (modified, modified))
        text = inspect(cohort, capsys, as_json=False)
        assert "TRAIN 50/275 completed" in text and "EVAL 17/300 prompts" not in text
    after = saved(cohort)
    changed = {name for name in before if before[name] != after[name]}
    assert changed == {training}


def test_corrupt_checkpoint_progress_does_not_relabel_valid_inputs_as_invalid(tmp_path, capsys):
    cohort = make_cohort("gemma4_it", tmp_path)
    folder = finish(cohort, "math")
    queue = folder.parent / ".queue"
    atomic_json(queue / "tasks/seed-5.on_policy.json", {
        "task": "seed-5.on_policy", "status": "running", "started": time.time() - 10})
    curve = folder / "reward-curves/on_policy/cost-receipts/step-50/eval.json"
    curve.parent.mkdir(parents=True)
    curve.write_text("broken JSON")
    with lease(queue / "leases/seed-5.on_policy.lock"), patch.dict(os.environ, cohort.env, clear=True):
        code = model_status.main([cohort.model, "--root", str(cohort.root), "--json"])
    output = json.loads(capsys.readouterr().out)
    assert code == 1
    assert any("checkpoint eval progress" in error for error in output["datasets"][0]["errors"])
    assert not any(row["status"] == "invalid" for row in output["datasets"][0]["tasks"])


def test_snapshot_write_failure_does_not_hide_current_status(tmp_path, capsys):
    cohort = make_cohort("gemma4_it", tmp_path)
    with patch.dict(os.environ, cohort.env, clear=True), \
            patch("srgc_research.dispatch.status_current.atomic_json",
                  side_effect=PermissionError("read-only status directory")):
        code = model_status.main([cohort.model, "--root", str(cohort.root)])
    output = capsys.readouterr()
    assert code == 1 and "Tasks" in output.out
    assert "STATUS SAVE ERROR" in output.err
    assert "read-only status directory" in output.err


def test_tty_replaces_screen_and_json_has_no_terminal_escape_codes(tmp_path, capsys):
    cohort = make_cohort("gemma4_it", tmp_path)
    with patch.dict(os.environ, cohort.env, clear=True), patch.object(sys.stdout, "isatty", return_value=True):
        assert model_status.main([cohort.model, "--root", str(cohort.root)]) == 0
        output = capsys.readouterr()
        assert output.out.startswith("\x1b[H\x1b[2J")
        assert model_status.main([cohort.model, "--root", str(cohort.root), "--json"]) == 0
        output = capsys.readouterr()
        assert "\x1b" not in output.out and json.loads(output.out)["saved_results"]
