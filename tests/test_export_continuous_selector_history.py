import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SPEC = importlib.util.spec_from_file_location("export_e5_checkpoint_curves", SCRIPTS / "export_e5_checkpoint_curves.py")
curves = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = curves
SPEC.loader.exec_module(curves)
SPEC = importlib.util.spec_from_file_location("continuous_history", SCRIPTS / "export_continuous_selector_history.py")
history = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(history)


def put(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


@pytest.fixture
def run(tmp_path):
    root = tmp_path / "runs"
    directory = root / "math400-d0/s0"
    put(directory / "experiment.json", {"seed": 0, "drift": 0, "steps": 200,
                                        "eval_k": 8, "eval_prompts": 300})
    put(directory / "evaluation.json", {"val": [1, 2]})
    with (directory / "downstream_results.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(("selector", "seed", "drift", "reward_before", "reward_after"))
        for arm in ("fresh_r", "passrate_beta"):
            writer.writerow((arm, 0, 0, .2, .4))
    for arm in ("fresh_r", "passrate_beta"):
        policy = directory / arm / "policy"
        policy.mkdir(parents=True)
        log = ''.join(json.dumps({"step": t, "reward_mean": .2 + t / 1000,
                                 "step_seconds": 2., "grad_norm": 1., "loss": .1}) + '\n'
                      for t in range(1, 201))
        (policy / "grpo_stats.jsonl").write_text(log)
        put(policy / "policy_train.json", {"seed": 0, "start_step": 0, "completed_steps": 200,
            "world_size": 4, "grpo_stats_sha256": hashlib.sha256(log.encode()).hexdigest()})
        put(directory / arm / "curve.json", {"k": 8, "points": {
            str(t): {"updates": t, "reward": .2 + t / 1000} for t in (0, 25, 50, 100, 150, 200)}})
    return root, directory


def test_full_trajectory_has_no_100_step_cap(run):
    report = history.export(run[0], [25, 50, 100])
    assert report["end_step_limit"] is None
    assert report["branch_start_step"] == 0
    for arm in report["experiments"][0]["arms"].values():
        assert arm["history"]["last_logged_step"] == 200
        assert arm["points"][-1]["step"] == 200
        assert len(arm["history"]["training_log"]) == 200
        assert arm["history"]["training_log"][0]["timed_update_gpu_seconds"] == 8


def test_cutoffs_do_not_include_future_observations(run):
    report = history.export(run[0], [25, 50, 100])
    for arm in report["experiments"][0]["arms"].values():
        for view in arm["as_of"]:
            t = view["decision_step"]
            assert all(point["step"] <= t for point in view["points"])
            assert all(row["step"] <= t for row in view["training_log"])
            assert all(checkpoint["step"] <= t for checkpoint in view["checkpoints"])


def test_nonzero_experiments_are_not_stitched_or_read(run):
    path = run[0] / "math400-d25/s0/experiment.json"
    path.parent.mkdir(parents=True)
    path.write_text("invalid and must not be read")
    report = history.export(run[0], [25])
    assert len(report["experiments"]) == 1
    assert report["experiments"][0]["start_updates"] == 0


def test_no_target_filter_or_training_reward_substitution(run):
    report = history.export(run[0], [25])
    assert not report["training_rewards_used"]
    arm = report["experiments"][0]["arms"]["cached"]
    assert len(arm["points"]) == 6
    assert arm["points"][0]["reward"] < .35 < arm["points"][-1]["reward"]
    assert "H_gpu_hours" not in arm


def test_read_only_and_separate_nonoverwriting_outputs(run, tmp_path):
    root, _ = run
    before = {str(p): p.read_bytes() for p in curves.files(root)}
    report = history.export(root, [25, 50])
    out = tmp_path / "exports/new"
    history.write_outputs(report, out)
    assert before == {str(p): p.read_bytes() for p in curves.files(root)}
    assert len(list(csv.DictReader((out / "training-log.csv").open()))) == 400
    with pytest.raises(FileExistsError):
        history.write_outputs(report, out)
    with pytest.raises(ValueError, match="separate"):
        history.write_outputs(report, root / "bad")


def test_archived_duplicate_logs_are_deduplicated(run):
    _, directory = run
    policy = directory / "fresh_r/policy"
    checkpoint = policy / "checkpoint-000025"
    put(checkpoint / "checkpoint_state.json", {"completed_steps": 25, "world_size": 4})
    (checkpoint / "grpo_stats.jsonl").write_text(''.join((policy / "grpo_stats.jsonl").read_text().splitlines(keepends=True)[:25]))
    result = history.training_history(directory / "fresh_r")
    assert len(result["training_log"]) == 200
    assert not result["issues"]
    assert len(result["training_log"][0]["sources"]) == 2


def test_hash_mismatch_does_not_supply_history(run):
    _, directory = run
    path = directory / "fresh_r/policy/grpo_stats.jsonl"
    path.write_text(path.read_text() + "\n")
    result = history.training_history(directory / "fresh_r")
    assert not result["training_log"]
    assert "hash mismatch" in result["issues"][0]


def test_missing_world_size_is_not_zero_cost(run):
    _, directory = run
    manifest = directory / "fresh_r/policy/policy_train.json"
    value = json.loads(manifest.read_text())
    value.pop("world_size")
    put(manifest, value)
    result = history.training_history(directory / "fresh_r")
    assert result["training_log"][0]["timed_update_gpu_seconds"] is None


def test_bash_entry_from_other_directory(run, tmp_path):
    result = subprocess.run(["bash", str(SCRIPTS / "export_continuous_selector_history.sh"),
                             "--root", str(run[0]), "--output", str(tmp_path / "new output")],
        cwd=tmp_path, env={**os.environ, "HISTORY_EXPORT_PYTHON": sys.executable},
        text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "last logged step=200" in result.stdout
    assert "JSON:" in result.stdout
