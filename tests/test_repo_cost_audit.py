"""Missing cost boundaries must not be charged to a neighboring stage."""

import os
from datetime import UTC, datetime, timedelta

import pytest

import cost_accounting as costs


def test_progress_gap_does_not_merge_different_stages():
    start = datetime(2026, 10, 6, tzinfo=UTC)
    events = [{"time": start + timedelta(hours=i), "stage": s, "total": 8, "label": str(s)}
              for i, s in enumerate((1, 2, 4, 5, 6, 7, 8))]
    result = costs.stage_durations(events)
    assert "behavior_rollout" not in result["stages"]
    assert result["stages"]["fresh_rollout"] == 3600


def test_clock_reversal_is_unknown_not_negative_compute():
    start = datetime(2026, 10, 6, tzinfo=UTC)
    result = costs.stage_durations([
        {"time": start, "stage": 1, "total": 8, "label": "prep"},
        {"time": start - timedelta(seconds=1), "stage": 2, "total": 8, "label": "behavior"},
    ])
    assert "prep" not in result["stages"]


def artifact(root, name, seconds):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("test")
    os.utime(path, (1_700_000_000 + seconds, 1_700_000_000 + seconds))


def test_artifact_gap_does_not_absorb_missing_stage(tmp_path):
    artifact(tmp_path, "prompts.json", 0)
    artifact(tmp_path, "rollouts_behavior_train.jsonl", 1800)
    artifact(tmp_path, "rollouts_fresh_train.jsonl", 9000)
    result = costs.stage_durations_from_artifacts(tmp_path, 400)
    assert result["stages"]["behavior_rollout"] == 1800
    assert "fresh_rollout" not in result["stages"]


def test_joint_gradient_stage_requires_both_end_artifacts(tmp_path):
    artifact(tmp_path, "rollouts_fresh_train.jsonl", 0)
    artifact(tmp_path, "oracle_micro_groups.pt", 1800)
    artifact(tmp_path, "scores_offpolicy.json", 3600)
    result = costs.stage_durations_from_artifacts(tmp_path, 400)
    assert "oracle_val_gradients" not in result["stages"]
    assert "offpolicy_scores" not in result["stages"]
    artifact(tmp_path, "val_gradient.pt", 2400)
    result = costs.stage_durations_from_artifacts(tmp_path, 400)
    assert result["stages"]["oracle_val_gradients"] == 2400
    assert result["stages"]["offpolicy_scores"] == 1200


def test_d0_has_no_training_stage_to_require(tmp_path):
    artifact(tmp_path, "rollouts_behavior_train.jsonl", 1800)
    artifact(tmp_path, "rollouts_fresh_train.jsonl", 9000)
    result = costs.stage_durations_from_artifacts(tmp_path, 0)
    assert result["stages"]["fresh_rollout"] == 7200


@pytest.mark.parametrize("seconds", [-1, float("nan"), float("inf"), True])
def test_invalid_update_timer_is_rejected(tmp_path, seconds):
    import json
    stats = tmp_path / "grpo_stats.jsonl"
    stats.write_text(json.dumps({"step": 1, "step_seconds": seconds}) + "\n")
    with pytest.raises(ValueError, match="timer"):
        costs.step_seconds(stats)
