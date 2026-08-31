"""GRPO objective and fail-closed policy artifact tests."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from artifact_contract import sha256_file
from train_policy_grpo import (
    CHECKPOINT_SCHEMA,
    POLICY_SCHEMA,
    _checkpoint_step,
    _latest_checkpoint,
    _save_checkpoint,
    clipped_grpo_loss,
    standardized_group_advantages,
    validate_policy_manifest,
)


def test_group_advantages_are_centered_and_zero_for_constant_rewards() -> None:
    advantages = standardized_group_advantages(torch.tensor([0.0, 1.0, 1.0, 0.0]))
    assert float(advantages.mean()) == pytest.approx(0.0, abs=1e-6)
    assert advantages.tolist()[0] < 0 < advantages.tolist()[1]
    assert standardized_group_advantages(torch.ones(8)).tolist() == [0.0] * 8


def test_clipped_grpo_has_the_correct_sign_and_asymmetric_clip() -> None:
    positive = torch.tensor([math.log(2.0)], requires_grad=True)
    loss, stats = clipped_grpo_loss(
        [positive], [torch.zeros(1)], torch.tensor([1.0]), 0.2
    )
    assert float(loss) == pytest.approx(-1.2)
    assert stats["clip_fraction"] == 1.0
    loss.backward()
    assert float(positive.grad) == 0.0  # positive improvement is capped

    negative = torch.tensor([math.log(2.0)], requires_grad=True)
    loss, _ = clipped_grpo_loss(
        [negative], [torch.zeros(1)], torch.tensor([-1.0]), 0.2
    )
    assert float(loss) == pytest.approx(2.0)
    loss.backward()
    assert float(negative.grad) > 0  # gradient descent lowers bad-response log-probability


def _policy_artifact(path: Path, *, objective: str = "grpo", active: int = 1) -> None:
    path.mkdir()
    (path / "adapter_config.json").write_text("{}\n")
    (path / "adapter_model.safetensors").write_bytes(b"adapter")
    (path / "optimizer.pt").write_bytes(b"optimizer")
    (path / "grpo_stats.jsonl").write_text(
        json.dumps({"step": 25, "nonzero_advantage_groups": active}) + "\n"
    )
    manifest = {
        "schema": POLICY_SCHEMA,
        "training_objective": objective,
        "policy_update": "clipped_policy_gradient",
        "reward_source": "verifier",
        "reference_kl_beta": 0.0,
        "supervised_loss": False,
        "positive_only_filter": False,
        "completed_steps": 25,
        "start_step": 0,
        "world_size": 4,
        "adapter_sha256": sha256_file(path / "adapter_model.safetensors"),
        "parent_policy": None,
        "parent_policy_manifest_sha256": None,
        "parent_adapter_sha256": None,
        "parent_optimizer_sha256": None,
    }
    (path / "policy_train.json").write_text(json.dumps(manifest))


def test_policy_manifest_rejects_sft_and_no_reward_variation(tmp_path: Path) -> None:
    valid = tmp_path / "valid"
    _policy_artifact(valid)
    validate_policy_manifest(valid, target_steps=25, world_size=4)

    sft = tmp_path / "sft"
    _policy_artifact(sft, objective="sft")
    with pytest.raises(ValueError, match="training_objective"):
        validate_policy_manifest(sft, target_steps=25, world_size=4)

    no_signal = tmp_path / "no-signal"
    _policy_artifact(no_signal, active=0)
    with pytest.raises(ValueError, match="no nonzero-advantage"):
        validate_policy_manifest(no_signal, target_steps=25, world_size=4)


def test_latest_checkpoint_ignores_partial_and_target_checkpoint(tmp_path: Path) -> None:
    contract = {
        "schema": CHECKPOINT_SCHEMA,
        "training_objective": "grpo",
        "seed": 7,
    }
    for step in (5, 10, 25):
        checkpoint = tmp_path / f"checkpoint-{step:06d}"
        checkpoint.mkdir()
        for name in ("adapter_config.json", "adapter_model.safetensors", "optimizer.pt"):
            (checkpoint / name).write_text("x")
        (checkpoint / "checkpoint_state.json").write_text(
            json.dumps({
                **contract,
                "completed_steps": step,
                "adapter_sha256": sha256_file(checkpoint / "adapter_model.safetensors"),
                "optimizer_sha256": sha256_file(checkpoint / "optimizer.pt"),
            })
        )
    (tmp_path / "checkpoint-000020").mkdir()
    path, step = _latest_checkpoint(tmp_path, 25, contract)
    assert step == 10
    assert path.name == "checkpoint-000010"

    path, step = _latest_checkpoint(tmp_path, 25, {**contract, "seed": 8})
    assert path is None
    assert step == 0

    (tmp_path / "checkpoint-000010/optimizer.pt").write_text("corrupt")
    path, step = _latest_checkpoint(tmp_path, 25, contract)
    assert step == 5
    assert path.name == "checkpoint-000005"


def test_checkpoint_publish_replaces_an_invalid_same_step_directory(tmp_path: Path) -> None:
    class Model:
        @staticmethod
        def save_pretrained(path: Path, *, safe_serialization: bool) -> None:
            assert safe_serialization
            (path / "adapter_config.json").write_text("{}\n")
            (path / "adapter_model.safetensors").write_bytes(b"valid-adapter")

    class Optimizer:
        @staticmethod
        def state_dict() -> dict:
            return {"state": {}, "param_groups": []}

    contract = {
        "schema": CHECKPOINT_SCHEMA,
        "training_objective": "grpo",
        "seed": 7,
    }
    stale = tmp_path / "checkpoint-000005"
    stale.mkdir()
    (stale / "checkpoint_state.json").write_text("{}\n")

    _save_checkpoint(Model(), Optimizer(), tmp_path, 5, 0, contract)

    assert _checkpoint_step(stale, contract) == 5
    assert (stale / "adapter_model.safetensors").read_bytes() == b"valid-adapter"


def test_canonical_path_contains_no_supervised_drift() -> None:
    experiment = (ROOT / "src/experiment.py").read_text()
    rollout = (ROOT / "src/rollout.py").read_text()
    point = (ROOT / "scripts/run_point.sh").read_text()
    assert "train_drift_lora" not in experiment + rollout
    assert 'choices=["prep", "rollout-behavior", "drift"' not in experiment
    assert "torch.distributed.run" in point
    assert "train_policy_grpo.py" in point
