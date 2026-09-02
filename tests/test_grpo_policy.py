"""GRPO objective and fail-closed policy artifact tests."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from artifact_contract import sha256_file
from train_policy_grpo import (
    CHECKPOINT_SCHEMA,
    POLICY_SCHEMA,
    _checkpoint_step,
    _chunks,
    _latest_checkpoint,
    _response_logps,
    _response_logps_batch,
    _save_checkpoint,
    centered_group_advantages,
    clipped_grpo_loss,
    policy_update_for_objective,
    rloo_group_advantages,
    rloo_loss,
    standardized_group_advantages,
    validate_policy_lineage,
    validate_policy_manifest,
)


class ToyCausalLM(torch.nn.Module):
    def __init__(self, vocab_size: int = 17) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        assert attention_mask is not None
        return SimpleNamespace(logits=self.embedding(input_ids))


def test_batched_response_logps_match_single_variable_length_scoring() -> None:
    torch.manual_seed(7)
    model = ToyCausalLM()
    sequences = [
        torch.tensor([1, 2, 3, 4, 5]),
        torch.tensor([1, 2, 6, 7, 8, 9, 10]),
        torch.tensor([1, 11, 12, 13]),
    ]
    starts = [3, 3, 2]
    expected = [
        _response_logps(model, sequence, start)
        for sequence, start in zip(sequences, starts, strict=True)
    ]
    actual = _response_logps_batch(
        model, sequences, starts, pad_token_id=0
    )
    assert [value.shape for value in actual] == [value.shape for value in expected]
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right)


def test_micro_batch_partition_preserves_group_loss_and_gradients() -> None:
    advantages = torch.tensor([-1.0, -0.5, 0.5, 1.0])
    old = [torch.zeros(length) for length in (2, 3, 4, 5)]
    reference = [
        torch.full((length,), 0.05, requires_grad=True)
        for length in (2, 3, 4, 5)
    ]
    reference_loss, _ = clipped_grpo_loss(reference, old, advantages, 0.2)
    reference_loss.backward()
    reference_gradients = [value.grad.clone() for value in reference]

    chunked = [
        torch.full((length,), 0.05, requires_grad=True)
        for length in (2, 3, 4, 5)
    ]
    chunked_loss = torch.zeros(())
    for chunk in _chunks(len(chunked), 2):
        indices = list(chunk)
        loss, _ = clipped_grpo_loss(
            [chunked[index] for index in indices],
            [old[index] for index in indices],
            advantages[indices],
            0.2,
        )
        weighted = loss * len(indices) / len(chunked)
        weighted.backward()
        chunked_loss = chunked_loss + weighted.detach()

    assert float(chunked_loss) == pytest.approx(
        float(reference_loss.detach()), abs=1e-7
    )
    for actual, expected in zip(chunked, reference_gradients, strict=True):
        torch.testing.assert_close(actual.grad, expected)


def test_group_advantages_are_centered_and_zero_for_constant_rewards() -> None:
    advantages = standardized_group_advantages(torch.tensor([0.0, 1.0, 1.0, 0.0]))
    assert float(advantages.mean()) == pytest.approx(0.0, abs=1e-6)
    assert advantages.tolist()[0] < 0 < advantages.tolist()[1]
    assert standardized_group_advantages(torch.ones(8)).tolist() == [0.0] * 8
    dr = centered_group_advantages(torch.tensor([0.0, 1.0, 1.0, 0.0]))
    assert dr.tolist() == [-0.5, 0.5, 0.5, -0.5]
    rloo = rloo_group_advantages(torch.tensor([0.0, 1.0, 1.0, 0.0]))
    assert rloo.tolist() == pytest.approx([-2 / 3, 2 / 3, 2 / 3, -2 / 3])


def test_on_policy_grpo_has_zero_scalar_loss_but_nonzero_gradient() -> None:
    advantages = standardized_group_advantages(torch.tensor([0.0, 1.0, 1.0, 0.0]))
    old = [torch.zeros(3) for _ in range(4)]
    current = [torch.zeros(3, requires_grad=True) for _ in range(4)]

    loss, stats = clipped_grpo_loss(current, old, advantages, 0.2)
    assert float(loss.detach()) == pytest.approx(0.0, abs=1e-7)
    assert stats["mean_ratio"] == 1.0

    loss.backward()
    gradient_norm = torch.linalg.vector_norm(
        torch.cat([value.grad for value in current if value.grad is not None])
    )
    assert float(gradient_norm) > 0


def test_clipped_grpo_has_the_correct_sign_and_asymmetric_clip() -> None:
    positive = torch.tensor([math.log(2.0)], requires_grad=True)
    loss, stats = clipped_grpo_loss(
        [positive], [torch.zeros(1)], torch.tensor([1.0]), 0.2
    )
    assert float(loss.detach()) == pytest.approx(-1.2)
    assert stats["clip_fraction"] == 1.0
    loss.backward()
    assert float(positive.grad) == 0.0  # positive improvement is capped

    negative = torch.tensor([math.log(2.0)], requires_grad=True)
    loss, _ = clipped_grpo_loss(
        [negative], [torch.zeros(1)], torch.tensor([-1.0]), 0.2
    )
    assert float(loss.detach()) == pytest.approx(2.0)
    loss.backward()
    assert float(negative.grad.detach()) > 0  # gradient descent lowers bad-response log-probability


def test_dr_grpo_uses_a_fixed_token_budget_not_response_length() -> None:
    current = torch.zeros(2, requires_grad=True)
    loss, _ = clipped_grpo_loss(
        [current], [torch.zeros(2)], torch.tensor([1.0]), 0.2, token_normalizer=8
    )
    assert float(loss.detach()) == pytest.approx(-0.25)


def test_rloo_uses_sequence_log_probability_and_leave_one_out_baseline() -> None:
    logps = torch.tensor([-0.2, -0.3], requires_grad=True)
    loss = rloo_loss([logps], torch.tensor([2 / 3]))
    assert float(loss.detach()) == pytest.approx(1 / 3)
    loss.backward()
    assert logps.grad.tolist() == pytest.approx([-2 / 3, -2 / 3])


def _policy_artifact(
    path: Path,
    *,
    objective: str = "grpo",
    active: int = 1,
    start_step: int = 0,
    completed_steps: int = 25,
    parent: Path | None = None,
) -> None:
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text("{}\n")
    (path / "adapter_model.safetensors").write_bytes(b"adapter")
    (path / "optimizer.pt").write_bytes(b"optimizer")
    normalization = {
        "grpo": ("group_std", "response_length"),
        "dr_grpo": ("none", "fixed_generation_budget"),
        "rloo": ("leave_one_out", "sequence_sum"),
    }.get(objective, ("invalid", "invalid"))
    stats = [
        {
            "step": step,
            "training_objective": objective,
            "advantage_normalization": normalization[0],
            "token_normalization": normalization[1],
            "nonzero_advantage_groups": active if step == start_step + 1 else 0,
            "groups": 4,
            "samples": 32,
            "reward_mean": 0.5,
            "rank_reward_std_mean": 0.5,
            "loss": 0.1,
            "grad_norm": 0.2,
            "clip_fraction": 0.0,
            "mean_ratio": 1.0,
            "approx_kl": 0.0,
        }
        for step in range(start_step + 1, completed_steps + 1)
    ]
    (path / "grpo_stats.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in stats)
    )
    manifest = {
        "schema": POLICY_SCHEMA,
        "training_objective": objective,
        "policy_update": (
            policy_update_for_objective(objective)
            if objective in {"grpo", "dr_grpo", "rloo"}
            else "supervised_cross_entropy"
        ),
        "reward_source": "verifier",
        "reference_kl_beta": 0.0,
        "supervised_loss": False,
        "positive_only_filter": False,
        "parameterization": "lora",
        "advantage_normalization": normalization[0],
        "token_normalization": normalization[1],
        "completed_steps": completed_steps,
        "start_step": start_step,
        "world_size": 4,
        "config": {"group_size": 8},
        "adapter_sha256": sha256_file(path / "adapter_model.safetensors"),
        "parent_policy": str(parent.resolve()) if parent else None,
        "parent_policy_manifest_sha256": (
            sha256_file(parent / "policy_train.json") if parent else None
        ),
        "parent_adapter_sha256": (
            sha256_file(parent / "adapter_model.safetensors") if parent else None
        ),
        "parent_optimizer_sha256": (
            sha256_file(parent / "optimizer.pt") if parent else None
        ),
    }
    (path / "policy_train.json").write_text(json.dumps(manifest))


def test_policy_manifest_rejects_sft_and_records_no_reward_variation(tmp_path: Path) -> None:
    valid = tmp_path / "valid"
    _policy_artifact(valid)
    validate_policy_manifest(valid, target_steps=25, world_size=4)

    sft = tmp_path / "sft"
    _policy_artifact(sft, objective="sft")
    with pytest.raises(ValueError, match="training_objective"):
        validate_policy_manifest(sft, target_steps=25, world_size=4)

    no_signal = tmp_path / "no-signal"
    _policy_artifact(no_signal, active=0)
    manifest = validate_policy_manifest(no_signal, target_steps=25, world_size=4)
    assert manifest["training_objective"] == "grpo"

    dr_grpo = tmp_path / "dr-grpo"
    _policy_artifact(dr_grpo, objective="dr_grpo")
    validate_policy_manifest(
        dr_grpo, target_steps=25, world_size=4, training_objective="dr_grpo"
    )
    with pytest.raises(ValueError, match="training_objective"):
        validate_policy_manifest(dr_grpo, target_steps=25, world_size=4)

    rloo = tmp_path / "rloo"
    _policy_artifact(rloo, objective="rloo")
    validate_policy_manifest(
        rloo, target_steps=25, world_size=4, training_objective="rloo"
    )


def test_policy_lineage_requires_the_exact_previous_interval(tmp_path: Path) -> None:
    parent = tmp_path / "d25" / "policy_step_25"
    _policy_artifact(parent)
    child = tmp_path / "d100" / "policy_step_100"
    _policy_artifact(
        child,
        start_step=25,
        completed_steps=100,
        parent=parent,
    )
    validate_policy_lineage(
        child,
        target_steps=100,
        world_size=4,
        training_objective="grpo",
        expected_start_step=25,
        expected_parent=parent,
    )

    independent = tmp_path / "independent" / "policy_step_100"
    _policy_artifact(independent, completed_steps=100)
    with pytest.raises(ValueError, match="start_step"):
        validate_policy_lineage(
            independent,
            target_steps=100,
            world_size=4,
            training_objective="grpo",
            expected_start_step=25,
            expected_parent=parent,
        )

    wrong_parent = tmp_path / "wrong-parent" / "policy_step_25"
    _policy_artifact(wrong_parent)
    with pytest.raises(ValueError, match="parent_policy"):
        validate_policy_lineage(
            child,
            target_steps=100,
            world_size=4,
            training_objective="grpo",
            expected_start_step=25,
            expected_parent=wrong_parent,
        )

    (parent / "optimizer.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="parent_optimizer_sha256"):
        validate_policy_lineage(
            child,
            target_steps=100,
            world_size=4,
            training_objective="grpo",
            expected_start_step=25,
            expected_parent=parent,
        )


def test_policy_contract_binds_model_seed_prompts_optimizer_and_stats(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    prompts = tmp_path / "prompts.json"
    prompts.write_text(
        '{"train":[{"question":"q","answer":"a"}],"val":[]}\n'
    )
    policy = tmp_path / "policy"
    _policy_artifact(policy)
    config = {
        "group_size": 8,
        "clip_epsilon": 0.2,
        "learning_rate": 1e-5,
        "epochs_per_batch": 2,
        "max_grad_norm": 1.0,
        "advantage_epsilon": 1e-4,
        "lora_rank": 16,
        "lora_alpha": 32,
        "checkpoint_every": 5,
    }
    manifest_path = policy / "policy_train.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        {
            "base_model": str(model.resolve()),
            "seed": 7,
            "max_new_tokens": 512,
            "config": config,
            "samples_per_step": 32,
            "prompts_sha256": sha256_file(prompts),
            "optimizer_sha256": sha256_file(policy / "optimizer.pt"),
            "grpo_stats_sha256": sha256_file(policy / "grpo_stats.jsonl"),
        }
    )
    manifest_path.write_text(json.dumps(manifest))

    contract = {
        "target_steps": 25,
        "world_size": 4,
        "training_objective": "grpo",
        "expected_start_step": 0,
        "expected_parent": None,
        "expected_model": model,
        "expected_seed": 7,
        "expected_max_new_tokens": 512,
        "expected_config": config,
        "expected_prompts": prompts,
        "require_complete_hashes": True,
    }
    validate_policy_lineage(policy, **contract)
    with pytest.raises(ValueError, match="seed"):
        validate_policy_lineage(policy, **{**contract, "expected_seed": 8})

    (policy / "optimizer.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="optimizer.pt hash"):
        validate_policy_lineage(policy, **contract)


def test_latest_checkpoint_accepts_complete_target_and_ignores_partial(tmp_path: Path) -> None:
    contract = {
        "schema": CHECKPOINT_SCHEMA,
        "training_objective": "grpo",
        "seed": 7,
    }
    for step in (5, 10, 25):
        checkpoint = tmp_path / f"checkpoint-{step:06d}"
        checkpoint.mkdir()
        for name in (
            "adapter_config.json",
            "adapter_model.safetensors",
            "optimizer.pt",
            "grpo_stats.jsonl",
        ):
            (checkpoint / name).write_text("x")
        (checkpoint / "checkpoint_state.json").write_text(
            json.dumps({
                **contract,
                "completed_steps": step,
                "adapter_sha256": sha256_file(checkpoint / "adapter_model.safetensors"),
                "optimizer_sha256": sha256_file(checkpoint / "optimizer.pt"),
                "grpo_stats_sha256": sha256_file(checkpoint / "grpo_stats.jsonl"),
            })
        )
    (tmp_path / "checkpoint-000020").mkdir()
    path, step = _latest_checkpoint(tmp_path, 25, contract)
    assert step == 25
    assert path.name == "checkpoint-000025"

    path, step = _latest_checkpoint(tmp_path, 25, {**contract, "seed": 8})
    assert path is None
    assert step == 0

    (tmp_path / "checkpoint-000025/optimizer.pt").write_text("corrupt")
    path, step = _latest_checkpoint(tmp_path, 25, contract)
    assert step == 10
    assert path.name == "checkpoint-000010"

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
    (tmp_path / "grpo_stats.jsonl").write_text('{"step":5}\n')
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
