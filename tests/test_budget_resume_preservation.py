"""A rejected local checkpoint must never become a silent parent restart."""

import argparse
import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

import train_selection_gate_grpo as driver

CONTRACT = {"seed": 3, "start_step": 25, "config": {"group_size": 4}}


def json_file(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def checkpoint(out, step=30, contract=None):
    directory = out / f"checkpoint-{step:06d}"
    directory.mkdir(parents=True)
    json_file(directory / "adapter_config.json", {"peft_type": "LORA"})
    (directory / "adapter_model.safetensors").write_bytes(b"fixture adapter")
    (directory / "optimizer.pt").write_bytes(b"fixture optimizer")
    json_file(directory / "grpo_stats.jsonl", {"step": step})
    state = {**(CONTRACT if contract is None else contract), "completed_steps": step}
    for key, name in (("adapter_sha256", "adapter_model.safetensors"),
                      ("optimizer_sha256", "optimizer.pt"),
                      ("grpo_stats_sha256", "grpo_stats.jsonl")):
        state[key] = hashlib.sha256((directory / name).read_bytes()).hexdigest()
    json_file(directory / "checkpoint_state.json", state)
    return directory


def snapshot(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*")
            if path.is_file() and not path.is_symlink()}


def test_fresh_directory_can_start_from_parent(tmp_path):
    assert driver._preserved_local_checkpoint(tmp_path, 100, CONTRACT) == (None, 0)
    (tmp_path / "worker.log").write_text("initializing\n")
    (tmp_path / "grpo_stats.jsonl").write_bytes(b"")
    assert driver._preserved_local_checkpoint(tmp_path, 100, CONTRACT) == (None, 0)


@pytest.mark.parametrize("name", ["optimizer.pt", "adapter_model.safetensors", "adapter_config.json",
                                  "grpo_stats.jsonl", "checkpoint_state.json", "policy_train.json"])
def test_local_training_evidence_blocks_parent_restart_without_checkpoint(tmp_path, name):
    (tmp_path / name).write_bytes(b"already saved training evidence\n")
    before = snapshot(tmp_path)
    with pytest.raises(ValueError):
        driver._preserved_local_checkpoint(tmp_path, 100, CONTRACT)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("name", ["checkpoint-000030", ".checkpoint-000030.tmp", "curve-checkpoints/step-30"])
def test_unfinished_checkpoint_directory_blocks_parent_restart(tmp_path, name):
    (tmp_path / name).mkdir(parents=True)
    with pytest.raises(ValueError):
        driver._preserved_local_checkpoint(tmp_path, 100, CONTRACT)
    assert (tmp_path / name).is_dir()


@pytest.mark.parametrize("name", ["optimizer.pt", "adapter_model.safetensors", "checkpoint-000030"])
def test_broken_saved_artifact_link_is_not_a_fresh_start(tmp_path, name):
    target = tmp_path / name
    target.symlink_to(tmp_path / "missing-external-artifact")
    with pytest.raises(ValueError):
        driver._preserved_local_checkpoint(tmp_path, 100, CONTRACT)
    assert target.is_symlink()


@pytest.mark.parametrize("failure", ["contract", "hash", "missing-optimizer", "future-step", "bad-state"])
def test_invalid_local_checkpoint_is_preserved_and_blocks_fallback(tmp_path, failure):
    directory = checkpoint(tmp_path, step=105 if failure == "future-step" else 30,
                           contract={**CONTRACT, "seed": 99} if failure == "contract" else None)
    if failure == "hash":
        (directory / "adapter_model.safetensors").write_bytes(b"damaged model")
    elif failure == "missing-optimizer":
        (directory / "optimizer.pt").unlink()
    elif failure == "bad-state":
        (directory / "checkpoint_state.json").write_text("{interrupted")
    before = snapshot(tmp_path)
    with pytest.raises(ValueError):
        driver._preserved_local_checkpoint(tmp_path, 100, CONTRACT)
    assert snapshot(tmp_path) == before


def test_valid_local_checkpoint_is_reused_without_changes(tmp_path):
    directory = checkpoint(tmp_path)
    json_file(tmp_path / "grpo_stats.jsonl", {"step": 33})
    before = snapshot(tmp_path)
    assert driver._preserved_local_checkpoint(tmp_path, 100, CONTRACT) == (directory, 30)
    assert snapshot(tmp_path) == before


def test_valid_older_checkpoint_survives_corrupt_newer_checkpoint(tmp_path):
    older = checkpoint(tmp_path, step=30)
    newer = checkpoint(tmp_path, step=35)
    (newer / "optimizer.pt").unlink()
    before = snapshot(tmp_path)
    assert driver._preserved_local_checkpoint(tmp_path, 100, CONTRACT) == (older, 30)
    assert snapshot(tmp_path) == before


def test_parent_only_budget_stop_is_not_local_training(tmp_path):
    json_file(tmp_path / "budget_stop.json", {"completed_steps": 25, "use_parent_policy": True,
                                             "stop_reason": "no_block_fits"})
    assert driver._preserved_local_checkpoint(tmp_path, 100, CONTRACT) == (None, 0)


def test_local_policy_budget_stop_cannot_be_restarted_from_parent(tmp_path):
    json_file(tmp_path / "budget_stop.json", {"completed_steps": 30, "use_parent_policy": False})
    with pytest.raises(ValueError):
        driver._preserved_local_checkpoint(tmp_path, 100, CONTRACT)


def test_stats_permission_error_fails_closed_without_any_rewrite(tmp_path, monkeypatch):
    stats = tmp_path / "grpo_stats.jsonl"
    stats.write_text('{"step":30}\n')
    original_stat = Path.stat

    def protected_stat(path, *args, **kwargs):
        if path == stats:
            raise PermissionError("fixture denied training stats")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", protected_stat)
    with pytest.raises((OSError, ValueError)):
        driver._preserved_local_checkpoint(tmp_path, 100, CONTRACT)


def test_train_runs_preservation_guard_before_loading_model_or_rewriting_stats(tmp_path, monkeypatch):
    pytest.importorskip("peft")
    output = tmp_path / "out"
    json_file(output / "grpo_stats.jsonl", {"step": 30})
    before = snapshot(output)
    monkeypatch.setattr(driver, "_distributed_setup", lambda *_: (0, 0, 1))
    monkeypatch.setattr(driver, "_checkpoint_contract", lambda *_: CONTRACT)
    monkeypatch.setattr(driver, "load_model", lambda *a, **k: pytest.fail("model loaded before preservation guard"))
    args = argparse.Namespace(
        output=str(output), model=str(tmp_path / "model"), prompts=str(tmp_path / "prompts.json"),
        wall_budget_deadline=None, budget_save_reserve=30., expected_world_size=1,
        objective="grpo", group_size=4, clip_epsilon=.2, learning_rate=1e-5,
        epochs_per_batch=1, max_grad_norm=1., advantage_epsilon=1e-4,
        lora_rank=16, lora_alpha=32, checkpoint_every=5, logprob_micro_batch=1,
        target_steps=100, start_step=25, resume_adapter=str(tmp_path / "parent"),
        resume_optimizer=str(tmp_path / "parent/optimizer.pt"), max_new_tokens=64,
        seed=3, disable_gradient_checkpointing=True,
    )
    with pytest.raises(ValueError):
        driver.train(args)
    assert snapshot(output) == before
