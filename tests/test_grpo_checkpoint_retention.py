"""Retention must never evict valid recovery points in favor of broken folders."""

import json

import pytest

pytest.importorskip("torch")

from artifact_contract import sha256_file
import train_policy_grpo as trainer
import train_selection_gate_grpo as budget_trainer


CONTRACT = {"schema": trainer.CHECKPOINT_SCHEMA, "training_objective": "grpo", "seed": 3}


class Model:
    @staticmethod
    def save_pretrained(path, *, safe_serialization):
        assert safe_serialization
        (path / "adapter_config.json").write_text('{}\n')
        (path / "adapter_model.safetensors").write_bytes(b"CPU fixture adapter")


class Optimizer:
    @staticmethod
    def state_dict():
        return {"state": {}, "param_groups": []}


def checkpoint(root, step, *, contract=None):
    path = root / f"checkpoint-{step:06d}"
    path.mkdir(parents=True)
    for name in ("adapter_config.json", "adapter_model.safetensors", "optimizer.pt"):
        (path / name).write_bytes(b"saved fixture payload")
    (path / "grpo_stats.jsonl").write_text(json.dumps({"step": step}) + "\n")
    record = {**(CONTRACT if contract is None else contract), "completed_steps": step,
              "adapter_sha256": sha256_file(path / "adapter_model.safetensors"),
              "optimizer_sha256": sha256_file(path / "optimizer.pt"),
              "grpo_stats_sha256": sha256_file(path / "grpo_stats.jsonl")}
    (path / "checkpoint_state.json").write_text(json.dumps(record) + "\n")
    return path


def save(root, step):
    (root / "grpo_stats.jsonl").write_text(json.dumps({"step": step}) + "\n")
    trainer._save_checkpoint(Model(), Optimizer(), root, step, 0, CONTRACT)
    return root / f"checkpoint-{step:06d}"


def contents(path):
    if path.is_file():
        return path.read_bytes()
    return {str(item.relative_to(path)): item.read_bytes()
            for item in path.rglob("*") if item.is_file()}


def test_normal_retention_keeps_two_latest_valid_checkpoints(tmp_path):
    first = save(tmp_path, 5)
    previous = save(tmp_path, 10)
    newest = save(tmp_path, 15)
    assert not first.exists()
    assert sorted(tmp_path.glob("checkpoint-*")) == [previous, newest]
    assert trainer._latest_checkpoint(tmp_path, 100, CONTRACT) == (newest, 15)


@pytest.mark.parametrize("damage", ["missing-state", "corrupt-hash", "foreign-contract"])
def test_new_checkpoint_is_not_deleted_to_keep_two_invalid_future_folders(tmp_path, damage):
    previous = checkpoint(tmp_path, 30)
    bad = []
    for step in (40, 45):
        path = checkpoint(tmp_path, step, contract={**CONTRACT, "seed": 99} if damage == "foreign-contract" else None)
        if damage == "missing-state":
            (path / "checkpoint_state.json").unlink()
        elif damage == "corrupt-hash":
            (path / "optimizer.pt").write_bytes(b"damaged optimizer")
        bad.append(path)
    before = {path: contents(path) for path in bad}
    assert budget_trainer._preserved_local_checkpoint(tmp_path, 100, CONTRACT) == (previous, 30)
    newest = save(tmp_path, 35)
    assert previous.is_dir()
    assert trainer._checkpoint_step(newest, CONTRACT) == 35
    assert budget_trainer._preserved_local_checkpoint(tmp_path, 100, CONTRACT) == (newest, 35)
    assert {path: contents(path) for path in bad} == before
    assert (newest / "grpo_stats.jsonl").read_bytes() == (tmp_path / "grpo_stats.jsonl").read_bytes()


@pytest.mark.parametrize("artifact", ["partial", "foreign", "file", "symlink", "noncanonical"])
def test_pruning_valid_predecessors_preserves_unrecognized_older_artifacts(tmp_path, artifact):
    if artifact == "partial":
        unexpected = tmp_path / "checkpoint-000001"
        unexpected.mkdir()
        (unexpected / "optimizer.pt").write_bytes(b"incomplete saved work")
    elif artifact == "foreign":
        unexpected = checkpoint(tmp_path, 1, contract={**CONTRACT, "seed": 99})
    elif artifact == "file":
        unexpected = tmp_path / "checkpoint-000001"
        unexpected.write_bytes(b"unexpected saved artifact")
    elif artifact == "symlink":
        target = checkpoint(tmp_path / "external", 1)
        unexpected = tmp_path / "checkpoint-000001"
        unexpected.symlink_to(target, target_is_directory=True)
    else:
        unexpected = checkpoint(tmp_path, 1)
        renamed = tmp_path / "checkpoint-000002"
        unexpected.rename(renamed)
        unexpected = renamed
    before = contents(unexpected)
    old = checkpoint(tmp_path, 20)
    previous = checkpoint(tmp_path, 30)
    newest = save(tmp_path, 35)
    assert not old.exists()
    assert previous.is_dir() and newest.is_dir()
    assert contents(unexpected) == before
    if artifact == "symlink":
        assert unexpected.is_symlink()


def test_valid_future_checkpoints_do_not_evict_newly_committed_checkpoint(tmp_path):
    previous = checkpoint(tmp_path, 30)
    future = [checkpoint(tmp_path, step) for step in (40, 45)]
    before = {path: contents(path) for path in future}
    newest = save(tmp_path, 35)
    assert previous.is_dir() and trainer._checkpoint_step(newest, CONTRACT) == 35
    assert {path: contents(path) for path in future} == before
    assert budget_trainer._preserved_local_checkpoint(tmp_path, 35, CONTRACT) == (newest, 35)


def test_failed_checkpoint_write_never_prunes_existing_recovery_points(tmp_path):
    checkpoint(tmp_path, 20)
    checkpoint(tmp_path, 30)
    before = {path: contents(path) for path in tmp_path.glob("checkpoint-*")}

    class FailedModel:
        @staticmethod
        def save_pretrained(*_args, **_kwargs):
            raise OSError("fixture interrupted save")

    with pytest.raises(OSError, match="interrupted save"):
        trainer._save_checkpoint(FailedModel(), Optimizer(), tmp_path, 35, 0, CONTRACT)
    assert {path: contents(path) for path in tmp_path.glob("checkpoint-*")} == before
