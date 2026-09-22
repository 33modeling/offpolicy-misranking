"""Exercise the actual checkpoint I/O functions without loading GPU libraries."""
import ast
import hashlib
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import selection_switch_curve_train as archive

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/train_policy_grpo.py"


def checkpoint_io():
    tree = ast.parse(SOURCE.read_text())
    names = {"_atomic_json", "_checkpoint_step", "_save_checkpoint", "_latest_checkpoint"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == len(names)
    namespace = dict(Path=Path, json=json, shutil=shutil,
                     sha256_file=lambda p: hashlib.sha256(p.read_bytes()).hexdigest(),
                     compact_adapter=lambda _: None,
                     torch=SimpleNamespace(save=lambda value, path: path.write_text(json.dumps(value))))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


class Model:
    def save_pretrained(self, path, **kwargs):
        (path / "adapter_model.safetensors").write_bytes(b"fixture weights")
        (path / "adapter_config.json").write_text('{"r":16}')


def save(io, root, step):
    with (root / "grpo_stats.jsonl").open("a") as handle:
        handle.write(json.dumps({"step": step}) + "\n")
    io["_save_checkpoint"](Model(), SimpleNamespace(state_dict=lambda: {"step": step}),
                           root, step, 0, {"seed": 0})
    return root / f"checkpoint-{step:06d}"


def test_every_published_checkpoint_survives_and_resume_uses_latest(tmp_path):
    io = checkpoint_io()
    published = [save(io, tmp_path, step) for step in range(5, 105, 5)]
    assert sorted(tmp_path.glob("checkpoint-*")) == published
    for step, path in zip(range(5, 105, 5), published):
        assert io["_checkpoint_step"](path, {"seed": 0}) == step
    assert io["_latest_checkpoint"](tmp_path, 100, {"seed": 0}) == (published[-1], 100)


@pytest.mark.parametrize("kind", ["directory", "file", "symlink"])
def test_existing_incompatible_checkpoint_is_never_deleted(tmp_path, kind):
    io = checkpoint_io()
    target = tmp_path / "checkpoint-000005"
    original = tmp_path / "original"
    original.write_bytes(b"preserve this evidence")
    if kind == "directory":
        target.mkdir()
        (target / "evidence").write_bytes(original.read_bytes())
    elif kind == "symlink":
        target.symlink_to(original)
    else:
        target.write_bytes(original.read_bytes())
    with pytest.raises(ValueError, match="preserve and review"):
        save(io, tmp_path, 5)
    assert original.read_bytes() == b"preserve this evidence"
    assert (target / "evidence" if kind == "directory" else target).read_bytes() == original.read_bytes()


def test_failed_save_and_valid_repeat_preserve_previous_files(tmp_path):
    io = checkpoint_io()
    first = save(io, tmp_path, 5)
    original = {p.name: p.read_bytes() for p in first.iterdir()}
    save(io, tmp_path, 5)
    assert {p.name: p.read_bytes() for p in first.iterdir()} == original

    class Failure(Model):
        def save_pretrained(self, *args, **kwargs):
            raise OSError("interrupted")

    with pytest.raises(OSError, match="interrupted"):
        io["_save_checkpoint"](Failure(), None, tmp_path, 10, 0, {"seed": 0})
    assert {p.name: p.read_bytes() for p in first.iterdir()} == original


@pytest.mark.parametrize("name", ["train_policy_grpo.py", "train_selection_gate_grpo.py", "train_mopps_grpo.py"])
def test_final_publication_does_not_delete_checkpoints(name):
    tree = ast.parse((ROOT / "src" / name).read_text())
    train = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "train")
    calls = [ast.unparse(node.func) for node in ast.walk(train) if isinstance(node, ast.Call)]
    assert "shutil.rmtree" not in calls
    assert any("save_pretrained" in call for call in calls)


def test_pair_publishes_curve_immediately_without_deleting_source(tmp_path, monkeypatch):
    io = checkpoint_io()
    original_writer = io["_atomic_json"]

    def with_receipt(path, value):
        if path.name == "checkpoint_state.json":
            original_writer(path.parent / "cost-receipt.json", {"step": value["completed_steps"]})
        original_writer(path, value)

    io["_atomic_json"] = with_receipt
    monkeypatch.setattr(archive, "KEEP", (*archive.KEEP, "cost-receipt.json"))
    io["_save_checkpoint"] = archive.archive_after_save(io["_save_checkpoint"])
    for step in (5, 10, 15):
        source = save(io, tmp_path, step)
        view = tmp_path / "curve-checkpoints" / f"step-{step}"
        assert source.is_dir() and (source / "optimizer.pt").is_file()
        for name in archive.KEEP:
            assert (view / name).read_bytes() == (source / name).read_bytes()
        archive.archive_then_remove(source)
        assert source.is_dir()


def test_install_wraps_both_trainer_bindings_once(monkeypatch):
    save_fn = checkpoint_io()["_save_checkpoint"]
    canonical = SimpleNamespace(_save_checkpoint=save_fn)
    budget = SimpleNamespace(_save_checkpoint=save_fn)
    monkeypatch.setitem(sys.modules, "train_policy_grpo", canonical)
    monkeypatch.setitem(sys.modules, "train_selection_gate_grpo", budget)
    monkeypatch.setattr(shutil, "rmtree", shutil.rmtree)
    archive.install()
    installed = canonical._save_checkpoint
    archive.install()
    assert budget._save_checkpoint is canonical._save_checkpoint is installed


def test_nonleader_does_not_publish_curve(tmp_path):
    io = checkpoint_io()
    wrapped = archive.archive_after_save(io["_save_checkpoint"])
    wrapped(None, None, tmp_path, 5, 1, {})
    assert list(tmp_path.iterdir()) == []
