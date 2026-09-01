"""Pinned model manifests bind every tokenizer and weight file by content."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model_matrix import (
    PINNED_OFFICIAL_FILES,
    _file_records,
    _load_config,
    _seal_local_snapshot,
    _verify_file_records,
    _weight_shards,
)


def test_generalization_configs_have_complete_fixed_protocols() -> None:
    root = Path(__file__).resolve().parents[1]
    expected = {
        "generalization_logic.json": "kk",
        "generalization_science.json": "arc-challenge",
        "generalization_knowledge.json": "mmlu-pro-nonmath",
    }
    for filename, dataset in expected.items():
        config = _load_config(root / "configs" / filename)
        experiment = config["experiment"]
        assert experiment["policy_method"] == "grpo"
        assert experiment["datasets"] == [dataset]
        assert experiment["n_train_by_dataset"] == {dataset: 512}
        assert experiment["seeds"] == [0, 1, 2]
        assert experiment["drifts"] == [0, 25, 100, 400]
        assert experiment["temperature"] == 1.0
        assert experiment["top_p"] == 1.0
        assert experiment["fresh_k"] // experiment["micro_group"] == 8
        assert experiment["n_val"] == 100
        assert experiment["first_bootstrap"] >= 10_000
        assert experiment["grpo"]["epochs_per_batch"] == 1


def test_weight_manifest_detects_content_corruption() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        root = Path(raw_tmp)
        (root / "model.safetensors").write_bytes(b"weights")
        (root / "config.json").write_text("{}\n", encoding="utf-8")
        shards = _weight_shards(root)
        records = _file_records(root, [root / "config.json", *shards])
        _verify_file_records(root, records)
        (root / "model.safetensors").write_bytes(b"damaged")
        try:
            _verify_file_records(root, records)
        except ValueError as exc:
            assert "mismatch" in str(exc)
        else:
            raise AssertionError("modified weight file passed the content manifest")


def test_weight_index_rejects_path_traversal() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        root = Path(raw_tmp)
        (root / "model.safetensors.index.json").write_text(
            '{"weight_map":{"x":"../outside.safetensors"}}\n', encoding="utf-8"
        )
        try:
            _weight_shards(root)
        except ValueError as exc:
            assert "unsafe" in str(exc)
        else:
            raise AssertionError("unsafe shard path was accepted")


def test_transfer_config_rejects_wrong_types_before_launch() -> None:
    source = Path(__file__).resolve().parents[1] / "configs/generalization_logic.json"
    with tempfile.TemporaryDirectory() as raw_tmp:
        root = Path(raw_tmp)
        config = json.loads(source.read_text(encoding="utf-8"))
        config["experiment"]["skip_hybrid"] = "yes"
        path = root / "bad.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        try:
            _load_config(path)
        except TypeError as exc:
            assert "skip_hybrid" in str(exc)
        else:
            raise AssertionError("wrongly typed experiment setting was accepted")


def test_transfer_config_requires_one_candidate_count_per_dataset() -> None:
    source = Path(__file__).resolve().parents[1] / "configs/generalization_logic.json"
    with tempfile.TemporaryDirectory() as raw_tmp:
        root = Path(raw_tmp)
        config = json.loads(source.read_text(encoding="utf-8"))
        del config["experiment"]["n_train_by_dataset"]["kk"]
        path = root / "bad.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        try:
            _load_config(path)
        except ValueError as exc:
            assert "cover exactly" in str(exc)
        else:
            raise AssertionError("incomplete per-dataset candidate counts were accepted")


def test_generalization_models_change_architecture_and_capacity() -> None:
    root = Path(__file__).resolve().parents[1]
    configs = [
        _load_config(root / "configs" / filename)
        for filename in (
            "generalization_logic.json",
            "generalization_science.json",
            "generalization_knowledge.json",
        )
    ]
    expected_models = {
        ("allenai/OLMoE-1B-7B-0125", "olmoe"),
        ("Qwen/Qwen2.5-14B", "qwen2"),
    }
    for config in configs:
        assert {(row["repository"], row["model_type"]) for row in config["models"]} == expected_models
        assert all(row["prompt_format"] == "verifiable_completion" for row in config["models"])


def test_local_hub_snapshot_can_only_be_sealed_from_exact_revision_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    revision = "a" * 40
    spec = {
        "key": "model",
        "repository": "owner/model",
        "revision": revision,
        "lora_targets": ["q_proj"],
    }
    (tmp_path / "config.json").write_text("{}\n")
    (tmp_path / "tokenizer_config.json").write_text("{}\n")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    metadata = tmp_path / ".cache/huggingface/download"
    metadata.mkdir(parents=True)
    for name in ("config.json", "tokenizer_config.json", "model.safetensors"):
        digest = hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        (metadata / f"{name}.metadata").write_text(f"{revision}\n{digest}\n0\n")

    monkeypatch.setattr(
        "model_matrix._check_snapshot",
        lambda model_spec, path: {"revision": model_spec["revision"], "path": str(path)},
    )
    result = _seal_local_snapshot(spec, tmp_path)
    assert result["revision"] == revision
    manifest = json.loads((tmp_path / ".om_snapshot.json").read_text())
    assert manifest["revision"] == revision
    assert set(manifest["files"]) == {
        "config.json",
        "model.safetensors",
        "tokenizer_config.json",
    }

    (tmp_path / ".om_snapshot.json").unlink()
    digest = hashlib.sha256((tmp_path / "model.safetensors").read_bytes()).hexdigest()
    (metadata / "model.safetensors.metadata").write_text(
        f"{'b' * 40}\n{digest}\n0\n"
    )
    with pytest.raises(ValueError, match="revision mismatch"):
        _seal_local_snapshot(spec, tmp_path)


def test_uploaded_snapshot_seals_without_hub_metadata(tmp_path: Path, monkeypatch) -> None:
    files = {
        "config.json": b"{}\n",
        "tokenizer_config.json": b"{}\n",
        "model.safetensors": b"weights",
    }
    for name, content in files.items():
        (tmp_path / name).write_bytes(content)
    spec = {
        "key": "model",
        "repository": "owner/model",
        "revision": "a" * 40,
        "lora_targets": ["q_proj"],
    }
    monkeypatch.setitem(
        PINNED_OFFICIAL_FILES,
        (spec["repository"], spec["revision"]),
        {
            name: {
                "size": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            for name, content in files.items()
        },
    )
    monkeypatch.setattr(
        "model_matrix._check_snapshot",
        lambda model_spec, path: {"revision": model_spec["revision"], "path": str(path)},
    )

    result = _seal_local_snapshot(spec, tmp_path)

    assert result["revision"] == spec["revision"]
    assert (tmp_path / ".om_snapshot.json").is_file()
    assert not (tmp_path / ".cache").exists()

    (tmp_path / ".om_snapshot.json").unlink()
    (tmp_path / "model.safetensors").write_bytes(b"damaged")
    with pytest.raises(ValueError, match="hash mismatch"):
        _seal_local_snapshot(spec, tmp_path)
