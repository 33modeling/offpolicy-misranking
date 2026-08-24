"""Pinned model manifests bind every tokenizer and weight file by content."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model_matrix import (
    _file_records,
    _load_config,
    _verify_file_records,
    _weight_shards,
)


def test_transfer_config_has_a_complete_fixed_protocol() -> None:
    root = Path(__file__).resolve().parents[1]
    config = _load_config(root / "configs/domain_transfer.json")
    experiment = config["experiment"]
    assert experiment["drifts"][0] == 0
    assert experiment["temperature"] == 1.0
    assert experiment["top_p"] == 1.0
    assert experiment["fresh_k"] // experiment["micro_group"] % 2 == 0


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
    source = Path(__file__).resolve().parents[1] / "configs/domain_transfer.json"
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
