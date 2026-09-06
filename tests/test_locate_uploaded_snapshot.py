"""A hand-uploaded Qwen3.5-9B directory is found by content and exposed at the pinned path."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from locate_uploaded_snapshot import discover, link_missing_shards
from model_matrix import PINNED_OFFICIAL_FILES, _load_specs

ROOT = Path(__file__).resolve().parents[1]
SPEC = next(iter(_load_specs(ROOT / "configs/qwen35_9b_grpo.json").values()))
OFFICIAL = PINNED_OFFICIAL_FILES[(SPEC["repository"], SPEC["revision"])]


def fake_upload(directory: Path, shard_prefix: str) -> None:
    directory.mkdir(parents=True)
    config = {"model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"]}
    text = json.dumps(config)
    # pad to the official size so content identification succeeds
    text += " " * (OFFICIAL["config.json"]["size"] - len(text.encode()))
    (directory / "config.json").write_text(text)
    for index in range(1, 5):
        name = f"{shard_prefix}-0000{index}-of-00004.safetensors"
        size = OFFICIAL[f"model.safetensors-0000{index}-of-00004.safetensors"]["size"]
        with (directory / name).open("wb") as stream:
            stream.truncate(size)  # sparse file of the official size


def test_nonstandard_directory_and_shard_names_are_linked(tmp_path: Path) -> None:
    models = tmp_path / "models"
    fake_upload(models / "Qwen3.5-9B", "model")          # wrong dir name, wrong shard names
    (models / "other").mkdir()
    (models / "other/config.json").write_text(json.dumps({"model_type": "olmo3"}))
    found, _ = discover(models, SPEC)
    assert found == (models / "Qwen3.5-9B").resolve()
    standard = models / SPEC["local_directory"]
    os.symlink(found, standard)
    actions = link_missing_shards(standard.resolve(), OFFICIAL)
    assert all(action.startswith("linked") for action in actions), actions
    for index in range(1, 5):
        assert (standard / f"model.safetensors-0000{index}-of-00004.safetensors").is_file()


def test_standard_directory_wins_and_ambiguity_fails(tmp_path: Path) -> None:
    models = tmp_path / "models"
    fake_upload(models / SPEC["local_directory"], "model.safetensors")
    found, _ = discover(models, SPEC)
    assert found == models / SPEC["local_directory"]
    models2 = tmp_path / "models2"
    fake_upload(models2 / "a", "model")
    fake_upload(models2 / "b", "model")
    found, scanned = discover(models2, SPEC)
    assert found is None and len(scanned) == 2


def test_size_mismatch_is_reported_not_linked(tmp_path: Path) -> None:
    models = tmp_path / "models"
    fake_upload(models / "x", "model")
    bad = models / "x/model-00002-of-00004.safetensors"
    with bad.open("wb") as stream:
        stream.truncate(123)
    actions = link_missing_shards(models / "x", OFFICIAL)
    assert any(action.startswith("missing model.safetensors-00002") for action in actions)


def test_two_qwen_uploads_are_told_apart_by_folder_name(tmp_path: Path) -> None:
    models = tmp_path / "models"
    fake_upload(models / "Qwen3.5-9B", "model.safetensors")
    fake_upload(models / "Qwen3.8-27B", "model")   # same model_type, different repo
    found, _ = discover(models, SPEC)
    assert found == (models / "Qwen3.5-9B").resolve()


def _write_safetensors(path: Path, tensors: list[str]) -> None:
    import json as _json, struct
    header = {name: {"dtype": "F32", "shape": [1], "data_offsets": [4 * i, 4 * i + 4]}
              for i, name in enumerate(tensors)}
    blob = _json.dumps(header).encode()
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(blob)) + blob + b"\0" * (4 * len(tensors)))


def test_index_is_rebuilt_from_actual_shard_headers(tmp_path: Path) -> None:
    import json as _json
    from locate_uploaded_snapshot import ensure_index
    d = tmp_path / "m"; d.mkdir()
    _write_safetensors(d / "part-a.safetensors", ["model.layers.0.w", "model.layers.1.w"])
    _write_safetensors(d / "part-b.safetensors", ["lm_head.weight"])
    (d / "model.safetensors.index.json").write_text(_json.dumps(
        {"weight_map": {"model.layers.0.w": "model.safetensors-00001-of-00002.safetensors"}}))
    actions = ensure_index(d)
    assert any(a.startswith("rebuilt") for a in actions), actions
    index = _json.loads((d / "model.safetensors.index.json").read_text())
    assert index["weight_map"] == {"model.layers.0.w": "part-a.safetensors",
                                   "model.layers.1.w": "part-a.safetensors",
                                   "lm_head.weight": "part-b.safetensors"}
    assert (d / "model.safetensors.index.json.orig").exists()
    assert ensure_index(d) == []  # consistent now: untouched


def test_weightless_pinned_dir_does_not_shadow_the_real_upload(tmp_path: Path) -> None:
    models = tmp_path / "models"
    fake_upload(models / "Qwen3.5-9B", "model.safetensors")
    for other in ("Qwen3.5-2B", "Qwen3.5-4B", "Qwen3.6-27B", "Qwen3.8-27B-BF16"):
        fake_upload(models / other, "model")
    pinned = models / SPEC["local_directory"]
    pinned.mkdir()
    (pinned / "config.json").write_text('{"model_type": "qwen3_5"}')  # old prepare: no weights
    found, _ = discover(models, SPEC)
    assert found == (models / "Qwen3.5-9B").resolve()
    assert not pinned.exists()
    assert list(models.glob(".stale-Qwen3.5-9B-pinned-*"))


def test_snapshot_nested_at_any_depth_is_found(tmp_path: Path) -> None:
    models = tmp_path / "models"
    deep = models / "uploads/2026-09/Qwen3.5-9B/snapshots/abc123"
    fake_upload(deep, "model")
    (models / ".downloads").mkdir(parents=True)
    fake_upload(models / ".downloads/Qwen3.5-9B", "model")  # must be ignored
    found, _ = discover(models, SPEC)
    assert found == deep.resolve()


def test_model_outside_the_model_roots_is_found_via_volume_scan(tmp_path, monkeypatch) -> None:
    volume = tmp_path / "vol"
    models = volume / "models"          # configured root, empty
    models.mkdir(parents=True)
    elsewhere = volume / "SR-Coredata/models/Qwen3.5-9B"
    fake_upload(elsewhere, "model")
    monkeypatch.setenv("GROUP_VOLUME", str(volume))
    monkeypatch.delenv("OM_WORK", raising=False)
    monkeypatch.delenv("OM_USER", raising=False)
    found, _ = discover(models, SPEC)
    assert found == elsewhere.resolve()
