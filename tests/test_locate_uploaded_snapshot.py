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
