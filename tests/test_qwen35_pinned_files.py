"""An uploaded Qwen3.5-9B snapshot must be sealable offline from pinned hashes."""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model_matrix import PINNED_OFFICIAL_FILES, _load_specs

ROOT = Path(__file__).resolve().parents[1]
SHARDS = [f"model.safetensors-0000{i}-of-00004.safetensors" for i in range(1, 5)]
# Everything _manifest_files() would include for this repository layout.
MANIFEST_NAMES = SHARDS + [
    "config.json", "tokenizer_config.json", "model.safetensors.index.json",
    "tokenizer.json", "merges.txt", "vocab.json", "chat_template.jinja",
]
BASE = ("Qwen/Qwen3.5-9B-Base", "68c46c4b3498877f3ef123c856ecfde50c39f404")


def _well_formed(files: dict) -> None:
    for name, record in files.items():
        assert set(record) == {"size", "sha256"} or set(record) == {"size", "git_blob_sha1"}, name
        assert record["size"] > 0
        digest = record.get("sha256") or record.get("git_blob_sha1")
        assert re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", digest), name
    for shard in SHARDS:
        assert "sha256" in files[shard]
    assert sum(files[s]["size"] for s in SHARDS) > 19_000_000_000


def test_qwen35_spec_has_pinned_official_files():
    spec = next(iter(_load_specs(ROOT / "configs/qwen35_9b_grpo.json").values()))
    assert spec["repository"] == "Qwen/Qwen3.5-9B"
    files = PINNED_OFFICIAL_FILES[(spec["repository"], spec["revision"])]
    for name in MANIFEST_NAMES:
        assert name in files, name
    _well_formed(files)


def test_qwen35_base_table_is_registered_and_distinct_from_posttrained():
    """The pretrained base is available as a one-line config switch; it shares
    config.json with the post-trained release but not the weights."""
    base = PINNED_OFFICIAL_FILES[BASE]
    post = PINNED_OFFICIAL_FILES[("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a")]
    assert "chat_template.jinja" not in base
    for name in [n for n in MANIFEST_NAMES if n != "chat_template.jinja"]:
        assert name in base, name
    _well_formed(base)
    assert base["config.json"] == post["config.json"]
    assert {base[s]["sha256"] for s in SHARDS}.isdisjoint({post[s]["sha256"] for s in SHARDS})


def test_qwen38_spec_has_pinned_official_files():
    spec = next(iter(_load_specs(ROOT / "configs/qwen38_27b_grpo.json").values()))
    files = PINNED_OFFICIAL_FILES[(spec["repository"], spec["revision"])]
    shards = [f"model-{i:05d}-of-00018.safetensors" for i in range(1, 19)]
    for name in shards + ["config.json", "tokenizer_config.json", "generation_config.json",
                          "model.safetensors.index.json", "tokenizer.json", "merges.txt",
                          "vocab.json", "chat_template.jinja"]:
        assert name in files, name
    assert files["config.json"]["size"] != PINNED_OFFICIAL_FILES[
        ("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a")]["config.json"]["size"]
