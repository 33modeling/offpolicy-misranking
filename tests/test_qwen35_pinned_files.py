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


def test_qwen35_spec_has_pinned_official_files():
    spec = next(iter(_load_specs(ROOT / "configs/qwen35_9b_grpo.json").values()))
    files = PINNED_OFFICIAL_FILES[(spec["repository"], spec["revision"])]
    for name in MANIFEST_NAMES:
        assert name in files, name
    for name, record in files.items():
        assert set(record) == {"size", "sha256"} or set(record) == {"size", "git_blob_sha1"}, name
        assert record["size"] > 0
        digest = record.get("sha256") or record.get("git_blob_sha1")
        assert re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", digest), name
    for shard in SHARDS:
        assert "sha256" in files[shard]
    assert sum(files[s]["size"] for s in SHARDS) > 19_000_000_000
