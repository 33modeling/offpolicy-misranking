"""Pinned snapshot download patterns must cover every shard naming scheme in use."""
import fnmatch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model_matrix import DOWNLOAD_PATTERNS


def covered(name: str) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in DOWNLOAD_PATTERNS)


def test_shard_naming_schemes_are_downloaded():
    for name in (
        "model.safetensors",
        "model-00001-of-00003.safetensors",            # OLMo-3
        "model.safetensors-00001-of-00004.safetensors",  # Qwen3.5-9B / Qwen3.8
        "model.safetensors.index.json",
        "chat_template.jinja",
        "preprocessor_config.json",
    ):
        assert covered(name), name


def test_unrelated_large_files_are_not_downloaded():
    for name in ("pytorch_model.bin", "consolidated.00.pth", "training_args.bin"):
        assert not covered(name), name
