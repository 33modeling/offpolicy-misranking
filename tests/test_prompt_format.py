"""Prompt-format families resolve per dataset everywhere the format is compared."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prompt_format as pf  # noqa: E402
from locate_uploaded_snapshot import discover  # noqa: E402
from model_matrix import PINNED_OFFICIAL_FILES, _load_specs  # noqa: E402
from regime_contract import expected_run_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_family_resolves_per_dataset_and_concrete_values_pass_through() -> None:
    assert pf.resolve_prompt_format("olmo_rlzero", "math500") == "olmo_rlzero_math"
    assert pf.resolve_prompt_format("olmo_rlzero", "mbpp") == "olmo_rlzero_code"
    assert pf.resolve_prompt_format("tokenizer_chat", "mbpp") == "tokenizer_chat"
    assert pf.resolve_prompt_format("verifiable_completion", "kk") == "verifiable_completion"
    with pytest.raises(ValueError):
        pf.resolve_prompt_format("olmo_rlzero", "kk")
    with pytest.raises(ValueError):
        pf.resolve_prompt_format("no_such_format", "math500")


def test_cli_prints_the_concrete_template(capsys: pytest.CaptureFixture[str]) -> None:
    assert pf.main(["olmo_rlzero", "mbpp"]) == 0
    assert capsys.readouterr().out.strip() == "olmo_rlzero_code"
    assert pf.main(["olmo_rlzero", "kk"]) == 1
    assert pf.main(["only-one-arg"]) == 2


def test_contract_expects_the_resolved_template_for_each_dataset() -> None:
    config = json.loads((ROOT / "configs/qwen35_9b_grpo.json").read_text())
    experiment = config["experiment"]
    matrix = {
        "git": "0" * 40,
        "experiment": experiment,
        "model": {
            "path": "/models/Qwen3.5-9B-Base",
            "config_sha256": "c" * 64,
            "tokenizer_config_sha256": "t" * 64,
            "generation_config_sha256": None,
            "snapshot_manifest_sha256": None,
            "prompt_format": "olmo_rlzero",
            "lora_targets": ["q_proj", "v_proj"],
        },
    }
    assert expected_run_config(matrix, "math500", 0, 0)["prompt_format"] == "olmo_rlzero_math"
    assert expected_run_config(matrix, "mbpp", 0, 25)["prompt_format"] == "olmo_rlzero_code"


def test_base_and_posttrained_uploads_are_told_apart_by_shard_sizes(tmp_path: Path) -> None:
    spec = next(iter(_load_specs(ROOT / "configs/qwen35_9b_grpo.json").values()))
    base_files = PINNED_OFFICIAL_FILES[(spec["repository"], spec["revision"])]
    post_files = PINNED_OFFICIAL_FILES[("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a")]
    # the two repositories ship the identical config.json, so use a fixture config
    # whose sha256 both specs accept and let the shard sizes decide
    text = json.dumps({"model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"]})
    text += " " * (base_files["config.json"]["size"] - len(text.encode()))
    import hashlib

    official = dict(base_files)
    official["config.json"] = {"size": len(text), "sha256": hashlib.sha256(text.encode()).hexdigest()}
    spec = {**spec, "official_files": official}
    models = tmp_path / "models"
    for name, table in (("Qwen3.5-9B", post_files), ("uploads/Qwen3.5-9B-Base", base_files)):
        directory = models / name
        directory.mkdir(parents=True)
        (directory / "config.json").write_text(text)
        for index in range(1, 5):
            shard = f"model.safetensors-0000{index}-of-00004.safetensors"
            with (directory / shard).open("wb") as stream:
                stream.truncate(table[shard]["size"])
    found, _ = discover(models, spec)
    assert found == (models / "uploads/Qwen3.5-9B-Base").resolve()
