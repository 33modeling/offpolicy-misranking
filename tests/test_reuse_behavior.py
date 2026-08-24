"""Behavior-rollout reuse must preserve the complete generation contract."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from artifact_contract import sha256_file
from reuse_behavior import reuse


def make_run(root: Path, name: str, drift: int) -> Path:
    run = root / name
    run.mkdir()
    config = {
        "model": "/models/test-model",
        "model_resolved": "/models/test-model",
        "model_config_sha256": "model-hash",
        "tokenizer_config_sha256": "tokenizer-hash",
        "generation_config_sha256": "generation-hash",
        "dataset": "gsm8k",
        "n_train": 2,
        "behavior_k": 2,
        "fresh_k": 4,
        "val_k": 2,
        "max_new_tokens": 16,
        "temperature": 1.0,
        "top_p": 1.0,
        "thinking": "off",
        "drift": drift,
    }
    (run / "run_config.json").write_text(json.dumps(config))
    (run / "prompts.json").write_text(
        json.dumps(
            {
                "train": [{"question": "a"}, {"question": "b"}],
                "val": [{"question": "v"}],
            },
            sort_keys=True,
        )
    )
    return run


def add_behavior(run: Path) -> None:
    artifact = run / "rollouts_behavior_train.jsonl"
    rows = []
    for prompt_idx in range(2):
        for rollout_idx in range(2):
            rows.append(
                {
                    "prompt_idx": prompt_idx,
                    "rollout_idx": rollout_idx,
                    "input_ids": [1, 2, 3, 99],
                    "resp_start": 2,
                    "resp_end": 4,
                    "reward": float(rollout_idx),
                }
            )
    artifact.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = {
        "explicit_kwargs": {
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "repetition_penalty": 1.0,
            "no_repeat_ngram_size": 0,
            "max_new_tokens": 16,
        },
        "eos_token_ids": [99],
        "model_name_or_path": "/models/test-model",
        "k": 2,
        "n_prompts": 2,
        "idx_offset": 0,
        "artifact_file": artifact.name,
        "artifact_sha256": sha256_file(artifact),
    }
    (run / "rollouts_behavior_train.manifest.json").write_text(json.dumps(manifest))


def test_reuse_copies_and_revalidates(tmp_path: Path) -> None:
    source = make_run(tmp_path, "source", 0)
    target = make_run(tmp_path, "target", 100)
    add_behavior(source)
    result = reuse(source, target)
    assert result["status"] == "copied-and-validated"
    assert result["validated_rows"] == 4
    assert (target / "behavior_reuse.json").exists()
    assert reuse(source, target)["status"] == "already-valid"


def test_reuse_rejects_prompt_drift(tmp_path: Path) -> None:
    source = make_run(tmp_path, "source", 0)
    target = make_run(tmp_path, "target", 100)
    add_behavior(source)
    (target / "prompts.json").write_text(json.dumps({"train": [], "val": []}))
    try:
        reuse(source, target)
    except ValueError as exc:
        assert "prompts.json" in str(exc)
    else:
        raise AssertionError("prompt mismatch must be rejected")


def test_reuse_rejects_a_different_valid_behavior_sample(tmp_path: Path) -> None:
    source = make_run(tmp_path, "source", 0)
    target = make_run(tmp_path, "target", 100)
    add_behavior(source)
    add_behavior(target)
    artifact = target / "rollouts_behavior_train.jsonl"
    rows = [json.loads(line) for line in artifact.read_text().splitlines()]
    rows[0]["input_ids"] = [1, 2, 4, 99]
    artifact.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest_path = target / "rollouts_behavior_train.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifact_sha256"] = sha256_file(artifact)
    manifest_path.write_text(json.dumps(manifest))
    try:
        reuse(source, target)
    except ValueError as exc:
        assert "valid but different" in str(exc)
    else:
        raise AssertionError("a different valid behavior sample must be rejected")


def test_reuse_resumes_an_identical_partial_copy(tmp_path: Path) -> None:
    source = make_run(tmp_path, "source", 0)
    target = make_run(tmp_path, "target", 100)
    add_behavior(source)
    shutil.copy2(
        source / "rollouts_behavior_train.jsonl",
        target / "rollouts_behavior_train.jsonl",
    )
    result = reuse(source, target)
    assert result["status"] == "copied-and-validated"
