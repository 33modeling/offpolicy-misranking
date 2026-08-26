"""Completed runs should retain canonical artifacts, not redundant copies."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from artifact_contract import sha256_file, validate_generation_contract
from compact_artifacts import (
    compact_adapter,
    compact_analysis_shards,
    compact_rollout_shards,
    merge_divergence_shards,
)


def rollout_row(prompt_idx: int, rollout_idx: int) -> dict:
    return {
        "prompt_idx": prompt_idx,
        "rollout_idx": rollout_idx,
        "input_ids": [1, 2, 3, 99],
        "resp_start": 2,
        "resp_end": 4,
        "reward": float(rollout_idx),
    }


def shard_manifest(artifact: Path, prompt_idx: int) -> dict:
    return {
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
        "n_prompts": 1,
        "idx_offset": prompt_idx,
        "artifact_file": artifact.name,
        "artifact_sha256": sha256_file(artifact),
    }


def make_sharded_rollout(run: Path) -> None:
    run.mkdir()
    (run / "run_config.json").write_text(
        json.dumps(
            {
                "behavior_k": 2,
                "fresh_k": 4,
                "val_k": 2,
                "max_new_tokens": 16,
                "model": "/models/test-model",
            }
        )
    )
    (run / "prompts.json").write_text(
        json.dumps({"train": [{}, {}], "val": [{}]})
    )
    merged_rows = []
    for prompt_idx in range(2):
        rows = [rollout_row(prompt_idx, rollout_idx) for rollout_idx in range(2)]
        merged_rows.extend(rows)
        artifact = run / f"rollouts_behavior_train.shard{prompt_idx}.jsonl"
        artifact.write_text("".join(json.dumps(row) + "\n" for row in rows))
        manifest = artifact.with_name(artifact.stem + ".manifest.json")
        manifest.write_text(json.dumps(shard_manifest(artifact, prompt_idx)))
    (run / "rollouts_behavior_train.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in merged_rows)
    )


def test_rollout_compaction_publishes_manifest_then_removes_shards(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    make_sharded_rollout(run)
    removed = compact_rollout_shards(run, "rollouts_behavior_train")
    assert len(removed) == 4
    assert not list(run.glob("rollouts_behavior_train.shard*"))
    result = validate_generation_contract(run, ("rollouts_behavior_train",))
    assert result["validated_rows"] == 4
    assert result["manifests"] == ["rollouts_behavior_train.manifest.json"]


def test_rollout_compaction_preserves_shards_when_validation_fails(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    make_sharded_rollout(run)
    merged = run / "rollouts_behavior_train.jsonl"
    merged.write_text(merged.read_text().replace('"reward": 0.0', '"reward": 1.0', 1))
    try:
        compact_rollout_shards(run, "rollouts_behavior_train")
    except ValueError as exc:
        assert "differs from bound shard" in str(exc)
    else:
        raise AssertionError("invalid merged rollout must not delete source shards")
    assert len(list(run.glob("rollouts_behavior_train.shard*"))) == 4
    assert not (run / "rollouts_behavior_train.manifest.json").exists()


def test_adapter_compaction_removes_base_model_duplicates(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    for name in (
        "adapter_config.json",
        "adapter_model.safetensors",
        "README.md",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ):
        (adapter / name).write_text(name)
    removed = compact_adapter(adapter)
    assert {path.name for path in removed} == {
        "README.md",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    }
    assert {path.name for path in adapter.iterdir()} == {
        "adapter_config.json",
        "adapter_model.safetensors",
    }


def test_analysis_compaction_merges_exact_divergence_and_removes_shards(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    weights = ((1.0, 2.0), (3.0, 4.0))
    for shard, shard_weights in enumerate(weights):
        document = {
            "token_kl_beta_pi": 0.1 + 0.2 * shard,
            "traj_ess_frac_g11": 0.0,
            "rollouts": 2,
            "tokens": 10 + 10 * shard,
            "traj_logw_logsumexp": math.log(sum(shard_weights)),
            "traj_logw2_logsumexp": math.log(
                sum(weight * weight for weight in shard_weights)
            ),
            "clipfrac_g11": 0.2 + 0.4 * shard,
        }
        (run / f"divergence_stats.shard{shard}.json").write_text(
            json.dumps(document)
        )
        (run / f"scores_offpolicy.shard{shard}.json").write_text("{}")
        (run / f"score_protocol.shard{shard}.json").write_text("{}")
        (run / f"oracle_micro_groups.shard{shard}.pt").write_text("tensor")

    merge_divergence_shards(run)
    for name in (
        "scores_offpolicy.json",
        "score_protocol.json",
        "oracle_micro_groups.pt",
    ):
        (run / name).write_text("canonical")
    removed = compact_analysis_shards(run)
    result = json.loads((run / "divergence_stats.json").read_text())
    assert math.isclose(result["token_kl_beta_pi"], 7 / 30)
    assert math.isclose(result["clipfrac_g11"], 0.4)
    assert math.isclose(result["traj_ess_frac_g11"], 100 / 30 / 4)
    assert len(removed) == 8
    assert not list(run.glob("*.shard*"))
