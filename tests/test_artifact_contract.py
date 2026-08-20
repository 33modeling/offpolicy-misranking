"""Generation manifest and exact prompt/K coverage regressions (CPU only)."""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from artifact_contract import sha256_file, validate_generation_contract


def write_source(run: Path, prefix: str, n: int, k: int) -> None:
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
        "k": k,
        "n_prompts": n,
        "idx_offset": 0,
    }
    manifest_path = run / f"{prefix}.manifest.json"
    rows = []
    for prompt_idx in range(n):
        for rollout_idx in range(k):
            rows.append({
                "prompt_idx": prompt_idx,
                "rollout_idx": rollout_idx,
                "input_ids": [1, 2, 3, 99],
                "resp_start": 2,
                "resp_end": 4,
                "reward": 0.0,
            })
    artifact_path = run / f"{prefix}.jsonl"
    artifact_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    manifest.update({
        "artifact_file": artifact_path.name,
        "artifact_sha256": sha256_file(artifact_path),
    })
    manifest_path.write_text(json.dumps(manifest))


def make_run(run: Path) -> None:
    (run / "run_config.json").write_text(json.dumps({
        "behavior_k": 2,
        "fresh_k": 4,
        "val_k": 2,
        "max_new_tokens": 16,
        "model": "/models/test-model",
    }))
    (run / "prompts.json").write_text(json.dumps({
        "train": [{}, {}],
        "val": [{}],
    }))
    write_source(run, "rollouts_behavior_train", 2, 2)
    write_source(run, "rollouts_fresh_train", 2, 4)
    write_source(run, "rollouts_fresh_val", 1, 2)


def test_generation_contract_and_exact_coverage():
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp)
        make_run(run)
        result = validate_generation_contract(run)
        assert result["validated_rows"] == 14
        assert result["manifest_sha256"]
        assert result["artifact_sha256"]
        assert result["generation_hash_missing"] == []
        assert validate_generation_contract(
            run, ("rollouts_behavior_train",)
        )["validated_rows"] == 4

        behavior = run / "rollouts_behavior_train.jsonl"
        original = behavior.read_text()
        behavior.write_text("\n".join(original.splitlines()[:-1]) + "\n")
        try:
            validate_generation_contract(run, ("rollouts_behavior_train",))
        except ValueError as exc:
            assert "artifact hash mismatch" in str(exc)
        else:
            raise AssertionError("partial rollout artifact must be rejected")
        behavior.write_text(original)

        manifest_path = run / "rollouts_behavior_train.manifest.json"
        manifest = json.loads(manifest_path.read_text())
        rows = [json.loads(line) for line in behavior.read_text().splitlines()]
        rows[-1]["rollout_idx"] = 0
        behavior.write_text("".join(json.dumps(row) + "\n" for row in rows))
        manifest["artifact_sha256"] = sha256_file(behavior)
        manifest_path.write_text(json.dumps(manifest))
        try:
            validate_generation_contract(run, ("rollouts_behavior_train",))
        except ValueError as exc:
            assert "duplicate rollout key" in str(exc)
        else:
            raise AssertionError("duplicate rollout indices must be rejected")

        behavior.write_text(original)
        manifest["artifact_sha256"] = sha256_file(behavior)
        manifest["explicit_kwargs"]["top_k"] = 20
        manifest_path.write_text(json.dumps(manifest))
        try:
            validate_generation_contract(run, ("rollouts_behavior_train",))
        except ValueError as exc:
            assert "generation contract mismatch" in str(exc)
        else:
            raise AssertionError("sampling mismatch must be rejected")


def test_legacy_manifest_is_hashed_at_validation_time():
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp)
        make_run(run)
        manifest_path = run / "rollouts_behavior_train.manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest.pop("artifact_file")
        manifest.pop("artifact_sha256")
        manifest_path.write_text(json.dumps(manifest))
        result = validate_generation_contract(run, ("rollouts_behavior_train",))
        assert result["generation_hash_missing"] == [manifest_path.name]
        assert result["artifact_sha256"]


def test_merged_rollouts_must_match_bound_shards():
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp)
        make_run(run)
        prefix = "rollouts_behavior_train"
        merged = run / f"{prefix}.jsonl"
        rows = [json.loads(line) for line in merged.read_text().splitlines()]
        base_manifest_path = run / f"{prefix}.manifest.json"
        base_manifest = json.loads(base_manifest_path.read_text())
        base_manifest_path.unlink()

        for shard_index in (0, 1):
            artifact = run / f"{prefix}.shard{shard_index}.jsonl"
            artifact.write_text("".join(
                json.dumps(row) + "\n"
                for row in rows if row["prompt_idx"] == shard_index
            ))
            manifest = {
                **base_manifest,
                "idx_offset": shard_index,
                "n_prompts": 1,
                "artifact_file": artifact.name,
                "artifact_sha256": sha256_file(artifact),
            }
            (run / f"{prefix}.shard{shard_index}.manifest.json").write_text(
                json.dumps(manifest)
            )

        validate_generation_contract(run, (prefix,))
        rows[0]["reward"] = 1.0
        merged.write_text("".join(json.dumps(row) + "\n" for row in rows))
        try:
            validate_generation_contract(run, (prefix,))
        except ValueError as exc:
            assert "merged row differs from bound shard" in str(exc)
        else:
            raise AssertionError("merged content drift from shards must be rejected")


if __name__ == "__main__":
    test_generation_contract_and_exact_coverage()
    test_legacy_manifest_is_hashed_at_validation_time()
    test_merged_rollouts_must_match_bound_shards()
    print("PASS generation contract and exact coverage")
