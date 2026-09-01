"""Artifact-only generation provenance and exact-coverage validation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from rollout_contract import ROLLOUT_SEED_SCHEME, rollout_seed_base

PRIMARY_SOURCES = (
    "rollouts_behavior_train",
    "rollouts_fresh_train",
    "rollouts_fresh_val",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_generation_contract(
    run: Path,
    source_names: tuple[str, ...] = PRIMARY_SOURCES,
    *,
    require_policy_binding: bool = False,
    require_rng_binding: bool = False,
) -> dict[str, object]:
    config = json.loads((run / "run_config.json").read_text())
    prompts = json.loads((run / "prompts.json").read_text())
    expected_kwargs = {
        "do_sample": True,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "max_new_tokens": int(config["max_new_tokens"]),
    }
    expected_model = Path(str(config.get("model_resolved", config.get("model", "")))).name
    all_sources = {
        "rollouts_behavior_train": (len(prompts["train"]), int(config["behavior_k"])),
        "rollouts_fresh_train": (len(prompts["train"]), int(config["fresh_k"])),
        "rollouts_fresh_val": (len(prompts["val"]), int(config["val_k"])),
    }
    unknown = sorted(set(source_names) - set(all_sources))
    if unknown:
        raise ValueError(f"unknown primary rollout sources: {unknown}")
    sources = {name: all_sources[name] for name in source_names}

    manifest_artifacts: dict[Path, Path] = {}
    unbound_manifests: list[str] = []
    unbound_policy_manifests: list[str] = []
    unbound_rng_manifests: list[str] = []
    drift = int(config.get("drift", 0))
    expected_policy = run / f"policy_step_{drift}" if drift > 0 else None

    def source_manifests(prefix: str, n_prompts: int, k: int) -> list[Path]:
        merged = run / f"{prefix}.manifest.json"
        manifests = [merged] if merged.exists() else sorted(
            run.glob(f"{prefix}.shard*.manifest.json")
        )
        if not manifests:
            raise ValueError(f"{prefix}: generation manifests are missing")
        covered: set[int] = set()
        for path in manifests:
            document = json.loads(path.read_text())
            expected_artifact_name = path.name.removesuffix(".manifest.json") + ".jsonl"
            recorded_artifact_name = document.get("artifact_file")
            if (recorded_artifact_name is not None
                    and recorded_artifact_name != expected_artifact_name):
                raise ValueError(
                    f"{path.name}: artifact_file mismatch: expected "
                    f"{expected_artifact_name!r}, got {document.get('artifact_file')!r}"
                )
            artifact = run / expected_artifact_name
            if not artifact.is_file():
                raise ValueError(f"{path.name}: bound artifact is missing: {artifact.name}")
            recorded_hash = document.get("artifact_sha256")
            if recorded_hash and recorded_hash != sha256_file(artifact):
                raise ValueError(f"{path.name}: rollout artifact hash mismatch")
            if not recorded_hash:
                unbound_manifests.append(path.name)
            manifest_artifacts[path] = artifact
            if int(document.get("k", -1)) != k:
                raise ValueError(f"{path.name}: expected K={k}, got {document.get('k')}")
            recorded_seed = document.get("sampling_seed_base")
            recorded_scheme = document.get("sampling_seed_scheme")
            expected_seed = rollout_seed_base(int(config.get("seed", 0)), drift, prefix)
            if recorded_seed is None and recorded_scheme is None:
                unbound_rng_manifests.append(path.name)
            elif (
                recorded_seed != expected_seed
                or recorded_scheme != ROLLOUT_SEED_SCHEME
            ):
                raise ValueError(f"{path.name}: rollout RNG binding mismatch")
            recorded_policy = document.get("policy_adapter")
            source_uses_policy = prefix.startswith("rollouts_fresh_")
            if not source_uses_policy or expected_policy is None:
                if recorded_policy is not None:
                    raise ValueError(
                        f"{path.name}: base-policy rollout unexpectedly records an adapter"
                    )
            elif recorded_policy is None:
                unbound_policy_manifests.append(path.name)
            else:
                expected_binding = {
                    "path": str(expected_policy.resolve()),
                    "adapter_sha256": sha256_file(
                        expected_policy / "adapter_model.safetensors"
                    ),
                    "policy_manifest_sha256": sha256_file(
                        expected_policy / "policy_train.json"
                    ),
                }
                if recorded_policy != expected_binding:
                    raise ValueError(
                        f"{path.name}: rollout policy adapter binding mismatch"
                    )
            lo = int(document.get("idx_offset", -1))
            count = int(document.get("n_prompts", -1))
            indices = set(range(lo, lo + count))
            if covered & indices:
                raise ValueError(f"{prefix}: overlapping manifest prompt ranges")
            covered |= indices
        expected_indices = set(range(n_prompts))
        if covered != expected_indices:
            raise ValueError(
                f"{prefix}: manifest prompt coverage mismatch; "
                f"missing={sorted(expected_indices - covered)[:5]}, "
                f"extra={sorted(covered - expected_indices)[:5]}"
            )
        return manifests

    manifests_by_source = {
        prefix: source_manifests(prefix, n_prompts, k)
        for prefix, (n_prompts, k) in sources.items()
    }
    manifests = [path for paths in manifests_by_source.values() for path in paths]
    for path in manifests:
        document = json.loads(path.read_text())
        kwargs = document.get("explicit_kwargs", {})
        mismatches = {
            key: {"expected": value, "recorded": kwargs.get(key)}
            for key, value in expected_kwargs.items()
            if kwargs.get(key) != value
        }
        if mismatches:
            raise ValueError(f"{path.name}: generation contract mismatch: {mismatches}")
        if not document.get("eos_token_ids"):
            raise ValueError(f"{path.name}: eos_token_ids are missing")
        recorded_model = Path(str(document.get("model_name_or_path", ""))).name
        if not expected_model or recorded_model != expected_model:
            raise ValueError(
                f"{path.name}: model mismatch: expected {expected_model!r}, "
                f"recorded {recorded_model!r}"
            )

    # A merged JSONL must contain exactly the rows bound by its shard manifests.
    # Coverage alone would not detect a modified reward/token sequence.
    shard_rows: dict[str, dict[tuple[int, int], dict]] = {}
    for prefix, source_manifests_for_prefix in manifests_by_source.items():
        artifacts = [manifest_artifacts[path] for path in source_manifests_for_prefix]
        merged_path = run / f"{prefix}.jsonl"
        if artifacts == [merged_path]:
            continue
        expected: dict[tuple[int, int], dict] = {}
        for artifact in artifacts:
            for lineno, line in enumerate(artifact.open(), 1):
                row = json.loads(line)
                key = (int(row["prompt_idx"]), int(row["rollout_idx"]))
                if key in expected:
                    raise ValueError(
                        f"{artifact.name}:{lineno}: duplicate rollout key {key} in shards"
                    )
                expected[key] = row
        shard_rows[prefix] = expected

    rows = 0
    merged = [run / f"{prefix}.jsonl" for prefix in sources]
    for (prefix, (n_prompts, expected_k)), path in zip(
        sources.items(), merged, strict=True
    ):
        if not path.is_file():
            raise ValueError(f"merged rollout file is missing: {path.name}")
        rollout_indices: dict[int, set[int]] = {}
        seen_keys: set[tuple[int, int]] = set()
        expected_rows = shard_rows.get(prefix)
        for lineno, line in enumerate(path.open(), 1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path.name}:{lineno}: rollout row is not an object")
            if "resp_end" not in row:
                raise ValueError(f"{path.name}:{lineno}: resp_end is missing")
            if int(row["resp_end"]) != len(row["input_ids"]):
                raise ValueError(f"{path.name}:{lineno}: input_ids are not trimmed at resp_end")
            if not 0 <= int(row["resp_start"]) < int(row["resp_end"]):
                raise ValueError(f"{path.name}:{lineno}: invalid response boundary")
            prompt_idx = int(row["prompt_idx"])
            rollout_idx = int(row["rollout_idx"])
            if not 0 <= prompt_idx < n_prompts:
                raise ValueError(
                    f"{path.name}:{lineno}: prompt_idx is outside the prompt set"
                )
            input_ids = row.get("input_ids")
            if (
                not isinstance(input_ids, list)
                or not input_ids
                or any(
                    not isinstance(token, int) or isinstance(token, bool)
                    for token in input_ids
                )
            ):
                raise ValueError(f"{path.name}:{lineno}: invalid input_ids")
            reward = row.get("reward")
            if (
                not isinstance(reward, (int, float))
                or isinstance(reward, bool)
                or not math.isfinite(float(reward))
                or float(reward) not in {0.0, 1.0}
            ):
                raise ValueError(f"{path.name}:{lineno}: reward is not binary and finite")
            key = (prompt_idx, rollout_idx)
            if key in seen_keys:
                raise ValueError(f"{path.name}:{lineno}: duplicate rollout key {key}")
            seen_keys.add(key)
            rollout_indices.setdefault(prompt_idx, set()).add(rollout_idx)
            if expected_rows is not None and expected_rows.get(key) != row:
                raise ValueError(
                    f"{path.name}:{lineno}: merged row differs from bound shard {key}"
                )
            rows += 1
        if expected_rows is not None and seen_keys != set(expected_rows):
            missing = sorted(set(expected_rows) - seen_keys)
            extra = sorted(seen_keys - set(expected_rows))
            raise ValueError(
                f"{path.name}: merged/shard row-set mismatch; "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
        expected_indices = set(range(expected_k))
        bad = sorted(
            idx for idx in set(rollout_indices) | set(range(n_prompts))
            if rollout_indices.get(idx) != expected_indices
        )
        if bad:
            raise ValueError(
                f"{path.name}: prompt/exact-K coverage mismatch at {bad[:5]} "
                f"(expected K={expected_k})"
            )
    if require_policy_binding and unbound_policy_manifests:
        raise ValueError(
            "fresh rollout manifests are not bound to the trained policy: "
            + ", ".join(sorted(unbound_policy_manifests))
        )
    if require_rng_binding and unbound_rng_manifests:
        raise ValueError(
            "rollout manifests are not bound to independent RNG streams: "
            + ", ".join(sorted(unbound_rng_manifests))
        )
    return {
        "manifests": [path.name for path in manifests],
        "rollout_files": [path.name for path in merged],
        "validated_rows": rows,
        "manifest_sha256": {
            path.name: sha256_file(path) for path in manifests
        },
        "artifact_sha256": {
            path.name: sha256_file(path)
            for path in dict.fromkeys([*manifest_artifacts.values(), *merged])
        },
        "generation_hash_missing": sorted(unbound_manifests),
        "policy_binding_missing": sorted(unbound_policy_manifests),
        "rng_binding_missing": sorted(unbound_rng_manifests),
    }
