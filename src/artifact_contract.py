"""Artifact-only generation provenance and exact-coverage validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


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
) -> dict[str, list[str] | int]:
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
            if int(document.get("k", -1)) != k:
                raise ValueError(f"{path.name}: expected K={k}, got {document.get('k')}")
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

    manifests = [
        path
        for prefix, (n_prompts, k) in sources.items()
        for path in source_manifests(prefix, n_prompts, k)
    ]
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

    rows = 0
    merged = [run / f"{prefix}.jsonl" for prefix in sources]
    for path, (n_prompts, expected_k) in zip(merged, sources.values(), strict=True):
        if not path.is_file():
            raise ValueError(f"merged rollout file is missing: {path.name}")
        counts: dict[int, int] = {}
        for lineno, line in enumerate(path.open(), 1):
            row = json.loads(line)
            if "resp_end" not in row:
                raise ValueError(f"{path.name}:{lineno}: resp_end is missing")
            if int(row["resp_end"]) != len(row["input_ids"]):
                raise ValueError(f"{path.name}:{lineno}: input_ids are not trimmed at resp_end")
            if not 0 <= int(row["resp_start"]) < int(row["resp_end"]):
                raise ValueError(f"{path.name}:{lineno}: invalid response boundary")
            prompt_idx = int(row["prompt_idx"])
            counts[prompt_idx] = counts.get(prompt_idx, 0) + 1
            rows += 1
        expected_counts = {idx: expected_k for idx in range(n_prompts)}
        if counts != expected_counts:
            bad = sorted(
                idx for idx in set(counts) | set(expected_counts)
                if counts.get(idx) != expected_counts.get(idx)
            )
            raise ValueError(
                f"{path.name}: prompt/K coverage mismatch at {bad[:5]} "
                f"(expected K={expected_k})"
            )
    return {
        "manifests": [path.name for path in manifests],
        "rollout_files": [path.name for path in merged],
        "validated_rows": rows,
    }
