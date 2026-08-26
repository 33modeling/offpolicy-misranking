#!/usr/bin/env python3
"""Safely reuse one immutable behavior-rollout pool across a drift sweep."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from artifact_contract import sha256_file, validate_generation_contract
from compact_artifacts import compact_rollout_shards

MATCHED_CONFIG = (
    "model_resolved",
    "model_config_sha256",
    "tokenizer_config_sha256",
    "generation_config_sha256",
    "dataset",
    "n_train",
    "behavior_k",
    "max_new_tokens",
    "temperature",
    "top_p",
    "thinking",
)


def compatibility_errors(source: Path, target: Path) -> list[str]:
    source_config = json.loads((source / "run_config.json").read_text())
    target_config = json.loads((target / "run_config.json").read_text())
    errors = []
    for key in MATCHED_CONFIG:
        source_value = source_config.get(key)
        target_value = target_config.get(key)
        if source_value != target_value:
            errors.append(f"{key}: source={source_value!r} target={target_value!r}")
    source_prompts = source / "prompts.json"
    target_prompts = target / "prompts.json"
    if sha256_file(source_prompts) != sha256_file(target_prompts):
        errors.append("prompts.json: content hash differs")
    return errors


def behavior_files(run: Path) -> list[Path]:
    return [
        path
        for path in (
            run / "rollouts_behavior_train.jsonl",
            run / "rollouts_behavior_train.manifest.json",
        )
        if path.is_file()
    ]


def check(run: Path) -> bool:
    try:
        validate_generation_contract(run, ("rollouts_behavior_train",))
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False
    return True


def reuse(source: Path, target: Path) -> dict:
    source = source.resolve()
    target = target.resolve()
    if source == target:
        raise ValueError("source and target run must differ")
    validate_generation_contract(source, ("rollouts_behavior_train",))
    compact_rollout_shards(source, "rollouts_behavior_train")
    errors = compatibility_errors(source, target)
    if errors:
        raise ValueError("behavior reuse contract mismatch: " + "; ".join(errors))
    sources = behavior_files(source)
    if not sources or not (source / "rollouts_behavior_train.jsonl").is_file():
        raise ValueError("source has no merged behavior rollout artifact")
    source_by_name = {path.name: path for path in sources}
    if check(target):
        compact_rollout_shards(target, "rollouts_behavior_train")
        target_by_name = {path.name: path for path in behavior_files(target)}
        if set(target_by_name) != set(source_by_name) or any(
            sha256_file(path) != sha256_file(target_by_name[name])
            for name, path in source_by_name.items()
        ):
            raise ValueError(
                "target has a valid but different behavior sample; drift-family "
                "points must use the exact source artifact"
            )
        return {"status": "already-valid", "source": str(source), "target": str(target)}

    existing = behavior_files(target)
    mismatched = [
        path.name
        for path in existing
        if path.name not in source_by_name
        or sha256_file(path) != sha256_file(source_by_name[path.name])
    ]
    if mismatched:
        raise ValueError(
            "target contains incomplete or invalid behavior artifacts; use a new run "
            f"directory instead of overwriting them: {mismatched}"
        )
    storage = {}
    for path in sources:
        destination = target / path.name
        if destination.exists():
            storage[path.name] = "existing"
            continue
        temporary = target / f"{path.name}.reuse.tmp"
        temporary.unlink(missing_ok=True)
        try:
            os.link(path, temporary)
            storage[path.name] = "hardlink"
        except OSError:
            shutil.copy2(path, temporary)
            storage[path.name] = "copy"
        temporary.replace(destination)
    validation = validate_generation_contract(target, ("rollouts_behavior_train",))
    compact_rollout_shards(target, "rollouts_behavior_train")
    record = {
        "schema": "offpolicy-behavior-reuse/v1",
        "status": "copied-and-validated",
        "source": str(source),
        "target": str(target),
        "source_config_sha256": sha256_file(source / "run_config.json"),
        "target_config_sha256": sha256_file(target / "run_config.json"),
        "prompts_sha256": sha256_file(target / "prompts.json"),
        "validated_rows": validation["validated_rows"],
        "artifact_sha256": validation["artifact_sha256"],
        "manifest_sha256": validation["manifest_sha256"],
        "storage": storage,
    }
    tmp = target / "behavior_reuse.json.tmp"
    tmp.write_text(json.dumps(record, indent=1))
    tmp.replace(target / "behavior_reuse.json")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, nargs="?")
    parser.add_argument("target", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.check:
        return 0 if check(args.target) else 1
    if args.source is None:
        parser.error("source is required unless --check is used")
    try:
        result = reuse(args.source, args.target)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"[behavior-reuse-abort] {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
