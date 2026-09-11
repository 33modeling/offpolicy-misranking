"""Publish a canonical manifest for a proven local model-path alias.

Called by the supervisor while holding the point lock, never by status. The
pinned generator, run configuration, rollout bytes and partials stay unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from artifact_contract import (
    PRIMARY_SOURCES,
    record_rollout_ready,
    sha256_file,
    validate_generation_contract,
)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def encoded(document: dict) -> bytes:
    return (json.dumps(document, sort_keys=True, indent=2) + "\n").encode()


def prove_model(config: dict, documents: list[dict]) -> str:
    expected = Path(config.get("model_resolved", ""))
    if not expected.is_absolute() or not expected.is_dir():
        raise ValueError("recorded model_resolved is not an accessible absolute directory")
    if expected.resolve(strict=True) != expected:
        raise ValueError("recorded canonical model path now resolves elsewhere")
    for value in [config.get("model", ""), *(d.get("model_name_or_path", "") for d in documents)]:
        alias = Path(value)
        if not alias.is_absolute() or alias.resolve(strict=True) != expected:
            raise ValueError(f"model alias cannot be proven identical: {value!r} -> {str(expected)!r}")
    if not config.get("model_config_sha256"):
        raise ValueError("run configuration has no model config hash")
    for key, name in (
        ("model_config_sha256", "config.json"),
        ("tokenizer_config_sha256", "tokenizer_config.json"),
        ("generation_config_sha256", "generation_config.json"),
        ("model_snapshot_manifest_sha256", ".om_snapshot.json"),
    ):
        if key in config:
            path = expected / name
            actual = sha256_file(path) if path.is_file() else None
            if actual != config[key]:
                raise ValueError(f"model identity changed: {name} hash mismatch")
    return str(expected)


def repair(run: Path) -> int:
    if not (run / "run_config.json").is_file():
        return 0
    config = read(run / "run_config.json")
    expected_name = Path(config.get("model_resolved", config.get("model", ""))).name
    plans = []
    for prefix in PRIMARY_SOURCES:
        # An interrupted, not-yet-merged generation is not a repair candidate.
        if not (run / f"{prefix}.jsonl").is_file():
            continue
        target = run / f"{prefix}.manifest.json"
        paths = [target] if target.exists() else sorted(run.glob(f"{prefix}.shard*.manifest.json"))
        documents = [read(path) for path in paths]
        if not any(Path(d.get("model_name_or_path", "")).name != expected_name for d in documents):
            continue
        model = prove_model(config, documents)
        # Never invalidate previously bound scores, training inputs or reports.
        protected = ("DONE", "*.pt", "scores*.json", "*protocol*.json", "report.*",
                     ".regime_validated.json")
        if prefix == "rollouts_behavior_train":
            protected += ("behavior_reuse.json",)
        if any(list(run.glob(pattern)) for pattern in protected):
            raise ValueError("model alias repair refused: derived/bound artifacts already exist")
        originals = {path.name: path.read_bytes() for path in paths}
        with tempfile.TemporaryDirectory(prefix=".model-alias-validation-", dir=run) as tmp:
            view = Path(tmp)
            for name in ("run_config.json", "prompts.json"):
                (view / name).symlink_to((run / name).resolve())
            for path in run.glob("policy_step_*"):
                if path.is_dir():
                    (view / path.name).symlink_to(path.resolve(), target_is_directory=True)
            for path in run.glob(f"{prefix}*.jsonl"):
                (view / path.name).symlink_to(path.resolve())
            normalized = [dict(document, model_name_or_path=model) for document in documents]
            for path, document in zip(paths, normalized, strict=True):
                (view / path.name).write_bytes(encoded(document))
            # Validate the original shard coverage, hashes, merged rows, RNG,
            # adapter and sampling contract before publishing a merged sidecar.
            validation = validate_generation_contract(view, (prefix,))
            if validation["generation_hash_missing"]:
                raise ValueError("model alias repair requires hash-bound rollout artifacts")
            varying = {"artifact_file", "artifact_sha256", "idx_offset", "n_prompts"}
            common = {key: value for key, value in normalized[0].items() if key not in varying}
            if any({key: value for key, value in d.items() if key not in varying} != common
                   for d in normalized[1:]):
                raise ValueError(f"{prefix}: shard manifests disagree")
            prompts = read(run / "prompts.json")
            canonical = dict(common, artifact_file=f"{prefix}.jsonl",
                             artifact_sha256=sha256_file(run / f"{prefix}.jsonl"),
                             idx_offset=0, n_prompts=len(prompts["val" if prefix.endswith("_val") else "train"]))
            payload = encoded(canonical)
            (view / target.name).write_bytes(payload)
            validate_generation_contract(view, (prefix,))
        plans.append((target, originals, payload))
    # Prove and validate every candidate before changing any manifest.
    for target, originals, payload in plans:
        digest = hashlib.sha256(payload).hexdigest()
        archive = run / "logs" / "model-alias-repair" / digest
        for name, data in originals.items():
            atomic_bytes(archive / name, data)
        atomic_bytes(archive / "receipt.json", encoded({
            "schema": "model-alias-repair-v1", "target": target.name,
            "canonical_manifest_sha256": digest,
            "original_manifest_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in originals.items()},
            "run_config_sha256": sha256_file(run / "run_config.json"),
            "repair_code_sha256": sha256_file(Path(__file__)),
            "generation_git": config.get("git"),
            "model_resolved": config["model_resolved"],
        }))
        atomic_bytes(target, payload)
        document = json.loads(payload)
        record_rollout_ready(run / document["artifact_file"], document["artifact_sha256"])
        print(f"[model-alias-repair] {target.name}: verified canonical model path; rollout bytes preserved; audit={archive}", flush=True)
    return len(plans)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--require-change", action="store_true")
    args = parser.parse_args()
    try:
        count = repair(args.run.resolve())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"[permanent-contract] model alias repair refused: {exc}", flush=True)
        return 43
    return 43 if args.require_change and not count else 0


if __name__ == "__main__":
    raise SystemExit(main())
