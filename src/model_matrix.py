#!/usr/bin/env python3
"""Pinned model-family download and static compatibility checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "domain_transfer.json"
DOWNLOAD_PATTERNS = [
    "README.md",
    "config.json",
    "generation_config.json",
    "model-*.safetensors",
    "model.safetensors.index.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "merges.txt",
    "vocab.json",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_specs(config_path: Path) -> dict[str, dict]:
    config = json.loads(config_path.read_text())
    specs = {row["key"]: row for row in config["models"]}
    if len(specs) != len(config["models"]):
        raise ValueError("duplicate model key")
    return specs


def _snapshot_path(spec: dict, models_dir: Path) -> Path:
    return models_dir / spec["local_directory"]


def _check_snapshot(spec: dict, path: Path) -> dict:
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    required = [
        path / "config.json",
        path / "tokenizer_config.json",
        path / "model.safetensors.index.json",
        path / ".om_snapshot.json",
    ]
    missing = [p.name for p in required if not p.is_file()]
    if missing:
        raise ValueError(f"{spec['key']}: missing files: {', '.join(missing)}")

    manifest = json.loads((path / ".om_snapshot.json").read_text())
    if manifest.get("repository") != spec["repository"] or manifest.get("revision") != spec["revision"]:
        raise ValueError(f"{spec['key']}: snapshot provenance mismatch")
    if manifest.get("config_sha256") != _sha256(path / "config.json"):
        raise ValueError(f"{spec['key']}: config hash mismatch")

    tokenizer_config = json.loads((path / "tokenizer_config.json").read_text())
    if not tokenizer_config.get("chat_template"):
        raise ValueError(f"{spec['key']}: tokenizer chat_template missing")
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    if config.model_type != spec["model_type"]:
        raise ValueError(
            f"{spec['key']}: model_type={config.model_type}, expected={spec['model_type']}"
        )

    index = json.loads((path / "model.safetensors.index.json").read_text())
    shards = sorted(set(index.get("weight_map", {}).values()))
    if not shards or any(not (path / shard).is_file() for shard in shards):
        raise ValueError(f"{spec['key']}: safetensors shard set incomplete")
    weight_bytes = sum((path / shard).stat().st_size for shard in shards)
    if weight_bytes < 1_000_000_000:
        raise ValueError(f"{spec['key']}: implausibly small weight snapshot")

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    module_suffixes = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    absent = [target for target in spec["lora_targets"] if target not in module_suffixes]
    if absent:
        raise ValueError(f"{spec['key']}: LoRA targets missing: {absent}")
    return {
        "key": spec["key"],
        "path": str(path),
        "revision": spec["revision"],
        "model_type": config.model_type,
        "weight_shards": len(shards),
        "weight_bytes": weight_bytes,
        "lora_targets": spec["lora_targets"],
    }


def _write_manifest(spec: dict, path: Path) -> None:
    manifest = {
        "schema_version": 1,
        "repository": spec["repository"],
        "revision": spec["revision"],
        "config_sha256": _sha256(path / "config.json"),
        "tokenizer_config_sha256": _sha256(path / "tokenizer_config.json"),
        "weight_index_sha256": _sha256(path / "model.safetensors.index.json"),
    }
    target = path / ".om_snapshot.json"
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _download(spec: dict, models_dir: Path) -> dict:
    from huggingface_hub import snapshot_download

    destination = _snapshot_path(spec, models_dir)
    try:
        return _check_snapshot(spec, destination)
    except (OSError, ValueError, json.JSONDecodeError):
        pass

    models_dir.mkdir(parents=True, exist_ok=True)
    downloads = models_dir / ".downloads"
    downloads.mkdir(exist_ok=True)
    temporary = downloads / f"{spec['local_directory']}.{os.getpid()}"
    if temporary.exists():
        quarantine = models_dir / ".quarantine"
        quarantine.mkdir(exist_ok=True)
        temporary.replace(quarantine / f"{temporary.name}.{time.time_ns()}")
    temporary.mkdir()
    try:
        snapshot_download(
            repo_id=spec["repository"],
            revision=spec["revision"],
            local_dir=temporary,
            allow_patterns=DOWNLOAD_PATTERNS,
        )
        _write_manifest(spec, temporary)
        checked = _check_snapshot(spec, temporary)
        if destination.exists():
            quarantine = models_dir / ".quarantine"
            quarantine.mkdir(exist_ok=True)
            destination.replace(quarantine / f"{destination.name}.{time.time_ns()}")
        temporary.replace(destination)
        checked["path"] = str(destination)
        return checked
    except Exception:
        if temporary.exists():
            failed = downloads / f"{temporary.name}.failed.{time.time_ns()}"
            temporary.replace(failed)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--models-dir", type=Path, default=os.environ.get("MODELS_DIR"))
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("check", "download"):
        p = sub.add_parser(command)
        p.add_argument("models", nargs="*")
    p = sub.add_parser("field")
    p.add_argument("model")
    p.add_argument("name", choices=["path", "lora_targets", "repository", "revision"])
    args = parser.parse_args()

    specs = _load_specs(args.config)
    if args.models_dir is None:
        parser.error("--models-dir or MODELS_DIR is required")
    models = getattr(args, "models", None) or list(specs)
    unknown = sorted(set(models) - set(specs))
    if unknown:
        raise ValueError(f"unknown models: {', '.join(unknown)}")

    if args.command == "field":
        spec = specs[args.model]
        if args.name == "path":
            print(_snapshot_path(spec, args.models_dir))
        elif args.name == "lora_targets":
            print(",".join(spec["lora_targets"]))
        else:
            print(spec[args.name])
        return

    for key in models:
        spec = specs[key]
        result = (
            _download(spec, args.models_dir)
            if args.command == "download"
            else _check_snapshot(spec, _snapshot_path(spec, args.models_dir))
        )
        print(
            f"[{args.command}] {key}: {result['model_type']} "
            f"{result['weight_shards']} shards {result['weight_bytes'] / 1e9:.1f} GB "
            f"revision={result['revision'][:12]}"
        )


if __name__ == "__main__":
    try:
        main()
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[model-abort] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
