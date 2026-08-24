#!/usr/bin/env python3
"""Pinned model-family download and static compatibility checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "domain_transfer.json"
DOWNLOAD_PATTERNS = [
    "README.md",
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "model-*.safetensors",
    "model.safetensors.index.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "merges.txt",
    "vocab.json",
]

EXPERIMENT_FIELDS = {
    "datasets",
    "seeds",
    "drifts",
    "n_train",
    "n_val",
    "behavior_k",
    "fresh_k",
    "val_k",
    "micro_group",
    "max_new_tokens",
    "proj_dim",
    "grad_layers",
    "clip_cap",
    "topk_frac",
    "temperature",
    "top_p",
    "thinking",
    "attn",
    "skip_hybrid",
    "first_bootstrap",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config(config_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("domain-transfer config must be a JSON object")
    if config.get("schema_version") != 1:
        raise ValueError("unsupported domain-transfer config schema")
    if not isinstance(config.get("models"), list) or not config["models"]:
        raise ValueError("domain-transfer model list is empty")
    experiment = config.get("experiment")
    if not isinstance(experiment, dict):
        raise TypeError("domain-transfer experiment must be a JSON object")
    missing = sorted(EXPERIMENT_FIELDS - set(experiment or {}))
    if missing:
        raise ValueError(f"domain-transfer experiment fields missing: {missing}")
    for key in ("datasets", "seeds", "drifts"):
        values = experiment[key]
        if not isinstance(values, list) or not values:
            raise ValueError(f"experiment.{key} must be a non-empty unique list")
        try:
            unique = len(values) == len(set(values))
        except TypeError as exc:
            raise TypeError(f"experiment.{key} values must be scalar") from exc
        if not unique:
            raise ValueError(f"experiment.{key} must be a non-empty unique list")
    if not all(isinstance(value, str) and value for value in experiment["datasets"]):
        raise TypeError("experiment.datasets must contain non-empty strings")
    if experiment["drifts"][0] != 0 or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in experiment["drifts"]
    ):
        raise ValueError("experiment.drifts must begin with integer positive control 0")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in experiment["seeds"]
    ):
        raise ValueError("experiment.seeds must contain non-negative integers")
    for key in (
        "n_train", "n_val", "behavior_k", "fresh_k", "val_k",
        "micro_group", "max_new_tokens", "proj_dim", "grad_layers",
        "first_bootstrap",
    ):
        if (
            isinstance(experiment[key], bool)
            or not isinstance(experiment[key], int)
            or experiment[key] <= 0
        ):
            raise ValueError(f"experiment.{key} must be a positive integer")
    if experiment["fresh_k"] % experiment["micro_group"]:
        raise ValueError("fresh_k must be divisible by micro_group")
    if (experiment["fresh_k"] // experiment["micro_group"]) % 2:
        raise ValueError("fresh_k/micro_group must produce an even group count")
    if experiment["temperature"] != 1.0 or experiment["top_p"] != 1.0:
        raise ValueError("transfer matrix must use the raw-softmax sampling contract")
    if not isinstance(experiment["clip_cap"], (int, float)) or experiment["clip_cap"] < 1:
        raise ValueError("experiment.clip_cap must be numeric and >= 1")
    if not isinstance(experiment["topk_frac"], (int, float)) or not (
        0 < experiment["topk_frac"] <= 1
    ):
        raise ValueError("experiment.topk_frac must be in (0, 1]")
    if experiment["thinking"] not in {"off", "on"}:
        raise ValueError("experiment.thinking must be 'off' or 'on'")
    if experiment["attn"] not in {"eager", "sdpa", "flash_attention_2"}:
        raise ValueError("experiment.attn is unsupported")
    if not isinstance(experiment["skip_hybrid"], bool):
        raise TypeError("experiment.skip_hybrid must be boolean")
    return config


def _load_specs(config_path: Path) -> dict[str, dict]:
    config = _load_config(config_path)
    if not all(isinstance(row, dict) and isinstance(row.get("key"), str) for row in config["models"]):
        raise TypeError("each domain-transfer model must be an object with a string key")
    specs = {row["key"]: row for row in config["models"]}
    if len(specs) != len(config["models"]):
        raise ValueError("duplicate model key")
    for key, spec in specs.items():
        if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", key):
            raise ValueError(f"invalid model key: {key!r}")
        revision = spec.get("revision", "")
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError(f"{key}: revision must be a full immutable commit SHA")
        local = Path(str(spec.get("local_directory", "")))
        if not local.name or local.is_absolute() or len(local.parts) != 1:
            raise ValueError(f"{key}: local_directory must be one safe path component")
        targets = spec.get("lora_targets")
        if not isinstance(targets, list) or not targets or not all(
            isinstance(target, str) and target for target in targets
        ):
            raise ValueError(f"{key}: lora_targets must be a non-empty string list")
    return specs


def _snapshot_path(spec: dict, models_dir: Path) -> Path:
    return models_dir / spec["local_directory"]


def _weight_shards(path: Path) -> list[Path]:
    index_path = path / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        names = sorted(set(index.get("weight_map", {}).values()))
        if not names:
            raise ValueError("weight index contains no shards")
        shards = []
        for name in names:
            candidate = Path(str(name))
            if candidate.is_absolute() or len(candidate.parts) != 1:
                raise ValueError(f"unsafe weight shard path in index: {name!r}")
            shards.append(path / candidate)
        return shards
    single = path / "model.safetensors"
    if single.is_file():
        return [single]
    raise ValueError("model has neither a safetensors index nor model.safetensors")


def _manifest_files(path: Path, shards: list[Path]) -> list[Path]:
    required = [path / "config.json", path / "tokenizer_config.json", *shards]
    for optional in (
        "generation_config.json",
        "model.safetensors.index.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "merges.txt",
        "vocab.json",
    ):
        candidate = path / optional
        if candidate.is_file():
            required.append(candidate)
    return sorted(set(required))


def _file_records(path: Path, files: list[Path]) -> dict[str, dict[str, int | str]]:
    return {
        str(file.relative_to(path)): {
            "size": file.stat().st_size,
            "sha256": _sha256(file),
        }
        for file in files
    }


def _verify_file_records(path: Path, records: dict) -> None:
    if not isinstance(records, dict) or not records:
        raise ValueError("snapshot manifest has no file integrity records")
    for name, expected in records.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe manifest file path: {name!r}")
        file = path / relative
        if not file.is_file():
            raise ValueError(f"snapshot file missing: {name}")
        if file.stat().st_size != int(expected.get("size", -1)):
            raise ValueError(f"snapshot file size mismatch: {name}")
        if _sha256(file) != expected.get("sha256"):
            raise ValueError(f"snapshot file hash mismatch: {name}")


def _check_snapshot(spec: dict, path: Path) -> dict:
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    required = [
        path / "config.json",
        path / "tokenizer_config.json",
        path / ".om_snapshot.json",
    ]
    missing = [p.name for p in required if not p.is_file()]
    if missing:
        raise ValueError(f"{spec['key']}: missing files: {', '.join(missing)}")

    manifest = json.loads((path / ".om_snapshot.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2:
        raise ValueError(f"{spec['key']}: obsolete snapshot manifest schema")
    if manifest.get("repository") != spec["repository"] or manifest.get("revision") != spec["revision"]:
        raise ValueError(f"{spec['key']}: snapshot provenance mismatch")
    _verify_file_records(path, manifest.get("files"))

    tokenizer_config = json.loads((path / "tokenizer_config.json").read_text(encoding="utf-8"))
    if not tokenizer_config.get("chat_template"):
        raise ValueError(f"{spec['key']}: tokenizer chat_template missing")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Reply with OK."}],
        add_generation_prompt=True,
        tokenize=True,
    )
    if not rendered or not all(isinstance(token, int) for token in rendered):
        raise ValueError(f"{spec['key']}: tokenizer chat template produced no token IDs")
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    if config.model_type != spec["model_type"]:
        raise ValueError(
            f"{spec['key']}: model_type={config.model_type}, expected={spec['model_type']}"
        )

    shards = _weight_shards(path)
    if any(not shard.is_file() for shard in shards):
        raise ValueError(f"{spec['key']}: safetensors shard set incomplete")
    weight_bytes = sum(shard.stat().st_size for shard in shards)
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
    shards = _weight_shards(path)
    manifest = {
        "schema_version": 2,
        "repository": spec["repository"],
        "revision": spec["revision"],
        "files": _file_records(path, _manifest_files(path, shards)),
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
    p = sub.add_parser("experiment-field")
    p.add_argument("name", choices=sorted(EXPERIMENT_FIELDS))
    args = parser.parse_args()

    config = _load_config(args.config)
    specs = _load_specs(args.config)
    if args.command == "experiment-field":
        value = config["experiment"][args.name]
        if isinstance(value, list):
            print(" ".join(map(str, value)))
        elif isinstance(value, bool):
            print("1" if value else "0")
        else:
            print(value)
        return
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
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"[model-abort] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
