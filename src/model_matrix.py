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
DEFAULT_CONFIG = ROOT / "configs" / "generalization_logic.json"
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
    "chat_template.jinja",
]

PINNED_OFFICIAL_FILES = {
    (
        "allenai/Olmo-3-1025-7B",
        "a81bae42db3975be1671e27b9c9a56da1a9f980f",
    ): {
        "config.json": {
            "size": 1620,
            "git_blob_sha1": "5e1778fbc278e8f47217ba485e9a075689207f0f",
        },
        "generation_config.json": {
            "size": 69,
            "git_blob_sha1": "33b71a9c3ddf78cfa1c6721826775ae02d06d64d",
        },
        "merges.txt": {
            "size": 916646,
            "git_blob_sha1": "354558edcdbd64ca7abd407b8be3d5d09d39d781",
        },
        "model-00001-of-00003.safetensors": {
            "size": 4969984976,
            "sha256": "0490d6668e613a29b23367e3a7aa9cc6aced3d162694445bb969ed7622b3c4e2",
        },
        "model-00002-of-00003.safetensors": {
            "size": 4981161496,
            "sha256": "e127ea479fb6e208fe9d48d23b11212b5722f4873f6eef9c009b7a855866c641",
        },
        "model-00003-of-00003.safetensors": {
            "size": 4644917240,
            "sha256": "f3ddff10052ffe5de5c6b4cad45c422c0d898acc6beb21b1b8531244adfb3c70",
        },
        "model.safetensors.index.json": {
            "size": 29630,
            "git_blob_sha1": "421a80b181a130ccbc579a328fb349d8792a32ce",
        },
        "special_tokens_map.json": {
            "size": 207,
            "git_blob_sha1": "48f174b441a37b588e40d794c437adad1624a311",
        },
        "tokenizer.json": {
            "size": 7137177,
            "git_blob_sha1": "5fe172127988c3709a49d8d2ce20e11bb266cd57",
        },
        "tokenizer_config.json": {
            "size": 4308,
            "git_blob_sha1": "5599723dac37d9f0b7e496de66d15e0a762babe9",
        },
        "vocab.json": {
            "size": 1611056,
            "git_blob_sha1": "51135344eec01a62fc4deaca39c72ac08f5b9709",
        },
    }
}

EXPERIMENT_FIELDS = {
    "policy_method",
    "datasets",
    "seeds",
    "drifts",
    "n_train",
    "n_train_by_dataset",
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

GRPO_FIELDS = {
    "world_size",
    "group_size",
    "clip_epsilon",
    "learning_rate",
    "reference_kl_beta",
    "epochs_per_batch",
    "max_grad_norm",
    "advantage_epsilon",
    "lora_rank",
    "lora_alpha",
}

RUNTIME_DEFAULTS = {
    "generation_batch": 4,
    "gradient_micro_batch": 1,
    "logprob_micro_batch": 1,
    "gradient_checkpointing": True,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {path.stat().st_size}\0".encode())
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
    dataset_sizes = experiment["n_train_by_dataset"]
    if not isinstance(dataset_sizes, dict) or set(dataset_sizes) != set(
        experiment["datasets"]
    ):
        raise ValueError(
            "experiment.n_train_by_dataset must cover exactly experiment.datasets"
        )
    for dataset, value in dataset_sizes.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"experiment.n_train_by_dataset.{dataset} must be a positive integer"
            )
    if experiment["policy_method"] not in {"grpo", "dr_grpo", "rloo"}:
        raise ValueError(
            "experiment.policy_method must be 'grpo', 'dr_grpo', or 'rloo'"
        )
    if experiment["drifts"][0] != 0 or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in experiment["drifts"]
    ):
        raise ValueError("experiment.drifts must begin with integer positive control 0")
    if experiment["drifts"] != sorted(experiment["drifts"]):
        raise ValueError("experiment.drifts must be strictly increasing")
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
    fresh_groups = experiment["fresh_k"] // experiment["micro_group"]
    if fresh_groups < 8 or fresh_groups % 4:
        raise ValueError(
            "fresh_k/micro_group must produce at least eight groups divisible by four"
        )
    if experiment["n_val"] < 8 or experiment["n_val"] % 4:
        raise ValueError("n_val must be at least eight and divisible by four")
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
    grpo = experiment.get("grpo")
    if not isinstance(grpo, dict):
        raise TypeError("experiment.grpo must be a JSON object")
    missing_grpo = sorted(GRPO_FIELDS - set(grpo))
    if missing_grpo:
        raise ValueError(f"domain-transfer GRPO fields missing: {missing_grpo}")
    if (
        experiment["policy_method"] == "rloo"
        and grpo["epochs_per_batch"] != 1
    ):
        raise ValueError("RLOO requires grpo.epochs_per_batch=1")
    for key in (
        "world_size",
        "group_size",
        "epochs_per_batch",
        "lora_rank",
        "lora_alpha",
    ):
        if isinstance(grpo[key], bool) or not isinstance(grpo[key], int) or grpo[key] <= 0:
            raise ValueError(f"experiment.grpo.{key} must be a positive integer")
    if grpo["group_size"] < 2:
        raise ValueError("experiment.grpo.group_size must be at least two")
    for key in (
        "clip_epsilon",
        "learning_rate",
        "max_grad_norm",
        "advantage_epsilon",
    ):
        if (
            isinstance(grpo[key], bool)
            or not isinstance(grpo[key], (int, float))
            or grpo[key] <= 0
        ):
            raise ValueError(f"experiment.grpo.{key} must be positive numeric")
    if not 0 < grpo["clip_epsilon"] < 1:
        raise ValueError("experiment.grpo.clip_epsilon must be in (0, 1)")
    if grpo["reference_kl_beta"] != 0.0:
        raise ValueError("current verifier-reward trainer requires reference_kl_beta=0.0")
    runtime = experiment.get("runtime", {})
    if not isinstance(runtime, dict):
        raise TypeError("experiment.runtime must be a JSON object")
    unknown_runtime = sorted(set(runtime) - set(RUNTIME_DEFAULTS))
    if unknown_runtime:
        raise ValueError(f"unsupported experiment.runtime fields: {unknown_runtime}")
    for key in ("generation_batch", "gradient_micro_batch", "logprob_micro_batch"):
        value = runtime.get(key, RUNTIME_DEFAULTS[key])
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"experiment.runtime.{key} must be a positive integer")
    checkpointing = runtime.get(
        "gradient_checkpointing", RUNTIME_DEFAULTS["gradient_checkpointing"]
    )
    if not isinstance(checkpointing, bool):
        raise TypeError("experiment.runtime.gradient_checkpointing must be boolean")
    if runtime.get("logprob_micro_batch", 1) > grpo["group_size"]:
        raise ValueError(
            "experiment.runtime.logprob_micro_batch cannot exceed grpo.group_size"
        )
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
        formatter = spec.get("prompt_format", "tokenizer_chat")
        if formatter not in {"tokenizer_chat", "olmo_rlzero", "verifiable_completion"}:
            raise ValueError(f"{key}: unsupported prompt_format={formatter!r}")
        official_files = spec.get("official_files")
        if official_files is not None:
            if not isinstance(official_files, dict) or not official_files:
                raise TypeError(f"{key}: official_files must be a non-empty object")
            for name, record in official_files.items():
                relative = Path(name)
                if (
                    not isinstance(name, str)
                    or relative.is_absolute()
                    or len(relative.parts) != 1
                    or not isinstance(record, dict)
                ):
                    raise ValueError(f"{key}: invalid official file record: {name!r}")
                size = record.get("size")
                hashes = {
                    field: record.get(field)
                    for field, length in (("sha256", 64), ("git_blob_sha1", 40))
                    if isinstance(record.get(field), str)
                    and re.fullmatch(rf"[0-9a-f]{{{length}}}", record[field])
                }
                if (
                    isinstance(size, bool)
                    or not isinstance(size, int)
                    or size <= 0
                    or len(hashes) != 1
                    or len(record) != 2
                ):
                    raise ValueError(f"{key}: invalid official file record: {name!r}")
    return specs


def _snapshot_path(spec: dict, models_dir: Path) -> Path:
    return models_dir / spec["local_directory"]


def _require_runtime(spec: dict) -> None:
    if spec.get("model_type") != "olmo3":
        return
    from packaging.version import Version
    from transformers import __version__ as transformers_version

    if Version(transformers_version) < Version("4.57.0"):
        raise ValueError(
            f"{spec['key']}: OLMo-3 requires transformers>=4.57.0, "
            f"found {transformers_version}; update the shared venv before GPU allocation"
        )


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
        "chat_template.jinja",
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

    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if spec.get("prompt_format", "tokenizer_chat") == "tokenizer_chat":
        if not getattr(tokenizer, "chat_template", None):
            raise ValueError(f"{spec['key']}: tokenizer chat_template missing")
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
        "prompt_format": spec.get("prompt_format", "tokenizer_chat"),
    }


def _write_manifest(
    spec: dict,
    path: Path,
    records: dict[str, dict[str, int | str]] | None = None,
) -> None:
    shards = _weight_shards(path)
    manifest = {
        "schema_version": 2,
        "repository": spec["repository"],
        "revision": spec["revision"],
        "files": records or _file_records(path, _manifest_files(path, shards)),
    }
    target = path / ".om_snapshot.json"
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _seal_local_snapshot(spec: dict, path: Path) -> dict:
    """Seal a Hub ``local_dir`` download without making a network request."""
    shards = _weight_shards(path)
    files = _manifest_files(path, shards)
    official_files = spec.get("official_files") or PINNED_OFFICIAL_FILES.get(
        (spec["repository"], spec["revision"])
    )
    if official_files:
        records: dict[str, dict[str, int | str]] = {}
        for file in files:
            relative = str(file.relative_to(path))
            expected = official_files.get(relative)
            if expected is None:
                raise ValueError(
                    f"{spec['key']}: file is not registered for the pinned model: {relative}"
                )
            if not file.is_file():
                raise ValueError(f"{spec['key']}: model file missing: {relative}")
            size = file.stat().st_size
            if size != expected["size"]:
                raise ValueError(f"{spec['key']}: model file size mismatch: {relative}")
            if "sha256" in expected:
                sha256 = _sha256(file)
                valid = sha256 == expected["sha256"]
            else:
                valid = _git_blob_sha1(file) == expected["git_blob_sha1"]
                sha256 = _sha256(file)
            if not valid:
                raise ValueError(f"{spec['key']}: model file hash mismatch: {relative}")
            records[relative] = {"size": size, "sha256": sha256}

        manifest = path / ".om_snapshot.json"
        _write_manifest(spec, path, records)
        try:
            return _check_snapshot(spec, path)
        except Exception:
            manifest.unlink(missing_ok=True)
            raise

    metadata_root = path / ".cache" / "huggingface" / "download"
    missing: list[str] = []
    wrong: list[str] = []
    corrupt: list[str] = []
    for file in files:
        relative = file.relative_to(path)
        metadata = metadata_root / f"{relative}.metadata"
        try:
            lines = metadata.read_text(encoding="utf-8").splitlines()
            revision, etag = lines[0], lines[1]
        except (OSError, IndexError):
            missing.append(str(relative))
            continue
        if revision != spec["revision"]:
            wrong.append(f"{relative}={revision}")
        if re.fullmatch(r"[0-9a-f]{64}", etag):
            valid_content = _sha256(file) == etag
        elif re.fullmatch(r"[0-9a-f]{40}", etag):
            valid_content = _git_blob_sha1(file) == etag
        else:
            valid_content = False
        if not valid_content:
            corrupt.append(str(relative))
    if missing or wrong or corrupt:
        detail = []
        if missing:
            detail.append("missing metadata: " + ", ".join(missing[:5]))
        if wrong:
            detail.append("revision mismatch: " + ", ".join(wrong[:5]))
        if corrupt:
            detail.append("content hash mismatch: " + ", ".join(corrupt[:5]))
        raise ValueError(
            f"{spec['key']}: cannot prove local Hub revision ({'; '.join(detail)})"
        )

    manifest = path / ".om_snapshot.json"
    _write_manifest(spec, path)
    try:
        return _check_snapshot(spec, path)
    except Exception:
        manifest.unlink(missing_ok=True)
        raise


def _download(spec: dict, models_dir: Path) -> dict:
    from huggingface_hub import snapshot_download

    destination = _snapshot_path(spec, models_dir)
    try:
        return _check_snapshot(spec, destination)
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    if destination.is_dir():
        try:
            return _seal_local_snapshot(spec, destination)
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
    parser.add_argument(
        "--snapshot-path",
        type=Path,
        help="exact local path override for a single check/seal/field operation",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("check", "download", "seal"):
        p = sub.add_parser(command)
        p.add_argument("models", nargs="*")
    p = sub.add_parser("field")
    p.add_argument("model")
    p.add_argument(
        "name", choices=["path", "lora_targets", "repository", "revision", "prompt_format"]
    )
    p = sub.add_parser("experiment-field")
    p.add_argument("name", choices=sorted(EXPERIMENT_FIELDS))
    p = sub.add_parser("dataset-n-train")
    p.add_argument("dataset")
    sub.add_parser("list-models")
    p = sub.add_parser("grpo-field")
    p.add_argument(
        "name",
        choices=sorted(GRPO_FIELDS),
    )
    p = sub.add_parser("runtime-field")
    p.add_argument("name", choices=sorted(RUNTIME_DEFAULTS))
    args = parser.parse_args()

    config = _load_config(args.config)
    specs = _load_specs(args.config)
    if args.command == "list-models":
        print("\n".join(specs))
        return
    if args.command == "experiment-field":
        value = config["experiment"][args.name]
        if isinstance(value, list):
            print(" ".join(map(str, value)))
        elif isinstance(value, bool):
            print("1" if value else "0")
        else:
            print(value)
        return
    if args.command == "dataset-n-train":
        sizes = config["experiment"]["n_train_by_dataset"]
        if args.dataset not in sizes:
            raise ValueError(f"dataset is outside matrix: {args.dataset}")
        print(sizes[args.dataset])
        return
    if args.command == "grpo-field":
        print(config["experiment"]["grpo"][args.name])
        return
    if args.command == "runtime-field":
        value = config["experiment"].get("runtime", {}).get(
            args.name, RUNTIME_DEFAULTS[args.name]
        )
        if isinstance(value, bool):
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
            print(args.snapshot_path or _snapshot_path(spec, args.models_dir))
        elif args.name == "lora_targets":
            print(",".join(spec["lora_targets"]))
        elif args.name == "prompt_format":
            print(spec.get("prompt_format", "tokenizer_chat"))
        else:
            print(spec[args.name])
        return

    if args.snapshot_path is not None and len(models) != 1:
        parser.error("--snapshot-path requires exactly one selected model")
    if args.snapshot_path is not None and args.command == "download":
        parser.error("--snapshot-path is supported by check/seal, not download")
    for key in models:
        _require_runtime(specs[key])
    for key in models:
        spec = specs[key]
        snapshot_path = args.snapshot_path or _snapshot_path(spec, args.models_dir)
        if args.command == "download":
            result = _download(spec, args.models_dir)
        elif args.command == "seal":
            result = _seal_local_snapshot(spec, snapshot_path)
        else:
            result = _check_snapshot(spec, snapshot_path)
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
