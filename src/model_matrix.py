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
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    # Qwen3.5/3.8 shards are named "model.safetensors-00001-of-00004.safetensors",
    # which "model-*.safetensors" never matched: prepare downloaded config/tokenizer
    # only and check failed with "safetensors shard set incomplete".
    "*.safetensors",
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
    },
    # Qwen3.5-9B: lets a manually uploaded snapshot (no .cache/huggingface metadata)
    # be sealed offline on the compute node, exactly like OLMo. Values are the Hub
    # tree at the pinned revision (LFS sha256 for big files, git blob sha1 otherwise).
    (
        "Qwen/Qwen3.5-9B",
        "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    ): {
        "chat_template.jinja": {
            "size": 7756,
            "git_blob_sha1": "a585dec894e63da457d9440ec6aa7caa16d20860",
        },
        "config.json": {
            "size": 3126,
            "git_blob_sha1": "273ce437e01baf96a07cd9eb3d5f48bac8d7c657",
        },
        "merges.txt": {
            "size": 3353259,
            "git_blob_sha1": "a494e019ca1502219fd0128658b979e5f05ae8e8",
        },
        "model.safetensors-00001-of-00004.safetensors": {
            "size": 5276436216,
            "sha256": "db6f444b43d318c92f360a13a25561a6a65b10c0631b8ed305a426dbaa6c380e",
        },
        "model.safetensors-00002-of-00004.safetensors": {
            "size": 5335161512,
            "sha256": "31c7d7e2dd5d207840b31cc59083c8f4c4718959149e0358c0364052bb9a0330",
        },
        "model.safetensors-00003-of-00004.safetensors": {
            "size": 5368717440,
            "sha256": "7ec36ba3a4176a44c3c0876ad80c56a2f70c84bf008d82e9501df642f17dadec",
        },
        "model.safetensors-00004-of-00004.safetensors": {
            "size": 3325995712,
            "sha256": "b62b0c4cd7e44edee103ee8f4fe225f246d5e768e07bfd5f25b63a8aa1fdd0c6",
        },
        "model.safetensors.index.json": {
            "size": 79657,
            "git_blob_sha1": "e4c1cb7dba5096b43b9d92bc781aba5e3aa8acd8",
        },
        "preprocessor_config.json": {
            "size": 390,
            "git_blob_sha1": "2ea84a437d448ff71b08df68fdd949d5cc4ebb64",
        },
        "tokenizer.json": {
            "size": 12807982,
            "sha256": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
        },
        "tokenizer_config.json": {
            "size": 16710,
            "git_blob_sha1": "eda48d3e75a8e59a8479ee4ec8b37f76e711d9c1",
        },
        "video_preprocessor_config.json": {
            "size": 385,
            "git_blob_sha1": "3ba673a5ad7d4d13f54155ecd38b2a94a6dac8fe",
        },
        "vocab.json": {
            "size": 6722759,
            "git_blob_sha1": "0aa0ce0658d60ac4a5d609f4eadb0e8e43514176",
        },
    },
    # Qwen3.8-27B (same purpose; shards use the classic model-NNNNN-of-NNNNN naming).
    (
        "Qwen/Qwen3.8-27B",
        "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    ): {
        "chat_template.jinja": {
            "size": 8952,
            "git_blob_sha1": "c0c686f9c38d70d179fb7b5f5aa7530bc913dda3",
        },
        "config.json": {
            "size": 4312,
            "git_blob_sha1": "706cebd746c4b6f2b1d1f892630867acfdfd3df8",
        },
        "generation_config.json": {
            "size": 202,
            "git_blob_sha1": "023756cfadf88e5bf69eefeee3e172f38c448d64",
        },
        "merges.txt": {
            "size": 3353259,
            "git_blob_sha1": "a494e019ca1502219fd0128658b979e5f05ae8e8",
        },
        "model-00001-of-00018.safetensors": {
            "size": 3966730552,
            "sha256": "ba0ce20aae489ad196733da5064bcdf159a1fe84f53336648196e1ebb7751b1c",
        },
        "model-00002-of-00018.safetensors": {
            "size": 3043080328,
            "sha256": "06a148c01bfbe3faa14a5f184a7ff29a706f7ae1c8b2705d2058e26d17a001fb",
        },
        "model-00003-of-00018.safetensors": {
            "size": 2542796952,
            "sha256": "2e1bf62cbcd406eaa64b60d10353e1f0ef4039d0976e56f05cabe953454f9968",
        },
        "model-00004-of-00018.safetensors": {
            "size": 3988973152,
            "sha256": "511e34063187882659753c4d93f3859f93c019fd438d8813071921c81d9a3f1a",
        },
        "model-00005-of-00018.safetensors": {
            "size": 2099339864,
            "sha256": "635cb53446dc74f219740fc59e18b774f877b803b9722e289ca62575a6efa701",
        },
        "model-00006-of-00018.safetensors": {
            "size": 3979553696,
            "sha256": "0bc5214fac607f0e6cc92eec3789d4b8559410ef9fce66621ba8158e8410dae0",
        },
        "model-00007-of-00018.safetensors": {
            "size": 2108759344,
            "sha256": "80b0c49033e9a0d5762562aa12f4acdb7f54da586f3d0110f28c48d91cf07892",
        },
        "model-00008-of-00018.safetensors": {
            "size": 3979553696,
            "sha256": "7192c5b66185d3592927daabee1cc19e6f6e0ce75988ee20e824b624765fda79",
        },
        "model-00009-of-00018.safetensors": {
            "size": 2108759344,
            "sha256": "af3c48cc37af44f3db6ae0579baf019180d48d9c527caa0a1f03ff85813a56d8",
        },
        "model-00010-of-00018.safetensors": {
            "size": 3979553696,
            "sha256": "163490a76f3bea3a40855b7efc04ce6d27afaf1a34f0bbde495b9491f76457c9",
        },
        "model-00011-of-00018.safetensors": {
            "size": 2108759344,
            "sha256": "5f3ae1b948aeee39da77aec558e8236cd65fe4d7cb7686a76bb007acc563c6d8",
        },
        "model-00012-of-00018.safetensors": {
            "size": 3979553696,
            "sha256": "a3de1c7114677a8f5ac5c4892c90e8238ea5c1e2038c80e757dfc87c3902ca55",
        },
        "model-00013-of-00018.safetensors": {
            "size": 2108759344,
            "sha256": "06ab79a41f74c9c5cb734816feb0c7fc364104b227165ee7391231e1155aa02a",
        },
        "model-00014-of-00018.safetensors": {
            "size": 3979553696,
            "sha256": "4138ed94603065ba884bbcadedb04d7718bb40117e85e6f5c6fc5b9c05b7a85b",
        },
        "model-00015-of-00018.safetensors": {
            "size": 2108759344,
            "sha256": "69224e27b9de4e7dbf6fc936c6eaae08447bda3b80a6c31a871ab451173afd22",
        },
        "model-00016-of-00018.safetensors": {
            "size": 3979564040,
            "sha256": "73cb9a1089fb6155cb648609478d6633be8a5c7d9ca5a05bc8925ce8a553cefe",
        },
        "model-00017-of-00018.safetensors": {
            "size": 2108759344,
            "sha256": "beb51f01056142ac4984bd800507b0dd0fd18de57f8e9ef6ea41d1a3598983a8",
        },
        "model-00018-of-00018.safetensors": {
            "size": 3392197344,
            "sha256": "1d3479509e21494658f9b64d317f5ea8e55c4025d28c702d6c4d0b356ce8ea06",
        },
        "model.safetensors.index.json": {
            "size": 112216,
            "git_blob_sha1": "da35e3c564457dface7d138f0b6cac284ff8958c",
        },
        "preprocessor_config.json": {
            "size": 390,
            "git_blob_sha1": "2ea84a437d448ff71b08df68fdd949d5cc4ebb64",
        },
        "tokenizer.json": {
            "size": 12809320,
            "sha256": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
        },
        "tokenizer_config.json": {
            "size": 17928,
            "git_blob_sha1": "5de744b3fca2129d7186979ae47c06be33903243",
        },
        "video_preprocessor_config.json": {
            "size": 385,
            "git_blob_sha1": "3ba673a5ad7d4d13f54155ecd38b2a94a6dac8fe",
        },
        "vocab.json": {
            "size": 6722759,
            "git_blob_sha1": "0aa0ce0658d60ac4a5d609f4eadb0e8e43514176",
        },
    },
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
    """Fail before GPU allocation when the venv cannot load this model family."""
    model_type = spec.get("model_type")
    if model_type not in {"olmo3", "qwen3_5"}:
        return
    from packaging.version import Version
    from transformers import __version__ as transformers_version

    if model_type == "olmo3" and Version(transformers_version) < Version("4.57.0"):
        raise ValueError(
            f"{spec['key']}: OLMo-3 requires transformers>=4.57.0, "
            f"found {transformers_version}; update the shared venv before GPU allocation"
        )
    if model_type == "qwen3_5":
        # rollout.load_model falls back to AutoModelForMultimodalLM and
        # check_27b_fla.py imports transformers.models.qwen3_5; both exist only in
        # transformers 5.x. requirements.txt's >=4.57 floor is the OLMo minimum,
        # not evidence of Qwen3.5 support (QWEN38_27B_RUNBOOK.md).
        try:
            from transformers import AutoModelForMultimodalLM  # noqa: F401
            from transformers.models.qwen3_5 import modeling_qwen3_5  # noqa: F401
        except ImportError as exc:
            raise ValueError(
                f"{spec['key']}: Qwen3.5 needs transformers>=5 with the qwen3_5 "
                f"multimodal classes (found {transformers_version}): {exc}"
            ) from exc
        try:
            from importlib.metadata import version

            fla = version("fla-core")
        except Exception as exc:  # noqa: BLE001 - any metadata failure means FLA is absent
            raise ValueError(
                f"{spec['key']}: fla-core (flash-linear-attention 0.5.2) is not "
                f"installed; GatedDeltaNet layers need it: {exc}"
            ) from exc
        if fla != "0.5.2":
            print(f"[model] WARNING {spec['key']}: fla-core {fla} != registered 0.5.2",
                  file=sys.stderr)


def _safetensors_tensor_names(path: Path) -> list[str] | None:
    """Tensor names from a safetensors header (8-byte LE length + JSON)."""
    import struct

    try:
        with path.open("rb") as stream:
            (length,) = struct.unpack("<Q", stream.read(8))
            if not 0 < length < 200_000_000:
                return None
            header = json.loads(stream.read(length).decode("utf-8"))
    except (OSError, ValueError, struct.error):
        return None
    return [name for name in header if name != "__metadata__"]


def _weight_shards(path: Path) -> list[Path]:
    index_path = path / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        names = sorted(set(index.get("weight_map", {}).values()))
        shards = []
        for name in names:
            candidate = Path(str(name))
            if candidate.is_absolute() or len(candidate.parts) != 1:
                raise ValueError(f"unsafe weight shard path in index: {name!r}")
            shards.append(path / candidate)
        if shards and all(shard.is_file() for shard in shards):
            return shards
    single = path / "model.safetensors"
    if single.is_file():
        return [single]
    # An uploaded snapshot may carry no index, or one naming files that were
    # renamed on upload. The shards themselves are what matters here; the
    # tokenizer/config checks and the loader validate the rest.
    present = sorted(
        candidate for candidate in path.glob("*.safetensors") if candidate.is_file()
    )
    if present:
        return present
    raise ValueError(
        f"no *.safetensors in {path} (and no usable model.safetensors.index.json)"
    )


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
        if name == "__provenance__":
            raise ValueError("unverified snapshot records are not admissible")
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


def validate_snapshot_provenance(spec: dict, path: Path) -> dict:
    """Reject old trust-local manifests, changed files and wrong pinned weights."""
    manifest = json.loads((path / ".om_snapshot.json").read_text())
    if (manifest.get("schema_version") != 2
            or manifest.get("repository") != spec["repository"]
            or manifest.get("revision") != spec["revision"]):
        raise ValueError("snapshot provenance mismatch")
    records = manifest.get("files")
    if not isinstance(records, dict) or "__provenance__" in records:
        raise ValueError("unverified snapshot cannot enter a registered matrix")
    shards = _weight_shards(path)
    names = {str(file.relative_to(path)) for file in _manifest_files(path, shards)}
    if set(records) != names:
        raise ValueError("snapshot manifest does not bind the complete current file set")
    _verify_file_records(path, records)
    official = spec.get("official_files") or PINNED_OFFICIAL_FILES.get(
        (spec["repository"], spec["revision"])
    )
    if official:
        expected_shards = {name for name in official if name.endswith(".safetensors")}
        if {file.name for file in shards} != expected_shards:
            raise ValueError("snapshot shard set differs from pinned revision")
        for name in names:
            expected = official.get(name)
            actual = records[name]
            if expected is None or actual["size"] != expected["size"]:
                raise ValueError(f"unregistered or wrong-size pinned file: {name}")
            if "sha256" in expected:
                valid = actual["sha256"] == expected["sha256"]
            else:
                valid = _git_blob_sha1(path / name) == expected["git_blob_sha1"]
            if not valid:
                raise ValueError(f"pinned model file hash mismatch: {name}")
    elif manifest.get("provenance") != "pinned-hub-revision":
        raise ValueError("legacy manifest must be resealed from exact Hub metadata")
    return manifest


def _check_snapshot(spec: dict, path: Path) -> dict:
    """Strict pinned identity plus tokenizer and architecture compatibility."""
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    validate_snapshot_provenance(spec, path)

    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if spec.get("prompt_format", "tokenizer_chat") == "tokenizer_chat":
        if not getattr(tokenizer, "chat_template", None):
            raise ValueError(f"{spec['key']}: tokenizer chat_template missing")
        # transformers 5 returns a BatchEncoding (dict) for tokenize=True, so render
        # text first and tokenize separately — the same path rollout.chat_ids uses.
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Reply with OK."}],
            add_generation_prompt=True,
            tokenize=False,
        )
        rendered = tokenizer(text, add_special_tokens=False).input_ids
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
        if config.model_type == "qwen3_5":
            from transformers import AutoModelForMultimodalLM

            model = AutoModelForMultimodalLM.from_config(config)
        else:
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
        "provenance": "pinned-hub-revision",
        "repository": spec["repository"],
        "revision": spec["revision"],
        "files": records or _file_records(path, _manifest_files(path, shards)),
    }
    target = path / ".om_snapshot.json"
    temporary = target.with_name(f"{target.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def _seal_local_snapshot(spec: dict, path: Path) -> dict:
    """Seal a Hub ``local_dir`` download without making a network request."""
    shards = _weight_shards(path)
    files = _manifest_files(path, shards)
    official_files = spec.get("official_files") or PINNED_OFFICIAL_FILES.get(
        (spec["repository"], spec["revision"])
    )
    if official_files:
        records: dict[str, dict[str, int | str]] = {}
        try:
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
                    raise ValueError(
                        f"{spec['key']}: model file size mismatch: {relative} "
                        f"({size} B on disk, official {expected['size']} B)"
                    )
                if "sha256" in expected:
                    sha256 = _sha256(file)
                    valid = sha256 == expected["sha256"]
                else:
                    valid = _git_blob_sha1(file) == expected["git_blob_sha1"]
                    sha256 = _sha256(file)
                if not valid:
                    raise ValueError(f"{spec['key']}: model file hash mismatch: {relative}")
                records[relative] = {"size": size, "sha256": sha256}
        except ValueError:
            raise

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
