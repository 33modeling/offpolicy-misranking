"""Remove redundant run artifacts only after canonical outputs validate."""

from __future__ import annotations

import json
import math
from pathlib import Path

from artifact_contract import sha256_file, validate_generation_contract

REDUNDANT_ADAPTER_FILES = (
    "README.md",
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
)


def _atomic_json(path: Path, document: dict) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    temporary.replace(path)


def compact_adapter(adapter: Path) -> list[Path]:
    """Keep PEFT reload inputs, never a duplicate base-model tokenizer."""
    config = adapter / "adapter_config.json"
    weights = (adapter / "adapter_model.safetensors", adapter / "adapter_model.bin")
    if not config.is_file() or config.stat().st_size == 0 or not any(
        path.is_file() and path.stat().st_size > 0 for path in weights
    ):
        raise ValueError(f"adapter is incomplete: {adapter}")
    removed = []
    for name in REDUNDANT_ADAPTER_FILES:
        path = adapter / name
        if path.is_file():
            path.unlink()
            removed.append(path)
    return removed


def compact_rollout_shards(run: Path, prefix: str) -> list[Path]:
    """Publish one validated manifest and remove redundant rollout shards."""
    artifact = run / f"{prefix}.jsonl"
    manifest = run / f"{prefix}.manifest.json"
    if not artifact.is_file():
        raise ValueError(f"merged rollout is missing: {artifact}")

    published_manifest = False
    if not manifest.exists():
        validation = validate_generation_contract(run, (prefix,))
        shard_manifests = [run / name for name in validation["manifests"]]
        if not shard_manifests or any(
            ".shard" not in path.name for path in shard_manifests
        ):
            raise ValueError(f"{prefix}: shard manifests are missing")
        documents = [
            json.loads(path.read_text(encoding="utf-8")) for path in shard_manifests
        ]
        varying = {"artifact_file", "artifact_sha256", "idx_offset", "n_prompts"}
        canonical = {
            key: value for key, value in documents[0].items() if key not in varying
        }
        for document in documents[1:]:
            comparable = {
                key: value for key, value in document.items() if key not in varying
            }
            if comparable != canonical:
                raise ValueError(f"{prefix}: shard manifests disagree")

        prompts = json.loads((run / "prompts.json").read_text(encoding="utf-8"))
        split = "val" if prefix.endswith("_val") else "train"
        canonical.update(
            {
                "artifact_file": artifact.name,
                "artifact_sha256": sha256_file(artifact),
                "idx_offset": 0,
                "n_prompts": len(prompts[split]),
            }
        )
        _atomic_json(manifest, canonical)
        published_manifest = True

    try:
        validate_generation_contract(run, (prefix,))
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        if published_manifest:
            manifest.unlink(missing_ok=True)
        raise
    removed = []
    for path in sorted(run.glob(f"{prefix}.shard*")):
        if path.is_file():
            path.unlink()
            removed.append(path)
    return removed


def _logsumexp(values: list[float]) -> float:
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def merge_divergence_shards(run: Path) -> Path | None:
    """Create exact aggregate divergence statistics before shard deletion."""
    shards = sorted(run.glob("divergence_stats.shard*.json"))
    if not shards:
        return None
    documents = [json.loads(path.read_text(encoding="utf-8")) for path in shards]
    total_tokens = sum(int(document.get("tokens", 0)) for document in documents)
    total_rollouts = sum(int(document.get("rollouts", 0)) for document in documents)
    if total_tokens <= 0 or total_rollouts <= 0:
        raise ValueError("divergence shards have no token or rollout observations")

    result: dict[str, float | int] = {
        "token_kl_beta_pi": sum(
            float(document["token_kl_beta_pi"]) * int(document["tokens"])
            for document in documents
        )
        / total_tokens,
        "rollouts": total_rollouts,
        "tokens": total_tokens,
    }
    shared_keys = set.intersection(*(set(document) for document in documents))
    for key in sorted(key for key in shared_keys if key.startswith("clipfrac_")):
        result[key] = sum(
            float(document[key]) * int(document["rollouts"])
            for document in documents
        ) / total_rollouts

    logsum_keys = ("traj_logw_logsumexp", "traj_logw2_logsumexp")
    if all(all(key in document for key in logsum_keys) for document in documents):
        log_w = _logsumexp(
            [float(document["traj_logw_logsumexp"]) for document in documents]
        )
        log_w2 = _logsumexp(
            [float(document["traj_logw2_logsumexp"]) for document in documents]
        )
        result["traj_logw_logsumexp"] = log_w
        result["traj_logw2_logsumexp"] = log_w2
        result["traj_ess_frac_g11"] = (
            math.exp(2 * log_w - log_w2) / total_rollouts
        )

    target = run / "divergence_stats.json"
    _atomic_json(target, result)
    return target


def compact_analysis_shards(run: Path) -> list[Path]:
    """Delete score/gradient shards after canonical replacements exist."""
    groups = (
        (
            (
                "scores_offpolicy.shard*.json",
                "score_protocol.shard*.json",
                "divergence_stats.shard*.json",
            ),
            (
                run / "scores_offpolicy.json",
                run / "score_protocol.json",
                run / "divergence_stats.json",
            ),
        ),
        (("oracle_micro_groups.shard*.pt",), (run / "oracle_micro_groups.pt",)),
    )
    removed = []
    for patterns, required in groups:
        shards = [path for pattern in patterns for path in sorted(run.glob(pattern))]
        if not shards:
            continue
        if not all(path.is_file() and path.stat().st_size > 0 for path in required):
            raise ValueError(
                "refusing to delete shards without canonical artifacts: "
                + ", ".join(str(path) for path in required)
            )
        for path in shards:
            path.unlink()
            removed.append(path)
    return removed
