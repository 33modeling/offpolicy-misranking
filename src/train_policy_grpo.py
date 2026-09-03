#!/usr/bin/env python3
"""Distributed verifier-reward GRPO policy training.

LoRA is used only as a parameter-efficient representation of the policy update.
The optimization objective is clipped GRPO over online samples from the current
policy. No supervised labels or positive-only filtering enter this path.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist

from artifact_contract import sha256_file
from compact_artifacts import compact_adapter
from data import reward
from rollout import SAMPLING, _lora_targets, chat_ids, load_model, prompt_format
from rollout_contract import eos_ids_of, gen_kwargs, resp_end_index

POLICY_SCHEMA = "offpolicy-rlvr-policy/v1"
CHECKPOINT_SCHEMA = "offpolicy-grpo-checkpoint/v2"
RLVR_METHODS = ("grpo", "dr_grpo", "rloo")


def policy_update_for_objective(objective: str) -> str:
    if objective in {"grpo", "dr_grpo"}:
        return "clipped_policy_gradient"
    if objective == "rloo":
        return "reinforce_leave_one_out"
    raise ValueError(f"unsupported RLVR method: {objective}")


def advantage_normalization_for_objective(objective: str) -> str:
    return {
        "grpo": "group_std",
        "dr_grpo": "none",
        "rloo": "leave_one_out",
    }[objective]


def token_normalization_for_objective(objective: str) -> str:
    return {
        "grpo": "response_length",
        "dr_grpo": "fixed_generation_budget",
        "rloo": "sequence_sum",
    }[objective]


@dataclass(frozen=True)
class GrpoConfig:
    group_size: int = 8
    clip_epsilon: float = 0.2
    learning_rate: float = 1e-5
    epochs_per_batch: int = 2
    max_grad_norm: float = 1.0
    advantage_epsilon: float = 1e-4
    lora_rank: int = 16
    lora_alpha: int = 32
    checkpoint_every: int = 5


def standardized_group_advantages(
    rewards: torch.Tensor, epsilon: float = 1e-4
) -> torch.Tensor:
    """Return the GRPO within-group standardized reward advantages."""
    if rewards.ndim != 1 or rewards.numel() < 2:
        raise ValueError("GRPO requires a one-dimensional reward group with K >= 2")
    if epsilon <= 0:
        raise ValueError("advantage epsilon must be positive")
    centered = rewards.float() - rewards.float().mean()
    return centered / (rewards.float().std(unbiased=False) + epsilon)


def centered_group_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """Return Dr.GRPO centered rewards without per-question std scaling."""
    if rewards.ndim != 1 or rewards.numel() < 2:
        raise ValueError("Dr.GRPO requires a one-dimensional reward group with K >= 2")
    values = rewards.float()
    return values - values.mean()


def rloo_group_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """Use every other online response as an unbiased per-prompt baseline."""
    if rewards.ndim != 1 or rewards.numel() < 2:
        raise ValueError("RLOO requires a one-dimensional reward group with K >= 2")
    values = rewards.float()
    return values - (values.sum() - values) / (values.numel() - 1)


def rloo_loss(
    sequence_logps: list[torch.Tensor], advantages: torch.Tensor
) -> torch.Tensor:
    """Sequence-level REINFORCE loss with a leave-one-out reward baseline."""
    if len(sequence_logps) != advantages.numel() or not sequence_logps:
        raise ValueError("RLOO log-prob groups and advantages must have equal lengths")
    losses = []
    for logps, advantage in zip(sequence_logps, advantages, strict=True):
        if logps.ndim != 1 or logps.numel() == 0:
            raise ValueError("each RLOO response must have non-empty token log-probs")
        advantage = advantage.to(device=logps.device, dtype=logps.dtype)
        losses.append(-advantage * logps.sum())
    return torch.stack(losses).mean()


def clipped_grpo_loss(
    current_logps: list[torch.Tensor],
    old_logps: list[torch.Tensor],
    advantages: torch.Tensor,
    clip_epsilon: float,
    token_normalizer: int | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute a clipped group-policy surrogate for one prompt group.

    Vanilla GRPO averages over each response length. Dr.GRPO instead passes the
    fixed generation budget as ``token_normalizer``, eliminating response-length
    reweighting while keeping gradient scale bounded across batches.
    """
    if not 0 < clip_epsilon < 1:
        raise ValueError("clip epsilon must be in (0, 1)")
    if len(current_logps) != len(old_logps) or len(current_logps) != advantages.numel():
        raise ValueError("log-prob groups and advantages must have equal lengths")
    if not current_logps:
        raise ValueError("GRPO group cannot be empty")
    if token_normalizer is not None and token_normalizer < 1:
        raise ValueError("fixed token normalizer must be positive")

    losses: list[torch.Tensor] = []
    ratios: list[torch.Tensor] = []
    log_ratios: list[torch.Tensor] = []
    approximate_kls: list[torch.Tensor] = []
    for current, old, advantage in zip(
        current_logps, old_logps, advantages, strict=True
    ):
        if current.ndim != 1 or current.shape != old.shape or current.numel() == 0:
            raise ValueError("each response must have matching non-empty token log-probs")
        old = old.to(device=current.device, dtype=current.dtype)
        log_ratio = current - old
        ratio = torch.exp(log_ratio)
        clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
        advantage = advantage.to(device=current.device, dtype=current.dtype)
        surrogate = torch.minimum(ratio * advantage, clipped * advantage)
        losses.append(
            -surrogate.mean()
            if token_normalizer is None
            else -surrogate.sum() / token_normalizer
        )
        ratios.append(ratio.detach())
        log_ratios.append(log_ratio.detach())
        approximate_kls.append((old - current.detach()).mean())

    all_ratios = torch.cat(ratios)
    stats = {
        "clip_fraction": float(
            ((all_ratios < 1.0 - clip_epsilon) | (all_ratios > 1.0 + clip_epsilon))
            .float()
            .mean()
        ),
        "mean_ratio": float(all_ratios.mean()),
        "approx_kl": float(torch.stack(approximate_kls).mean()),
        "max_abs_log_ratio": float(torch.cat(log_ratios).abs().max()),
    }
    return torch.stack(losses).mean(), stats


def _distributed_metric_row(local: torch.Tensor, world_size: int) -> dict[str, float]:
    """Summarize an already all-reduced training-stat vector."""
    if local.numel() != 10 or world_size < 1:
        raise ValueError("invalid distributed training-stat vector")
    token_count = float(local[9])
    if token_count <= 0:
        raise ValueError("distributed training-stat vector has no response tokens")
    return {
        "reward_mean": float(local[0] / local[1]),
        "rank_reward_std_mean": float(local[2] / world_size),
        "loss": float(local[4] / world_size),
        "grad_norm": float(local[5] / world_size),
        "clip_fraction": float(local[6] / token_count),
        "mean_ratio": float(local[7] / token_count),
        "approx_kl": float(local[8] / token_count),
    }


def validate_policy_manifest(
    adapter_dir: Path,
    *,
    target_steps: int | None = None,
    world_size: int | None = None,
    training_objective: str = "grpo",
    verify_hash: bool = True,
    require_complete_hashes: bool = False,
) -> dict:
    """Fail closed unless an adapter used the requested verifier-RL objective."""
    if training_objective not in RLVR_METHODS:
        raise ValueError(f"unsupported RLVR method: {training_objective}")
    manifest_path = adapter_dir / "policy_train.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "schema": POLICY_SCHEMA,
        "training_objective": training_objective,
        "policy_update": policy_update_for_objective(training_objective),
        "reward_source": "verifier",
        "reference_kl_beta": 0.0,
        "supervised_loss": False,
        "positive_only_filter": False,
        "parameterization": "lora",
        "advantage_normalization": advantage_normalization_for_objective(
            training_objective
        ),
        "token_normalization": token_normalization_for_objective(training_objective),
    }
    errors = [
        f"{key}={manifest.get(key)!r}, expected {value!r}"
        for key, value in required.items()
        if manifest.get(key) != value
    ]
    start_step = manifest.get("start_step")
    if not isinstance(start_step, int) or start_step < 0:
        errors.append(f"invalid start_step={start_step!r}")
    parent_fields = (
        "parent_policy",
        "parent_policy_manifest_sha256",
        "parent_adapter_sha256",
        "parent_optimizer_sha256",
    )
    if isinstance(start_step, int) and start_step > 0:
        if not manifest.get("parent_policy"):
            errors.append("resumed policy has no parent_policy")
        for key in parent_fields[1:]:
            value = manifest.get(key)
            if not isinstance(value, str) or len(value) != 64:
                errors.append(f"resumed policy has invalid {key}")
    elif any(manifest.get(key) is not None for key in parent_fields):
        errors.append("base policy unexpectedly records a parent policy")
    if target_steps is not None and manifest.get("completed_steps") != target_steps:
        errors.append(
            f"completed_steps={manifest.get('completed_steps')!r}, expected {target_steps}"
        )
    if world_size is not None and manifest.get("world_size") != world_size:
        errors.append(f"world_size={manifest.get('world_size')!r}, expected {world_size}")
    for name in ("adapter_config.json", "adapter_model.safetensors", "optimizer.pt"):
        path = adapter_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing or empty {name}")
    stats_path = adapter_dir / "grpo_stats.jsonl"
    stats_rows = []
    if not stats_path.is_file() or stats_path.stat().st_size == 0:
        errors.append("missing or empty grpo_stats.jsonl")
    else:
        try:
            stats_rows = [
                json.loads(line)
                for line in stats_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"invalid grpo_stats.jsonl: {exc}")
        if stats_rows:
            completed_steps = manifest.get("completed_steps")
            if isinstance(start_step, int) and isinstance(completed_steps, int):
                expected_steps = list(range(start_step + 1, completed_steps + 1))
                recorded_steps = [row.get("step") for row in stats_rows]
                if recorded_steps != expected_steps:
                    errors.append(
                        "grpo_stats.jsonl step coverage does not match the policy interval"
                    )
            for row in stats_rows:
                if row.get("training_objective") != training_objective:
                    errors.append("grpo_stats.jsonl objective mismatch")
                    break
                if row.get("advantage_normalization") != required[
                    "advantage_normalization"
                ]:
                    errors.append("grpo_stats.jsonl advantage normalization mismatch")
                    break
                if row.get("token_normalization") != required["token_normalization"]:
                    errors.append("grpo_stats.jsonl token normalization mismatch")
                    break
                numeric = (
                    "reward_mean",
                    "rank_reward_std_mean",
                    "loss",
                    "grad_norm",
                    "clip_fraction",
                    "mean_ratio",
                    "approx_kl",
                )
                if any(
                    not isinstance(row.get(key), (int, float))
                    or not math.isfinite(float(row[key]))
                    for key in numeric
                ):
                    errors.append("grpo_stats.jsonl contains missing or non-finite metrics")
                    break
                active = row.get("nonzero_advantage_groups")
                groups = row.get("groups")
                samples = row.get("samples")
                expected_samples = (
                    int(manifest["world_size"])
                    * int(manifest.get("config", {}).get("group_size", 0))
                )
                if (
                    not isinstance(active, int)
                    or isinstance(active, bool)
                    or not 0 <= active <= int(manifest["world_size"])
                    or groups != manifest["world_size"]
                    or samples != expected_samples
                ):
                    errors.append("grpo_stats.jsonl group/sample accounting mismatch")
                    break
        else:
            errors.append("missing GRPO training statistics")
    if errors:
        raise ValueError("invalid GRPO policy artifact: " + "; ".join(errors))
    if verify_hash:
        hash_artifacts = {
            "adapter_sha256": "adapter_model.safetensors",
            "optimizer_sha256": "optimizer.pt",
            "grpo_stats_sha256": "grpo_stats.jsonl",
        }
        hash_errors = []
        for key, name in hash_artifacts.items():
            recorded = manifest.get(key)
            if recorded is None and not require_complete_hashes and key != "adapter_sha256":
                continue
            if not isinstance(recorded, str) or len(recorded) != 64:
                hash_errors.append(f"missing or invalid {key}")
            elif recorded != sha256_file(adapter_dir / name):
                hash_errors.append(f"{name} hash does not match policy_train.json")
        if hash_errors:
            raise ValueError("invalid GRPO policy hashes: " + "; ".join(hash_errors))
    return manifest


def validate_policy_lineage(
    adapter_dir: Path,
    *,
    target_steps: int,
    world_size: int,
    training_objective: str,
    expected_start_step: int,
    expected_parent: Path | None,
    expected_model: Path | None = None,
    expected_seed: int | None = None,
    expected_max_new_tokens: int | None = None,
    expected_prompt_format: str | None = None,
    expected_config: dict | None = None,
    expected_prompts: Path | None = None,
    require_complete_hashes: bool = False,
) -> dict:
    """Validate the exact parent adapter and optimizer for a policy interval."""
    manifest = validate_policy_manifest(
        adapter_dir,
        target_steps=target_steps,
        world_size=world_size,
        training_objective=training_objective,
        require_complete_hashes=require_complete_hashes,
    )
    errors = []
    if manifest.get("start_step") != expected_start_step:
        errors.append(
            f"start_step={manifest.get('start_step')!r}, expected {expected_start_step}"
        )
    if expected_parent is None:
        expected = {
            "parent_policy": None,
            "parent_policy_manifest_sha256": None,
            "parent_adapter_sha256": None,
            "parent_optimizer_sha256": None,
        }
    else:
        parent = expected_parent.resolve()
        validate_policy_manifest(
            parent,
            target_steps=expected_start_step,
            world_size=world_size,
            training_objective=training_objective,
        )
        expected = {
            "parent_policy": str(parent),
            "parent_policy_manifest_sha256": sha256_file(parent / "policy_train.json"),
            "parent_adapter_sha256": sha256_file(
                parent / "adapter_model.safetensors"
            ),
            "parent_optimizer_sha256": sha256_file(parent / "optimizer.pt"),
        }
    errors.extend(
        f"{key}={manifest.get(key)!r}, expected {value!r}"
        for key, value in expected.items()
        if manifest.get(key) != value
    )
    run_expected = {}
    if expected_model is not None:
        run_expected["base_model"] = str(expected_model.resolve())
    if expected_seed is not None:
        run_expected["seed"] = expected_seed
    if expected_max_new_tokens is not None:
        run_expected["max_new_tokens"] = expected_max_new_tokens
    if expected_prompt_format is not None:
        run_expected["prompt_format"] = expected_prompt_format
    if expected_config is not None:
        run_expected["config"] = expected_config
        run_expected["samples_per_step"] = world_size * int(
            expected_config["group_size"]
        )
    errors.extend(
        f"{key}={manifest.get(key)!r}, expected {value!r}"
        for key, value in run_expected.items()
        if manifest.get(key) != value
    )
    if expected_prompts is not None:
        prompt_hash = sha256_file(expected_prompts)
        recorded_hash = manifest.get("prompts_sha256")
        if recorded_hash is None and require_complete_hashes:
            errors.append("prompts_sha256 is missing")
        elif recorded_hash is not None and recorded_hash != prompt_hash:
            errors.append(
                f"prompts_sha256={recorded_hash!r}, expected {prompt_hash!r}"
            )
    if errors:
        raise ValueError("invalid GRPO policy lineage: " + "; ".join(errors))
    return manifest


def _response_logps(model, ids: torch.Tensor, response_start: int) -> torch.Tensor:
    return _response_logps_batch(model, [ids], [response_start], pad_token_id=0)[0]


def _response_logps_batch(
    model,
    sequences: list[torch.Tensor],
    response_starts: list[int],
    *,
    pad_token_id: int,
) -> list[torch.Tensor]:
    """Score variable-length responses together without changing their losses."""
    if not sequences or len(sequences) != len(response_starts):
        raise ValueError("sequences and response starts must be non-empty and aligned")
    if any(sequence.ndim != 1 or sequence.numel() < 2 for sequence in sequences):
        raise ValueError("each generated sequence must contain at least two tokens")
    lengths = [int(sequence.numel()) for sequence in sequences]
    if any(
        start < 1 or start >= length
        for start, length in zip(response_starts, lengths)
    ):
        raise ValueError("response start must identify a non-empty generated suffix")

    device = next(model.parameters()).device
    width = max(lengths)
    batch = torch.full(
        (len(sequences), width),
        int(pad_token_id),
        dtype=sequences[0].dtype,
        device=device,
    )
    attention = torch.zeros_like(batch)
    for row, sequence in enumerate(sequences):
        length = lengths[row]
        batch[row, :length] = sequence.to(device)
        attention[row, :length] = 1

    logits = model(batch, attention_mask=attention).logits[:, :-1].float()
    targets = batch[:, 1:]
    token_logps = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    token_logps = token_logps - logits.logsumexp(dim=-1)
    return [
        token_logps[row, start - 1 : length - 1]
        for row, (start, length) in enumerate(zip(response_starts, lengths))
    ]


def _chunks(size: int, chunk_size: int) -> list[range]:
    if size < 1 or chunk_size < 1:
        raise ValueError("batch and micro-batch sizes must be positive")
    return [
        range(start, min(size, start + chunk_size))
        for start in range(0, size, chunk_size)
    ]


@torch.no_grad()
def _sample_group(model, tokenizer, prompt: dict, config: GrpoConfig, max_new_tokens: int):
    inputs = chat_ids(tokenizer, prompt["question"]).to(next(model.parameters()).device)
    response_start = int(inputs.numel())
    batch = inputs.unsqueeze(0).expand(config.group_size, -1)
    kwargs = gen_kwargs(1.0, SAMPLING["top_p"], max_new_tokens, tokenizer.eos_token_id)
    generated = model.generate(
        batch,
        attention_mask=torch.ones_like(batch),
        use_cache=True,
        **kwargs,
    )
    eos_ids = eos_ids_of(model, tokenizer, pad_id=tokenizer.eos_token_id)
    sequences: list[torch.Tensor] = []
    rewards: list[float] = []
    for sequence in generated:
        end = resp_end_index(sequence, response_start, eos_ids)
        sequence = sequence[:end].detach()
        text = tokenizer.decode(sequence[response_start:], skip_special_tokens=True)
        sequences.append(sequence)
        rewards.append(reward(text, prompt["answer"]))
    return sequences, torch.tensor(rewards, dtype=torch.float32)


def _atomic_json(path: Path, document: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _checkpoint_contract(
    args: argparse.Namespace, config: GrpoConfig, world_size: int
) -> dict:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "training_objective": args.objective,
        "base_model": str(Path(args.model).resolve()),
        "seed": args.seed,
        "world_size": world_size,
        "start_step": args.start_step,
        "target_steps": args.target_steps,
        "prompts_sha256": sha256_file(Path(args.prompts)),
        "max_new_tokens": args.max_new_tokens,
        "prompt_format": prompt_format(),
        "resume_adapter": (
            str(Path(args.resume_adapter).resolve()) if args.resume_adapter else None
        ),
        "resume_optimizer": (
            str(Path(args.resume_optimizer).resolve()) if args.resume_optimizer else None
        ),
        "config": asdict(config),
    }


def _checkpoint_step(path: Path, expected_contract: dict) -> int | None:
    try:
        state = json.loads((path / "checkpoint_state.json").read_text())
        if not isinstance(state, dict):
            return None
        step = int(state["completed_steps"])
        if not all(state.get(key) == value for key, value in expected_contract.items()):
            return None
        for name in (
            "adapter_config.json",
            "adapter_model.safetensors",
            "optimizer.pt",
            "grpo_stats.jsonl",
        ):
            artifact = path / name
            if not artifact.is_file() or artifact.stat().st_size == 0:
                return None
        if state.get("adapter_sha256") != sha256_file(
            path / "adapter_model.safetensors"
        ) or state.get("optimizer_sha256") != sha256_file(
            path / "optimizer.pt"
        ) or state.get("grpo_stats_sha256") != sha256_file(
            path / "grpo_stats.jsonl"
        ):
            return None
        return step
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _save_checkpoint(
    model,
    optimizer,
    out_dir: Path,
    completed_steps: int,
    rank: int,
    contract: dict,
) -> None:
    if rank != 0:
        return
    target = out_dir / f"checkpoint-{completed_steps:06d}"
    if target.exists():
        if target.is_dir() and _checkpoint_step(target, contract) == completed_steps:
            return
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    temporary = out_dir / f".checkpoint-{completed_steps:06d}.tmp"
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    model.save_pretrained(temporary, safe_serialization=True)
    compact_adapter(temporary)
    torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
    stats_source = out_dir / "grpo_stats.jsonl"
    if not stats_source.is_file() or stats_source.stat().st_size == 0:
        raise ValueError("cannot checkpoint without durable GRPO statistics")
    shutil.copy2(stats_source, temporary / "grpo_stats.jsonl")
    state = {
        **contract,
        "completed_steps": completed_steps,
        "adapter_sha256": sha256_file(temporary / "adapter_model.safetensors"),
        "optimizer_sha256": sha256_file(temporary / "optimizer.pt"),
        "grpo_stats_sha256": sha256_file(temporary / "grpo_stats.jsonl"),
    }
    _atomic_json(
        temporary / "checkpoint_state.json",
        state,
    )
    temporary.rename(target)
    checkpoints = sorted(out_dir.glob("checkpoint-*"))
    for stale in checkpoints[:-2]:
        shutil.rmtree(stale)


def _latest_checkpoint(
    out_dir: Path, upper_bound: int, expected_contract: dict
) -> tuple[Path | None, int]:
    candidates: list[tuple[int, Path]] = []
    for path in out_dir.glob("checkpoint-*"):
        step = _checkpoint_step(path, expected_contract)
        if step is not None and 0 < step <= upper_bound:
            candidates.append((step, path))
    candidates.sort(key=lambda item: item[0])
    return (candidates[-1][1], candidates[-1][0]) if candidates else (None, 0)


def _distributed_setup(expected_world_size: int) -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != expected_world_size:
        raise RuntimeError(
            f"GRPO requires exactly {expected_world_size} processes, got {world_size}; "
            "launch with torchrun"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("GRPO training requires CUDA")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    return rank, local_rank, world_size


def train(args: argparse.Namespace) -> None:
    from peft import LoraConfig, PeftModel, get_peft_model
    from torch.nn.parallel import DistributedDataParallel

    rank, local_rank, world_size = _distributed_setup(args.expected_world_size)
    config = GrpoConfig(
        group_size=args.group_size,
        clip_epsilon=args.clip_epsilon,
        learning_rate=args.learning_rate,
        epochs_per_batch=args.epochs_per_batch,
        max_grad_norm=args.max_grad_norm,
        advantage_epsilon=args.advantage_epsilon,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        checkpoint_every=args.checkpoint_every,
    )
    if args.objective not in RLVR_METHODS:
        raise ValueError(f"unsupported RLVR method: {args.objective}")
    if config.group_size < 2 or config.epochs_per_batch < 1 or args.target_steps < 1:
        raise ValueError("group size must be >=2 and epochs/target steps must be positive")
    if not 1 <= args.logprob_micro_batch <= config.group_size:
        raise ValueError("log-prob micro-batch must be in [1, group size]")
    if args.objective == "rloo" and config.epochs_per_batch != 1:
        raise ValueError("canonical sequence-level RLOO requires exactly one epoch per batch")
    if SAMPLING["top_p"] != 1.0:
        raise ValueError("canonical GRPO requires OM_TOP_P=1.0")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    published_error = None
    if (out_dir / "policy_train.json").is_file():
        try:
            validate_policy_lineage(
                out_dir,
                target_steps=args.target_steps,
                world_size=world_size,
                training_objective=args.objective,
                expected_start_step=args.start_step,
                expected_parent=(
                    Path(args.resume_adapter) if args.resume_adapter else None
                ),
                expected_model=Path(args.model),
                expected_seed=args.seed,
                expected_max_new_tokens=args.max_new_tokens,
                expected_prompt_format=prompt_format(),
                expected_config=asdict(config),
                expected_prompts=Path(args.prompts),
                require_complete_hashes=True,
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            published_error = exc
        else:
            if rank == 0:
                print(f"[grpo] validated completed policy: {out_dir}", flush=True)
            dist.destroy_process_group()
            return

    has_parent_adapter = args.resume_adapter is not None
    has_parent_optimizer = args.resume_optimizer is not None
    if args.start_step > 0 and not (has_parent_adapter and has_parent_optimizer):
        raise ValueError("a positive start step requires both parent adapter and optimizer")
    if args.start_step == 0 and (has_parent_adapter or has_parent_optimizer):
        raise ValueError("parent adapter/optimizer require a positive start step")

    checkpoint_contract = _checkpoint_contract(args, config, world_size)
    local_checkpoint, local_step = _latest_checkpoint(
        out_dir, args.target_steps, checkpoint_contract
    )
    if published_error is not None and local_checkpoint is None:
        raise ValueError(
            "published policy is invalid and no complete local checkpoint can repair it: "
            f"{published_error}"
        ) from published_error
    if published_error is not None and rank == 0:
        print(
            f"[grpo-resume] repairing invalid final publication from {local_checkpoint}: "
            f"{published_error}",
            flush=True,
        )
    resume_adapter = local_checkpoint or (Path(args.resume_adapter) if args.resume_adapter else None)
    completed_steps = local_step or args.start_step
    if completed_steps > args.target_steps:
        raise ValueError("resume step must not exceed target steps")
    if local_checkpoint is None and completed_steps:
        previous = validate_policy_manifest(
            resume_adapter,
            target_steps=completed_steps,
            world_size=world_size,
            training_objective=args.objective,
        )
        if previous["training_objective"] != args.objective:
            raise ValueError("resume policy uses a different RLVR method")

    model, tokenizer = load_model(args.model, device=f"cuda:{local_rank}")
    if resume_adapter:
        model = PeftModel.from_pretrained(
            model,
            str(resume_adapter),
            is_trainable=True,
            local_files_only=True,
        )
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                target_modules=_lora_targets(),
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
    if not args.disable_gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model.enable_input_require_grads()
    model.config.use_cache = False
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("GRPO policy has no trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate)
    optimizer_source = (
        local_checkpoint / "optimizer.pt"
        if local_checkpoint
        else (Path(args.resume_optimizer) if args.resume_optimizer else None)
    )
    if optimizer_source:
        optimizer.load_state_dict(torch.load(optimizer_source, map_location="cpu", weights_only=True))
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(torch.device(f"cuda:{local_rank}"))

    ddp = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )
    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))["train"]
    if not prompts:
        raise ValueError("training prompt set is empty")
    stats_path = out_dir / "grpo_stats.jsonl"
    if rank == 0 and local_checkpoint is not None:
        checkpoint_stats = local_checkpoint / "grpo_stats.jsonl"
        temporary = stats_path.with_name(stats_path.name + ".checkpoint.tmp")
        shutil.copy2(checkpoint_stats, temporary)
        temporary.replace(stats_path)
    if rank == 0 and stats_path.exists():
        retained = []
        for line in stats_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                break
            if int(row.get("step", -1)) <= completed_steps:
                retained.append(json.dumps(row, sort_keys=True))
        temporary = stats_path.with_name(stats_path.name + ".tmp")
        temporary.write_text("\n".join(retained) + ("\n" if retained else ""))
        temporary.replace(stats_path)
    if world_size > 1:
        dist.barrier()
    stats_stream = stats_path.open("a", encoding="utf-8") if rank == 0 else None
    initial_step = args.start_step
    parent_hashes = {}
    if rank == 0:
        parent_policy = Path(args.resume_adapter).resolve() if args.resume_adapter else None
        parent_hashes = {
            "parent_policy": str(parent_policy) if parent_policy else None,
            "parent_policy_manifest_sha256": (
                sha256_file(parent_policy / "policy_train.json") if parent_policy else None
            ),
            "parent_adapter_sha256": (
                sha256_file(parent_policy / "adapter_model.safetensors")
                if parent_policy
                else None
            ),
            "parent_optimizer_sha256": (
                sha256_file(Path(args.resume_optimizer)) if args.resume_optimizer else None
            ),
        }

    try:
        for step in range(completed_steps, args.target_steps):
            step_started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats(local_rank)
            sample_seed = (args.seed * 1_000_003 + step * 7_919 + rank * 104_729 + 17) & 0x7FFFFFFF
            random.seed(sample_seed)
            torch.manual_seed(sample_seed)
            prompt_index = (args.seed * 37 + step * world_size + rank) % len(prompts)
            model.eval()
            sequences, rewards = _sample_group(
                model, tokenizer, prompts[prompt_index], config, args.max_new_tokens
            )
            if args.objective == "rloo":
                old_logps = [None] * len(sequences)
                advantages = rloo_group_advantages(rewards)
            else:
                with torch.no_grad():
                    response_start = int(
                        chat_ids(tokenizer, prompts[prompt_index]["question"]).numel()
                    )
                    old_logps = []
                    for chunk in _chunks(len(sequences), args.logprob_micro_batch):
                        indices = list(chunk)
                        old_logps.extend(
                            value.cpu()
                            for value in _response_logps_batch(
                                model,
                                [sequences[index] for index in indices],
                                [response_start] * len(indices),
                                pad_token_id=tokenizer.eos_token_id,
                            )
                        )
                advantages = (
                    standardized_group_advantages(rewards, config.advantage_epsilon)
                    if args.objective == "grpo"
                    else centered_group_advantages(rewards)
                )
            token_normalizer = args.max_new_tokens if args.objective == "dr_grpo" else None

            epoch_stats = None
            loss_value = 0.0
            grad_norm_value = 0.0
            model.train()
            for epoch_index in range(config.epochs_per_batch):
                optimizer.zero_grad(set_to_none=True)
                response_start = int(chat_ids(tokenizer, prompts[prompt_index]["question"]).numel())
                token_count = 0
                clip_count = 0
                ratio_sum = 0.0
                kl_sum = 0.0
                max_abs_log_ratio = 0.0
                epoch_loss = 0.0
                chunks = _chunks(len(sequences), args.logprob_micro_batch)
                for chunk_index, chunk in enumerate(chunks):
                    indices = list(chunk)
                    sync = chunk_index == len(chunks) - 1
                    sync_context = contextlib.nullcontext() if sync else ddp.no_sync()
                    with sync_context:
                        current_logps = _response_logps_batch(
                            ddp,
                            [sequences[index] for index in indices],
                            [response_start] * len(indices),
                            pad_token_id=tokenizer.eos_token_id,
                        )
                        if args.objective == "rloo":
                            chunk_loss = rloo_loss(
                                current_logps, advantages[indices]
                            )
                            chunk_stats = {
                                "clip_fraction": 0.0,
                                "mean_ratio": 1.0,
                                "approx_kl": 0.0,
                                "max_abs_log_ratio": 0.0,
                            }
                        else:
                            chunk_loss, chunk_stats = clipped_grpo_loss(
                                current_logps,
                                [old_logps[index] for index in indices],
                                advantages[indices],
                                config.clip_epsilon,
                                token_normalizer=token_normalizer,
                            )
                        weight = len(indices) / len(sequences)
                        (chunk_loss * weight).backward()
                    count = sum(int(value.numel()) for value in current_logps)
                    token_count += count
                    clip_count += round(chunk_stats["clip_fraction"] * count)
                    ratio_sum += chunk_stats["mean_ratio"] * count
                    max_abs_log_ratio = max(
                        max_abs_log_ratio, chunk_stats["max_abs_log_ratio"]
                    )
                    if args.objective != "rloo":
                        kl_sum += sum(
                            float(
                                (
                                    old_logps[index].to(
                                        device=current.device, dtype=current.dtype
                                    )
                                    - current.detach()
                                ).mean()
                            )
                            * int(current.numel())
                            for index, current in zip(indices, current_logps, strict=True)
                        )
                    epoch_loss += float(chunk_loss.detach()) * weight
                epoch_stats = {
                    "clip_fraction": clip_count / token_count,
                    "mean_ratio": ratio_sum / token_count,
                    "approx_kl": kl_sum / token_count,
                    "max_abs_log_ratio": max_abs_log_ratio,
                }
                ratio_deviation = torch.tensor(
                    [max_abs_log_ratio], device=local_rank
                )
                if world_size > 1:
                    dist.all_reduce(ratio_deviation, op=dist.ReduceOp.MAX)
                if (
                    args.objective != "rloo"
                    and epoch_index == 0
                    and (
                        not math.isfinite(float(ratio_deviation[0]))
                        or float(ratio_deviation[0]) > 5e-3
                    )
                ):
                    raise RuntimeError(
                        "first policy-loss evaluation is not on-policy: "
                        f"max_abs_log_ratio={float(ratio_deviation[0]):.6g}"
                    )
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
                optimizer.step()
                loss_value = epoch_loss
                grad_norm_value = float(grad_norm)

            step_seconds = time.perf_counter() - step_started
            peak_allocated = torch.cuda.max_memory_allocated(local_rank) / 1e9
            peak_reserved = torch.cuda.max_memory_reserved(local_rank) / 1e9

            local = torch.tensor(
                [
                    float(rewards.sum()),
                    float(rewards.numel()),
                    float(rewards.std(unbiased=False)),
                    float(bool(advantages.abs().max() > 0)),
                    loss_value,
                    grad_norm_value,
                    float(clip_count),
                    float(ratio_sum),
                    float(kl_sum),
                    float(token_count),
                ],
                device=local_rank,
            )
            if world_size > 1:
                dist.all_reduce(local)
            if int(local[3]) > 0 and not float(local[5]) > 0:
                raise RuntimeError(
                    "nonzero reward advantages produced a zero or non-finite gradient norm"
                )
            runtime_max = torch.tensor(
                [step_seconds, peak_allocated, peak_reserved], device=local_rank
            )
            if world_size > 1:
                dist.all_reduce(runtime_max, op=dist.ReduceOp.MAX)
            if rank == 0:
                metric_row = _distributed_metric_row(local, world_size)
                row = {
                    "step": step + 1,
                    "nonzero_advantage_groups": int(local[3]),
                    "groups": world_size,
                    "samples": int(local[1]),
                    **metric_row,
                    "max_abs_log_ratio": float(ratio_deviation[0]),
                    "response_tokens": int(local[9]),
                    "step_seconds": float(runtime_max[0]),
                    "response_tokens_per_second": float(local[9] / runtime_max[0]),
                    "gpu_peak_allocated_gb": float(runtime_max[1]),
                    "gpu_peak_reserved_gb": float(runtime_max[2]),
                    "logprob_micro_batch": args.logprob_micro_batch,
                    "training_objective": args.objective,
                    "advantage_normalization": advantage_normalization_for_objective(
                        args.objective
                    ),
                    "token_normalization": token_normalization_for_objective(
                        args.objective
                    ),
                }
                stats_stream.write(json.dumps(row, sort_keys=True) + "\n")
                stats_stream.flush()
                print(
                    f"[{args.objective}] step {step + 1}/{args.target_steps} "
                    f"reward={row['reward_mean']:.3f} active_groups="
                    f"{row['nonzero_advantage_groups']}/{world_size} "
                    f"loss={row['loss']:.3e} grad_norm={row['grad_norm']:.3e} "
                    f"ratio={row['mean_ratio']:.6f} optimizer_step=applied",
                    flush=True,
                )
            if world_size > 1:
                dist.barrier()
            if (step + 1) % config.checkpoint_every == 0:
                _save_checkpoint(
                    model,
                    optimizer,
                    out_dir,
                    step + 1,
                    rank,
                    checkpoint_contract,
                )
            if world_size > 1:
                dist.barrier()

        if rank == 0:
            model.save_pretrained(out_dir, safe_serialization=True)
            compact_adapter(out_dir)
            torch.save(optimizer.state_dict(), out_dir / "optimizer.pt")
            manifest = {
                "schema": POLICY_SCHEMA,
                "training_objective": args.objective,
                "policy_update": policy_update_for_objective(args.objective),
                "reward_source": "verifier",
                "reference_kl_beta": 0.0,
                "supervised_loss": False,
                "positive_only_filter": False,
                "parameterization": "lora",
                "base_model": str(Path(args.model).resolve()),
                "world_size": world_size,
                "start_step": initial_step,
                "completed_steps": args.target_steps,
                "samples_per_step": world_size * config.group_size,
                "max_new_tokens": args.max_new_tokens,
                "prompt_format": prompt_format(),
                "seed": args.seed,
                "config": asdict(config),
                "runtime": {
                    "logprob_micro_batch": args.logprob_micro_batch,
                    "gradient_checkpointing": not args.disable_gradient_checkpointing,
                },
                "advantage_normalization": advantage_normalization_for_objective(
                    args.objective
                ),
                "token_normalization": token_normalization_for_objective(args.objective),
                "adapter_sha256": sha256_file(out_dir / "adapter_model.safetensors"),
                "optimizer_sha256": sha256_file(out_dir / "optimizer.pt"),
                "grpo_stats_sha256": sha256_file(out_dir / "grpo_stats.jsonl"),
                "prompts_sha256": sha256_file(Path(args.prompts)),
                **parent_hashes,
            }
            _atomic_json(out_dir / "policy_train.json", manifest)
            validate_policy_lineage(
                out_dir,
                target_steps=args.target_steps,
                world_size=world_size,
                training_objective=args.objective,
                expected_start_step=args.start_step,
                expected_parent=(
                    Path(args.resume_adapter) if args.resume_adapter else None
                ),
                expected_model=Path(args.model),
                expected_seed=args.seed,
                expected_max_new_tokens=args.max_new_tokens,
                expected_prompt_format=prompt_format(),
                expected_config=asdict(config),
                expected_prompts=Path(args.prompts),
                require_complete_hashes=True,
            )
            for checkpoint in out_dir.glob("checkpoint-*"):
                shutil.rmtree(checkpoint)
            print(f"[grpo] published {out_dir}", flush=True)
        if world_size > 1:
            dist.barrier()
    finally:
        if stats_stream is not None:
            stats_stream.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--objective", choices=RLVR_METHODS, default="grpo")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-steps", type=int, required=True)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--resume-adapter")
    parser.add_argument("--resume-optimizer")
    parser.add_argument("--expected-world-size", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--epochs-per-batch", type=int, default=2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--advantage-epsilon", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--logprob-micro-batch", type=int, default=1)
    parser.add_argument("--disable-gradient-checkpointing", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
