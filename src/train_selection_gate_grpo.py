"""Independent budgeted training driver for the one-shot gate study.

The training loop is derived from train_policy_grpo.py at bf4c7f8.
Losses, sampling, optimizer checks, and artifact validation use its helpers.
Only this driver adds resource-budget termination; the canonical driver is unchanged.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import shutil
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist

from artifact_contract import sha256_file
from compact_artifacts import compact_adapter
from rollout import SAMPLING, _lora_targets, chat_ids, load_model, prompt_format
from train_policy_grpo import (
    LOG_EVERY_STEPS,
    POLICY_SCHEMA,
    RLVR_METHODS,
    GrpoConfig,
    _atomic_json,
    _checkpoint_contract,
    _chunks,
    _distributed_metric_row,
    _distributed_setup,
    _latest_checkpoint,
    _response_logps_batch,
    _sample_group,
    _save_checkpoint,
    advantage_normalization_for_objective,
    centered_group_advantages,
    checked_optimizer_step,
    clipped_grpo_loss,
    policy_update_for_objective,
    rloo_group_advantages,
    rloo_loss,
    standardized_group_advantages,
    token_normalization_for_objective,
    validate_policy_lineage,
    validate_policy_manifest,
)


def train(args: argparse.Namespace) -> None:
    from peft import LoraConfig, PeftModel, get_peft_model
    from torch.nn.parallel import DistributedDataParallel

    from selection_gate_budget import stop_before_step

    budget_deadline = getattr(args, "wall_budget_deadline", None)
    budget_reserve = getattr(args, "budget_save_reserve", 30.)
    stop_before_step(budget_deadline, time.monotonic(), budget_reserve, 0.)

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
            existing_target = args.target_steps
            if budget_deadline is not None:
                previous_manifest = json.loads((out_dir / "policy_train.json").read_text())
                if previous_manifest.get("training_budget", {}).get("requested_target_steps") == args.target_steps:
                    existing_target = previous_manifest["completed_steps"]
            validate_policy_lineage(
                out_dir,
                target_steps=existing_target,
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
        if previous.get("config") is not None:
            current = asdict(config)
            mismatched = {
                key: (previous["config"][key], current.get(key))
                for key in previous["config"]
                if key in current and previous["config"][key] != current[key]
            }
            if mismatched:
                raise ValueError(
                    f"resume policy was trained under a different GRPO config: {mismatched}"
                )
            for key in ("weight_decay",):
                if key not in previous["config"]:
                    print(f"[grpo] parent manifest predates the {key} record; current value {current[key]}", flush=True)

    model, tokenizer = load_model(args.model, device=f"cuda:{local_rank}")
    if resume_adapter:
        model = PeftModel.from_pretrained(
            model,
            str(resume_adapter),
            is_trainable=True,
            local_files_only=True,
        )
    else:
        # LoRA B is zero but A is random: seed it so the same --seed reproduces
        # the same initial adapter (DDP broadcasts rank 0 to the others).
        torch.manual_seed((args.seed * 1_000_003 + 8_675_309) & 0x7FFFFFFF)
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
    # torch's AdamW default weight_decay=0.01 is not part of the registered
    # contract; state it explicitly as zero.
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=config.weight_decay)
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
        # A loaded optimizer state carries the parent's hyper-parameters; the
        # registered config, not the checkpoint, owns them.
        for group in optimizer.param_groups:
            group["lr"] = config.learning_rate
            group["weight_decay"] = config.weight_decay

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

    run_seconds = 0.0
    run_steps = 0
    actual_completed = completed_steps
    last_step_seconds = 0.
    stop_reason = "step_limit"
    try:
        for step in range(completed_steps, args.target_steps):
            if budget_deadline is not None:
                stop = torch.tensor([int(stop_before_step(budget_deadline, time.monotonic(),
                                                         budget_reserve, last_step_seconds)) if rank == 0 else 0],
                                    device=local_rank)
                if world_size > 1:
                    dist.broadcast(stop, src=0)
                if int(stop.item()):
                    stop_reason = "no_block_fits"
                    break
            step_started = time.perf_counter()
            if rank == 0 and (step == completed_steps or (step + 1) % LOG_EVERY_STEPS == 0):
                print(f"[{args.objective}] utc={datetime.now(timezone.utc).isoformat()} "
                      f"step-start={step + 1}/{args.target_steps} phase=rollout", flush=True)
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
                grad_norm = checked_optimizer_step(
                    optimizer, trainable, config.max_grad_norm, epoch_loss,
                    advantages.abs().max() > 0,
                )
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
            last_step_seconds = float(runtime_max[0])
            actual_completed = step + 1
            if rank == 0:
                metric_row = _distributed_metric_row(local, world_size)
                row = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
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
                run_seconds += float(row["step_seconds"])
                run_steps += 1
                if (step + 1) % LOG_EVERY_STEPS == 0 or step + 1 == args.target_steps:
                    remaining = args.target_steps - (step + 1)
                    eta_min = remaining * (run_seconds / run_steps) / 60.0
                    # Every field is in grpo_stats.jsonl; the log keeps what an
                    # operator needs to judge progress at a glance.
                    print(
                        f"[{args.objective}] step {step + 1}/{args.target_steps} "
                        f"reward={row['reward_mean']:.3f} "
                        f"active={row['nonzero_advantage_groups']}/{world_size} "
                        f"loss={row['loss']:.2e} gnorm={row['grad_norm']:.2e} "
                        f"{row['step_seconds']:.0f}s/step eta={eta_min:.0f}min "
                        f"peakGB={row['gpu_peak_allocated_gb']:.1f}",
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

        budget_record = {"requested_target_steps": args.target_steps, "completed_steps": actual_completed,
                         "stop_reason": stop_reason, "deadline_monotonic": budget_deadline,
                         "save_reserve_seconds": budget_reserve}
        if budget_deadline is not None and actual_completed == args.start_step:
            if rank == 0:
                _atomic_json(out_dir / "budget_stop.json", {**budget_record, "use_parent_policy": True})
                print("[grpo-budget] no complete update fits; retained the parent policy", flush=True)
            if world_size > 1:
                dist.barrier()
            return

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
                "completed_steps": actual_completed,
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
            if budget_deadline is not None:
                manifest["training_budget"] = budget_record
            _atomic_json(out_dir / "policy_train.json", manifest)
            validate_policy_lineage(
                out_dir,
                target_steps=actual_completed,
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
            if budget_deadline is not None:
                _atomic_json(out_dir / "budget_stop.json", {**budget_record, "use_parent_policy": False})
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
    parser.add_argument("--wall-budget-deadline", type=float)
    parser.add_argument("--budget-save-reserve", type=float, default=30.)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
