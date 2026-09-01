#!/usr/bin/env python3
"""Qualify non-degenerate verifier signal from the exact RL-Zero runtime."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import torch

from artifact_contract import sha256_file
from data import load_prompts
from qualify_domain_data import SPECS, _adopt_official_upload

PROMPT_FORMAT = {
    "math500": "olmo_rlzero_math",
    "mbpp": "olmo_rlzero_code",
}
ROOT = Path(__file__).resolve().parents[1]


def git_head() -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def dataset_manifest(data_root: Path, dataset: str) -> Path:
    data_file, _ = _adopt_official_upload(dataset, data_root)
    os.environ[SPECS[dataset]["env"]] = str(data_file.parent)
    path = data_file.parent / "dataset_manifest.json"
    if not path.is_file():
        raise ValueError(f"dataset manifest missing: {path}")
    return path


def fingerprint(args: argparse.Namespace) -> dict:
    model = args.model.resolve()
    manifest = dataset_manifest(args.data_root.resolve(), args.dataset)
    return {
        "schema": "offpolicy-rlzero-signal/v1",
        "git": git_head(),
        "model": str(model),
        "model_snapshot_sha256": sha256_file(model / ".om_snapshot.json"),
        "dataset": args.dataset,
        "dataset_manifest_sha256": sha256_file(manifest),
        "prompt_format": PROMPT_FORMAT[args.dataset],
        "prompt_count": args.prompt_count,
        "group_size": args.group_size,
        "max_new_tokens": args.max_new_tokens,
        "generation_batch": args.generation_batch,
        "gradient_micro_batch": args.gradient_micro_batch,
        "grad_layers": args.grad_layers,
        "seed": args.seed,
        "temperature": 1.0,
        "top_p": 1.0,
    }


def cached_report(path: Path, expected: dict) -> dict | None:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        artifact = path.parent / report["rollout_file"]
        manifest = path.parent / report["rollout_manifest_file"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return None
    if report.get("fingerprint") != expected or report.get("status") != "qualified":
        return None
    if not artifact.is_file() or sha256_file(artifact) != report.get("rollout_sha256"):
        return None
    if not manifest.is_file() or sha256_file(manifest) != report.get(
        "rollout_manifest_sha256"
    ):
        return None
    try:
        actual_stats = read_rewards(
            artifact, expected["prompt_count"], expected["group_size"]
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if report.get("stats") != actual_stats:
        return None
    probe = report.get("runtime_probe")
    if (
        not isinstance(probe, dict)
        or probe.get("gradient_micro_batch") != expected["gradient_micro_batch"]
        or any(
            not isinstance(probe.get(key), (int, float))
            or not math.isfinite(float(probe[key]))
            for key in (
                "projected_gradient_norm",
                "gpu_peak_allocated_gb",
                "gpu_peak_reserved_gb",
            )
        )
    ):
        return None
    return report


def read_rewards(path: Path, prompt_count: int, group_size: int) -> dict:
    grouped = {index: [] for index in range(prompt_count)}
    for line_number, line in enumerate(path.open(encoding="utf-8"), 1):
        row = json.loads(line)
        prompt = int(row["prompt_idx"])
        if prompt not in grouped:
            raise ValueError(f"rollout row {line_number} has unexpected prompt_idx={prompt}")
        grouped[prompt].append(float(row["reward"]))
    if any(len(values) != group_size for values in grouped.values()):
        counts = {key: len(values) for key, values in grouped.items()}
        raise ValueError(f"rollout group coverage mismatch: {counts}")
    rewards = [value for values in grouped.values() for value in values]
    correct = sum(value > 0.5 for value in rewards)
    mixed = sum(
        any(value > 0.5 for value in values)
        and any(value <= 0.5 for value in values)
        for values in grouped.values()
    )
    return {
        "samples": len(rewards),
        "correct": correct,
        "incorrect": len(rewards) - correct,
        "reward_mean": sum(rewards) / len(rewards),
        "mixed_prompt_groups": mixed,
        "per_prompt_correct": [sum(value > 0.5 for value in grouped[i]) for i in grouped],
    }


def qualify(args: argparse.Namespace, expected: dict) -> dict:
    os.environ["OM_PROMPT_FORMAT"] = PROMPT_FORMAT[args.dataset]
    os.environ["OM_GEN_BATCH"] = str(args.generation_batch)
    os.environ["OM_TOP_P"] = "1.0"
    os.environ["OM_ONLINE"] = "0"
    from rollout import collect_rollouts, load_model

    prompts = load_prompts(
        args.dataset, args.prompt_count, 1, seed=0
    )["train"]
    if len(prompts) != args.prompt_count:
        raise ValueError("qualified prompt loader returned the wrong count")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    stem = f".{args.output.stem}.{os.getpid()}"
    rollout = args.output.parent / f"{stem}.jsonl"
    model, tokenizer = load_model(str(args.model.resolve()), device="cuda")
    collect_rollouts(
        model,
        tokenizer,
        prompts,
        args.group_size,
        args.max_new_tokens,
        1.0,
        rollout,
        batch_prompts=1,
        sampling_seed_base=args.seed,
    )
    from experiment import read_rollouts
    from grads import ProjectionSpec, grad_params, loo_advantages, prompt_gradient

    groups = read_rollouts(rollout)
    probe_rows = next(
        (
            rows
            for rows in groups.values()
            if len({float(row["reward"]) for row in rows}) > 1
        ),
        next(iter(groups.values())),
    )
    rewards = torch.tensor([float(row["reward"]) for row in probe_rows])
    advantages = loo_advantages(rewards)
    weights = [
        torch.full(
            (int(row["input_ids"].numel()) - int(row["resp_start"]),),
            float(advantage),
        )
        for row, advantage in zip(probe_rows, advantages, strict=True)
    ]
    model.config.use_cache = False
    params = grad_params(model, args.grad_layers)
    torch.cuda.reset_peak_memory_stats()
    projected = prompt_gradient(
        model,
        params,
        probe_rows,
        weights,
        ProjectionSpec(dim=256),
        micro_batch=args.gradient_micro_batch,
    )
    runtime_probe = {
        "gradient_micro_batch": args.gradient_micro_batch,
        "projected_gradient_norm": float(projected.norm()),
        "gpu_peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        "gpu_peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
    }
    if not all(math.isfinite(float(value)) for value in runtime_probe.values()):
        raise ValueError(f"non-finite H100 runtime probe: {runtime_probe}")
    del model, tokenizer

    stats = read_rewards(rollout, args.prompt_count, args.group_size)
    if stats["correct"] == 0:
        raise ValueError(f"{args.dataset}: verifier produced no positive reward: {stats}")
    if stats["incorrect"] == 0:
        raise ValueError(f"{args.dataset}: verifier produced no negative reward: {stats}")
    if stats["mixed_prompt_groups"] == 0:
        raise ValueError(f"{args.dataset}: no within-prompt reward variation: {stats}")

    final_rollout = args.output.with_suffix(".rollouts.jsonl")
    rollout.replace(final_rollout)
    temporary_manifest = rollout.parent / f"{rollout.stem}.manifest.json"
    final_manifest = args.output.with_suffix(".rollouts.manifest.json")
    temporary_manifest.replace(final_manifest)
    report = {
        "status": "qualified",
        "fingerprint": expected,
        "rollout_file": final_rollout.name,
        "rollout_sha256": sha256_file(final_rollout),
        "rollout_manifest_file": final_manifest.name,
        "rollout_manifest_sha256": sha256_file(final_manifest),
        "stats": stats,
        "runtime_probe": runtime_probe,
    }
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", choices=sorted(PROMPT_FORMAT), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-count", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--generation-batch", type=int, default=4)
    parser.add_argument("--gradient-micro-batch", type=int, default=1)
    parser.add_argument("--grad-layers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=314159)
    args = parser.parse_args()
    if min(
        args.prompt_count,
        args.group_size,
        args.max_new_tokens,
        args.gradient_micro_batch,
        args.grad_layers,
    ) <= 0:
        parser.error("prompt, batch, layer, and max-token values must be positive")
    try:
        expected = fingerprint(args)
        report = cached_report(args.output, expected)
        if report is None:
            report = qualify(args, expected)
        print(
            f"[signal-qualified] {args.dataset}: "
            f"correct={report['stats']['correct']}/{report['stats']['samples']} "
            f"mixed={report['stats']['mixed_prompt_groups']} "
            f"gen_batch={args.generation_batch} "
            f"grad_batch={report['runtime_probe']['gradient_micro_batch']} "
            f"peak={report['runtime_probe']['gpu_peak_allocated_gb']:.1f}GB "
            f"report={args.output}"
        )
        return 0
    except (KeyError, OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"[signal-abort] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
