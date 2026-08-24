#!/usr/bin/env python3
"""Reject mixed or unexpected 27B runs before publishing v4 results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

FIXED_CONFIG = {
    "behavior_k": 8,
    "fresh_k": 32,
    "val_k": 8,
    "micro_group": 4,
    "hybrid_prompts": 64,
    "k_cell": 8,
    "drift": 100,
    "max_new_tokens": 512,
    "proj_dim": 4096,
    "grad_layers": 4,
    "clip_cap": 10.0,
    "temperature": 1.0,
    "topk_frac": 0.10,
    "radius_mode": "gaussian",
    "top_p": 1.0,
    "thinking": "off",
    "attn": "eager",
    "gen_batch": "4",
    "lora_targets": "all-linear",
    "skip_hybrid": "1",
    "linear_attention_backend": "fla",
    "fla_core_version": "0.5.2",
    "n_val": 100,
}


def _config_digest(config: dict) -> str:
    payload = {key: value for key, value in config.items() if key != "digest"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_run(
    run: Path,
    expected_git: str,
    expected_model_hash: str,
    seed: int,
    dataset: str,
) -> None:
    path = run / "run_config.json"
    try:
        config = json.loads(path.read_text())
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"invalid run config: {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise TypeError(f"invalid run config object: {path}")

    expected = {
        **FIXED_CONFIG,
        "git": expected_git,
        "git_status": "",
        "git_diff_sha256": hashlib.sha256(b"").hexdigest(),
        "model_config_sha256": expected_model_hash,
        "seed": seed,
        "dataset": dataset,
        "n_train": 512 if dataset == "gsm8k" else 400,
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if config.get("digest") != _config_digest(config):
        mismatches["digest"] = (config.get("digest"), "recomputed digest")
    if mismatches:
        details = ", ".join(
            f"{key}={actual!r} expected {wanted!r}"
            for key, (actual, wanted) in sorted(mismatches.items())
        )
        raise ValueError(f"unexpected 27B provenance in {run.name}: {details}")


def validate_matrix(
    runs_root: Path, expected_git: str, expected_model_hash: str
) -> list[Path]:
    runs: list[Path] = []
    for seed in range(5):
        for dataset, suffix in (
            ("gsm8k", ""),
            ("math500", "-math500"),
        ):
            run = runs_root / f"v4-27b-s{seed}{suffix}"
            validate_run(run, expected_git, expected_model_hash, seed, dataset)
            runs.append(run)
    return runs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs_root", type=Path)
    parser.add_argument("--expected-git", required=True)
    parser.add_argument("--expected-model-hash", required=True)
    parser.add_argument("--single-run", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dataset", choices=("gsm8k", "math500"))
    args = parser.parse_args()
    try:
        if args.single_run is not None:
            if args.seed is None or args.dataset is None:
                parser.error("--single-run requires --seed and --dataset")
            validate_run(
                args.single_run,
                args.expected_git,
                args.expected_model_hash,
                args.seed,
                args.dataset,
            )
            runs = [args.single_run]
        else:
            runs = validate_matrix(
                args.runs_root, args.expected_git, args.expected_model_hash
            )
    except (TypeError, ValueError) as exc:
        print(f"[v4-27b-provenance-abort] {exc}")
        return 1
    print(
        f"[v4-27b-provenance] {len(runs)} runs, "
        f"commit={args.expected_git[:12]}, FLA=0.5.2"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
