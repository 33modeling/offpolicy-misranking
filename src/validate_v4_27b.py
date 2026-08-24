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


def validate_matrix(
    runs_root: Path, expected_git: str, expected_model_hash: str
) -> list[Path]:
    runs: list[Path] = []
    empty_diff = hashlib.sha256(b"").hexdigest()
    for seed in range(5):
        for dataset, suffix, n_train in (
            ("gsm8k", "", 512),
            ("math500", "-math500", 400),
        ):
            run = runs_root / f"v4-27b-s{seed}{suffix}"
            path = run / "run_config.json"
            try:
                config = json.loads(path.read_text())
            except (OSError, ValueError, TypeError) as exc:
                raise ValueError(f"invalid run config: {path}: {exc}") from exc

            expected = {
                **FIXED_CONFIG,
                "git": expected_git,
                "git_status": "",
                "git_diff_sha256": empty_diff,
                "model_config_sha256": expected_model_hash,
                "seed": seed,
                "dataset": dataset,
                "n_train": n_train,
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
            runs.append(run)
    return runs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs_root", type=Path)
    parser.add_argument("--expected-git", required=True)
    parser.add_argument("--expected-model-hash", required=True)
    args = parser.parse_args()
    try:
        runs = validate_matrix(
            args.runs_root, args.expected_git, args.expected_model_hash
        )
    except ValueError as exc:
        print(f"[v4-27b-provenance-abort] {exc}")
        return 1
    print(
        f"[v4-27b-provenance] {len(runs)} runs, "
        f"commit={args.expected_git[:12]}, FLA=0.5.2"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
