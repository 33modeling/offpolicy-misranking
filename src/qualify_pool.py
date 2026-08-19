"""Independently requalify a prescreened pool from the main-run rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

from select_rules import topk_count


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def qualify(run: Path, pool: Path, topk_frac: float = 0.10) -> dict:
    prompts_path = run / "prompts.json"
    behavior_path = run / "rollouts_behavior_train.jsonl"
    config_path = run / "run_config.json"
    pool_manifest_path = pool.with_name(pool.name + ".manifest.json")
    required_paths = (
        pool, pool_manifest_path, prompts_path, behavior_path, config_path
    )
    if not all(path.is_file() for path in required_paths):
        raise ValueError(
            "pool, pool manifest, prompts, merged behavior rollouts, and run_config are required"
        )
    config = json.loads(config_path.read_text())
    if config.get("pool_sha256") != sha256_file(pool):
        raise ValueError("run_config does not match the selected pool")
    if config.get("pool_manifest_sha256") != sha256_file(pool_manifest_path):
        raise ValueError("run_config does not match the pool provenance sidecar")
    pool_manifest = json.loads(pool_manifest_path.read_text())
    if pool_manifest.get("dataset") != config.get("dataset"):
        raise ValueError("pool and main run datasets do not match")
    if int(pool_manifest["seed"]) == int(config["seed"]):
        raise ValueError("prescreen and main-run rollout seeds must be different")
    expected_k = int(config["behavior_k"])
    prompts = json.loads(prompts_path.read_text())["train"]
    rewards: dict[int, dict[int, float]] = defaultdict(dict)
    for lineno, line in enumerate(behavior_path.open(), 1):
        row = json.loads(line)
        prompt_idx = int(row["prompt_idx"])
        rollout_idx = int(row["rollout_idx"])
        reward = float(row["reward"])
        if not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
            raise ValueError(f"line {lineno}: reward must be finite and in [0,1]")
        if rollout_idx in rewards[prompt_idx]:
            raise ValueError(f"line {lineno}: duplicate prompt/rollout key")
        rewards[prompt_idx][rollout_idx] = reward

    expected = set(range(expected_k))
    rates = []
    for prompt_idx in range(len(prompts)):
        got = set(rewards.get(prompt_idx, {}))
        if got != expected:
            raise ValueError(
                f"prompt {prompt_idx}: expected rollout_idx 0..{expected_k - 1}, got {sorted(got)}"
            )
        rates.append(sum(rewards[prompt_idx].values()) / expected_k)
    unexpected = sorted(set(rewards) - set(range(len(prompts))))
    if unexpected:
        raise ValueError(f"unexpected prompt indices: {unexpected[:8]}")

    mixed = sum(0.0 < rate < 1.0 for rate in rates)
    required = topk_count(len(prompts), topk_frac)
    result = {
        "schema": "offpolicy-pool-qualification/v1",
        "pool": str(pool.resolve()),
        "pool_sha256": sha256_file(pool),
        "pool_manifest_sha256": sha256_file(pool_manifest_path),
        "prescreen_seed": int(pool_manifest["seed"]),
        "qualification_seed": int(config["seed"]),
        "behavior_rollouts_sha256": sha256_file(behavior_path),
        "prompt_count": len(prompts),
        "behavior_k": expected_k,
        "topk_frac": topk_frac,
        "required_mixed_prompts": required,
        "mixed_prompts": mixed,
        "mixed_fraction": mixed / len(prompts) if prompts else 0.0,
        "passed": mixed >= required,
    }
    output = run / "pool_qualification.json"
    tmp = output.with_name(output.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(result, indent=1))
    tmp.replace(output)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("pool", type=Path)
    parser.add_argument("--topk-frac", type=float, default=0.10)
    args = parser.parse_args(argv)
    try:
        result = qualify(args.run, args.pool, args.topk_frac)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"[qualification-abort] {exc}", file=sys.stderr)
        return 1
    print(
        "[pool qualification] independent mixed prompts "
        f"{result['mixed_prompts']}/{result['prompt_count']} "
        f"(required {result['required_mixed_prompts']})"
    )
    if not result["passed"]:
        print("[qualification-abort] prescreen liveness did not replicate", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
