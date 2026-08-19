"""Build and validate a model-specific hard-slice prompt pool.

Build:
    python3 src/make_hard_pool.py RUN OUT [LO=0.0] [HI=1.0] \
        --model MODEL --dataset DATASET --expected-k 8 --seed 0

Validate before a downstream run:
    python3 src/make_hard_pool.py --validate OUT --model MODEL --dataset DATASET

The JSONL remains directly consumable by ``OM_POOL_FILE``.  A required
``OUT.manifest.json`` sidecar binds it to the source prompts, behavior model,
sampling count, and selection thresholds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path


SCHEMA = "offpolicy-hard-pool/v1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_path(pool: Path) -> Path:
    return pool.with_name(pool.name + ".manifest.json")


def model_identity(model: Path) -> dict[str, str | None]:
    config = model / "config.json"
    if not config.is_file():
        raise ValueError(f"model config not found: {config}")
    return {
        "model": str(model),
        "model_resolved": str(model.resolve()),
        "model_config_sha256": sha256_file(config),
    }


def _rollout_files(run: Path) -> list[Path]:
    shards = sorted(run.glob("rollouts_behavior_train*.jsonl"))
    merged = run / "rollouts_behavior_train.jsonl"
    if merged in shards:
        return [merged]
    return shards


def _rollout_manifest_info(run: Path) -> tuple[int | None, dict[str, str]]:
    values: set[int] = set()
    contracts: set[str] = set()
    hashes: dict[str, str] = {}
    for path in sorted(run.glob("rollouts_behavior_train*.manifest.json")):
        doc = json.loads(path.read_text())
        value = doc.get("k")
        if value is not None:
            values.add(int(value))
        contract = {
            key: doc.get(key)
            for key in (
                "explicit_kwargs",
                "eos_token_ids",
                "model_name_or_path",
                "contract",
            )
        }
        contracts.add(json.dumps(contract, sort_keys=True))
        hashes[path.name] = sha256_file(path)
    if len(values) > 1:
        raise ValueError(f"rollout manifests disagree on K: {sorted(values)}")
    if len(contracts) > 1:
        raise ValueError("rollout manifests disagree on the generation contract")
    return (next(iter(values)) if values else None), hashes


def build_pool(
    run: Path,
    out: Path,
    lo: float,
    hi: float,
    *,
    model: Path,
    dataset: str,
    expected_k: int | None,
    seed: int,
) -> dict:
    if not 0.0 <= lo < hi <= 1.0:
        raise ValueError(f"invalid open pass-rate interval: ({lo}, {hi})")
    prompt_path = run / "prompts.json"
    prompt_doc = json.loads(prompt_path.read_text())
    prompts = prompt_doc["train"]
    shards = _rollout_files(run)
    if not shards:
        raise ValueError(f"rollout files not found: {run}/rollouts_behavior_train*.jsonl")

    recorded_k, rollout_manifest_hashes = _rollout_manifest_info(run)
    if expected_k is None:
        expected_k = recorded_k
    elif recorded_k is not None and recorded_k != expected_k:
        raise ValueError(
            f"requested K={expected_k}, but rollout manifest records K={recorded_k}"
        )
    if expected_k is None or expected_k < 1:
        raise ValueError("expected K is unknown; pass --expected-k or retain rollout manifests")

    acc: dict[int, dict[int, float]] = defaultdict(dict)
    for shard in shards:
        for lineno, line in enumerate(shard.open(), 1):
            row = json.loads(line)
            if "rollout_idx" not in row:
                raise ValueError(f"{shard}:{lineno}: rollout_idx is required")
            prompt_idx = int(row["prompt_idx"])
            rollout_idx = int(row["rollout_idx"])
            if not 0 <= prompt_idx < len(prompts):
                raise ValueError(f"{shard}:{lineno}: unexpected prompt_idx={prompt_idx}")
            if rollout_idx in acc[prompt_idx]:
                raise ValueError(
                    f"duplicate rollout key: prompt_idx={prompt_idx}, rollout_idx={rollout_idx}"
                )
            reward = float(row["reward"])
            if not math.isfinite(reward):
                raise ValueError(f"{shard}:{lineno}: non-finite reward")
            acc[prompt_idx][rollout_idx] = reward

    expected_rollouts = set(range(expected_k))
    for prompt_idx in range(len(prompts)):
        got = set(acc.get(prompt_idx, {}))
        if got != expected_rollouts:
            missing = sorted(expected_rollouts - got)
            extra = sorted(got - expected_rollouts)
            raise ValueError(
                f"prompt_idx={prompt_idx}: expected rollout_idx 0..{expected_k - 1}; "
                f"missing={missing[:8]}, extra={extra[:8]}"
            )

    rows: list[dict] = []
    hist: dict[float, int] = defaultdict(int)
    for prompt_idx, item in enumerate(prompts):
        rewards = [acc[prompt_idx][i] for i in range(expected_k)]
        successes = sum(rewards)
        rate = successes / expected_k
        hist[round(rate, 3)] += 1
        if lo < rate < hi:
            row = dict(item)
            question = str(item.get("question", ""))
            row["_prescreen"] = {
                "source_prompt_idx": prompt_idx,
                "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
                "pass_rate": rate,
                "successes": successes,
                "rollout_count": expected_k,
            }
            rows.append(row)

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + f".tmp.{os.getpid()}")
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(out)

    source_manifest = run / "manifest.json"
    metadata = {
        "schema": SCHEMA,
        "dataset": dataset,
        **model_identity(model),
        "source_run": str(run.resolve()),
        "source_run_manifest_sha256": (
            sha256_file(source_manifest) if source_manifest.is_file() else None
        ),
        "source_rollout_manifest_sha256": rollout_manifest_hashes,
        "source_prompts_sha256": sha256_file(prompt_path),
        "source_prompt_count": len(prompts),
        "behavior_k": expected_k,
        "seed": seed,
        "selection": {"pass_rate_gt": lo, "pass_rate_lt": hi},
        "selected_count": len(rows),
        "saturated": not rows,
        "pool_sha256": sha256_file(out),
    }
    sidecar = manifest_path(out)
    sidecar_tmp = sidecar.with_name(sidecar.name + f".tmp.{os.getpid()}")
    sidecar_tmp.write_text(json.dumps(metadata, indent=1, ensure_ascii=False))
    sidecar_tmp.replace(sidecar)
    print("pass-rate distribution:", dict(sorted(hist.items())))
    print(f"hard-slice: {len(rows)}/{len(prompts)} -> {out}")
    print(f"provenance: {sidecar}")
    if not rows:
        print("[warn] hard-slice is empty: all behavior pass rates are at an interval boundary")
    elif len(rows) < 620:
        print(f"[warn] pool has {len(rows)} rows < 620 (512 train + 100 val + margin)")
    return metadata


def validate_pool(pool: Path, *, model: Path, dataset: str) -> dict:
    sidecar = manifest_path(pool)
    if not pool.is_file() or not sidecar.is_file():
        raise ValueError(f"pool or provenance sidecar missing: {pool}, {sidecar}")
    metadata = json.loads(sidecar.read_text())
    identity = model_identity(model)
    expected = {
        "schema": SCHEMA,
        "dataset": dataset,
        "model_resolved": identity["model_resolved"],
        "model_config_sha256": identity["model_config_sha256"],
        "pool_sha256": sha256_file(pool),
    }
    mismatches = {
        key: {"expected": value, "recorded": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    row_count = sum(1 for line in pool.open() if line.strip())
    if metadata.get("selected_count") != row_count:
        mismatches["selected_count"] = {
            "expected": row_count,
            "recorded": metadata.get("selected_count"),
        }
    if bool(metadata.get("saturated")) != (row_count == 0):
        mismatches["saturated"] = {
            "expected": row_count == 0,
            "recorded": metadata.get("saturated"),
        }
    if mismatches:
        raise ValueError(f"hard-pool provenance mismatch: {json.dumps(mismatches)}")
    print(
        f"hard-pool valid: dataset={dataset}, rows={row_count}, "
        f"K={metadata.get('behavior_k')}, model={model.name}"
    )
    return metadata


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("lo", nargs="?", type=float, default=0.0)
    parser.add_argument("hi", nargs="?", type=float, default=1.0)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--expected-k", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    if not args.validate and args.run is None:
        parser.error("RUN is required when building a pool")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.validate:
            validate_pool(args.out, model=args.model, dataset=args.dataset)
        else:
            build_pool(
                args.run,
                args.out,
                args.lo,
                args.hi,
                model=args.model,
                dataset=args.dataset,
                expected_k=args.expected_k,
                seed=args.seed,
            )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"[abort] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
