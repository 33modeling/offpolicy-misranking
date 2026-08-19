"""Rescore a completed legacy run under the corrected validation protocol.

Expensive behavior/fresh rollouts and stored oracle micro-gradients are reused
only after their generation contract passes. Off-policy scores are recomputed on
GPU; oracle scores, split halves, and the report are then regenerated from stored
gradients.

    python3 src/rescore_completed_run.py RUN_DIR
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

from artifact_contract import validate_generation_contract
from experiment import stage_score
from recompute_oracle_scores import recompute


def clean_code_head() -> str:
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "src", "scripts"], text=True
    ).strip()
    if status and os.environ.get("OM_ALLOW_DIRTY", "0") != "1":
        raise ValueError("src/scripts worktree is dirty; commit the correction before rescoring")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def score_args(run: Path, config: dict) -> Namespace:
    drift = int(config.get("drift", config.get("drift_steps", 100)))
    adapter = run / f"drift_{drift}"
    if not (adapter / "adapter_config.json").is_file():
        raise ValueError(f"drift adapter is missing: {adapter}")
    required = {
        "model", "proj_dim", "grad_layers", "clip_cap", "micro_group",
        "topk_frac", "seed",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"run_config.json is missing keys: {missing}")
    return Namespace(
        model=str(config["model"]),
        adapter=str(adapter),
        proj_dim=int(config["proj_dim"]),
        grad_layers=int(config["grad_layers"]),
        clip_cap=float(config["clip_cap"]),
        micro_batch=1,
        micro_group=int(config["micro_group"]),
        topk_frac=float(config["topk_frac"]),
        seed=int(config["seed"]),
        radius_mode=str(config.get("radius_mode", "gaussian")),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args(argv)
    try:
        head = clean_code_head()
        config = json.loads((args.run / "run_config.json").read_text())
        generation = validate_generation_contract(args.run)
        print(
            f"[contract-ok] {generation['validated_rows']} rollout rows; "
            f"rescoring at {head[:12]}",
            flush=True,
        )
        stage_score(score_args(args.run, config), args.run)
        result = recompute(args.run)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"[rescore-abort] {exc}", file=sys.stderr)
        return 1
    print(
        "[rescore-ok] corrected off-policy, oracle, split-half, and report artifacts "
        f"written at {result['postprocess_git'][:12]}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
