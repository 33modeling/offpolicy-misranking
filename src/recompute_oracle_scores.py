"""Recompute corrected oracle split-halves and reports from completed artifacts.

This path preserves expensive rollouts only when their manifests prove the
raw-softmax generation contract and every row records an EOS-trimmed boundary.

    python3 src/recompute_oracle_scores.py RUN_DIR
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path

import torch

from artifact_contract import sha256_file, validate_generation_contract
from experiment import (
    ORACLE_PROTOCOL_SCHEMA,
    SCORE_PROTOCOL_SCHEMA,
    _atomic_text,
    oracle_protocol_document,
    score_oracle_microgroups,
    split_validation_directions,
    stage_report,
)


def recompute(run: Path) -> dict:
    required = (
        run / "run_config.json",
        run / "prompts.json",
        run / "oracle_micro_groups.pt",
        run / "val_groups.pt",
        run / "scores_offpolicy.json",
        run / "score_protocol.json",
    )
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"required artifacts are missing: {missing}")
    generation = validate_generation_contract(run)
    config = json.loads((run / "run_config.json").read_text())
    score_protocol = json.loads((run / "score_protocol.json").read_text())
    if score_protocol.get("schema") != SCORE_PROTOCOL_SCHEMA:
        raise ValueError(
            "off-policy scores predate the independent validation split; rerun the score stage"
        )
    prompts = json.loads((run / "prompts.json").read_text())["train"]
    micro = torch.load(run / "oracle_micro_groups.pt", map_location="cpu", weights_only=True)
    val_groups = torch.load(run / "val_groups.pt", map_location="cpu", weights_only=True)
    expected_ids = set(range(len(prompts)))
    if set(micro) != expected_ids:
        raise ValueError("oracle micro-group prompt coverage does not match prompts.json")
    val_half_a, val_half_b = split_validation_directions(val_groups)
    oracle, halves = {}, {}
    for prompt_idx, stack in sorted(micro.items()):
        oracle[prompt_idx], halves[prompt_idx] = score_oracle_microgroups(
            stack.float(), val_half_b.float(), val_half_a.float(), val_half_b.float()
        )
    _atomic_text(run / "scores_splithalf.json", json.dumps(halves, indent=1))
    _atomic_text(run / "scores_oracle.json", json.dumps(oracle, indent=1))
    oracle_protocol = oracle_protocol_document(val_groups, generation)
    if oracle_protocol["schema"] != ORACLE_PROTOCOL_SCHEMA:
        raise AssertionError("oracle protocol schema drift")
    _atomic_text(run / "oracle_protocol.json", json.dumps(oracle_protocol, indent=1))

    args = Namespace(
        topk_frac=float(config["topk_frac"]),
        seed=int(config["seed"]),
        micro_group=int(config["micro_group"]),
        radius_mode=str(config.get("radius_mode", "gaussian")),
    )
    stage_report(args, run)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    inputs = {
        path.name: sha256_file(path)
        for path in required[1:]
    }
    result = {
        "schema": "offpolicy-corrected-postprocess/v1",
        "timestamp_unix": int(time.time()),
        "source_run_git": config.get("git"),
        "postprocess_git": head,
        "generation_validation": generation,
        "input_sha256": inputs,
        "outputs": {
            name: sha256_file(run / name)
            for name in ("scores_oracle.json", "scores_splithalf.json", "report.json")
        },
    }
    _atomic_text(run / "postprocess_manifest.json", json.dumps(result, indent=1))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args(argv)
    try:
        result = recompute(args.run)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"[postprocess-abort] {exc}", file=sys.stderr)
        return 1
    print(
        "[postprocess-ok] independent candidate/validation split-halves and report "
        f"regenerated at {result['postprocess_git'][:12]}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
