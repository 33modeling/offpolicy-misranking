#!/usr/bin/env python3
"""Fail-closed qualification for the fixed non-math dataset snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from data import load_prompts

SPECS = {
    "mbpp": {
        "revision": "4bb6404fdc6cacfda99d4ac4205087b89d32030c",
        "file": "mbpp.jsonl",
        "answer_prefix": "assert",
        "env": "MBPP_DIR",
    },
    "kk": {
        "revision": "2f68547989981b1af37cb3dde5fdefa847aa8619",
        "file": "kk.jsonl",
        "answer_prefix": "KK:",
        "env": "KK_DIR",
    },
    "arc-challenge": {
        "revision": "210d026faf9955653af8916fad021475a3f00453",
        "file": "arc_challenge.jsonl",
        "answer_prefix": "ARC:",
        "env": "ARC_CHALLENGE_DIR",
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _question_hashes(rows: list[dict]) -> list[str]:
    return [hashlib.sha256(r["question"].strip().encode()).hexdigest() for r in rows]


def qualify(dataset: str, root: Path, n_train: int, n_val: int, seeds: list[int]) -> dict:
    spec = SPECS[dataset]
    dataset_root = root / dataset
    data_file = dataset_root / spec["file"]
    manifest_file = dataset_root / "dataset_manifest.json"
    if not data_file.is_file() or not manifest_file.is_file():
        raise ValueError(f"{dataset}: 고정 snapshot/manifest 없음: {dataset_root}")

    manifest = json.loads(manifest_file.read_text())
    digest = _sha256(data_file)
    if manifest.get("source_revision") != spec["revision"]:
        raise ValueError(f"{dataset}: source revision 불일치")
    if manifest.get("sha256") != digest:
        raise ValueError(f"{dataset}: manifest SHA-256 불일치")
    rows = sum(1 for line in data_file.open() if line.strip())
    if manifest.get("rows") != rows:
        raise ValueError(f"{dataset}: manifest row count 불일치")

    old = os.environ.get(spec["env"])
    os.environ[spec["env"]] = str(dataset_root)
    try:
        # experiment.py intentionally keeps the prompt split at seed 0; experiment
        # seeds affect rollout, drift training, and tie breaking on this fixed pool.
        split = load_prompts(dataset, n_train, n_val, seed=0)
        repeated = load_prompts(dataset, n_train, n_val, seed=0)
        train_hashes = _question_hashes(split["train"])
        val_hashes = _question_hashes(split["val"])
        if train_hashes != _question_hashes(repeated["train"]):
            raise ValueError(f"{dataset}: prompt split 비결정적")
        if len(set(train_hashes)) != len(train_hashes):
            raise ValueError(f"{dataset}: train prompt 중복")
        if len(set(val_hashes)) != len(val_hashes):
            raise ValueError(f"{dataset}: validation prompt 중복")
        if set(train_hashes) & set(val_hashes):
            raise ValueError(f"{dataset}: train/validation prompt 중복")
        answers = [str(r["answer"]) for part in split.values() for r in part]
        if not all(a.lstrip().startswith(spec["answer_prefix"]) for a in answers):
            raise ValueError(f"{dataset}: reward contract 불일치")
    finally:
        if old is None:
            os.environ.pop(spec["env"], None)
        else:
            os.environ[spec["env"]] = old

    return {
        "dataset": dataset,
        "status": "qualified",
        "source_revision": spec["revision"],
        "snapshot_sha256": digest,
        "snapshot_rows": rows,
        "n_train": n_train,
        "n_val": n_val,
        "prompt_split": {
            "seed": 0,
            "train_prompt_set_sha256": hashlib.sha256("".join(sorted(train_hashes)).encode()).hexdigest(),
            "validation_prompt_set_sha256": hashlib.sha256("".join(sorted(val_hashes)).encode()).hexdigest(),
        },
        "experiment_seeds": seeds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*", choices=sorted(SPECS))
    parser.add_argument("--data-root", default=os.environ.get("DATASETS_DIR"))
    parser.add_argument("--n-train", type=int, default=512)
    parser.add_argument("--n-val", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output")
    args = parser.parse_args()
    if not args.data_root:
        parser.error("--data-root or DATASETS_DIR is required")

    root = Path(args.data_root)
    datasets = args.datasets or list(SPECS)
    report = {
        "schema_version": 1,
        "status": "qualified",
        "datasets": [
            qualify(ds, root, args.n_train, args.n_val, args.seeds)
            for ds in datasets
        ],
    }
    output = Path(args.output) if args.output else root / "domain_dataset_qualification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    tmp.replace(output)
    for row in report["datasets"]:
        print(
            f"[qualified] {row['dataset']}: rows={row['snapshot_rows']} "
            f"sha256={row['snapshot_sha256'][:12]} seeds={len(row['experiment_seeds'])}"
        )
    print(f"[qualified] report={output}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[qualification-abort] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
