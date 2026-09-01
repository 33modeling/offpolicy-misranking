"""Legacy loaders can reproduce every immutable d0 prompt split."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import load_prompts
from materialize_prompt_dataset import materialize

ENV_NAMES = {
    "gsm8k": "GSM8K_DIR",
    "math500": "MATH500_DIR",
    "mbpp": "MBPP_DIR",
    "kk": "KK_DIR",
    "arc-challenge": "ARC_CHALLENGE_DIR",
}


def _raw_rows(dataset: str) -> list[dict]:
    if dataset == "gsm8k":
        return [
            {
                "question": f"arithmetic question {index}",
                "answer": f"work\n#### {index}",
            }
            for index in range(9)
        ]
    if dataset == "math500":
        return [
            {"problem": f"math problem {index}", "answer": f"x_{index}"}
            for index in range(9)
        ]
    if dataset == "mbpp":
        return [
            {
                "text": f"Return the integer {index}.",
                "test_list": [f"assert solve() == {index}", "assert callable(solve)"],
            }
            for index in range(9)
        ]
    if dataset == "kk":
        return [
            {
                "quiz": f"Logic puzzle {index}.",
                "names": [f"Alex{index}", f"Blair{index}"],
                "solution": [index % 2 == 0, index % 2 != 0],
            }
            for index in range(9)
        ]
    return [
        {
            "question": f"Science question {index}?",
            "choices": {
                "label": ["A", "B", "C", "D"],
                "text": [f"choice {letter} for {index}" for letter in "ABCD"],
            },
            "answerKey": "ABCD"[index % 4],
        }
        for index in range(9)
    ]


def _filename(dataset: str) -> str:
    return {
        "gsm8k": "gsm8k_train.jsonl",
        "math500": "math500_test.jsonl",
        "mbpp": "mbpp.jsonl",
        "kk": "kk.jsonl",
        "arc-challenge": "arc_challenge.jsonl",
    }[dataset]


def _load_with_root(dataset: str, root: Path, seed: int) -> dict:
    env_name = ENV_NAMES[dataset]
    old_root = os.environ.get(env_name)
    old_pool = os.environ.pop("OM_POOL_FILE", None)
    os.environ[env_name] = str(root)
    try:
        return load_prompts(dataset, 6, 3, seed=seed)
    finally:
        if old_root is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = old_root
        if old_pool is not None:
            os.environ["OM_POOL_FILE"] = old_pool


def test_materialized_snapshots_reproduce_all_additional_splits(tmp_path: Path) -> None:
    for dataset in ENV_NAMES:
        raw_root = tmp_path / "raw" / dataset
        raw_root.mkdir(parents=True)
        rows = _raw_rows(dataset)
        (raw_root / _filename(dataset)).write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        for seed in (0, 1, 2):
            prompts = _load_with_root(dataset, raw_root, seed)
            source = tmp_path / "prompts" / dataset / f"s{seed}.json"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text(json.dumps(prompts), encoding="utf-8")

            snapshot = materialize(source, dataset, tmp_path / "snapshots", seed=seed)
            assert _load_with_root(dataset, snapshot, seed) == prompts


def test_materialized_snapshot_is_seed_specific(tmp_path: Path) -> None:
    prompts = {
        "train": [
            {"question": f"q{index}", "answer": str(index)} for index in range(6)
        ],
        "val": [{"question": f"v{index}", "answer": str(index)} for index in range(3)],
    }
    source = tmp_path / "prompts.json"
    source.write_text(json.dumps(prompts), encoding="utf-8")
    roots = [
        materialize(source, "gsm8k", tmp_path / "snapshots", seed) for seed in (0, 1, 2)
    ]
    assert len(set(roots)) == 3
    for seed, root in enumerate(roots):
        assert _load_with_root("gsm8k", root, seed) == prompts
