#!/usr/bin/env python3
"""Materialize the exact d0 prompt split as a legacy math-loader snapshot."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path

FILENAMES = {
    "gsm8k": "gsm8k_train.jsonl",
    "math500": "math500_test.jsonl",
}


def _unshuffle(desired: list[dict]) -> list[dict]:
    indices = list(range(len(desired)))
    random.Random(0).shuffle(indices)
    original: list[dict | None] = [None] * len(desired)
    for position, source_index in enumerate(indices):
        original[source_index] = desired[position]
    return [row for row in original if row is not None]


def materialize(source: Path, dataset: str, output_root: Path) -> Path:
    if dataset not in FILENAMES:
        raise ValueError(f"unsupported legacy prompt dataset: {dataset}")
    raw = source.read_bytes()
    prompts = json.loads(raw)
    if not isinstance(prompts, dict):
        raise TypeError("prompts.json must contain an object")
    desired = prompts.get("train", []) + prompts.get("val", [])
    if not desired or not all(
        isinstance(row, dict) and row.get("question") and "answer" in row
        for row in desired
    ):
        raise ValueError("prompts.json has invalid train/val rows")

    digest = hashlib.sha256(raw).hexdigest()
    target_dir = output_root / f"{dataset}-{digest}"
    target = target_dir / FILENAMES[dataset]
    target_dir.mkdir(parents=True, exist_ok=True)
    lock_path = target_dir / ".materialize.lock"
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        rows = _unshuffle(desired)
        encoded = []
        for row in rows:
            if dataset == "gsm8k":
                document = {
                    "question": row["question"],
                    "answer": f"#### {row['answer']}",
                }
            else:
                document = {"problem": row["question"], "answer": str(row["answer"])}
            encoded.append(json.dumps(document, ensure_ascii=False) + "\n")
        payload = "".join(encoded).encode()
        if target.exists() and target.read_bytes() != payload:
            raise ValueError(f"existing prompt snapshot differs: {target}")
        if not target.exists():
            fd, temporary_name = tempfile.mkstemp(dir=target_dir, prefix=".prompts.")
            temporary = Path(temporary_name)
            try:
                with os.fdopen(fd, "wb") as output:
                    output.write(payload)
                    output.flush()
                    os.fsync(output.fileno())
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)

    check_rows = rows.copy()
    random.Random(0).shuffle(check_rows)
    normalized = [
        {
            "question": row.get("question", row.get("problem")),
            "answer": (
                str(row["answer"]).split("####")[-1].strip().replace(",", "")
                if dataset == "gsm8k"
                else str(row["answer"])
            ),
        }
        for row in check_rows
    ]
    if normalized != desired:
        raise AssertionError("materialized prompt snapshot does not reproduce d0 order")
    return target_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("dataset", choices=sorted(FILENAMES))
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    try:
        print(materialize(args.source, args.dataset, args.output_root))
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
