#!/usr/bin/env python3
"""Materialize an exact d0 prompt split as a legacy loader snapshot."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import random
import re
import tempfile
from pathlib import Path

FILENAMES = {
    "gsm8k": "gsm8k_train.jsonl",
    "math500": "math500_test.jsonl",
    "mbpp": "mbpp.jsonl",
    "kk": "kk.jsonl",
    "arc-challenge": "arc_challenge.jsonl",
}

MBPP_PREFIX = "Write a Python function for the task below.\n\n"
MBPP_TESTS = "\n\nYour code should satisfy these tests:\n"
MBPP_SUFFIX = "\n\nReturn the complete function in a ```python code block."
KK_SUFFIX = (
    "\n\nDetermine each person's identity. Reason step by step, "
    "then after '####' state your final answer as e.g. "
    "'Zoey is a knight, Oliver is a knave.'"
)
ARC_PREFIX = (
    "Answer the following multiple-choice science question. Reason step by step, "
    "then write only the option label after '####'.\n\nQuestion: "
)
ARC_CHOICES = "\n\nChoices:\n"


def _unshuffle(desired: list[dict], seed: int) -> list[dict]:
    indices = list(range(len(desired)))
    random.Random(seed).shuffle(indices)
    original: list[dict | None] = [None] * len(desired)
    for position, source_index in enumerate(indices):
        original[source_index] = desired[position]
    return [row for row in original if row is not None]


def _document(row: dict, dataset: str) -> dict:
    question = str(row["question"])
    answer = str(row["answer"])
    if dataset == "gsm8k":
        return {"question": question, "answer": f"#### {answer}"}
    if dataset == "math500":
        return {"problem": question, "answer": answer}
    if dataset == "mbpp":
        marker = f"{MBPP_TESTS}{answer}{MBPP_SUFFIX}"
        if not question.startswith(MBPP_PREFIX) or not question.endswith(marker):
            raise ValueError("MBPP prompt does not match the legacy loader template")
        text = question[len(MBPP_PREFIX) : -len(marker)]
        if not text or not answer:
            raise ValueError("MBPP prompt has empty task text or tests")
        return {"text": text, "test_list": answer.split("\n")}
    if dataset == "kk":
        if not question.endswith(KK_SUFFIX) or not answer.startswith("KK:"):
            raise ValueError("KK prompt does not match the legacy loader template")
        quiz = question[: -len(KK_SUFFIX)]
        names: list[str] = []
        solution: list[bool] = []
        for assignment in answer[3:].split(";"):
            name, separator, role = assignment.partition("=")
            if not separator or not name or role not in {"knight", "knave"}:
                raise ValueError("KK answer does not match the legacy loader format")
            names.append(name)
            solution.append(role == "knight")
        if not quiz or not names:
            raise ValueError("KK prompt has empty quiz or solution")
        return {"quiz": quiz, "names": names, "solution": solution}
    if dataset == "arc-challenge":
        if not question.startswith(ARC_PREFIX) or not answer.startswith("ARC:"):
            raise ValueError("ARC prompt does not match the legacy loader template")
        body = question[len(ARC_PREFIX) :]
        raw_question, separator, options = body.rpartition(ARC_CHOICES)
        if not separator or not raw_question or not options:
            raise ValueError("ARC prompt has no question or choices")
        matches = list(re.finditer(r"(?m)^([A-H]|[1-9])\. ", options))
        if not matches or matches[0].start() != 0:
            raise ValueError("ARC choices do not match the legacy loader format")
        labels: list[str] = []
        texts: list[str] = []
        for index, match in enumerate(matches):
            end = (
                matches[index + 1].start() if index + 1 < len(matches) else len(options)
            )
            text = options[match.end() : end]
            if index + 1 < len(matches) and text.endswith("\n"):
                text = text[:-1]
            if not text or text != text.strip():
                raise ValueError("ARC choice text cannot be reconstructed exactly")
            labels.append(match.group(1))
            texts.append(text)
        answer_key = answer[4:]
        if answer_key not in labels:
            raise ValueError("ARC answer label is absent from the choices")
        return {
            "question": raw_question,
            "choices": {"label": labels, "text": texts},
            "answerKey": answer_key,
        }
    raise ValueError(f"unsupported legacy prompt dataset: {dataset}")


def _normalize(document: dict, dataset: str) -> dict:
    if dataset == "gsm8k":
        return {
            "question": document["question"],
            "answer": str(document["answer"])
            .split("####")[-1]
            .strip()
            .replace(",", ""),
        }
    if dataset == "math500":
        return {"question": document["problem"], "answer": str(document["answer"])}
    if dataset == "mbpp":
        tests = "\n".join(document["test_list"])
        question = f"{MBPP_PREFIX}{document['text']}{MBPP_TESTS}{tests}{MBPP_SUFFIX}"
        return {"question": question.strip(), "answer": tests}
    if dataset == "kk":
        gold = "KK:" + ";".join(
            f"{name}={'knight' if value else 'knave'}"
            for name, value in zip(document["names"], document["solution"], strict=True)
        )
        return {
            "question": f"{document['quiz']}{KK_SUFFIX}".strip(),
            "answer": gold,
        }
    labels = document["choices"]["label"]
    texts = document["choices"]["text"]
    options = "\n".join(
        f"{label}. {text}" for label, text in zip(labels, texts, strict=True)
    )
    return {
        "question": f"{ARC_PREFIX}{document['question']}{ARC_CHOICES}{options}",
        "answer": f"ARC:{document['answerKey']}",
    }


def materialize(source: Path, dataset: str, output_root: Path, seed: int = 0) -> Path:
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
    target_dir = output_root / f"{dataset}-s{seed}-{digest}"
    target = target_dir / FILENAMES[dataset]
    target_dir.mkdir(parents=True, exist_ok=True)
    lock_path = target_dir / ".materialize.lock"
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        rows = _unshuffle(desired, seed)
        documents = [_document(row, dataset) for row in rows]
        encoded = [json.dumps(row, ensure_ascii=False) + "\n" for row in documents]
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

    check_rows = documents.copy()
    random.Random(seed).shuffle(check_rows)
    normalized = [_normalize(row, dataset) for row in check_rows]
    if normalized != desired:
        raise AssertionError("materialized prompt snapshot does not reproduce d0 order")
    return target_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("dataset", choices=sorted(FILENAMES))
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    try:
        print(materialize(args.source, args.dataset, args.output_root, seed=args.seed))
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
