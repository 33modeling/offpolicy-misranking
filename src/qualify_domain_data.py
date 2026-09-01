#!/usr/bin/env python3
"""Fail-closed qualification for fixed cross-domain dataset snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from data import (
    MMLU_PRO_NONMATH_CATEGORIES,
    _boxed,
    _code_sandbox_backend,
    _load_rows_any,
    load_prompts,
    reward,
)

MMLU_PRO_QUOTAS = {
    category: 77 if index < 4 else 76
    for index, category in enumerate(MMLU_PRO_NONMATH_CATEGORIES)
}
MMLU_PRO_SELECTION = (
    "test split; stable-hash sample without duplicate questions; non-math categories "
    "business/economics/health/history=77 each and "
    "law/other/philosophy/psychology=76 each"
)

SPECS = {
    "gsm8k": {
        "revision": "740312add88f781978c0658806c59bc2815b9866",
        "repository": "openai/gsm8k",
        "file": "gsm8k_train.jsonl",
        "answer_prefix": None,
        "env": "GSM8K_DIR",
    },
    "math500": {
        "revision": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
        "repository": "HuggingFaceH4/MATH-500",
        "file": "math500_test.jsonl",
        "answer_prefix": None,
        "env": "MATH500_DIR",
        "aliases": ("math500", "MATH-500", "math-500", "math_500"),
        "rows": 500,
        "content_sha256": "8fe39a4f83b2331b2fdee1c5fa9dbc2f0f3523b5f0644f8d4bb8dd180f464c14",
        "selection": "test split",
    },
    "mbpp": {
        "revision": "4bb6404fdc6cacfda99d4ac4205087b89d32030c",
        "repository": "google-research-datasets/mbpp",
        "file": "mbpp.jsonl",
        "answer_prefix": "assert",
        "env": "MBPP_DIR",
        "aliases": ("mbpp", "MBPP"),
        "rows": 974,
        "content_sha256": "d6405fb43c7314126f24a98206789736a8713eeeca8886e9d888963d553b2aba",
        "selection": "all published splits, full config",
    },
    "kk": {
        "revision": "2f68547989981b1af37cb3dde5fdefa847aa8619",
        "repository": "K-and-K/knights-and-knaves",
        "file": "kk.jsonl",
        "answer_prefix": "KK:",
        "env": "KK_DIR",
    },
    "arc-challenge": {
        "revision": "210d026faf9955653af8916fad021475a3f00453",
        "repository": "allenai/ai2_arc",
        "file": "arc_challenge.jsonl",
        "answer_prefix": "ARC:",
        "env": "ARC_CHALLENGE_DIR",
    },
    "mmlu-pro-nonmath": {
        "revision": "b189ec765aa7ed75c8acfea42df31fdae71f97be",
        "repository": "TIGER-Lab/MMLU-Pro",
        "file": "mmlu_pro_nonmath.jsonl",
        "answer_prefix": "MMLU:",
        "env": "MMLU_PRO_DIR",
        "aliases": ("mmlu-pro-nonmath", "MMLU-Pro", "mmlu_pro"),
        "rows": 612,
        "content_sha256": "4c8958a934fb8f17ef43d1826e140c5acc6f60d130a3657637f6331ac524623e",
        "selection": MMLU_PRO_SELECTION,
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _question_hashes(rows: list[dict]) -> list[str]:
    return [hashlib.sha256(r["question"].strip().encode()).hexdigest() for r in rows]


def _jsonl_rows(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.open(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise TypeError(f"{path.name}:{line_number}: row must be an object")
        rows.append(row)
    return rows


def _canonical_rows(dataset: str, rows: list[dict]) -> list[dict]:
    canonical = []
    for index, row in enumerate(rows):
        try:
            if dataset == "math500":
                problem = row.get("problem") or row.get("question")
                answer = row.get("answer")
                if answer is None and isinstance(row.get("solution"), str):
                    answer = _boxed(row["solution"])
                if not isinstance(problem, str) or answer is None:
                    raise ValueError("problem/answer missing")
                canonical.append({"problem": problem, "answer": str(answer)})
            elif dataset == "mbpp":
                tests = row.get("test_list")
                if not isinstance(tests, list) or not all(
                    isinstance(test, str) for test in tests
                ):
                    raise ValueError("test_list must be a string list")
                canonical.append(
                    {
                        "task_id": row["task_id"],
                        "text": row["text"],
                        "code": row["code"],
                        "test_list": tests,
                    }
                )
            elif dataset == "mmlu-pro-nonmath":
                question = row.get("question")
                options = row.get("options")
                category = str(row.get("category", "")).strip().lower()
                answer = str(row.get("answer", "")).strip().upper()
                answer_index = row.get("answer_index")
                if (
                    not isinstance(question, str)
                    or not question.strip()
                    or not isinstance(options, list)
                    or not 2 <= len(options) <= 10
                    or not all(isinstance(option, str) and option.strip() for option in options)
                    or category not in MMLU_PRO_QUOTAS
                    or isinstance(answer_index, bool)
                    or not isinstance(answer_index, int)
                    or not 0 <= answer_index < len(options)
                    or answer != chr(ord("A") + answer_index)
                ):
                    raise ValueError("question/options/category/answer fields are inconsistent")
                canonical.append(
                    {
                        "question_id": row.get("question_id"),
                        "question": question.strip(),
                        "options": [option.strip() for option in options],
                        "answer": answer,
                        "answer_index": answer_index,
                        "category": category,
                        "src": str(row.get("src", "")),
                    }
                )
            else:
                return rows
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{dataset}: row {index} schema mismatch: {exc}") from exc
    return canonical


def _select_mmlu_pro_nonmath(rows: list[dict]) -> list[dict]:
    canonical = _canonical_rows("mmlu-pro-nonmath", rows)
    grouped = {category: [] for category in MMLU_PRO_NONMATH_CATEGORIES}
    for row in canonical:
        grouped[row["category"]].append(row)

    selected = []
    seen_questions = set()
    for category in MMLU_PRO_NONMATH_CATEGORIES:
        candidates = sorted(
            grouped[category],
            key=lambda row: hashlib.sha256(
                json.dumps(
                    row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest(),
        )
        for row in candidates:
            if row["question"] in seen_questions:
                continue
            selected.append(row)
            seen_questions.add(row["question"])
            if sum(item["category"] == category for item in selected) == MMLU_PRO_QUOTAS[category]:
                break
        actual = sum(item["category"] == category for item in selected)
        if actual != MMLU_PRO_QUOTAS[category]:
            raise ValueError(
                f"mmlu-pro-nonmath: category {category!r} has {actual} unique rows; "
                f"expected {MMLU_PRO_QUOTAS[category]}"
            )
    return selected


def _content_sha256(rows: list[dict]) -> str:
    digest = hashlib.sha256()
    lines = [
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for row in rows
    ]
    for line in sorted(lines):
        digest.update(line.encode("utf-8") + b"\n")
    return digest.hexdigest()


def _source_candidates(dataset: str, root: Path) -> list[Path]:
    spec = SPECS[dataset]
    filename = spec["file"]
    aliases = spec.get("aliases", (dataset,))
    candidates: list[Path] = []
    explicit = os.environ.get(spec["env"])
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend([root / dataset / filename, root / dataset, root / filename])
    candidates.extend(root / alias for alias in aliases)
    om_data = os.environ.get("OM_DATA")
    if om_data:
        base = Path(om_data)
        candidates.extend([base / filename, base / dataset / filename, base / dataset])
        candidates.extend(base / alias for alias in aliases)
    # Folder names are not a contract. Check exact filenames first, then every
    # plausible local artifact/directory by its parsed schema and content hash.
    if root.is_dir():
        candidates.extend(sorted(root.rglob(filename)))
        candidates.extend(
            path
            for path in sorted(root.rglob("*.jsonl"))
            if path.name != "dataset_manifest.json"
        )
        candidates.extend(sorted(root.rglob("*.parquet")))
        candidates.extend(sorted({path.parent for path in root.rglob("*.parquet")}))
        candidates.extend(sorted({path.parent for path in root.rglob("state.json")}))
        for pattern in ("*", "*/*", "*/*/*"):
            candidates.extend(path for path in sorted(root.glob(pattern)) if path.is_dir())
    unique = []
    seen = set()
    for candidate in candidates:
        marker = str(candidate)
        if marker not in seen:
            seen.add(marker)
            unique.append(candidate)
    return unique


def _write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _adopt_official_upload(dataset: str, root: Path) -> tuple[Path, list[dict]]:
    spec = SPECS[dataset]
    expected_hash = spec.get("content_sha256")
    if expected_hash is None:
        target = root / dataset / spec["file"]
        return target, _jsonl_rows(target)

    tried = []
    for source in _source_candidates(dataset, root):
        if not source.exists():
            continue
        try:
            rows = _load_rows_any(source)
            if not rows:
                raise ValueError("no rows")
            canonical = (
                _select_mmlu_pro_nonmath(rows)
                if dataset == "mmlu-pro-nonmath"
                else _canonical_rows(dataset, rows)
            )
            actual_hash = _content_sha256(canonical)
            if len(canonical) != spec["rows"] or actual_hash != expected_hash:
                raise ValueError(
                    f"rows={len(canonical)} content_sha256={actual_hash[:12]}"
                )
        except (OSError, TypeError, ValueError) as exc:
            tried.append(f"{source} ({exc})")
            continue

        target = root / dataset / spec["file"]
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + f".tmp.{os.getpid()}")
        with temporary.open("w", encoding="utf-8") as stream:
            for row in canonical:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(target)
        raw_rows = _jsonl_rows(target)
        manifest = {
            "schema_version": 1,
            "dataset": dataset,
            "source_repository": spec["repository"],
            "source_revision": spec["revision"],
            "selection": spec["selection"],
            "artifact": spec["file"],
            "rows": len(raw_rows),
            "sha256": _sha256(target),
            "content_sha256": expected_hash,
            "verification": "official-content-sha256-v1",
        }
        _write_json_atomic(target.parent / "dataset_manifest.json", manifest)
        return target, raw_rows

    locations = "\n  ".join(tried) if tried else "no candidate exists"
    raise ValueError(
        f"{dataset}: official local dataset not found or content mismatch. Checked:\n  "
        f"{locations}"
    )


def _verify_reward_runtime(dataset: str, raw_rows: list[dict], split: dict) -> dict:
    """Exercise the real reward path before any GPU process is started."""
    if dataset == "mbpp":
        synthetic = reward(
            "```python\ndef add(a, b):\n    return a + b\n```",
            "assert add(2, 3) == 5",
        )
        if synthetic != 1.0:
            raise ValueError(
                f"mbpp: {_code_sandbox_backend()} code sandbox cannot execute Python"
            )
        checked = 0
        for row in raw_rows:
            code, tests = row.get("code"), row.get("test_list")
            if not code or not isinstance(tests, list) or not tests:
                continue
            if reward(f"```python\n{code}\n```", "\n".join(map(str, tests))) != 1.0:
                raise ValueError(
                    f"mbpp: published reference solution failed sandbox tests "
                    f"(task_id={row.get('task_id')})"
                )
            checked += 1
            if checked == 3:
                break
        if checked < 3:
            raise ValueError("mbpp: fewer than three executable reference rows")
        return {
            "kind": "isolated-python-execution",
            "backend": _code_sandbox_backend(),
            "reference_rows": checked,
        }

    gold = str(split["train"][0]["answer"])
    if dataset in {"gsm8k", "math500"}:
        canonical = f"Reasoning\n#### {gold}"
        if reward(canonical, gold) != 1.0 or reward("Reasoning\n#### not-a-number", gold) != 0.0:
            raise ValueError(f"{dataset}: positive/negative math reward self-test failed")
        if reward("Reasoning\n#### 0.5", r"\frac{1}{2}") != 1.0:
            raise ValueError(f"{dataset}: symbolic-equivalence reward self-test failed")
        return {"kind": "math-verify-equivalence", "reference_rows": 1}
    if dataset == "kk":
        canonical = "#### " + ", ".join(
            f"{name} is a {role}" for name, role in (
                part.split("=", 1) for part in gold[3:].split(";") if "=" in part
            )
        )
        wrong = canonical.replace("knight", "wrong", 1).replace("knave", "wrong", 1)
    else:
        label = gold.split(":", 1)[1]
        canonical = f"Reasoning\n#### {label}"
        wrong_label = next(candidate for candidate in "ABCDEFGHIJ" if candidate != label.upper())
        wrong = f"Reasoning\n#### {wrong_label}"
    if reward(canonical, gold) != 1.0 or reward(wrong, gold) != 0.0:
        raise ValueError(f"{dataset}: positive/negative reward self-test failed")
    return {"kind": "exact-structured-match", "reference_rows": 1}


def qualify(dataset: str, root: Path, n_train: int, n_val: int, seeds: list[int]) -> dict:
    spec = SPECS[dataset]
    data_file, raw_rows = _adopt_official_upload(dataset, root)
    dataset_root = data_file.parent
    manifest_file = dataset_root / "dataset_manifest.json"
    if not manifest_file.is_file():
        raise ValueError(f"{dataset}: 고정 snapshot/manifest 없음: {dataset_root}")

    manifest = json.loads(manifest_file.read_text())
    digest = _sha256(data_file)
    if manifest.get("dataset") != dataset:
        raise ValueError(f"{dataset}: manifest dataset name mismatch")
    if manifest.get("source_repository") != spec["repository"]:
        raise ValueError(f"{dataset}: source repository mismatch")
    if manifest.get("source_revision") != spec["revision"]:
        raise ValueError(f"{dataset}: source revision 불일치")
    if manifest.get("artifact") != spec["file"]:
        raise ValueError(f"{dataset}: manifest artifact name mismatch")
    if manifest.get("sha256") != digest:
        raise ValueError(f"{dataset}: manifest SHA-256 불일치")
    if manifest.get("rows") != len(raw_rows):
        raise ValueError(f"{dataset}: manifest row count 불일치")
    if spec.get("content_sha256") and (
        _content_sha256(
            _select_mmlu_pro_nonmath(raw_rows)
            if dataset == "mmlu-pro-nonmath"
            else _canonical_rows(dataset, raw_rows)
        )
        != spec["content_sha256"]
    ):
        raise ValueError(f"{dataset}: official content SHA-256 mismatch")
    if spec.get("selection") and manifest.get("selection") != spec["selection"]:
        raise ValueError(f"{dataset}: manifest selection mismatch")

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
        if not all(answers):
            raise ValueError(f"{dataset}: empty normalized answer")
        if spec["answer_prefix"] is not None and not all(
            a.lstrip().startswith(spec["answer_prefix"]) for a in answers
        ):
            raise ValueError(f"{dataset}: reward contract 불일치")
        reward_runtime = _verify_reward_runtime(dataset, raw_rows, split)
    finally:
        if old is None:
            os.environ.pop(spec["env"], None)
        else:
            os.environ[spec["env"]] = old

    return {
        "dataset": dataset,
        "status": "qualified",
        "source_revision": spec["revision"],
        "source_path": str(data_file),
        "snapshot_sha256": digest,
        "snapshot_rows": len(raw_rows),
        "n_train": n_train,
        "n_val": n_val,
        "prompt_split": {
            "seed": 0,
            "train_prompt_set_sha256": hashlib.sha256("".join(sorted(train_hashes)).encode()).hexdigest(),
            "validation_prompt_set_sha256": hashlib.sha256("".join(sorted(val_hashes)).encode()).hexdigest(),
            "train_prompt_order_sha256": hashlib.sha256("".join(train_hashes).encode()).hexdigest(),
            "validation_prompt_order_sha256": hashlib.sha256("".join(val_hashes).encode()).hexdigest(),
        },
        "experiment_seeds": seeds,
        "reward_runtime": reward_runtime,
    }


def _dataset_train_sizes(
    datasets: list[str], default: int, overrides: list[str]
) -> dict[str, int]:
    sizes = {dataset: default for dataset in datasets}
    seen = set()
    for raw in overrides:
        dataset, separator, value = raw.partition("=")
        if not separator or dataset not in sizes or dataset in seen:
            raise ValueError(f"invalid --dataset-n-train override: {raw!r}")
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ValueError(f"invalid --dataset-n-train override: {raw!r}") from exc
        if parsed <= 0:
            raise ValueError(f"invalid --dataset-n-train override: {raw!r}")
        sizes[dataset] = parsed
        seen.add(dataset)
    return sizes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*", choices=sorted(SPECS))
    parser.add_argument("--data-root", default=os.environ.get("DATASETS_DIR"))
    parser.add_argument("--n-train", type=int, default=512)
    parser.add_argument("--dataset-n-train", action="append", default=[])
    parser.add_argument("--n-val", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output")
    args = parser.parse_args()
    if not args.data_root:
        parser.error("--data-root or DATASETS_DIR is required")

    root = Path(args.data_root)
    datasets = args.datasets or list(SPECS)
    train_sizes = _dataset_train_sizes(
        datasets, args.n_train, args.dataset_n_train
    )
    report = {
        "schema_version": 1,
        "status": "qualified",
        "datasets": [
            qualify(ds, root, train_sizes[ds], args.n_val, args.seeds)
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
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"[qualification-abort] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
