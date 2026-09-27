"""Build an ``srgc-inputs-v1`` bundle from MATH train, MBPP, GSM8K or JSONL.

The bundle is what ``run_experiment`` consumes: 400 candidates, a 100-prompt
validation pool with an explicit ranking-validation subset, 300 evaluation
questions, formatted RL-Zero prompts, and eight cached initial-policy rewards
per candidate. Rewards come either from an existing cache (``--cache``) or are
left for ``build_cache`` to generate on GPUs; ``--check`` validates a finished
bundle with the same validator the experiment uses.

    python -m srgc_rebuttal.build_inputs --dataset mbpp --seed 5 \
        --output srgc_rebuttal/inputs/mbpp-seed-5.json [--cache CACHE.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path

from .plan import validate_inputs

SCHEMA = "srgc-inputs-v1"
SIZES = {"candidates": 400, "validation": 100, "evaluation": 300}
RANKING_VALIDATION = 50  # fixed for new runs, not inferred from old online measurements
MATH_DATASET = "EleutherAI/hendrycks_math"
MATH_REVISION = "21a5633873b6a120296cce3e2df9d5550074f4a3"
MATH_SUBSETS = ("algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                "number_theory", "prealgebra", "precalculus")
# Released OLMo RL-Zero prompt formats, reproduced verbatim from the experiment code.
MATH_PROMPT = ("Solve the following problem step by step. The last line of your response "
               "should be the answer to the problem in form Answer: $Answer (without quotes) "
               "where $Answer is the answer to the problem.\n\n"
               "{question}\n\nRemember to put your answer on its own line after \"Answer:\"")
CODE_PROMPT = ("Solve the following code problem step by step. The last part of your response "
               "should be the solution to the problem in form ```\npython\nCODE\n``` where CODE "
               "is the solution for the problem.\n\n"
               "{question}\n\nRemember to put your solution inside the ```\npython\nCODE\n``` tags")
DATASETS = {"math_train": ("math", "srgc_rebuttal.verifiers:math_reward"),
            "math500": ("math", "srgc_rebuttal.verifiers:math_reward"),
            "gsm8k": ("math", "srgc_rebuttal.verifiers:math_reward"),
            "mbpp": ("code", "srgc_rebuttal.verifiers:code_reward"),
            "jsonl": (None, None)}


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def gsm8k_answer(answer: str) -> str:
    return answer.split("####")[-1].strip().replace(",", "")


def math_answer(solution: str) -> str:
    """Extract the final boxed gold answer, preserving nested LaTeX braces."""
    matches = list(re.finditer(r"\\(?:boxed|fbox)\s*", solution))
    if not matches:
        raise ValueError("MATH solution has no boxed answer")
    start = matches[-1].end()
    if start >= len(solution):
        raise ValueError("empty boxed answer")
    if solution[start] != "{":
        if solution[start] in "\\}$":
            raise ValueError("unsupported unbraced gold answer")
        return "$\\boxed{" + solution[start] + "}$"
    start, depth = start + 1, 1
    for end in range(start, len(solution)):
        if solution[end] == "{" and solution[end - 1] != "\\":
            depth += 1
        elif solution[end] == "}" and solution[end - 1] != "\\":
            depth -= 1
            if depth == 0:
                return "$\\boxed{" + solution[start:end] + "}$"
    raise ValueError("unbalanced boxed gold answer")


def load_rows(dataset: str, rows_path: Path | None) -> list[dict]:
    """Rows of {question, answer}; ``answer`` is the verifier gold (LaTeX/number or assert tests)."""
    if rows_path is not None:
        rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
        if any(not r.get("question") or r.get("answer") is None for r in rows):
            raise ValueError("every JSONL row needs 'question' and 'answer'")
        return [{"question": str(r["question"]), "answer": str(r["answer"])} for r in rows]
    if dataset == "math500":
        raise ValueError("MATH-500 has only 500 evaluation questions; use math_train for 400/100/300 disjoint splits")
    from datasets import load_dataset
    if dataset == "math_train":
        rows = []
        for subset in MATH_SUBSETS:
            for r in load_dataset(MATH_DATASET, subset, split="train", revision=MATH_REVISION):
                rows.append({"question": r["problem"], "answer": math_answer(r["solution"])})
        return rows
    if dataset == "gsm8k":
        return [{"question": r["question"], "answer": gsm8k_answer(r["answer"])} for r in load_dataset("openai/gsm8k", "main", split="train")]
    if dataset == "mbpp":
        ds = load_dataset("google-research-datasets/mbpp", "full")
        rows = []
        for split in ds:
            for r in ds[split]:
                tests = "\n".join(r["test_list"])
                question = (f"Write a Python function for the task below.\n\n{r['text']}\n\n"
                            f"Your code should satisfy these tests:\n{tests}\n\n"
                            "Return the complete function in a ```python code block.")
                rows.append({"question": question, "answer": tests})
        return rows
    raise ValueError(f"unknown dataset {dataset!r}; pass --rows for a custom JSONL file")


def has_gold(row: dict) -> bool:
    answer = str(row.get("answer", "")).strip()
    return bool(answer) and re.fullmatch(r"\$?\\(?:boxed|fbox)\{\s*\}\$?", answer) is None


def dedupe(rows: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in rows:
        key = normalized(r["question"])
        if key and key not in seen and has_gold(r):
            seen.add(key)
            out.append(r)
    return out


def record_id(dataset: str, question: str) -> str:
    return f"{dataset}-{hashlib.sha256(normalized(question).encode()).hexdigest()[:16]}"


def build(dataset: str, rows: list[dict], *, split_seed: int, kind: str, ranking_validation: int,
          cache: dict[str, list[int]] | None, provenance: dict) -> dict:
    if type(ranking_validation) is not int or not 1 <= ranking_validation <= SIZES["validation"]:
        raise ValueError("ranking-validation count must be between 1 and 100")
    rows = dedupe(rows)
    need = sum(SIZES.values())
    if len(rows) < need:
        raise ValueError(f"{dataset}: {len(rows)} distinct rows < {need} needed")
    rng = random.Random(split_seed)
    rng.shuffle(rows)
    template = MATH_PROMPT if kind == "math" else CODE_PROMPT
    records, groups = {}, {}
    cursor = 0
    for name, size in SIZES.items():
        ids = []
        for r in rows[cursor:cursor + size]:
            rid = record_id(dataset, r["question"])
            records[rid] = {"question": r["question"], "prompt": template.format(question=r["question"]),
                            "answer": r["answer"]}
            ids.append(rid)
        groups[name] = ids
        cursor += size
    ranking = groups["validation"][:ranking_validation]
    cached = {}
    if cache is not None:
        missing = [i for i in groups["candidates"] if i not in cache]
        if missing:
            raise ValueError(f"cache lacks {len(missing)} candidates (e.g. {missing[0]}); build it with build_cache")
        cached = {i: list(cache[i]) for i in groups["candidates"]}
    bundle = {"schema": SCHEMA, "dataset": dataset, "records": records,
            "candidate_ids": groups["candidates"], "validation_pool_ids": groups["validation"],
            "ranking_validation_ids": ranking, "evaluation_ids": groups["evaluation"],
            "cached_rewards": cached,
            "provenance": {**provenance, "split_seed": split_seed, "prompt_format": f"olmo_rlzero_{kind}",
                           "ranking_validation": f"first {ranking_validation} of the shuffled validation pool",
                           "cache": provenance.get("cache", "pending: run build_cache")}}
    validate_inputs(bundle, require_cache=cache is not None)
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--rows", type=Path, help="JSONL with question/answer rows (required for --dataset jsonl)")
    parser.add_argument("--kind", choices=("math", "code"), help="prompt/verifier kind for --dataset jsonl")
    parser.add_argument("--seed", type=int, required=True, help="experiment seed; also the default split seed")
    parser.add_argument("--split-seed", type=int)
    parser.add_argument("--ranking-validation", type=int, default=RANKING_VALIDATION)
    parser.add_argument("--cache", type=Path, help="JSON {record_id: [8 binary rewards]} from build_cache or an earlier bundle")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="validate an existing bundle and exit")
    args = parser.parse_args()
    if args.check:
        validate_inputs(json.loads(args.output.read_text()))
        print(f"PASS: {args.output} is a valid {SCHEMA} bundle")
        return
    kind, verifier = DATASETS[args.dataset]
    if args.dataset == "jsonl":
        if args.rows is None or args.kind is None:
            parser.error("--dataset jsonl needs --rows and --kind")
        kind = args.kind
        verifier = DATASETS["math500" if kind == "math" else "mbpp"][1]
    cache = None
    if args.cache is not None:
        loaded = json.loads(args.cache.read_text())
        cache = loaded.get("cached_rewards", loaded)
    bundle = build(args.dataset, load_rows(args.dataset, args.rows),
                   split_seed=args.seed if args.split_seed is None else args.split_seed,
                   kind=kind, ranking_validation=args.ranking_validation, cache=cache,
                   provenance={"source": str(args.rows) if args.rows else f"huggingface:{args.dataset}",
                               "verifier": verifier, "experiment_seed": args.seed,
                               **({"dataset_revision": MATH_REVISION, "source_split": "train"}
                                  if args.dataset == "math_train" and args.rows is None else {}),
                               **({"source_sha256": hashlib.sha256(args.rows.read_bytes()).hexdigest()} if args.rows else {}),
                               **({"cache": str(args.cache)} if args.cache else {})})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bundle, ensure_ascii=False, indent=1) + "\n")
    if cache is not None:
        validate_inputs(bundle)
        print(f"PASS: wrote complete bundle {args.output}")
    else:
        print(f"wrote {args.output} without cached rewards; next: torchrun --standalone --nproc_per_node=4 "
              f"-m srgc_rebuttal.build_cache --bundle {args.output} --verifier {verifier}")


if __name__ == "__main__":
    main()
