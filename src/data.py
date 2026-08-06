"""데이터 로딩 — 파일럿 기본 GSM8K, 옵션 MATH-500. 본실험은 DAPO-Math-17k로 확장.

H100 클러스터는 GitHub egress가 없으므로 HF 미러를 쓴다:
  export HF_ENDPOINT=<사내 미러>  (huggingface_hub가 자동 인식)
"""

from __future__ import annotations

import random
import re


def _gsm8k_answer(ans: str) -> str:
    return ans.split("####")[-1].strip().replace(",", "")


def load_prompts(dataset: str, n_train: int, n_val: int, seed: int = 0) -> dict:
    from datasets import load_dataset

    if dataset == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="train")
        items = [
            {"question": row["question"], "answer": _gsm8k_answer(row["answer"])}
            for row in ds
        ]
    elif dataset == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        items = [{"question": row["problem"], "answer": str(row["answer"])} for row in ds]
    else:
        raise ValueError(f"unknown dataset {dataset}")

    rng = random.Random(seed)
    rng.shuffle(items)
    need = n_train + n_val
    if len(items) < need:
        raise ValueError(f"{dataset}: {len(items)} < 필요 {need}")
    return {"train": items[:n_train], "val": items[n_train : n_train + n_val]}


PROMPT_TEMPLATE = (
    "Solve the following math problem. Reason step by step, then give the final "
    "answer after '####'.\n\nProblem: {question}\n"
)


ANSWER_RE = re.compile(r"####\s*([^\n]+)")


def extract_answer(text: str) -> str | None:
    m = ANSWER_RE.search(text)
    if m:
        return m.group(1).strip().rstrip(".").replace(",", "").replace("$", "")
    # fallback: \boxed{...}
    m = re.search(r"\\boxed\{([^{}]+)\}", text)
    return m.group(1).strip() if m else None


def reward(text: str, gold: str) -> float:
    pred = extract_answer(text)
    if pred is None:
        return 0.0
    gold = gold.strip().rstrip(".").replace(",", "").replace("$", "")
    if pred == gold:
        return 1.0
    try:  # 수치 동등 (예: 3.0 == 3)
        return 1.0 if abs(float(pred) - float(gold)) < 1e-6 else 0.0
    except ValueError:
        return 0.0
