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
    import json
    import os
    from pathlib import Path

    # 사전 구성 풀 오버라이드 (예: 27B hard-slice) — {"question","answer"} jsonl.
    # dataset 이름은 reward 분기용으로 그대로 쓰이고, 풀 내용만 이 파일이 대체한다.
    pool = os.environ.get("OM_POOL_FILE")
    if pool:
        pf = Path(pool)
        if not pf.is_file():
            raise ValueError(f"OM_POOL_FILE 없음: {pool}")
        rows = [json.loads(l) for l in pf.open()]
        items = [{"question": r["question"], "answer": str(r["answer"])} for r in rows]
        return _split(items, n_train, n_val, seed, f"pool({pf.name})")

    # 로컬 사본 — provision.sh($OM_DATA) 또는 fetch_datasets.sh($DATASETS_DIR/<이름>/)
    local_names = {"gsm8k": "gsm8k_train.jsonl", "math500": "math500_test.jsonl"}
    fname = local_names.get(dataset, "_none_")
    local = next((c for c in (Path(os.environ.get("OM_DATA", "")) / fname,
                              Path(os.environ.get("DATASETS_DIR", "")) / dataset / fname)
                  if c.is_file()), None)
    if local is not None:
        rows = [json.loads(l) for l in local.open()]
        if dataset == "gsm8k":
            items = [
                {"question": r["question"], "answer": _gsm8k_answer(r["answer"])}
                for r in rows
            ]
        else:  # math500
            items = [{"question": r["problem"], "answer": str(r["answer"])} for r in rows]
        return _split(items, n_train, n_val, seed, f"{dataset}(local)")

    # datasets 라이브러리는 허브 폴백에서만 지연 import — 로컬 사본 경로는
    # 라이브러리 상태(미설치·구버전 오프라인 미지원)와 무관하게 동작해야 한다

    if dataset == "gsm8k":
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="train")
        items = [
            {"question": row["question"], "answer": _gsm8k_answer(row["answer"])}
            for row in ds
        ]
    elif dataset == "math500":
        # 사전 배치본 우선 — $MATH500_DIR 또는 데이터셋 베이스들 아래 통상 이름들
        names = ("math500", "MATH-500", "math-500", "math_500", "math", "MATH")
        tried = []
        root = os.environ.get("MATH500_DIR")
        if not root:
            for base in _dataset_bases():
                tried += [base / n for n in names]
            root = next((c for c in tried if c.exists()), None)
        rows = _load_rows_any(root) if root else None
        if rows is None:
            try:
                from datasets import load_dataset
                ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
            except Exception as e:
                raise ValueError(
                    "math500 로컬 사본을 못 찾았고 허브도 실패. 찾아본 위치:\n  "
                    + "\n  ".join(str(t) for t in tried)
                    + f"\n실제 위치가 다르면 MATH500_DIR=<경로> 지정. (허브 오류: {e})"
                ) from e
            rows = list(ds)
        items = []
        for r in rows:
            q = r.get("problem") or r.get("question")
            a = r.get("answer")
            if a is None and r.get("solution"):  # 원본 MATH 형식 — \boxed에서 추출
                a = _boxed(r["solution"])
            if q and a is not None:
                items.append({"question": q, "answer": str(a)})
        if not items:
            raise ValueError(f"math500 스키마 파싱 실패 (root={root})")
    elif dataset == "mbpp":
        # 클러스터 사전 배치본($DATASETS_DIR/mbpp) 우선 — jsonl/parquet/HF 스냅샷 전부 수용
        tried = [Path(os.environ["MBPP_DIR"])] if os.environ.get("MBPP_DIR") \
            else [b / "mbpp" for b in _dataset_bases()]
        root = next((c for c in tried if c.exists()), None)
        rows = _load_rows_any(root) if root else None
        if rows is None:
            try:
                from datasets import load_dataset
                ds = load_dataset("google-research-datasets/mbpp", "full")
            except Exception as e:
                raise ValueError(
                    "mbpp 로컬 사본을 못 찾았고 허브도 실패. 찾아본 위치:\n  "
                    + "\n  ".join(str(t) for t in tried)
                    + f"\nMBPP_DIR=<경로>로 지정 가능. (허브 오류: {e})"
                ) from e
            rows = [r for split in ds for r in ds[split]]
        items = []
        for r in rows:
            text = r.get("text") or r.get("prompt") or r.get("description")
            tests = r.get("test_list") or r.get("tests") or r.get("test")
            if isinstance(tests, str):
                tests = [tests]
            if not (text and tests):
                continue
            tests_str = "\n".join(tests)
            q = (f"Write a Python function for the task below.\n\n{text}\n\n"
                 f"Your code should satisfy these tests:\n{tests_str}\n\n"
                 "Return the complete function in a ```python code block.")
            # answer 자리에 assert 테스트를 넣는다 — reward()가 실행 채점으로 분기
            items.append({"question": q, "answer": tests_str})
        if not items:
            raise ValueError(f"mbpp 스키마 파싱 실패 (root={root}) — 필드명을 확인할 것")
    elif dataset == "kk":
        # Knights & Knaves 논리 퍼즐 — 사전 배치본($DATASETS_DIR/kk 등)
        tried = [Path(os.environ["KK_DIR"])] if os.environ.get("KK_DIR") \
            else [b / n for b in _dataset_bases()
                  for n in ("kk", "knights-and-knaves", "knights_and_knaves")]
        root = next((c for c in tried if c.exists()), None)
        rows = _load_rows_any(root) if root else None
        if rows is None:
            raise ValueError("kk 로컬 사본 없음. 찾아본 위치:\n  "
                             + "\n  ".join(str(t) for t in tried)
                             + "\nKK_DIR=<경로>로 지정 가능.")
        items = []
        for r in rows:
            quiz, names, sol = r.get("quiz"), r.get("names"), r.get("solution")
            if not (quiz and names and isinstance(sol, list) and len(sol) == len(names)):
                continue
            gold = "KK:" + ";".join(
                f"{n}={'knight' if s else 'knave'}" for n, s in zip(names, sol))
            q = (f"{quiz}\n\nDetermine each person's identity. Reason step by step, "
                 "then after '####' state your final answer as e.g. "
                 "'Zoey is a knight, Oliver is a knave.'")
            items.append({"question": q, "answer": gold})
        if not items:
            raise ValueError(f"kk 스키마 파싱 실패 (root={root}) — quiz/names/solution 필드 확인")
    elif dataset == "dapo-math":
        # 본실험용. 사전 배치본 우선. 스키마가 릴리스마다 달라 방어적으로 파싱한다.
        tried = [Path(os.environ["DAPO_DIR"])] if os.environ.get("DAPO_DIR") \
            else [b / n for b in _dataset_bases()
                  for n in ("dapo-math", "dapo-math-17k", "DAPO-Math-17k", "dapo_math")]
        root = next((c for c in tried if c.exists()), None)
        rows = _load_rows_any(root) if root else None
        if rows is None:
            try:
                from datasets import load_dataset
                rows = list(load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train"))
            except Exception as e:
                raise ValueError(
                    "dapo-math 로컬 사본을 못 찾았고 허브도 실패. 찾아본 위치:\n  "
                    + "\n  ".join(str(t) for t in tried)
                    + f"\nDAPO_DIR=<경로>로 지정 가능. (허브 오류: {e})"
                ) from e
        items = []
        for row in rows:
            q = row.get("prompt") or row.get("question") or row.get("problem")
            if isinstance(q, list):  # chat 형식이면 user 내용만
                q = next((m["content"] for m in q if m.get("role") == "user"), None)
            rm = row.get("reward_model") or {}
            a = rm.get("ground_truth") or row.get("answer") or row.get("solution")
            if q and a is not None:
                items.append({"question": str(q), "answer": str(a)})
        if not items:
            raise ValueError("DAPO-Math-17k 스키마 파싱 실패 — 필드명을 확인할 것")
    elif dataset == "apps":
        # 경쟁 프로그래밍 — stdin/stdout 실행 채점. 사전 배치본만 (fetch_datasets.sh apps).
        tried = [Path(os.environ["APPS_DIR"])] if os.environ.get("APPS_DIR") \
            else [b / "apps" for b in _dataset_bases()]
        root = next((c for c in tried if c.exists()), None)
        rows = _load_rows_any(root) if root else None
        if rows is None:
            raise ValueError("apps 로컬 사본 없음. 찾아본 위치:\n  "
                             + "\n  ".join(str(t) for t in tried)
                             + "\nAPPS_DIR=<경로>로 지정 가능. (fetch_datasets.sh apps)")
        items = []
        for r in rows:
            io = r.get("input_output")
            if isinstance(io, str):
                try:
                    io = json.loads(io)
                except ValueError:
                    continue
            if not isinstance(io, dict) or io.get("fn_name"):
                continue  # call-based 문제 제외 — stdin/stdout형만 채점 지원
            ins, outs = io.get("inputs") or [], io.get("outputs") or []
            if not ins or len(ins) != len(outs) or not all(isinstance(x, str) for x in ins):
                continue
            q = ("Write a Python program that reads from standard input and writes "
                 "the answer to standard output.\n\n" + str(r.get("question", ""))
                 + "\n\nReturn the complete program in a ```python code block.")
            # 테스트 8개 캡 — 채점 비용 상한 (APPS는 문제당 수십~수백 테스트)
            gold = "APPS:" + json.dumps(
                {"inputs": ins[:8], "outputs": [str(o) for o in outs[:8]]})
            items.append({"question": q, "answer": gold})
        if not items:
            raise ValueError(f"apps 스키마 파싱 실패 (root={root}) — input_output 필드 확인")
    else:
        raise ValueError(f"unknown dataset {dataset}")

    return _split(items, n_train, n_val, seed, dataset)


def _dataset_bases() -> list:
    """사전 배치본을 찾을 베이스 경로들 — 환경변수·공용·사용자 폴더 전부."""
    import os
    from pathlib import Path

    gv = os.environ.get("GROUP_VOLUME", "/group-volume")
    user = os.environ.get("OM_USER", "minsoo3.kim")
    om_work = os.environ.get("OM_WORK")
    cand = [os.environ.get("DATASETS_DIR"), f"{gv}/datasets", f"{gv}/{user}/datasets",
            f"{om_work}/data" if om_work else None]
    out, seen = [], set()
    for c in cand:
        if c and c not in seen:
            seen.add(c)
            out.append(Path(c))
    return out


def _boxed(solution: str) -> str | None:
    """\\boxed{...}의 내용물 — 중괄호 중첩을 세면서 뽑는다 (원본 MATH 정답 형식)."""
    i = solution.rfind("\\boxed{")
    if i == -1:
        return None
    depth, j = 1, i + len("\\boxed{")
    out = []
    while j < len(solution) and depth:
        ch = solution[j]
        depth += ch == "{"
        depth -= ch == "}"
        if depth:
            out.append(ch)
        j += 1
    return "".join(out) or None


def _load_rows_any(root) -> list[dict] | None:
    """디렉토리/파일에서 형식 무관하게 행을 읽는다 — jsonl > parquet > HF 스냅샷."""
    import json
    from pathlib import Path

    root = Path(root)
    if root.is_file():
        return [json.loads(l) for l in root.open()]
    files = sorted(root.rglob("*.jsonl"))
    if files:
        return [json.loads(l) for f in files for l in f.open()]
    pq = sorted(str(p) for p in root.rglob("*.parquet"))
    if pq:
        from datasets import load_dataset
        return list(load_dataset("parquet", data_files=pq, split="train"))
    # 원본 MATH 등 문제당 개별 .json 파일 트리 (HF 메타 파일은 제외)
    meta = {"dataset_info.json", "dataset_infos.json", "dataset_dict.json",
            "state.json", "config.json"}
    jf = [p for p in sorted(root.rglob("*.json")) if p.name not in meta]
    if jf:
        rows = []
        for p in jf:
            try:
                obj = json.loads(p.read_text())
            except (ValueError, OSError):
                continue
            if isinstance(obj, list):
                rows += [r for r in obj if isinstance(r, dict)]
            elif isinstance(obj, dict):
                rows.append(obj)
        if rows:
            return rows
    try:
        from datasets import load_from_disk
        obj = load_from_disk(str(root))
    except Exception:
        return None
    if hasattr(obj, "keys") and not hasattr(obj, "features"):  # DatasetDict
        return [r for k in obj.keys() for r in obj[k]]
    return list(obj)


def _split(items: list[dict], n_train: int, n_val: int, seed: int, name: str) -> dict:
    rng = random.Random(seed)
    rng.shuffle(items)
    need = n_train + n_val
    if len(items) < need:
        raise ValueError(f"{name}: {len(items)} < 필요 {need}")
    return {"train": items[:n_train], "val": items[n_train : n_train + n_val]}


PROMPT_TEMPLATE = (
    "Solve the following math problem. Reason step by step, then give the final "
    "answer after '####'.\n\nProblem: {question}\n"
)

_CODE_MARKER = "Your code should satisfy these tests"
_APPS_MARKER = "reads from standard input"


def build_user_msg(question: str) -> str:
    """수학 문제는 #### 템플릿으로 감싸고, 이미 완전한 지시문(mbpp/kk/apps)은 그대로."""
    if _CODE_MARKER in question or "####" in question or _APPS_MARKER in question:
        return question
    return PROMPT_TEMPLATE.format(question=question)


ANSWER_RE = re.compile(r"####\s*([^\n]+)")


def extract_answer(text: str) -> str | None:
    m = ANSWER_RE.search(text)
    if m:
        return m.group(1).strip().rstrip(".").replace(",", "").replace("$", "")
    # fallback: \boxed{...}
    m = re.search(r"\\boxed\{([^{}]+)\}", text)
    return m.group(1).strip() if m else None


def _extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    if m:
        return m.group(1)
    i = text.find("def ")
    return text[i:] if i != -1 else text


def _code_reward(text: str, tests: str) -> float:
    """생성 코드 + assert 테스트를 서브프로세스로 실행 — 전부 통과해야 1.0.

    타임아웃 8초(무한루프 방지), 실행은 -I(isolated)로 사용자 site 격리.
    """
    import subprocess
    import sys

    src = _extract_code(text) + "\n\n" + tests + "\n"
    try:
        # returncode만 필요 — 출력은 버린다 (print 폭주 코드의 메모리 폭발 방지)
        p = subprocess.run([sys.executable, "-I", "-c", src],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=8)
        return 1.0 if p.returncode == 0 else 0.0
    except (subprocess.TimeoutExpired, OSError):
        return 0.0


def _apps_reward(text: str, gold: str) -> float:
    """APPS stdin/stdout 채점 — 캡된 테스트 전부에서 stdout이 일치해야 1.0.

    타임아웃 8초/테스트. stdout 비교는 줄 단위 우측 공백·전체 앞뒤 공백 무시.
    (capture라 print 폭주 시 메모리 위험이 있으나 타임아웃이 상한 — mbpp와 달리
    출력 자체가 채점 대상이라 DEVNULL 불가.)
    """
    import json as _json
    import subprocess
    import sys

    io = _json.loads(gold[5:])
    code = _extract_code(text)
    if not code.strip():
        return 0.0
    for inp, want in zip(io["inputs"], io["outputs"]):
        try:
            p = subprocess.run([sys.executable, "-I", "-c", code],
                               input=inp, capture_output=True, text=True, timeout=8)
        except (subprocess.TimeoutExpired, OSError):
            return 0.0
        if p.returncode != 0:
            return 0.0
        got = "\n".join(l.rstrip() for l in p.stdout.strip().splitlines())
        exp = "\n".join(l.rstrip() for l in str(want).strip().splitlines())
        if got != exp:
            return 0.0
    return 1.0


def _kk_reward(text: str, gold: str) -> float:
    """이름별 knight/knave 전원 정답이어야 1.0 — 각 이름의 마지막 언급으로 판정."""
    pairs = [p.split("=", 1) for p in gold[3:].split(";") if "=" in p]
    seg = text.split("####")[-1] if "####" in text else text
    for name, role in pairs:
        ms = list(re.finditer(
            rf"\b{re.escape(name)}\b\s+is\s+(?:a|an)\s+(knight|knave)", seg, re.I))
        if not ms or ms[-1].group(1).lower() != role:
            return 0.0
    return 1.0


def reward(text: str, gold: str) -> float:
    if gold.lstrip().startswith("assert"):  # mbpp — 실행 채점
        return _code_reward(text, gold)
    if gold.startswith("APPS:"):  # apps — stdin/stdout 실행 채점
        return _apps_reward(text, gold)
    if gold.startswith("KK:"):  # knights & knaves — 전원 신원 매치
        return _kk_reward(text, gold)
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
