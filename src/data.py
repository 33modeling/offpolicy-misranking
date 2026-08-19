"""데이터 로딩 — 파일럿 기본 GSM8K, 옵션 MATH-500. 본실험은 DAPO-Math-17k로 확장.

H100 클러스터는 GitHub egress가 없으므로 HF 미러를 쓴다:
  export HF_ENDPOINT=<사내 미러>  (huggingface_hub가 자동 인식)
"""

from __future__ import annotations

import json
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
    for local in (Path(os.environ.get("OM_DATA", "")) / fname,
                  Path(os.environ.get("DATASETS_DIR", "")) / dataset / fname):
        # '존재하는 첫 파일' 채택은 손상/빈 사본이 정상 사본을 가린다(E4형) —
        # 판독까지 성공해야 채택, 실패하면 다음 후보→일반 탐색으로 넘어간다
        if not local.is_file():
            continue
        try:
            rows = [json.loads(l) for l in local.open()]
            if dataset == "gsm8k":
                items = [
                    {"question": r["question"], "answer": _gsm8k_answer(r["answer"])}
                    for r in rows
                ]
            else:  # math500
                items = [{"question": r["problem"], "answer": str(r["answer"])} for r in rows]
        except (ValueError, KeyError, OSError) as e:
            print(f"[data] 로컬 사본 판독 실패, 건너뜀: {local} ({type(e).__name__}: {e})")
            continue
        if items:
            return _split(items, n_train, n_val, seed, f"{dataset}(local)")
        print(f"[data] 로컬 사본 0행, 건너뜀: {local}")

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
        # fuzzy에서 "math"를 빼는 이유: dapo-math 폴더가 부분 일치로 잡힌다
        tried = _candidate_roots("MATH500_DIR",
                                 ("math500", "MATH-500", "math-500", "math_500", "math", "MATH"),
                                 fuzzy=("math500", "math-500", "math_500"))
        root, rows = _load_rows_first(tried)
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
        tried = _candidate_roots("MBPP_DIR", ("mbpp",))
        root, rows = _load_rows_first(tried)
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
            text = (r.get("text") or r.get("prompt") or r.get("description")
                    or r.get("instruction") or r.get("task_description"))
            tests = (r.get("test_list") or r.get("tests") or r.get("test")
                     or r.get("challenge_test_list"))
            if isinstance(tests, str):
                tests = _maybe_json_list(tests) or [tests]
            if not (text and tests):
                continue
            tests_str = "\n".join(tests)
            q = (f"Write a Python function for the task below.\n\n{text}\n\n"
                 f"Your code should satisfy these tests:\n{tests_str}\n\n"
                 "Return the complete function in a ```python code block.")
            # answer 자리에 assert 테스트를 넣는다 — reward()가 실행 채점으로 분기
            items.append({"question": q, "answer": tests_str})
        if not items:
            keys = sorted(rows[0].keys()) if rows else []
            raise ValueError(f"mbpp 스키마 파싱 실패 (root={root}, rows={len(rows or [])}, "
                             f"첫 행 필드={keys}) — 필드명을 확인할 것")
    elif dataset == "kk":
        # Knights & Knaves 논리 퍼즐 — 사전 배치본($DATASETS_DIR/kk 등)
        tried = _candidate_roots("KK_DIR", ("kk", "knights-and-knaves", "knights_and_knaves"),
                                 fuzzy=("knights",))
        root, rows = _load_rows_first(tried)
        if rows is None:
            raise ValueError("kk 로컬 사본 없음. 찾아본 위치:\n  "
                             + "\n  ".join(str(t) for t in tried)
                             + "\nKK_DIR=<경로>로 지정 가능.")
        items = []
        for r in rows:
            quiz, names, sol = r.get("quiz"), r.get("names"), r.get("solution")
            if isinstance(names, str):
                names = _maybe_json_list(names)
            if isinstance(sol, str):
                sol = _maybe_json_list(sol)
            if not (quiz and names and isinstance(sol, list) and len(sol) == len(names)):
                continue
            gold = "KK:" + ";".join(
                f"{n}={'knight' if _kk_truthy(s) else 'knave'}"
                for n, s in zip(names, sol, strict=True))
            q = (f"{quiz}\n\nDetermine each person's identity. Reason step by step, "
                 "then after '####' state your final answer as e.g. "
                 "'Zoey is a knight, Oliver is a knave.'")
            items.append({"question": q, "answer": gold})
        if not items:
            keys = sorted(rows[0].keys()) if rows else []
            raise ValueError(f"kk 스키마 파싱 실패 (root={root}, rows={len(rows or [])}, "
                             f"첫 행 필드={keys}) — quiz/names/solution 필드 확인")
    elif dataset == "dapo-math":
        # 본실험용. 사전 배치본 우선. 스키마가 릴리스마다 달라 방어적으로 파싱한다.
        tried = _candidate_roots("DAPO_DIR",
                                 ("dapo-math", "dapo-math-17k", "DAPO-Math-17k", "dapo_math"),
                                 fuzzy=("dapo",))
        root, rows = _load_rows_first(tried)
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
        tried = _candidate_roots("APPS_DIR", ("apps",))
        root, rows = _load_rows_first(tried)
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


def _maybe_json_list(s: str) -> list | None:
    """'["a", "b"]' 꼴 문자열-인코딩 리스트 복원 — parquet/jsonl 내보내기 시
    리스트 필드가 문자열로 변형된 사본(pandas to_json 등)을 수용한다."""
    s = s.strip()
    if not (s.startswith("[") and s.endswith("]")):
        return None
    try:
        v = json.loads(s)
    except ValueError:
        try:
            import ast
            v = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return None
    return v if isinstance(v, list) else None


def _kk_truthy(s) -> bool:
    """kk solution 원소 — bool/int 외에 'knight'/'true' 문자열 변형 수용."""
    if isinstance(s, str):
        return s.strip().lower() in ("knight", "true", "1")
    return bool(s)


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
        # HF 스냅샷은 설정 폴더별(full/·sanitized/ 등)로 컬럼이 달라 통짜 로드가
        # CastError로 죽는다 — 그룹 키는 마지막 폴더명(중첩 스냅샷의
        # snapshots/<해시>/<설정>도 설정명으로 잡힘). full/이 있으면 그것만
        # (sanitized 등 부분집합 뷰와 중복 방지), 없으면 로드되는 그룹 전부 병합
        # (kk처럼 설정=분할인 데이터셋의 침묵 부분 로드 방지).
        groups: dict[str, list[str]] = {}
        for f in pq:
            parent = Path(f).parent
            groups.setdefault("." if parent == root else parent.name, []).append(f)
        order = sorted(groups, key=lambda g: (not g.startswith("full"), g))
        rows: list[dict] = []
        for g in order:
            try:
                part = list(load_dataset("parquet", data_files=groups[g], split="train"))
            except Exception:
                continue
            rows += part
            if g.startswith("full"):
                break
        if not rows:
            for f in pq:
                try:
                    rows += list(load_dataset("parquet", data_files=f, split="train"))
                except Exception:
                    continue
        if rows:
            return rows
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


def _candidate_roots(env_var: str, names: tuple, fuzzy: tuple | None = None) -> list:
    """<ENV> 지정 시 그 경로만. 아니면 베이스 아래 통상 이름 + 이름이 든 하위
    폴더(깊이 2)까지 — hf download가 만드는 datasets--org--name/snapshots/<해시>/
    같은 중첩·변형 구조도 후보로 잡는다. fuzzy로 부분 일치 키를 좁힐 수 있다
    (예: math500이 "math" 매칭으로 dapo-math를 잡는 오인 방지)."""
    import os
    from pathlib import Path

    if os.environ.get(env_var):
        return [Path(os.environ[env_var])]
    bases = _dataset_bases()
    tried = [b / n for b in bases for n in names]
    keys = tuple(k.lower() for k in (fuzzy or names))
    found = set()
    for b in bases:
        if not b.is_dir():
            continue
        for pat in ("*", "*/*"):
            for p in b.glob(pat):
                try:
                    if p.is_dir() and any(k in p.name.lower() for k in keys) \
                            and p not in tried:
                        found.add(p)
                except OSError:
                    continue
    return tried + sorted(found, key=lambda p: (len(p.name), str(p)))


def _load_rows_first(tried: list) -> tuple:
    """후보를 순서대로 실제로 읽어, 행이 나오는 첫 경로를 채택한다.

    '존재하는 첫 경로' 선택은 중단된 fetch가 남긴 빈 디렉터리가 뒤의 실사본을
    가리는 버그를 만든다(E4 재발형) — 존재가 아니라 읽기 성공이 기준이어야 한다."""
    from pathlib import Path

    for c in tried:
        if not Path(c).exists():
            continue
        try:
            rows = _load_rows_any(c)
        except Exception as e:  # 손상/부분 파일(비원자 쓰기 잔재)은 다음 후보로
            print(f"[data] 후보 판독 실패, 건너뜀: {c} ({type(e).__name__}: {e})")
            continue
        if rows:
            return c, rows
    return None, None


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


def _run_untrusted_python(code: str, stdin: str = "", timeout: int = 8):
    """Run generated code in a fail-closed bubblewrap sandbox."""
    import os
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path

    bwrap = shutil.which("bwrap")
    python = "/usr/bin/python3"
    runner = Path(__file__).resolve().parents[1] / "scripts" / "code_sandbox.py"
    if not bwrap or not os.path.isfile(python) or not runner.is_file():
        return None

    command = [
        bwrap,
        "--unshare-all", "--die-with-parent", "--new-session",
        "--clearenv", "--setenv", "PATH", "/usr/bin", "--setenv", "LANG", "C.UTF-8",
        "--ro-bind", "/usr", "/usr",
        "--ro-bind", "/lib", "/lib",
        "--ro-bind", str(runner), "/runner.py",
    ]
    for path in ("/lib64", "/etc/ld.so.cache"):
        if os.path.exists(path):
            command += ["--ro-bind", path, path]
    command += [
        "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
        "--chdir", "/tmp", python, "-I", "/runner.py", str(timeout), code,
    ]

    try:
        with tempfile.TemporaryFile() as output:
            process = subprocess.run(
                command,
                input=stdin.encode(),
                stdout=output,
                stderr=subprocess.DEVNULL,
                timeout=timeout,
                check=False,
            )
            output.seek(0)
            stdout = output.read(1024 * 1024 + 1)
        if len(stdout) > 1024 * 1024:
            return None
        return process.returncode, stdout.decode(errors="replace")
    except (subprocess.TimeoutExpired, OSError, ValueError):
        return None


def _code_reward(text: str, tests: str) -> float:
    """Run generated functions and asserts in an isolated sandbox."""
    src = _extract_code(text) + "\n\n" + tests + "\n"
    result = _run_untrusted_python(src)
    return 1.0 if result is not None and result[0] == 0 else 0.0


def _apps_reward(text: str, gold: str) -> float:
    """APPS stdin/stdout 채점 — 캡된 테스트 전부에서 stdout이 일치해야 1.0.

    타임아웃 8초/테스트. stdout 비교는 줄 단위 우측 공백·전체 앞뒤 공백 무시.
    출력은 1 MiB로 제한하고 호스트 파일·네트워크 접근을 허용하지 않는다.
    """
    import json as _json

    io = _json.loads(gold[5:])
    code = _extract_code(text)
    if not code.strip():
        return 0.0
    inputs, outputs = io["inputs"], io["outputs"]
    if len(inputs) != len(outputs):
        return 0.0
    for inp, want in zip(inputs, outputs, strict=True):
        result = _run_untrusted_python(code, stdin=inp)
        if result is None or result[0] != 0:
            return 0.0
        got = "\n".join(l.rstrip() for l in result[1].strip().splitlines())
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
