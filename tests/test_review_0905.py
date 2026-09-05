"""2026-09-05 검수 수정분 회귀 테스트 (torch 불필요).

    PYTHONPATH=src python3 tests/test_review_0905.py
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import extract_answer, reward  # noqa: E402
from gate_rules import run_seed  # noqa: E402

FAIL = 0


def check(name, cond):
    global FAIL
    print(("  ok  " if cond else "FAIL  ") + name)
    if not cond:
        FAIL += 1


# §4 — 마지막 '####'가 최종 답
check("last #### wins (multi-line)", extract_answer("a #### 12\nb #### 42.") == "42")
check("last #### wins (same line)", extract_answer("#### 3 then #### 4") == "4")
check("boxed fallback", extract_answer("\\boxed{7}") == "7")
check("comma/dollar normalised", reward("#### $1,000", "1000") == 1.0)
check("wrong answer still 0", reward("#### 41", "42") == 0.0)

# §3 — 동률 jitter 시드는 run_config.seed 하나
d = Path(tempfile.mkdtemp())
check("run_seed default 0 without config", run_seed(d) == 0)
(d / "run_config.json").write_text(json.dumps({"seed": 2}))
check("run_seed reads run_config", run_seed(d) == 2)
(d / "run_config.json").write_text("not json")
check("run_seed tolerates malformed config", run_seed(d) == 0)

# §1 — drift 학습 표본은 계약 검증 소스로 등록돼야 한다
import artifact_contract  # noqa: E402
src = Path(artifact_contract.__file__).read_text()
check("rollouts_drift_train registered as a contract source", '"rollouts_drift_train"' in src)
check("drift source not forced on legacy rescore", "rollouts_drift_train" not in artifact_contract.PRIMARY_SOURCES)

print("PASS" if FAIL == 0 else f"FAIL ({FAIL})")
sys.exit(1 if FAIL else 0)
