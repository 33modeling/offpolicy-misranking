"""reversal_freq 손계산 회귀 테스트 (모델 불필요, CPU 1초 미만).

    PYTHONPATH=src python3 tests/test_reversal_freq.py

합성 run(n=20, k=2, w=1, 동점 없음)에서 전 수치를 손으로 계산해 고정:
반전 집계·경계 대역·결정 피해(양수 승격만)·불일치 경보 Fisher·
g11 McNemar·닻(자기 불일치)까지 전부 정확 일치를 요구한다.
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reversal_freq import analyze_run, binom_2sided, fisher_exact_2x2  # noqa: E402

FAIL = 0


def check(name, cond):
    global FAIL
    print(("  ok  " if cond else "FAIL  ") + name)
    if not cond:
        FAIL += 1


ORACLE = {0: 0.9, 1: 0.8, 2: 0.7, 3: 0.6, 4: 0.5, 5: 0.4, 6: 0.3, 7: 0.2,
          8: 0.1, 9: -0.1, 10: -0.2, 11: -0.3, 12: -0.4, 13: -0.5,
          14: 0.0, 15: 0.0, 16: 0.05, 17: -0.05, 18: 0.15, 19: -0.15}
# g00: id0·9·19 반전, id8 무신호, id19가 est 1위(양수 승격 피해 검증)
G00 = {0: -0.5, 1: 0.64, 2: 0.56, 3: 0.48, 4: 0.40, 5: 0.32, 6: 0.24,
       7: 0.16, 8: 0.0, 9: 0.25, 10: -0.16, 11: -0.24, 12: -0.32,
       13: -0.40, 14: 0.1, 15: 0.0, 16: 0.22, 17: -0.04, 18: 0.12, 19: 0.9}
# g10: id2·3·9 반전 / g01: id9만 반전 → 불일치 D={2,3}, id9는 이중반전=일치
G10 = {i: (-0.2 if i == 2 else -0.15 if i == 3 else 0.1 if i == 9
           else ORACLE[i] * 0.7) for i in ORACLE}
G01 = {i: (0.12 if i == 9 else ORACLE[i] * 0.6) for i in ORACLE}
# g11: id2만 반전, id16 무신호
G11 = {i: (-0.1 if i == 2 else 0.0 if i == 16 else ORACLE[i] * 0.5)
       for i in ORACLE}
# 닻: id5·17 반부호, id14 a=0·id15 양쪽 0 → 분모 18, 반부호 2
HALF = {i: {"a": (0.0 if i == 14 else abs(ORACLE[i]) or 0.0),
            "b": (-0.1 if i in (5, 17) else (abs(ORACLE[i]) or 0.0))}
        for i in ORACLE}

with tempfile.TemporaryDirectory() as td:
    run = Path(td)
    (run / "score_protocol.json").write_text(json.dumps(
        {"schema": "offpolicy-score-validation-split/v1",
         "generation_validation": {"validated_rows": 1}}))
    (run / "oracle_protocol.json").write_text(json.dumps(
        {"schema": "offpolicy-oracle-validation-split/v1",
         "generation_validation": {"validated_rows": 1}}))
    (run / "scores_oracle.json").write_text(json.dumps(
        {str(i): {"score": v, "norm": 1.0} for i, v in ORACLE.items()}))
    (run / "scores_offpolicy.json").write_text(json.dumps(
        {est: {str(i): {"score": v, "norm": 1.0} for i, v in d.items()}
         for est, d in [("g00", G00), ("g10", G10), ("g01", G01), ("g11", G11)]}))
    (run / "scores_splithalf.json").write_text(json.dumps(
        {str(i): v for i, v in HALF.items()}))
    r = analyze_run(run)

check("n=20 k=2 w=1 무신호 2", (r["n"], r["k"], r["w"], r["oracle_zero"]) == (20, 2, 1, 2))
g = r["est"]
check("g00 반전 3/17", (g["g00"]["rev"], g["g00"]["nonzero"]) == (3, 17))
check("g00 경계 0/2", (g["g00"]["band_rev"], g["g00"]["band_n"]) == (0, 2))
check("g00 otop 1 (id0 음수화)", g["g00"]["otop_flipped"] == 1)
check("g00 etop 1 (id19 양수 승격만)", g["g00"]["etop_wrongdir"] == 1)
check("g10 반전 3/18·경계 1/2", (g["g10"]["rev"], g["g10"]["nonzero"],
                                 g["g10"]["band_rev"], g["g10"]["band_n"]) == (3, 18, 1, 2))
check("g01 반전 1/18", (g["g01"]["rev"], g["g01"]["nonzero"]) == (1, 18))
check("g11 반전 1/17 (id16 제외)", (g["g11"]["rev"], g["g11"]["nonzero"]) == (1, 17))

al = r["alarm"]
check("경보 분모 18·불일치 2", (al["base"], al["disagree"]) == (18, 2))
check("g10 경보표 [2,0,1,15]", al["by_est"]["g10"]["table"] == [2, 0, 1, 15])
check("g10 Fisher p=16/816", abs(al["by_est"]["g10"]["p"] - 16 / 816) < 1e-12)
check("g01 경보표 [0,2,1,15]·p=1", al["by_est"]["g01"]["table"] == [0, 2, 1, 15]
      and al["by_est"]["g01"]["p"] == 1.0)

vf = r["vs_full"]
check("McNemar g00 b3 c1 p=10/16", (vf["g00"]["b"], vf["g00"]["c"]) == (3, 1)
      and abs(vf["g00"]["p"] - 0.625) < 1e-12)
check("McNemar g10 b2 c0 p=0.5", (vf["g10"]["b"], vf["g10"]["c"]) == (2, 0)
      and abs(vf["g10"]["p"] - 0.5) < 1e-12)
check("McNemar g01 b1 c1 p=1", (vf["g01"]["b"], vf["g01"]["c"]) == (1, 1)
      and vf["g01"]["p"] == 1.0)

a = r["anchor"]
check("닻 2/18·경계 0/2", (a["flip"], a["n"], a["band_flip"], a["band_n"]) == (2, 18, 0, 2))
check("binom_2sided(0,0)=1", binom_2sided(0, 0) == 1.0)
check("fisher 빈 표=1", fisher_exact_2x2(0, 0, 0, 0) == 1.0)

print(("\n전체 통과" if FAIL == 0 else f"\n실패 {FAIL}건"))


def test_reversal_frequency_regressions() -> None:
    assert FAIL == 0


if __name__ == "__main__":
    sys.exit(1 if FAIL else 0)
