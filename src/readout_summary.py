"""사람이 읽는 판독 보고서 생성 — READOUT.md의 본문.

    python3 src/readout_summary.py <runs_root>

구성: ① 한눈 요약 표(run × 수치 × 평문 판정) ② 자동 결론 ③ 용어 설명
④ 상세(원시 judge 출력). PASS/FAIL 이중부정 없이 전부 평문으로 쓴다.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from gate_rules import evaluate_causal_run, has_valid_analysis_protocol

def precisions(run: Path) -> tuple[dict, int, float] | None:
    if not has_valid_analysis_protocol(run):
        return None
    try:
        oracle = {int(i): v["score"] for i, v in
                  json.loads((run / "scores_oracle.json").read_text()).items()}
        off = json.loads((run / "scores_offpolicy.json").read_text())
    except Exception:
        return None
    n = len(oracle)
    from select_rules import overlap_under_independent_ties, topk_count
    k = topk_count(n, 0.10)
    out = {}
    for est in ("g00", "g10", "g01", "g11"):
        if est not in off:
            continue
        sc = {int(i): v["score"] for i, v in off[est].items() if int(i) in oracle}
        out[est] = overlap_under_independent_ties(oracle, sc, k, seed=0).mean
    return out, k, k / n


def main() -> int:
    root = Path(sys.argv[1])
    runs = [d for d in sorted(root.glob("v2-*"))
            if (d / "DONE").exists() and "smoke" not in d.name]

    rows, details, concl = [], [], []
    for d in runs:
        state = evaluate_causal_run(d)
        rep = state["report"] or {}
        floor = rep.get("noise_floor")
        pr = precisions(d)
        if pr is None:
            continue
        prec, k, chance = pr

        # judge와 같은 사전 문턱, 같은 run의 joint predicate를 그대로 사용한다.
        if state["axis_failures"] is not None:
            onesided = "예 (사전 문턱)" if state["joint_failure"] else "아니오"
        else:
            onesided = "판정 불가"

        valid_hybrid = [r for r in state["hybrid_results"] if "error" not in r]
        eligible_hybrid = [r for r in valid_hybrid if r["eligible"]]
        if valid_hybrid:
            dip_votes = [
                max(r["precision"]["bp"], r["precision"]["pb"])
                < r["precision"]["bb"]
                for r in valid_hybrid
            ]
            if not state["joint_failure"]:
                hyb = "C1 미충족"
            elif state["witnesses"]:
                hyb = f"예 (cut={state['causal_cut']})"
            elif not eligible_hybrid:
                hyb = "사전고정 cut 없음"
            else:
                hyb = "아니오"
            dip = "예" if all(dip_votes) else ("일부" if any(dip_votes) else "아니오")
        else:
            hyb, dip = "데이터 없음", "-"

        jd = subprocess.run(
            [sys.executable, "src/judge.py", str(d)],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        ).stdout

        f_str = f"{floor:.3f}" if isinstance(floor, (int, float)) else "?"
        if rep.get("_recomputed"):
            f_str += "†"
        rows.append(
            f"| {d.name} | {f_str} | {chance:.2f} | "
            + " | ".join(f"{prec.get(e, float('nan')):.3f}" for e in ("g00", "g10", "g01", "g11"))
            + f" | {onesided} | {hyb} | {dip} |")
        details.append(f"<details><summary>{d.name} 원시 출력</summary>\n\n```\n{jd.strip()}\n```\n</details>\n")

        tag = "gsm8k" if "dapo" not in d.name and "math500" not in d.name else \
              ("dapo" if "dapo" in d.name else "math500")
        concl.append((tag, onesided.startswith("예"), hyb))

    print("# 판독 보고서\n")
    print("## 한눈 요약\n")
    print("| run | floor | chance | g00 | g10 | g01 | g11 | one-sided가 더 나쁜가 | hybrid 회복 | mixed-dip |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(r)

    print("\n## 자동 결론\n")
    for tag in ("gsm8k", "dapo", "math500"):
        sub = [c for c in concl if c[0] == tag]
        if not sub:
            continue
        yes = sum(1 for c in sub if c[1])
        print(f"- **{tag}**: one-sided 열세 {yes}/{len(sub)} run에서 관찰. "
              f"hybrid 회복: {', '.join(c[2] for c in sub)}")
    print("\n(주의: run 수가 적으면 위 관찰은 통계적 확정이 아님 — 5-seed 전승이 유의선)")

    print("\n## 용어 — 표를 읽는 법\n")
    print("- **floor**: oracle 절반끼리의 일치도. †는 원시 점수에서 독립 tie stream으로 재계산한 정본")
    print("- **chance**: 아무거나 찍었을 때의 기대 precision")
    print("- **g00/g10/g01/g11**: 무보정 / prefix만 / suffix만 / 전부 보정의 top-k precision")
    print("- **one-sided가 더 나쁜가**: 동일 run에서 g10·g01 모두 floor보다 0.15 이상 낮으면 '예'")
    print("- **hybrid 회복**: C1을 만족한 동일 run의 사전고정 cut=0.5에서 pp가 pb·bp보다 모두 높으면 '예'")
    print("- **mixed-dip**: 혼합 셀(bp·pb)이 순수 stale(bb)보다 낮으면 '예'")

    print("\n## 상세 (원시 출력)\n")
    for dt in details:
        print(dt)
    if not rows:
        print("[abort] corrected protocol을 만족하는 run이 없음", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
