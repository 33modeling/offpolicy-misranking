"""사람이 읽는 판독 보고서 생성 — READOUT.md의 본문.

    python3 src/readout_summary.py <runs_root>

구성: ① 한눈 요약 표(run × 수치 × 평문 판정) ② 자동 결론 ③ 용어 설명
④ 상세(원시 judge 출력). PASS/FAIL 이중부정 없이 전부 평문으로 쓴다.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

def precisions(run: Path) -> tuple[dict, int, float] | None:
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
    from run_select import describe_skips, iter_runs
    runs = iter_runs(root)
    if not runs:
        print(f"# 판독 대상 없음 — {root}", file=sys.stderr)
        for line in describe_skips(root, runs):
            print(f"#   {line}", file=sys.stderr)
        return 1

    rows, details, concl = [], [], []
    for d in runs:
        rep = {}
        try:
            rep = json.loads((d / "report.json").read_text())
        except Exception:
            pass
        floor = rep.get("noise_floor")
        # 교정 floor — scores_splithalf에서 절반별 독립 jitter로 재계산
        floor_fixed = None
        try:
            hv = {int(i): v for i, v in
                  json.loads((d / "scores_splithalf.json").read_text()).items()}
            from select_rules import overlap_under_independent_ties, topk_count
            kk = topk_count(len(hv), 0.10)
            floor_fixed = overlap_under_independent_ties(
                {i: h["a"] for i, h in hv.items()},
                {i: h["b"] for i, h in hv.items()},
                kk, seed=0,
            ).mean
        except Exception:
            pass
        pr = precisions(d)
        if pr is None:
            continue
        prec, k, chance = pr

        # 평문 판정 1 — one-sided가 무보정보다 나쁜가 (논문 방향)
        if prec.get("g10") is not None and prec.get("g01") is not None:
            worse = prec["g10"] < prec["g00"] and prec["g01"] < prec["g00"]
            onesided = "예 (논문 방향)" if worse else "아니오"
        else:
            onesided = "판정 불가"

        # 평문 판정 2 — hybrid 회복 (judge 출력에서 셀 수치 파싱, 최신 컷 기준)
        jd = subprocess.run([sys.executable, "src/judge.py", str(d)],
                            capture_output=True, text=True, timeout=600).stdout
        cells = re.findall(
            r"cut=([\d.]+): bb=([\d.]+) bp=([\d.]+) pb=([\d.]+) pp=([\d.]+)", jd)
        if cells:
            rec_votes = [(float(pp) > float(pb)) and (float(pp) > float(bp))
                         for _, bb, bp, pb, pp in cells]
            dip_votes = [max(float(bp), float(pb)) < float(bb)
                         for _, bb, bp, pb, pp in cells]
            hyb = ("예" if all(rec_votes) else
                   "아니오" if not any(rec_votes) else
                   f"부분 ({sum(rec_votes)}/{len(rec_votes)}컷)")
            dip = "예" if all(dip_votes) else ("일부" if any(dip_votes) else "아니오")
        else:
            hyb, dip = "데이터 없음", "-"

        f_str = f"{floor:.3f}" if isinstance(floor, (int, float)) else "?"
        if floor_fixed is not None:
            f_str += f"→{floor_fixed:.3f}(교정)"
        rows.append(
            f"| {d.name} | {f_str} | {chance:.2f} | "
            + " | ".join(f"{prec[e]:.3f}" if e in prec else "산출없음"
                         for e in ("g00", "g10", "g01", "g11"))
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
    print("- **floor**: oracle 절반끼리의 일치도. `구값→교정값` 표기 — 구값은 동점 절단 공유-jitter 버그로 부풀려질 수 있음(동점 많은 체제), **교정값(독립 jitter)이 정본**")
    print("- **chance**: 아무거나 찍었을 때의 기대 precision")
    print("- **g00/g10/g01/g11**: 무보정 / prefix만 / suffix만 / 전부 보정의 top-k precision")
    print("- **one-sided가 더 나쁜가**: g10·g01 둘 다 g00보다 낮으면 '예' = 논문 주장 방향")
    print("- **hybrid 회복**: 빠진 반쪽을 복원한 pp가 pb·bp보다 높으면 '예' = 인과 주장 방향")
    print("- **mixed-dip**: 혼합 셀(bp·pb)이 순수 stale(bb)보다 낮으면 '예'")

    print("\n## 상세 (원시 출력)\n")
    for dt in details:
        print(dt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
