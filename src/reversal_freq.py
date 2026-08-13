"""부호반전 빈도 재집계 — 기존 산출물만으로 (CPU, 표준 라이브러리만).

    python3 src/reversal_freq.py <run_dir>            # 단일 run
    python3 src/reversal_freq.py <runs_root>          # v2-*/DONE 전체 + 풀링

리뷰어 질문("반전은 얼마나 자주 일어나는가, 경계 후보 중 몇 %가 피해자인가")에
답하는 프롬프트 단위 표를 만든다. 추정량별(g00/g10/g01/g11) 산출:

  1) 전체 부호반전율 — sign(est) != sign(oracle), 양쪽 비영(非零) 프롬프트 대상.
     무신호(score==0) 프롬프트는 분모에서 빼고 개수만 따로 보고.
  2) 경계 대역 반전율 — oracle 순위 k±w 대역(w=round(k/2))의 반전율.
     "top-k 경계 후보 중 반전 피해자 비율"의 직접 측정.
  3) 결정 피해 — oracle top-k인데 est가 음수(부호 뒤집혀 배제 방향),
     est top-k인데 oracle이 음수(잘못된 방향으로 승격).
  4) 불일치 경보 성능 — g10/g01 부호 불일치 시 vs 일치 시의 조건부 반전율
     + Fisher 정확검정. 주의: 스칼라 점수에서 이중 반전(둘 다 oracle 반대)이면
     두 셀은 서로 "일치"한다 — 즉 일치 시 반전율이 곧 경보의 사각지대 크기다.
     (본문 Prop. disagreement의 벡터 코사인 판과 estimand가 다름을 명시할 것.)

top-k 동점 처리는 readout_summary와 동일(Random(0) jitter) — 판정 무결성 보존.
"""

from __future__ import annotations

import json
import random
import sys
from math import comb
from pathlib import Path

ESTS = ("g00", "g10", "g01", "g11")


def topk_ids(scores: dict[int, float], k: int, rng: random.Random) -> set[int]:
    jit = {i: rng.random() for i in scores}
    return set(sorted(scores, key=lambda i: (-scores[i], jit[i]))[:k])


def ranks_desc(scores: dict[int, float], rng: random.Random) -> dict[int, int]:
    jit = {i: rng.random() for i in scores}
    order = sorted(scores, key=lambda i: (-scores[i], jit[i]))
    return {i: r + 1 for r, i in enumerate(order)}


def fisher_exact_2x2(a: int, b: int, c: int, d: int) -> float:
    """양측 Fisher 정확검정 p — 초기하 열거(관측 확률 이하 표 합산)."""
    n = a + b + c + d
    r1, c1 = a + b, a + c
    if n == 0 or r1 == 0 or r1 == n or c1 == 0 or c1 == n:
        return 1.0
    denom = comb(n, c1)

    def prob(x: int) -> float:
        return comb(r1, x) * comb(n - r1, c1 - x) / denom

    lo, hi = max(0, c1 - (n - r1)), min(r1, c1)
    p_obs = prob(a)
    return min(1.0, sum(p for x in range(lo, hi + 1)
                        if (p := prob(x)) <= p_obs * (1 + 1e-9)))


def load_run(run: Path) -> tuple[dict[int, float], dict[str, dict[int, float]]] | None:
    try:
        oracle = {int(i): v["score"] for i, v in
                  json.loads((run / "scores_oracle.json").read_text()).items()}
        off = json.loads((run / "scores_offpolicy.json").read_text())
    except Exception:
        return None
    ests = {e: {int(i): v["score"] for i, v in off[e].items() if int(i) in oracle}
            for e in ESTS if e in off}
    return oracle, ests


def analyze_run(run: Path) -> dict | None:
    loaded = load_run(run)
    if loaded is None:
        return None
    oracle, ests = loaded
    n = len(oracle)
    k = max(1, round(0.10 * n))
    w = max(1, round(k / 2))
    rng = random.Random(0)
    o_top = topk_ids(oracle, k, rng)
    o_rank = ranks_desc(oracle, random.Random(0))
    band = {i for i, r in o_rank.items() if k - w + 1 <= r <= k + w}
    o_nz = {i for i, s in oracle.items() if s != 0.0}

    out = {"run": run.name, "n": n, "k": k, "w": w,
           "oracle_zero": n - len(o_nz), "est": {}}

    for e, sc in ests.items():
        both = [i for i in o_nz if sc.get(i, 0.0) != 0.0]
        rev = [i for i in both if sc[i] * oracle[i] < 0]
        band_both = [i for i in both if i in band]
        band_rev = [i for i in band_both if sc[i] * oracle[i] < 0]
        e_top = topk_ids(sc, k, random.Random(0))
        out["est"][e] = {
            "nonzero": len(both), "rev": len(rev),
            "band_n": len(band_both), "band_rev": len(band_rev),
            "otop_flipped": sum(1 for i in o_top if sc.get(i, 0.0) < 0 < oracle[i]),
            "etop_wrongdir": sum(1 for i in e_top if oracle.get(i, 0.0) < 0),
        }

    # 불일치 경보 — g10 vs g01 부호 불일치가 (각 셀의) oracle 대비 반전을 예측하는가
    if "g10" in ests and "g01" in ests:
        base = [i for i in o_nz
                if ests["g10"].get(i, 0.0) != 0.0 and ests["g01"].get(i, 0.0) != 0.0]
        dis = {i for i in base if ests["g10"][i] * ests["g01"][i] < 0}
        alarm = {}
        for e in ("g10", "g01"):
            a = sum(1 for i in dis if ests[e][i] * oracle[i] < 0)          # 불일치·반전
            b = len(dis) - a                                               # 불일치·정상
            c = sum(1 for i in base if i not in dis
                    and ests[e][i] * oracle[i] < 0)                        # 일치·반전
            d = (len(base) - len(dis)) - c                                 # 일치·정상
            alarm[e] = {"table": [a, b, c, d], "p": fisher_exact_2x2(a, b, c, d)}
        out["alarm"] = {"base": len(base), "disagree": len(dis), "by_est": alarm}
    return out


def pct(num: int, den: int) -> str:
    return f"{num}/{den}" + (f" ({num / den:.0%})" if den else " (-)")


def report(results: list[dict]) -> str:
    L = ["# 부호반전 빈도 재집계 (reversal_freq)", ""]
    L.append("경계 대역 = oracle 순위 k±w (w=round(k/2)). 무신호(score 0) 제외, "
             "분모는 각 칸에 명시. top-k jitter는 readout과 동일(Random(0)).")
    for r in results:
        L += ["", f"## {r['run']} — n={r['n']}, k={r['k']}, w={r['w']}, "
                  f"oracle 무신호 {r['oracle_zero']}개", "",
              "| est | 전체 반전 | 경계 대역 반전 | oracle top-k 중 부호 뒤집혀 배제 방향 | est top-k 중 잘못된 방향 승격 |",
              "|---|---|---|---|---|"]
        for e, s in r["est"].items():
            L.append(f"| {e} | {pct(s['rev'], s['nonzero'])} "
                     f"| {pct(s['band_rev'], s['band_n'])} "
                     f"| {s['otop_flipped']}/{r['k']} | {s['etop_wrongdir']}/{r['k']} |")
        if "alarm" in r:
            al = r["alarm"]
            L += ["", f"불일치 경보 (g10↔g01, 분모 {al['base']}, 불일치 {al['disagree']}):"]
            for e, v in al["by_est"].items():
                a, b, c, d = v["table"]
                L.append(f"- {e}: P(반전|불일치)={pct(a, a + b)} vs "
                         f"P(반전|일치)={pct(c, c + d)} — Fisher 양측 p={v['p']:.3g}")
            L.append("- 주의: 스칼라 점수에서 이중 반전은 정의상 '일치'로 나타난다 — "
                     "P(반전|일치)가 경보 사각지대의 크기다.")
    # 풀링 (run 2개 이상일 때)
    if len(results) > 1:
        L += ["", "## 풀링 (전 run 합산)", "",
              "| est | 전체 반전 | 경계 대역 반전 |", "|---|---|---|"]
        for e in ESTS:
            rows = [r["est"][e] for r in results if e in r["est"]]
            if rows:
                L.append(f"| {e} | {pct(sum(s['rev'] for s in rows), sum(s['nonzero'] for s in rows))} "
                         f"| {pct(sum(s['band_rev'] for s in rows), sum(s['band_n'] for s in rows))} |")
    L += ["", "```json", json.dumps(results, ensure_ascii=False), "```"]
    return "\n".join(L)


def main() -> int:
    root = Path(sys.argv[1])
    if (root / "scores_oracle.json").exists():
        runs = [root]
    else:
        runs = [d for d in sorted(root.glob("v2-*"))
                if (d / "DONE").exists() and "smoke" not in d.name]
    results = [a for d in runs if (a := analyze_run(d))]
    if not results:
        print(f"산출물 없음: {root} (scores_oracle.json 또는 v2-*/DONE 필요)")
        return 1
    print(report(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
