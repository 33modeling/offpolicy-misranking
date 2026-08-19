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
     est top-k인데 oracle이 음수(잘못된 방향 승격 — est 점수가 양수인 것만
     집계: 무신호 우세 pool에서 동점 jitter로 편입된 0점 프롬프트 제외).
     est top-k 동점 처리는 readout_summary와 같은 단일 jitter 스트림.
  4) 불일치 경보 성능 — g10/g01 부호 불일치 시 vs 일치 시의 조건부 반전율
     + Fisher 정확검정. 주의: 스칼라 점수에서 이중 반전(둘 다 oracle 반대)이면
     두 셀은 서로 "일치"한다 — 즉 일치 시 반전율이 곧 경보의 사각지대 크기다.
     (본문 Prop. disagreement의 벡터 코사인 판과 estimand가 다름을 명시할 것.)
  5) 닻 — oracle 자기 부호 불일치율(scores_splithalf의 a·b 반부호 비율).
     추정량 반전율은 이 값 이하로 내려갈 이유가 없다(oracle 노이즈 기여분).
     닻과의 차이만 추정량 결함으로 읽을 것 — below-chance 단독 과판매 금지
     원칙과 같은 취지.

비영 프롬프트가 10개 미만인 run(무신호 우세)의 결정 칸(top-k 두 열)은 동점
jitter 인공물이므로 †로 표시하고 해석하지 않는다.

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


def binom_2sided(b: int, n: int) -> float:
    """McNemar 정확검정 — 불일치쌍 n 중 b, 양측(관측 확률 이하 합산)."""
    if n == 0:
        return 1.0
    pmf = [comb(n, i) / 2 ** n for i in range(n + 1)]
    return min(1.0, sum(p for p in pmf if p <= pmf[b] * (1 + 1e-9)))


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
    from select_rules import topk_count
    k = topk_count(n, 0.10)
    w = max(1, round(k / 2))
    rng = random.Random(0)
    o_top = topk_ids(oracle, k, rng)
    o_rank = ranks_desc(oracle, random.Random(0))
    band = {i for i, r in o_rank.items() if k - w + 1 <= r <= k + w}
    o_nz = {i for i, s in oracle.items() if s != 0.0}

    out = {"run": run.name, "n": n, "k": k, "w": w,
           "oracle_zero": n - len(o_nz), "est": {}}

    # 닻 — oracle 자기 부호 불일치 (split-half a·b 반부호)
    try:
        half = {int(i): v for i, v in
                json.loads((run / "scores_splithalf.json").read_text()).items()}
        hb = [i for i, v in half.items() if v["a"] != 0.0 and v["b"] != 0.0]
        out["anchor"] = {
            "n": len(hb),
            "flip": sum(1 for i in hb if half[i]["a"] * half[i]["b"] < 0),
            "band_n": len([i for i in hb if i in band]),
            "band_flip": sum(1 for i in hb
                             if i in band and half[i]["a"] * half[i]["b"] < 0),
        }
    except Exception:
        pass

    for e, sc in ests.items():
        both = [i for i in o_nz if sc.get(i, 0.0) != 0.0]
        rev = [i for i in both if sc[i] * oracle[i] < 0]
        band_both = [i for i in both if i in band]
        band_rev = [i for i in band_both if sc[i] * oracle[i] < 0]
        e_top = topk_ids(sc, k, rng)
        out["est"][e] = {
            "nonzero": len(both), "rev": len(rev),
            "band_n": len(band_both), "band_rev": len(band_rev),
            "otop_flipped": sum(1 for i in o_top if sc.get(i, 0.0) < 0 < oracle[i]),
            "etop_wrongdir": sum(1 for i in e_top
                                 if sc[i] > 0 and oracle.get(i, 0.0) < 0),
        }

    # g11 대비 짝지은 초과 반전 — 같은 프롬프트에서 one-sided만 반전(b) vs
    # g11만 반전(c), McNemar 정확검정. oracle 노이즈가 양변에 동일하게 걸려
    # 닻 없이도 성립하는 대비.
    if "g11" in ests:
        vs_full = {}
        for e in ("g00", "g10", "g01"):
            if e not in ests:
                continue
            both = [i for i in o_nz
                    if ests[e].get(i, 0.0) != 0.0 and ests["g11"].get(i, 0.0) != 0.0]
            b = sum(1 for i in both
                    if ests[e][i] * oracle[i] < 0 <= ests["g11"][i] * oracle[i])
            c = sum(1 for i in both
                    if ests["g11"][i] * oracle[i] < 0 <= ests[e][i] * oracle[i])
            vs_full[e] = {"b": b, "c": c, "p": binom_2sided(b, b + c)}
        out["vs_full"] = vs_full

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
        degen = any(s["nonzero"] < 10 for s in r["est"].values())
        for e, s in r["est"].items():
            dg = "†" if s["nonzero"] < 10 else ""
            L.append(f"| {e} | {pct(s['rev'], s['nonzero'])} "
                     f"| {pct(s['band_rev'], s['band_n'])} "
                     f"| {s['otop_flipped']}/{r['k']}{dg} | {s['etop_wrongdir']}/{r['k']}{dg} |")
        if "anchor" in r:
            a = r["anchor"]
            L.append(f"| **닻: oracle 자기 불일치** | {pct(a['flip'], a['n'])} "
                     f"| {pct(a['band_flip'], a['band_n'])} | — | — |")
        if degen:
            L.append("")
            L.append("† 비영 프롬프트 10개 미만 — top-k가 동점 jitter로 채워져 "
                     "결정 칸은 인공물, 해석 금지.")
        if "vs_full" in r:
            L += ["", "g11 대비 짝지은 초과 반전 (McNemar 정확, one-sided만 반전 b vs g11만 반전 c):"]
            for e, v in r["vs_full"].items():
                L.append(f"- {e}: b={v['b']} vs c={v['c']} — 양측 p={v['p']:.3g}")
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
        anchors = [r["anchor"] for r in results if "anchor" in r]
        if anchors:
            L.append(f"| **닻: oracle 자기 불일치** | "
                     f"{pct(sum(a['flip'] for a in anchors), sum(a['n'] for a in anchors))} | "
                     f"{pct(sum(a['band_flip'] for a in anchors), sum(a['band_n'] for a in anchors))} |")
        L.append("")
        L.append("풀링 주의: 같은 프롬프트 풀의 seed 반복이면 독립 표본이 아니다 — "
                 "재현성 확인용이지 p-값 결합 근거가 아님.")
    L += ["", "```json", json.dumps(results, ensure_ascii=False), "```"]
    return "\n".join(L)


def main() -> int:
    root = Path(sys.argv[1])
    if (root / "scores_oracle.json").exists():
        runs = [root]
    else:
        # v2-*는 DONE 필수, v1 계열(drift*/gate-*)은 산출물 존재로 완결 판정
        runs = [d for d in sorted(root.iterdir()) if d.is_dir()
                and "smoke" not in d.name
                and (d / "scores_oracle.json").exists()
                and ((d / "DONE").exists() or not d.name.startswith("v2-"))]
    results = [a for d in runs if (a := analyze_run(d))]
    if not results:
        print(f"산출물 없음: {root} (scores_oracle.json 또는 v2-*/DONE 필요)")
        return 1
    print(report(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
