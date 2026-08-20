"""D6 — GPU 0 통계 보강: 기존 산출물만으로 정확 p-값과 부트스트랩 CI.

    python3 src/stats_extra.py <run_dir> [frac] [boot] [seed]   # 기본 0.10 2000 0
    python3 src/stats_extra.py --sign <wins> <losses> [ties]    # 부호검정만

출력 3종 (전부 CPU, 표준 라이브러리만):
  1) 추정량별 top-k precision + 초기하 p-값 P(overlap <= 관측 | 무작위 선택)
     — "below chance" 주장의 정확 유의성. 단독 셀은 약할 수 있음(예: n=256,
     k=25에서 overlap 0의 p=0.067) — 다조건 결합·부호검정과 함께 쓸 것.
  2) 프롬프트 부트스트랩 95% CI — seed 반복 없이 프롬프트 재표집으로 오차대
     (seed 분산과 다른 분산원임을 본문에 명시할 것).
  3) 부호검정: hybrid 축별 회복 w승/l패/t동률의 exact one-sided binomial p.
"""

from __future__ import annotations

import random
import sys
from math import comb
from pathlib import Path

from gate_rules import has_valid_analysis_protocol
from score_artifacts import ESTIMATORS, load_complete_score_artifacts


def hyp_p_le(n: int, big_k: int, k: int, x: int) -> float:
    """P(X <= x), X ~ Hypergeom(모집단 n, 성공 big_k, 추출 k)."""
    tot = comb(n, k)
    return sum(comb(big_k, i) * comb(n - big_k, k - i)
               for i in range(0, min(x, big_k, k) + 1)) / tot


def topk(scores: dict, k: int, rng: random.Random) -> set:
    jit = {i: rng.random() for i in scores}
    return set(sorted(scores, key=lambda i: (-scores[i], jit[i]))[:k])


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "--sign":
        if len(sys.argv) < 4:
            raise ValueError("--sign requires wins and losses; ties are optional")
        w, loss = int(sys.argv[2]), int(sys.argv[3])
        ties = int(sys.argv[4]) if len(sys.argv) > 4 else 0
        n = w + loss
        p_value = sum(comb(n, i) for i in range(w, n + 1)) / (2 ** n) if n else 1.0
        print(f"부호검정: {w}승 {loss}패 {ties}동률(제외) → "
              f"one-sided p = {p_value:.3e}")
        return 0
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    run = Path(sys.argv[1])
    if not has_valid_analysis_protocol(run):
        raise ValueError("corrected independent-validation score protocol is missing")
    frac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.10
    boot = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    rng = random.Random(seed)

    artifacts = load_complete_score_artifacts(run)
    oracle = artifacts.oracle
    n = len(oracle)
    from select_rules import topk_count
    k = topk_count(n, frac)
    print(f"run={run.name}  n={n}  k={k}  chance={k / n:.3f}")
    print(f"{'est':>6} {'prec':>6} {'overlap':>9} {'P(<=x|rand)':>16}   bootstrap95%CI")

    idx = list(oracle)
    for est in ESTIMATORS:
        sc = artifacts.offpolicy[est]
        from select_rules import overlap_under_independent_ties
        overlap = overlap_under_independent_ties(oracle, sc, k, seed=seed)
        xs = [round(value * k) for value in overlap.values]
        ps = [hyp_p_le(n, k, k, x) for x in xs]
        # 프롬프트 부트스트랩: (oracle, est) 점수 쌍을 재표집해 precision 분포
        bs = []
        for _ in range(boot):
            samp = [idx[rng.randrange(n)] for _ in range(n)]
            o2 = {j: oracle[i] for j, i in enumerate(samp)}
            e2 = {j: sc.get(i, min(sc.values())) for j, i in enumerate(samp)}
            bs.append(len(topk(o2, k, rng) & topk(e2, k, rng)) / k)
        bs.sort()
        lo, hi = bs[int(0.025 * boot)], bs[int(0.975 * boot) - 1]
        x_text = f"{sum(xs) / len(xs):.1f}/{k}"
        p_text = f"{sum(ps) / len(ps):.4f}[{min(ps):.4f},{max(ps):.4f}]"
        print(f"{est:>6} {overlap.mean:>6.3f} {x_text:>9} {p_text:>16}   "
              f"[{lo:.3f}, {hi:.3f}]")

    print("\n해석 지침: 단독 셀 p는 약할 수 있음 — 다조건 결합과 hybrid 부호검정"
          "(--sign)을 주 방어로, below-chance는 음수 retention과 함께 서술할 것.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
