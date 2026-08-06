"""CertaGrad — 순위 경계에만 fresh rollout을 배분하는 순차 top-k 인증 (concept 6·7절).

관측 단위는 micro-group(크기 G)의 projected LOO group gradient다. 각 프롬프트의
fresh pool(사전 수집)에서 micro-group을 하나씩 꺼내 쓰는 시뮬레이션으로 구현하므로
GPU 재호출 없이 배분 정책만 비교할 수 있다.

score: s_i = cos(μ_i, v). confidence ball ||μ̂-μ|| ≤ r 에서 각도 반경
α = arcsin(r/||μ̂||) (r < ||μ̂||), score 구간은 [cos(φ+α_i+α_v), cos(φ-α_i-α_v)].
공유 validation 오차 α_v를 후보마다 더한다 (독립 취급 금지 — concept 7절).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch


def eb_radius(samples: torch.Tensor, delta: float, mode: str = "gaussian") -> float:
    """confidence ball 반경 ||μ̂-μ|| ≤ r.

    mode="gaussian" (파일럿 기본): 평균 오차를 등방 가우시안으로 근사하고
      Laurent–Massart χ² 꼬리 r² = (tr(Σ̂)/(d·n))·(d + 2√(d·t) + 2t), t=ln(1/δ).
      근사이므로 coverage는 게이트의 반복실험 검증 항목(≥0.90)으로 실측한다.
    mode="hoeffding" (보수): concept 7절 방향의 노름 Hoeffding 판본. 가산 B항이
      지배해 파일럿 규모에서는 거의 인증 불가 — 본실험 정식 보증 비교용.
    """
    n, d = samples.shape
    if n < 2:
        return float("inf")
    var_total = float(samples.var(dim=0).sum())
    if mode == "gaussian":
        t = math.log(1.0 / delta)
        return math.sqrt((var_total / (d * n)) * (d + 2 * math.sqrt(d * t) + 2 * t))
    b = float(samples.norm(dim=1).max()) * 2.0
    return math.sqrt(2 * var_total * math.log(3 / delta) / n) + 3 * b * math.log(3 / delta) / n


def angle_radius(mean: torch.Tensor, r: float) -> float:
    n = float(mean.norm())
    if r >= n or n == 0:
        return math.pi  # 방향 미정 — score 구간 [-1, 1]
    return math.asin(r / n)


@dataclass
class Candidate:
    pool: torch.Tensor  # (n_groups, dim) — fresh micro-group projected gradients
    used: int = 0
    obs: list = field(default_factory=list)

    def draw(self) -> bool:
        if self.used >= self.pool.shape[0]:
            return False
        self.obs.append(self.pool[self.used])
        self.used += 1
        return True

    def stats(self, delta: float) -> tuple[torch.Tensor, float]:
        x = torch.stack(self.obs)
        return x.mean(dim=0), eb_radius(x, delta)


def score_interval(mu_i, alpha_i, mu_v, alpha_v) -> tuple[float, float]:
    cos_phi = float((mu_i @ mu_v) / (mu_i.norm() * mu_v.norm() + 1e-12))
    phi = math.acos(max(-1.0, min(1.0, cos_phi)))
    lo = math.cos(min(math.pi, phi + alpha_i + alpha_v))
    hi = math.cos(max(0.0, phi - alpha_i - alpha_v))
    return lo, hi


def certagrad(
    cand_pools: list[torch.Tensor],
    val_pool: torch.Tensor,
    k: int,
    delta: float = 0.05,
    init_groups: int = 1,
    max_rounds: int = 10_000,
) -> dict:
    """순차 인증. 반환: 선택 집합, 사용 micro-group 수, 인증 성공 여부."""
    m = len(cand_pools)
    cands = [Candidate(pool) for pool in cand_pools]
    val = Candidate(val_pool)
    per = delta / (m + 1)

    for c in cands:
        for _ in range(init_groups):
            c.draw()
    val.draw()

    for _ in range(max_rounds):
        mu_v, r_v = val.stats(per)
        a_v = angle_radius(mu_v, r_v)
        intervals = []
        for c in cands:
            mu, r = c.stats(per)
            intervals.append(score_interval(mu, angle_radius(mu, r), mu_v, a_v))
        mid = sorted(range(m), key=lambda i: -(intervals[i][0] + intervals[i][1]))
        sel, rest = set(mid[:k]), mid[k:]
        lo_min = min(intervals[i][0] for i in sel)
        hi_max = max(intervals[i][1] for i in rest)
        if lo_min > hi_max:
            return {
                "selected": sorted(sel),
                "certified": True,
                "fresh_groups": sum(c.used for c in cands) + val.used,
            }
        # 경계에 걸린 후보 / 공유 validation 중 불확실성 기여가 큰 쪽에 배분 (concept 6절 5항)
        boundary = [i for i in sel if intervals[i][0] <= hi_max] + [
            i for i in rest if intervals[i][1] >= lo_min
        ]
        boundary_alphas = []
        for i in boundary:
            mu_i, r_i = cands[i].stats(per)
            boundary_alphas.append((angle_radius(mu_i, r_i), i))
        if boundary_alphas and a_v >= max(a for a, _ in boundary_alphas):
            if val.draw():
                continue
        progressed = False
        for _, i in sorted(boundary_alphas, reverse=True):
            if cands[i].draw():
                progressed = True
                break
        if not progressed and not val.draw():
            return {
                "selected": sorted(sel),
                "certified": False,  # 예산 소진 — 인증 실패를 숨기지 않는다
                "fresh_groups": sum(c.used for c in cands) + val.used,
            }
    return {"selected": sorted(sel), "certified": False, "fresh_groups": sum(c.used for c in cands) + val.used}


def uniform_baseline(cand_pools: list[torch.Tensor], val_pool: torch.Tensor, k: int, groups_each: int) -> dict:
    """GradAlign matched — 모든 후보에 같은 수의 fresh micro-group 균등 배분."""
    mu_v = val_pool[: max(1, groups_each)].mean(dim=0)
    scores = []
    for pool in cand_pools:
        mu = pool[:groups_each].mean(dim=0)
        scores.append(float((mu @ mu_v) / (mu.norm() * mu_v.norm() + 1e-12)))
    sel = sorted(range(len(cand_pools)), key=lambda i: -scores[i])[:k]
    return {
        "selected": sorted(sel),
        "fresh_groups": groups_each * len(cand_pools) + max(1, groups_each),
    }
