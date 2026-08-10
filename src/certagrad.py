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

    def stats(self, delta: float, mode: str = "gaussian") -> tuple[torch.Tensor, float]:
        x = torch.stack(self.obs)
        return x.mean(dim=0), eb_radius(x, delta, mode)


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
    radius_mode: str = "gaussian",
    max_fresh: int | None = None,
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

    # 후보 통계를 (m,d) 행렬로 유지, 라운드당 계산을 전부 벡터화 (느린 CPU 대응)
    dim = cand_pools[0].shape[1]
    MU = torch.zeros(m, dim)
    ALPHA = torch.zeros(m)
    dirty = set(range(m))
    val_dirty = True
    mu_v = None
    a_v = 0.0
    _pi = math.pi

    for _ in range(max_rounds):
        if max_fresh is not None and sum(c.used for c in cands) + val.used > max_fresh:
            break  # 예산 상한 초과 — C2 기준(≤0.5×)상 이미 실패 확정이므로 조기 종료
        if val_dirty:
            mu_v, r_v = val.stats(per, radius_mode)
            a_v = angle_radius(mu_v, r_v)
            val_dirty = False
        for i in dirty:
            mu, r = cands[i].stats(per, radius_mode)
            MU[i] = mu
            ALPHA[i] = angle_radius(mu, r)
        dirty.clear()
        norms = MU.norm(dim=1)
        vn = float(mu_v.norm())
        cosphi = (MU @ mu_v) / (norms * vn + 1e-12)
        phi = torch.arccos(cosphi.clamp(-1.0, 1.0))
        width = ALPHA + a_v
        lo = torch.cos((phi + width).clamp(max=_pi))
        hi = torch.cos((phi - width).clamp(min=0.0))
        mid_order = torch.argsort(lo + hi, descending=True)
        sel_idx = mid_order[:k]
        rest_idx = mid_order[k:]
        lo_min = float(lo[sel_idx].min())
        hi_max = float(hi[rest_idx].max()) if rest_idx.numel() else float("-inf")
        if lo_min > hi_max:
            return {
                "selected": sorted(int(i) for i in sel_idx),
                "certified": True,
                "fresh_groups": sum(c.used for c in cands) + val.used,
            }
        # 경계에 걸린 후보 / 공유 validation 중 불확실성 기여가 큰 쪽에 배분 (concept 6절 5항)
        sel_mask = torch.zeros(m, dtype=torch.bool)
        sel_mask[sel_idx] = True
        boundary_mask = (sel_mask & (lo <= hi_max)) | (~sel_mask & (hi >= lo_min))
        boundary = boundary_mask.nonzero(as_tuple=True)[0]
        sel = {int(i) for i in sel_idx}
        if boundary.numel() and a_v >= float(ALPHA[boundary].max()):
            if val.draw():
                val_dirty = True
                continue
        progressed = False
        drawn = 0
        for i in boundary[torch.argsort(ALPHA[boundary], descending=True)].tolist():
            if cands[i].draw():
                dirty.add(i)
                progressed = True
                drawn += 1
                if drawn >= 4:  # 라운드당 경계 상위 4곳 동시 관측 — 라운드 수 1/4
                    break
        if not progressed:
            if val.draw():
                val_dirty = True
                continue
            # 예산 소진 — 인증 실패를 숨기지 않되, 선택 자체는 구간 중점이 아니라
            # 지금까지 관측한 전체 평균 점수로 반환한다 (uniform과 동일 기준).
            mu_v_f, _ = val.stats(per, radius_mode)
            final_scores = []
            for c in cands:
                mu, _ = c.stats(per, radius_mode)
                final_scores.append(float((mu @ mu_v_f) / (mu.norm() * mu_v_f.norm() + 1e-12)))
            sel = set(sorted(range(m), key=lambda i: -final_scores[i])[:k])
            return {
                "selected": sorted(sel),
                "certified": False,
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
