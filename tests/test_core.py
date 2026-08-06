"""핵심 로직 테스트 (모델 불필요, CPU 수 초).

    PYTHONPATH=src python3 tests/test_core.py
"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from certagrad import certagrad, uniform_baseline  # noqa: E402
from grads import loo_advantages, token_weights  # noqa: E402

FAIL = 0


def check(name, cond):
    global FAIL
    print(("  ok  " if cond else "FAIL  ") + name)
    if not cond:
        FAIL += 1


# 1) LOO advantage — 상수 이동 불변·K=1 안전
a = loo_advantages(torch.tensor([1.0, 0.0, 1.0, 1.0]))
check("LOO K=4", torch.allclose(a, torch.tensor([1 / 3, -1.0, 1 / 3, 1 / 3])))
check("LOO K=1 → 0", loo_advantages(torch.tensor([1.0])).item() == 0.0)

# 2) 2×2 토큰 가중치 — 정의와 곱 관계 g10·g01/g00 = g11
lp_pi = torch.tensor([-1.0, -2.0, -0.5])
lp_b = torch.tensor([-1.5, -1.0, -1.0])
w = {e: token_weights(lp_pi, lp_b, 1.0, e, clip_cap=100.0) for e in ("g00", "g10", "g01", "g11")}
lr = lp_pi - lp_b
check("g00 = r_t", torch.allclose(w["g00"], lr.exp()))
check("g10 = cumprod prefix", torch.allclose(w["g10"], lr.cumsum(0).exp()))
check("g01 = suffix product", torch.allclose(w["g01"], (lr.sum() - lr.cumsum(0) + lr).exp()))
check("g11 = full ratio 상수", torch.allclose(w["g11"], lr.sum().exp().expand(3)))
check("g10·g01/g00 = g11", torch.allclose(w["g10"] * w["g01"] / w["g00"], w["g11"], atol=1e-5))

# 3) CertaGrad — 경계 마진 크면 인증+절약, 빡빡하면 정직한 실패
torch.manual_seed(0)
dim, n_groups, k = 32, 64, 3
v = torch.randn(dim)
v /= v.norm()


def make_pools(thetas, noise, groups):
    pools = []
    for th in thetas:
        u = torch.randn(dim)
        u -= (u @ v) * v
        u /= u.norm()
        m = 3.0 * (math.cos(math.radians(th)) * v + math.sin(math.radians(th)) * u)
        pools.append(torch.stack([m + noise * torch.randn(dim) for _ in range(groups)]))
    return pools


pools = make_pools([10, 15, 20, 80, 85, 90, 95, 100, 105, 110, 115, 120], 0.4, n_groups)
val_pool = torch.stack([3.0 * v + 0.4 * torch.randn(dim) for _ in range(n_groups)])
res = certagrad(pools, val_pool, k=k)
uni = uniform_baseline(pools, val_pool, k=k, groups_each=n_groups)
check("넓은 마진 인증 성공", res["certified"])
check("정답 top-k 선택", set(res["selected"]) == {0, 1, 2})
check("fresh ≤ 50% of uniform", res["fresh_groups"] / uni["fresh_groups"] <= 0.5)

pools2 = make_pools([10, 15, 20, 25, 30, 35, 40, 45], 0.5, 16)
res2 = certagrad(pools2, torch.stack([3.0 * v + 0.5 * torch.randn(dim) for _ in range(16)]), k=3)
check("좁은 마진 → 인증 실패 정직 보고", not res2["certified"])

print(("PASS" if FAIL == 0 else "FAIL") + f" (실패 {FAIL})")
sys.exit(1 if FAIL else 0)
