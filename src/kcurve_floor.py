"""P4-0 — oracle K-스케일링 floor 곡선 (GPU 0, 기존 산출물만 사용).

    python src/kcurve_floor.py <runs_root> [--gate]

목적: v2 전 체제의 floor≈chance 가 (a) oracle 표본 K가 부족해 진짜 순위를
분해 못 하는 것인지, (b) 프롬프트 간 순위 신호가 구조적으로 없는 것인지 판별.

재료: run마다 저장된 oracle_micro_groups.pt(프롬프트당 G개 micro-group
gradient, JL 투영)와 val_gradient.pt. split-half floor의 정의(절반 평균
gradient의 val 방향 cosine → top-k 겹침)를 임의의 절반 크기 m(그룹 수)으로
정확히 재계산한다 — K'=2·m·(그룹당 롤아웃 수) 롤아웃의 floor와 동일.

확장 예측: 절반 크기 m=1의 split-half 상관 r1에 Spearman–Brown
  r_m = m·r1 / (1 + (m-1)·r1)
을 적용, 이변량 정규 시뮬로 top-k 겹침으로 환산해 K=64~256 floor를 예측.
관측 가능한 최대 m에서 예측-관측 보정 오차를 같이 출력한다(신뢰도 자가 진단).

사전 등록 판정 (concept.md P4 설계서와 동일):
  - 대상: gsm8k 계열 완주 run
  - 확장 권고(exit 0): 과반 run에서 "예측 floor >= 2×chance 가 되는 최소
    K' <= 256" 존재 → P4-1(FRESH_K 증량 oracle 재실행) 가치 있음
  - 구조적 부재(exit 3): 그 외 — K를 아무리 키워도 2×chance 도달 근거 없음
    → GPU 확장 없이 사전 등록 분기(부정적 결과+방법론 재편)로
  - 판정 불가(exit 4): 대상 run 0개
--gate: exit code만 (0/3/4)
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch

T_REP = 30          # 관측 floor: 그룹 배정 재표집 횟수
S_SIM = 40          # 예측 floor: 이변량 정규 시뮬 횟수
GO_MULT = 2.0
K_MAX_PRED = 256    # 이 K까지 예측해 확장 가치 판정


def topk_set(scores: list[float], k: int, rng: random.Random) -> set:
    jit = [rng.random() for _ in scores]
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], jit[i]))
    return set(order[:k])


def overlap(a: list[float], b: list[float], k: int, j: int) -> float:
    ta = topk_set(a, k, random.Random(1000 + j))
    tb = topk_set(b, k, random.Random(104729 + j))
    return len(ta & tb) / k


def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a - a.mean()
    b = b - b.mean()
    d = a.norm() * b.norm()
    return float((a @ b) / d) if d > 0 else 0.0


def observed_curve(stack: torch.Tensor, v: torch.Tensor):
    """stack [P,G,D], v [D] → {m: (floor_mean, lo, hi, r_mean)} for m=1..G//2."""
    P, G, _ = stack.shape
    k = max(1, round(0.10 * P))
    vn = v / v.norm()
    out = {}
    for m in range(1, G // 2 + 1):
        floors, rs = [], []
        for j in range(T_REP):
            g = torch.Generator().manual_seed(9000 + j)
            perm = torch.argsort(torch.rand(P, G, generator=g), dim=1)[:, : 2 * m]
            idx = perm.unsqueeze(-1).expand(-1, -1, stack.shape[2])
            picked = torch.gather(stack, 1, idx)
            ma = picked[:, :m].mean(dim=1)
            mb = picked[:, m:].mean(dim=1)
            a = (ma @ vn) / ma.norm(dim=1).clamp_min(1e-12)
            b = (mb @ vn) / mb.norm(dim=1).clamp_min(1e-12)
            floors.append(overlap(a.tolist(), b.tolist(), k, j))
            rs.append(pearson(a, b))
        out[m] = (sum(floors) / T_REP, min(floors), max(floors), sum(rs) / T_REP)
    return out, k, k / P


def predicted_floor(r: float, n: int, k: int) -> float:
    """상관 r인 이변량 정규 점수쌍의 top-k 겹침 기대치 (시뮬)."""
    r = max(0.0, min(0.999, r))
    vals = []
    for s in range(S_SIM):
        g = torch.Generator().manual_seed(3000 + s)
        t = torch.randn(n, generator=g)
        e1 = torch.randn(n, generator=g)
        e2 = torch.randn(n, generator=g)
        w = r ** 0.5
        a = (w * t + (1 - r) ** 0.5 * e1).tolist()
        b = (w * t + (1 - r) ** 0.5 * e2).tolist()
        vals.append(overlap(a, b, k, s))
    return sum(vals) / S_SIM


def sb(r1: float, m: int) -> float:
    if r1 <= 0:
        return 0.0
    return m * r1 / (1 + (m - 1) * r1)


def find_fresh_k(run: Path, G: int) -> int:
    for name in ("manifest.json", "report.json"):
        try:
            obj = json.loads((run / name).read_text())
        except Exception:
            continue
        stack = [obj]
        while stack:
            o = stack.pop()
            if isinstance(o, dict):
                for kk, vv in o.items():
                    if kk in ("fresh_k", "fresh-k") and isinstance(vv, int):
                        return vv
                    stack.append(vv)
            elif isinstance(o, list):
                stack.extend(o)
    return 32


def tag_of(name: str) -> str:
    if "dapo" in name:
        return "dapo"
    if "math500" in name:
        return "math500"
    if "hard" in name or "27b" in name:
        return "other"
    return "gsm8k"


def main() -> int:
    gate = "--gate" in sys.argv
    root = Path([a for a in sys.argv[1:] if not a.startswith("--")][0])
    runs = [d for d in sorted(root.glob("v2-*"))
            if (d / "DONE").exists() and "smoke" not in d.name
            and (d / "oracle_micro_groups.pt").exists()
            and (d / "val_gradient.pt").exists()]

    reports, votes = [], []
    for d in runs:
        micro = torch.load(d / "oracle_micro_groups.pt", map_location="cpu")
        v = torch.load(d / "val_gradient.pt", map_location="cpu").float()
        G = min(t.shape[0] for t in micro.values())
        stack = torch.stack([micro[i][:G].float() for i in sorted(micro)])
        curve, k, chance = observed_curve(stack, v)
        P = stack.shape[0]
        gsize = max(1, find_fresh_k(d, G) // G)
        r1 = curve[1][3]
        mmax = G // 2
        # 보정 자가진단: 최대 관측 m에서 SB+BVN 예측 vs 실측
        pred_at_mmax = predicted_floor(sb(r1, mmax), P, k)
        obs_at_mmax = curve[mmax][0]
        # 확장 예측: K'=64..K_MAX_PRED (절반 m = K'/(2·gsize))
        preds, k_need = [], None
        for kp in (64, 128, 256):
            if kp > K_MAX_PRED:
                break
            m = max(1, kp // (2 * gsize))
            f = predicted_floor(sb(r1, m), P, k)
            preds.append((kp, f))
            if k_need is None and f >= GO_MULT * chance:
                k_need = kp
        tag = tag_of(d.name)
        if tag == "gsm8k":
            votes.append(k_need is not None)
        reports.append((d.name, tag, P, G, gsize, chance, curve, r1,
                        pred_at_mmax, obs_at_mmax, preds, k_need))

    eligible = sum(1 for r in reports if r[1] == "gsm8k")
    go = eligible > 0 and sum(votes) * 2 > eligible
    code = 4 if eligible == 0 else (0 if go else 3)
    if gate:
        return code

    print("# P4-0 — oracle K-스케일링 floor 곡선 (기존 산출물, GPU 0)\n")
    print("질문: floor≈chance 는 'oracle 표본 K 부족'인가 '순위 신호의 구조적 부재'인가.")
    print("micro-group gradient 재조합으로 K'별 floor를 정확 재계산하고, K=64~256은")
    print("Spearman–Brown+정규 시뮬로 예측한다(관측 최대점에서 보정 오차 병기).\n")
    for name, tag, P, G, gsize, chance, curve, r1, pm, om, preds, k_need in reports:
        print(f"## {name}  ({tag}, n={P}, 그룹 {G}×{gsize}롤아웃, chance={chance:.3f})\n")
        print("| K'(롤아웃) | floor 관측 [범위] | split-half r |")
        print("|---|---|---|")
        for m, (fm, lo, hi, rm) in curve.items():
            print(f"| {2 * m * gsize} | {fm:.3f} [{lo:.3f}~{hi:.3f}] | {rm:.3f} |")
        print(f"\n- r1(절반=1그룹) = {r1:.3f} → SB 예측: "
              + ", ".join(f"K'={kp}: floor≈{f:.3f}" for kp, f in preds))
        print(f"- 보정 자가진단 (K'={2 * (G // 2) * gsize}): 예측 {pm:.3f} vs 관측 {om:.3f}"
              f" — 차이가 크면 예측은 참고로만")
        if k_need:
            print(f"- **{GO_MULT}×chance 도달 예상 최소 K' = {k_need}**\n")
        else:
            print(f"- K'={K_MAX_PRED}까지도 {GO_MULT}×chance 도달 근거 없음\n")

    print("## 판정 (사전 등록 규칙)\n")
    print(f"- 대상(gsm8k 계열): {eligible}개, 'K'<=256에서 도달 예상' {sum(votes)}개")
    if code == 0:
        ks = [r[11] for r in reports if r[1] == "gsm8k" and r[11]]
        print(f"- **확장 권고** — FRESH_K={max(ks)} oracle 재실행(P4-1)이 신호 체제를")
        print("  만들 수 있다는 근거 있음. P4-1 스크립트를 준비해 진행.")
    elif code == 3:
        print("- **구조적 부재** — K를 키워도 순위 신호가 살아날 근거 없음. GPU 확장")
        print("  없이 사전 등록 분기로: 이 결과 자체가 '포화 태스크의 프롬프트 선택")
        print("  신호는 잡음과 구분 불가하며 대시보드는 이를 알려주지 않는다'의 실증.")
        print("  (이론 + 인증 불가 + 평가 함정 3종 + 본 판별 = 재편 논문의 골격)")
    else:
        print("- **판정 불가** — micro-group 산출물이 있는 완주 run 없음.")
    return code


if __name__ == "__main__":
    sys.exit(main())
