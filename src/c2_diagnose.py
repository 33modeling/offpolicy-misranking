"""C2 실패 원인 진단 — 경계 margin vs 신뢰반경으로 '필요 관측 깊이'를 계산.

    python3 src/c2_diagnose.py <OUT_ROOT>

출력: k번째 경계의 각도 margin, 후보 각도반경 α(현 풀 깊이), α_v(검증 방향
오차), 인증에 필요한 관측 수 추정 — 풀 깊이보다 크면 "심화 수집"이 처방.
run 하나에서 예외가 나도 나머지는 계속 진단한다.
"""

from __future__ import annotations

import math
import sys
import traceback
from pathlib import Path

import torch

from certagrad import angle_radius, eb_radius


def diagnose_run(run: Path) -> None:
    mg = run / "oracle_micro_groups.pt"
    vg = run / "val_groups.pt"
    oc = run / "scores_oracle.json"
    missing = [p.name for p in (mg, vg, oc) if not p.exists()]
    if missing:
        print(f"[{run.name}] 산출물 누락으로 진단 불가: {missing}")
        return

    micro = torch.load(mg, weights_only=True)
    val_pool = torch.load(vg, weights_only=True).float()
    order = sorted(micro)
    m = len(order)
    per = 0.05 / (m + 1)

    mu_v = val_pool.mean(dim=0)
    r_v = eb_radius(val_pool, per)
    a_v = math.degrees(angle_radius(mu_v, r_v))

    alphas, phis, ns, vars_, norms = [], [], [], [], []
    for i in order:
        stack = micro[i].float()
        mu = stack.mean(dim=0)
        r = eb_radius(stack, per)
        alphas.append(math.degrees(angle_radius(mu, r)))
        c = float((mu @ mu_v) / (mu.norm() * mu_v.norm() + 1e-12))
        phis.append(math.degrees(math.acos(max(-1.0, min(1.0, c)))))
        ns.append(stack.shape[0])
        vars_.append(float(stack.var(dim=0).sum()))
        norms.append(float(mu.norm()))

    # 무신호(gradient 노름 ≈ 0 = 보상 분산 0) 프롬프트 — GRPO도 버리는 데이터
    live = [j for j in range(m) if norms[j] > 1e-6]
    print(f"[{run.name}] 유신호 {len(live)}/{m} prompts (무신호 {m - len(live)}개 = 전부 정답/오답 그룹)")

    frac = 0.10
    k = max(1, int(m * frac))
    srt = sorted(range(m), key=lambda j: phis[j])
    gap = phis[srt[k]] - phis[srt[k - 1]]
    # 유신호만의 경계 margin (실질 인증 문제)
    if len(live) > k:
        srt_l = sorted(live, key=lambda j: phis[j])
        gap_live = phis[srt_l[k]] - phis[srt_l[k - 1]]
        print(f"[{run.name}] 유신호 한정 margin={gap_live:.2f}° (전체 포함 {gap:.2f}°)")
        gap = gap_live
        band = srt_l[max(0, k - 10): k + 10]
    else:
        band = srt[max(0, k - 10): k + 10]
    med_var = sorted(vars_[j] for j in band)[len(band) // 2]
    med_norm = sorted(norms[j] for j in band)[len(band) // 2]
    a_req = (gap - 2.0 * a_v) / 2.0
    if a_req <= 0:
        need = "α_v(검증 방향 오차)만으로 margin 초과 → **val 관측부터 심화** (가장 싼 처방)"
    else:
        s = med_norm * math.sin(math.radians(a_req))
        n_req = med_var / (s * s) if s > 0 else float("inf")
        need = f"경계 후보당 필요 관측 ≈ {n_req:.0f} micro-groups (현재 {ns[0]})"
    print(f"[{run.name}] frac={frac}: k경계 margin={gap:.2f}° · "
          f"후보 α(n={ns[0]})={sorted(alphas)[m // 2]:.2f}° · α_v={a_v:.2f}° → {need}")


def main() -> int:
    out_root = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/pilot")
    runs = sorted(d for d in out_root.glob("drift*")
                  if d.is_dir() and not d.name.startswith("drift_")) or [out_root]
    for run in runs:
        try:
            diagnose_run(run)
        except Exception:
            print(f"[{run.name}] 진단 중 예외 — 아래 traceback을 그대로 전달할 것")
            traceback.print_exc()
    return 0


if __name__ == "__main__":
    sys.exit(main())
