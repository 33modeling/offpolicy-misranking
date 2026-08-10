"""C2 실패 원인 진단 — 경계 margin vs 신뢰반경으로 '필요 관측 깊이'를 계산.

    python3 src/c2_diagnose.py <OUT_ROOT>

출력: k번째 경계의 각도 margin, 후보 평균 각도반경 α(n=8), 인증에 필요한
관측 수 추정 n_req — 풀 깊이(8)보다 크면 "경계 심화 수집"이 처방이다.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

from certagrad import angle_radius, eb_radius


def main() -> int:
    out_root = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/pilot")
    runs = sorted(d for d in out_root.glob("drift*")
                  if d.is_dir() and not d.name.startswith("drift_")) or [out_root]
    for run in runs:
        mg, vg, oc = run / "oracle_micro_groups.pt", run / "val_groups.pt", run / "scores_oracle.json"
        if not (mg.exists() and vg.exists() and oc.exists()):
            continue
        micro = torch.load(mg, weights_only=True)
        val_pool = torch.load(vg, weights_only=True).float()
        oracle = {int(i): v["score"] for i, v in json.loads(oc.read_text()).items()}
        order = sorted(micro)
        m = len(order)
        per = 0.05 / (m + 1)

        mu_v = val_pool.mean(dim=0)
        r_v = eb_radius(val_pool, per)
        a_v = math.degrees(angle_radius(mu_v, r_v))

        # 후보별 전체 풀 사용 시 각도반경과 φ(val과의 각)
        alphas, phis, ns, vars_ = [], [], [], []
        for i in order:
            stack = micro[i].float()
            mu = stack.mean(dim=0)
            r = eb_radius(stack, per)
            alphas.append(math.degrees(angle_radius(mu, r)))
            c = float((mu @ mu_v) / (mu.norm() * mu_v.norm() + 1e-12))
            phis.append(math.degrees(math.acos(max(-1, min(1, c)))))
            ns.append(stack.shape[0])
            vars_.append(float(stack.var(dim=0).sum()))

        for frac in (0.10, 0.25):
            k = max(1, int(m * frac))
            srt = sorted(range(m), key=lambda j: phis[j])  # 각도 작을수록 상위
            gap = phis[srt[k]] - phis[srt[k - 1]]  # k경계 각도 margin
            # 경계 부근 후보들의 필요 깊이: 2α(n)+2α_v ≤ gap → α_req = (gap-2α_v)/2 (후보 2개 몫)
            band = srt[max(0, k - 10):k + 10]
            med_var = sorted(vars_[j] for j in band)[len(band) // 2]
            med_norm = sorted(
                float(micro[order[j]].float().mean(dim=0).norm()) for j in band
            )[len(band) // 2]
            a_req = (gap - 2 * a_v) / 2.0
            if a_req <= 0:
                need = "불가능 — α_v(검증 방향 오차)만으로 margin 초과 → val 관측부터 심화"
            else:
                # α ≈ asin( sqrt(var/n)/norm )  →  n_req ≈ var / (norm·sin(α_req))²
                s = med_norm * math.sin(math.radians(a_req))
                n_req = med_var / (s * s) if s > 0 else float("inf")
                need = f"경계 후보당 필요 관측 ≈ {n_req:.0f} micro-groups (현재 {ns[0]})"
            print(f"[{run.name}] frac={frac}: k경계 margin={gap:.2f}° · 후보 α(n={ns[0]})="
                  f"{sorted(alphas)[m//2]:.2f}° · α_v={a_v:.2f}° → {need}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
