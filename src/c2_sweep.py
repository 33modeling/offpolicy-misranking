"""C2(CertaGrad) 파라미터 스윕 — 저장된 micro-group 풀에서 CPU로 재판정.

    python3 src/c2_sweep.py <OUT_ROOT>

게이트 C2 기준: certified=True, fresh ≤ 0.5× uniform, precision 손실 ≤ 0.02.
델타·top-k 비율·초기 관측 수·반경 모드를 격자로 돌려 통과 조합을 찾는다.
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

from certagrad import certagrad, uniform_baseline


def _one(job):
    import time
    pools, val_pool, k, delta, init, max_fresh = job
    torch.set_num_threads(1)
    t0 = time.time()
    r = certagrad(pools, val_pool, k, delta=delta, init_groups=init, max_fresh=max_fresh)
    r["elapsed"] = time.time() - t0
    return delta, init, r


def main() -> int:
    out_root = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/pilot")
    runs = sorted(d for d in out_root.glob("drift*")
                  if d.is_dir() and not d.name.startswith("drift_")) or [out_root]
    for run in runs:
        mg, vg = run / "oracle_micro_groups.pt", run / "val_groups.pt"
        oc = run / "scores_oracle.json"
        if not (mg.exists() and vg.exists() and oc.exists()):
            continue
        micro = torch.load(mg, weights_only=True)
        val_pool = torch.load(vg, weights_only=True).float()
        oracle = {int(i): v["score"] for i, v in json.loads(oc.read_text()).items()}
        order = sorted(micro)
        pools = [micro[i].float() for i in order]
        n_pool = pools[0].shape[0]
        print(f"\n===== {run.name}: 후보 {len(pools)}개 × micro-group {n_pool} =====")
        best = None
        frac = 0.10  # 게이트 기준은 top-10% — 판정에 필요한 축만 돈다
        k = max(1, int(len(pools) * frac))
        o_top = set(sorted(oracle, key=lambda i: -oracle[i])[:k])
        uni = uniform_baseline(pools, val_pool, k, groups_each=n_pool)
        uni_prec = len({order[i] for i in uni["selected"]} & o_top) / k
        cap = int(uni["fresh_groups"] * 0.55)  # 0.5× 초과 = FAIL 확정 → 조기 중단
        jobs = [(pools, val_pool, k, delta, init, cap)
                for delta in (0.05, 0.20, 0.32) for init in (1, 2)]
        with ProcessPoolExecutor(max_workers=min(len(jobs), os.cpu_count() or 4)) as ex:
            for delta, init, r in ex.map(_one, jobs):
                sel = {order[i] for i in r["selected"]}
                prec = len(sel & o_top) / k
                frac_fresh = r["fresh_groups"] / uni["fresh_groups"]
                ok = r["certified"] and frac_fresh <= 0.5 and prec >= uni_prec - 0.02
                line = (f"  frac={frac} δ={delta} init={init}: certified={r['certified']} "
                        f"fresh={frac_fresh:.2f}× prec={prec:.3f} (uniform {uni_prec:.3f}) "
                        f"[{r.get('elapsed', 0):.0f}s]"
                        + ("  ← C2 PASS" if ok else ""))
                print(line, flush=True)
                if ok and (best is None or frac_fresh < best[0]):
                    best = (frac_fresh, line)
        print("→ " + ("통과 조합 존재: " + best[1].strip() if best else
                      "통과 조합 없음 — margin/신호 자체가 부족 (설계 한계 판정 재료)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
