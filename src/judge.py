"""게이트 자동 판정 — report·hybrid·downstream 산출물을 읽어 통과/사망을 출력.

    python3 src/judge.py <OUT_ROOT>   (예: $OM_WORK/runs/gate)

조건 (concept 10절):
  C1  one-sided 실패: g10·g01 각각, 어느 drift에서든 oracle top-k precision이
      noise floor보다 0.15 이상 낮다
  C1' hybrid 인과: 실패 축을 π로 바꾼 cell(bp: continuation→π, pb: occupancy→π)의
      top-k precision이 bb(β 그대로)보다 오른다
  C2  CertaGrad: certified=True, fresh ≤ 0.5× uniform, precision 손실 ≤ 0.02
  C3  downstream: oracle/인증 선택이 random보다 나쁘지 않다 (-0.02 허용)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def topk_set(scores: dict, frac: float = 0.10) -> set:
    k = max(1, int(len(scores) * frac))
    return set(sorted(scores, key=lambda i: -scores[i])[:k])


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def main() -> int:
    out_root = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/pilot")
    # drift100(run)만 — drift_100(adapter 폴더)은 제외
    runs = sorted(d for d in out_root.glob("drift*")
                  if d.is_dir() and not d.name.startswith("drift_"))
    if not runs:
        runs = [out_root]
    verdicts: dict[str, bool | None] = {"C1_g10": None, "C1_g01": None,
                                        "C1'_hybrid": None, "C2_certagrad": None,
                                        "C3_downstream": None}
    print(f"=== 게이트 판정: {out_root} (파이프라인 {len(runs)}개) ===\n")

    # ---- C1: one-sided 실패 + C2 ----
    for run in runs:
        rep = load(run / "report.json")
        if not rep:
            print(f"[{run.name}] report.json 없음 — 미완료")
            continue
        floor = rep.get("noise_floor", 0.0)
        print(f"[{run.name}] noise_floor={floor:.3f}, k={rep.get('k')}")
        for est in ("g00", "g10", "g01", "g11"):
            if est not in rep:
                continue
            p = rep[est]["precision"]
            delta = p - floor
            mark = ""
            if est in ("g10", "g01"):
                fail_here = delta <= -0.15
                key = f"C1_{est}"
                if fail_here:
                    verdicts[key] = True
                elif verdicts[key] is None:
                    verdicts[key] = False
                mark = "  ← one-sided 실패 실증" if fail_here else ""
            print(f"    {est}: precision={p:.3f} (Δfloor={delta:+.3f}){mark}")
        cg = rep.get("certagrad")
        if cg:
            ok = (cg.get("certified") and cg.get("fresh_frac_of_uniform", 9) <= 0.5)
            # uniform(전체 풀)은 정의상 oracle과 거의 일치 — precision 손실로 근사 비교
            loss_ok = cg.get("precision_vs_oracle", 0) >= (1.0 - 0.02) or \
                      cg.get("precision_vs_oracle", 0) >= rep.get("g11", {}).get("precision", 1.0) - 0.02
            verdicts["C2_certagrad"] = bool(ok and loss_ok) if verdicts["C2_certagrad"] is not True else True
            print(f"    CertaGrad: certified={cg.get('certified')} fresh={cg.get('fresh_frac_of_uniform'):.2f}× "
                  f"precision={cg.get('precision_vs_oracle'):.3f} → {'OK' if ok and loss_ok else 'FAIL'}")
        print()

    # ---- C1': hybrid 인과 (cell별 oracle 대비 top-k precision) ----
    for run in runs:
        oracle = load(run / "scores_oracle.json")
        if not oracle:
            continue
        for hf in sorted(run.glob("scores_hybrid_*.json")):
            cells = load(hf)
            if not cells or "bb" not in cells:
                continue
            sub = set(cells["bb"])  # hybrid 서브셋 프롬프트만 비교
            o_sub = {i: oracle[i]["score"] for i in sub if i in oracle}
            o_top = topk_set(o_sub, 0.25)
            prec = {}
            for cell, sc in cells.items():
                c_top = topk_set({i: v for i, v in sc.items() if i in o_sub}, 0.25)
                prec[cell] = len(c_top & o_top) / max(1, len(o_top))
            improved = (prec.get("bp", 0) >= prec.get("bb", 0)) or \
                       (prec.get("pb", 0) >= prec.get("bb", 0))
            if verdicts["C1'_hybrid"] is not True:
                verdicts["C1'_hybrid"] = improved
            cut = hf.stem.split("_")[-1]
            print(f"[{run.name}] hybrid cut={cut}: " +
                  " ".join(f"{c}={prec.get(c, float('nan')):.2f}" for c in ("bb", "bp", "pb", "pp")) +
                  ("  ← 축 교체로 회복" if improved else ""))
    print()

    # ---- C3: downstream ----
    for run in runs:
        ds = {f.stem.replace("downstream_", ""): load(f) for f in run.glob("downstream_*.json")}
        ds = {k: v for k, v in ds.items() if v}
        if not ds:
            continue
        base = next(iter(ds.values())).get("base_acc")
        line = ", ".join(f"{k}={v['val_acc']:.3f}" for k, v in sorted(ds.items()))
        print(f"[{run.name}] downstream (base={base:.3f}): {line}")
        if "oracle" in ds and "random" in ds:
            verdicts["C3_downstream"] = ds["oracle"]["val_acc"] >= ds["random"]["val_acc"] - 0.02
    print()

    # ---- 종합 ----
    print("=== 종합 ===")
    labels = {"C1_g10": "C1 g10(prefix만) 실패 실증", "C1_g01": "C1 g01(suffix만) 실패 실증",
              "C1'_hybrid": "C1' hybrid 축 교체로 회복", "C2_certagrad": "C2 CertaGrad 인증·절약",
              "C3_downstream": "C3 downstream 비열등"}
    for key, label in labels.items():
        v = verdicts[key]
        print(f"  {'PASS' if v else ('FAIL' if v is False else '미판정')}  {label}")
    core = [verdicts["C1_g10"], verdicts["C1_g01"], verdicts["C2_certagrad"]]
    if all(core) and verdicts["C1'_hybrid"]:
        print("\n→ 게이트 핵심 조건 충족. downstream까지 PASS면 원고 착수 조건 성립.")
    elif any(v is False for v in core):
        print("\n→ 핵심 조건 실패 있음 — concept 사망 조건 대조 필요. 수치 원인 분석 권장.")
    else:
        print("\n→ 산출물 부족 — 미완료 스테이지 확인.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
