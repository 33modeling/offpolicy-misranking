"""게이트 자동 판정 — report·hybrid·downstream 산출물을 읽어 통과/사망을 출력.

    python3 src/judge.py <OUT_ROOT>   (예: $OM_WORK/runs/gate)

조건 (concept 10절):
  C1  one-sided 실패: g10·g01 각각, 어느 drift에서든 oracle top-k precision이
      noise floor보다 0.15 이상 낮다
  C1' hybrid 인과: g10(pb)의 빠진 continuation을 복원한 pp, g01(bp)의 빠진
      occupancy를 복원한 pp의 top-k precision이 각각 엄격히 오른다
  C2  CertaGrad: certified=True, fresh ≤ 0.5× uniform, precision 손실 ≤ 0.02
  C3  downstream: oracle/인증 선택이 random보다 나쁘지 않다 (-0.02 허용)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def topk_set(scores: dict, frac: float = 0.10) -> set:
    from select_rules import topk_count
    k = topk_count(len(scores), frac)
    return set(sorted(scores, key=lambda i: -scores[i])[:k])


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _complete_verdict(results: list[bool], expected: int) -> bool | None:
    """모든 필수 결과가 있을 때만 PASS하고, 하나라도 실패하면 즉시 FAIL한다."""
    if any(result is False for result in results):
        return False
    if len(results) < expected:
        return None
    return True


def judge(out_root: Path) -> dict[str, bool | None]:
    # drift100(run)만 — drift_100(adapter 폴더)은 제외
    runs = sorted(d for d in out_root.glob("drift*")
                  if d.is_dir() and not d.name.startswith("drift_"))
    if not runs:
        runs = [out_root]
    verdicts: dict[str, bool | None] = {"C1_g10": None, "C1_g01": None,
                                        "C1'_hybrid": None, "C2_certagrad": None,
                                        "C3_downstream": None}
    one_sided_failures = {"g10": set(), "g01": set()}
    complete_reports = 0
    c2_results: list[bool] = []
    print(f"=== 게이트 판정: {out_root} (파이프라인 {len(runs)}개) ===\n")

    # ---- C1: one-sided 실패 + C2 ----
    for run in runs:
        rep = load(run / "report.json")
        if not rep:
            print(f"[{run.name}] report.json 없음 — 미완료")
            continue
        complete_reports += 1
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
                    one_sided_failures[est].add(run.name)
                mark = "  ← one-sided 실패 실증" if fail_here else ""
            print(f"    {est}: precision={p:.3f} (Δfloor={delta:+.3f}){mark}")
        cg = rep.get("certagrad")
        if cg:
            fresh = float(cg.get("fresh_frac_of_uniform", 9.0))
            precision = float(cg.get("precision_vs_oracle", 0.0))
            uniform_precision = cg.get("uniform_precision_vs_oracle")
            # 기존 산출물에 uniform precision이 없으면 oracle=1.0을 보수적으로 적용한다.
            reference = float(uniform_precision) if uniform_precision is not None else 1.0
            ok = bool(cg.get("certified")) and fresh <= 0.5
            loss_ok = precision >= reference - 0.02
            c2_results.append(bool(ok and loss_ok))
            legacy = " (legacy: uniform=1.0 가정)" if uniform_precision is None else ""
            print(f"    CertaGrad: certified={cg.get('certified')} fresh={fresh:.2f}× "
                  f"precision={precision:.3f} uniform={reference:.3f}{legacy} "
                  f"→ {'OK' if ok and loss_ok else 'FAIL'}")
        print()

    for est in ("g10", "g01"):
        key = f"C1_{est}"
        if verdicts[key] is not True:
            verdicts[key] = False if complete_reports == len(runs) else None
    verdicts["C2_certagrad"] = _complete_verdict(c2_results, len(runs))

    # ---- C1': hybrid 인과 (cell별 oracle 대비 top-k precision) ----
    hybrid_seen = {"g10": False, "g01": False}
    hybrid_recovered = {"g10": False, "g01": False}
    for run in runs:
        oracle = load(run / "scores_oracle.json")
        if not oracle:
            continue
        for hf in sorted(run.glob("scores_hybrid_*.json")):
            cells = load(hf)
            required = {"bb", "bp", "pb", "pp"}
            if not cells or not required.issubset(cells):
                print(f"[{run.name}] {hf.name}: hybrid cell 부족 — 미판정")
                continue
            sub = set(cells["bb"])  # hybrid 서브셋 프롬프트만 비교
            o_sub = {i: oracle[i]["score"] for i in sub if i in oracle}
            o_top = topk_set(o_sub, 0.25)
            prec = {}
            for cell, sc in cells.items():
                c_top = topk_set({i: v for i, v in sc.items() if i in o_sub}, 0.25)
                prec[cell] = len(c_top & o_top) / max(1, len(o_top))
            recovered = {
                "g10": prec["pp"] > prec["pb"],  # continuation: β → π
                "g01": prec["pp"] > prec["bp"],  # occupancy: β → π
            }
            marks = []
            for est in ("g10", "g01"):
                if run.name not in one_sided_failures[est]:
                    continue
                hybrid_seen[est] = True
                hybrid_recovered[est] |= recovered[est]
                marks.append(f"{est} {'회복' if recovered[est] else '미회복'}")
            cut = hf.stem.split("_")[-1]
            print(f"[{run.name}] hybrid cut={cut}: " +
                  " ".join(f"{c}={prec.get(c, float('nan')):.2f}" for c in ("bb", "bp", "pb", "pp")) +
                  (f"  ← {', '.join(marks)}" if marks else ""))
    axis_verdicts = {
        est: (hybrid_recovered[est] if hybrid_seen[est] else None)
        for est in ("g10", "g01")
    }
    if any(v is False for v in axis_verdicts.values()):
        verdicts["C1'_hybrid"] = False
    elif all(v is True for v in axis_verdicts.values()):
        verdicts["C1'_hybrid"] = True
    print("    hybrid 축별: " + ", ".join(
        f"{est}={'PASS' if value else ('FAIL' if value is False else '미판정')}"
        for est, value in axis_verdicts.items()
    ))
    print()

    # ---- C3: downstream ----
    c3_results: list[bool] = []
    for run in runs:
        ds = {f.stem.replace("downstream_", ""): load(f) for f in run.glob("downstream_*.json")}
        ds = {k: v for k, v in ds.items() if v}
        if not ds:
            continue
        base = next(iter(ds.values())).get("base_acc")
        line = ", ".join(f"{k}={v['val_acc']:.3f}" for k, v in sorted(ds.items()))
        print(f"[{run.name}] downstream (base={base:.3f}): {line}")
        if "oracle" in ds and "random" in ds:
            c3_results.append(ds["oracle"]["val_acc"] >= ds["random"]["val_acc"] - 0.02)
    if c3_results:
        verdicts["C3_downstream"] = all(c3_results)
    print()

    # ---- 종합 ----
    print("=== 종합 ===")
    labels = {"C1_g10": "C1 g10(prefix만) 실패 실증", "C1_g01": "C1 g01(suffix만) 실패 실증",
              "C1'_hybrid": "C1' hybrid 축 교체로 회복", "C2_certagrad": "C2 CertaGrad 인증·절약",
              "C3_downstream": "C3 downstream 비열등"}
    for key, label in labels.items():
        v = verdicts[key]
        print(f"  {'PASS' if v else ('FAIL' if v is False else '미판정')}  {label}")
    core = [verdicts["C1_g10"], verdicts["C1_g01"],
            verdicts["C1'_hybrid"], verdicts["C2_certagrad"]]
    if all(core) and verdicts["C3_downstream"] is True:
        print("\n→ 전체 게이트 조건 충족 — 원고 착수 조건 성립.")
    elif all(core) and verdicts["C3_downstream"] is False:
        print("\n→ 핵심 조건은 충족했지만 downstream 비열등 조건 실패.")
    elif all(core):
        print("\n→ 게이트 핵심 조건 충족. downstream 산출물 대기 중.")
    elif any(v is False for v in core):
        print("\n→ 핵심 조건 실패 있음 — concept 사망 조건 대조 필요. 수치 원인 분석 권장.")
    else:
        print("\n→ 산출물 부족 — 미완료 스테이지 확인.")
    return verdicts


def main() -> int:
    out_root = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/pilot")
    judge(out_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
