"""B1 — K-스케일링 floor 곡선 전 조건 확장 (GPU 0, 기존 산출물만 사용).

    python src/kcurve_all.py <runs_root>

kcurve_floor.py(P4-0)의 확장판. 차이 두 가지:

1) run 발견: 이름 규약 대신 **산출물 존재**로 찾는다 — runs_root 아래 깊이 2까지
   oracle_micro_groups.pt + val_gradient.pt + scores_splithalf.json 이 모두 있는
   디렉토리 전부(smoke 제외). v1 게이트(gate*, DONE 없음)·math500·drift 스윕이
   전부 포함된다.

2) 판정 무결성: **사전 등록된 P4-0 판정은 바꾸지 않는다.** 판정 절은
   kcurve_floor.py와 동일한 대상(v2-* + DONE + gsm8k 계열)으로만 계산해 그대로
   재출력하고, 나머지 run은 "확장 증거" 표로만 보고한다(표결 아님 — 원고의
   증거 폭 확장용). v1 run을 표결에 사후 편입하는 것은 사전 등록 위반이라
   의도적으로 하지 않는다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from kcurve_floor import GO_MULT, find_fresh_k, observed_curve, predicted_floor, sb, tag_of

ARTIFACTS = ("oracle_micro_groups.pt", "val_gradient.pt", "scores_splithalf.json")


def has_artifacts(d: Path) -> bool:
    return all((d / a).exists() for a in ARTIFACTS)


def discover(root: Path) -> list[Path]:
    found = set()
    for depth_glob in ("*", "*/*"):
        for d in root.glob(depth_glob):
            if d.is_dir() and "smoke" not in d.name and has_artifacts(d):
                found.add(d)
    return sorted(found)


def analyze(d: Path):
    micro = torch.load(d / "oracle_micro_groups.pt", map_location="cpu")
    v = torch.load(d / "val_gradient.pt", map_location="cpu").float()
    G = min(t.shape[0] for t in micro.values())
    if G < 2:
        return None
    stack = torch.stack([micro[i][:G].float() for i in sorted(micro)])
    curve, k, chance = observed_curve(stack, v)
    P = stack.shape[0]
    gsize = max(1, find_fresh_k(d, G) // G)
    r1 = curve[1][3]
    preds, k_need = [], None
    for kp in (64, 128, 256):
        m = max(1, kp // (2 * gsize))
        f = predicted_floor(sb(r1, m), P, k)
        preds.append((kp, f))
        if k_need is None and f >= GO_MULT * chance:
            k_need = kp
    mmax = G // 2
    pred_at_mmax = predicted_floor(sb(r1, mmax), P, k)
    return dict(name=d.name, rel=str(d), tag=tag_of(d.name), P=P, G=G,
                gsize=gsize, chance=chance, curve=curve, r1=r1,
                pred_at_mmax=pred_at_mmax, obs_at_mmax=curve[mmax][0],
                preds=preds, k_need=k_need)


def main() -> int:
    root = Path(sys.argv[1])
    all_runs = discover(root)
    prereg = [d for d in all_runs
              if d.parent == root and __import__("run_select").is_generation_run(d.name)
              and (d / "DONE").exists()]
    extra = [d for d in all_runs if d not in prereg]

    rows_pre = [r for r in (analyze(d) for d in prereg) if r]
    rows_ext = [r for r in (analyze(d) for d in extra) if r]

    votes = [r["k_need"] is not None for r in rows_pre if r["tag"] == "gsm8k"]
    eligible = len(votes)
    go = eligible > 0 and sum(votes) * 2 > eligible

    print("# B1 — K-스케일링 floor 곡선, 전 조건 확장 (기존 산출물, GPU 0)\n")
    print("사전 등록 판정(P4-0 규칙·대상 불변)과 확장 증거(v1 게이트 포함, 표결 아님)를")
    print("분리 보고한다. 곡선이 K'에서 **하강**하면 포화-공유 인공 겹침(함정 4호)의")
    print("시그니처로 읽는다 — 그 run의 floor는 상한으로만 해석.\n")

    def table(rows, title):
        print(f"## {title}\n")
        if not rows:
            print("(해당 run 없음)\n")
            return
        print("| run | 계열 | n | chance | r1 | 관측 K'=8→최대 | 예측 K'=128/256 | 2×chance 도달 | 곡선 방향 |")
        print("|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            ms = sorted(r["curve"])
            first, last = r["curve"][ms[0]][0], r["curve"][ms[-1]][0]
            kmax = 2 * ms[-1] * r["gsize"]
            shape = "상승" if last > first + 0.01 else ("하강⚠️" if first > last + 0.01 else "평탄")
            pred = {kp: f for kp, f in r["preds"]}
            reach = f"K'={r['k_need']}" if r["k_need"] else "없음(≤256)"
            print(f"| {r['name']} | {r['tag']} | {r['P']} | {r['chance']:.3f} | {r['r1']:.3f} "
                  f"| {first:.3f}→{last:.3f} (K'={kmax}) "
                  f"| {pred.get(128, float('nan')):.3f} / {pred.get(256, float('nan')):.3f} "
                  f"| {reach} | {shape} |")
        print()

    table(rows_pre, "사전 등록 대상 (P4-0과 동일: v2-* 완주)")
    table(rows_ext, "확장 증거 (v1 게이트·기타 — 표결에 불포함)")

    print("## 판정 (사전 등록 규칙 재출력 — 확장분 미반영)\n")
    print(f"- 대상(gsm8k 계열): {eligible}개, 'K'<=256에서 도달 예상' {sum(votes)}개 → "
          + ("**확장 권고**" if go else "**구조적 부재**" if eligible else "**판정 불가**"))
    n_signal = sum(1 for r in rows_ext
                   if r["curve"][max(r["curve"])][0] >= GO_MULT * r["chance"])
    print(f"- 확장 증거 요약: {len(rows_ext)}개 run 중 최대 K'에서 floor ≥ 2×chance 인 것 "
          f"{n_signal}개 — 원고 '실현 지도' 증거 폭 확장용.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
