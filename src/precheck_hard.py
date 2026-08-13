"""P3-0 사전 검력 체크 — go_hard(hard-slice 준-개입)를 GPU로 돌리기 전 판정.

    python3 src/precheck_hard.py <runs_root> [--gate]

원리: go_hard가 만들 풀(live 프롬프트만 = 0 < β pass-rate < 1)은 이미 완주한
run 안에 부분집합으로 존재한다. 그 부분집합만으로 oracle split-half 일치도
(교정 floor, 독립 jitter)를 다시 재면, 같은 필터로 풀을 새로 짜도 floor가
오를지 안 오를지를 GPU 없이 미리 추정할 수 있다.

사전 등록 규칙 (concept.md P3 설계서와 동일 — 실행 전 등록):
  - 대상: gsm8k 계열 완주 run 중 live 부분집합 |H| >= 80 인 것
  - GO   : 대상 run의 과반에서 live 조건부 floor(20-jitter 평균) >= 2×chance
  - NO-GO: 그 외 → go_hard를 건너뛰고 P4(oracle K-스케일링 floor 곡선)로
  - 판정 불가: 대상 run이 0개 (exit 4) → prescreen만 먼저 돌려 풀 분포 확인

--gate: 보고서 없이 exit code만 반환 (0=GO, 3=NO-GO, 4=판정 불가)

한계(설계서에 명시): 부분집합 floor는 |H|가 작아 노이즈가 크고(부트스트랩
CI 병기), go_hard 풀은 2000-프리스크린에서 오므로 분포가 약간 다르다.
이 체크는 "오를 가능성의 근거"를 재는 것이지 본실행 결과의 보증이 아니다.
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

N_JITTER = 20
N_BOOT = 200
MIN_H = 80          # 이보다 작으면 floor 추정 자체가 불안정 → 대상 제외
GO_MULT = 2.0       # floor >= GO_MULT × chance 이면 "신호 체제" 후보


def load_passrates(run: Path) -> dict[int, float]:
    """β pass-rate per prompt — make_hard_pool.py와 동일한 중복 제거 규칙."""
    shards = sorted(run.glob("rollouts_behavior_train*.jsonl"))
    merged = run / "rollouts_behavior_train.jsonl"
    if merged in shards:
        shards = [merged]
    acc: dict[int, list[float]] = defaultdict(list)
    seen: set[tuple[int, int]] = set()
    for sh in shards:
        for line in sh.open():
            r = json.loads(line)
            key = (r["prompt_idx"], r.get("rollout_idx", len(acc[r["prompt_idx"]])))
            if key in seen:
                continue
            seen.add(key)
            acc[r["prompt_idx"]].append(float(r["reward"]))
    return {i: sum(v) / len(v) for i, v in acc.items() if v}


def topk(scores: dict, k: int, rng: random.Random) -> set:
    jit = {i: rng.random() for i in scores}
    return set(sorted(scores, key=lambda i: (-scores[i], jit[i]))[:k])


def floor_on(hv: dict[int, dict], idxs: list[int]) -> tuple[float, float, float, int, float]:
    """부분집합 조건부 교정 floor — 20개 독립 jitter쌍 평균과 범위."""
    k = max(1, round(0.10 * len(idxs)))
    a = {i: hv[i]["a"] for i in idxs}
    b = {i: hv[i]["b"] for i in idxs}
    vals = []
    for j in range(N_JITTER):
        ta = topk(a, k, random.Random(1000 + j))
        tb = topk(b, k, random.Random(104729 + j))
        vals.append(len(ta & tb) / k)
    return sum(vals) / len(vals), min(vals), max(vals), k, k / len(idxs)


def floor_boot_ci(hv: dict[int, dict], idxs: list[int]) -> tuple[float, float]:
    """프롬프트 부트스트랩 95% CI (재표집마다 jitter 새로)."""
    vals = []
    for bidx in range(N_BOOT):
        rng = random.Random(50000 + bidx)
        sample = [rng.choice(idxs) for _ in idxs]      # 위치 키로 중복 허용
        a = {p: hv[i]["a"] for p, i in enumerate(sample)}
        b = {p: hv[i]["b"] for p, i in enumerate(sample)}
        k = max(1, round(0.10 * len(sample)))
        ta = topk(a, k, random.Random(60000 + bidx))
        tb = topk(b, k, random.Random(70000 + bidx))
        vals.append(len(ta & tb) / k)
    vals.sort()
    return vals[int(0.025 * N_BOOT)], vals[int(0.975 * N_BOOT) - 1]


def precisions_on(run: Path, idxs: list[int]) -> dict[str, float] | None:
    """탐색적 미리보기 — H 제한 estimator top-k precision (판정에 쓰지 않음)."""
    try:
        oracle = {int(i): v["score"] for i, v in
                  json.loads((run / "scores_oracle.json").read_text()).items()}
        off = json.loads((run / "scores_offpolicy.json").read_text())
    except Exception:
        return None
    sub = [i for i in idxs if i in oracle]
    if len(sub) < MIN_H:
        return None
    k = max(1, round(0.10 * len(sub)))
    out = {}
    for est in ("g00", "g10", "g01", "g11"):
        if est not in off:
            continue
        sc = {int(i): v["score"] for i, v in off[est].items() if int(i) in oracle}
        vals = []
        for j in range(N_JITTER):
            ot = topk({i: oracle[i] for i in sub}, k, random.Random(1000 + j))
            et = topk({i: sc[i] for i in sub if i in sc}, k, random.Random(104729 + j))
            vals.append(len(ot & et) / k)
        out[est] = sum(vals) / len(vals)
    out["_k"] = k
    out["_chance"] = k / len(sub)
    return out


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
            if (d / "DONE").exists() and "smoke" not in d.name]

    rows, votes = [], []
    for d in runs:
        try:
            hv = {int(i): v for i, v in
                  json.loads((d / "scores_splithalf.json").read_text()).items()}
        except Exception:
            continue
        rates = load_passrates(d)
        if not rates:
            continue
        all_idx = [i for i in hv if i in rates]
        live = [i for i in all_idx if 0.0 < rates[i] < 1.0]
        mid = [i for i in all_idx if 0.25 <= rates[i] <= 0.75]

        f_all = floor_on(hv, all_idx) if len(all_idx) >= MIN_H else None
        f_live = floor_on(hv, live) if len(live) >= MIN_H else None
        f_mid = floor_on(hv, mid) if len(mid) >= MIN_H else None
        ci = floor_boot_ci(hv, live) if len(live) >= MIN_H else None
        prev = precisions_on(d, live) if len(live) >= MIN_H else None

        tag = tag_of(d.name)
        if tag == "gsm8k" and f_live is not None:
            votes.append(f_live[0] >= GO_MULT * f_live[4])
        rows.append((d.name, tag, len(all_idx), len(live), len(mid),
                     f_all, f_live, ci, f_mid, prev))

    eligible = len(votes)
    go = eligible > 0 and sum(votes) * 2 > eligible
    code = 4 if eligible == 0 else (0 if go else 3)
    if gate:
        return code

    def fmt(f):
        if f is None:
            return "표본 부족"
        m, lo, hi, k, ch = f
        return f"{m:.3f} [{lo:.3f}~{hi:.3f}] (k={k}, chance={ch:.3f})"

    print("# P3-0 사전 검력 체크 — hard 부분집합 조건부 floor\n")
    print("go_hard(풀 필터 준-개입)를 돌리기 전에, 기존 완주 run에서 같은 필터")
    print("(0<β pass-rate<1)를 적용한 부분집합의 oracle 재현성(교정 floor)을 잰다.")
    print("여기서 floor가 오르지 않으면 새 풀에서도 오를 근거가 없다.\n")
    print("| run | 계열 | n | live | floor(전체) | floor(live) | live 95% CI | floor(중간대 0.25~0.75) |")
    print("|---|---|---|---|---|---|---|---|")
    for name, tag, n, nl, nm, fa, fl, ci, fm, _ in rows:
        ci_s = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "-"
        print(f"| {name} | {tag} | {n} | {nl} | {fmt(fa)} | {fmt(fl)} | {ci_s} | {fmt(fm)} |")

    print("\n## 탐색적 미리보기 — live 제한 estimator precision (판정 아님)\n")
    print("사후 부분집합이라 확증용으로 쓰지 않는다. 본판정은 go_hard 3-seed에서만.\n")
    print("| run | g00 | g10 | g01 | g11 | chance |")
    print("|---|---|---|---|---|---|")
    for name, *_, prev in rows:
        if prev:
            print(f"| {name} | " + " | ".join(
                f"{prev.get(e, float('nan')):.3f}" for e in ("g00", "g10", "g01", "g11"))
                + f" | {prev['_chance']:.3f} |")

    print("\n## 판정 (사전 등록 규칙)\n")
    print(f"- 대상(gsm8k 계열, |H|>={MIN_H}): {eligible}개 run, "
          f"floor(live) >= {GO_MULT}×chance 충족 {sum(votes)}개")
    if code == 0:
        print("- **GO** — live 필터로 신호 체제가 형성될 근거 있음. `bash scripts/go_hard.sh` 진행.")
    elif code == 3:
        print("- **NO-GO** — live 부분집합에서도 floor가 chance 근처. 같은 필터로 풀을")
        print("  새로 짜도 오를 근거가 없다. go_hard는 건너뛰고 P4(oracle K-스케일링")
        print("  floor 곡선: K=32→64→128에서 floor가 오르는지 — '신호가 있는데 K 부족'")
        print("  인지 '구조적으로 없는지' 판별)로 간다.")
    else:
        print("- **판정 불가** — 조건을 만족하는 완주 run이 없다. 먼저 prescreen만 돌려")
        print("  풀 크기·pass-rate 분포를 확인할 것.")
    return code


if __name__ == "__main__":
    sys.exit(main())
