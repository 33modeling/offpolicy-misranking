"""Fresh-audit 비용–품질 frontier 사후 분석 (concept 'F 계획', GPU 불필요).

    python3 src/frontier.py <run_dir> [<run_dir> ...]

v2 산출물만으로 selection 정책 스펙트럼을 시뮬레이션한다:
  stale(g00/g10/g01/g11) · passrate(Beta-posterior 난이도) · random
  · fresh(m)               — 프롬프트 전수에 m개 fresh 관측
  · audit_random(p, m)     — stale 순위 + 무작위 p-부분집합만 fresh로 교체
  · audit_boundary(p, m)   — stale 순위 + top-k 경계 근접 p-부분집합만 교체

누수 차단 프로토콜: oracle micro-group을 반으로 갈라 **짝수 그룹 = 정책 관측**,
**홀수 그룹 = 진실(truth)** 로만 쓴다 — 정책이 진실과 표본을 공유하지 않는다
(감사 blocker C의 교훈). 진실 top-k는 홀수-절반 평균 gradient의 val 정렬 순위.

출력: $OM_RESULTS(없으면 $OM_WORK/results)/FRONTIER.md + frontier.json
  F1 run별 정책×예산 precision/regret   F2 dataset 집계(seed 평균±sd)
  F3 predictor-family vs gradient-family   F4 조건 지표(Q1 위상도 재료)
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch

EST = ("g00", "g10", "g01", "g11")
FRAC = 0.10
REPEATS = 20          # 확률 정책(random/audit 표집)의 반복 수
AUDIT_FRACS = (0.01, 0.05, 0.10, 0.25)
FRESH_MS = (1, 2, 4)  # 짝수-절반(최대 4그룹) 안에서의 관측 깊이


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    na, nb = a.norm(), b.norm()
    if na == 0 or nb == 0:
        return 0.0
    return float((a @ b) / (na * nb))


def topk_ids(scores: dict, k: int, rng: random.Random) -> set:
    jit = {i: rng.random() for i in scores}
    return set(sorted(scores, key=lambda i: (-scores[i], jit[i]))[:k])


class Run:
    """한 run의 산출물 로드 + 짝/홀 분리 점수."""

    def __init__(self, root: Path):
        self.root = root
        self.name = root.name
        micro = torch.load(root / "oracle_micro_groups.pt", weights_only=True)
        self.val = torch.load(root / "val_gradient.pt", weights_only=True).float()
        self.ids = sorted(micro)
        self.n = len(self.ids)
        self.k = max(1, int(self.n * FRAC))
        # 짝수 그룹 = 정책 관측 풀, 홀수 그룹 = 진실 전용
        self.obs = {i: micro[i][0::2].float() for i in self.ids}
        tru = {i: micro[i][1::2].float() for i in self.ids}
        self.truth_score = {i: cos(tru[i].mean(0), self.val) for i in self.ids}
        h = next(iter(tru.values())).shape[0] // 2
        ha = {i: cos(tru[i][:h].mean(0), self.val) for i in self.ids}
        hb = {i: cos(tru[i][h:].mean(0), self.val) for i in self.ids}
        r0 = random.Random(7)
        self.truth_reliability = len(topk_ids(ha, self.k, r0)
                                     & topk_ids(hb, self.k, random.Random(7))) / self.k
        off = json.loads((root / "scores_offpolicy.json").read_text())
        self.stale = {e: {int(i): v["score"] for i, v in off[e].items()} for e in EST}
        # β pass-rate (behavior rollout 재사용 — fresh 비용 0)
        agg: dict[int, list[float]] = defaultdict(list)
        bpath = root / "rollouts_behavior_train.jsonl"
        if bpath.exists():
            for line in bpath.open():
                r = json.loads(line)
                agg[r["prompt_idx"]].append(r["reward"])
        self.passrate = {i: (1 + sum(agg[i])) / (2 + len(agg[i])) for i in self.ids}
        self.max_obs = next(iter(self.obs.values())).shape[0]

    # ---- 정책들: (선택집합, fresh 관측 수) ----
    def obs_score(self, i: int, m: int) -> float:
        return cos(self.obs[i][:m].mean(0), self.val)

    def pol_stale(self, est: str, rng) -> tuple[set, int]:
        return topk_ids(self.stale[est], self.k, rng), 0

    def pol_passrate(self, rng) -> tuple[set, int]:
        sc = {i: -abs(self.passrate[i] - 0.5) for i in self.ids}  # 중간 난이도 선호
        return topk_ids(sc, self.k, rng), 0

    def pol_random(self, rng) -> tuple[set, int]:
        return set(rng.sample(self.ids, self.k)), 0

    def pol_fresh(self, m: int, rng) -> tuple[set, int]:
        m = min(m, self.max_obs)
        sc = {i: self.obs_score(i, m) for i in self.ids}
        return topk_ids(sc, self.k, rng), self.n * m

    def pol_audit(self, est: str, p: float, m: int, boundary: bool, rng) -> tuple[set, int]:
        m = min(m, self.max_obs)
        n_audit = max(1, int(self.n * p))
        base = self.stale[est]
        if boundary:  # stale 점수의 k-경계에 가까운 순
            kth = sorted(base.values(), reverse=True)[self.k - 1]
            cand = sorted(self.ids, key=lambda i: abs(base[i] - kth))[:n_audit]
        else:
            cand = rng.sample(self.ids, n_audit)
        sc = dict(base)
        for i in cand:  # audit된 프롬프트는 fresh 관측으로 점수 교체
            sc[i] = self.obs_score(i, m)
        return topk_ids(sc, self.k, rng), n_audit * m

    # ---- 평가 ----
    def evaluate(self, sel: set) -> dict:
        rng = random.Random(11)
        t_top = topk_ids(self.truth_score, self.k, rng)
        prec = len(sel & t_top) / self.k
        best = sum(sorted((self.truth_score[i] for i in self.ids), reverse=True)[:self.k]) / self.k
        got = sum(self.truth_score[i] for i in sel) / self.k
        return {"precision": prec, "regret": best - got}


def simulate(run: Run) -> list[dict]:
    rows = []

    def add(policy: str, fn, stochastic: bool):
        ps, rs, cost = [], [], 0
        for r in range(REPEATS if stochastic else 1):
            rng = random.Random(1000 + r)
            sel, cost = fn(rng)
            ev = run.evaluate(sel)
            ps.append(ev["precision"]); rs.append(ev["regret"])
        mean = lambda xs: sum(xs) / len(xs)
        sd = lambda xs: (sum((x - mean(xs)) ** 2 for x in xs) / max(1, len(xs) - 1)) ** 0.5
        rows.append({"run": run.name, "policy": policy, "fresh_groups": cost,
                     "budget_frac": cost / (run.n * run.max_obs),
                     "precision": mean(ps), "precision_sd": sd(ps) if stochastic else 0.0,
                     "regret": mean(rs)})

    for e in EST:
        add(f"stale_{e}", lambda rng, e=e: run.pol_stale(e, rng), True)
    add("passrate_beta", run.pol_passrate, True)
    add("random", run.pol_random, True)
    for m in FRESH_MS:
        add(f"fresh_m{m}", lambda rng, m=m: run.pol_fresh(m, rng), True)
    base_est = "g00"  # 실무 기본(무보정)을 audit의 stale 기반으로
    for p in AUDIT_FRACS:
        for m in (2,):
            add(f"audit_rand_p{int(p*100)}_m{m}",
                lambda rng, p=p, m=m: run.pol_audit(base_est, p, m, False, rng), True)
            add(f"audit_bnd_p{int(p*100)}_m{m}",
                lambda rng, p=p, m=m: run.pol_audit(base_est, p, m, True, rng), True)
    return rows


def condition_row(run: Run) -> dict:
    row = {"run": run.name, "n": run.n, "k": run.k,
           "truth_reliability": round(run.truth_reliability, 3)}
    # live: β 보상이 혼합인 프롬프트 비율 (Beta 사후평균이 0/1 경계에서 떨어짐)
    live = sum(1 for i in run.ids if 0.05 < run.passrate[i] < 0.95)
    row["live_frac_beta"] = round(live / run.n, 3)
    ds = sorted(run.root.glob("divergence_stats*.json"))
    if ds:
        agg: dict[str, list] = defaultdict(list)
        for p in ds:
            for kk, vv in json.loads(p.read_text()).items():
                if isinstance(vv, (int, float)):
                    agg[kk].append(vv)
        for kk in ("token_kl_beta_pi", "traj_ess_frac_g11", "clipfrac_g11", "clipfrac_g10"):
            if kk in agg:
                row[kk] = round(sum(agg[kk]) / len(agg[kk]), 4)
    # 진실 점수의 k-경계 margin
    vals = sorted(run.truth_score.values(), reverse=True)
    row["truth_margin_k"] = round(vals[run.k - 1] - vals[run.k], 4) if len(vals) > run.k else None
    return row


def to_md(all_rows: list[dict], conds: list[dict]) -> str:
    lines = ["# Fresh-audit 비용–품질 frontier", ""]
    lines.append("진실=홀수 micro-group 전용, 정책 관측=짝수 그룹 — 표본 비공유 프로토콜. "
                 f"확률 정책은 {REPEATS}회 평균±sd.")
    runs = sorted({r["run"] for r in all_rows})

    lines.append("\n## F1. run별 정책 × 예산")
    lines.append("| run | policy | fresh(관측) | 예산% | precision | ±sd | regret |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in all_rows:
        lines.append(f"| {r['run']} | {r['policy']} | {r['fresh_groups']} | "
                     f"{r['budget_frac']:.0%} | {r['precision']:.3f} | "
                     f"{r['precision_sd']:.3f} | {r['regret']:.4f} |")

    lines.append("\n## F2. dataset 집계 (seed 평균)")
    def tag(run_name: str) -> str:
        import re
        return re.sub(r"-s\d+", "", run_name)
    grp: dict[tuple, list] = defaultdict(list)
    for r in all_rows:
        grp[(tag(r["run"]), r["policy"])].append(r["precision"])
    lines.append("| dataset | policy | precision(seed평균) | seed-sd | seeds |")
    lines.append("|---|---|---|---|---|")
    for (t, pol), xs in sorted(grp.items()):
        m = sum(xs) / len(xs)
        sd = (sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1)) ** 0.5
        lines.append(f"| {t} | {pol} | {m:.3f} | {sd:.3f} | {len(xs)} |")

    lines.append("\n## F3. family 비교 (run별 최고 정책)")
    fam = {"gradient(stale)": lambda p: p.startswith("stale_"),
           "predictor(passrate)": lambda p: p.startswith("passrate"),
           "audit(random)": lambda p: p.startswith("audit_rand"),
           "audit(boundary)": lambda p: p.startswith("audit_bnd"),
           "fresh": lambda p: p.startswith("fresh_"),
           "random": lambda p: p == "random"}
    lines.append("| run | " + " | ".join(fam) + " |")
    lines.append("|---|" + "---|" * len(fam))
    for rn in runs:
        cells = []
        for _, match in fam.items():
            xs = [r["precision"] for r in all_rows if r["run"] == rn and match(r["policy"])]
            cells.append(f"{max(xs):.3f}" if xs else "—")
        lines.append(f"| {rn} | " + " | ".join(cells) + " |")

    lines.append("\n## F4. 조건 지표 (Q1 위상도 재료)")
    if conds:
        keys = sorted({k for c in conds for k in c} - {"run"})
        lines.append("| run | " + " | ".join(keys) + " |")
        lines.append("|---|" + "---|" * len(keys))
        for c in conds:
            lines.append(f"| {c['run']} | " + " | ".join(str(c.get(k, "—")) for k in keys) + " |")
    return "\n".join(lines)


def main() -> int:
    roots = [Path(p) for p in sys.argv[1:]]
    all_rows, conds = [], []
    for root in roots:
        need = ["oracle_micro_groups.pt", "val_gradient.pt", "scores_offpolicy.json"]
        missing = [f for f in need if not (root / f).exists()]
        if missing:
            print(f"[skip] {root.name}: 산출물 누락 {missing}")
            continue
        run = Run(root)
        print(f"[run] {run.name}: n={run.n} k={run.k} obs_max={run.max_obs} "
              f"truth_rel={run.truth_reliability:.3f}")
        all_rows += simulate(run)
        conds.append(condition_row(run))
    if not all_rows:
        print("분석 가능한 run 없음")
        return 1
    base = os.environ.get("OM_RESULTS") or (
        os.environ.get("OM_WORK", "") and os.environ["OM_WORK"] + "/results") or "results"
    out = Path(base)
    out.mkdir(parents=True, exist_ok=True)
    md = to_md(all_rows, conds)
    (out / "FRONTIER.md").write_text(md)
    (out / "frontier.json").write_text(json.dumps(
        {"rows": all_rows, "conditions": conds}, indent=1))
    print(md)
    print(f"\n[저장됨] {out / 'FRONTIER.md'} · frontier.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
