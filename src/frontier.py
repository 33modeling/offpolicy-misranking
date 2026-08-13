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

    def features(self, i: int) -> list[float]:
        """2D-REFRESH 특징: 셀 점수 3종 + 축별 불일치 + β 난이도."""
        s00, s10, s01 = (self.stale[e][i] for e in ("g00", "g10", "g01"))
        p = self.passrate[i]
        return [s00, s10, s01,
                abs(s10 - s00), abs(s01 - s00), abs(s10 - s01),
                p, p * (1 - p)]

    def pol_2d_refresh(self, budget_groups: int, mode: str, rng) -> tuple[set, int]:
        """감사(랭커 선택)+획득함수 refresh — concept '2D-REFRESH' v0.1 시뮬.

        v0 교훈(합성 seed1): 소표본 ridge로 전체 순위를 대체하면 멀쩡한 stale을
        망가뜨린다. v0.1: audit은 **랭커 선택기** — 후보(셀 3종·passrate·ridge[
        audit≥12일 때만])의 audit 실측 상관을 재서 최고를 base로 쓰고, 그 위에
        불확실·불일치·경계 획득함수로 refresh. 최악에도 최선-stale 수준 보장.
        mode: full | margin_only | disagree_only (획득함수 ablation)
        """
        m_a = 2
        spent = 0
        n_a = min(max(8, budget_groups // (4 * m_a)), budget_groups // (2 * m_a))
        if n_a < 3:
            return topk_ids(self.stale["g00"], self.k, rng), 0
        by_p = sorted(self.ids, key=lambda i: self.passrate[i])
        terciles = [by_p[j::3] for j in range(3)]
        # audit 이중 목적: 층화(선택기 정보) + 각 층에서 g00 경계 근접 우선(경계 교정)
        kth00 = sorted(self.stale["g00"].values(), reverse=True)[self.k - 1]
        audit: list[int] = []
        for t in terciles:
            near = sorted(t, key=lambda i: abs(self.stale["g00"][i] - kth00))
            pool_t = near[:max(2, 2 * (n_a // 3 + 1))]
            audit += rng.sample(pool_t, min(len(pool_t), max(1, n_a // 3)))
        audit = audit[:n_a]
        spent += len(audit) * m_a
        y = {i: self.obs_score(i, m_a) for i in audit}

        # ---- 후보 랭커와 audit 상관 ----
        def corr(sc: dict) -> float:
            xs = torch.tensor([sc[i] for i in audit], dtype=torch.float64)
            ys = torch.tensor([y[i] for i in audit], dtype=torch.float64)
            xs, ys = xs - xs.mean(), ys - ys.mean()
            d = xs.norm() * ys.norm()
            return float((xs @ ys) / d) if d > 0 else 0.0

        cands: dict[str, dict] = {e: self.stale[e] for e in ("g00", "g10", "g01")}
        cands["passrate"] = {i: -abs(self.passrate[i] - 0.5) for i in self.ids}
        if len(audit) >= 12:  # ridge는 표본이 있을 때만 후보에 진입
            X = torch.tensor([self.features(i) for i in audit], dtype=torch.float64)
            Y = torch.tensor([y[i] for i in audit], dtype=torch.float64)
            Xm, Xs = X.mean(0), X.std(0).clamp_min(1e-6)
            Z = (X - Xm) / Xs
            A = Z.T @ Z + 1.0 * torch.eye(Z.shape[1], dtype=torch.float64)
            w = torch.linalg.solve(A, Z.T @ (Y - Y.mean()))
            b0 = float(Y.mean())
            F = torch.tensor([self.features(i) for i in self.ids], dtype=torch.float64)
            P = ((F - Xm) / Xs) @ w + b0
            cands["ridge"] = {i: float(P[j]) for j, i in enumerate(self.ids)}
        # 스위치 문턱: 소표본 상관 노이즈로 멀쩡한 기본 랭커를 갈아타는 것 방지
        corrs = {nm: corr(sc) for nm, sc in cands.items()}
        base_name = "g00"
        best = max(corrs, key=corrs.get)
        if corrs[best] >= corrs["g00"] + 0.15:
            base_name = best
        base = cands[base_name]
        base_corr = corrs[base_name]
        # audit 잔차 규모(불확실도 대용) — 3분위별
        res_by_t = []
        for t in terciles:
            rs = [abs(base[i] - y[i]) for i in t if i in y]
            res_by_t.append(sum(rs) / len(rs) if rs else 0.1)
        tvix = {i: j for j, t in enumerate(terciles) for i in t}
        r_i = {i: res_by_t[tvix[i]] for i in self.ids}
        d_i = {i: max(self.features(i)[3:6]) for i in self.ids}

        # ---- 획득 루프: base 순위 위에 fresh 교체 ----
        score = dict(base)
        for i in audit:
            score[i] = y[i]
        refreshed = set(audit)
        # corr-적응 지수: audit이 base 순위를 신뢰할수록 경계(gap)에 집중하고,
        # 신뢰가 낮으면 gap을 무시하고 불일치·불확실 쪽을 탐색한다 (anti-순위 체제 대응)
        w_gap = min(1.0, max(0.0, base_corr))
        while spent + m_a <= budget_groups:
            kth = sorted(score.values(), reverse=True)[self.k - 1]
            def acq(i: int) -> float:
                gap = abs(score[i] - kth) + 1e-3
                if mode == "margin_only":
                    return 1.0 / gap
                if mode == "disagree_only":
                    return d_i[i]
                return (r_i[i] + 0.5 * d_i[i]) / (gap ** w_gap + 1e-3)
            cand = [i for i in self.ids if i not in refreshed]
            if not cand:
                break
            batch = sorted(cand, key=acq, reverse=True)[:8]
            took = False
            for i in batch:
                if spent + m_a > budget_groups:
                    break
                score[i] = self.obs_score(i, m_a)
                refreshed.add(i)
                spent += m_a
                took = True
            if not took:
                break
        sel = topk_ids(score, self.k, rng)
        self.last_base = (base_name, round(base_corr, 3))  # 진단용
        return sel, spent

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

    def pol_floor_gated(self, rng, diag_prompts: int = 48,
                        tau_mult: float = 2.0, p: float = 0.10, m: int = 2) -> tuple[set, int]:
        """제안 절차(floor-gated audit): ① 진단 — 소표본(diag_prompts)에 fresh를
        부어 짝수-절반 split-half로 floor(회복 가능 신호 크기)를 추정
        ② 분기 — floor_est < tau_mult×chance면 선택 은퇴(random, 추가 비용 0);
        아니면 stale(g00) 순위의 top-k 경계만 audit(p, m)으로 재채점.

        비용 = 진단(diag_prompts×max_obs) + 분기별 추가. 홀수 절반(truth)은 결코
        쓰지 않는다(누수 없음). tau_mult=2.0은 사전 지정 규칙 — 관찰된 두 체제
        (0.14~0.18 vs 0.75~0.80)로 튜닝한 것이 아니라 chance의 배수로 고정.
        """
        diag = rng.sample(self.ids, min(diag_prompts, self.n))
        h = self.max_obs // 2
        if h >= 1:
            ka = {i: cos(self.obs[i][:h].mean(0), self.val) for i in diag}
            kb = {i: cos(self.obs[i][h:].mean(0), self.val) for i in diag}
            kd = max(1, int(len(diag) * FRAC))
            floor_est = len(topk_ids(ka, kd, rng) & topk_ids(kb, kd, rng)) / kd
        else:
            floor_est = 0.0
        diag_cost = len(diag) * self.max_obs
        if floor_est < tau_mult * FRAC:
            sel, _ = self.pol_random(rng)
            return sel, diag_cost
        sel, cost = self.pol_audit("g00", p, m, True, rng)
        return sel, diag_cost + cost

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
    add("floor_gated(제안)", lambda rng: run.pol_floor_gated(rng), True)
    for m in FRESH_MS:
        add(f"fresh_m{m}", lambda rng, m=m: run.pol_fresh(m, rng), True)
    base_est = "g00"  # 실무 기본(무보정)을 audit의 stale 기반으로
    for p in AUDIT_FRACS:
        for m in (2,):
            add(f"audit_rand_p{int(p*100)}_m{m}",
                lambda rng, p=p, m=m: run.pol_audit(base_est, p, m, False, rng), True)
            add(f"audit_bnd_p{int(p*100)}_m{m}",
                lambda rng, p=p, m=m: run.pol_audit(base_est, p, m, True, rng), True)
    # 2D-REFRESH와 획득함수 ablation — audit_*와 동일 예산 격자(B = p·n·2)
    for p in AUDIT_FRACS:
        B = max(1, int(p * run.n)) * 2
        for mode, tagm in (("full", "2dref"), ("margin_only", "2dref_marginonly"),
                           ("disagree_only", "2dref_disagreeonly")):
            add(f"{tagm}_p{int(p*100)}",
                lambda rng, B=B, mode=mode: run.pol_2d_refresh(B, mode, rng), True)
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
           "2D-REFRESH": lambda p: p.startswith("2dref_p") or p.startswith("2dref_") and "only" not in p,
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
