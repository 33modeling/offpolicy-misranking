"""결과 테이블 생성기 — 게이트 산출물에서 논문용 표 7종을 CPU만으로 뽑는다.

    python3 src/make_tables.py <run_root> [<run_root> ...]

run_root가 drift* 자식(어댑터 폴더 drift_* 제외)을 가지면 각각을 run으로,
없으면 자신을 단일 run으로 취급한다. 출력: results/TABLES.md (+ 콘솔 echo).

표 목록 (concept '개선 백로그' B절 구현):
  T1 게이트 요약        — floor·k·4추정량 precision(Δfloor)
  T2 정규화 재판정(B3)  — rand 대비 상대 신호 보존율, 절대/상대 문턱 판정 비교
  T3 floor-vs-관측(B2)  — micro-group subsample로 floor 곡선 (판정 가능 최소 관측)
  T4 live fraction(B4)  — 전부정답/전부오답/유신호 프롬프트 비율
  T5 hybrid 인과        — cut별 bb/bp/pb/pp precision (축 교체 회복량)
  T6 C2·margin          — CertaGrad 판정 + 유신호 경계 margin vs α_v
  T7 downstream         — 선택 소스별 val 정확도

각 표는 독립적으로 생성한다 — 산출물이 없거나 손상돼도 그 표에만 사유를 적고
나머지는 계속 만든다 (원칙: 침묵 대신 명시).
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch

from certagrad import angle_radius, eb_radius
from gate_rules import (
    HYBRID_PROTOCOL_SCHEMA,
    has_valid_analysis_protocol,
)
from select_rules import overlap_under_independent_ties, topk_count

FRAC = 0.10
EST = ("g00", "g10", "g01", "g11")


def collect_runs(roots: list[str]) -> list[tuple[str, Path]]:
    runs: list[tuple[str, Path]] = []
    for r in roots:
        root = Path(r)
        if not root.exists():
            continue
        subs = sorted(d for d in root.glob("drift*")
                      if d.is_dir() and not d.name.startswith("drift_"))
        if subs:
            runs += [(f"{root.name}/{d.name}", d) for d in subs]
        else:
            runs.append((root.name, root))
    return runs


def jload(p: Path):
    return json.loads(p.read_text()) if p.exists() else None

def gate_numbers(run: Path) -> dict | None:
    """report.json 우선, 없으면 원시 점수에서 재계산."""
    if not has_valid_analysis_protocol(run):
        return None
    rep = jload(run / "report.json")
    if rep and "noise_floor" in rep:
        return rep
    oracle = jload(run / "scores_oracle.json")
    off = jload(run / "scores_offpolicy.json")
    halves = jload(run / "scores_splithalf.json")
    if not (oracle and off and halves):
        return None
    oracle = {int(i): v for i, v in oracle.items()}
    halves = {int(i): v for i, v in halves.items()}
    k = topk_count(len(oracle), FRAC)
    oracle_scores = {i: v["score"] for i, v in oracle.items()}
    floor = overlap_under_independent_ties(
        {i: h["a"] for i, h in halves.items()},
        {i: h["b"] for i, h in halves.items()},
        k, seed=0,
    )
    out = {"noise_floor": floor.mean, "k": k, "_recomputed": True}
    for est in EST:
        sc = {int(i): v["score"] for i, v in off.get(est, {}).items()}
        if sc:
            overlap = overlap_under_independent_ties(oracle_scores, sc, k, seed=0)
            out[est] = {
                "precision": overlap.mean,
                "jaccard": sum(v / (2 - v) for v in overlap.values) / len(overlap.values),
            }
    return out


def n_universe(run: Path) -> int | None:
    oracle = jload(run / "scores_oracle.json")
    return len(oracle) if oracle else None


# ---------- 표 생성기들 ----------

def t1_gate(runs) -> list[str]:
    rows = ["| run | floor | k | g00 | g10 | g01 | g11 |", "|---|---|---|---|---|---|---|"]
    for name, run in runs:
        rep = gate_numbers(run)
        if not rep:
            rows.append(f"| {name} | (report·점수 없음) | | | | | |")
            continue
        f = rep["noise_floor"]
        cells = []
        for est in EST:
            if est in rep:
                p = rep[est]["precision"]
                cells.append(f"{p:.3f} ({p - f:+.3f})")
            else:
                cells.append("—")
        tag = "†" if rep.get("_recomputed") else ""
        rows.append(f"| {name}{tag} | {f:.3f} | {rep.get('k')} | " + " | ".join(cells) + " |")
    rows.append("")
    rows.append("괄호는 Δfloor. †는 report.json 없이 원시 점수에서 재계산한 run.")
    return rows


def t2_relative(runs) -> list[str]:
    rows = ["| run | floor | rand | " + " | ".join(f"{e} 신호보존" for e in EST)
            + " | 절대문턱(g10/g01) | 상대문턱(g10/g01) |",
            "|---|---|---|" + "---|" * (len(EST) + 2)]
    for name, run in runs:
        rep = gate_numbers(run)
        n = n_universe(run)
        if not (rep and n):
            rows.append(f"| {name} | (산출물 없음) |" + " |" * (len(EST) + 3))
            continue
        f, k = rep["noise_floor"], rep["k"]
        rand = k / n
        sig = f - rand
        rels = {}
        for est in EST:
            if est in rep and sig > 1e-9:
                rels[est] = (rep[est]["precision"] - rand) / sig
        cells = [f"{rels[e]:+.2f}" if e in rels else "판정불능" for e in EST]
        a_v = [f"{'FAIL실증' if rep[e]['precision'] <= f - 0.15 else '미달'}"
               for e in ("g10", "g01") if e in rep]
        r_v = [f"{'FAIL실증' if rels.get(e, 9) <= 0.5 else '미달'}"
               if sig > 1e-9 else "판정불능" for e in ("g10", "g01")]
        rows.append(f"| {name} | {f:.3f} | {rand:.3f} | " + " | ".join(cells)
                    + f" | {'/'.join(a_v) or '—'} | {'/'.join(r_v)} |")
    rows += ["", "신호보존 = (precision−rand)/(floor−rand): 1.0=oracle급, 0=무작위급, "
                 "음수=무작위보다 못함. 상대문턱은 보존율 ≤0.5(신호 절반 상실)를 실증으로 본다 "
                 "— **사후 분석**이므로 원고에는 사전 문턱과 구분해 표기."]
    return rows


def t3_floor_curve(runs) -> list[str]:
    rows = ["| run | 관측 그룹 m (× micro-group) → floor |", "|---|---|"]
    for name, run in runs:
        mg, vg = run / "oracle_micro_groups.pt", run / "val_groups.pt"
        if not (mg.exists() and vg.exists()):
            rows.append(f"| {name} | (micro/val 산출물 없음) |")
            continue
        micro = torch.load(mg, weights_only=True)
        val_groups = torch.load(vg, weights_only=True).float()
        if val_groups.ndim != 2 or val_groups.shape[0] < 2:
            rows.append(f"| {name} | (독립 validation 절반 부족) |")
            continue
        val_a = val_groups[0::2].mean(0)
        val_b = val_groups[1::2].mean(0)
        n_pool = min(s.shape[0] for s in micro.values())
        k = topk_count(len(micro), FRAC)
        pts = []
        for m in range(2, n_pool + 1, 2):
            a_sc, b_sc = {}, {}
            for i, stack in micro.items():
                st = stack[:m].float()
                a_mu, b_mu = st[0::2].mean(0), st[1::2].mean(0)
                a_sc[i] = float((a_mu @ val_a) / (a_mu.norm() * val_a.norm() + 1e-12))
                b_sc[i] = float((b_mu @ val_b) / (b_mu.norm() * val_b.norm() + 1e-12))
            fl = overlap_under_independent_ties(a_sc, b_sc, k, seed=0).mean
            pts.append(f"m={m}: {fl:.3f}")
        rows.append(f"| {name} | " + " · ".join(pts) + " |")
    rows += ["", "split-half floor를 관측 그룹 수 m으로 subsample 재계산 — 곡선이 늦게 "
                 "포화하면 그 run은 관측 부족 체제. '판정 가능 최소 관측 K*' 처방의 근거."]
    return rows


def t4_live(runs) -> list[str]:
    rows = ["| run | 파일 | prompts | 전부오답 | 전부정답 | 유신호(혼합) |", "|---|---|---|---|---|---|"]
    for name, run in runs:
        for fname in ("rollouts_fresh_train.jsonl", "rollouts_behavior_train.jsonl"):
            p = run / fname
            if not p.exists():
                continue
            agg: dict[int, list[float]] = {}
            for line in p.open():
                r = json.loads(line)
                agg.setdefault(r["prompt_idx"], []).append(r["reward"])
            n = len(agg)
            if not n:
                continue
            all0 = sum(1 for v in agg.values() if max(v) < 0.5)
            all1 = sum(1 for v in agg.values() if min(v) > 0.5)
            live = n - all0 - all1
            tag = "fresh(π)" if "fresh" in fname else "behavior(β)"
            rows.append(f"| {name} | {tag} | {n} | {all0} ({all0 / n:.0%}) | "
                        f"{all1} ({all1 / n:.0%}) | {live} ({live / n:.0%}) |")
    rows += ["", "전부정답 비율↑ = 포화(14B GSM8K형), 전부오답 비율↑ = 난이도 과잉. "
                 "선택 신호는 '유신호'에서만 산다 — A2 생존 창 서사의 정량 근거."]
    return rows


def t5_hybrid(runs) -> list[str]:
    rows = ["| run | cut | bb | bp | pb | pp | 회복(g10: pp−pb) | 회복(g01: pp−bp) |",
            "|---|---|---|---|---|---|---|---|"]
    any_row = False
    for name, run in runs:
        oracle = jload(run / "scores_oracle.json")
        if not oracle:
            continue
        for hf in sorted(run.glob("scores_hybrid_*.json")):
            cut = hf.stem.split("_")[-1]
            protocol = jload(run / f"hybrid_protocol_{cut}.json")
            if not protocol or protocol.get("schema") != HYBRID_PROTOCOL_SCHEMA:
                continue
            cells = jload(hf)
            if not cells or not {"bb", "bp", "pb", "pp"}.issubset(cells):
                continue
            sub = set(cells["bb"])
            o_sub = {i: oracle[i]["score"] for i in sub if i in oracle}
            k = topk_count(len(o_sub), 0.25)
            prec = {}
            for cell in ("bb", "bp", "pb", "pp"):
                c_scores = {i: v for i, v in cells[cell].items() if i in o_sub}
                prec[cell] = overlap_under_independent_ties(
                    o_sub, c_scores, k, seed=0
                ).mean
            rows.append(f"| {name} | {cut} | " +
                        " | ".join(f"{prec[c]:.2f}" for c in ("bb", "bp", "pb", "pp")) +
                        f" | {prec['pp'] - prec['pb']:+.2f} | {prec['pp'] - prec['bp']:+.2f} |")
            any_row = True
    if not any_row:
        rows.append("| (hybrid 산출물 없음) | | | | | | | |")
    rows += ["", "bb=β/β, bp=β경로+π마무리, pb=π경로+β마무리, pp=π/π. "
                 "회복량>0 = 그 축을 π로 되돌리면 순위가 회복 — one-sided 구조가 원인이라는 인과 증거."]
    return rows


def t6_c2_margin(runs) -> list[str]:
    rows = ["| run | certified | fresh(×uniform) | prec vs uniform | 유신호 | margin(live) | α_v |",
            "|---|---|---|---|---|---|---|"]
    for name, run in runs:
        rep = jload(run / "report.json") or {}
        cg = rep.get("certagrad", {})
        cert = cg.get("certified", "—")
        fresh = f"{cg['fresh_frac_of_uniform']:.2f}×" if "fresh_frac_of_uniform" in cg else "—"
        pv = (f"{cg['precision_vs_oracle']:.2f}/{cg.get('uniform_precision_vs_oracle', float('nan')):.2f}"
              if "precision_vs_oracle" in cg else "—")
        mg, vg = run / "oracle_micro_groups.pt", run / "val_groups.pt"
        live_s = margin_s = av_s = "—"
        if mg.exists() and vg.exists():
            micro = torch.load(mg, weights_only=True)
            val_pool = torch.load(vg, weights_only=True).float()[0::2]
            mu_v = val_pool.mean(0)
            per = 0.05 / (len(micro) + 1)
            av = math.degrees(angle_radius(mu_v, eb_radius(val_pool, per)))
            phis, norms = {}, {}
            for i, stack in micro.items():
                mu = stack.float().mean(0)
                c = float((mu @ mu_v) / (mu.norm() * mu_v.norm() + 1e-12))
                phis[i] = math.degrees(math.acos(max(-1.0, min(1.0, c))))
                norms[i] = float(mu.norm())
            live = [i for i in micro if norms[i] > 1e-6]
            k = topk_count(len(micro), FRAC)
            if len(live) > k:
                srt = sorted(live, key=lambda i: phis[i])
                margin_s = f"{phis[srt[k]] - phis[srt[k - 1]]:.2f}°"
            live_s = f"{len(live)}/{len(micro)}"
            av_s = f"{av:.2f}°"
        rows.append(f"| {name} | {cert} | {fresh} | {pv} | {live_s} | {margin_s} | {av_s} |")
    rows += ["", "margin(live) < α_v 이면 어떤 인증도 그 관측 수준에서 불성립 (§6 margin-collapse). "
                 "margin은 유신호 한정 k경계 간극."]
    return rows


def t7_downstream(runs) -> list[str]:
    rows = ["| run | base | oracle | g10 | g01 | random |", "|---|---|---|---|---|---|"]
    any_row = False
    for name, run in runs:
        ds = {f.stem.replace("downstream_", ""): jload(f)
              for f in run.glob("downstream_*.json")}
        ds = {k: v for k, v in ds.items() if v}
        if not ds:
            continue
        base = next(iter(ds.values())).get("base_acc", float("nan"))
        cell = {s: (f"{ds[s]['val_acc']:.3f}" if s in ds else "—")
                for s in ("oracle", "g10", "g01", "random")}
        rows.append(f"| {name} | {base:.3f} | {cell['oracle']} | {cell['g10']} | "
                    f"{cell['g01']} | {cell['random']} |")
        any_row = True
    if not any_row:
        rows.append("| (downstream 산출물 없음 — 14B 축소 게이트는 미실행 설계) | | | | | |")
    return rows


TABLES = [
    ("T1. 게이트 요약", t1_gate),
    ("T2. 정규화 재판정 (B3 — floor 대비 신호 보존율)", t2_relative),
    ("T3. floor-vs-관측 곡선 (B2)", t3_floor_curve),
    ("T4. live fraction (B4 — 신호 생존 창)", t4_live),
    ("T5. hybrid 인과 (cut별 축 교체 회복량)", t5_hybrid),
    ("T6. C2 인증·margin", t6_c2_margin),
    ("T7. downstream", t7_downstream),
]


def main() -> int:
    roots = sys.argv[1:] or ["outputs/pilot"]
    runs = collect_runs(roots)
    invalid = [name for name, run in runs
               if (run / "scores_offpolicy.json").exists()
               and not has_valid_analysis_protocol(run)]
    if invalid:
        print(f"[abort] corrected score protocol 없는 run: {invalid}", file=sys.stderr)
        return 2
    lines = [f"# 게이트 결과 테이블  ({time.strftime('%F %T')})", ""]
    lines.append("대상 run: " + ", ".join(f"`{n}`" for n, _ in runs) if runs
                 else "대상 run 없음")
    lines.append("")
    errors = []
    for title, fn in TABLES:
        lines.append(f"## {title}")
        try:
            lines += fn(runs)
        except Exception as e:  # 표 하나가 죽어도 나머지는 계속
            lines.append(f"⚠️ 생성 실패: {type(e).__name__}: {e}")
            errors.append((title, e))
        lines.append("")
    import os
    # 저장 위치: OM_RESULTS > $OM_WORK/results (group-volume) > ./results
    base = os.environ.get("OM_RESULTS") or (
        os.environ.get("OM_WORK", "") and os.environ["OM_WORK"] + "/results") or "results"
    out = Path(base)
    out.mkdir(parents=True, exist_ok=True)
    md = "\n".join(lines)
    (out / "TABLES.md").write_text(md)
    print(md)
    print(f"\n[저장됨] {out / 'TABLES.md'}")
    if errors:
        print(f"[abort] {len(errors)}개 표 생성 실패", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
