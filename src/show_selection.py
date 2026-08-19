"""선택 결과 열람 — 방법별 top-k 프롬프트(문제 미리보기)와 방법 간 겹침.

    python3 src/show_selection.py <OUT_ROOT 또는 run 디렉토리> [--topk-frac 0.10]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from gate_rules import has_valid_analysis_protocol
from select_rules import jittered_topk, topk_count


def topk(scores: dict[int, float], frac: float) -> list[int]:
    k = topk_count(len(scores), frac)
    return sorted(jittered_topk(scores, k, seed=0))


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/pilot")
    frac = 0.10
    if "--topk-frac" in sys.argv:
        frac = float(sys.argv[sys.argv.index("--topk-frac") + 1])
    runs = sorted(d for d in root.glob("drift*")
                  if d.is_dir() and not d.name.startswith("drift_")) or [root]

    invalid = False
    for run in runs:
        pj = run / "prompts.json"
        if not pj.exists():
            continue
        if not has_valid_analysis_protocol(run):
            print(f"[{run.name}] corrected score/oracle protocol 없음 — 열람 거부")
            invalid = True
            continue
        prompts = json.loads(pj.read_text())["train"]
        def question_preview(idx: int, rows: list[dict] = prompts) -> str:
            return rows[idx]["question"].replace("\n", " ")[:70]

        sel: dict[str, list[int]] = {}
        op = run / "scores_oracle.json"
        if op.exists():
            oracle = {int(i): v["score"] for i, v in json.loads(op.read_text()).items()}
            sel["oracle"] = topk(oracle, frac)
        fp = run / "scores_offpolicy.json"
        if fp.exists():
            off = json.loads(fp.read_text())
            for est, sc in off.items():
                sel[est] = topk({int(i): v["score"] for i, v in sc.items()}, frac)
        for f in run.glob("downstream_*.json"):
            d = json.loads(f.read_text())
            sel[f"DS:{d['source']}"] = sorted(d["selected"])

        if not sel:
            print(f"[{run.name}] 선택 산출물 없음")
            continue

        print(f"\n===== {run.name} — 방법별 top-{int(frac*100)}% 선택 =====")
        # β rollout 정답률(난이도 감각)을 같이 보여준다
        acc = {}
        bp = run / "rollouts_behavior_train.jsonl"
        if bp.exists():
            agg: dict[int, list] = {}
            for line in bp.open():
                r = json.loads(line)
                agg.setdefault(r["prompt_idx"], []).append(r["reward"])
            acc = {i: sum(v) / len(v) for i, v in agg.items()}

        if "oracle" in sel:
            print(f"\n[oracle 선택 {len(sel['oracle'])}개] (idx · β정답률 · 문제)")
            for i in sel["oracle"]:
                print(f"  {i:4d} · {acc.get(i, float('nan')):.2f} · {question_preview(i)}")
        for name in ("g00", "g10", "g01", "g11"):
            if name not in sel or "oracle" not in sel:
                continue
            inter = set(sel[name]) & set(sel["oracle"])
            only = [i for i in sel[name] if i not in inter][:5]
            print(f"\n[{name}] oracle과 겹침 {len(inter)}/{len(sel[name])}"
                  + (f" — {name}만 뽑은 예: " + ", ".join(str(i) for i in only) if only else ""))
            for i in only[:3]:
                print(f"    {i:4d} · β정답률 {acc.get(i, float('nan')):.2f} · {question_preview(i)}")

        names = list(sel)
        print("\n[겹침 행렬] (교집합 크기)")
        print("        " + " ".join(f"{n[:7]:>7}" for n in names))
        for a in names:
            row = " ".join(f"{len(set(sel[a]) & set(sel[b])):7d}" for b in names)
            print(f"{a[:7]:>7} {row}")
    return 2 if invalid else 0


if __name__ == "__main__":
    sys.exit(main())
