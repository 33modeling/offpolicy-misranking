"""사람이 읽는 판독 보고서 생성 — READOUT.md의 본문.

    python3 src/readout_summary.py <runs_root>

구성: ① 한눈 요약 표(run × 수치 × 평문 판정) ② 자동 결론 ③ 용어 설명
④ 상세(원시 judge 출력). PASS/FAIL 이중부정 없이 전부 평문으로 쓴다.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from gate_rules import evaluate_causal_run, has_valid_analysis_protocol
from run_select import is_generation_run, iter_runs
from score_artifacts import (
    ESTIMATORS,
    ScoreArtifactError,
    load_complete_score_artifacts,
)
from select_rules import overlap_under_independent_ties, topk_count


def precisions(run: Path) -> tuple[dict[str, float], int, float]:
    if not has_valid_analysis_protocol(run):
        raise ScoreArtifactError("corrected score/oracle protocol is missing")
    artifacts = load_complete_score_artifacts(run)
    oracle = artifacts.oracle
    n = len(oracle)
    k = topk_count(n, 0.10)
    out: dict[str, float] = {}
    for est in ESTIMATORS:
        sc = artifacts.offpolicy[est]
        out[est] = overlap_under_independent_ties(oracle, sc, k, seed=0).mean
    return out, k, k / n


def _completed_runs(root: Path) -> list[Path]:
    if not root.is_dir():
        raise ScoreArtifactError(f"runs root does not exist: {root}")
    return iter_runs(root, include_legacy=True)


def _dataset_tag(name: str) -> str:
    generation = name.split("-", 1)[0] if is_generation_run(name) else "legacy"
    model_match = re.match(r"^v\d+-(.+)-s\d+(?:-|$)", name)
    model = model_match.group(1) if model_match else None
    if "dapo" in name:
        dataset = "dapo"
    elif "math500" in name:
        dataset = "math500"
    elif "gsm8k" in name or is_generation_run(name):
        dataset = "gsm8k"
    else:
        dataset = "other"
    parts = [generation]
    if model:
        parts.append(model)
    parts.append(dataset)
    return "/".join(parts)


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: readout_summary.py RUNS_ROOT", file=sys.stderr)
        return 2
    root = Path(sys.argv[1])
    try:
        runs = _completed_runs(root)
    except ScoreArtifactError as exc:
        print(f"[abort] {exc}", file=sys.stderr)
        return 2

    rows, details, concl = [], [], []
    skipped: list[str] = []
    errors: list[str] = []
    for d in runs:
        if not has_valid_analysis_protocol(d):
            skipped.append(f"{d.name}: corrected score/oracle protocol 없음")
            continue
        try:
            prec, _, chance = precisions(d)
            state = evaluate_causal_run(d)
            rep = state["report"] or {}
            if not rep.get("_recomputed") or not isinstance(
                rep.get("noise_floor"), (int, float)
            ):
                raise ScoreArtifactError(
                    "raw split-half 기반 canonical report 재계산 실패"
                )
            hybrid_errors = [
                result["error"]
                for result in state["hybrid_results"]
                if "error" in result
            ]
            if hybrid_errors:
                raise ScoreArtifactError(
                    "hybrid artifact 오류: " + "; ".join(hybrid_errors)
                )
            floor = float(rep["noise_floor"])

            # judge와 같은 사전 문턱, 같은 run의 joint predicate를 그대로 사용한다.
            if state["axis_failures"] is not None:
                onesided = "예 (사전 문턱)" if state["joint_failure"] else "아니오"
            else:
                onesided = "판정 불가"

            valid_hybrid = state["hybrid_results"]
            eligible_hybrid = [r for r in valid_hybrid if r["eligible"]]
            if valid_hybrid:
                dip_votes = [
                    max(r["precision"]["bp"], r["precision"]["pb"])
                    < r["precision"]["bb"]
                    for r in valid_hybrid
                ]
                if not state["joint_failure"]:
                    hyb = "C1 미충족"
                elif state["witnesses"]:
                    hyb = f"예 (cut={state['causal_cut']})"
                elif not eligible_hybrid:
                    hyb = "사전고정 cut 없음"
                else:
                    hyb = "아니오"
                dip = "예" if all(dip_votes) else ("일부" if any(dip_votes) else "아니오")
            else:
                hyb, dip = "데이터 없음", "-"

            judge = subprocess.run(
                [sys.executable, str(Path(__file__).with_name("judge.py")), str(d)],
                capture_output=True,
                text=True,
                timeout=600,
                check=False,
            )
            if judge.returncode != 0 or not judge.stdout.strip():
                reason = judge.stderr.strip() or "judge stdout 없음"
                raise ScoreArtifactError(
                    f"judge 실패(exit={judge.returncode}): {reason}"
                )
            jd = judge.stdout

            rows.append(
                f"| {d.name} | {floor:.3f}† | {chance:.2f} | "
                + " | ".join(f"{prec[e]:.3f}" for e in ESTIMATORS)
                + f" | {onesided} | {hyb} | {dip} |"
            )
            details.append(
                f"<details><summary>{d.name} 원시 출력</summary>\n\n"
                f"```\n{jd.strip()}\n```\n</details>\n"
            )
            concl.append((_dataset_tag(d.name), onesided.startswith("예"), hyb))
        except (ScoreArtifactError, subprocess.TimeoutExpired, ValueError) as exc:
            errors.append(f"{d.name}: {type(exc).__name__}: {exc}")

    print("# 판독 보고서\n")
    print("## 한눈 요약\n")
    print("| run | floor | chance | g00 | g10 | g01 | g11 | one-sided가 더 나쁜가 | hybrid 회복 | mixed-dip |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(r)

    print("\n## 자동 결론\n")
    for tag in sorted({item[0] for item in concl}):
        sub = [c for c in concl if c[0] == tag]
        yes = sum(1 for c in sub if c[1])
        print(f"- **{tag}**: one-sided 열세 {yes}/{len(sub)} run에서 관찰. "
              f"hybrid 회복: {', '.join(c[2] for c in sub)}")
    print("\n(주의: run 수가 적으면 위 관찰은 통계적 확정이 아님 — 5-seed 전승이 유의선)")

    print("\n## 용어 — 표를 읽는 법\n")
    print("- **floor**: oracle 절반끼리의 일치도. †는 원시 점수에서 독립 tie stream으로 재계산한 정본")
    print("- **chance**: 아무거나 찍었을 때의 기대 precision")
    print("- **g00/g10/g01/g11**: 무보정 / prefix만 / suffix만 / 전부 보정의 top-k precision")
    print("- **one-sided가 더 나쁜가**: 동일 run에서 g10·g01 모두 floor보다 0.15 이상 낮으면 '예'")
    print("- **hybrid 회복**: C1을 만족한 동일 run의 사전고정 cut=0.5에서 pp가 pb·bp보다 모두 높으면 '예'")
    print("- **mixed-dip**: 혼합 셀(bp·pb)이 순수 stale(bb)보다 낮으면 '예'")

    if skipped:
        print("\n## 제외된 historical run\n")
        for reason in skipped:
            print(f"- {reason}")

    if errors:
        print("\n## 산출물 오류\n")
        for error in errors:
            print(f"- {error}")

    print("\n## 상세 (원시 출력)\n")
    for dt in details:
        print(dt)
    if not rows:
        print("[abort] corrected protocol을 만족하는 run이 없음", file=sys.stderr)
        return 2
    if errors:
        print(f"[abort] corrected run 산출물 오류 {len(errors)}개", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
