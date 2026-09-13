"""Compute-cost accounting from the artifacts and logs already on disk (CPU).

E5 seed directories (runs/e5-reduced/math500-d<drift>/s<seed>):
  training      sum of step_seconds in <arm>/policy/grpo_stats.jsonl, times the
                four allocated GPUs (GPU-seconds); pilot blocks of gate arms
  evaluation    per shard, done.json mtime minus contract.json mtime (one GPU)
  benchmarks    per shard, the recorded elapsed_seconds (one GPU)
  logging       reliability-logging overhead: mean step_seconds of the random
                arm under math500-d<drift>-rlog minus the benchmark's random arm

Matrix points (family-*/…-d<drift>/logs/main.log): stage wall times from the
runner's [progress] lines of the last attempt (behavior rollout, GRPO, fresh
rollout with validation, oracle and validation gradients, off-policy scores),
times the GPUs the point used. Per-prompt values are research-stage costs,
not marginal selector costs: fresh includes evaluation-only work, and the
off-policy stage computes four estimators jointly. They cannot populate a
gate rule without separately measuring and allocating the deployed work.

    python src/cost_accounting.py --work $OM_WORK [--matrix ROOT] [--out PREFIX]

Writes PREFIX.json and PREFIX.csv and prints the tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

PROGRESS = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] \[progress\] \S+\s+(\d+)/(\d+) (.*?)\s+\+(\d+)min\s*$")
STAGE_NAMES = {1: "prep", 2: "behavior_rollout", 3: "grpo", 4: "fresh_rollout", 5: "oracle_val_gradients",
               6: "offpolicy_scores", 7: "merge_report", 8: "done"}
GPUS_PER_NODE = 4


# ---------------------------------------------------------------- matrix
def parse_progress(text: str) -> list[dict]:
    events = []
    for line in text.splitlines():
        m = PROGRESS.match(line.strip())
        if m:
            events.append({"time": datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"), "stage": int(m.group(2)),
                           "total": int(m.group(3)), "label": m.group(4).strip(), "minutes": int(m.group(5))})
    return events


def stage_durations(events: list[dict]) -> dict:
    """Seconds per stage from the last attempt (the last 'prep' line onwards);
    a stage's duration is the gap to the next progress line."""
    if not events:
        return {"stages": {}, "attempts": 0, "wall_seconds_all_attempts": None, "complete": False}
    starts = [i for i, e in enumerate(events) if e["stage"] == 1]
    attempt = events[starts[-1]:] if starts else events
    stages = {}
    for current, following in zip(attempt, attempt[1:]):
        name = STAGE_NAMES.get(current["stage"], str(current["stage"]))
        stages[name] = stages.get(name, 0.0) + (following["time"] - current["time"]).total_seconds()
    complete = attempt[-1]["stage"] == attempt[-1]["total"]
    return {"stages": stages, "attempts": len(starts), "complete": complete,
            "wall_seconds_all_attempts": (events[-1]["time"] - events[0]["time"]).total_seconds(),
            "labels": {STAGE_NAMES.get(e["stage"], str(e["stage"])): e["label"] for e in attempt}}


def point_costs(run: Path) -> dict | None:
    log = run / "logs" / "main.log"
    if not log.is_file():
        return None
    config = json.loads((run / "run_config.json").read_text()) if (run / "run_config.json").is_file() else {}
    prompts = json.loads((run / "prompts.json").read_text()) if (run / "prompts.json").is_file() else {}
    n = len(prompts.get("train", [])) or None
    durations = stage_durations(parse_progress(log.read_text(encoding="utf-8", errors="replace")))
    gpus = int(config.get("grpo_world_size") or GPUS_PER_NODE)
    stages = durations["stages"]
    per_prompt = {}
    if n:
        def per_candidate(*names):
            if not all(name in stages for name in names):
                return None
            return sum(stages[name] for name in names) * gpus / n

        per_prompt = {"behavior_cache_gpu_seconds": per_candidate("behavior_rollout"),
                      "fresh_scoring_gpu_seconds": per_candidate("fresh_rollout", "oracle_val_gradients"),
                      "reuse_scoring_gpu_seconds": per_candidate("offpolicy_scores")}
    return {"run": str(run), "dataset": config.get("dataset"), "seed": config.get("seed"), "drift": config.get("drift"),
            "candidates": n, "gpus": gpus, "complete": durations["complete"], "attempts": durations["attempts"],
            "stage_seconds": stages, "wall_seconds_all_attempts": durations["wall_seconds_all_attempts"],
            "per_prompt": per_prompt,
            "cost_scope": "research stages; fresh includes reference/evaluation work; reuse covers all four estimators"}


def matrix_costs(root: Path) -> list[dict]:
    rows = []
    for run in sorted(root.glob("family-*/*-d*")):
        if not (run / "DONE").is_file():
            continue
        record = point_costs(run)
        if record:
            rows.append(record)
    return rows


# -------------------------------------------------------------------- E5
def step_seconds(stats: Path) -> list[float]:
    if not stats.is_file():
        return []
    values = []
    for line in stats.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if "step_seconds" in row:
                values.append(float(row["step_seconds"]))
    return values


def evaluation_seconds(arm_dir: Path) -> tuple[float | None, int]:
    """Sum over shards of done.json mtime minus contract.json mtime (one GPU each)."""
    total, shards = 0.0, 0
    for shard in range(4):
        done = arm_dir / "evaluation" / f"shard-{shard}.done.json"
        contract = arm_dir / "evaluation" / f"shard-{shard}.contract.json"
        if done.is_file() and contract.is_file():
            total += max(0.0, done.stat().st_mtime - contract.stat().st_mtime)
            shards += 1
    return (total if shards else None), shards


def benchmark_seconds(arm_dir: Path) -> float | None:
    total, found = 0.0, False
    for done in (arm_dir / "benchmark").glob("*/shard-*.done.json") if (arm_dir / "benchmark").is_dir() else []:
        record = json.loads(done.read_text())
        value = record.get("elapsed_seconds")
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            return None
        total += value
        found = True
    return total if found else None


def seed_costs(seed_dir: Path) -> list[dict]:
    contract = json.loads((seed_dir / "experiment.json").read_text())
    arms = ["before"] + list(contract["selectors"])
    extra = seed_dir / "arms.json"
    if extra.is_file():
        arms += [a for a in json.loads(extra.read_text())["selectors"] if a not in arms]
    rows = []
    for arm in arms:
        arm_dir = seed_dir / arm
        train = step_seconds(arm_dir / "policy" / "grpo_stats.jsonl")
        pilot = step_seconds(arm_dir / "pilot" / "grpo_stats.jsonl")
        eval_s, shards = evaluation_seconds(arm_dir)
        rows.append({"branch": seed_dir.parent.name, "seed": contract["seed"], "arm": arm,
                     "train_steps": len(train), "train_gpu_seconds": sum(train) * GPUS_PER_NODE if train else None,
                     "step_seconds_mean": statistics.fmean(train) if train else None,
                     "pilot_steps": len(pilot), "pilot_gpu_seconds": sum(pilot) * GPUS_PER_NODE if pilot else None,
                     "eval_gpu_seconds": eval_s, "eval_shards": shards,
                     "benchmark_gpu_seconds": benchmark_seconds(arm_dir)})
    return rows


def logging_overhead(work: Path) -> list[dict]:
    """Reliability-logging overhead per seed: rlog random arm versus the benchmark random arm."""
    rows = []
    for rlog in sorted(work.glob("runs/e5-reduced/math500-d*-rlog*")):
        base = rlog.name.split("-rlog")[0]
        for seed_dir in sorted(rlog.glob("s*")):
            logged = step_seconds(seed_dir / "random" / "policy" / "grpo_stats.jsonl")
            plain = step_seconds(work / "runs" / "e5-reduced" / base / seed_dir.name / "random" / "policy" / "grpo_stats.jsonl")
            if logged and plain:
                rows.append({"branch": base, "logging_root": rlog.name, "seed": seed_dir.name[1:],
                             "logged_step_seconds": statistics.fmean(logged), "plain_step_seconds": statistics.fmean(plain),
                             "overhead_seconds_per_step": statistics.fmean(logged) - statistics.fmean(plain),
                             "overhead_fraction": statistics.fmean(logged) / statistics.fmean(plain) - 1.0})
    return rows


def e5_costs(work: Path) -> list[dict]:
    rows = []
    for branch in sorted(work.glob("runs/e5-reduced/math500-d*")):
        if "-rlog" in branch.name:
            continue
        for seed_dir in sorted(branch.glob("s*")):
            if (seed_dir / "experiment.json").is_file():
                rows.extend(seed_costs(seed_dir))
    return rows


def suggested_rule_costs(points: list[dict]) -> dict:
    """Stage logs do not identify marginal costs of the deployed selectors."""
    return {}


def render(report: dict) -> str:
    f = lambda v, w=9: ("-" if v is None else f"{v:,.0f}").rjust(w)  # noqa: E731
    lines = ["E5 arms (GPU-seconds: training = step seconds x 4 GPUs; evaluation and benchmarks = one GPU per shard)",
             "  branch        seed arm             steps  train_gpu_s  step_s  pilot_gpu_s   eval_gpu_s  bench_gpu_s"]
    for r in report["e5"]:
        lines.append(f"  {r['branch']:13s} {r['seed']:4d} {r['arm']:15s} {r['train_steps']:5d} {f(r['train_gpu_seconds'], 12)} "
                     f"{('-' if r['step_seconds_mean'] is None else f'{r['step_seconds_mean']:.1f}').rjust(7)} {f(r['pilot_gpu_seconds'], 12)} "
                     f"{f(r['eval_gpu_seconds'], 12)} {f(r['benchmark_gpu_seconds'], 12)}")
    lines.append("reliability-logging overhead (random arm, rlog root vs benchmark root)")
    for r in report["logging_overhead"]:
        lines.append(f"  {r['branch']:13s} seed {r['seed']}: logged {r['logged_step_seconds']:.1f}s plain {r['plain_step_seconds']:.1f}s "
                     f"overhead {r['overhead_seconds_per_step']:+.1f}s per step ({100 * r['overhead_fraction']:+.1f}%)")
    if not report["logging_overhead"]:
        lines.append("  (no seed has both a logged and a plain random arm yet)")
    lines.append("matrix points (stage wall seconds of the last attempt; per-prompt GPU-seconds = stage x GPUs / candidates)")
    lines.append("  dataset  seed drift  behavior     grpo    fresh  oracle+val  offpolicy | per prompt: cache   fresh   reuse")
    for p in report["matrix"]:
        s, pp = p["stage_seconds"], p["per_prompt"]
        lines.append(f"  {str(p['dataset']):8s} {str(p['seed']):>4s} {str(p['drift']):>5s} {f(s.get('behavior_rollout'))} {f(s.get('grpo'))} "
                     f"{f(s.get('fresh_rollout'))} {f(s.get('oracle_val_gradients'), 11)} {f(s.get('offpolicy_scores'), 10)} | "
                     f"{f(pp.get('behavior_cache_gpu_seconds'), 8)} {f(pp.get('fresh_scoring_gpu_seconds'), 7)} {f(pp.get('reuse_scoring_gpu_seconds'), 7)}"
                     + ("" if p["complete"] else "  [last attempt incomplete]"))
    if not report["matrix"]:
        lines.append("  (no point with logs/main.log found)")
    lines.append("suggested cost_per_prompt_seconds for config/gate_rule.json: "
                 + json.dumps(report["suggested_rule_costs"]))
    lines.append("  no automatic rule costs: joint research stages are not marginal selector measurements")
    return "\n".join(lines)


def build(work: Path, matrix: Path | None) -> dict:
    points = matrix_costs(matrix) if matrix and matrix.is_dir() else []
    return {"schema": "offpolicy-cost-accounting/v1", "work": str(work), "matrix": points, "e5": e5_costs(work),
            "logging_overhead": logging_overhead(work), "suggested_rule_costs": suggested_rule_costs(points),
            "units": "GPU-seconds on the allocated hardware; evaluation shards use one GPU each; training uses four"}


def write_outputs(report: dict, prefix: Path) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    rows = report["e5"]
    with prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["arm"])
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work", type=Path, required=True, help="$OM_WORK")
    parser.add_argument("--matrix", type=Path, help="matrix root with family-*/ point directories")
    parser.add_argument("--out", type=Path, help="output prefix (default: <work>/exports/cost-accounting)")
    args = parser.parse_args(argv)
    try:
        report = build(args.work.resolve(), args.matrix.resolve() if args.matrix else None)
        prefix = args.out or (args.work.resolve() / "exports" / "cost-accounting")
        write_outputs(report, prefix)
        print(render(report))
        print(f"[cost] written: {prefix.with_suffix('.json')} and .csv")
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"[abort] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
