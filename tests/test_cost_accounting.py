"""CPU contracts for the cost accounting over E5 artifacts and matrix logs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import cost_accounting as ca

ROOT = Path(__file__).resolve().parents[1]

MAIN_LOG = """[2026-09-08 10:00:00] retry=1 GPU order=0 1 2 3
[2026-09-08 10:00:00] [progress] point  1/8 prep  +0min
[2026-09-08 10:01:00] [progress] point  2/8 behavior-rollout 400x8 on 4 GPUs  +1min
[2026-09-08 10:31:00] [progress] point  3/8 grpo steps 0->400 (see log)  +31min
[2026-09-08 11:00:00] [stage-fail] pid=1 rc=1
[2026-09-08 11:00:00] [progress] point  1/8 prep  +0min
[2026-09-08 11:01:00] [progress] point  2/8 behavior-rollout reused  +1min
[2026-09-08 11:02:00] [progress] point  3/8 grpo steps 0->400 (see log)  +2min
[2026-09-08 13:02:00] [progress] point  4/8 fresh-rollout 400x32 + val 100x8 on 4 GPUs (longest stage)  +122min
[2026-09-08 17:02:00] [progress] point  5/8 oracle+val gradients  +362min
[2026-09-08 18:02:00] [progress] point  6/8 off-policy scores (4 estimators)  +422min
[2026-09-08 19:02:00] [progress] point  7/8 merge + report  +482min
[2026-09-08 19:03:00] [progress] point  8/8 DONE  +483min
"""


def test_stage_durations_use_the_last_attempt():
    events = ca.parse_progress(MAIN_LOG)
    assert len(events) == 11
    d = ca.stage_durations(events)
    assert d["attempts"] == 2 and d["complete"]
    assert d["stages"]["fresh_rollout"] == 4 * 3600 and d["stages"]["offpolicy_scores"] == 3600
    assert d["stages"]["behavior_rollout"] == 60  # the reused cache in the last attempt
    assert d["wall_seconds_all_attempts"] == 9 * 3600 + 3 * 60
    assert ca.stage_durations([]) == {"stages": {}, "attempts": 0, "wall_seconds_all_attempts": None, "complete": False}


def test_point_costs_per_prompt(tmp_path):
    run = tmp_path / "family-math500-s0" / "point-s0-math500-d400"
    (run / "logs").mkdir(parents=True)
    (run / "logs" / "main.log").write_text(MAIN_LOG)
    (run / "run_config.json").write_text(json.dumps({"dataset": "math500", "seed": 0, "drift": 400, "grpo_world_size": 4}))
    (run / "prompts.json").write_text(json.dumps({"train": [{"question": str(i), "answer": "1"} for i in range(400)], "val": []}))
    (run / "DONE").write_text("ok")
    record = ca.point_costs(run)
    assert record["candidates"] == 400 and record["gpus"] == 4
    assert record["per_prompt"]["fresh_scoring_gpu_seconds"] == pytest.approx((4 * 3600 + 3600) * 4 / 400)
    assert record["per_prompt"]["reuse_scoring_gpu_seconds"] == pytest.approx(3600 * 4 / 400)
    rows = ca.matrix_costs(tmp_path)
    assert len(rows) == 1
    assert ca.suggested_rule_costs(rows) == {}  # joint research work is not a selector's marginal cost


def test_missing_stages_are_unknown_not_zero_cost(tmp_path):
    run = tmp_path / "point"
    (run / "logs").mkdir(parents=True)
    (run / "logs/main.log").write_text(MAIN_LOG.split("[2026-09-08 17:02:00]")[0])
    (run / "prompts.json").write_text(json.dumps({"train": [{}] * 400}))
    row = ca.point_costs(run)
    assert not row["complete"]
    assert row["per_prompt"]["fresh_scoring_gpu_seconds"] is None
    assert row["per_prompt"]["reuse_scoring_gpu_seconds"] is None
    assert ca.suggested_rule_costs([row]) == {}


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -1])
def test_unknown_benchmark_timer_is_not_free(tmp_path, value):
    bench = tmp_path / "benchmark/aime24"
    bench.mkdir(parents=True)
    (bench / "shard-0.done.json").write_text(json.dumps({"elapsed_seconds": value}))
    assert ca.benchmark_seconds(tmp_path) is None


def _stats(path: Path, seconds: float, steps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"step": s + 1, "step_seconds": seconds}) + "\n" for s in range(steps)))


def test_e5_costs_and_logging_overhead(tmp_path):
    work = tmp_path / "work"
    seed = work / "runs" / "e5-reduced" / "math500-d400" / "s0"
    seed.mkdir(parents=True)
    (seed / "experiment.json").write_text(json.dumps({"seed": 0, "selectors": ["random", "passrate_beta"]}))
    (seed / "arms.json").write_text(json.dumps({"selectors": ["gate_passrate"]}))
    _stats(seed / "random" / "policy" / "grpo_stats.jsonl", 70.0, 100)
    _stats(seed / "gate_passrate" / "policy" / "grpo_stats.jsonl", 70.0, 90)
    _stats(seed / "gate_passrate" / "pilot" / "grpo_stats.jsonl", 75.0, 10)
    ev = seed / "random" / "evaluation"
    ev.mkdir(parents=True)
    for shard in range(4):
        (ev / f"shard-{shard}.contract.json").write_text("{}")
        (ev / f"shard-{shard}.done.json").write_text("{}")
        os.utime(ev / f"shard-{shard}.contract.json", (1_000_000, 1_000_000))
        os.utime(ev / f"shard-{shard}.done.json", (1_000_000, 1_000_000 + 600))
    bench = seed / "random" / "benchmark" / "aime24"
    bench.mkdir(parents=True)
    (bench / "shard-0.done.json").write_text(json.dumps({"elapsed_seconds": 120.0}))
    _stats(work / "runs" / "e5-reduced" / "math500-d400-rlog" / "s0" / "random" / "policy" / "grpo_stats.jsonl", 73.5, 100)
    report = ca.build(work, None)
    rows = {r["arm"]: r for r in report["e5"]}
    assert set(rows) == {"before", "random", "passrate_beta", "gate_passrate"}
    assert rows["random"]["train_gpu_seconds"] == pytest.approx(70.0 * 100 * 4) and rows["random"]["eval_gpu_seconds"] == pytest.approx(2400.0)
    assert rows["random"]["benchmark_gpu_seconds"] == 120.0 and rows["passrate_beta"]["train_gpu_seconds"] is None
    assert rows["gate_passrate"]["pilot_gpu_seconds"] == pytest.approx(75.0 * 10 * 4) and rows["gate_passrate"]["train_steps"] == 90
    overhead = report["logging_overhead"][0]
    assert overhead["overhead_seconds_per_step"] == pytest.approx(3.5) and overhead["overhead_fraction"] == pytest.approx(0.05)
    ca.write_outputs(report, tmp_path / "out" / "cost")
    assert (tmp_path / "out" / "cost.csv").is_file()
    text = ca.render(report)
    assert "gate_passrate" in text and "overhead +3.5s per step" in text


def test_cli_and_script_syntax(tmp_path):
    work = tmp_path / "work"
    (work / "runs" / "e5-reduced").mkdir(parents=True)
    env = {"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"}
    result = subprocess.run([sys.executable, str(ROOT / "src/cost_accounting.py"), "--work", str(work)], capture_output=True, text=True, env=env)
    assert result.returncode == 0 and "suggested cost_per_prompt_seconds" in result.stdout, result.stdout + result.stderr
    assert (work / "exports" / "cost-accounting.json").is_file()
    subprocess.run(["bash", "-n", str(ROOT / "scripts/run_cost_accounting.sh")], check=True)
