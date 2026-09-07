"""Progress is what lands on disk; liveness is not progress."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import training_progress as tp  # noqa: E402


def _point(root: Path, family: str, name: str, *, done: bool = False, steps: int = 0, rollout_bytes: int = 0, age: float = 0.0) -> Path:
    run = root / family / name
    run.mkdir(parents=True, exist_ok=True)
    (run / "run_config.json").write_text("{}")
    (run / "logs").mkdir(exist_ok=True)
    (run / "logs/main.log").write_text("noise\n")           # logs never count
    (run / "keepalive.log").write_text("noise\n")           # keepalive never counts
    if steps:
        stats = run / "policy_step_25"
        stats.mkdir(exist_ok=True)
        (stats / "grpo_stats.jsonl").write_text("".join(json.dumps({"step": i}) + "\n" for i in range(1, steps + 1)))
    if rollout_bytes:
        (run / "rollouts_fresh_train.shard0.partial").write_bytes(b"x" * rollout_bytes)
    if done:
        (run / "DONE").write_text("done\n")
    if age:
        stamp = time.time() - age
        for path in run.rglob("*"):
            os.utime(path, (stamp, stamp))
        os.utime(run, (stamp, stamp))
    return run


def test_not_started_then_training_then_done(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    word, line, _ = tp.verdict(root, total_points=4, stall_seconds=1800)
    assert word == "NOT STARTED" and line.startswith("NOT STARTED")

    _point(root, "family-math500-s0", "tag-s0-math500-d0", steps=3, rollout_bytes=1000)
    word, line, sig = tp.verdict(root, total_points=4, stall_seconds=1800, record_probe=True)
    assert word == "TRAINING", line
    assert sig.grpo_steps == 3 and sig.rollout_bytes == 1000 and sig.points_done == 0
    assert "points 0/4" in line and "grpo 3 steps" in line and "last write" in line

    for name in ("d0", "d25", "d100", "d400"):
        _point(root, "family-math500-s0", f"tag-s0-math500-{name}", done=True)
    word, line, _ = tp.verdict(root, total_points=4, stall_seconds=1800)
    assert word == "DONE" and "points 4/4" in line


def test_old_artifacts_with_no_change_are_not_training(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _point(root, "family-mbpp-s1", "tag-s1-mbpp-d25", steps=25, rollout_bytes=5000, age=3 * 3600)
    # a fresh log line does not rescue it: logs are excluded from "durable"
    (root / "family-mbpp-s1/tag-s1-mbpp-d25/logs/main.log").write_text("still alive\n")
    word, line, _ = tp.verdict(root, total_points=40, stall_seconds=1800, record_probe=True)
    assert word == "NOT TRAINING", line
    assert line.startswith("NOT TRAINING for 3h")
    assert "grpo 25 steps" in line and "last write 3h" in line


def test_history_gives_deltas_between_probes(tmp_path: Path) -> None:
    root = tmp_path / "root"
    run = _point(root, "family-math500-s2", "tag-s2-math500-d25", steps=10, rollout_bytes=1000)
    now = time.time()
    # a probe 40 minutes ago saw 10 steps; since then 25 more landed on disk
    tp.record(root, tp.probe(root, 40), now - 2400)
    (run / "policy_step_25/grpo_stats.jsonl").write_text("".join(json.dumps({"step": i}) + "\n" for i in range(1, 36)))
    word, line, _ = tp.verdict(root, total_points=40, stall_seconds=1800, now=now)
    assert word == "TRAINING"
    assert "+25 grpo steps" in line


def test_unchanged_signature_across_probes_is_a_stall(tmp_path: Path) -> None:
    root = tmp_path / "root"
    run = _point(root, "family-math500-s2", "tag-s2-math500-d25", steps=35, rollout_bytes=1000)
    now = time.time()
    old = now - 3000
    for path in run.rglob("*"):
        os.utime(path, (old, old))
    # two earlier probes saw exactly this signature; nothing has been written since
    tp.record(root, tp.probe(root, 40), now - 3000)
    tp.record(root, tp.probe(root, 40), now - 1900)
    word, line, _ = tp.verdict(root, total_points=40, stall_seconds=1800, now=now)
    assert word == "NOT TRAINING", line
    assert line.startswith("NOT TRAINING for 50m")
    assert "+0 grpo steps" in line and "+0 points" in line


def test_cli_exit_status_and_watch_tag(tmp_path: Path, capsys) -> None:
    root = tmp_path / "root"
    root.mkdir()
    assert tp.main(["--root", str(root), "--total-points", "40"]) == 1
    assert capsys.readouterr().out.startswith("NOT STARTED")
    _point(root, "f", "p", steps=1)
    assert tp.main(["--root", str(root), "--total-points", "40", "--record"]) == 0
    assert capsys.readouterr().out.startswith("TRAINING")
    assert tp.history_path(root).is_file()
    assert tp.main(["--root", str(root), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["verdict"] == "TRAINING" and payload["grpo_steps"] == 1
