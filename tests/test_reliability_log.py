"""CPU contracts for the reliability log: chunking, per-row math, and the trajectory summary."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import reliability_trajectory as rt
from train_policy_grpo import _chunks, _half_chunks, reliability_row

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("size,micro", [(8, 4), (8, 1), (8, 8), (8, 3), (7, 2), (2, 1)])
def test_half_chunks_cover_the_group_without_crossing_the_half(size, micro):
    first, second = _half_chunks(size, micro)
    covered = [i for c in first + second for i in c]
    assert covered == list(range(size))
    half = size // 2
    assert all(c.stop <= half for c in first) and all(c.start >= half for c in second)
    assert len(first + second) >= len(_chunks(size, micro))


def test_half_chunks_need_two_responses():
    with pytest.raises(ValueError):
        _half_chunks(1, 1)


def test_reliability_row_recovers_second_half_and_batch_geometry():
    grad_a = torch.tensor([1.0, 0.0, 0.0])
    grad_b = torch.tensor([0.0, 1.0, 0.0])
    total = grad_a + grad_b
    others = torch.tensor([1.0, 1.0, 0.0])
    world = 4
    mean_total = (total + (world - 1) * others) / world
    rewards = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    row = reliability_row(grad_a, total, mean_total, world, rewards, step=3, rank=1, prompt_index=7)
    assert row["step"] == 3 and row["rank"] == 1 and row["prompt_index"] == 7
    assert row["pass_a"] == 0.75 and row["pass_b"] == 0.25 and row["pass"] == 0.5 and row["mixed"]
    assert row["cos_ab"] == pytest.approx(0.0)
    assert row["norm_b"] == pytest.approx(1.0)
    assert row["cos_a_others"] == pytest.approx(1 / math.sqrt(2))
    assert row["cos_b_others"] == pytest.approx(1 / math.sqrt(2))
    assert row["cos_total_others"] == pytest.approx(1.0)


def test_reliability_row_single_rank_has_no_batch_direction():
    grad_a = torch.ones(4)
    row = reliability_row(grad_a, 2 * grad_a, 2 * grad_a, 1, torch.zeros(8), 1, 0, 0)
    assert row["cos_a_others"] is None and row["cos_total_others"] is None
    assert not row["mixed"] and row["cos_ab"] == pytest.approx(1.0)


def _synthetic_rows(tmp_path: Path, rho: float, steps: int = 60, ranks: int = 4, seed: int = 0) -> Path:
    """Half measurements y = theta + noise with a known split-half correlation rho."""
    rng = np.random.default_rng(seed)
    policy = tmp_path / "policy"
    policy.mkdir()
    noise_sd = math.sqrt((1 - rho) / rho) if rho > 0 else 10.0
    streams = {r: (policy / f"reliability_log.rank{r}.jsonl").open("w") for r in range(ranks)}
    for step in range(1, steps + 1):
        for rank in range(ranks):
            theta = rng.normal()
            a, b = theta + noise_sd * rng.normal(), theta + noise_sd * rng.normal()
            p = float(np.clip(0.5 + 0.2 * theta, 0.05, 0.95))
            pa, pb = rng.binomial(4, p) / 4, rng.binomial(4, p) / 4
            row = {"step": step, "rank": rank, "prompt_index": step * ranks + rank,
                   "pass_a": pa, "pass_b": pb, "pass": (pa + pb) / 2, "mixed": pa != pb or 0 < pa < 1,
                   "cos_ab": 0.1, "norm_a": 1.0, "norm_b": 1.0, "norm_total": 2.0,
                   "cos_a_others": a, "cos_b_others": b, "cos_total_others": (a + b) / 2}
            streams[rank].write(json.dumps(row) + "\n")
    for stream in streams.values():
        stream.close()
    return policy


def test_trajectory_recovers_a_known_split_half_reliability(tmp_path):
    policy = _synthetic_rows(tmp_path, rho=0.6, steps=200)
    rows = rt.load_rows(policy)
    records = rt.trajectory(rows, window=200, reps=200)
    assert len(records) == 1
    r = records[0]
    assert abs(r["grad_r_half"] - 0.6) < 0.08
    assert r["grad_r_half_lo"] < 0.6 < r["grad_r_half_hi"]
    assert r["grad_r_full"] == pytest.approx(rt.spearman_brown(r["grad_r_half"]))
    assert 0 < r["pass_r_half"] < 1 and 0 < r["mixed_fraction"] <= 1
    assert r["rows"] == 800


def test_trajectory_windows_slide_and_outputs_are_written(tmp_path):
    policy = _synthetic_rows(tmp_path, rho=0.2, steps=60)
    rows = rt.load_rows(policy)
    records = rt.trajectory(rows, window=20, reps=50)
    # stride is half the window; the last full window ends at the final step and
    # no partial trailing window is emitted
    assert [r["step_start"] for r in records] == [1, 11, 21, 31, 41]
    assert records[-1]["step_end"] == 60
    target = rt.write_outputs(records, policy)
    assert target.is_file() and (policy / "reliability_trajectory.dat").is_file()
    header = (policy / "reliability_trajectory.dat").read_text().splitlines()[0].split()
    assert header[:3] == ["step", "pass_r_full", "grad_r_full"]
    result = subprocess.run([sys.executable, str(ROOT / "src/reliability_trajectory.py"), "--policy", str(policy),
                             "--window", "30", "--reps", "20"], capture_output=True, text=True,
                            env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"})
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[reliability]" in result.stdout


def test_spearman_brown_edges():
    assert rt.spearman_brown(0.0) == 0.0
    assert rt.spearman_brown(1.0) == 1.0
    assert rt.spearman_brown(0.5) == pytest.approx(2 / 3)
    assert math.isnan(rt.spearman_brown(float("nan")))
