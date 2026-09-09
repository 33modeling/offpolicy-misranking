"""Reliability-versus-budget sizing on synthetic micro-group artifacts (CPU)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import reliability_budget as rb  # noqa: E402

N, K = 400, 40          # registered MATH-500 design so the frozen lookup applies
GROUPS, GSIZE, DIM, VAL = 8, 4, 48, 100


def synthetic_artifacts(noise: float, seed: int = 0, groups: int = GROUPS, val_noise: float = 0.5):
    """Latent utility t_i along one direction u; every micro-group adds isotropic noise."""
    generator = torch.Generator().manual_seed(seed)
    direction = torch.randn(DIM, generator=generator)
    direction = direction / direction.norm()
    latent = torch.randn(N, generator=generator)
    base = latent[:, None] * direction[None, :] + 0.3 * torch.randn(N, DIM, generator=generator)
    stack = base[:, None, :] + noise * torch.randn(N, groups, DIM, generator=generator)
    val_groups = direction[None, :] + val_noise * torch.randn(VAL, DIM, generator=generator)
    return stack, val_groups


def write_run(tmp_path: Path, stack: torch.Tensor, val_groups: torch.Tensor, *, parked: bool = False,
              behavior: bool = False, fresh_k: int = GROUPS * GSIZE) -> Path:
    run = tmp_path / "point"
    run.mkdir(parents=True, exist_ok=True)
    micro = {idx: stack[idx].clone() for idx in range(stack.shape[0])}
    target = run / "pinned-scoring" / "20260908T233347Z" if parked else run
    target.mkdir(parents=True, exist_ok=True)
    torch.save(micro, target / "oracle_micro_groups.pt")
    torch.save(val_groups, target / "val_groups.pt")
    (run / "run_config.json").write_text(json.dumps(
        {"dataset": "math500", "fresh_k": fresh_k, "val_k": 8, "micro_group": GSIZE, "seed": 0, "drift": 0}
    ))
    if behavior:
        rows = []
        for idx in range(stack.shape[0]):
            rewards = [1.0] * 8 if idx % 10 == 0 else ([0.0] * 8 if idx % 10 == 1 else [1.0, 0.0] * 4)
            rows.extend(json.dumps({"prompt_idx": idx, "rollout_idx": j, "reward": r}) for j, r in enumerate(rewards))
        (run / "rollouts_behavior_train.jsonl").write_text("\n".join(rows) + "\n")
    return run


def test_floor_rises_with_half_size_and_falls_with_noise():
    stack, val = synthetic_artifacts(noise=2.0)
    small = rb.floor_statistic(stack, val, k=K, groups_per_half=1, val_prompts_per_half=25,
                               mode="both", reps=6, pairs=3, seed=1)
    large = rb.floor_statistic(stack, val, k=K, groups_per_half=4, val_prompts_per_half=25,
                               mode="both", reps=6, pairs=3, seed=1)
    assert large.mean > small.mean + 0.05
    assert large.correlation > small.correlation
    noisy_stack, noisy_val = synthetic_artifacts(noise=60.0)
    noisy = rb.floor_statistic(noisy_stack, noisy_val, k=K, groups_per_half=4, val_prompts_per_half=25,
                               mode="both", reps=6, pairs=3, seed=1)
    assert abs(noisy.mean - K / N) < 0.06          # chance = 0.10
    clean_stack, clean_val = synthetic_artifacts(noise=0.01, val_noise=0.01)
    clean = rb.floor_statistic(clean_stack, clean_val, k=K, groups_per_half=1, val_prompts_per_half=25,
                               mode="both", reps=3, pairs=2, seed=1)
    assert clean.mean > 0.9


def test_single_axis_modes_bound_the_registered_construction():
    stack, val = synthetic_artifacts(noise=2.0, val_noise=1.5)
    common = dict(k=K, groups_per_half=2, val_prompts_per_half=25, reps=6, pairs=3, seed=3)
    both = rb.floor_statistic(stack, val, mode="both", **common)
    candidate = rb.floor_statistic(stack, val, mode="candidate", **common)
    validation = rb.floor_statistic(stack, val, mode="validation", **common)
    assert candidate.correlation >= both.correlation - 0.02
    assert validation.correlation >= both.correlation - 0.02


def test_spearman_brown_round_trip_and_lookup_monotone():
    for r_unit in (0.05, 0.2, 0.6):
        for factor in (2, 4, 8):
            assert rb.unit_correlation(rb.spearman_brown(r_unit, factor), factor) == pytest.approx(r_unit, abs=1e-9)
    assert rb.spearman_brown(0.0, 8) == 0.0
    overlaps = [rb.overlap_from_correlation(rho, N, K) for rho in (0.0, 0.1, 0.3, 0.6, 0.9)]
    assert overlaps == sorted(overlaps)
    assert overlaps[0] == pytest.approx(K / N, abs=1e-6)
    assert rb.correlation_from_overlap(rb.overlap_from_correlation(0.3, N, K), N, K) == pytest.approx(0.3, abs=0.02)
    # unregistered design falls back to simulation and stays monotone
    fallback = [rb.overlap_from_correlation(rho, 120, 12) for rho in (0.0, 0.5, 0.95)]
    assert fallback[0] < fallback[1] < fallback[2]


def test_analyze_run_predicts_within_tolerance_and_reports_budget(tmp_path):
    stack, val = synthetic_artifacts(noise=3.0, val_noise=1.0)
    run = write_run(tmp_path, stack, val, behavior=True)
    artifacts = rb.load_run(run)
    assert artifacts.scoring == "current"
    assert artifacts.group_size == GSIZE
    readout = rb.analyze_run(artifacts, reps=8, pairs=3, target=0.20, seed=5)
    assert readout.n == N and readout.k == K and readout.groups == GROUPS
    assert readout.registered is not None
    assert (2, 25) in readout.curves["both"]
    assert readout.coupling is not None and readout.coupling > 0
    assert readout.budget_table[(8, 25)] == pytest.approx(readout.registered.mean, abs=0.06)
    # more responses per half can only raise the prediction
    column = [readout.budget_table[(r, 25)] for r in rb.CANDIDATE_HALF_RESPONSES]
    assert column == sorted(column)
    assert readout.self_check is not None
    predicted, observed = readout.self_check
    assert abs(predicted - observed) < 0.12
    assert readout.strata is not None
    assert readout.strata["mixed"] == 320 and readout.strata["all_right"] == 40 and readout.strata["all_wrong"] == 40
    assert readout.strata["mixed_floor"] is not None
    text = rb.render_report([readout], 0.20, 8, 3)
    assert "KEY point:" in text and "registered geometry" in text and "limiting side now" in text


def test_pure_noise_reports_no_reachable_budget(tmp_path):
    stack, val = synthetic_artifacts(noise=80.0, val_noise=80.0)
    run = write_run(tmp_path, stack, val, parked=True)
    artifacts = rb.load_run(run)
    assert artifacts.scoring == "pinned"
    readout = rb.analyze_run(artifacts, reps=6, pairs=3, target=0.20, seed=9)
    assert readout.registered.mean < 0.16
    assert readout.needed_candidate is None or readout.needed_candidate >= 128
    text = rb.render_report([readout], 0.20, 6, 3)
    assert "KEY point:" in text


def test_cli_writes_report(tmp_path, capsys):
    stack, val = synthetic_artifacts(noise=3.0)
    run = write_run(tmp_path, stack, val)
    out = tmp_path / "exports" / "reliability.txt"
    code = rb.main([str(run), "--out", str(out), "--reps", "4", "--pairs", "2", "--label", "math500/s0 d0"])
    assert code == 0
    assert out.is_file()
    body = out.read_text()
    assert "KEY math500/s0 d0:" in body and "KEY math500:" in body
    assert "reliability-budget" in capsys.readouterr().out


def test_cli_skips_unreadable_runs(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert rb.main([str(empty), "--reps", "2", "--pairs", "1"]) == 1


def test_half_scores_rejects_impossible_geometry():
    stack, val = synthetic_artifacts(noise=1.0)
    generator = torch.Generator().manual_seed(0)
    with pytest.raises(ValueError):
        rb.half_scores(stack, val, groups_per_half=5, val_prompts_per_half=25, mode="both", generator=generator)
    with pytest.raises(ValueError):
        rb.half_scores(stack, val, groups_per_half=1, val_prompts_per_half=60, mode="both", generator=generator)
    with pytest.raises(ValueError):
        rb.half_scores(stack, val, groups_per_half=1, val_prompts_per_half=25, mode="nope", generator=generator)
