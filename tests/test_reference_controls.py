"""Signal-resolution control (src/reference_controls.py, scripts/reference_controls.sh)."""

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import reference_controls as rc


def write_rollouts(path: Path, matrix: torch.Tensor) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for p in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                handle.write(json.dumps({"prompt_idx": p, "rollout_idx": j, "reward": float(matrix[p, j]),
                                         "response": '"reward": 1, "prompt_idx": 999'}) + "\n")


def bernoulli_pool(n: int, responses: int, seed: int, spread: bool = True) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    p = torch.linspace(0.0, 1.0, n) if spread else torch.full((n,), 0.3)
    return (torch.rand(n, responses, generator=generator) < p[:, None]).double()


def test_read_rewards_uses_json_fields_and_validates(tmp_path):
    matrix = torch.tensor([[1.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0]])
    write_rollouts(tmp_path / "r.jsonl", matrix)
    assert torch.equal(rc.read_rewards(tmp_path / "r.jsonl"), matrix.double())
    (tmp_path / "dup.jsonl").write_text((json.dumps({"prompt_idx": 0, "rollout_idx": 0, "reward": 1}) + "\n") * 2)
    with pytest.raises(ValueError, match="duplicate"):
        rc.read_rewards(tmp_path / "dup.jsonl")
    (tmp_path / "bad.jsonl").write_text(json.dumps({"prompt_idx": 0, "rollout_idx": 0, "reward": 0.5}) + "\n")
    with pytest.raises(ValueError, match="finite binary"):
        rc.read_rewards(tmp_path / "bad.jsonl")
    (tmp_path / "gap.jsonl").write_text("".join(json.dumps({"prompt_idx": p, "rollout_idx": j, "reward": 0}) + "\n"
                                                for p in range(2) for j in (0, 1, 2, 4)))
    with pytest.raises(ValueError, match="noncontiguous"):
        rc.read_rewards(tmp_path / "gap.jsonl")


def test_registered_halves_are_the_last_two_quarters():
    a, b = rc.registered_halves(32)
    assert (a.start, a.stop, b.start, b.stop) == (16, 24, 24, 32)


def test_spread_pass_rates_are_reproducible_and_constant_pool_is_at_chance():
    spread = rc.overlap_by_ranking(bernoulli_pool(400, 32, 1), k_frac=0.1, reps=5, pairs=5, seed=0)
    chance = 40 / 400
    # eight-response halves resolve p to 1/8, so ties at the boundary cap the
    # overlap well below one; "reproducible" here means far above chance
    assert spread["pass-rate"].registered > chance + 0.3
    assert spread["hardest"].registered > chance + 0.3
    # the mixed-difficulty band is dense and tie-heavy (p resolved to 1/8), so
    # learnability is only modestly reproducible even with a true spread
    assert spread["learnability"].registered > chance + 0.05
    flat = rc.overlap_by_ranking(bernoulli_pool(400, 32, 2, spread=False), k_frac=0.1, reps=5, pairs=5, seed=0)
    # every prompt shares one success probability: no ordering to reproduce
    assert abs(flat["pass-rate"].resampled_mean - chance) < 0.08
    for stat in flat.values():
        assert 0.0 <= stat.registered <= 1.0 and stat.boundary_ties >= 1


def test_audit_point_reads_floors_and_counts_degenerate_prompts(tmp_path):
    run = tmp_path / "tag-s0-math500-d0"
    run.mkdir()
    matrix = bernoulli_pool(50, 8, 3)
    matrix[0] = 0.0
    matrix[1] = 1.0
    write_rollouts(run / "rollouts_fresh_train.jsonl", matrix)
    (run / "report.json").write_text(json.dumps({"noise_floor": 0.175, "k": 5}))
    parked = run / "pinned-scoring" / "20260908T000000Z"
    parked.mkdir(parents=True)
    (parked / "report.json").write_text(json.dumps({"noise_floor": 0.2, "k": 5}))
    row = rc.audit_point(run, "math500/s0", k_frac=0.1, reps=3, pairs=3, seed=0)
    assert row.floors == {"current": 0.175, "pinned": 0.2}
    means = matrix.mean(dim=1)
    assert row.all_wrong == int((means == 0).sum()) >= 1
    assert row.all_right == int((means == 1).sum()) >= 1
    assert (row.k, row.chance) == (5, 0.1)
    assert row.zero_groups is None  # no gradients stored: reported as absent, not zero


def test_main_writes_report_outside_the_point_and_refuses_inside(tmp_path, capsys):
    run = tmp_path / "tag-s1-mbpp-d0"
    run.mkdir()
    write_rollouts(run / "rollouts_fresh_train.jsonl", bernoulli_pool(60, 16, 4))
    out = tmp_path / "exports" / "controls.txt"
    assert rc.main([str(run), "--label", "mbpp/s1", "--out", str(out), "--reps", "2", "--pairs", "2"]) == 0
    text = out.read_text()
    assert " mbpp/s1 " in text and "KEY means over 1 point(s)" in text and "gradient -" in text
    with pytest.raises(ValueError, match="inside an experiment point"):
        rc.main([str(run), "--out", str(run / "controls.txt"), "--reps", "1", "--pairs", "1"])


def test_shell_wrapper_discovers_d0_points_and_exports(tmp_path):
    work = tmp_path / "work"
    tag = "olmo3-1025-7b-base-rlzero-grpo-h100-v2"
    for dataset, seed in (("math500", 0), ("mbpp", 3)):
        run = work / "runs" / tag / f"family-{dataset}-s{seed}" / f"{tag}-s{seed}-{dataset}-d0"
        run.mkdir(parents=True)
        write_rollouts(run / "rollouts_fresh_train.jsonl", bernoulli_pool(40, 8, seed))
    (work / "runs" / tag / "family-mbpp-s4").mkdir()  # no responses yet: skipped, not fatal
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/reference_controls.sh")],
        cwd=ROOT, capture_output=True, text=True, timeout=120, check=False,
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "OM_WORK": str(work),
             "VENV_DIR": str(Path(sys.executable).parent.parent), "RC_REPS": "2", "RC_PAIRS": "2"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[skip] mbpp/s4" in result.stdout
    reports = list((work / "exports").glob(f"reference-controls-{tag}-*.txt"))
    assert len(reports) == 1
    text = reports[0].read_text()
    assert " math500/s0 " in text and " mbpp/s3 " in text and "KEY means over 2 point(s)" in text
    key = next(line for line in text.splitlines() if line.startswith("KEY means"))
    assert not math.isnan(float(key.split("chance ")[-1].split()[0]))


def test_partition_agreement_is_one_for_identical_halves_and_near_chance_when_independent():
    pa = torch.tensor([0.0, 0.0, 0.5, 0.25, 1.0, 1.0, 0.75, 0.0])
    same = rc.partition_agreement(pa, pa.clone())
    assert same.kappa == pytest.approx(1.0)
    assert all(same.jaccard[b] == pytest.approx(1.0) for b in rc.BANDS)
    generator = torch.Generator().manual_seed(0)
    # independent halves with all three bands populated: 0, 1, or a mixed value
    def draw():
        kind = torch.randint(0, 3, (4000,), generator=generator)
        mixed_value = 0.25 + 0.5 * torch.rand(4000, generator=generator)
        return torch.where(kind == 0, 0.0, torch.where(kind == 2, 1.0, mixed_value))
    a, b = draw(), draw()
    indep = rc.partition_agreement(a, b)
    assert abs(indep.kappa) < 0.05
    for band in rc.BANDS:
        assert abs(indep.jaccard[band] - indep.jaccard_chance[band]) < 0.05


def test_utility_comparison_rewards_a_band_that_carries_the_signal():
    n = 200
    generator = torch.Generator().manual_seed(1)
    mixed = torch.zeros(n, dtype=torch.bool)
    mixed[:80] = True
    noise = torch.randn(n, generator=generator) * 0.05
    truth = torch.where(mixed, torch.full((n,), 0.5), torch.zeros(n)) + noise
    scores = {"truth": {i: float(truth[i]) for i in range(n)},
              "fresh": {i: float(truth[i] + 0.2 * torch.randn(1, generator=generator)) for i in range(n)},
              "g11": {i: float(torch.randn(1, generator=generator)) for i in range(n)}}
    result = rc.utility_comparison(scores, {"fresh-band": mixed}, k_frac=0.1, draws=50, seed=0)
    assert result.k == 20 and result.band_size["fresh-band"] == 80
    assert result.gain["fresh-band-random"] > 0.25
    assert result.gain["fresh-topk"] > result.gain["fresh-band-random"] - 0.1
    assert abs(result.gain["stale-g11"]) < 0.15
    assert result.fresh_topk_in_band["fresh-band"] > 0.9
    empty = rc.utility_comparison(scores, {"fresh-band": torch.zeros(n, dtype=torch.bool)}, k_frac=0.1, draws=5, seed=0)
    assert math.isnan(empty.gain["fresh-band-random"])


def test_audit_point_uses_split_half_scores_when_present(tmp_path):
    run = tmp_path / "tag-s2-math500-d0"
    run.mkdir()
    n = 40
    matrix = bernoulli_pool(n, 32, 5)
    write_rollouts(run / "rollouts_fresh_train.jsonl", matrix)
    write_rollouts(run / "rollouts_behavior_train.jsonl", bernoulli_pool(n, 8, 6))
    (run / "scores_splithalf.json").write_text(json.dumps(
        {str(i): {"r": float(i) / n, "a": float(i) / n + 0.01, "b": float(i) / n - 0.01} for i in range(n)}))
    (run / "scores_offpolicy.json").write_text(json.dumps(
        {name: {str(i): {"score": float(n - i), "norm": 1.0} for i in range(n)} for name in rc.ESTIMATORS}))
    row = rc.audit_point(run, "math500/s2", k_frac=0.1, reps=2, pairs=2, seed=0, draws=10)
    assert row.utility is not None and row.notes == []
    assert row.utility.gain["fresh-topk"] > 0 > row.utility.gain["stale-g11"]  # stale ranking is reversed on purpose
    assert set(row.utility.band_size) == {"fresh-band", "behavior-band"}
    text = rc.render([row], reps=2, pairs=2, k_frac=0.1)
    assert "## 2." in text and " math500/s2 " in text and "KEY utility gain over uniform" in text
