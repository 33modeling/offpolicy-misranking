"""CPU-only tests for utility-based regime characterization."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from regime_map import (
    SCHEMA,
    analyze_run,
    exact_topk_margin_implication,
    summarize_regimes,
)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value))


def make_run(root: Path, seed: int = 0, generation_git: str = "a" * 40) -> Path:
    run = root / f"regime-s{seed}"
    run.mkdir()
    n = 40
    write_json(
        run / "run_config.json",
        {
            "git": generation_git,
            "model": "/models/test",
            "dataset": "math500",
            "seed": seed,
            "drift": 25,
        },
    )
    write_json(run / "manifest.json", {"git": generation_git})
    write_json(
        run / "score_protocol.json",
        {
            "schema": "offpolicy-score-validation-split/v2",
            "source_run_git": generation_git,
            "generation_validation": {"validated_rows": n},
        },
    )
    write_json(
        run / "oracle_protocol.json",
        {
            "schema": "offpolicy-oracle-validation-split/v3",
            "generation_validation": {"validated_rows": n},
        },
    )
    truth = {idx: float(idx) for idx in range(n)}
    write_json(
        run / "scores_oracle.json",
        {str(idx): {"score": score} for idx, score in truth.items()},
    )
    write_json(
        run / "scores_splithalf.json",
        {
            str(idx): {
                "r": score,
                "r_high_budget": score,
                "a": score,
                "b": score,
            }
            for idx, score in truth.items()
        },
    )
    write_json(
        run / "scores_offpolicy.json",
        {
            "g00": {str(idx): {"score": score} for idx, score in truth.items()},
            "g10": {str(idx): {"score": -score} for idx, score in truth.items()},
            "g01": {str(idx): {"score": score} for idx, score in truth.items()},
            "g11": {str(idx): {"score": score} for idx, score in truth.items()},
        },
    )
    with (run / "rollouts_behavior_train.jsonl").open("w") as stream:
        for idx in range(n):
            for rollout_idx in range(8):
                # Half the pool has mixed rewards; half has identical rewards.
                reward = float((idx < 20 and rollout_idx < 4) or (idx >= 20))
                stream.write(
                    json.dumps(
                        {
                            "prompt_idx": idx,
                            "rollout_idx": rollout_idx,
                            "reward": reward,
                        }
                    )
                    + "\n"
                )
    write_json(
        run / "divergence_stats.json",
        {
            "token_kl_beta_pi": 0.01,
            "traj_ess_frac_g11": 0.9,
            "clipfrac_g11": 0.02,
            "tokens": 100,
            "rollouts": 320,
        },
    )
    return run


def test_regime_map_recognizes_positive_and_negative_selectors(tmp_path: Path) -> None:
    assert SCHEMA == "offpolicy-regime-map/v4"
    rows = analyze_run(make_run(tmp_path))
    all_rows = {row["policy"]: row for row in rows if row["stratum"] == "all"}
    assert all_rows["stale_g00"]["utility_gain"] > 0
    assert all_rows["stale_g00"]["utility_retention"] == 1.0
    assert all_rows["stale_g00"]["point_status"] == "provisional_effective_candidate"
    assert all_rows["stale_g10"]["utility_gain"] < 0
    assert all_rows["stale_g10"]["point_status"] == "provisional_ineffective_candidate"
    assert {row["stratum"] for row in rows} == {
        "all",
        "mixed_reward",
        "identical_reward",
    }


def test_summary_partitions_different_generation_commits(tmp_path: Path) -> None:
    rows = analyze_run(make_run(tmp_path, 0, "a" * 40))
    rows.extend(analyze_run(make_run(tmp_path, 1, "b" * 40)))
    summary = [
        row
        for row in summarize_regimes(rows)
        if row["stratum"] == "all" and row["policy"] == "stale_g00"
    ]
    assert {row["generation_git"] for row in summary} == {"a" * 40, "b" * 40}
    assert {row["seeds"] for row in summary} == {1}


def test_replicated_status_requires_at_least_three_seeds(tmp_path: Path) -> None:
    rows = []
    for seed in range(5):
        rows.extend(analyze_run(make_run(tmp_path, seed)))
    summary = summarize_regimes(rows)
    all_rows = {
        (row["stratum"], row["policy"]): row
        for row in summary
        if row["dataset"] == "math500"
    }
    assert all_rows[("all", "stale_g00")]["status"] == "provisional_effective"
    assert all_rows[("all", "stale_g10")]["status"] == "provisional_ineffective"

    for row in rows:
        row["final_resampling"] = True
    final = {
        (row["stratum"], row["policy"]): row
        for row in summarize_regimes(rows)
        if row["dataset"] == "math500"
    }
    assert final[("all", "stale_g00")]["status"] == "effective"
    assert final[("all", "stale_g10")]["status"] == "ineffective"


def test_uniform_error_margin_is_a_valid_sufficient_condition() -> None:
    truth = {0: 5.0, 1: 4.0, 2: 0.0, 3: -1.0}
    estimate = {0: 4.8, 1: 4.1, 2: 0.1, 3: -0.8}
    sufficient, exact = exact_topk_margin_implication(truth, estimate, 2)
    assert sufficient and exact

    bad = {0: 0.0, 1: 4.0, 2: 5.0, 3: -1.0}
    sufficient, exact = exact_topk_margin_implication(truth, bad, 2)
    assert not sufficient and not exact


def test_regime_map_rejects_v1_or_missing_ranking_split(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    protocol = json.loads((run / "score_protocol.json").read_text())
    protocol["schema"] = "offpolicy-score-validation-split/v1"
    write_json(run / "score_protocol.json", protocol)
    with pytest.raises(ValueError, match="corrected score/oracle protocols"):
        analyze_run(run)

    protocol["schema"] = "offpolicy-score-validation-split/v2"
    write_json(run / "score_protocol.json", protocol)
    halves = json.loads((run / "scores_splithalf.json").read_text())
    for row in halves.values():
        row.pop("r")
    write_json(run / "scores_splithalf.json", halves)
    with pytest.raises(ValueError, match="matched R or high-budget"):
        analyze_run(run)


def test_regime_map_uses_bootstrap_endpoints_for_candidate_labels(
    tmp_path: Path,
) -> None:
    run = make_run(tmp_path)
    micro = {}
    for idx in range(40):
        vector = torch.tensor([float(idx + 1), 8.0])
        micro[idx] = torch.stack([vector] * 8)
    torch.save(micro, run / "oracle_micro_groups.pt")
    torch.save(
        torch.tensor([[1.0, 0.0]] * 8),
        run / "val_groups.pt",
    )
    rows = analyze_run(run, first_bootstrap=100)
    aligned = next(
        row for row in rows if row["stratum"] == "all" and row["policy"] == "stale_g00"
    )
    reversed_row = next(
        row for row in rows if row["stratum"] == "all" and row["policy"] == "stale_g10"
    )
    assert aligned["floor_lower_one_sided_95"] == 1
    assert aligned["utility_gain_lower_one_sided_95"] > 0
    assert aligned["retention_margin_lower_one_sided_95"] >= 0
    assert aligned["point_status"] == "provisional_effective_candidate"
    assert reversed_row["utility_gain_upper_one_sided_95"] < 0
    assert reversed_row["point_status"] == "provisional_ineffective_candidate"
