"""Hierarchical FIRST bootstrap regression tests."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from first_interval import (
    bootstrap_floor_intervals,
    bootstrap_regime_intervals,
    percentile,
)


def test_percentile_interpolates() -> None:
    assert percentile([0.0, 1.0], 0.5) == 0.5


def test_identical_independent_halves_have_unit_lower_bound(tmp_path: Path) -> None:
    micro = {}
    for idx in range(20):
        vector = torch.tensor([float(idx + 1), 1.0])
        micro[idx] = torch.stack([vector, vector, vector, vector])
    torch.save(micro, tmp_path / "oracle_micro_groups.pt")
    torch.save(
        torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
        tmp_path / "val_groups.pt",
    )
    result = bootstrap_floor_intervals(
        tmp_path,
        {"all": list(range(20))},
        samples=100,
        seed=3,
        device="cpu",
    )
    assert result["all"]["lower_one_sided_95"] == 1.0
    assert result["all"]["upper_two_sided_95"] == 1.0


def test_regime_interval_separates_aligned_and_reversed_selectors(
    tmp_path: Path,
) -> None:
    micro = {}
    for idx in range(40):
        vector = torch.tensor([float(idx + 1), 8.0])
        micro[str(idx)] = torch.stack([vector, vector, vector, vector])
    torch.save(micro, tmp_path / "oracle_micro_groups.pt")
    torch.save(
        torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
        tmp_path / "val_groups.pt",
    )
    aligned = {idx: float(idx) for idx in range(40)}
    reversed_scores = {idx: -float(idx) for idx in range(40)}
    result = bootstrap_regime_intervals(
        tmp_path,
        {"all": list(range(40))},
        {"aligned": aligned, "reversed": reversed_scores},
        samples=100,
        seed=7,
        device="cpu",
    )["all"]
    assert result["fresh_gain"]["lower_one_sided_95"] > 0
    assert result["selectors"]["aligned"]["gain"]["lower_one_sided_95"] > 0
    assert result["selectors"]["aligned"]["retention"]["lower_one_sided_95"] == 1
    assert result["selectors"]["reversed"]["gain"]["upper_one_sided_95"] < 0
