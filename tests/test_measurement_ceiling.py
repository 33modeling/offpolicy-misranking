"""Regression tests for the registered Gaussian ceiling lookup."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from measurement_ceiling import (
    SCHEMA,
    gaussian_ceiling_from_reliability,
    gaussian_ceiling_interval,
)


def test_mbpp_registered_anchor_reproduces_manuscript_mapping() -> None:
    result = gaussian_ceiling_from_reliability(0.17110686, 512, 51)
    assert result is not None
    assert result["schema"] == SCHEMA
    assert result["rho_half"] == pytest.approx(0.20)
    assert result["ceiling"] == pytest.approx(0.373, abs=0.003)


def test_ceiling_interval_is_monotone_and_clamps_below_chance() -> None:
    result = gaussian_ceiling_interval(0.05, 0.20, 0.40, 400, 40)
    assert result is not None
    assert result["reliability_clamped"] is False
    assert result["ceiling_lower_two_sided_95"] == pytest.approx(0.10)
    assert result["ceiling_lower_two_sided_95"] <= result["ceiling"]
    assert result["ceiling"] <= result["ceiling_upper_two_sided_95"]


def test_unregistered_design_does_not_use_an_implicit_approximation() -> None:
    assert gaussian_ceiling_from_reliability(0.5, 40, 4) is None
