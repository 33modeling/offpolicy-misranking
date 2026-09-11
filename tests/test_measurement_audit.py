import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import measurement_audit as audit


def test_exact_reversal_is_not_removed_by_independent_evaluation():
    e = audit.examples()["cosine_counterexample"]
    assert e["population_A"] > e["population_B"]
    assert e["finite_A"] < e["finite_B"]
    assert e["population_gain_select_B"] < 0 < e["finite_gain_select_B"]
    assert audit.examples()["tied_pool"]["expected_overlap"] < .2


def test_exact_binomial_expectation_and_no_noise():
    assert audit.exact_score([[1., 0.]], 1., 1, val_count=1)[0] == pytest.approx(1 / np.sqrt(2))
    assert audit.exact_score([[1., 0.]], 1., 2, val_count=1)[0] == pytest.approx(.5 + .5 / np.sqrt(2))
    assert audit.exact_score([[.8, .6]], 0., 2)[0] == pytest.approx(.8)


def test_calibration_uses_actual_core_on_cpu_and_reports_two_targets(monkeypatch):
    calls = []
    original = audit._bootstrap_scores
    def wrapped(candidate, validation, draws, generator):
        calls.append((str(candidate.device), candidate.shape, validation.shape))
        return original(candidate, validation, draws, generator)
    monkeypatch.setattr(audit, "_bootstrap_scores", wrapped)
    result = audit.calibration(trials=2, draws=100)
    assert len(calls) == 8 and all(c[0] == "cpu" for c in calls)
    for row in result["trials"]:
        if row["scenario"] == "no_noise":
            assert row["covers_population"] and row["covers_finite_budget"]
    assert result["scenarios"][0]["population"]["wilson_interval"][1] <= 1.0000001
    assert result == audit.calibration(trials=2, draws=100)


def test_invalid_calibration_budget():
    with pytest.raises(ValueError):
        audit.calibration(1, 100)
