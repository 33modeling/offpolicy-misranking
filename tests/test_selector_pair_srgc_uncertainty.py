import numpy as np
import pytest

import selector_pair_srgc_uncertainty as uncertainty


def test_constant_prompt_projections_give_zero_uncertainty(monkeypatch):
    data = {
        "candidate-a": {0: [1.], 1: [1.], 2: [2.], 3: [2.]},
        "candidate-b": {0: [1.], 1: [1.], 2: [2.], 3: [2.]},
        "validation-a": {10: [1.], 11: [1.]},
        "validation-b": {12: [1.], 13: [1.]},
    }
    monkeypatch.setattr(uncertainty.score, "projections",
                        lambda directory, stage: {i: np.asarray(v) for i, v in data[stage].items()})
    result = uncertainty.estimate(None, {"on_policy": [0, 1], "cached": [2, 3]}, 1)
    assert result["d"] == -1.
    assert result["standard_error"] == 0.
    assert result["upper"] == -1.
    assert result["confirmed_sr"] is True


def test_wider_later_bound_and_shared_candidate_ids(monkeypatch):
    data = {
        "candidate-a": {0: [1.], 1: [2.], 2: [3.]},
        "candidate-b": {0: [2.], 1: [1.], 2: [4.]},
        "validation-a": {10: [1.], 11: [2.], 12: [3.]},
        "validation-b": {13: [2.], 14: [3.], 15: [4.]},
    }
    monkeypatch.setattr(uncertainty.score, "projections",
                        lambda directory, stage: {i: np.asarray(v) for i, v in data[stage].items()})
    sets = {"on_policy": [0, 1], "cached": [1, 2]}
    first = uncertainty.estimate(None, sets, 1)
    later = uncertainty.estimate(None, sets, 3)
    assert first["standard_error"] > 0
    assert later["d"] == first["d"]
    assert later["upper"] > first["upper"]
    assert later["alpha_at_check"] < first["alpha_at_check"]
    with pytest.raises(ValueError, match="check index"):
        uncertainty.estimate(None, sets, 0)
