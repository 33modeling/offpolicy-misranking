from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from correction_selector import (
    box_quadratic,
    correction_components,
    crossfit_scores,
    directional_difference,
    endpoint_groups,
    fit_anchor_corrections,
    fit_corrections,
    select,
)


def test_box_solver_matches_dense_grid_and_boundaries():
    rng = np.random.default_rng(12)
    a = rng.normal(size=(3, 3))
    h = a.T @ a
    b = np.array([2.0, -0.3, 1.0])
    x = box_quadratic(h, b)
    grid = np.stack(np.meshgrid(*([np.linspace(0, 1, 31)] * 3)), axis=-1).reshape(-1, 3)
    losses = np.einsum("ni,ij,nj->n", grid, h, grid) - 2 * grid @ b
    assert x @ h @ x - 2 * b @ x <= losses.min() + 1e-10
    np.testing.assert_allclose(box_quadratic(np.eye(3), [-2, 0.4, 3]), [0, 0.4, 1])


def test_no_drift_reduces_exactly_to_same_endpoint():
    rng = np.random.default_rng(1)
    direction = rng.normal(size=(10, 8, 3))
    rewards = rng.integers(0, 2, size=(10, 8))
    g = endpoint_groups(np.zeros_like(direction), direction, rewards)
    np.testing.assert_allclose(g, np.repeat(g[:, :, :1], 4, axis=2))
    scores, _ = crossfit_scores(g)
    np.testing.assert_allclose(scores, g[:, :, 0].mean(axis=1))


def test_components_reconstruct_full_even_after_clipping():
    rng = np.random.default_rng(2)
    g = endpoint_groups(
        rng.normal(size=(8, 16, 5)) * 3,
        rng.normal(size=(8, 16, 5)),
        rng.integers(0, 2, (8, 16)),
    )
    np.testing.assert_allclose(
        correction_components(g).sum(axis=-1), g[:, :, 3], atol=1e-12
    )


def test_directional_finite_difference_matches_logistic_derivative():
    logits = np.array([-0.2, 0.5, 1.0])
    tangent = np.array([1.0, -2.0, 0.3])
    eps = 1e-4
    logp = lambda x: -np.logaddexp(0, -x)
    observed = directional_difference(
        logp(logits + eps * tangent), logp(logits - eps * tangent), eps
    )
    np.testing.assert_allclose(observed, tangent / (1 + np.exp(logits)), rtol=1e-7)


def test_calibration_does_not_read_scoring_half():
    g = np.random.default_rng(3).normal(size=(12, 8, 4))
    _, folds = crossfit_scores(g, seed=8)
    first = folds[0]
    changed = g.copy()
    changed[:, first["score_groups"]] *= 100
    _, new_folds = crossfit_scores(changed, seed=8)
    np.testing.assert_array_equal(first["coefficients"], new_folds[0]["coefficients"])
    assert not set(first["fit_groups"]) & set(first["score_groups"])


def test_noiseless_signal_retains_required_corrections():
    rng = np.random.default_rng(4)
    z = rng.normal(size=(100, 4))
    g = np.stack(
        [z[:, 0], z[:, 0] + z[:, 1], z[:, 0] + z[:, 2], z.sum(axis=1)], axis=-1
    )
    repeated = np.repeat(g[:, None, :], 4, axis=1)
    weights, _ = fit_corrections(repeated, 4)
    np.testing.assert_allclose(weights, 1, atol=1e-10)
    scores, _ = crossfit_scores(repeated)
    np.testing.assert_allclose(scores, g[:, 3], atol=1e-10)


def test_common_offsets_do_not_change_correction_fit_or_selection():
    g = np.random.default_rng(5).normal(size=(20, 8, 4))
    offset = np.array([4.0, -7.0, 2.0, 8.0])
    old, new = select(g, 5), select(g + offset, 5)
    assert old["selected_indices"] == new["selected_indices"]
    for a, b in zip(old["folds"], new["folds"]):
        np.testing.assert_allclose(a["coefficients"], b["coefficients"], atol=1e-10)


def test_ties_exploration_and_seed_are_reproducible():
    g = np.zeros((40, 4, 4))
    result = select(g, 10, seed=10, exploration_fraction=0.2)
    assert result == select(g, 10, seed=10, exploration_fraction=0.2)
    assert len(result["selected_indices"]) == 10
    assert result["exploration_count"] == 2
    assert result["selected_indices"] != select(g, 10, seed=11)["selected_indices"]


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_rejects_nonfinite_input(value):
    g = np.zeros((8, 4, 4))
    g[0, 0, 0] = value
    with pytest.raises(ValueError, match="finite"):
        select(g, 2)


def test_rejects_inadequate_independent_groups():
    with pytest.raises(ValueError, match="four independent"):
        select(np.zeros((8, 3, 4)), 2)
    with pytest.raises(ValueError, match="divisible"):
        endpoint_groups(np.zeros((8, 7, 2)), np.zeros((8, 7, 2)), np.zeros((8, 7)))


def test_anchor_fit_recovers_known_mixed_correction():
    rng = np.random.default_rng(19)
    z = rng.normal(size=(40, 4))
    g = np.stack(
        [z[:, 0], z[:, 0] + z[:, 1], z[:, 0] + z[:, 2], z.sum(axis=1)], axis=-1
    )
    g = np.repeat(g[:, None, :], 4, axis=1)
    ids = np.arange(12)
    expected = np.array([0.3, 0.7, 0.2])
    y = z[ids, 0] + z[ids, 1:] @ expected + 10
    actual, _ = fit_anchor_corrections(g, ids, y)
    np.testing.assert_allclose(actual, expected, atol=1e-10)
    scores, _ = crossfit_scores(g, anchor_ids=ids, anchor_scores=y)
    np.testing.assert_allclose(scores, z[:, 0] + z[:, 1:] @ expected, atol=1e-10)


def test_anchors_do_not_expose_unqueried_labels_or_scoring_groups():
    g = np.random.default_rng(20).normal(size=(20, 8, 4))
    ids, y = np.arange(8), np.arange(8) * 0.03
    _, folds = crossfit_scores(g, anchor_ids=ids, anchor_scores=y)
    changed = g.copy()
    changed[:, folds[0]["score_groups"]] *= 100
    _, new_folds = crossfit_scores(changed, anchor_ids=ids, anchor_scores=y)
    np.testing.assert_array_equal(
        folds[0]["coefficients"], new_folds[0]["coefficients"]
    )
    with pytest.raises(ValueError, match="at least eight"):
        crossfit_scores(g, anchor_ids=ids[:4], anchor_scores=y[:4])


@pytest.mark.parametrize("anchored", [False, True])
def test_cli_preserves_prompt_ids_provenance_and_existing_output(tmp_path, anchored):
    source = tmp_path / "input.npz"
    output = tmp_path / "selection.json"
    rng = np.random.default_rng(21)
    ids = np.array([f"prompt-{i}" for i in range(20)])
    np.savez(source, endpoint_groups=rng.normal(size=(20, 8, 4)), prompt_ids=ids)
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "src/correction_selector.py"),
        str(source),
        "--k",
        "5",
        "--output",
        str(output),
    ]
    if anchored:
        anchors = tmp_path / "anchors.npz"
        np.savez(anchors, anchor_ids=np.arange(8), anchor_scores=rng.normal(size=8))
        command += ["--anchors", str(anchors)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    original = output.read_bytes()
    result = json.loads(original)
    assert len(set(result["selected_indices"])) == 5
    assert result["selected_prompt_ids"] == ids[result["selected_indices"]].tolist()
    assert result["input"] == str(source.resolve())
    assert result["method"] == ("anchor_cfcs" if anchored else "cfcs")
    if anchored:
        assert result["anchors_input"] == str(anchors.resolve())
    repeated = subprocess.run(command, capture_output=True, text=True, check=False)
    assert repeated.returncode != 0
    assert output.read_bytes() == original


def test_synthetic_evaluator_cannot_change_training_and_counts_costs(monkeypatch):
    import correction_selector_study as study

    config = study.StudyConfig(n_candidates=40, n_validation=64, dim=16, horizon=6)
    rows, _ = study.run_cell(config, 1.0, 8, 31, 2)
    original = study.exact_gradients
    monkeypatch.setattr(
        study, "exact_gradients", lambda *args: np.zeros_like(original(*args))
    )
    changed, _ = study.run_cell(config, 1.0, 8, 31, 2)
    for first, second in zip(rows, changed):
        assert first["method"] == second["method"]
        assert first["test_reward_after"] == second["test_reward_after"]
        assert second["population_directional_regret"] == 0
        assert first["finite_difference_max_error"] < 1e-5
    by_method = {row["method"]: row for row in rows}
    assert by_method["cfcs"]["fresh_candidate_generation_tokens"] == 0
    assert by_method["cfcs"]["candidate_forward_token_evaluations"] == 3 * 40 * 8 * 6
    for method in ("anchor_cfcs", "anchor_best"):
        assert by_method[method]["fresh_candidate_generation_tokens"] == 8 * 32 * 6
        assert by_method[method]["fresh_candidate_gradient_tokens"] == 8 * 32 * 6
    assert (
        by_method["fresh_direction"]["fresh_candidate_generation_tokens"] == 40 * 8 * 6
    )
    assert len({row["training_generation_tokens"] for row in rows}) == 1
