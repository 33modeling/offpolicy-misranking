"""Tests for the 2026-09-07 registered extensions (E1-E6 analysis modules).

    PYTHONPATH=src python3 -m pytest tests/test_additional_experiments.py -q
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import pytest

import diagnostics_vs_retention as dvr
import downstream_compare as dc
import drift_curve
import margin_condition as mc
import reversal_matrix as rm
import synthetic_pool_study as sps
from rescore_variants import parse_variant

ESTIMATORS = ("g00", "g10", "g01", "g11")


# ---------------------------------------------------------------- synthetic study
def test_synthetic_replicate_is_deterministic_and_finite():
    config = sps.StudyConfig(n_candidates=60, n_validation=16, dim=16, horizon=3)
    a = sps.run_replicate(config, 0.1, 4, seed=3)
    b = sps.run_replicate(config, 0.1, 4, seed=3)
    np.testing.assert_equal(a, b)  # nan-aware equality
    for selector, metrics in a.items():
        assert set(metrics) >= {"precision", "utility_gain", "retention", "reversal_vs_fresh", "floor", "measurable"}
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["reversal_vs_fresh"] <= 1.0
        assert 0.0 <= metrics["floor"] <= 1.0
        assert math.isfinite(metrics["utility_gain"])
    assert a["fresh_r"]["precision"] == 1.0
    assert a["fresh_r"]["reversal_vs_fresh"] == 0.0


def test_synthetic_zero_drift_makes_all_corrections_identical():
    config = sps.StudyConfig(n_candidates=60, n_validation=16, dim=16, horizon=3)
    rows = sps.run_replicate(config, 0.0, 8, seed=11)
    reference = rows["g11"]
    for est in ("g00", "g10", "g01"):
        for key in ("precision", "utility_gain", "reversal_vs_fresh"):
            assert rows[est][key] == pytest.approx(reference[key], abs=1e-12)


def test_synthetic_token_weights_match_grads_definitions():
    log_r = np.array([[[0.3, -0.2, 0.5]]])  # (n=1, k=1, T=3)
    prefix = np.array([[[0.3, 0.1, 0.6]]])
    suffix = np.array([[[0.6, 0.3, 0.5]]])
    assert np.allclose(sps.token_log_weights(log_r, "g00"), log_r)
    assert np.allclose(sps.token_log_weights(log_r, "g10"), prefix)
    assert np.allclose(sps.token_log_weights(log_r, "g01"), suffix)
    assert np.allclose(sps.token_log_weights(log_r, "g11"), np.full_like(log_r, 0.6))
    with pytest.raises(ValueError):
        sps.token_log_weights(log_r, "g12")


def test_synthetic_exact_gradient_matches_monte_carlo():
    config = sps.StudyConfig(n_candidates=6, n_validation=8, dim=8, horizon=3)
    rng = np.random.default_rng(0)
    env = sps.Environment(config, 0.0, rng)
    ids = env.candidates
    exact = sps.exact_gradients(env, ids)
    actions = env.sample(env.theta_pi, ids, 40_000, rng)
    z = env.score_function(ids, actions).sum(axis=2)  # (n, k, dim)
    reward = actions[:, :, -1].astype(float)
    monte_carlo = (reward[:, :, None] * z).mean(axis=1)
    assert np.allclose(exact, monte_carlo, atol=0.01)


def test_synthetic_behavior_equals_current_at_zero_drift():
    config = sps.StudyConfig(n_candidates=5, n_validation=8, dim=8, horizon=4)
    env = sps.Environment(config, 0.0, np.random.default_rng(1))
    actions = env.sample(env.theta_pi, env.candidates, 6, np.random.default_rng(2))
    assert np.all(env.log_ratios(env.candidates, actions) == 0.0)
    drifted = sps.Environment(config, 1.0, np.random.default_rng(1))
    assert np.any(drifted.log_ratios(drifted.candidates, actions) != 0.0)


def test_synthetic_loo_advantages_are_leave_one_out():
    rewards = np.array([[1.0, 0.0, 0.0, 1.0]])
    adv = sps.loo_advantages(rewards)
    assert np.allclose(adv, [[1 - 1 / 3, -2 / 3, -2 / 3, 1 - 1 / 3]])
    assert np.all(sps.loo_advantages(np.array([[1.0]])) == 0.0)


def test_synthetic_study_writes_summary_and_dat(tmp_path: Path):
    config = sps.StudyConfig(n_candidates=40, n_validation=8, dim=8, horizon=2)
    summary = sps.run_study(config, [0.0, 0.2], [2, 8], replicates=2, seed=5, output_dir=tmp_path)
    assert len(summary) == 2 * 2 * len(sps.SELECTORS)
    for name in ("synthetic_pool_records.csv", "synthetic_pool_summary.csv", "retention_vs_delta.dat",
                 "reversal_vs_fresh_vs_delta.dat", "clip_fraction_vs_delta.dat", "retention_vs_kb.dat",
                 "floor_vs_kb.dat", "synthetic_pool_config.json"):
        assert (tmp_path / name).is_file(), name
    header = (tmp_path / "retention_vs_delta.dat").read_text().splitlines()[0].split()
    assert header[0] == "delta" and "g11_mean" in header and "g11_sd" in header


# ---------------------------------------------------------------- run fixtures
def _write_run(root: Path, n: int = 40, seed: int = 0, drift: int = 25, dataset: str = "math500",
               flip: float = 0.0, reward_rate: float = 0.5) -> Path:
    """A run directory with score artifacts, run_config and a behavior pool.

    ``flip`` is the fraction of prompts whose stale score has the opposite sign
    of the fresh score.
    """
    rng = random.Random(seed)
    run = root / f"m-s{seed}-{dataset}-d{drift}"
    run.mkdir(parents=True)
    fresh = {i: rng.uniform(-1, 1) for i in range(n)}
    halves = {str(i): {"r": fresh[i], "r_high_budget": fresh[i] * 0.9,
                       "a": fresh[i] + rng.gauss(0, 0.05), "b": fresh[i] + rng.gauss(0, 0.05)} for i in range(n)}
    off = {}
    for est in ESTIMATORS:
        off[est] = {}
        for i in range(n):
            value = fresh[i] * (1.0 + rng.gauss(0, 0.05))
            if rng.random() < flip:
                value = -value
            off[est][str(i)] = {"score": value, "norm": 1.0}
    oracle = {str(i): {"score": (halves[str(i)]["a"] + halves[str(i)]["b"]) / 2, "norm": 1.0} for i in range(n)}
    (run / "scores_splithalf.json").write_text(json.dumps(halves))
    (run / "scores_offpolicy.json").write_text(json.dumps(off))
    (run / "scores_oracle.json").write_text(json.dumps(oracle))
    (run / "run_config.json").write_text(json.dumps({"dataset": dataset, "drift": drift, "seed": seed,
                                                     "n_train": n, "model": "tiny"}))
    with (run / "rollouts_behavior_train.jsonl").open("w") as handle:
        for i in range(n):
            for j in range(8):
                reward = 1.0 if rng.random() < reward_rate else 0.0
                handle.write(json.dumps({"prompt_idx": i, "rollout_idx": j, "reward": reward}) + "\n")
    (run / "divergence_stats.json").write_text(json.dumps({
        "token_kl_beta_pi": 0.01 * drift, "rollouts": n * 8, "tokens": n * 800,
        "clipfrac_g00": 0.0, "clipfrac_g10": 0.1, "clipfrac_g01": 0.1, "clipfrac_g11": 0.2,
        "traj_ess_frac_g11": 0.5}))
    (run / "prompts.json").write_text(json.dumps({
        "train": [{"question": f"q{i}", "answer": str(i)} for i in range(n)],
        "val": [{"question": f"v{i}", "answer": str(i)} for i in range(8)]}))
    return run


# ---------------------------------------------------------------- reversal matrix
def test_reversal_matrix_counts_flips_and_anchor(tmp_path: Path):
    run = _write_run(tmp_path, flip=0.0)
    result = rm.analyze_run(run)
    for est in ESTIMATORS:
        assert result["selectors"][est]["flipped"] == 0
        assert result["selectors"][est]["rate"] == 0.0
    assert result["anchor"]["nonzero"] > 0
    flipped = _write_run(tmp_path / "flipped", flip=1.0, seed=1)
    result = rm.analyze_run(flipped)
    for est in ESTIMATORS:
        assert result["selectors"][est]["rate"] == 1.0
        assert result["selectors"][est]["band_rate"] == 1.0
    rows = rm.aggregate([rm.analyze_run(run), result])
    assert {row["selector"] for row in rows} == set(ESTIMATORS) | {"anchor"}
    table = rm.render_markdown(rows)
    assert "| math500 | 25 | g11 |" in table


# ---------------------------------------------------------------- margin condition
def test_margin_condition_recovers_when_condition_holds(tmp_path: Path):
    run = _write_run(tmp_path, flip=0.0)
    rows = mc.analyze_run(run)
    assert len(rows) == len(ESTIMATORS)
    for row in rows:
        assert row["ratio"] >= 0.0
        if row["condition_holds"]:
            assert row["recovered"], row
        assert not row["violation"]
    summary = mc.summarize(rows)
    assert summary["cells"] == len(ESTIMATORS)
    assert summary["violations"] == []


def test_margin_condition_flags_a_violation(tmp_path: Path):
    run = _write_run(tmp_path, flip=0.0, seed=2)
    # Forge a selector that agrees with the fresh scores except for one prompt
    # pushed out of the top-k while the reported error stays small: an
    # inconsistent artifact must surface as a violation, never be silently
    # accepted.
    off = json.loads((run / "scores_offpolicy.json").read_text())
    halves = json.loads((run / "scores_splithalf.json").read_text())
    fresh = {int(i): h["r"] for i, h in halves.items()}
    top = max(fresh, key=fresh.get)
    for est in ESTIMATORS:
        off[est] = {str(i): {"score": fresh[i], "norm": 1.0} for i in fresh}
        off[est][str(top)]["score"] = min(fresh.values()) - 1e-9
    (run / "scores_offpolicy.json").write_text(json.dumps(off))
    rows = mc.analyze_run(run)
    assert all(not row["recovered"] for row in rows)
    assert all(row["ratio"] < 1.0 for row in rows)  # the moved prompt makes the error large
    assert mc.summarize(rows)["violations"] == []


def test_margin_helpers():
    assert mc.margin_of({0: 0.9, 1: 0.5, 2: 0.1}, 1) == pytest.approx(0.4)
    assert mc.margin_of({0: 0.9}, 1) == float("inf")
    assert mc._unit_scale({0: 2.0, 1: -4.0}) == {0: 0.5, 1: -1.0}
    assert mc._unit_scale({0: 0.0}) == {0: 0.0}


# ---------------------------------------------------------------- diagnostics
def test_spearman_rank_correlation():
    assert dvr.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert dvr.spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    assert dvr.spearman([1, 2], [1, 2]) is None
    assert dvr.spearman([1, 1, 1], [1, 2, 3]) is None
    assert dvr.spearman([1, 2, 3, float("nan")], [1, 2, 3, 4]) == pytest.approx(1.0)


def test_diagnostics_correlations_cover_every_selector_and_diagnostic():
    rows = []
    for i in range(4):
        for selector in dvr.SELECTORS:
            rows.append({"selector": selector, "utility_retention": 1.0 - 0.2 * i, "topk_precision": 0.9 - 0.1 * i,
                         "token_kl_beta_pi": 0.1 * i, "traj_ess_frac_g11": 1.0 - 0.2 * i,
                         "clipfrac_g11": 0.05 * i, "clipfrac_selector": 0.02 * i})
    corr = dvr.correlations(rows)
    assert len(corr) == len(dvr.SELECTORS) * len(dvr.DIAGNOSTICS) * 2
    kl = next(c for c in corr if c["selector"] == "stale_g11" and c["diagnostic"] == "token_kl_beta_pi"
              and c["target"] == "utility_retention")
    assert kl["spearman"] == pytest.approx(-1.0)


# ---------------------------------------------------------------- drift curve
def test_drift_curve_point_metrics_and_summary(tmp_path: Path):
    runs = [_write_run(tmp_path, seed=s, drift=d) for s in (0, 1) for d in (0, 25)]
    rows = []
    for run in runs:
        rows.extend(drift_curve.point_metrics(run))
    assert len(rows) == 4 * len(ESTIMATORS)
    assert all(row["source"] == "registered" for row in rows)
    summary = drift_curve.summarize(rows)
    assert {(row["drift"], row["selector"]) for row in summary} == {(d, e) for d in (0, 25) for e in ESTIMATORS}
    assert all(row["seeds"] == 2 for row in summary)
    drift_curve.write_dat(tmp_path / "curve.dat", summary, "registered", "math500", "utility_retention")
    lines = (tmp_path / "curve.dat").read_text().splitlines()
    assert lines[0].split()[0] == "drift" and len(lines) == 3


def test_drift_curve_keeps_curve_and_registered_points_apart(tmp_path: Path):
    registered = _write_run(tmp_path / "reg", drift=25)
    curve = _write_run(tmp_path / "curve", drift=25, seed=1)
    rows = drift_curve.point_metrics(registered) + drift_curve.point_metrics(curve, source="curve")
    summary = drift_curve.summarize(rows)
    assert {row["source"] for row in summary} == {"registered", "curve"}
    assert all(row["seeds"] == 1 for row in summary)


# ---------------------------------------------------------------- downstream
def test_downstream_subsets_cover_every_selector(tmp_path: Path):
    run = _write_run(tmp_path, n=40)
    written = dc.write_subsets(run, tmp_path / "subsets", frac=0.10, seed=0)
    assert set(written) == set(dc.SELECTORS)
    for name, path in written.items():
        payload = json.loads(path.read_text())
        assert payload["selector"] == name
        assert len(payload["train"]) == payload["k"] == 4
        assert len(payload["val"]) == 8
        assert sorted(payload["selected_idx"]) == payload["selected_idx"]
    fresh = json.loads(written["fresh_r"].read_text())["selected_idx"]
    halves = json.loads((run / "scores_splithalf.json").read_text())
    top = sorted(range(40), key=lambda i: -halves[str(i)]["r"])[:4]
    assert sorted(top) == fresh


def test_downstream_summarize_pairs_against_fresh(tmp_path: Path):
    (tmp_path / "eval-before.json").write_text(json.dumps({"mean_reward": 0.30}))
    for name, after in (("fresh_r", 0.40), ("g11", 0.38), ("random", 0.31)):
        (tmp_path / name).mkdir()
        (tmp_path / name / "eval-after.json").write_text(json.dumps({"mean_reward": after}))
    rows = dc.summarize(tmp_path)
    by = {row["selector"]: row for row in rows}
    assert by["fresh_r"]["change_minus_fresh"] == pytest.approx(0.0)
    assert by["g11"]["change_minus_fresh"] == pytest.approx(-0.02)
    assert by["random"]["reward_change"] == pytest.approx(0.01)


# ---------------------------------------------------------------- rescore variants
def test_variant_specs_parse():
    assert parse_variant("bk4", 10.0) == (4, 10.0)
    assert parse_variant("clip3", 10.0) == (None, 3.0)
    assert parse_variant("bk2-clip30", 10.0) == (2, 30.0)
    for bad in ("", "bk1", "clip0.5", "k4", "bk4clipx"):
        with pytest.raises(ValueError):
            parse_variant(bad, 10.0)
