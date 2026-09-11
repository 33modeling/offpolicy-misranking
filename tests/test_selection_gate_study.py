import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from test_selection_gate import scope

import selection_gate as sg
import selection_gate_study as study

ROOT = Path(__file__).resolve().parents[1]


def point(name, progress, role="development", trajectory=None):
    delta = .04 if progress < .5 else -.03
    lineage = {"parent_sha256": f"parent-{name}", "optimizer_sha256": f"optimizer-{name}",
               "pool_sha256": scope()["pool_sha256"], "evaluation_sha256": "diagnostic-evaluation",
               "learner_config_sha256": "same-learner", "gpu_type": scope()["gpu_type"]}
    rewards = {"random_full": .5, "random_reduced": .49, "selection_reduced": .49+delta}
    branches = {arm: {**lineage, "complete": True, "stop_reason": "budget_exhausted",
                      "budget_gpu_seconds": 100. if arm == "random_full" else 98.,
                      "used_gpu_seconds": 100. if arm == "random_full" else 98.,
                      "rewards": {"q0": r-.01, "q1": r+.01}} for arm, r in rewards.items()}
    return {"id": name, "trajectory_id": trajectory or name, "role": role, "step": 100,
            "feature_step": 100, "scope": scope(), "budget_gpu_seconds": 100.,
            "full_pool_coverage": True, "decision_schedule": "once_before_training",
            "measurement_gpu_seconds": 2., "branches": branches,
            "features": {**{f: .5 for f in sg.FEATURES}, "success_rate_p50": progress}}


def fixture():
    points = [point(f"train-{i}", i/7) for i in range(8)]
    points += [point("cal-a", .2, "calibration"), point("cal-b", .8, "calibration"),
               point("test-a", .2, "test"), point("test-b", .8, "test")]
    return {"schema": study.STUDY_SCHEMA, "data_kind": "synthetic", "points": points}


def test_fit_analyze_and_exact_cost_accounting():
    data = fixture()
    fitted = study.fit(data)
    assert len(fitted["nodes"]) <= 7
    assert fitted["fit_trajectories"] == [f"train-{i}" for i in range(8)]
    report = study.analyze(data, fitted)
    test = [r for r in report["rows"] if r["role"] == "test"]
    assert [r["action"] for r in test] == ["select", "random"]
    assert test[0]["gate_minus_random_full"] == pytest.approx(.03)
    assert test[1]["gate_minus_random_full"] == pytest.approx(-.01), "fallback still pays measurement"
    assert report["by_role"]["test"]["mean_gate_contrast"] == pytest.approx(.01)
    assert not report["certified"] and report["data_kind"] == "synthetic"
    assert all(abs(r["accounting_residual"]) < 1e-12 for r in report["rows"])


def test_no_model_bypasses_measurement_entirely():
    report = study.analyze(fixture())
    assert all(r["gate_minus_random_full"] == 0. and not r["measurement_performed"] for r in report["rows"])
    assert all(abs(r["accounting_residual"]) < 1e-12 for r in report["rows"])
    assert report["by_role"]["test"]["mean_weighted_error"] is None


def test_exported_tree_matches_sklearn_float32_threshold_semantics():
    from sklearn.tree import DecisionTreeRegressor
    data = fixture()
    fitted = study.fit(data, min_leaf=1, features=("success_rate_p50",))
    estimator = DecisionTreeRegressor(max_depth=2, min_samples_leaf=1, random_state=20260912)
    points = data["points"][:8]
    estimator.fit([[p["features"]["success_rate_p50"]] for p in points],
                  [study.validate_point(p)["net_selection_gain"] for p in points])
    for value in [*np.linspace(0, 1, 101), .5-1e-10, .5, .5+1e-10]:
        assert sg.predict(fitted, {"success_rate_p50": float(value)}) == pytest.approx(estimator.predict([[value]])[0])


def test_fit_does_not_read_test_rewards():
    data = fixture()
    original = study.fit(data)
    for p in data["points"]:
        if p["role"] != "development":
            p["branches"]["selection_reduced"]["rewards"] = {"q0": 0., "q1": 1.}
    assert study.fit(data) == original


@pytest.mark.parametrize("key", study.LINEAGE)
def test_parent_optimizer_evaluation_and_hardware_must_match(key):
    p = point("a", .2)
    p["branches"]["selection_reduced"][key] = "different"
    with pytest.raises(ValueError, match="mismatch"):
        study.validate_point(p)


@pytest.mark.parametrize("change", ["budget", "overshoot", "step_limit", "incomplete", "reward", "prompt_ids", "features", "future"])
def test_reject_invalid_causal_comparisons(change):
    p = point("a", .2)
    branch = p["branches"]["selection_reduced"]
    if change == "budget":
        branch["budget_gpu_seconds"] = 100.
    elif change == "overshoot":
        branch["used_gpu_seconds"] = 99.
    elif change == "step_limit":
        branch["stop_reason"] = "updates_completed"
    elif change == "incomplete":
        branch["complete"] = False
    elif change == "reward":
        branch["rewards"]["q0"] = float("nan")
    elif change == "prompt_ids":
        branch["rewards"] = {"other": .5}
    elif change == "features":
        p["features"]["final_test_reward"] = .8
    else:
        p["feature_step"] = 101
    with pytest.raises(ValueError):
        study.validate_point(p)


def test_whole_trajectories_and_renamed_parents_cannot_cross_splits():
    data = fixture()
    data["points"][-1]["trajectory_id"] = "train-0"
    with pytest.raises(ValueError, match="trajectory leakage"):
        study.validate_study(data)
    data = fixture()
    for branch in data["points"][-1]["branches"].values():
        branch["parent_sha256"] = "parent-train-0"
        branch["optimizer_sha256"] = "optimizer-train-0"
    with pytest.raises(ValueError, match="checkpoint leakage"):
        study.validate_study(data)


def test_repeated_checkpoints_are_grouped_not_counted_as_more_trajectories():
    data = fixture()
    extra = point("test-a-second", .3, "test", "test-a")
    data["points"].append(extra)
    report = study.analyze(data, study.fit(data))
    assert report["by_role"]["test"]["points"] == 3
    assert report["by_role"]["test"]["trajectories"] == 2
    assert report["by_role"]["test"]["mean_gate_contrast"] == pytest.approx(.01)


def test_synthetic_models_cannot_be_evaluated_as_observed_results():
    data = fixture()
    fitted = study.fit(data)
    data["data_kind"] = "observed"
    with pytest.raises(ValueError, match="data kind"):
        study.analyze(data, fitted)


def test_cached_features_are_exact_and_gpu_free(tmp_path):
    path = tmp_path / "rollouts.jsonl"
    rows = [{"prompt_idx": i, "rollout_idx": j, "reward": reward}
            for i, group in enumerate(([0, 0], [0, 1], [1, 1], [1, 0]))
            for j, reward in enumerate(group)]
    path.write_text("".join(json.dumps(r)+"\n" for r in rows))
    result = study.cached_features(path, step=10,
                                   expected_prompts=4, expected_responses=2)
    assert result["features"]["success_rate"] == .5
    assert result["features"]["mixed_group_fraction"] == .5
    assert result["success_count_histogram"] == [1, 2, 1]
    assert result["features"]["success_rate_std"] == pytest.approx(2**-.5/2)
    assert result["full_pool_coverage"]
    assert result["allocated_gpu_seconds"] == 0.
    assert result["wall_seconds"] >= 0 and result["source_sha256"]
    path.write_text(path.read_text()+json.dumps(rows[0])+"\n")
    with pytest.raises(ValueError, match="duplicate"):
        study.cached_features(path, step=10,
                              expected_prompts=4, expected_responses=2)


def test_incomplete_or_large_cache_is_rejected(tmp_path):
    path = tmp_path / "cache.jsonl"
    path.write_text(json.dumps({"prompt_idx": 0, "rollout_idx": 0, "reward": 1})+"\n")
    with pytest.raises(ValueError, match="coverage incomplete"):
        study.cached_features(path, step=0,
                              expected_prompts=4, expected_responses=2)
    with pytest.raises(ValueError, match="byte cap"):
        study.cached_features(path, step=0,
                              expected_prompts=4, expected_responses=2, max_bytes=2)


def test_core_imports_no_gpu_or_training_framework():
    code = "import sys,selection_gate,selection_gate_study; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code], env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "CUDA_VISIBLE_DEVICES": ""},
                            capture_output=True, text=True, check=False, timeout=10)
    assert result.returncode == 0, result.stderr


def test_shell_fit_analyze_and_state_status(tmp_path):
    path, fitted, report = tmp_path / "study.json", tmp_path / "model.json", tmp_path / "report.json"
    sg.atomic_json(path, fixture())
    env = {**os.environ, "GATE_PYTHON": sys.executable}
    def run(*args):
        return subprocess.run(["bash", "scripts/run_selection_gate.sh", *args], cwd=ROOT, env=env,
                              capture_output=True, text=True, timeout=30, check=False)
    assert run("plan").returncode == 0
    result = run("fit", "--study", str(path), "--out", str(fitted))
    assert result.returncode == 0, result.stderr
    result = run("analyze", "--study", str(path), "--model", str(fitted), "--out", str(report))
    assert result.returncode == 0, result.stderr
    assert not sg.read(report)["certified"]
    assert run("status", "--state", str(tmp_path / "missing.json")).returncode == 0
    assert run("fit", "--study", str(path), "--out", str(path)).returncode == 2
    assert sg.read(path) == fixture()


def test_legacy_results_are_read_but_not_fabricated_into_gate_labels(tmp_path):
    payload = {"complete": True, "rows": [
        {"selector": "random", "dataset": "math", "seed": 0, "drift": 400, "reward_after": .5},
        {"selector": "g11", "dataset": "math", "seed": 0, "drift": 400, "reward_after": .48}]}
    sg.atomic_json(tmp_path / "downstream_results.json", payload)
    result = study.discover([tmp_path])
    assert result["legacy_reports"][0]["comparisons"][0]["difference_vs_random"] == pytest.approx(-.02)
    assert result["matched_switch_labels"] == 0
    assert not result["legacy_reports"][0]["gate_labels_usable"]


def test_cost_identity_and_prediction_bound_across_many_finite_pools():
    rng = np.random.default_rng(8123)
    for _ in range(250):
        p = rng.dirichlet(np.ones(8))
        rand, selected, full = rng.random((3, 8))
        delta, ell = selected-rand, full-rand
        estimate = delta+rng.uniform(-.2, .2, 8)
        accept = estimate > 0
        gain = np.dot(p, np.where(accept, selected, rand)-full)
        assert gain == pytest.approx(np.dot(p, accept*delta-ell))
        lower = np.dot(p, np.maximum(delta, 0)-ell-np.abs(estimate-delta))
        assert gain >= lower-1e-12


def test_previously_fitted_trajectory_cannot_be_reused_as_test():
    data = fixture()
    fitted = study.fit(data)
    other = {**copy.deepcopy(data), "points": [point("new-point", .2, "test", "train-0")]}
    with pytest.raises(ValueError, match="used to fit"):
        study.analyze(other, fitted)


def test_renaming_a_fitted_parent_does_not_make_it_held_out():
    data = fixture()
    fitted = study.fit(data)
    p = point("unseen-name", .2, "test")
    for b in p["branches"].values():
        b["parent_sha256"], b["optimizer_sha256"] = "parent-train-0", "optimizer-train-0"
    with pytest.raises(ValueError, match="used to fit"):
        study.analyze({**data, "points": [p]}, fitted)


def cache(path, successes, k=2):
    path.write_text("".join(json.dumps({"prompt_idx": i, "rollout_idx": j, "reward": int(j < n)})+"\n"
                            for i, n in enumerate(successes) for j in range(k)))


def test_full_distribution_distinguishes_pools_with_the_same_mean(tmp_path):
    path = tmp_path / "cache.jsonl"
    cache(path, [0, 0, 2, 2])
    a = study.cached_features(path, step=100, expected_prompts=4, expected_responses=2)
    cache(path, [1, 1, 1, 1])
    b = study.cached_features(path, step=100, expected_prompts=4, expected_responses=2, allocated_gpus=4)
    assert a["features"]["success_rate"] == b["features"]["success_rate"] == .5
    assert a["features"]["mixed_group_fraction"] == 0
    assert b["features"]["mixed_group_fraction"] == 1
    assert a["features"]["success_rate_std"] == .5
    assert b["features"]["success_rate_std"] == 0
    assert b["allocated_gpu_seconds"] == b["wall_seconds"]*4


def test_initialize_only_scans_once_and_resume_needs_no_source(tmp_path, monkeypatch):
    from test_selection_gate import model
    fitted = model()
    config = sg.GateConfig(scope=scope(), total_gpu_seconds=1000., measurement_gpu_seconds=50.,
                          start_step=100, data_kind="synthetic", model_id=fitted["model_id"])
    path, state = tmp_path / "cache.jsonl", tmp_path / "state.json"
    cache(path, [0, 1, 1, 2])
    a = study.initialize(state, config, model=fitted, rollouts=path, prompts=4, responses=2, allocated_gpus=4)
    assert a["action"] == "select" and a["checks"] == 1
    path.unlink()
    monkeypatch.setattr(study, "cached_features", lambda *a, **k: pytest.fail("measured twice"))
    assert study.initialize(state, config) == a


def test_initialize_without_model_needs_no_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "cached_features", lambda *a, **k: pytest.fail("unnecessary measurement"))
    config = sg.GateConfig(scope=scope(), total_gpu_seconds=1000.)
    result = study.initialize(tmp_path / "state.json", config)
    assert result["action"] == "random" and result["measurement_used"] == 0.


def test_broken_pool_falls_back_once_instead_of_repeated_probes(tmp_path):
    from test_selection_gate import model
    fitted = model()
    config = sg.GateConfig(scope=scope(), total_gpu_seconds=1000., measurement_gpu_seconds=50.,
                          data_kind="synthetic", model_id=fitted["model_id"])
    state = tmp_path / "state.json"
    result = study.initialize(state, config, model=fitted, rollouts=tmp_path / "missing", prompts=4)
    assert result["action"] == "random" and result["reason"] == "invalid_pool_distribution"
    assert study.initialize(state, config) == result


def test_fit_learns_cost_inclusive_net_gain_not_just_selection_advantage():
    data = fixture()
    for p in data["points"]:
        p["branches"]["random_reduced"]["rewards"] = {"q0": .4, "q1": .4}
        p["branches"]["selection_reduced"]["rewards"] = {"q0": .45, "q1": .45}
    result = study.analyze(data, study.fit(data))
    assert all(r["action_advantage"] > 0 and r["net_selection_gain"] < 0 for r in result["rows"])
    assert all(r["action"] == "random" for r in result["rows"])


def test_gate_fit_at_a_different_budget_cannot_be_replayed():
    data = fixture()
    fitted = study.fit(data)
    fitted["budget_gpu_seconds"] *= 2
    fitted["model_id"] = sg.fingerprint({k: v for k, v in fitted.items() if k != "model_id"})
    with pytest.raises(ValueError, match="differ"):
        study.analyze(data, fitted)
