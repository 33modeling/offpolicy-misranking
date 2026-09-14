import copy
import json
import math

import numpy as np
import pytest

import selection_gate as core
import selection_switch as rule
import selection_switch_score as score


def development():
    scope = {"model": "model", "dataset": "math500", "selector": "fresh_r", "verifier": "math_verify",
             "pool_sha256": "pool", "gpu_type": "H100"}
    return [{"seed": seed, "step": step, "role": "development", "complete": True, "scope": scope,
             "trajectory_id": f"model:seed-{seed}", "parent": [f"p-{seed}-{step}", f"o-{seed}-{step}"],
             "budget_gpu_seconds": 1000., "features": {"recent_reward": .2+.1*seed,
                 "recent_active_fraction": .7-step/200, "success_rate_std": .2,
                 "log_prefix_updates": math.log1p(step)},
             "means": {"selection_reduced": .5+(50-step)/1000, "random_reduced": .5}}
            for seed in rule.DEV_SEEDS for step in rule.STEPS]


def test_registered_ridge_matches_weighted_normal_equations():
    rows = development()
    model = rule.fit(rows)
    x = np.array([rule.feature_vector(r["features"]) for r in rows])
    y = np.array([r["means"]["selection_reduced"]-.5 for r in rows])
    reg = model["ridge"]
    z = (x-reg["mean"])/reg["scale"]
    expected = np.linalg.solve(z.T@z/3+np.eye(4), z.T@(y-y.mean())/3)
    assert reg["coef"] == pytest.approx(expected)
    assert len(model["checkpoint_only"]["coef"]) == 1
    assert rule.choose(model, rows[0]["features"])["action"] == "select"
    assert rule.choose(model, rows[2]["features"])["action"] == "random"


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "test", "failed", "scope", "budget", "free_label", "extra_feature", "nan"])
def test_fit_rejects_invalid_or_leaked_study(mutation):
    rows = development()
    if mutation == "missing": rows.pop()
    elif mutation == "duplicate": rows[-1] = copy.deepcopy(rows[0])
    elif mutation == "test": rows[0]["role"] = "test"
    elif mutation == "failed": rows[0]["complete"] = False
    elif mutation == "scope": rows[0]["scope"] = {**rows[0]["scope"], "selector": "low_order"}
    elif mutation == "budget": rows[0]["budget_gpu_seconds"] = 1001.
    elif mutation == "free_label": rows[0]["means"]["random_full"] = .9
    elif mutation == "extra_feature": rows[0]["features"]["future_reward"] = .9
    else: rows[0]["features"]["recent_reward"] = float("nan")
    with pytest.raises(ValueError): rule.fit(rows)


def test_frozen_model_and_test_identity_are_enforced():
    model = rule.fit(development())
    rule.check_scope(model, model["scope"], 1000., trajectory="model:seed-3", parent=("new", "optim"), role="test", observed=True)
    for kwargs in ({"trajectory": "model:seed-0", "parent": ("new", "optim"), "role": "test"},
                   {"trajectory": "model:seed-3", "parent": model["fit_parent_states"][0], "role": "test"},
                   {"trajectory": "model:seed-3", "parent": ("new", "optim"), "role": "development"}):
        with pytest.raises(ValueError): rule.check_scope(model, model["scope"], 1000., **kwargs)
    model["ridge"]["coef"][0] += 1
    with pytest.raises(ValueError): rule.choose(model, development()[0]["features"])


def test_tie_switches_and_does_not_claim_population_optimality():
    rows = development()
    for r in rows: r["means"]["selection_reduced"] = .5
    model = rule.fit(rows)
    assert rule.choose(model, rows[0]["features"])["action"] == "random"


def cache_and_stats(tmp_path):
    cache, stats = tmp_path / "cache.jsonl", tmp_path / "stats.jsonl"
    cache.write_text("".join(json.dumps({"prompt_idx": i, "rollout_idx": j, "reward": int(j <= i)})+"\n"
                            for i in range(4) for j in range(8)))
    stats.write_text("".join(json.dumps({"step": t, "groups": 4, "nonzero_advantage_groups": 2,
                                        "reward_mean": .5, "step_seconds": 3.})+"\n" for t in range(1, 26)))
    return cache, stats


def test_one_scan_uses_only_registered_past_features(tmp_path):
    cache, stats = cache_and_stats(tmp_path)
    result = rule.measure(cache, stats=stats, step=25, prompts=4)
    assert set(result["features"]) == set(rule.FEATURES)
    assert result["features"]["recent_reward"] == .5
    assert result["features"]["log_prefix_updates"] == math.log1p(25)
    assert result["source_sha256"] and result["stats_sha256"]
    assert "difficulty_indices" not in result


@pytest.mark.parametrize("kind", ["future", "missing", "duplicate", "missing_cache", "window", "unregistered_step"])
def test_measurement_rejects_invalid_window_or_cache(tmp_path, kind):
    cache, stats = cache_and_stats(tmp_path)
    if kind == "future": stats.write_text(stats.read_text()+json.dumps({"step": 26})+"\n")
    elif kind == "missing": stats.write_text("\n".join(stats.read_text().splitlines()[:-1]))
    elif kind == "duplicate": stats.write_text(stats.read_text()+stats.read_text().splitlines()[-1]+"\n")
    elif kind == "missing_cache": cache.write_text("\n".join(cache.read_text().splitlines()[:-1]))
    with pytest.raises(ValueError):
        rule.measure(cache, stats=stats, step=26 if kind == "unregistered_step" else 25, prompts=4,
                     window=19 if kind == "window" else 20)


def test_wrong_switch_and_wrong_retention_measure_actual_reward_loss():
    means = {"selection_reduced": .6, "random_reduced": .5, "gated": .49, "random_full": .51, "selection_full": .61}
    a = rule.decision_audit(means, "random")
    assert a["wrong_switch_loss"] == pytest.approx(.1)
    assert a["gate_minus_random"] == pytest.approx(-.02)
    means["selection_reduced"] = .4
    a = rule.decision_audit(means, "select")
    assert a["wrong_retention_loss"] == pytest.approx(.1)
    assert a["wrong_switch_loss"] == 0


def test_fresh_r_exactly_matches_original_group_score():
    import torch
    from experiment import score_oracle_microgroups
    torch.manual_seed(7)
    stack, directions = torch.randn(8, 13), torch.randn(3, 13)
    _, original = score_oracle_microgroups(stack, *directions)
    assert score.matched_score(stack[:2], directions[0]) == original["r"]
    with pytest.raises(ValueError): score.matched_score(stack[:4], directions[0])


def test_fresh_r_uses_only_eight_candidates_and_ranking_validation():
    cfg = {"fresh_k": 32, "micro_group": 4, "behavior_k": 8, "val_k": 8}
    assert score.layout(cfg, {"val": [{}]*100})["ranking_validation_n"] == 50
    assert score.layout(cfg, {"val": [{}]*100})["candidate_k"] == 8
    with pytest.raises(ValueError): score.layout({**cfg, "micro_group": 8}, {"val": [{}]*100})


def test_validation_merge_weights_prompts_not_shards(tmp_path):
    import selection_gate_gpu as base
    core.atomic_json(tmp_path / "prompts.json", {"train": [{}]*40, "val": [{}]*100})
    cfg = {"fresh_k": 32, "micro_group": 4, "behavior_k": 8, "val_k": 8, "proj_dim": 2}
    core.atomic_json(tmp_path / "scoring.json", {"config": cfg, "prompts": str(tmp_path / "prompts.json")})
    for shard in range(4):
        part = {str(i): [float(i), 2.] for i in range(50*shard//4, 50*(shard+1)//4)}
        core.atomic_json(tmp_path / f"validation-{shard}.json", part)
        core.atomic_json(tmp_path / f"validation-{shard}.done.json", {"contract_sha256": base.digest(tmp_path / "scoring.json"),
            "stage": "validation", "shard": shard, "sha256": base.digest(tmp_path / f"validation-{shard}.json")})
    score.merge(tmp_path, "validation")
    assert core.read(tmp_path / "direction.json")["direction"] == [24.5, 2.]


def test_seed_clustering_does_not_count_checkpoints_as_seeds():
    rows = [{"seed": s, "audit": {k: .1 for k in ("delta", "decision_regret", "gate_minus_random", "gate_minus_selection")}}
            for s in (3, 4) for _ in range(3)]
    report = rule.clustered_summary(rows)
    assert report["independent_seeds"] == 2
    assert report["optimal_switch_time_claim"] is False


def test_figures_render_observed_rows_without_connecting_branch_states(tmp_path):
    pytest.importorskip("matplotlib")
    from selection_switch_plot import plot
    from PIL import Image
    means = {"selection_reduced": .6, "random_reduced": .5, "gated": .58, "random_full": .51, "selection_full": .61}
    rows = [{"seed": s, "step": t, "means": means, "audit": rule.decision_audit(means, "select"),
             "checkpoint_only_audit": rule.decision_audit(means, "random"), "intended_action": "select",
             "fallback": False, "measurement_gpu_seconds": 1.2}
            for s in (3, 4) for t in rule.STEPS]
    core.atomic_json(tmp_path / "test-report.json", {"rows": rows})
    plot(tmp_path)
    for name in ("switch-outcomes", "switch-regret-cost"):
        assert (tmp_path / "figures" / f"{name}.pdf").stat().st_size > 1000
        pixels = np.asarray(Image.open(tmp_path / "figures" / f"{name}.png"))
        assert pixels.std() > 5 and pixels.shape[0] > pixels.shape[1]*.8
