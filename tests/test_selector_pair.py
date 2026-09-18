import copy
import math

import pytest

import selection_gate as core
import selector_pair as pair


def allocation(key, phase, start, seconds, *, ledger="deployment", rc=0):
    common = {"event_id": key, "phase": phase, "ledger": ledger, "gpus": 4,
              "gpu_type": "H100", "host": "node-a"}
    return [{**common, "state": "started", "time": start},
            {**common, "state": "finished", "time": start+seconds, "seconds": seconds,
             "allocated_gpu_seconds": 4*seconds, "exit_code": rc}]


def points(cost=100., rewards=(.2, .4, .3)):
    return [{"updates": i*5, "reward": reward, "gpu_seconds": i*cost,
             "training_gpu_seconds": i*cost*.8, "scoring_gpu_seconds": i*cost*.2,
             "other_gpu_seconds": 0.} for i, reward in enumerate(rewards)]


def features(step=25):
    return {"recent_reward": .2, "recent_active_fraction": .7,
            "success_rate_std": .1, "log_prefix_updates": math.log1p(step)}


def development():
    return [{"seed": seed, "step": step, "role": "development", "protocol_id": "p",
             "state_id": f"{seed}-{step}", "features": features(step),
             "contrast": {"status": "observed", "h_gpu_seconds": 100.-step}}
            for seed in pair.DEV_SEEDS for step in pair.STEPS]


def contract():
    return {"config": {"seed": 0, "drift": 25}, "evaluation": {"val": ["q"]},
            "eval_k": 8, "eval_seed": 10, "n": 400, "budget_gpu_seconds": 1000., "max_steps": 100,
            "source_hashes": {"policy/adapter": "a", "policy/optimizer.pt": "o", "stats": "s"},
            "scope": {"selector": "fresh_r", "dataset": "math500", "gpu_type": "H100"},
            "selected_prefix": {"certificate_sha256": "c"}}


def test_matching_ignores_only_selector_and_paths_not_optimizer_or_data():
    a, b = contract(), contract()
    b["scope"]["selector"] = "difficulty"
    assert pair.matched_state([a, b])
    for field in ("policy/optimizer.pt", "policy/adapter", "stats"):
        changed = copy.deepcopy(b)
        changed["source_hashes"][field] = "changed"
        with pytest.raises(ValueError, match="same model/optimizer"):
            pair.matched_state([a, changed])
    b["evaluation"] = {"val": ["different"]}
    with pytest.raises(ValueError):
        pair.matched_state([a, b])


def test_actual_cost_includes_failed_work_scoring_and_startup_not_reporting():
    events = [*allocation("score", "fresh-r-candidate", 0, 10, ledger="reporting"),
              *allocation("fail", "train", 20, 20, rc=1),
              *allocation("retry", "train", 50, 100),
              *allocation("eval", "evaluate", 200, 200, ledger="reporting")]
    cost = pair.cost_at_checkpoint(events, {"event_id": "retry", "time": 75})
    assert cost["gpu_seconds"] == 40+80+100
    assert cost["scoring_gpu_seconds"] == 40
    assert cost["training_gpu_seconds"] == 180
    assert pair.cost_at_checkpoint(events, {"event_id": "retry", "time": 75}, final=True)["gpu_seconds"] == 520


def test_future_failed_attempt_is_not_charged_to_an_earlier_checkpoint():
    events = [*allocation("first", "train", 0, 100), *allocation("later", "train", 200, 100, rc=1)]
    assert pair.cost_at_checkpoint(events, {"event_id": "first", "time": 20})["gpu_seconds"] == 80


@pytest.mark.parametrize("change", ["open", "clock", "timestamp", "unknown", "overlap"])
def test_unknown_or_invalid_cost_never_becomes_zero(change):
    events = allocation("t", "train", 100, 100)
    receipt = {"event_id": "t", "time": 150}
    if change == "open":
        events.pop()
    elif change == "clock":
        events[-1]["time"] += 10
    elif change == "timestamp":
        receipt["time"] = 201
    elif change == "unknown":
        receipt["event_id"] = "missing"
    else:
        other = allocation("other", "train", 110, 80)
        events = [events[0], other[0], other[1], events[1]]
    with pytest.raises(ValueError):
        pair.cost_at_checkpoint(events, receipt)


def test_first_observed_crossing_no_linear_interpolation_and_nonmonotone_reward():
    hit = pair.crossing(points(), .35, diagnosis=7.)
    assert hit["updates"] == 5  # Not the interpolated 3.75 updates.
    assert hit["gpu_seconds"] == 107
    assert hit["diagnostic_gpu_seconds"] == 7
    assert hit["status"] == "reached"  # Later reward regression does not undo first crossing.


def test_censoring_and_target_already_reached_are_not_zero_labels():
    censored = pair.crossing(points(), .5)
    assert censored["status"] == "right_censored" and censored["gpu_seconds"] is None
    assert censored["observed_through_gpu_seconds"] == 200
    invalid = pair.crossing(points(), .2)
    assert invalid["status"] == "target_not_above_parent" and invalid["gpu_seconds"] is None
    assert pair.contrast(censored, pair.crossing(points(), .35))["h_gpu_seconds"] is None
    assert pair.contrast(invalid, censored)["status"] == "ineligible"


def test_zero_update_censoring_retains_real_spent_allocation():
    result = pair.crossing(points()[:1], .35, diagnosis=5., observed_gpu_seconds=900.)
    assert result["status"] == "right_censored" and result["gpu_seconds"] is None
    assert result["observed_through_updates"] == 0
    assert result["observed_through_gpu_seconds"] == 905.


@pytest.mark.parametrize("field,value", [("updates", 0), ("gpu_seconds", -1.), ("reward", float("nan"))])
def test_invalid_curve_rejected(field, value):
    curve = points()
    curve[1][field] = value
    with pytest.raises(ValueError):
        pair.crossing(curve, .35)


def test_h_in_gpu_seconds_and_expensive_gradient_can_lose_despite_learning_faster():
    g = pair.crossing(points(cost=300), .35)
    d = pair.crossing(points(cost=200, rewards=(.2, .3, .4)), .35)
    assert pair.contrast(g, d)["h_gpu_seconds"] == 100
    g["gpu_seconds"] = 500
    assert g["updates"] < d["updates"]
    assert pair.contrast(g, d) == {"status": "observed", "h_gpu_seconds": -100., "preferred": "cached"}


def test_frozen_ridge_reproducible_and_only_predecision_features():
    model = pair.fit(development(), "p")
    assert model == pair.fit(development(), "p")
    choice = pair.choose(model, features(), seed=3, state_id="test", protocol_id="p")
    assert choice["selector"] == "on_policy" and choice["h_hat_gpu_seconds"] > 0
    with pytest.raises(ValueError, match="unregistered"):
        pair.choose(model, {**features(), "future_reward": .9}, seed=3, state_id="test", protocol_id="p")


@pytest.mark.parametrize("change", ["test_seed", "test_role", "missing", "censored", "protocol", "nan"])
def test_fit_refuses_leakage_or_successful_states_only(change):
    rows = development()
    if change == "test_seed":
        rows[0]["seed"] = 3
    elif change == "test_role":
        rows[0]["role"] = "test"
    elif change == "missing":
        rows.pop()
    elif change == "censored":
        rows[0]["contrast"] = {"status": "censored", "h_gpu_seconds": None}
    elif change == "protocol":
        rows[0]["protocol_id"] = "changed"
    else:
        rows[0]["contrast"]["h_gpu_seconds"] = float("nan")
    with pytest.raises(ValueError):
        pair.fit(rows, "p")


@pytest.mark.parametrize("seed,state_id,protocol_id", [(0, "test", "p"), (3, "0-25", "p"), (3, "test", "changed")])
def test_prediction_rejects_development_state_and_changed_protocol(seed, state_id, protocol_id):
    with pytest.raises(ValueError):
        pair.choose(pair.fit(development(), "p"), features(), seed=seed, state_id=state_id, protocol_id=protocol_id)


def test_model_tampering_rejected():
    model = pair.fit(development(), "p")
    model["intercept"] = 1e8
    with pytest.raises(ValueError, match="invalid frozen"):
        pair.validate_model(model)


def test_actual_adaptive_cost_not_copied_from_winning_control():
    hit = lambda cost: {"status": "reached", "gpu_seconds": cost}
    row = {"decision": {"selector": "on_policy"}, "crossings": {
        "on_policy": hit(100), "cached": hit(200), "adaptive": hit(117), "random": hit(300)}}
    audit = pair.audit(row)
    assert audit["regret_gpu_seconds"] == 0
    assert audit["adaptive_saving_vs_on_policy"] == -17
    assert audit["adaptive_saving_vs_cached"] == 83
    row["crossings"]["adaptive"] = {"status": "right_censored", "gpu_seconds": None}
    assert pair.audit(row)["adaptive_saving_vs_cached"] is None


def test_summary_counts_eligible_states_without_treating_them_as_independent_seeds():
    keys = ("regret_gpu_seconds", "adaptive_saving_vs_on_policy", "adaptive_saving_vs_cached", "adaptive_saving_vs_random")
    rows = [{"seed": 3, "audit": {key: None for key in keys}}]
    summary = pair.summarize(rows)
    assert summary["complete_test_states"] == 1
    assert summary["seed_means"]["3"][keys[0]]["mean"] is None
    assert summary["seed_means"]["4"][keys[0]]["observed_states"] == 0
    assert not summary["optimal_switch_time_claim"]
