import concurrent.futures
import copy
from dataclasses import replace

import pytest

import selection_gate as sg


def scope():
    return {"model": "test-model", "dataset": "test-pool", "selector": "test-selector",
            "verifier": "test-verifier", "pool_sha256": "test-pool-hash", "gpu_type": "test-GPU"}


def model(value=.1):
    result = {"schema": sg.SCHEMA, "target": sg.TARGET, "scope": scope(),
              "budget_gpu_seconds": 1000.,
              "features": list(sg.FEATURES), "nodes": [{"value": value}],
              "feature_ranges": {f: [0., 1.] for f in sg.FEATURES},
              "fit_trajectories": ["train-a", "train-b"], "data_kind": "synthetic"}
    result["model_id"] = sg.fingerprint(result)
    return result


def config(m=None, **kwargs):
    m = model() if m is None else m
    return sg.GateConfig(scope(), 1000., 50., start_step=100, model_id=m["model_id"], data_kind="synthetic", **kwargs)


def observation(**kwargs):
    return {"event_id": "event-0", "step": 100, "feature_step": 100, "scope": scope(),
            "total_used": 10., "measurement_used_now": 2.,
            "measurement_wall_now": .1, "full_pool_coverage": True,
            "features": {f: .5 for f in sg.FEATURES}, **kwargs}


def test_initial_choice_stays_frozen_throughout_training():
    cfg, m = config(threshold=.05), model()
    state, result = sg.decide(cfg, sg.GateState(), observation(), m)
    assert (result["action"], state.checks, result["next_check_step"]) == ("select", 1, None)
    state, result = sg.decide(cfg, state, observation(step=500, feature_step=500, total_used=25.,
                                                     measurement_used_now=0., measurement_wall_now=0.), None)
    assert result["reason"] == "decision_frozen" and state.checks == 1
    state, result = sg.decide(cfg, state, observation(step=600, total_used=30., selector_valid=False,
                                                     measurement_used_now=0., measurement_wall_now=0.), m)
    assert result["action"] == "random" and result["reason"] == "invalid_selector"
    state, result = sg.decide(cfg, state, observation(step=700, total_used=40., measurement_used_now=0., measurement_wall_now=0.), m)
    assert result["reason"] == "decision_frozen" and state.checks == 1
    assert state.measurement_used == 2., "the single measurement remains charged"


@pytest.mark.parametrize("value", [0., -.1, .05])
def test_threshold_and_ties_fall_back(value):
    m = model(value)
    state, result = sg.decide(config(m, threshold=.05), sg.GateState(), observation(), m)
    assert state.mode == "random" and result["reason"] == "predicted_no_useful_gain"


def test_total_cap_records_overshoot_and_stops_training():
    state, result = sg.decide(config(), sg.GateState(), observation(total_used=1005.), model())
    assert state.mode == "done" and result["overshoot_gpu_seconds"] == 5.
    assert result["remaining_gpu_seconds"] == 0


def test_measurement_cap_stops_selection_only():
    cfg = replace(config(), measurement_gpu_seconds=1.)
    state, result = sg.decide(cfg, sg.GateState(), observation(), model())
    assert state.mode == "random" and result["remaining_gpu_seconds"] == 990.
    assert state.measurement_used == 2.
    assert not sg.measurement_allowed(cfg, state, reserved_gpu_seconds=0., step=110)


def test_preflight_permits_only_the_initial_measurement():
    cfg = config()
    assert sg.measurement_allowed(cfg, sg.GateState(), reserved_gpu_seconds=1., step=100)
    assert not sg.measurement_allowed(cfg, sg.GateState(), reserved_gpu_seconds=51., step=100)
    assert not sg.measurement_allowed(cfg, sg.GateState(), reserved_gpu_seconds=0., step=110)
    assert not sg.measurement_allowed(cfg, sg.GateState(), reserved_gpu_seconds=0., step=100, reserved_wall_seconds=31.)
    state, _ = sg.decide(cfg, sg.GateState(), observation(), model())
    assert not sg.measurement_allowed(cfg, state, reserved_gpu_seconds=1., step=110)
    with pytest.raises(ValueError, match="measurement repeated"):
        sg.decide(cfg, state, observation(step=110, feature_step=110, total_used=20.), model())


@pytest.mark.parametrize("change,reason", [
    ({"feature_step": 101}, "invalid_feature_time"),
    ({"features": {}}, "invalid_or_unsupported_features"),
    ({"features": {f: float("nan") for f in sg.FEATURES}}, "invalid_or_unsupported_features"),
    ({"measurement_status": "probe_timeout"}, "probe_timeout"),
    ({"scope": {**scope(), "dataset": "different"}}, "unsupported_scope"),
    ({"full_pool_coverage": False}, "incomplete_pool_distribution"),
    ({"measurement_wall_now": 31.}, "measurement_wall_budget_exhausted"),
])
def test_invalid_or_missing_measurement_falls_back(change, reason):
    state, result = sg.decide(config(), sg.GateState(), observation(**change), model())
    assert state.mode == "random" and result["reason"] == reason


def test_random_can_start_without_any_scores_or_model():
    state, result = sg.decide(replace(config(), model_id=None), sg.GateState(),
                              observation(features={}, measurement_used_now=0.), None)
    assert state.mode == "random" and result["reason"] == "no_fitted_model"


def test_synthetic_predictor_cannot_control_observed_training():
    _, result = sg.decide(replace(config(), data_kind="observed"), sg.GateState(), observation(), model())
    assert result["action"] == "random" and result["reason"] == "unsupported_model"


def test_common_corruption_is_not_disguised_as_random():
    with pytest.raises(ValueError, match="common training inputs"):
        sg.decide(config(), sg.GateState(), observation(common_inputs_valid=False), model())


@pytest.mark.parametrize("change", [{"total_used": float("nan")}, {"total_used": -1.},
                                     {"step": True}, {"measurement_used_now": 11.}])
def test_invalid_budget_or_step_is_rejected(change):
    with pytest.raises(ValueError):
        sg.decide(config(), sg.GateState(), observation(**change), model())


def test_out_of_order_steps_costs_and_early_paid_checks():
    state, _ = sg.decide(config(), sg.GateState(), observation(), model())
    for obs in (observation(step=99, total_used=20.), observation(step=110, total_used=5.),
                observation(step=105, total_used=20.)):
        with pytest.raises(ValueError):
            sg.decide(config(), state, obs, model())


def test_durable_duplicate_requests_bill_once_under_concurrency(tmp_path):
    path, cfg, obs, m = tmp_path / "state.json", config(), observation(), model()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: sg.durable_decide(path, cfg, obs, m), range(12)))
    assert all(r == results[0] for r in results)
    saved = sg.read(path)
    assert saved["state"]["checks"] == 1 and saved["state"]["measurement_used"] == 2.
    assert len(saved["events"]) == 1
    with pytest.raises(ValueError, match="changed inputs"):
        sg.durable_decide(path, cfg, observation(total_used=12.), m)
    with pytest.raises(ValueError, match="contract changed"):
        sg.durable_decide(path, replace(cfg, threshold=.2), obs, m)


def test_old_acceptance_cannot_be_replayed_after_stopping(tmp_path):
    path, cfg, m = tmp_path / "state.json", config(), model()
    sg.durable_decide(path, cfg, observation(), m)
    second = observation(event_id="event-1", step=110, feature_step=110, total_used=20., selector_valid=False)
    second.update(measurement_used_now=0., measurement_wall_now=0.)
    assert sg.durable_decide(path, cfg, second, m)["action"] == "random"
    with pytest.raises(ValueError, match="stale decision"):
        sg.durable_decide(path, cfg, observation(), m)


def test_tampered_tree_is_rejected():
    m = model()
    m["nodes"][0]["value"] = .2
    with pytest.raises(ValueError, match="hash changed"):
        sg.predict(m, {f: .5 for f in sg.FEATURES})
    m = model()
    m["nodes"] = [{"feature": "success_rate", "threshold": .5, "left": 0, "right": 0}]
    m["model_id"] = sg.fingerprint({k: v for k, v in m.items() if k != "model_id"})
    with pytest.raises(ValueError, match="tree structure"):
        sg.predict(m, {f: .5 for f in sg.FEATURES})


def cost(event, seconds, gpus, code=0, ledger="deployment"):
    base = {"event_id": event, "ledger": ledger, "gpus": gpus, "gpu_type": "test-GPU", "phase": "probe"}
    return [{**base, "state": "started"},
            {**base, "state": "finished", "seconds": seconds, "allocated_gpu_seconds": seconds*gpus, "exit_code": code}]


def test_cost_includes_failed_probes_and_separates_ledgers():
    rows = cost("success", 10., 4)+cost("failed", 5., 4, 1)+cost("fit", 9., 0, ledger="research")
    rows += copy.deepcopy(rows[-2:])
    result = sg.cost_summary(rows)
    assert result["complete"]
    assert result["ledgers"]["deployment"] == {"gpu_seconds": 60., "wall_seconds": 15., "failed_events": 1}
    assert result["ledgers"]["research"]["wall_seconds"] == 9.


def test_cost_incomplete_and_inconsistent_records():
    assert not sg.cost_summary(cost("pending", 5., 4)[:1])["complete"]
    assert not sg.cost_summary(cost("lost", 5., 4)[1:])["complete"]
    rows = cost("bad", 5., 4)
    rows[1]["allocated_gpu_seconds"] = 15.
    with pytest.raises(ValueError, match="does not match"):
        sg.cost_summary(rows)
    rows = cost("a", 5., 4)+cost("b", 1., 4)
    rows[-2]["gpu_type"] = rows[-1]["gpu_type"] = "different-GPU"
    with pytest.raises(ValueError, match="mixed GPU types"):
        sg.cost_summary(rows)
    rows = cost("a", 5., 4)
    rows.append({**rows[-1], "seconds": 6., "allocated_gpu_seconds": 24.})
    with pytest.raises(ValueError, match="conflicting duplicate"):
        sg.cost_summary(rows)


def test_five_seeds_cannot_produce_a_fake_positive_certificate():
    result = sg.certificate([1.]*5, family_size=3)
    assert result["radius"] == pytest.approx(1.3838340569276426)
    assert result["vacuous"] and not result["accepted"]
    assert sg.certificate([.9]*1000)["accepted"]
