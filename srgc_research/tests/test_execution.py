import copy
import hashlib
import json
from unittest.mock import patch

import pytest

from srgc_rebuttal.cost_ledger import PhaseLedger
from srgc_rebuttal.runtime import atomic_json
from srgc_rebuttal.srgc import Config
from srgc_rebuttal.timing import torch_meter
from srgc_rebuttal.torch_backend import TorchBackend
from srgc_research.cli import main, status
from srgc_research.design import Condition
from srgc_research.report import collect, coverage
from srgc_research.storage import (
    complete,
    publish_anchor,
    validate_costs,
    verify_anchor,
)
from srgc_research.study import NestedEngine, Trajectory
from srgc_research.tests.test_research import equal_tree, study_backend
from srgc_research.worker import execute


@pytest.fixture
def data():
    from srgc_research.tests import test_research
    return test_research.data.__wrapped__()


def manifest_for(data):
    return {"protocol": "srgc-literature-studies-v1", "dataset": "math", "seed": 5,
            "input_sha256": hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest(),
            "implementation_sha256": "test", "source_plan_sha256": "test", "plan": {"projection_dim": 16},
            "evaluation_ids": data["evaluation_ids"], "candidate_ids": data["candidate_ids"],
            "initial_state": "synthetic-test-not-experiment"}


def attach_meter(backend, folder, condition):
    backend.cost_meter = torch_meter(PhaseLedger(folder / condition.key / "cost-receipts").record, cuda=False)
    backend.rollout_cache_root = folder / condition.key / "rollouts"


@pytest.mark.parametrize("arm", ["on_policy", "switch", "sr", "random", "lesser", "arcus_adapted"])
def test_execute_checkpoint_endpoint_recovery(data, tmp_path, arm):
    b = study_backend()
    b.records = data["records"]
    initial = b.state_dict()
    folder = tmp_path / "math/seed-5"
    manifest = manifest_for(data)
    atomic_json(folder / "manifest.json", manifest)
    with patch.object(TorchBackend, "_rollout", side_effect=b.generate_test_rollout):
        cache = Condition("cache", "cache", updates=0)
        attach_meter(b, folder, cache)
        execute(folder, cache, manifest, copy.deepcopy(data), b)
        assert complete(folder, manifest, cache)
        condition = Condition("test-" + arm, "trajectory", arm=arm, updates=2)
        b.load_state_dict(initial)
        attach_meter(b, folder, condition)
        value = execute(folder, condition, manifest, copy.deepcopy(data), b)
        assert complete(folder, manifest, condition)
        assert value["cost_measurement_complete"] is True
        assert len(value["result"]["curve"]) == 2
        assert coverage(value["result"]["curve"][-1])["pass_at_k"]["1"] == .5
        expected = b.state_dict()
        (folder / condition.key / "endpoint.json").unlink()
        b.load_state_dict(initial)
        attach_meter(b, folder, condition)
        recovered = execute(folder, condition, manifest, copy.deepcopy(data), b)
        equal_tree(expected, b.state_dict())
        assert recovered["result"] == value["result"]
        assert complete(folder, manifest, condition)


def test_feature_audit_shared_rollouts_and_no_training(data, tmp_path):
    b = study_backend()
    b.records = data["records"]
    initial = b.state_dict()
    condition = Condition("n01-features", "features", updates=0)
    manifest = manifest_for(data)
    attach_meter(b, tmp_path, condition)
    with patch.object(TorchBackend, "_rollout", side_effect=b.generate_test_rollout) as generation:
        result = execute(tmp_path, condition, manifest, data, b)
        assert generation.call_count == 40 + len(data["ranking_validation_ids"])
    assert result["result"]["same_rollouts"]
    assert result["cost_role"] == "diagnostic"
    equal_tree(initial, b.state_dict())


def test_anchor_hash_and_wrong_stage_are_rejected(data, tmp_path):
    b = study_backend()
    config = Config(seed=5, projection_dim=16)
    engine = NestedEngine(b, data["candidate_ids"], data["ranking_validation_ids"], data["cached_rewards"], config=config)
    manifest = manifest_for(data)
    publish_anchor(tmp_path, manifest, engine.state_dict())
    path = verify_anchor(tmp_path, manifest, 0)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash differs"):
        verify_anchor(tmp_path, manifest, 0)


def test_incomplete_cost_never_becomes_exact_zero():
    costs = {"known_gpu_seconds": {"selection_gpu_seconds": 0., "training_gpu_seconds": 0.},
             "complete": False, "total_gpu_seconds": None, "unfinished_phases": [{"id": "interrupted"}]}
    validate_costs(costs)
    costs["total_gpu_seconds"] = 0.
    with pytest.raises(ValueError):
        validate_costs(costs)


def test_switch_stops_scoring_after_confirmed_transition(data):
    b = study_backend()
    b.records = data["records"]
    config = Config(seed=5, projection_dim=16, selection_interval=1, check_interval=1, first_check=1)
    condition = Condition("test-switch", "trajectory", arm="switch", updates=4)
    with patch.object(TorchBackend, "_rollout", side_effect=b.generate_test_rollout), \
            patch("srgc_rebuttal.srgc.gradient_contrast", return_value=-1.), \
            patch.object(b, "score_gradients", wraps=b.score_gradients) as scoring:
        study = Trajectory(b, data, config, condition)
        while study.step < 3:
            study.advance()
        assert study.engine.switched_at == 2
        calls = scoring.call_count
        while not study.done:
            study.advance()
        assert scoring.call_count == calls


def test_no_results_export_and_status_are_explicit(tmp_path):
    report = collect(tmp_path, ("math", "mbpp"), "n01")
    assert report["pending"] and not report["errors"]
    assert json.loads((tmp_path / "exports/math-mbpp-n01-results.json").read_text()) == report
    assert all(r["status"] == "not-started" for r in status(tmp_path, ("math",), "n01"))


def test_n08_accepts_only_matching_dataset_and_report_actions(tmp_path):
    source = tmp_path / "results.json"
    atomic_json(source, {"dataset": "mbpp", "source_results": {}})
    with pytest.raises(SystemExit):
        main(["math", "n08", "results", str(source)])
    with pytest.raises(SystemExit):
        main(["mbpp", "n08", "run", str(source)])


def test_results_collect_raw_json_and_reject_corruption(data, tmp_path):
    b = study_backend()
    b.records = data["records"]
    folder = tmp_path / "math/seed-5"
    manifest = manifest_for(data)
    atomic_json(folder / "manifest.json", manifest)
    condition = Condition("test-random", "trajectory", arm="random", updates=1)
    attach_meter(b, folder, condition)
    with patch.object(TorchBackend, "_rollout", side_effect=b.generate_test_rollout):
        execute(folder, condition, manifest, copy.deepcopy(data), b)
    with patch("srgc_research.report.tasks", return_value=[condition]):
        report = collect(tmp_path, ("math",), "n01")
        assert len(report["source_results"]) == len(report["rows"]) == 1
        assert len(report["pending"]) == 4 and not report["errors"]
        assert report["performance_summary"][0]["completed_seeds"] == [5]
        assert not report["performance_summary"][0]["all_seeds_complete"]
        endpoint = folder / condition.key / "endpoint.json"
        changed = json.loads(endpoint.read_text())
        changed["result"]["curve"][-1]["per_question_reward"].pop(data["evaluation_ids"][0])
        atomic_json(endpoint, changed)
        report = collect(tmp_path, ("math",), "n01")
        assert report["errors"] and not report["rows"]


def test_scale_conditions_and_shared_baselines(data):
    from srgc_research.design import conditions
    from srgc_research.study import make_engine
    b = study_backend()
    data["candidate_ids"] = [f"p{i}" for i in range(400)]
    data["ranking_validation_ids"] = ["ref0", "ref1"]
    data["cached_rewards"] = {i: [0, 1] * 4 for i in data["candidate_ids"]}
    draws = {}
    for condition in conditions("n05"):
        engine = make_engine(b, data, Config(seed=5, projection_dim=16), condition)
        ids = engine._draw_candidates()
        assert len(ids) == len(set(ids)) == condition.candidates
        if condition.arm in {"random", "sr"}:
            assert len(engine._training_batch(condition.arm, ids)) == condition.batch
        draws[condition.candidates] = ids
    assert draws[40] == draws[80][:40] == draws[160][:40]
    switch = next(c for c in conditions("n01") if c.arm == "switch")
    assert switch in conditions("n06")
    baseline = next(c for c in conditions("n01") if c.arm == "on_policy" and c.kind == "trajectory")
    assert baseline in conditions("n05")


def test_budget_readout_is_cold_and_does_not_interpolate():
    from srgc_research.report import budget_readout
    curve = [{"update": i, "reward": r, "costs_to_checkpoint": {"complete": True,
        "known_gpu_seconds": {"selection_gpu_seconds": c, "training_gpu_seconds": c}}}
        for i, r, c in [(0, .2, 0), (25, .4, 10), (50, .6, 20)]]
    report = budget_readout(curve, target=.5, budgets=(24, 25, 45), cache_cost=5)
    assert report["first_observed_target"] == {"gpu_seconds": 45, "update": 50}
    assert [r["last_observed"]["update"] for r in report["at_gpu_budgets"]] == [0, 25, 50]
    assert budget_readout(curve, .5, cache_cost=None)["first_observed_target"] is None


def test_new_export_n08_pairs_shared_baseline_and_variant():
    from srgc_research.report import analyze_archive
    common = {"seed": 5, "dataset": "math", "status": "complete", "source_plan_sha256": "p",
              "input_sha256": "i", "implementation_sha256": "c", "protocol": "v1",
              "result": {"curve": [{"per_question_reward": {"x": .5}, "reward": .5}]}}
    bundle = {"scope": "n01", "source_results": {str(i): {**common, "condition": {
        "kind": "trajectory", "key": arm, "updates": 275}} for i, arm in enumerate(("baseline-switch", "n01-lesser"))}}
    result = analyze_archive(bundle)
    assert len(result["comparisons"]) == 1 and not result["errors"]
    assert result["summary"][0]["paired_seeds"] == [5]


def test_results_reject_changed_cache_even_with_valid_endpoint(data, tmp_path):
    b = study_backend()
    b.records = data["records"]
    folder = tmp_path / "math/seed-5"
    manifest = manifest_for(data)
    atomic_json(folder / "manifest.json", manifest)
    condition = Condition("cache", "cache", updates=0)
    attach_meter(b, folder, condition)
    with patch.object(TorchBackend, "_rollout", side_effect=b.generate_test_rollout):
        execute(folder, condition, manifest, copy.deepcopy(data), b)
    (folder / "sr-cache.json").write_text("{}")
    with patch("srgc_research.report.tasks", return_value=[condition]):
        report = collect(tmp_path, ("math",), "n01")
        assert report["errors"] and not report["rows"] and not report["source_results"]


def test_n08_checks_dataset_inside_raw_export(tmp_path):
    source = tmp_path / "export.json"
    atomic_json(source, {"source_results": {"endpoint": {"dataset": "mbpp"}}})
    with pytest.raises(SystemExit):
        main(["math", "n08", "results", str(source)])


@pytest.mark.parametrize("kwargs", [{"key": "../bad"}, {"draw": 3}, {"batch": 41},
                                     {"stage": 25}, {"updates": -1}, {"kind": "unknown"}])
def test_invalid_conditions_fail_before_creating_work(kwargs):
    with pytest.raises(ValueError):
        Condition(**{"key": "test", "kind": "trajectory", **kwargs})
