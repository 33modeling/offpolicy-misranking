import copy
from unittest.mock import patch

import numpy as np
import pytest
import torch

from srgc_rebuttal.objectives import loo_advantages
from srgc_rebuttal.runtime import atomic_json, code_digest
from srgc_rebuttal.srgc import Config
from srgc_rebuttal.tests import test_torch_backend as torch_fixture
from srgc_research.arcus import Arcus
from srgc_research.backend import ResearchBackend
from srgc_research.cli import dependencies, run_queue
from srgc_research.design import (
    Condition,
    candidate_draw,
    conditions,
    matched_control,
    reference_sets,
    score_bins,
    tasks,
)
from srgc_research.report import analyze_archive, coverage, paired_questions, pass_at_k
from srgc_research.storage import (
    freeze,
    identity,
    root,
    validate_endpoint,
    verify_runtime,
)
from srgc_research.study import Diagnostic, NestedEngine, Trajectory


def equal_tree(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal_tree(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal_tree(a, b)
    else:
        assert left == right


@pytest.fixture
def backend():
    fixture = torch_fixture.ModelBackendTests()
    fixture.setUp()
    source = fixture.backend
    b = ResearchBackend(source.model, source.tokenizer, source.records, source.verifier,
                        projection_dim=16, max_new_tokens=2)
    b._rollout = fixture.deterministic_rollout
    return b


def study_backend():
    fixture = torch_fixture.ModelBackendTests()
    fixture.setUp()
    source = fixture.backend
    b = ResearchBackend(source.model, source.tokenizer, source.records, source.verifier,
                        projection_dim=16, max_new_tokens=2)
    # Keep ResearchBackend's capture and replay; replace generation only.
    b.generate_test_rollout = fixture.deterministic_rollout
    return b


@pytest.fixture
def data():
    ids = [f"p{i}" for i in range(90)]
    return {"candidate_ids": ids[:80], "validation_pool_ids": ids[80:86],
            "ranking_validation_ids": ids[80:83], "evaluation_ids": ids[86:90],
            "records": {i: {"prompt": f"problem {i}", "question": i, "answer": "3"} for i in ids},
            "cached_rewards": {i: [0, 1] * 4 for i in ids[:80]}}


def test_task_counts_and_scope_isolation():
    assert [len(conditions(f"n{i:02d}")) for i in range(1, 9)] == [4, 3, 9, 6, 15, 2, 3, 0]
    assert [t.key for t in tasks("n02")][:2] == ["cache", "anchors"]
    assert all(t.arm != "switch" for t in conditions("n05"))
    assert not any("fixed200" in t.key or "replicate" in t.key for t in tasks("n01"))
    with pytest.raises(ValueError):
        tasks("all")


def test_nested_draw_and_reference_split(data):
    ids = list(map(str, range(400)))
    small, large = candidate_draw(ids, 40, 5, 25), candidate_draw(ids, 160, 5, 25)
    assert small == large[:40] and len(set(large)) == 160
    assert small != candidate_draw(ids, 40, 5, 50)
    refs, overlap = reference_sets(data, 5)
    assert len({frozenset(r) for r in refs}) == 3
    assert all(len(r) == 3 for r in refs)
    assert overlap["0-2"] == 0
    data["validation_pool_ids"] = data["ranking_validation_ids"]
    with pytest.raises(ValueError):
        reference_sets(data, 5)


def test_bins_and_exact_matching():
    ids = list(map(str, range(40)))
    chosen, details = score_bins(ids, np.zeros(40), 4, 7)
    assert details["constant_scores"] and len(chosen) == 6
    for index in range(4):
        assert set(chosen[f"bin-{index}"]) <= set(details["bins"][index]["ids"])
    labels = {i: int(i) % 3 for i in ids}
    control, metadata = matched_control(ids, ids[:4], labels, 12)
    assert metadata["unmatched"] == 0 and len(set(control)) == 4
    assert sorted(labels[i] for i in control) == sorted(labels[i] for i in ids[:4])


def test_readout_is_projected_exact_head_ascent_gradient(backend):
    b = backend
    head = b.model.get_base_model().lm_head
    head.weight.requires_grad_(True)
    sequences, rewards, start = b._rollout("p0", 8, 1)
    objective = sum(lp.sum() * float(a) / 8 for lp, a in zip(
        b._logps_batch(sequences, start), loo_advantages(rewards, 4)))
    raw = torch.autograd.grad(objective, head.weight)[0]
    output, hidden = b._projections(head)
    expected = (output.T @ raw.float() @ hidden).detach().flatten().numpy()
    optimizer = copy.deepcopy(b.optimizer.state_dict())
    actual = b.readout_gradients(["p0"], responses=8, group_size=4, seed=1)["p0"]
    np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=2e-5)
    equal_tree(optimizer, b.optimizer.state_dict())
    assert np.linalg.norm(actual) > 0


def test_audit_restores_moments_steps_weights_and_trainability(backend):
    b = backend
    b.train(["p0", "p1"], responses=8, objective="grpo", seed=3)
    saved = b.state_dict()
    flags = [p.requires_grad for p in b.model.parameters()]
    vectors, metrics = b.gradient_audit(["p0", "p1"], seed=5)
    assert metrics["gradient_norm"] > 0
    assert vectors["zero_update"].norm() > 0
    torch.testing.assert_close(vectors["incremental_update"], vectors["update"] - vectors["zero_update"])
    equal_tree(saved, b.state_dict())
    assert flags == [p.requires_grad for p in b.model.parameters()]
    with patch.object(b, "train", side_effect=RuntimeError("test failure")), pytest.raises(RuntimeError):
        b.gradient_audit(["p0"], seed=6)
    equal_tree(saved, b.state_dict())


def test_arcus_is_without_replacement_updates_zero_groups_and_resumes():
    a = Arcus(list(map(str, range(120))))
    for step in range(30):
        ids = a.sample(step)
        assert len(set(ids)) == 5
        selected = a.observe({i: [0] * 8 if j % 2 else [0, 1] * 4 for j, i in enumerate(ids)})
        assert len(selected) <= 4
        assert all(i in ids[::2] for i in selected)
        assert np.isfinite(a.mu).all() and (a.variance > 0).all()
    snapshot = a.state_dict()
    resumed = Arcus(a.ids)
    resumed.load_state_dict(snapshot)
    assert a.sample(40) == resumed.sample(40)
    assert abs(a.inclusion(a.scores(a.target)).sum() - 5) < 1e-8


def test_rollout_cache_persists_interrupted_operation_and_separates_policies(tmp_path):
    b = study_backend()
    b.rollout_cache_root = tmp_path
    from srgc_rebuttal.torch_backend import TorchBackend
    with patch.object(TorchBackend, "_rollout", side_effect=b.generate_test_rollout) as generate:
        with b.operation("step-0"):
            first = b._rollout("p0", 8, 99)
            second = b._rollout("p0", 8, 99)
        assert generate.call_count == 1
        with b.operation("step-1"):
            b._rollout("p0", 8, 99)
        assert generate.call_count == 2
        equal_tree(first, second)


@pytest.mark.parametrize("arm", ["on_policy", "switch", "sr", "random", "lesser", "arcus_adapted"])
def test_trajectory_resume_matches_uninterrupted(data, arm):
    b = study_backend()
    b.records = data["records"]
    config = Config(seed=5, projection_dim=16)
    condition = Condition("test-" + arm, "trajectory", arm=arm, updates=2)
    from srgc_rebuttal.torch_backend import TorchBackend
    with patch.object(TorchBackend, "_rollout", side_effect=b.generate_test_rollout):
        study = Trajectory(b, data, config, condition)
        study.advance()
        study.advance()
        checkpoint = copy.deepcopy(study.state_dict())
        while not study.done:
            study.advance()
        expected = study.state_dict()
        replay = Trajectory(b, data, config, condition)
        replay.load_state_dict(checkpoint)
        while not replay.done:
            replay.advance()
        actual = replay.state_dict()
    # Timers are intentionally fresh measurements, not reproducible values.
    if actual["engine"]:
        for state in (actual, expected):
            state["engine"].pop("costs")
            for row in state["engine"]["history"]:
                row.pop("selection_gpu_seconds")
                row.pop("training_gpu_seconds")
    equal_tree(expected, actual)


@pytest.mark.parametrize("scope", ["n02", "n03", "n04", "n07"])
def test_diagnostics_complete_and_resume_at_every_phase(data, scope):
    b = study_backend()
    b.records = data["records"]
    config = Config(seed=5, projection_dim=16)
    carrier = NestedEngine(b, data["candidate_ids"], data["ranking_validation_ids"], data["cached_rewards"],
                           config=config, arm="on_policy")
    anchor = carrier.state_dict()
    initial = b.state_dict()
    condition = Condition(scope + "-stage-0", "diagnostic", arm=scope, stage=0, updates=25)
    study = Diagnostic(b, data, config, condition, anchor)
    from srgc_rebuttal.torch_backend import TorchBackend
    with patch.object(TorchBackend, "_rollout", side_effect=b.generate_test_rollout):
        iterations = 0
        while not study.done:
            study.advance()
            saved = copy.deepcopy(study.state_dict())
            resumed = Diagnostic(b, data, config, condition, anchor)
            resumed.load_state_dict(saved)
            equal_tree(saved, resumed.state_dict())
            study = resumed
            iterations += 1
            assert iterations < 250
    if scope == "n02":
        equal_tree(initial, b.state_dict())
        assert len(study.rows) == 3 and len(study.details["alignment"]) == 3
        assert len(study.centers) == 8
    elif scope == "n04":
        assert len(study.rows) == 6 and not study.selected
        equal_tree(initial, b.state_dict())
    else:
        count = 6 if scope == "n03" else 4
        assert len(study.rows) == count * 4
        assert {r["branch_updates"] for r in study.rows} == {0, 1, 5, 25}
        assert study.step == count * 25


def test_pairing_and_passk_dont_invent_raw_responses():
    result = paired_questions({"x": .5, "y": .25}, {"x": .25, "y": .5})
    assert result["mean_difference"] == 0 and result["observed_lower"] == result["observed_higher"] == 1
    with pytest.raises(ValueError):
        paired_questions({"x": .5}, {"y": .5})
    assert pass_at_k([0, 0, 1, 1], 2) == pytest.approx(5 / 6)
    with pytest.raises(ValueError):
        pass_at_k([0, 1], 3)
    assert coverage({"per_question_reward": {"x": .5}})["pass_at_k"] is None
    common = {"seed": 5, "plan_sha256": "a", "input_sha256": "b", "implementation_sha256": "c", "total_updates": 275,
              "sampling_protocol": "x", "per_question_reward": {"p0": .5}}
    bundle = {"source_results": {"a": dict(common, arm="sr"), "b": dict(common, arm="switch")}}
    assert len(analyze_archive(bundle)["comparisons"]) == 1
    bundle["source_results"]["b"]["input_sha256"] = "wrong"
    assert analyze_archive(bundle)["errors"]


def test_snapshot_tamper_and_group_storage_guard(tmp_path):
    before = code_digest()
    snapshot, sha = freeze(tmp_path)
    verify_runtime(snapshot, sha)
    assert (snapshot / "src/bootstrap_math_verify.py").exists()
    assert (snapshot / "srgc_research/rank.py").exists()
    (snapshot / "srgc_research/arcus.py").write_text("invalid")
    with pytest.raises(ValueError, match="differs"):
        verify_runtime(snapshot, sha)
    assert before == code_digest()
    with pytest.raises(ValueError, match="separate directory"):
        root({"GROUP_VOLUME": str(tmp_path), "SRGC_RESEARCH_ROOT": "/home/small-volume"})


def test_endpoint_identity_and_dependency_validation(tmp_path):
    manifest = {"protocol": "p", "dataset": "math", "seed": 5, "input_sha256": "i", "source_plan_sha256": "s", "implementation_sha256": "c"}
    cond = Condition("n01-features", "features", updates=0)
    costs = {"complete": True, "known_gpu_seconds": {"selection_gpu_seconds": 0., "training_gpu_seconds": 0.},
             "unfinished_phases": [], "total_gpu_seconds": 0.}
    value = {**identity(manifest, cond), "status": "complete", "result": {"same_rollouts": True,
             "representations": {"dense": {}, "lesser": {}}}, "cost_receipts": costs,
             "cost_measurement_complete": True}
    validate_endpoint(value, manifest, cond)
    value["seed"] = 6
    with pytest.raises(ValueError):
        validate_endpoint(value, manifest, cond)
    assert not dependencies(tmp_path, Condition("sr", "trajectory", arm="sr"), manifest)


def test_queue_claim_resume_and_failed_prerequisites(tmp_path):
    manifest = {"dataset": "math", "seed": 5}
    complete_set = set()
    condition = Condition("x", "trajectory", arm="random", updates=1)
    def launch(folder, spec, saved, lock):
        assert lock.fileno() >= 0
        complete_set.add(spec.key)
        return 0
    with patch("srgc_research.cli.tasks", return_value=[condition]), \
            patch("srgc_research.cli.complete", side_effect=lambda f, m, c: c.key in complete_set), \
            patch("srgc_research.cli.launch", side_effect=launch) as invoked, \
            patch("srgc_research.cli.time.sleep"):
        assert run_queue([(tmp_path, manifest)], "n01", poll=0) == 0
        assert run_queue([(tmp_path, manifest)], "n01", poll=0) == 0
        assert invoked.call_count == 1
    cache = Condition("cache", "cache", updates=0)
    atomic_json(tmp_path / "cache/queue.json", {"attempts": 3, "status": "failed"})
    with patch("srgc_research.cli.tasks", return_value=[cache, Condition("s", "trajectory", arm="sr")]), \
            patch("srgc_research.cli.complete", return_value=False):
        assert run_queue([(tmp_path, manifest)], "n01", poll=0) == 2
