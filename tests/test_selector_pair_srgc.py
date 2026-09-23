"""Real SR-GC decisions/barriers/scheduling, with only GPU projections replaced."""
import copy
from pathlib import Path

import pytest

import selection_gate as core
import selection_gate_gpu as base
import selector_pair_gpu as gpu
import selector_pair_srgc as srgc
import selector_pair_srgc_score as score
import selector_pair_parallel as parallel
from test_selector_pair_gpu import fake_study
from test_selector_pair_parallel import study


@pytest.mark.parametrize("a,b,selector", [(-2., 1., "cached"), (2., -1., "on_policy"),
                                         (0., 0., "on_policy"), (-1., 1., "on_policy")])
def test_zero_threshold_rule(a, b, selector):
    assert srgc.choose(a, b)["selector"] == selector
    assert "h_hat_gpu_seconds" not in srgc.choose(a, b)


@pytest.mark.parametrize("bad", [None, True, float("nan"), float("inf"), "0"])
def test_missing_or_invalid_reference_never_becomes_a_default(bad):
    with pytest.raises(ValueError):
        srgc.choose(bad, 1.)


def test_reference_contrast_matches_paper_equation_and_cancels_overlap(monkeypatch, tmp_path):
    import numpy as np
    shared = np.array([1000., -900.])
    projections = {
        "candidate-a": {0: np.array([2., 4.]), 1: shared, 2: np.array([6., 0.])},
        "validation-a": {2: np.array([1., 0.]), 3: np.array([3., 2.])},
        "candidate-b": {0: np.array([0., 2.]), 1: shared, 2: np.array([4., 6.])},
        "validation-b": {4: np.array([0., 1.]), 5: np.array([2., 3.])},
    }
    monkeypatch.setattr(score, "projections", lambda _, stage: projections[stage])
    result = srgc.reference_contrast(tmp_path, {"on_policy": [0, 1], "cached": [1, 2]})
    assert result == {"method": "SR-GC", "d_a": -2., "d_b": -6., "d": -4., "selector": "cached"}
    identical = srgc.reference_contrast(tmp_path, {"on_policy": [0, 1], "cached": [0, 1]})
    assert identical["d"] == 0. and identical["selector"] == "on_policy"
    for stage in ("candidate-a", "candidate-b"):
        projections[stage] = {i: 3 * v for i, v in projections[stage].items()}
    assert srgc.reference_contrast(tmp_path, {"on_policy": [0, 1], "cached": [1, 2]})["d"] == -12.


@pytest.fixture
def current_policy(tmp_path, monkeypatch):
    calls = []

    def inputs(entry):
        branch, out, c, protocol, suite = entry
        c = copy.deepcopy(c)
        run = tmp_path / "parents" / f"s{c['config']['seed']}-t{c['config']['drift']}"
        parent = run / f"policy_step_{c['config']['drift']}"
        parent.mkdir(parents=True, exist_ok=True)
        (parent / "adapter_model.safetensors").write_bytes(b"current saved parent")
        core.atomic_json(run / "prompts.json", {"train": list(range(20)), "val": list(range(8))})
        import json
        (run / "rollouts_behavior_train.jsonl").write_text("".join(json.dumps({
            "prompt_idx": i, "rollout_idx": j, "reward": int(i < 2 and j < 4)}) + "\n"
            for i in range(20) for j in range(8)))
        c.update(source_run=str(run), n=20)
        c["config"].update(fresh_k=32, micro_group=4, behavior_k=8, val_k=8, proj_dim=2, topk_frac=.1)
        core.atomic_json(out / "contract.json", c)
        return branch, out, c, protocol, suite

    def paid(directory, p, phase, commands, env):
        calls.append(phase)
        with gpu.pair_lease(tmp_path / ".pair-barrier.lock"):
            pass
        for command, device in commands:
            target = Path(command[command.index("--root") + 1])
            stage = command[command.index("--stage") + 1]
            shard = int(command[command.index("--shard") + 1])
            reference = (target / "reference.json").exists()
            contract_path = target / ("reference.json" if reference else "scoring.json")
            c = core.read(contract_path)
            prompts = core.read(Path(c["prompts"]))
            if reference:
                ids = score.indices(c, prompts, stage)
            else:
                ids = list(range(4 if stage == "validation" else 20))
            indices = ids[len(ids) * shard // 4:len(ids) * (shard + 1) // 4]
            sign = -1. if c["config"]["drift"] == 25 else 1.
            payload = {}
            for i in indices:
                if stage.startswith("validation"):
                    payload[str(i)] = [1., 0.]
                elif reference:
                    payload[str(i)] = [sign if i >= 18 else 0., 0.]
                else:
                    payload[str(i)] = i / 20.
            path = target / f"{stage}-{shard}.json"
            core.atomic_json(path, payload)
            binding = {"reference_sha256" if reference else "contract_sha256": base.digest(contract_path),
                       "stage": stage, "shard": shard, "sha256": base.digest(path)}
            core.atomic_json(target / f"{stage}-{shard}.done.json", binding)
        base.meter(directory, phase, p["gpu_type"], action=lambda: None, ledger="deployment")

    monkeypatch.setattr(srgc, "paid", paid)
    return inputs, calls


def test_independent_validation_partitions_and_selected_union_only():
    c = {"sets": {"on_policy": [2, 3], "cached": [1, 3]}}
    prompts = {"train": list(range(4)), "val": list(range(100))}
    assert score.indices(c, prompts, "candidate-a") == [1, 2, 3]
    assert score.indices(c, prompts, "candidate-b") == [1, 2, 3]
    assert score.indices(c, prompts, "validation-a") == list(range(50, 75))
    assert score.indices(c, prompts, "validation-b") == list(range(75, 100))


@pytest.fixture
def srgc_study(tmp_path, study, current_policy, monkeypatch):
    p, calls, original = study
    inputs, measurements = current_policy

    def states(root, seed, step):
        identity, entries = original(root, seed, step)
        return identity, {name: inputs(entry) for name, entry in entries.items()}

    monkeypatch.setattr(gpu, "verify_pair", states)
    monkeypatch.setattr(gpu, "fit", lambda *args: pytest.fail("SR-GC must never fit a regression"))
    monkeypatch.setattr(gpu.pair, "fit", lambda *args: pytest.fail("SR-GC must never read H labels"))
    return p, calls, measurements, states


def test_srgc_runs_before_development_and_resumes_without_regression(tmp_path, srgc_study):
    p, calls, measurements, _ = srgc_study
    old_decisions, old_select = gpu.decisions, gpu.switch.runtime.select_once
    with srgc.activated(tmp_path, p, ["0", "1", "2", "3"]):
        for _ in range(2):
            parallel.run_distributed(tmp_path, p, ["0", "1", "2", "3"], "run", srgc.run_stages)
        choices = gpu.decisions(tmp_path, p)
    assert gpu.decisions is old_decisions and gpu.switch.runtime.select_once is old_select
    assert len(calls) == 42
    assert all(seed in gpu.pair.TEST_SEEDS for _, seed, _, _ in calls[:24])
    assert sum(name.startswith("adaptive-") for name, *_ in calls) == 6
    assert len(measurements) == 6 * 6
    assert not (tmp_path / "model.json").exists()
    assert not (tmp_path / "fit-cost.json").exists()
    assert all(value["method"] == "SR-GC" and "h_hat_gpu_seconds" not in value for value in choices.values())
    assert choices["s3-t25"]["selector"] == "cached"
    assert choices["s3-t50"]["selector"] == "on_policy"
    assert not core.read(tmp_path / "report.json")["missing_states"]


def test_target_and_future_outcomes_do_not_enter_frozen_decisions(tmp_path, srgc_study):
    p, _, _, _ = srgc_study
    with srgc.activated(tmp_path, p, ["0", "1", "2", "3"]):
        srgc.freeze(tmp_path, p, ["0", "1", "2", "3"])
        before = {path: path.read_bytes() for path in (tmp_path / "sr-gc").glob("*/decision.json")}
        for seed in gpu.pair.DEV_SEEDS:
            for step in gpu.pair.STEPS:
                core.atomic_json(tmp_path / "development" / f"s{seed}-t{step}" / "result.json",
                                 {"contrast": {"status": "censored", "h_gpu_seconds": None}})
        core.atomic_json(tmp_path / "report.json", {"future_reward": -1e99})
        core.atomic_json(tmp_path / "model.json", {"unused_legacy_regression": 1e99})
        altered = {**p, "target_reward": .9999}
        srgc.freeze(tmp_path, altered, [])
        assert all(path.read_bytes() == saved for path, saved in before.items())


@pytest.mark.parametrize('hashes', [srgc.PRE_FAILURE_HANDLING_HASHES, srgc.PRE_BUDGET_RECOVERY_HASHES,
                                  srgc.PRE_SRGC_COST_RECOVERY_HASHES])
def test_previous_runtime_receipt_and_saved_decisions_are_preserved(tmp_path, srgc_study, hashes):
    p, _, measurements, _ = srgc_study
    core.atomic_json(tmp_path / srgc.RECEIPT, {
        **srgc.receipt(tmp_path, p), "code_sha256": hashes})
    receipt_before = (tmp_path / srgc.RECEIPT).read_bytes()
    with srgc.activated(tmp_path, p, ["0", "1", "2", "3"]):
        choices = srgc.freeze(tmp_path, p, ["0", "1", "2", "3"])
        before = {path: path.read_bytes() for path in (tmp_path / "sr-gc").glob("*/decision.json")}
        measured = len(measurements)
        assert srgc.freeze(tmp_path, p, []) == choices
        assert len(measurements) == measured
        assert all(path.read_bytes() == value for path, value in before.items())
    assert (tmp_path / srgc.RECEIPT).read_bytes() == receipt_before


def test_unknown_runtime_is_not_accepted_as_predecessor(tmp_path, study):
    p, _, _ = study
    value = srgc.receipt(tmp_path, p)
    value["code_sha256"]["selector_pair_srgc_score.py"] = "changed-scoring"
    core.atomic_json(tmp_path / srgc.RECEIPT, value)
    with pytest.raises(ValueError, match="runtime receipt"):
        srgc.activate(tmp_path, p)


def test_freeze_recovers_s4_t100_and_reuses_finished_reference_shards(tmp_path, srgc_study, monkeypatch):
    p, _, measurements, _ = srgc_study
    directory = tmp_path / 'sr-gc/s4-t100'
    original = base.meter
    def interrupted(target, phase, gpu_type, **kwargs):
        if target == directory and phase == 'sr-gc-aggregate':
            start = {'event_id': 'interrupted', 'state': 'started', 'phase': phase,
                     'ledger': 'deployment', 'gpus': 4, 'gpu_type': gpu_type,
                     'host': 'remote-stopped-worker', 'time': 100.}
            base.journal(target / 'cost.jsonl', start)
            core.atomic_json(target / 'progress.json', {**start, 'seconds': 12., 'updated': 112.})
            raise RuntimeError('simulated interrupted aggregate')
        return original(target, phase, gpu_type, **kwargs)
    with srgc.activated(tmp_path, p, list('0123')):
        monkeypatch.setattr(base, 'meter', interrupted)
        with pytest.raises(gpu.IncompletePairRun):
            srgc.freeze(tmp_path, p, list('0123'))
        shards = {path: path.read_bytes() for path in directory.glob('*.done.json')}
        assert len(shards) == 16 and not (directory / 'decision.json').exists()
        measured = len(measurements)
        monkeypatch.setattr(base, 'meter', original)
        choices = srgc.freeze(tmp_path, p, list('0123'))
        assert choices['s4-t100']['method'] == 'SR-GC'
        assert len(measurements) == measured
        assert all(path.read_bytes() == data for path, data in shards.items())
        assert not base.cost(directory)['incomplete_events']
        assert choices['s4-t100']['new_measurement_gpu_seconds'] >= 288.


@pytest.mark.parametrize("artifact", ["test-decisions.json", "decisions/s3-t25/decision.json",
    "branches/adaptive-cached/states/s3-t25/points/view-25/selection_full/execution.json"])
def test_legacy_decisions_or_training_are_never_relabelled(tmp_path, study, artifact):
    p, _, _ = study
    path = tmp_path / artifact
    core.atomic_json(path, {"legacy": True})
    before = path.read_bytes()
    with pytest.raises(ValueError, match="legacy"):
        srgc.activate(tmp_path, p)
    assert path.read_bytes() == before and not (tmp_path / srgc.RECEIPT).exists()


def test_executed_subset_is_the_exact_frozen_srgc_subset(tmp_path, srgc_study):
    p, _, _, states = srgc_study
    with srgc.activated(tmp_path, p, ["0", "1", "2", "3"]):
        choices = srgc.freeze(tmp_path, p, ["0", "1", "2", "3"])
        for step in gpu.pair.STEPS:
            value = choices[f"s3-t{step}"]
            _, entries = states(tmp_path, 3, step)
            _, out, c, protocol, _ = entries["adaptive-" + value["selector"]]
            # Fake study has no Switch manifests, but uses the real adaptive namespace.
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(gpu.switch, "switch_root", lambda _: tmp_path / "branches" / ("adaptive-" + value["selector"]))
                selected = gpu.switch.runtime.select_once(out, c, protocol, "selection_full", {}, {}, [])
            assert selected == value["sets"][value["selector"]]


def test_reference_tampering_is_rejected(tmp_path, srgc_study):
    p, _, _, _ = srgc_study
    with srgc.activated(tmp_path, p, ["0", "1", "2", "3"]):
        srgc.freeze(tmp_path, p, ["0", "1", "2", "3"])
        core.atomic_json(tmp_path / "sr-gc/s3-t25/candidate-a-0.json", {"forged": [99., 0.]})
        with pytest.raises(ValueError, match="projections changed"):
            gpu.decisions(tmp_path, p)


@pytest.mark.parametrize("failure", [ValueError, RuntimeError, TimeoutError])
def test_reference_failure_keeps_all_fixed_pair_work_runnable(tmp_path, srgc_study, monkeypatch, failure):
    p, calls, _, _ = srgc_study
    def unavailable(*args):
        raise failure("missing independent A/B measurements")
    monkeypatch.setattr(srgc, "measure", unavailable)
    with srgc.activated(tmp_path, p, ["0", "1", "2", "3"]):
        with pytest.raises(gpu.IncompletePairRun, match="missing independent A/B"):
            parallel.run_distributed(tmp_path, p, ["0", "1", "2", "3"], "run", srgc.run_stages)
    assert len(calls) == 36
    assert not any(name.startswith("adaptive-") for name, *_ in calls)
    assert not (tmp_path / "test-decisions.json").exists()
    statuses = list((tmp_path / "sr-gc").glob("*/measurement-status.json"))
    assert len(statuses) == 6
    assert all(core.read(path)["state"] == "BLOCKED" for path in statuses)


def test_peer_owned_measurement_preserves_other_decisions_and_resumes(tmp_path, srgc_study):
    p, _, measurements, _ = srgc_study
    with srgc.activated(tmp_path, p, ["0", "1", "2", "3"]):
        with gpu.pair_lease(tmp_path / "sr-gc/s3-t25/.decision.lock"):
            with pytest.raises(gpu.IncompletePairRun, match="owned by peer"):
                srgc.freeze(tmp_path, p, ["0", "1", "2", "3"])
        assert len(measurements) == 30
        before = {path: path.read_bytes() for path in (tmp_path / "sr-gc").glob("*/decision.json")}
        srgc.freeze(tmp_path, p, ["0", "1", "2", "3"])
        assert len(measurements) == 36
        assert all(path.read_bytes() == value for path, value in before.items())


def test_reuses_current_ranking_without_repeating_r_scoring_or_refunding_cost(tmp_path, srgc_study):
    import shutil
    p, _, measurements, states = srgc_study
    identity, entries = states(tmp_path, 3, 25)
    initial = tmp_path / "initial-measurement"
    srgc.measure(initial, identity, entries["on_policy"], ["0", "1", "2", "3"], p)
    source = entries["on_policy"][1] / "selection_full/fresh-r"
    shutil.copytree(initial / "ranking", source)
    for phase in ("fresh-r-validation", "fresh-r-candidate"):
        base.meter(source.parent, phase, p["gpu_type"], action=lambda: None, ledger="deployment")
    cost = srgc.ranking_cost(source.parent)
    before = (source.parent / "cost.jsonl").read_bytes()
    count = len(measurements)
    directory = tmp_path / "sr-gc/s3-t25"
    value = srgc.measure(directory, identity, entries["on_policy"], ["0", "1", "2", "3"], p)
    assert len(measurements) - count == 4
    assert not any(phase.startswith("sr-gc-r-") for phase in measurements[count:])
    assert value["reused_ranking_gpu_seconds"] == cost > 0
    assert value["diagnosis_gpu_seconds"] == cost + value["new_measurement_gpu_seconds"]
    assert (source.parent / "cost.jsonl").read_bytes() == before
    assert srgc.validate_choice(tmp_path, p, 3, 25) == value
