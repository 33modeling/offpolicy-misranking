"""Pending convergence curves are work prerequisites, not failed gate fits."""

import pytest

import selection_gate as core
import selection_gate_gpu as base
import selection_switch as rule
import selection_switch_gpu as switch
from test_selection_switch_gpu import simulated_queue
from test_mbpp_node_queue import queue_worker


@pytest.fixture(autouse=True)
def mbpp_queue_fit(monkeypatch):
    original = switch.fit_once
    monkeypatch.setattr(switch, "fit_once", lambda root: queue_worker.fit_when_ready(root, original))


def development_results(root, *, curves=False):
    directories = []
    for seed in rule.DEV_SEEDS:
        for step in rule.STEPS:
            child = switch.child_root(root, seed, step)
            out = child / f"points/view-{step}"
            core.atomic_json(out / "contract.json", {"seed": seed, "step": step})
            core.atomic_json(child / "suite.json", {"schema": base.SCHEMA, "points": [
                {"name": out.name, "sha256": base.digest(out / "contract.json")}]})
            for arm in rule.DEV_ARMS:
                directory = out / arm
                core.atomic_json(directory / "result.json", {"complete": True})
                if curves:
                    core.atomic_json(directory / "curve.json", {"ready": True})
                directories.append(directory)
    return directories


def snapshot(root):
    return {path: (path.read_bytes(), path.stat().st_mtime_ns)
            for path in root.rglob("*") if path.is_file()}


def test_pending_curves_skip_full_validation_and_write_nothing(tmp_path, monkeypatch):
    core.atomic_json(tmp_path / "switch.json", {"gate": "convergence", "dataset": "mbpp"})
    development_results(tmp_path)
    before = snapshot(tmp_path)
    monkeypatch.setattr(switch, "collect", lambda *args, **kwargs:
                        pytest.fail("pending curves triggered full policy/source validation"))
    assert switch.fit_once(tmp_path) is False
    assert snapshot(tmp_path) == before


def test_pending_curves_do_not_contend_for_fit_lock(tmp_path):
    core.atomic_json(tmp_path / "switch.json", {"gate": "convergence", "dataset": "mbpp"})
    directories = development_results(tmp_path, curves=True)
    (directories[-1] / "curve.json").unlink()
    with base.lease(tmp_path / ".fit.lock"):
        before = snapshot(tmp_path)
        assert switch.fit_once(tmp_path) is False
        assert snapshot(tmp_path) == before


@pytest.mark.parametrize("protocol", [
    {"dataset": "math500", "gate": "convergence"},
    {"dataset": "mbpp", "gate": "final"},
])
def test_other_protocols_keep_original_fit_behavior(tmp_path, protocol):
    core.atomic_json(tmp_path / "switch.json", protocol)
    calls = []
    assert queue_worker.fit_when_ready(tmp_path, lambda root: calls.append(root) or True)
    assert calls == [tmp_path]


def test_existing_model_is_still_validated_without_waiting_for_curves(tmp_path):
    core.atomic_json(tmp_path / "switch.json", {"dataset": "mbpp", "gate": "convergence"})
    core.atomic_json(tmp_path / "model.json", {"tampered": True})
    before = snapshot(tmp_path)

    def validate(root):
        raise ValueError("invalid frozen model")

    with pytest.raises(ValueError, match="invalid frozen model"):
        queue_worker.fit_when_ready(tmp_path, validate)
    assert snapshot(tmp_path) == before


def test_unregistered_point_does_not_block_a_ready_frozen_suite(tmp_path):
    core.atomic_json(tmp_path / "switch.json", {"dataset": "mbpp", "gate": "convergence"})
    development_results(tmp_path, curves=True)
    extra = switch.child_root(tmp_path, 0, 25) / "points/unregistered/selection_reduced/result.json"
    core.atomic_json(extra, {"historical": True})
    calls = []
    assert queue_worker.fit_when_ready(tmp_path, lambda root: calls.append(root) or True)
    assert calls == [tmp_path]


def test_unregistered_curve_cannot_make_the_frozen_suite_ready(tmp_path):
    core.atomic_json(tmp_path / "switch.json", {"dataset": "mbpp", "gate": "convergence"})
    directories = development_results(tmp_path, curves=True)
    (directories[0] / "curve.json").unlink()
    extra = switch.child_root(tmp_path, 0, 25) / "points/unregistered/selection_reduced/curve.json"
    core.atomic_json(extra, {"historical": True})
    assert queue_worker.fit_when_ready(tmp_path, lambda root: pytest.fail("unregistered curve used")) is False


def test_queue_entrypoint_defers_pending_fit_and_restores_callbacks(tmp_path, monkeypatch):
    core.atomic_json(tmp_path / "switch.json", {"dataset": "mbpp", "gate": "convergence"})
    development_results(tmp_path)
    original_fit, original_wait = switch.fit_once, switch.wait_for_peers
    monkeypatch.setattr(switch, "main", lambda: switch.fit_once(tmp_path))
    assert queue_worker.run() is False
    assert switch.fit_once is original_fit and switch.wait_for_peers is original_wait


@pytest.mark.parametrize("gate,curves", [("convergence", True), ("final", False)])
def test_ready_artifacts_still_enter_original_full_validation(tmp_path, monkeypatch, gate, curves):
    core.atomic_json(tmp_path / "switch.json", {"gate": gate, "dataset": "mbpp"})
    development_results(tmp_path, curves=curves)
    calls = []

    def validate(root, *, development):
        calls.append((root, development))
        raise ValueError("recorded policy hash changed")

    monkeypatch.setattr(switch, "collect", validate)
    with pytest.raises(ValueError, match="policy hash changed"):
        switch.fit_once(tmp_path)
    assert calls == [(tmp_path, True)]
    assert not (tmp_path / "model.json").exists()


def test_remaining_curves_unlock_gate_in_same_worker_without_retraining(tmp_path, monkeypatch):
    original_fit = switch.fit_once
    calls, gate_seen = simulated_queue(tmp_path, monkeypatch)
    protocol = switch.manifest(tmp_path)
    protocol.update(gate="convergence", dataset="mbpp")
    core.atomic_json(tmp_path / "switch.json", protocol)
    monkeypatch.setattr(switch, "fit_once", original_fit)
    monkeypatch.setattr(switch, "mbpp_resume_blocked", lambda *args: False)
    monkeypatch.setattr(rule, "validate_model", lambda _: None)
    monkeypatch.setattr(rule, "fit", lambda _: {"fitted": True})
    monkeypatch.setattr(switch, "fit_rows", lambda rows, protocol: rows)
    for seed in (*rule.DEV_SEEDS, *rule.TEST_SEEDS):
        for step in rule.STEPS:
            core.atomic_json(switch.prefix_dir(tmp_path, seed) / f"prefix-{step}.json", {})
            switch.publish_state(tmp_path, seed, step)
            if seed in rule.DEV_SEEDS:
                out = next(base.entries(switch.child_root(tmp_path, seed, step)))
                for arm in rule.DEV_ARMS:
                    core.atomic_json(out / arm / "result.json", {})
                    core.atomic_json(out / arm / "result.sha256.json", {
                        "sha256": base.digest(out / arm / "result.json")})
    saved = {path: path.read_bytes() for path in tmp_path.glob("states/*/points/*/*/result*.json")}
    fit_checks = []

    def collect(root, *, development):
        assert development
        for seed in rule.DEV_SEEDS:
            for step in rule.STEPS:
                out = next(base.entries(switch.child_root(root, seed, step)))
                for arm in rule.DEV_ARMS:
                    assert (out / arm / "curve.json").is_file(), "fit ran before required curves"
        fit_checks.append(True)
        return {"complete": True, "rows": []}

    original_run = switch.runtime.run_arm

    def resume(out, suite, p, arm, devices, env):
        if (out / arm / "result.json").exists():
            return
        original_run(out, suite, p, arm, devices, env)

    def curve(root, p, out, c, arm, suite, devices, env):
        core.atomic_json(out / arm / "curve.json", {"result_sha256": base.digest(out / arm / "result.json")})

    monkeypatch.setattr(switch, "collect", collect)
    monkeypatch.setattr(switch.runtime, "run_arm", resume)
    monkeypatch.setattr(switch, "curve_once", curve)
    assert switch.work(tmp_path, idle_timeout=0) == 0
    assert fit_checks == [True]
    assert all(seed in rule.TEST_SEEDS for seed, step, arm in calls)
    assert len(calls) == 30
    assert sum(arm == "gated" for _, _, arm in calls) == 6
    assert all(fitted for arm, fitted in gate_seen if arm == "gated")
    assert not (tmp_path / "gate-fit/failure.json").exists()
    assert saved == {path: path.read_bytes() for path in saved}
    before = len(calls)
    assert switch.work(tmp_path, idle_timeout=0) == 0
    assert len(calls) == before
