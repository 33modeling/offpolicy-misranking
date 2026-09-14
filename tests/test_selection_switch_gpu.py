import copy
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import net_gain_gate_gpu as runtime
import selection_gate as core
import selection_gate_gpu as base
import selection_switch as rule
import selection_switch_gpu as switch
from test_selection_gate_gpu import toy_source
from test_selection_switch import development


@pytest.fixture
def installed(monkeypatch):
    for target, names in ((runtime, ("net", "HERE", "TEST_ARMS", "SELECTORS", "CODE_FILES", "study", "protocol", "select_once", "measurement_worker", "decision")),
                          (base, ("verify",))):
        for name in names:
            monkeypatch.setattr(target, name, getattr(target, name))
    switch.install_runtime()


def test_new_runtime_keeps_primary_fresh_r_and_five_test_arms(installed):
    assert runtime.SELECTORS == ("fresh_r",)
    assert runtime.study.BRANCHES == rule.DEV_ARMS
    assert len(runtime.TEST_ARMS) == 5
    assert runtime.HERE == Path(switch.__file__).resolve()


def test_generic_parent_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(switch, "_verify", lambda _: {"config": {"seed": 0, "drift": 25}})
    with pytest.raises(ValueError, match="generic"):
        switch.verify(tmp_path)


def test_prefix_certificate_binds_actual_selected_history(tmp_path, monkeypatch):
    c = {"config": {"seed": 0, "drift": 25}, "selected_prefix": {"schema": rule.SCHEMA,
         "root": str(tmp_path), "certificate_sha256": core.fingerprint({"history": "selected"})}}
    monkeypatch.setattr(switch, "_verify", lambda _: c)
    monkeypatch.setattr(switch, "validate_prefix", lambda *a: {"history": "selected"})
    assert switch.verify(tmp_path) == c
    monkeypatch.setattr(switch, "validate_prefix", lambda *a: {"history": "random"})
    with pytest.raises(ValueError): switch.verify(tmp_path)


def test_gate_decision_precedes_every_control_and_resume_does_not_remeasure(tmp_path, monkeypatch):
    p = {"mode": "test", "arms": list(rule.TEST_ARMS)}
    order = []
    def decide(out, suite, protocol, arm, env):
        order.append(arm)
        base.bind(out / arm / "decision.json", {"action": "random"})
    monkeypatch.setattr(runtime, "decision", decide)
    frozen = switch.freeze_decisions(tmp_path, {}, p, {})
    assert order[0] == "gated" and set(order) == set(rule.TEST_ARMS)
    assert len(frozen["decisions"]) == 5
    monkeypatch.setattr(runtime, "decision", lambda *a: pytest.fail("diagnostic repeated"))
    assert switch.freeze_decisions(tmp_path, {}, p, {}) == frozen
    core.atomic_json(tmp_path / "gated/decision.json", {"action": "select"})
    with pytest.raises(ValueError): switch.freeze_decisions(tmp_path, {}, p, {})


@pytest.mark.parametrize("artifact", ["execution.json", "result.json"])
def test_barrier_rejects_outcomes_created_before_decision(tmp_path, artifact):
    core.atomic_json(tmp_path / "random_full" / artifact, {})
    with pytest.raises(ValueError, match="precede"):
        switch.freeze_decisions(tmp_path, {}, {"mode": "test", "arms": list(rule.TEST_ARMS)}, {})


def test_diagnostic_charge_once_per_paid_arm_not_free_arm(tmp_path, monkeypatch, installed):
    out, c = toy_source(tmp_path)
    p = {"mode": "test", "model": {}, "arms": list(rule.TEST_ARMS)}
    monkeypatch.setattr(runtime, "check_model", lambda *a: None)
    core.atomic_json(out / "gate_measurement/measurement.json", {"choice": {"action": "select", "reason": "frozen"}})
    monkeypatch.setattr(runtime, "measure_once", lambda *a: {"status": "complete", "gpu_seconds": 8., "report_sha256": "sha"})
    for arm in rule.TEST_ARMS:
        choice = runtime.decision(out, {}, p, arm, {})
        paid = arm in {*rule.DEV_ARMS, "gated"}
        assert choice["measurement_gpu_seconds"] == (8. if paid else 0.)
        assert choice["budget_gpu_seconds"] == (992. if paid else 1000.)
        assert runtime.decision(out, {}, p, arm, {}) == choice


def test_failed_diagnosis_has_paid_fallback_and_is_never_repeated(tmp_path, monkeypatch, installed):
    out, _ = toy_source(tmp_path)
    core.atomic_json(out / "net_inputs.json", {})
    p = {"mode": "test", "max_measurement_fraction": .01, "recent_window": 20, "model": {}}
    monkeypatch.setattr(runtime, "check_model", lambda *a: None)
    meter = base.meter
    def failed(directory, name, gpu_type, **kw):
        def fail(): raise RuntimeError("intentional failure")
        return meter(directory, name, gpu_type, action=fail, ledger=kw["ledger"])
    monkeypatch.setattr(base, "meter", failed)
    decision = runtime.decision(out, {"measurement_wall_seconds": 30.}, p, "gated", {})
    assert decision["action"] == "random" and decision["measurement_gpu_seconds"] > 0
    monkeypatch.setattr(base, "meter", lambda *a, **kw: pytest.fail("failed diagnostic was retried"))
    assert runtime.decision(out, {}, p, "gated", {}) == decision


def test_failed_test_diagnostic_does_not_block_actual_fallback(tmp_path, monkeypatch, installed):
    out, _ = toy_source(tmp_path)
    p = {"mode": "test", "arms": list(rule.TEST_ARMS), "model": {}}
    monkeypatch.setattr(runtime, "check_model", lambda *a: None)
    monkeypatch.setattr(runtime, "measure_once", lambda *a: {
        "status": "failed_no_retry", "gpu_seconds": 10., "report_sha256": None})
    frozen = switch.freeze_decisions(out, {}, p, {})
    assert set(frozen["decisions"]) == set(rule.TEST_ARMS)
    for arm in (*rule.DEV_ARMS, "gated"):
        assert core.read(out / arm / "decision.json")["budget_gpu_seconds"] == 990.
    assert core.read(out / "gated/decision.json")["reason"] == "measurement_failed_no_retry"


def test_completed_gate_uses_own_training_evaluation_not_control_reward(tmp_path, monkeypatch, installed):
    out, c = toy_source(tmp_path)
    p = {"mode": "test", "model": None}
    choice = {"binding": {"protocol_sha256": core.fingerprint(p), "contract_sha256": base.digest(out / "contract.json")},
              "action": "random", "reason": "frozen", "budget_gpu_seconds": 990., "measurement_gpu_seconds": 10.}
    core.atomic_json(out / "gated/decision.json", choice)
    monkeypatch.setattr(runtime, "decision", lambda *a: choice)
    monkeypatch.setattr(base, "verify", lambda _: c)
    monkeypatch.setattr(base, "policy", lambda *a: Path(c["source_run"]))
    monkeypatch.setattr(base, "train_command", lambda *a: ["fake"])
    calls = []
    meter = base.meter
    def execute(directory, name, gpu_type, **kwargs):
        if name in {"train", "evaluate"}:
            def act():
                calls.append(name)
                if name == "train":
                    core.atomic_json(directory / "policy/budget_stop.json", {
                        "completed_steps": 110, "stop_reason": "budget_exhausted", "use_parent_policy": False})
            return meter(directory, name, gpu_type, action=act, ledger=kwargs["ledger"])
        return meter(directory, name, gpu_type, **kwargs)
    monkeypatch.setattr(base, "meter", execute)
    monkeypatch.setattr(base, "rewards", lambda out, c, arm: {"q0": .3 if arm == "gated" else .9})
    runtime.run_arm(out, {"eval_timeout": 2.}, p, "gated", list("0123"), {})
    assert calls == ["train", "evaluate"]
    result = core.read(out / "gated/result.json")
    assert result["rewards"] == {"q0": .3}
    assert result["measurement_gpu_seconds"] == 10.


def test_nonblocking_task_lease_prevents_duplicate_nodes(tmp_path):
    import subprocess
    lock = tmp_path / "task.lock"
    script = "import sys; from pathlib import Path; from selection_gate_gpu import lease\nwith lease(Path(sys.argv[1])): pass"
    with base.lease(lock):
        result = subprocess.run([sys.executable, "-c", script, str(lock)], capture_output=True)
    assert result.returncode != 0 and b"BlockingIOError" in result.stderr


def test_all_48_tasks_auto_transition_only_after_18_dev_results(tmp_path, monkeypatch):
    seeds = (*rule.DEV_SEEDS, *rule.TEST_SEEDS)
    p = {"sources": {str(s): {"config": {}} for s in seeds}, "gpu_type": "H100"}
    monkeypatch.setattr(switch, "manifest", lambda _: p)
    monkeypatch.setattr(switch.subprocess, "check_output", lambda *a, **kw: "H100\n"*4)
    monkeypatch.setattr(switch, "status", lambda _: None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.setenv("OM_NODE_LOCK_HELD", "1")
    calls = []
    def build(root, seed, step, devices, env):
        assert (seed, step, "prefix") not in calls
        calls.append((seed, step, "prefix"))
        core.atomic_json(switch.prefix_dir(root, seed) / f"prefix-{step}.json", {})
    monkeypatch.setattr(switch, "build_prefix", build)
    def publish(root, seed, step):
        child = switch.child_root(root, seed, step)
        core.atomic_json(child / "points" / f"{seed}-{step}" / "contract.json", {"seed": seed, "step": step})
        core.atomic_json(child / "net_protocol.json", {"arms": list(rule.DEV_ARMS if seed in rule.DEV_SEEDS else rule.TEST_ARMS)})
        core.atomic_json(child / "suite.json", {})
    monkeypatch.setattr(switch, "publish_state", publish)
    monkeypatch.setattr(switch, "protocol", lambda child: core.read(child / "net_protocol.json"))
    monkeypatch.setattr(base, "entries", lambda child: iter((child / "points").iterdir()))
    def fit(root):
        if sum(1 for s, t, a in calls if s in rule.DEV_SEEDS and a in rule.DEV_ARMS) == 18:
            core.atomic_json(root / "model.json", {})
    monkeypatch.setattr(switch, "fit_once", fit)
    def freeze(out, *a):
        core.atomic_json(out / "decisions-frozen.json", {})
    monkeypatch.setattr(switch, "freeze_decisions", freeze)
    def run(out, suite, protocol, arm, devices, env):
        c = core.read(out / "contract.json")
        key = (c["seed"], c["step"], arm)
        assert key not in calls
        assert (out / "decisions-frozen.json").exists()
        if c["seed"] in rule.TEST_SEEDS: assert (tmp_path / "model.json").exists()
        calls.append(key)
        core.atomic_json(out / arm / "result.json", {})
    monkeypatch.setattr(runtime, "run_arm", run)
    assert switch.work(tmp_path, idle_timeout=0) == 0
    assert sum(a == "prefix" for _, _, a in calls) == 15
    assert sum(a != "prefix" for _, _, a in calls) == 48
    before = len(calls)
    assert switch.work(tmp_path, idle_timeout=0) == 0
    assert len(calls) == before


def test_existing_link_cannot_silently_point_to_other_policy(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir(); second.mkdir()
    switch.link(tmp_path / "parent", first)
    switch.link(tmp_path / "parent", first)
    with pytest.raises(ValueError): switch.link(tmp_path / "parent", second)
