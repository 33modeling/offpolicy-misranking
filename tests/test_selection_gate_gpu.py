"""CPU tests for process lifecycle, frozen GPU inputs and resource accounting."""

import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import selection_gate as gate
import selection_gate_gpu as gpu
from selection_gate_budget import stop_before_step

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("deadline,now,reserve,last,want", [
    (None, 100, 30, 20, False), (200, 100, 30, 20, False),
    (150, 100, 30, 20, True), (99, 100, 0, 0, True)])
def test_training_budget_check_does_not_run_a_gate(deadline, now, reserve, last, want):
    assert stop_before_step(deadline, now, reserve, last) is want


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, "100"])
def test_nonfinite_deadline_rejected(bad):
    with pytest.raises(ValueError):
        stop_before_step(bad, 10, 5, 1)


def test_measurement_charges_idle_gpu_allocation(tmp_path):
    assert gpu.meter(tmp_path, "cpu-summary", "H100", action=lambda: 42) == 42
    costs = gpu.cost(tmp_path)["ledgers"]["research"]
    assert costs["gpu_seconds"] == 4*costs["wall_seconds"] > 0


def test_failed_action_is_charged_and_incomplete_action_is_not_zero(tmp_path):
    def fail():
        raise ValueError("bad cache")
    with pytest.raises(ValueError, match="bad cache"):
        gpu.meter(tmp_path, "profile", "H100", action=fail)
    assert gpu.spent(tmp_path) > 0
    assert gpu.cost(tmp_path)["ledgers"]["research"]["failed_events"] == 1
    gpu.journal(tmp_path / "cost.jsonl", {"event_id": "unclean-kill", "state": "started"})
    with pytest.raises(ValueError, match="unknown cost"):
        gpu.spent(tmp_path)


def test_process_timeout_reaps_worker_and_records_failure(tmp_path):
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        gpu.meter(tmp_path, "slow", "H100", commands=[([sys.executable, "-c", "import time; time.sleep(20)"], "")],
                  timeout=.1, devices=0)
    assert time.monotonic()-started < 7
    assert gpu.cost(tmp_path)["complete"]
    assert gpu.cost(tmp_path)["ledgers"]["research"]["failed_events"] == 1


def test_failed_shard_cancels_sibling_and_does_not_loop(tmp_path):
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="worker failed"):
        gpu.meter(tmp_path, "shards", "H100", commands=[([sys.executable, "-c", "raise SystemExit(7)"], ""),
                  ([sys.executable, "-c", "import time; time.sleep(20)"], "")], timeout=10, devices=0)
    assert time.monotonic()-started < 7
    assert gpu.cost(tmp_path)["complete"]


def test_reporting_cost_is_not_branch_training_cost(tmp_path):
    gpu.meter(tmp_path, "eval", "H100", action=lambda: None, ledger="reporting")
    assert gpu.spent(tmp_path) == 0.
    assert gpu.cost(tmp_path)["ledgers"]["reporting"]["gpu_seconds"] > 0


def toy_source(tmp_path):
    run = tmp_path / "source"
    source = {"train": [{"question": f"q{i}", "answer": "2"} for i in range(40)], "val": []}
    gate.atomic_json(run / "prompts.json", source)
    c = {"source_run": str(run), "n": 40, "config": {"seed": 0, "drift": 100, "topk_frac": .1},
         "scope": {"gpu_type": "H100"}, "budget_gpu_seconds": 1000., "max_steps": 100000}
    out = tmp_path / "point"
    gate.atomic_json(out / "contract.json", c)
    return out, c


def test_random_subset_is_fixed_shared_between_random_arms_and_needs_no_scores(tmp_path):
    out, c = toy_source(tmp_path)
    one = gpu.freeze_subset(out, c, "random_full")
    first = one.read_bytes()
    assert gpu.freeze_subset(out, c, "random_full").read_bytes() == first
    two = gpu.freeze_subset(out, c, "random_reduced")
    assert gate.read(one)["selected_idx"] == gate.read(two)["selected_idx"]
    assert len(gate.read(one)["train"]) == 4
    assert not (Path(c["source_run"]) / "rollouts_behavior_train.jsonl").exists()
    with pytest.raises(ValueError, match="frozen contract"):
        gpu.freeze_subset(out, c, "random_full", [0, 1, 2, 3])


def test_random_full_never_profiles_or_scores(tmp_path, monkeypatch):
    out, c = toy_source(tmp_path)
    monkeypatch.setattr(gpu, "verify", lambda out: c)
    monkeypatch.setattr(gpu, "initial", lambda *a: pytest.fail("random baseline profiled"))
    monkeypatch.setattr(gpu, "select_once", lambda *a: pytest.fail("random baseline scored"))
    monkeypatch.setattr(gpu, "policy", lambda *a: Path(c["source_run"]))
    monkeypatch.setattr(gpu, "rewards", lambda *a: {"q0": .5})
    gate.atomic_json(out / "random_full/policy/budget_stop.json", {"completed_steps": 101, "stop_reason": "no_block_fits"})
    for i in range(4):
        gate.atomic_json(out / "random_full/evaluation" / f"shard-{i}.done.json", {})
    gpu.run_arm(out, {"mode": "study", "eval_timeout": 5}, "random_full", ["0", "1", "2", "3"], {})
    assert gate.read(out / "random_full/result.json")["matched_budget"]
    paid = gpu.spent(out / "random_full")
    gpu.run_arm(out, {"mode": "study"}, "random_full", [], {})
    assert gpu.spent(out / "random_full") == paid


def test_frozen_initial_is_reused_without_profiling(tmp_path, monkeypatch):
    out, c = toy_source(tmp_path)
    first = {"payload": {"action": "select"}, "gpu_seconds": 2.}
    gate.atomic_json(out / "initial.json", first)
    monkeypatch.setattr(gpu, "meter", lambda *a, **k: pytest.fail("initial decision repeated"))
    assert gpu.initial(out, {}, c) == first


def test_failure_on_one_arm_does_not_block_other_arms(tmp_path, monkeypatch):
    out, c = toy_source(tmp_path)
    gate.atomic_json(tmp_path / "suite.json", {"mode": "study"})
    c["evaluation"] = {"val": [], "provenance": {}}
    gate.atomic_json(out / "contract.json", c)
    monkeypatch.setitem(sys.modules, "additive_experiment", SimpleNamespace(model_environment=lambda c: {}))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.setenv("OM_NODE_LOCK_HELD", "1")
    monkeypatch.setattr(gpu.subprocess, "check_output", lambda *a, **k: "H100\n"*4)
    monkeypatch.setattr(gpu, "entries", lambda root: [out])
    monkeypatch.setattr(gpu, "status", lambda root: {})
    seen = []
    def run(out, suite, arm, devices, env):
        seen.append(arm)
        if arm == "random_full":
            raise ValueError("intentional")
    monkeypatch.setattr(gpu, "run_arm", run)
    assert gpu.work(tmp_path) == 1
    assert seen == list(gpu.study.BRANCHES)


def test_gpu_shell_plan_and_status_need_no_cuda_or_source(tmp_path):
    env = {**os.environ, "GATE_GPU_PYTHON": sys.executable, "GATE_ROOT": str(tmp_path / "unused"), "CUDA_VISIBLE_DEVICES": ""}
    for mode in ("plan", "status"):
        result = subprocess.run(["bash", "scripts/run_selection_gate_gpu.sh", mode], cwd=ROOT, env=env,
                                capture_output=True, text=True, timeout=15, check=False)
        assert result.returncode == 0, result.stderr
    assert not (tmp_path / "unused").exists()


def test_deployment_selector_failure_freezes_random_without_a_second_gate(tmp_path, monkeypatch):
    out, c = toy_source(tmp_path)
    monkeypatch.setattr(gpu, "verify", lambda out: c)
    called = []
    def initial(*args):
        called.append(1)
        return {"gpu_seconds": 2., "payload": {"action": "select"}}
    monkeypatch.setattr(gpu, "initial", initial)
    def bad_selector(*a, **k):
        raise ValueError("invalid score")
    monkeypatch.setattr(gpu, "select_once", bad_selector)
    monkeypatch.setattr(gpu, "policy", lambda *a: Path(c["source_run"]))
    monkeypatch.setattr(gpu, "rewards", lambda *a: {"q0": .5})
    gate.atomic_json(out / "deployment/policy/budget_stop.json", {"completed_steps": 101, "stop_reason": "no_block_fits"})
    for i in range(4):
        gate.atomic_json(out / "deployment/evaluation" / f"shard-{i}.done.json", {})
    suite = {"mode": "deploy", "eval_timeout": 5}
    gpu.run_arm(out, suite, "deployment", ["0", "1", "2", "3"], {})
    assert gate.read(out / "deployment/execution.json")["reason"] == "selector_failed"
    result = gate.read(out / "deployment/result.json")
    assert result["action"] == "random" and result["budget_gpu_seconds"] == 998.
    assert result["cost"]["ledgers"]["deployment"]["gpu_seconds"] > 0
    assert result["cost"]["ledgers"]["research"]["gpu_seconds"] == 0
    gpu.run_arm(out, suite, "deployment", [], {})
    assert len(called) == 1


def test_closed_initial_payload_recovers_publication_without_new_measurement(tmp_path, monkeypatch):
    out, c = toy_source(tmp_path)
    payload = {"action": "random"}
    gate.atomic_json(out / "measurement/payload.json", payload)
    gpu.meter(out / "measurement", "profile", "H100", action=lambda: None)
    monkeypatch.setattr(gpu, "meter", lambda *a, **k: pytest.fail("measured again"))
    first = gpu.initial(out, {}, c)
    assert first["payload"] == payload and first["gpu_seconds"] > 0
    assert gate.read(out / "initial.json") == first
