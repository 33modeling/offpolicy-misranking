"""Pair-specific operational regressions; never run a real GPU workload."""
import os
import subprocess
import sys

import pytest

import selection_gate as core
import selection_gate_gpu as base
import pinned_trainers
import selector_pair as pair
import selector_pair_gpu as gpu
from test_selector_pair import allocation
from test_selector_pair_gpu import bootstrap_predecessor, fake_study


def previous_runtime():
    value = gpu.code_hashes()
    value.update({
        "src/selection_switch_gpu.py": "48d593db8a6abd8d3363a5fa2b8421cb0528fd4f6e8068e2330d141c5f4f4122",
        "src/selector_pair_gpu.py": "c1b2d1478bdfb56ef1e7403df084011443552e825db626cde70bf42ed468aa45",
        "scripts/run_selector_pair.sh": "25e3f9156caa200205f12d1501d82da48299947a649b0c7abcf7e815dfac628c",
    })
    assert core.fingerprint(value) == gpu.PRE_PAIR_OPERATIONS_CODE
    return value


def frozen(root, hashes=None):
    p = {"schema": pair.SCHEMA, "target_reward": .35,
         "code_hashes": hashes or gpu.code_hashes(), "branch_manifests": {}}
    p["protocol_id"] = core.fingerprint(p)
    core.atomic_json(root / "pair.json", p)
    return p


@pytest.mark.parametrize("old_runtime", [False, True])
def test_live_status_is_readonly_and_does_not_need_controller_lock(tmp_path, old_runtime):
    frozen(tmp_path, previous_runtime() if old_runtime else None)
    with base.lease(tmp_path / ".pair.lock"):
        before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
        process = subprocess.run([sys.executable, pinned_trainers.COMMAND, str(base.ROOT / "src/selector_pair_gpu.py"),
            "status", "--root", str(tmp_path)], capture_output=True, text=True, timeout=15,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        assert process.returncode == 0, process.stderr
        assert "completed" in process.stdout
        assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


def test_status_missing_root_never_initializes(tmp_path, capsys):
    root = tmp_path / "absent"
    gpu.read_status(root)
    assert "not prepared" in capsys.readouterr().out
    assert not root.exists()


@pytest.mark.parametrize("migrated", [False, True])
def test_operations_upgrade_preserves_frozen_work_and_prior_receipts(tmp_path, monkeypatch, migrated):
    old = previous_runtime()
    p = frozen(tmp_path, bootstrap_predecessor() if migrated else old)
    if migrated:
        from test_selector_pair_lock_migration import prior_receipts
        prior_receipts(tmp_path, p["code_hashes"], old)
        # This exact historical runtime predates the operations receipt.
        (tmp_path / "pair-operations-runtime.json").unlink()
    core.atomic_json(tmp_path / "saved/decision.json", {"budget": 87120, "target": .35})
    (tmp_path / "saved/checkpoint.bin").write_bytes(b"preserve checkpoint")
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    for _ in range(2):
        assert gpu.manifest(tmp_path) == p
        assert all(path.read_bytes() == raw for path, raw in before.items())
    assert core.read(tmp_path / "pair-operations-runtime.json")["runtime_code_hashes"] == gpu.code_hashes()


def test_admission_exports_only_successful_probe_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(gpu.switch, "admitted_devices", lambda p: list("0123"))
    monkeypatch.setattr(gpu, "admission_probe", lambda root: {"NCCL_NVLS_ENABLE": "0"})
    monkeypatch.delenv("NCCL_NVLS_ENABLE", raising=False)
    assert gpu.admit_node(tmp_path, {}) == list("0123")
    assert os.environ["NCCL_NVLS_ENABLE"] == "0"


def test_failed_admission_cannot_start_development(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["pair", "run", "--root", str(tmp_path)])
    monkeypatch.setattr(gpu, "install_runtime", lambda: None)
    monkeypatch.setattr(gpu, "ensure_prepared", lambda root: {"gpu_type": "H100"})
    monkeypatch.setattr(gpu, "resource_diagnostics", lambda: None)
    monkeypatch.setattr(gpu.switch, "admitted_devices", lambda p: list("0123"))
    monkeypatch.setattr(gpu, "admission_probe", lambda root: (_ for _ in ()).throw(RuntimeError("bad NCCL")))
    monkeypatch.setattr(gpu, "develop", lambda *args: pytest.fail("work dispatched before admission"))
    with pytest.raises(gpu.NodeAdmissionError, match="bad NCCL"):
        gpu.main()


def test_transient_failure_rechecks_gpu_then_resumes_without_duplicate_results(tmp_path, fake_study, monkeypatch):
    p, calls, _ = fake_study
    original = gpu.execute
    events = []
    def execute(entry, arm, devices):
        events.append("execute")
        if len(events) == 1:
            raise RuntimeError("worker died")
        original(entry, arm, devices)
    monkeypatch.setattr(gpu, "execute", execute)
    monkeypatch.setattr(gpu, "admit_node", lambda *args: events.append("admit") or [])
    gpu.develop(tmp_path, p, [])
    assert events[:3] == ["execute", "admit", "execute"]
    assert len(calls) == 18
    assert len(list((tmp_path / "development").glob("*/result.json"))) == 9
    gpu.develop(tmp_path, p, [])
    assert len(calls) == 18


@pytest.mark.parametrize("error", [ValueError("budget exhausted"), RuntimeError("worker failed")])
def test_failed_branch_does_not_prevent_other_branches(tmp_path, fake_study, monkeypatch, error):
    p, calls, _ = fake_study
    original = gpu.execute
    attempts = []
    def execute(entry, arm, devices):
        key = (entry[0].name, entry[2]["config"]["seed"], entry[2]["config"]["drift"])
        attempts.append(key)
        if key == ("cached", 0, 25):
            raise error
        original(entry, arm, devices)
    admissions = []
    monkeypatch.setattr(gpu, "execute", execute)
    monkeypatch.setattr(gpu, "admit_node", lambda *args: admissions.append(1) or [])
    with pytest.raises(gpu.IncompletePairRun, match="1 task"):
        gpu.develop(tmp_path, p, [])
    assert len(calls) == 17
    assert len(list((tmp_path / "development").glob("*/result.json"))) == 8
    assert attempts.count(("cached", 0, 25)) == (2 if isinstance(error, RuntimeError) else 1)
    assert len(admissions) == (2 if isinstance(error, RuntimeError) else 0)
    assert core.read(tmp_path / "development-pass.json")["state"] == "WAIT"
    monkeypatch.setattr(gpu, "execute", original)
    gpu.develop(tmp_path, p, [])
    assert len(calls) == 18 and core.read(tmp_path / "development-pass.json")["state"] == "DONE"


def test_failed_re_admission_stops_before_next_branch(tmp_path, fake_study, monkeypatch):
    p, calls, _ = fake_study
    attempts = []
    def fail(*args):
        attempts.append(1)
        raise RuntimeError("CUDA worker failure")
    monkeypatch.setattr(gpu, "execute", fail)
    monkeypatch.setattr(gpu, "admit_node", lambda *args: (_ for _ in ()).throw(gpu.NodeAdmissionError("unhealthy node")))
    with pytest.raises(gpu.NodeAdmissionError, match="unhealthy"):
        gpu.develop(tmp_path, p, [])
    assert len(attempts) == 1 and not calls


def test_signal_is_not_retried_or_swallowed(tmp_path, fake_study, monkeypatch):
    p, calls, _ = fake_study
    monkeypatch.setattr(gpu, "execute", lambda *args: (_ for _ in ()).throw(KeyboardInterrupt()))
    monkeypatch.setattr(gpu, "admit_node", lambda *args: pytest.fail("signal must exit"))
    with pytest.raises(KeyboardInterrupt):
        gpu.develop(tmp_path, p, [])
    assert not calls


def test_heldout_failure_preserves_barrier_and_continues_other_states(tmp_path, fake_study, monkeypatch):
    p, calls, _ = fake_study
    gpu.develop(tmp_path, p, [])
    gpu.fit(tmp_path, p)
    gpu.freeze(tmp_path, p)
    barrier = (tmp_path / "test-decisions.json").read_bytes()
    original = gpu.execute
    def fail_one(entry, arm, devices):
        c = entry[2]["config"]
        if c["seed"] == 3 and c["drift"] == 25 and arm == "random_full":
            raise ValueError("invalid saved subset")
        original(entry, arm, devices)
    monkeypatch.setattr(gpu, "execute", fail_one)
    with pytest.raises(gpu.IncompletePairRun):
        gpu.test(tmp_path, p, [])
    assert len(calls) == 41 and len(list((tmp_path / "test").glob("*/result.json"))) == 5
    assert (tmp_path / "test-decisions.json").read_bytes() == barrier


def test_exhausted_pair_never_recharges_verification(tmp_path, monkeypatch):
    branch = tmp_path / "branches/on_policy"
    out = branch / "states/s0-t25/points/view-25"
    arm = "selection_reduced"
    directory = out / arm
    c = {"scope": {"gpu_type": "H100"}, "config": {"drift": 25}, "budget_gpu_seconds": 100.}
    for row in allocation("prior", "train", 100., 25., rc=1):
        base.journal(directory / "cost.jsonl", row)
    before = (directory / "cost.jsonl").read_bytes()
    monkeypatch.setattr(gpu, "manifest", lambda root: {})
    monkeypatch.setattr(gpu, "environment", lambda config: {})
    monkeypatch.setattr(gpu.switch.runtime, "run_arm", lambda *args: pytest.fail("exhausted work dispatched"))
    for _ in range(2):
        with pytest.raises(ValueError, match="allocation exhausted"):
            gpu.execute((branch, out, c, {}, {}), arm, list("0123"))
    assert (directory / "cost.jsonl").read_bytes() == before


@pytest.mark.parametrize("saved", ["policy/budget_stop.json", "policy/policy_train.json", "result.json"])
def test_exhausted_guard_allows_original_saved_policy_validation(tmp_path, monkeypatch, saved):
    branch = tmp_path / "branches/on_policy"
    out = branch / "states/s0-t25/points/view-25"
    directory = out / "selection_reduced"
    core.atomic_json(directory / saved, {})
    calls = []
    monkeypatch.setattr(gpu, "manifest", lambda root: {})
    monkeypatch.setattr(gpu, "environment", lambda c: {})
    monkeypatch.setattr(gpu.switch, "manifest", lambda root: {})
    monkeypatch.setattr(gpu.switch, "remaining_allocation", lambda *a: pytest.fail("reporting work blocked"))
    monkeypatch.setattr(gpu.switch.runtime, "run_arm", lambda *a: calls.append("validate/publish"))
    monkeypatch.setattr(gpu.switch, "curve_once", lambda *a: calls.append("curve"))
    monkeypatch.setattr(gpu.switch, "branch_finished", lambda *a: True)
    gpu.execute((branch, out, {}, {}, {}), "selection_reduced", list("0123"))
    assert calls == ["validate/publish", "curve"]
