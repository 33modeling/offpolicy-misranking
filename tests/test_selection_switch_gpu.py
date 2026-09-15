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


def cache_predecessor():
    hashes = switch.code_hashes()
    hashes.update({
        "src/grads.py": "112d6a18747d324d91d3fdea0ae316ba0b5248dc7f12eb72eaf51bb391688245",
        "src/selection_switch_gpu.py": "d2974888651e91badd30332569bd62c12c40c0bf6609e2fe10f250eed6a3276d",
        "src/selection_gate_gpu.py": "cdac209f9dae10513782d865be615e825ea392605551df88bfa94a213f8f3f13",
    })
    assert core.fingerprint(hashes) == switch.PRE_KV_CACHE_CODE
    return hashes


def initial_predecessor():
    hashes = cache_predecessor()
    hashes["src/selection_switch_gpu.py"] = "118f7d0a7ecfe6b3e9a06cf21ac93f29cb5c784c40689f7202f3c5f0da996c02"
    assert core.fingerprint(hashes) == switch.PRE_INITIAL_SCORE_CODE
    return hashes


def code_compat_predecessor():
    hashes = switch.code_hashes()
    hashes["src/selection_switch_gpu.py"] = "f87119d0f40cc0166f9095049b234c0b25a6fbaf0b910cc13bef688ce494f753"
    assert core.fingerprint(hashes) == switch.PRE_CODE_COMPAT_CODE
    return hashes


@pytest.mark.parametrize("predecessor", [initial_predecessor, code_compat_predecessor])
def test_code_compat_resumes_original_and_latest_frozen_runs(tmp_path, predecessor):
    frozen = {"schema": rule.SCHEMA, "code_hashes": predecessor(), "budget_gpu_seconds": 1000.}
    core.atomic_json(tmp_path / "switch.json", frozen)
    core.atomic_json(tmp_path / "prefixes/seed-0/prefix-25.json", {"checkpoint": "existing"})
    base.journal(tmp_path / "cost.jsonl", {"state": "started", "event_id": "unknown-cost"})
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert switch.manifest(tmp_path) == frozen
    assert switch.manifest(tmp_path) == frozen
    assert {p: p.read_bytes() for p in before} == before
    receipt = core.read(tmp_path / "code-compat-runtime.json")
    assert receipt["runtime_code_hashes"] == switch.code_hashes()
    assert receipt["switch_sha256"] == base.digest(tmp_path / "switch.json")
    with pytest.raises(ValueError, match="unknown cost"):
        base.spent(tmp_path)


@pytest.mark.parametrize("receipt_count", range(5))
def test_code_compat_preserves_latest_and_partially_written_receipt_chains(tmp_path, monkeypatch, receipt_count):
    frozen = {"schema": rule.SCHEMA, "code_hashes": cache_predecessor()}
    core.atomic_json(tmp_path / "switch.json", frozen)
    previous = code_compat_predecessor()
    with monkeypatch.context() as patch:
        patch.setattr(switch, "code_hashes", lambda: previous)
        switch.manifest(tmp_path)
    receipts = ["kv-cache-runtime.json", "cost-runtime.json", "prefix-resume-runtime.json",
                "worker-logs-runtime.json", "code-compat-runtime.json"]
    for name in receipts[receipt_count:]:
        (tmp_path / name).unlink()
    before = {p: p.read_bytes() for p in tmp_path.glob("*.json")}
    assert switch.manifest(tmp_path) == frozen
    assert switch.manifest(tmp_path) == frozen
    assert {p: p.read_bytes() for p in before} == before
    assert core.read(tmp_path / "code-compat-runtime.json")["runtime_code_hashes"] == switch.code_hashes()


@pytest.mark.parametrize("receipt", ["kv-cache", "cost", "prefix-resume", "worker-logs", "code-compat"])
def test_code_compat_rejects_tampered_receipts(tmp_path, receipt):
    core.atomic_json(tmp_path / "switch.json", {"schema": rule.SCHEMA, "code_hashes": initial_predecessor()})
    switch.manifest(tmp_path)
    path = tmp_path / f"{receipt}-runtime.json"
    value = core.read(path)
    value["cost_policy"] = "ignore previous work"
    core.atomic_json(path, value)
    with pytest.raises(ValueError, match="frozen contract changed"):
        switch.manifest(tmp_path)


def test_code_error_identifies_unreviewed_file_and_full_fingerprints(monkeypatch):
    recorded = initial_predecessor()
    current = switch.code_hashes()
    current["src/train_policy_grpo.py"] = "unreviewed"
    monkeypatch.setattr(switch, "code_hashes", lambda: current)
    with pytest.raises(ValueError, match="scientific code changed") as error:
        switch.validate_code_hashes(recorded)
    assert "src/train_policy_grpo.py" in str(error.value)
    assert "unreviewed" in str(error.value)
    assert core.fingerprint(recorded) in str(error.value)
    assert core.fingerprint(current) in str(error.value)


def test_code_compat_four_processes_share_one_migration(tmp_path):
    import subprocess

    frozen = {"schema": rule.SCHEMA, "code_hashes": initial_predecessor(), "budget_gpu_seconds": 1000.}
    core.atomic_json(tmp_path / "switch.json", frozen)
    base.journal(tmp_path / "cost.jsonl", {"state": "started", "event_id": "unknown-cost"})
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    command = [sys.executable, "-c", "import sys; from pathlib import Path; "
               "import selection_switch_gpu as s; s.manifest(Path(sys.argv[1])); s.manifest(Path(sys.argv[1]))",
               str(tmp_path)]
    env = {**os.environ, "PYTHONPATH": str(base.ROOT / "src"), "CUDA_VISIBLE_DEVICES": "",
           "PYTHONDONTWRITEBYTECODE": "1"}
    workers = []
    try:
        for _ in range(4):
            workers.append(subprocess.Popen(command, cwd=base.ROOT, env=env, stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE, text=True))
        for worker in workers:
            stdout, stderr = worker.communicate(timeout=20)
            assert worker.returncode == 0, stdout + stderr
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
            worker.communicate(timeout=20)
    assert switch.manifest(tmp_path) == frozen
    assert {p: p.read_bytes() for p in before} == before


@pytest.mark.parametrize("compatible", [True, False])
def test_check_code_launcher_is_read_only_and_needs_no_gpu(tmp_path, compatible):
    import json
    import subprocess

    recorded = initial_predecessor()
    if not compatible:
        recorded["src/selection_switch.py"] = "unreviewed"
    core.atomic_json(tmp_path / "switch.json", {"schema": rule.SCHEMA, "code_hashes": recorded})
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    env = {**os.environ, "SWITCH_ROOT": str(tmp_path), "SWITCH_PYTHON": sys.executable,
           "OM_WORK": str(tmp_path / "absent-work"), "CUDA_VISIBLE_DEVICES": ""}
    result = subprocess.run(["bash", "scripts/run_selection_switch.sh", "check-code"],
                            cwd=base.ROOT, env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == (0 if compatible else 1), result.stdout + result.stderr
    if compatible:
        assert json.loads(result.stdout)["status"] == "compatible"
    else:
        assert "src/selection_switch.py" in result.stderr
        assert core.fingerprint(recorded) in result.stderr
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
    assert not list(tmp_path.rglob("*.lock"))


def test_cache_fix_resumes_exact_predecessor_and_preserves_artifacts(tmp_path):
    frozen = {"schema": rule.SCHEMA, "code_hashes": cache_predecessor(), "budget_gpu_seconds": 1000.}
    core.atomic_json(tmp_path / "switch.json", frozen)
    core.atomic_json(tmp_path / "prefixes/seed-0/prefix-25.json", {"checkpoint": "existing"})
    base.journal(tmp_path / "cost.jsonl", {"gpu_seconds": 12.})
    original = {p: base.digest(p) for p in tmp_path.rglob("*") if p.is_file()}
    assert switch.manifest(tmp_path) == frozen
    receipt = core.read(tmp_path / "kv-cache-runtime.json")
    assert receipt["runtime_code_hashes"] == switch.code_hashes()
    assert receipt["original_code_hashes"] == frozen["code_hashes"]
    assert switch.manifest(tmp_path) == frozen
    assert {p: base.digest(p) for p in original} == original
    receipt["runtime_code_hashes"]["src/grads.py"] = "changed"
    core.atomic_json(tmp_path / "kv-cache-runtime.json", receipt)
    with pytest.raises(ValueError, match="frozen contract changed"):
        switch.manifest(tmp_path)


@pytest.mark.parametrize("filename", ["src/grads.py", "src/selection_gate_gpu.py", "src/selection_switch.py", "extra.py"])
def test_cache_fix_rejects_unrelated_runtime_changes(filename, monkeypatch):
    original = cache_predecessor()
    current = switch.code_hashes()
    current[filename] = "unreviewed"
    monkeypatch.setattr(switch, "code_hashes", lambda: current)
    with pytest.raises(ValueError, match="scientific code changed"):
        switch.validate_code_hashes(original)


def test_cache_fix_rejects_unknown_predecessor():
    original = cache_predecessor()
    original["src/grads.py"] = "unknown version"
    with pytest.raises(ValueError, match="scientific code changed"):
        switch.validate_code_hashes(original)


@pytest.mark.parametrize("already_patched", [False, True])
def test_cost_fix_preserves_preexisting_kv_runtime_receipt(tmp_path, already_patched):
    patched = cache_predecessor()
    patched.update({"src/grads.py": switch.KV_CACHE_GRADS,
                    "src/selection_switch_gpu.py": "52e2f4f6da51fe3693281ef4894b389b76811352795d97f5eea56e01f88acd90"})
    assert core.fingerprint(patched) == switch.PRE_COST_CODE
    frozen = {"schema": rule.SCHEMA, "code_hashes": patched if already_patched else cache_predecessor()}
    core.atomic_json(tmp_path / "switch.json", frozen)
    receipt = {"schema": "selection-switch-kv-cache-runtime/v1",
               "switch_sha256": base.digest(tmp_path / "switch.json"),
               "original_code_hashes": frozen["code_hashes"], "runtime_code_hashes": patched,
               "change": "teacher-forced scoring forwards explicitly disable KV cache",
               "cost_policy": "retain all previous costs and the original branch allocation"}
    core.atomic_json(tmp_path / "kv-cache-runtime.json", receipt)
    original = base.digest(tmp_path / "kv-cache-runtime.json")
    assert switch.manifest(tmp_path) == frozen
    assert base.digest(tmp_path / "kv-cache-runtime.json") == original
    assert core.read(tmp_path / "cost-runtime.json")["runtime_code_hashes"] == switch.code_hashes()
    assert switch.manifest(tmp_path) == frozen


@pytest.mark.parametrize("original_version", ["before_cache", "before_cost", "before_prefix"])
def test_prefix_resume_preserves_previous_runtime_receipts(tmp_path, original_version):
    previous = cache_predecessor()
    previous.update({"src/grads.py": switch.KV_CACHE_GRADS,
                     "src/selection_gate_gpu.py": "91b1d60ef7266dd59e5b534a0c7a0cf531075746b89d0d4126e455f935577005",
                     "src/selection_switch_gpu.py": "43ea3c42217f47817312876a9d6e8bea37b84e616d98e722bc786ac6c826250b"})
    assert core.fingerprint(previous) == switch.PRE_PREFIX_RESUME_CODE
    original = cache_predecessor() if original_version != "before_prefix" else previous
    if original_version == "before_cost":
        original.update({"src/grads.py": switch.KV_CACHE_GRADS,
                         "src/selection_switch_gpu.py": "52e2f4f6da51fe3693281ef4894b389b76811352795d97f5eea56e01f88acd90"})
    frozen = {"schema": rule.SCHEMA, "code_hashes": original}
    core.atomic_json(tmp_path / "switch.json", frozen)
    if original_version != "before_prefix":
        kv = {"schema": "selection-switch-kv-cache-runtime/v1",
              "switch_sha256": base.digest(tmp_path / "switch.json"),
              "original_code_hashes": original, "runtime_code_hashes": previous,
              "change": "teacher-forced scoring forwards explicitly disable KV cache",
              "cost_policy": "retain all previous costs and the original branch allocation"}
        core.atomic_json(tmp_path / "kv-cache-runtime.json", kv)
        core.atomic_json(tmp_path / "cost-runtime.json", {
            "schema": "selection-switch-cost-runtime/v1",
            "switch_sha256": base.digest(tmp_path / "switch.json"),
            "kv_cache_runtime_sha256": base.digest(tmp_path / "kv-cache-runtime.json"),
            "runtime_code_hashes": previous,
            "change": "protect phase startup, recover atomic finish receipts, skip busy publication tasks",
            "cost_policy": "recover only from completion evidence or operator-reported termination duration"})
    before = {path: base.digest(path) for path in tmp_path.glob("*.json")}
    assert switch.manifest(tmp_path) == frozen
    assert switch.manifest(tmp_path) == frozen
    assert {path: base.digest(path) for path in before} == before
    assert core.read(tmp_path / "prefix-resume-runtime.json")["runtime_code_hashes"] == switch.code_hashes()
    receipt = core.read(tmp_path / "cost-runtime.json")
    receipt["cost_policy"] = "ignore costs"
    core.atomic_json(tmp_path / "cost-runtime.json", receipt)
    with pytest.raises(ValueError, match="frozen contract changed"):
        switch.manifest(tmp_path)


@pytest.mark.parametrize("legacy", [False, True])
def test_switch_protocol_validates_design_and_parent_runtime(tmp_path, legacy):
    hashes = cache_predecessor() if legacy else switch.code_hashes()
    core.atomic_json(tmp_path / "switch.json", {"schema": rule.SCHEMA, "code_hashes": hashes})
    child = switch.child_root(tmp_path, 0, 25)
    p = {"schema": rule.SCHEMA, "schedule": rule.SCHEDULE, "mode": "study",
         "arms": list(rule.DEV_ARMS), "selector": "fresh_r", "model": None,
         "role": "development", "max_measurement_fraction": .01, "recent_window": 20,
         "code_hashes": hashes}
    core.atomic_json(child / "net_protocol.json", p)
    assert switch.protocol(child) == p
    p["selector"] = "low_order"
    core.atomic_json(child / "net_protocol.json", p)
    with pytest.raises(ValueError, match="invalid switch experimental design"):
        switch.protocol(child)
    p["selector"] = "fresh_r"
    p["code_hashes"] = switch.code_hashes() if legacy else cache_predecessor()
    core.atomic_json(child / "net_protocol.json", p)
    with pytest.raises(ValueError, match="state code binding differs"):
        switch.protocol(child)


@pytest.mark.parametrize("migrated", [False, True])
def test_worker_logging_upgrade_preserves_69bec8d_run_and_receipts(tmp_path, migrated):
    previous = cache_predecessor()
    previous.update({"src/grads.py": switch.KV_CACHE_GRADS,
                     "src/selection_gate_gpu.py": "91b1d60ef7266dd59e5b534a0c7a0cf531075746b89d0d4126e455f935577005",
                     "src/selection_switch_gpu.py": "a8d93b39270c576e9b29300863dcedc108a115c826748c8add65e4a14b86cb99"})
    assert core.fingerprint(previous) == switch.PRE_WORKER_LOGS_CODE
    frozen = {"schema": rule.SCHEMA, "code_hashes": cache_predecessor() if migrated else previous}
    core.atomic_json(tmp_path / "switch.json", frozen)
    if migrated:
        core.atomic_json(tmp_path / "kv-cache-runtime.json", {
            "schema": "selection-switch-kv-cache-runtime/v1", "switch_sha256": base.digest(tmp_path / "switch.json"),
            "original_code_hashes": frozen["code_hashes"], "runtime_code_hashes": previous,
            "change": "teacher-forced scoring forwards explicitly disable KV cache",
            "cost_policy": "retain all previous costs and the original branch allocation"})
        core.atomic_json(tmp_path / "cost-runtime.json", {
            "schema": "selection-switch-cost-runtime/v1", "switch_sha256": base.digest(tmp_path / "switch.json"),
            "kv_cache_runtime_sha256": base.digest(tmp_path / "kv-cache-runtime.json"), "runtime_code_hashes": previous,
            "change": "protect phase startup, recover atomic finish receipts, skip busy publication tasks",
            "cost_policy": "recover only from completion evidence or operator-reported termination duration"})
        core.atomic_json(tmp_path / "prefix-resume-runtime.json", {
            "schema": "selection-switch-prefix-resume-runtime/v1", "switch_sha256": base.digest(tmp_path / "switch.json"),
            "cost_runtime_sha256": base.digest(tmp_path / "cost-runtime.json"), "runtime_code_hashes": previous,
            "change": "resume research prefixes with explicitly unknown historical costs; keep active queue peers",
            "cost_policy": "preserve open research events; never waive deployment budget accounting"})
    before = {path: base.digest(path) for path in tmp_path.glob("*.json")}
    assert switch.manifest(tmp_path) == frozen
    assert switch.manifest(tmp_path) == frozen
    assert {path: base.digest(path) for path in before} == before
    assert core.read(tmp_path / "worker-logs-runtime.json")["runtime_code_hashes"] == switch.code_hashes()


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


QUEUE_WORKER = '''
import json, os, sys, time
from pathlib import Path
from types import SimpleNamespace
import selection_switch_gpu as s
sys.modules['additive_experiment'] = SimpleNamespace(model_environment=lambda c: {})
s.rule.DEV_SEEDS, s.rule.TEST_SEEDS, s.rule.STEPS = (0, 1, 2), (), (25,)
root = Path(sys.argv[1])
s.manifest = lambda _: {'sources': {str(i): {'config': {}} for i in range(3)}}
s.admitted_devices = lambda _: list('0123')
s.fit_once = lambda _: False
s.status = lambda _: None
def publish(root, seed, step):
    child = s.child_root(root, seed, step)
    out = child / 'points' / 'view-25'
    time.sleep(.05)
    s.core.atomic_json(out / 'contract.json', {'seed': seed})
    s.core.atomic_json(child / 'suite.json', {})
    s.core.atomic_json(child / 'net_protocol.json', {'mode': 'study', 'arms': list(s.rule.DEV_ARMS)})
s.publish_state = publish
s.protocol = lambda child: s.core.read(child / 'net_protocol.json')
s.base.entries = lambda child: iter((child / 'points').iterdir())
def decide(out, suite, protocol, arm, env):
    s.base.bind(out / arm / 'decision.json', {'action': 'random'})
s.runtime.decision = decide
def run(out, suite, protocol, arm, devices, env):
    directory = out / arm
    with (directory / 'claim.json').open('x') as handle:
        json.dump({'pid': os.getpid(), 'started': time.monotonic()}, handle)
    time.sleep(.15)
    s.core.atomic_json(directory / 'result.json', {'pid': os.getpid(), 'finished': time.monotonic()})
s.runtime.run_arm = run
if len(sys.argv) > 2:
    (root / f'ready-{os.getpid()}').touch()
    deadline = time.monotonic() + 5
    while not (root / 'go').exists():
        if time.monotonic() > deadline: raise RuntimeError('test start timeout')
        time.sleep(.01)
    sleep = time.sleep
    s.time.sleep = lambda seconds: sleep(.01 if seconds == 15 else seconds)
raise SystemExit(s.work(root, idle_timeout=1 if len(sys.argv) > 2 else 0))
'''


def ready_queue(root):
    for seed in range(3):
        core.atomic_json(switch.prefix_dir(root, seed) / "prefix-25.json", {})


def test_busy_publication_is_skipped_while_other_seeds_run(tmp_path):
    import subprocess
    ready_queue(tmp_path)
    child = switch.child_root(tmp_path, 0, 25)
    with base.lease(child / ".publish.lock"):
        result = subprocess.run([sys.executable, "-c", QUEUE_WORKER, str(tmp_path)],
                                capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (child / "net_protocol.json").exists()
    assert len(list(tmp_path.glob("states/*/points/*/*/result.json"))) == 4


def test_four_nodes_claim_switch_tasks_concurrently_without_duplicates(tmp_path):
    import subprocess
    import time
    ready_queue(tmp_path)
    workers = [subprocess.Popen([sys.executable, "-c", QUEUE_WORKER, str(tmp_path), "wait"],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(4)]
    try:
        deadline = time.monotonic() + 5
        while len(list(tmp_path.glob("ready-*"))) < 4 and time.monotonic() < deadline:
            time.sleep(.01)
        assert len(list(tmp_path.glob("ready-*"))) == 4
        (tmp_path / "go").touch()
        for worker in workers:
            stdout, stderr = worker.communicate(timeout=10)
            assert worker.returncode == 0, stdout + stderr
        claims = {path.parent: core.read(path) for path in tmp_path.glob("states/*/points/*/*/claim.json")}
        results = {path.parent: core.read(path) for path in tmp_path.glob("states/*/points/*/*/result.json")}
        assert len(claims) == len(results) == 6
        assert len({row["pid"] for row in results.values()}) == 4
        assert max(sum(claims[path]["started"] <= instant < results[path]["finished"] for path in claims)
                   for instant in (row["started"] for row in claims.values())) == 4
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
                worker.communicate()


def test_missing_development_labels_do_not_hold_fit_lock(tmp_path):
    with base.lease(tmp_path / ".fit.lock"):
        assert switch.fit_once(tmp_path) is False


def test_waiting_node_stays_for_active_peer_past_local_idle_limit(tmp_path, monkeypatch, capsys):
    import time
    directory = tmp_path / "states/s0-t25/points/view-25/selection_reduced"
    core.atomic_json(directory.parent / "measurement/progress.json", {
        "state": "running", "updated": time.time(), "host": "node-2", "pid": 123, "phase": "measure"})
    busy = switch.busy_task("s0/t25/selection_reduced", directory)
    sleeps = []
    monkeypatch.setattr(switch.time, "sleep", sleeps.append)
    assert switch.wait_for_peers([busy], last_progress=0., idle_timeout=0.)
    assert sleeps == [15]
    assert "node-2:123(measure)" in capsys.readouterr().out


@pytest.mark.parametrize("state,age", [("running", 61.), ("failed", 0.), ("finished", 0.)])
def test_dead_or_finished_peer_does_not_keep_node_waiting(tmp_path, monkeypatch, state, age):
    import time
    core.atomic_json(tmp_path / "progress.json", {"state": state, "updated": time.time()-age})
    monkeypatch.setattr(switch.time, "sleep", lambda _: pytest.fail("stale peer kept node waiting"))
    assert not switch.wait_for_peers([switch.busy_task("prefix", tmp_path)], last_progress=0., idle_timeout=0.)


def test_existing_link_cannot_silently_point_to_other_policy(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir(); second.mkdir()
    switch.link(tmp_path / "parent", first)
    switch.link(tmp_path / "parent", first)
    with pytest.raises(ValueError): switch.link(tmp_path / "parent", second)


@pytest.mark.parametrize("schema,recorded", [
    ("offpolicy-oracle-validation-split/v3", None),
    ("offpolicy-oracle-validation-split/v3", {"validated_rows": 0}),
    ("offpolicy-oracle-validation-split/v2", {"validated_rows": 32}),
])
def test_legacy_initial_scores_reconstructed_on_cpu_without_mutating_source(tmp_path, schema, recorded):
    import torch
    from experiment import score_oracle_microgroups, split_validation_directions
    cfg = {"proj_dim": 3}
    prompts = {"train": [{}, {}], "val": [{}]*8}
    torch.manual_seed(72)
    groups = {i: torch.randn(8, 3) for i in range(2)}
    validation = torch.randn(8, 3)
    torch.save(groups, tmp_path / "oracle_micro_groups.pt")
    torch.save(validation, tmp_path / "val_groups.pt")
    core.atomic_json(tmp_path / "oracle_protocol.json", {"schema": schema, "generation_validation": recorded})
    before = {p.name: base.digest(p) for p in tmp_path.iterdir()}
    scores, info = switch.initial_fresh_scores(tmp_path, cfg, prompts, {"validated_rows": 128})
    expected = {i: score_oracle_microgroups(g, *split_validation_directions(validation))[1]["r"] for i, g in groups.items()}
    assert scores == expected
    assert info["method"] == "cpu_reconstructed_fresh_r_from_saved_gradients"
    assert before == {p.name: base.digest(p) for p in tmp_path.iterdir()}


def test_verified_initial_scalar_scores_need_no_tensors(tmp_path):
    core.atomic_json(tmp_path / "oracle_protocol.json", {
        "schema": "offpolicy-oracle-validation-split/v3", "generation_validation": {"validated_rows": 16}})
    core.atomic_json(tmp_path / "scores_splithalf.json", {"0": {"r": .2}, "1": {"r": -.1}})
    scores, info = switch.initial_fresh_scores(tmp_path, {}, {"train": [{}, {}]}, {"validated_rows": 16})
    assert scores == {0: .2, 1: -.1}
    assert info["method"] == "verified_v3_scalar_scores"


def test_legacy_without_gradients_explains_exact_missing_evidence(tmp_path):
    core.atomic_json(tmp_path / "oracle_protocol.json", {"schema": "old"})
    with pytest.raises(ValueError, match="CPU repair needs.*oracle_micro_groups.pt"):
        switch.initial_fresh_scores(tmp_path, {"proj_dim": 3}, {"train": [{}]}, {"validated_rows": 8})


def test_initial_score_repair_never_ignores_changed_recorded_inputs(tmp_path):
    core.atomic_json(tmp_path / "oracle_protocol.json", {"schema": "old", "generation_validation": {
        "artifact_sha256": {"rollouts.jsonl": "wrong"}}})
    (tmp_path / "rollouts.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="input changed"):
        switch.initial_fresh_scores(tmp_path, {}, {"train": [{}]}, {"validated_rows": 8})
