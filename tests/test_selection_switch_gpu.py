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
raise SystemExit(s.work(root, idle_timeout=0))
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
        assert len({row["pid"] for row in results.values()}) >= 2
        assert any(left != right and claims[left]["pid"] != claims[right]["pid"]
                   and claims[left]["started"] < results[right]["finished"]
                   and claims[right]["started"] < results[left]["finished"]
                   for left in claims for right in claims)
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
                worker.communicate()


def test_missing_development_labels_do_not_hold_fit_lock(tmp_path):
    with base.lease(tmp_path / ".fit.lock"):
        assert switch.fit_once(tmp_path) is False


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
