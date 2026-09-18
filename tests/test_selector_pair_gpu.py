import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import selection_gate as core
import selection_gate_gpu as base
import selector_pair as pair
import selector_pair_gpu as gpu
import selector_pair_train as trainer
from test_selector_pair import allocation, contract, features, points


@pytest.fixture
def fake_study(tmp_path, monkeypatch):
    """Real orchestration/fit/freeze/report, replacing only the GPU backend."""
    p = {"protocol_id": "p", "target_reward": .35, "gpu_type": "H100"}
    curves, calls = {}, []
    def states(root, seed, step):
        entries = {}
        for name, selector in gpu.BRANCHES.items():
            branch = root / "branches" / name
            out = branch / "states" / f"s{seed}-t{step}" / "point"
            c = {**contract(), "config": {"seed": seed, "drift": step}}
            c["scope"]["selector"] = selector
            entries[name] = (branch, out, c, {"mode": "study" if seed < 3 else "test"}, {})
        return f"state-{seed}-{step}", entries
    def diagnose(entry, env):
        _, out, c, _, _ = entry
        directory = out / "measurement"
        if not (directory / "initial.json").exists():
            core.atomic_json(directory / "measurement.json", {"features": features(c["config"]["drift"])})
            base.meter(directory, "diagnose", "H100", action=lambda: None, ledger="deployment")
            core.atomic_json(directory / "initial.json", {"status": "complete",
                "report_sha256": base.digest(directory / "measurement.json"), "gpu_seconds": base.spent(directory)})
        return features(c["config"]["drift"]), core.read(directory / "initial.json"), directory
    def execute(entry, arm, devices):
        branch, out, c, _, _ = entry
        if (out / arm / "result.json").exists():
            return
        if c["config"]["seed"] in pair.TEST_SEEDS:
            assert (tmp_path / "test-decisions.json").exists()
            assert len(gpu.decisions(tmp_path, p)) == 6
        name = branch.name
        calls.append((name, c["config"]["seed"], c["config"]["drift"], arm))
        cost = 100 if "on_policy" in name else 200
        cost += 17 if name.startswith("adaptive-") else 0
        cost = 300 if arm == "random_full" else cost
        curve = {"points": points(cost), "artifact_hashes": {}, "path": str(out / arm)}
        curves[(str(out), arm)] = curve
        core.atomic_json(out / arm / "result.json", curve)
    def measured(entry, arm):
        return core.read(entry[1] / arm / "result.json")
    monkeypatch.setattr(gpu, "verify_pair", states)
    monkeypatch.setattr(gpu, "diagnostic", diagnose)
    monkeypatch.setattr(gpu, "environment", lambda _: {})
    monkeypatch.setattr(gpu, "execute", execute)
    monkeypatch.setattr(gpu, "measured_curve", measured)
    return p, calls, states


def test_full_cpu_workflow_runs_independent_adaptive_and_resumes(tmp_path, fake_study):
    p, calls, _ = fake_study
    for _ in range(2):
        gpu.develop(tmp_path, p, [])
        gpu.fit(tmp_path, p)
        gpu.freeze(tmp_path, p)
        gpu.test(tmp_path, p, [])
        gpu.report(tmp_path, p)
    assert len(calls) == 42
    assert sum(name.startswith("adaptive-") for name, *_ in calls) == 6
    report = core.read(tmp_path / "report.json")
    assert not report["missing_states"] and len(report["rows"]) == 6
    for row in report["rows"]:
        assert row["audit"]["adaptive_saving_vs_on_policy"] < -17
        assert row["curves"]["adaptive"]["path"] != row["curves"]["on_policy"]["path"]
    assert len((tmp_path / "curves.csv").read_text().splitlines()) == 1+6*4*3+9*2*3


@pytest.mark.parametrize("artifact", ["execution.json", "result.json", "policy/partial.json", "cached-select/selection.json", "cost.jsonl"])
def test_preexisting_heldout_work_is_rejected_before_freezing(tmp_path, fake_study, artifact):
    p, _, states = fake_study
    gpu.develop(tmp_path, p, [])
    gpu.fit(tmp_path, p)
    _, entries = states(tmp_path, 3, 25)
    core.atomic_json(entries["cached"][1] / "selection_full" / artifact, {})
    with pytest.raises(ValueError, match="precedes"):
        gpu.freeze(tmp_path, p)
    assert not (tmp_path / "test-decisions.json").exists()


def test_test_requires_barrier_and_model(tmp_path, fake_study):
    p, _, _ = fake_study
    with pytest.raises(FileNotFoundError):
        gpu.test(tmp_path, p, [])


def test_partial_decision_publication_resumes_without_repredicting(tmp_path, fake_study, monkeypatch):
    p, _, _ = fake_study
    gpu.develop(tmp_path, p, [])
    gpu.fit(tmp_path, p)
    original = base.bind
    class LostNode(BaseException):
        pass
    def interrupted(path, value):
        original(path, value)
        if path == tmp_path / "decisions/s3-t25/decision.json":
            raise LostNode()
    with monkeypatch.context() as patch:
        patch.setattr(base, "bind", interrupted)
        with pytest.raises(LostNode):
            gpu.freeze(tmp_path, p)
    before = (tmp_path / "decisions/s3-t25/decision.json").read_bytes()
    gpu.freeze(tmp_path, p)
    assert (tmp_path / "decisions/s3-t25/decision.json").read_bytes() == before
    assert len(gpu.decisions(tmp_path, p)) == 6


@pytest.mark.parametrize("tamper", ["model", "decision", "measurement", "barrier"])
def test_decision_artifacts_are_hash_bound(tmp_path, fake_study, tamper):
    p, _, _ = fake_study
    gpu.develop(tmp_path, p, [])
    gpu.fit(tmp_path, p)
    choices = gpu.freeze(tmp_path, p)
    if tamper == "model":
        path = tmp_path / "model.json"
    elif tamper == "decision":
        path = tmp_path / "decisions/s3-t25/decision.json"
    elif tamper == "measurement":
        path = Path(choices["s3-t25"]["diagnostic_path"]) / "measurement.json"
    else:
        path = tmp_path / "test-decisions.json"
    core.atomic_json(path, {**core.read(path), "tampered": True})
    # A barrier's own unrelated field is not scientifically relevant; change an
    # actual bound value for that case.
    if tamper == "barrier":
        core.atomic_json(path, {**core.read(path), "model_sha256": "changed"})
    with pytest.raises(ValueError):
        gpu.decisions(tmp_path, p)


def test_checkpoint_cost_record_is_inside_atomic_directory_publication(tmp_path, monkeypatch):
    monkeypatch.setenv("OM_SELECTION_COST_train123", "1")
    pending, published = tmp_path / ".checkpoint.tmp", tmp_path / "checkpoint-000030"
    pending.mkdir()
    state = {"completed_steps": 30, "adapter_sha256": "adapter"}
    def atomic_commit(path, value):
        assert (path.parent / "cost-receipt.json").exists()
        core.atomic_json(path, value)
    trainer.checkpoint_writer(atomic_commit)(pending / "checkpoint_state.json", state)
    pending.rename(published)
    receipt = core.read(published / "cost-receipt.json")
    assert receipt["event_id"] == "train123" and receipt["step"] == 30
    assert receipt["checkpoint_state_id"] == core.fingerprint(state)


def test_checkpoint_kill_before_state_commit_cannot_publish_unmetered_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("OM_SELECTION_COST_train123", "1")
    def lost(path, value):
        raise RuntimeError("lost node")
    with pytest.raises(RuntimeError):
        trainer.checkpoint_writer(lost)(tmp_path / ".checkpoint.tmp/checkpoint_state.json",
                                       {"completed_steps": 30, "adapter_sha256": "a"})
    assert not (tmp_path / ".checkpoint.tmp/checkpoint_state.json").exists()
    assert (tmp_path / ".checkpoint.tmp/cost-receipt.json").exists()


def test_final_cost_receipt_recovers_publication_gap_without_retraining(tmp_path):
    policy = tmp_path / "policy"
    core.atomic_json(policy / "adapter_model.safetensors", {"fixture": True})
    events = [*allocation("failed", "train", 0, 30, rc=1), *allocation("done", "train", 40, 100)]
    adapter = policy / "adapter_model.safetensors"
    receipt = gpu.final_receipt(tmp_path, events, 50, adapter)
    assert receipt["time"] == 140 and receipt["event_id"] == "done"
    assert gpu.final_receipt(tmp_path, events, 50, adapter) == receipt
    assert pair.cost_at_checkpoint(events, receipt, final=True)["gpu_seconds"] == 520


def test_active_cost_event_is_required(monkeypatch):
    for key in os.environ:
        if key.startswith("OM_SELECTION_COST_"):
            monkeypatch.delenv(key)
    with pytest.raises(ValueError):
        trainer.active_event()
    monkeypatch.setenv("OM_SELECTION_COST_a", "1")
    monkeypatch.setenv("OM_SELECTION_COST_b", "1")
    with pytest.raises(ValueError):
        trainer.active_event()


def test_shell_syntax_and_cpu_help_need_no_gpu():
    subprocess.run(["bash", "-n", "scripts/run_selector_pair.sh"], cwd=base.ROOT, check=True)
    process = subprocess.run([sys.executable, "src/selector_pair_gpu.py", "--help"], cwd=base.ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, capture_output=True, text=True, timeout=20)
    assert process.returncode == 0, process.stderr
    assert "--target-reward" in process.stdout


def test_prepare_uses_defaults_but_requires_real_source(tmp_path):
    root = tmp_path / "pair"
    process = subprocess.run([sys.executable, "src/selector_pair_gpu.py", "prepare", "--root", str(root)],
        cwd=base.ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": "",
                           "SWITCH_PREFIX_SOURCE": str(tmp_path / "missing-source")},
        capture_output=True, text=True, timeout=20)
    assert process.returncode != 0
    assert "certified prefix source is missing" in process.stderr
    assert core.read(root / "pair.json")["schema"] == gpu.BOOTSTRAP_SCHEMA
    for key, value in gpu.RUN_DEFAULTS.items():
        assert core.read(root / "pair.json")["configuration"][key] == value


def test_direct_entry_does_not_silently_ignore_changed_target(tmp_path):
    process = subprocess.run([sys.executable, "src/selector_pair_gpu.py", "run", "--root", str(tmp_path),
                              "--target-reward", ".2"], cwd=base.ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, capture_output=True, text=True, timeout=20)
    assert process.returncode != 0 and "cannot change a frozen run" in process.stderr


def test_new_trainer_does_not_change_legacy_command(monkeypatch, tmp_path):
    runtime = gpu.switch.runtime
    for target, names in ((runtime, ("net", "HERE", "TEST_ARMS", "SELECTORS", "CODE_FILES", "study", "protocol", "select_once", "measurement_worker", "decision")),
                          (base, ("verify", "train_command"))):
        for name in names:
            monkeypatch.setattr(target, name, getattr(target, name))
    legacy = gpu.switch.train_command
    monkeypatch.setattr(gpu.switch, "train_command", lambda *a: ["python", str(base.ROOT / gpu.switch.CURVE_TRAINER)])
    gpu.install_runtime()
    assert base.train_command(tmp_path, {}, "selection_full", 100)[1] == str(base.ROOT / gpu.TRAINER)
    assert gpu.switch.CURVE_TRAINER == "src/selection_switch_curve_train.py"


def test_report_rechecks_curves_instead_of_trusting_summary(tmp_path, fake_study):
    p, _, _ = fake_study
    gpu.develop(tmp_path, p, [])
    gpu.fit(tmp_path, p)
    gpu.freeze(tmp_path, p)
    gpu.test(tmp_path, p, [])
    path = tmp_path / "test/s3-t25/result.json"
    value = core.read(path)
    value["audit"]["h_gpu_seconds"] = 999999
    core.atomic_json(path, value)
    with pytest.raises(ValueError, match="frozen contract changed"):
        gpu.report(tmp_path, p)


def test_development_report_available_when_fit_is_censored(tmp_path, fake_study):
    p, _, _ = fake_study
    p["target_reward"] = .9
    gpu.develop(tmp_path, p, [])
    with pytest.raises(ValueError, match="unreached"):
        gpu.fit(tmp_path, p)
    gpu.report(tmp_path, p)
    report = core.read(tmp_path / "report.json")
    assert len(report["development_rows"]) == 9
    assert all(r["contrast"]["status"] == "censored" for r in report["development_rows"])
    assert len(report["missing_states"]) == 6
    assert not (tmp_path / "model.json").exists()


def test_prepare_freezes_four_separate_roots_and_rejects_changed_target(tmp_path, monkeypatch):
    prefix, matrix, root = tmp_path / "source", tmp_path / "matrix", tmp_path / "pair"
    core.atomic_json(prefix / "switch.json", {"fixture": True})
    calls = []
    def prepare(options):
        calls.append(options)
        core.atomic_json(options.root / "switch.json", {"selector": options.selector})
    monkeypatch.setattr(gpu.switch, "prepare", prepare)
    args = SimpleNamespace(root=root, matrix=matrix, prefix_source=prefix, target_reward=.35,
        budget_gpu_seconds=1000., curve_points=9, eval_k=8, gpu_type="H100", dataset="math500", eval_timeout=100.)
    gpu.prepare(args)
    frozen = (root / "pair.json").read_bytes()
    assert len(calls) == 4 and len({c.root for c in calls}) == 4
    assert all(c.prefix_source == prefix and c.curve_k == c.eval_k == 8 and c.accounting == "matched" for c in calls)
    gpu.prepare(args)
    assert (root / "pair.json").read_bytes() == frozen and len(calls) == 4
    args.target_reward = .3
    with pytest.raises(ValueError, match="frozen contract changed"):
        gpu.prepare(args)
    assert (root / "pair.json").read_bytes() == frozen


@pytest.mark.parametrize("field", ["code", "branch"])
def test_manifest_rejects_changed_runtime_or_child(tmp_path, monkeypatch, field):
    p = {"schema": pair.SCHEMA, "code_hashes": gpu.code_hashes(), "branch_manifests": {}}
    if field == "branch":
        child = tmp_path / "branches/on_policy/switch.json"
        core.atomic_json(child, {"selector": "fresh_r"})
        p["branch_manifests"]["on_policy"] = base.digest(child)
    p["protocol_id"] = core.fingerprint(p)
    core.atomic_json(tmp_path / "pair.json", p)
    if field == "code":
        monkeypatch.setattr(gpu, "code_hashes", lambda: {})
    else:
        core.atomic_json(child, {"selector": "difficulty"})
    with pytest.raises(ValueError):
        gpu.manifest(tmp_path)


def measured_fixture(tmp_path, monkeypatch):
    out, arm = tmp_path / "point", "selection_full"
    directory = out / arm
    c = {"config": {"drift": 25}, "eval_k": 8}
    events = [*allocation("score", "fresh-r-candidate", 0., 10., ledger="reporting"),
              *allocation("train", "train", 20., 100.),
              *allocation("evaluate", "evaluate", 150., 20., ledger="reporting")]
    result = {"completed_steps": 35, "rewards": {"q0": .5, "q1": .3}, "cost": core.cost_summary(events)}
    core.atomic_json(directory / "result.json", result)
    for event in events:
        base.journal(directory / "cost.jsonl", event)
    policy = directory / "policy"
    saved = policy / "curve-checkpoints/step-30"
    core.atomic_json(saved / "adapter_model.safetensors", {"step": 30})
    core.atomic_json(policy / "adapter_model.safetensors", {"step": 35})
    checkpoint = {"completed_steps": 30, "adapter_sha256": base.digest(saved / "adapter_model.safetensors")}
    core.atomic_json(saved / "checkpoint_state.json", checkpoint)
    core.atomic_json(saved / "cost-receipt.json", {"step": 30, "event_id": "train", "time": 60.,
        "adapter_sha256": checkpoint["adapter_sha256"], "checkpoint_state_id": core.fingerprint(checkpoint)})
    core.atomic_json(policy / "curve-cost/final.json", {"step": 35, "event_id": "train", "time": 110.,
        "adapter_sha256": base.digest(policy / "adapter_model.safetensors")})
    core.atomic_json(directory / "curve.json", {"points": {
        "25": {"updates": 0, "reward": .2}, "30": {"updates": 5, "reward": .3},
        "35": {"updates": 10, "reward": .4, "final": True}},
        "result_sha256": base.digest(directory / "result.json")})
    monkeypatch.setattr(gpu.switch.runtime, "validate_result", lambda *a: result)
    monkeypatch.setattr(base, "policy", lambda *a: policy)
    monkeypatch.setattr(gpu.switch, "curve_reward", lambda out, c, arm, step, k: {25: .2, 30: .3}[step])
    return (tmp_path, out, c, {}, {}), directory


def test_curve_joins_real_artifact_receipts_to_allocation_cost(tmp_path, monkeypatch):
    entry, directory = measured_fixture(tmp_path, monkeypatch)
    curve = gpu.measured_curve(entry, "selection_full")
    assert [p["gpu_seconds"] for p in curve["points"]] == [0., 200., 440.]
    assert pair.crossing(curve["points"], .35)["gpu_seconds"] == 440.
    assert curve["allocated_cost"]["ledgers"]["reporting"]["gpu_seconds"] == 120.


@pytest.mark.parametrize("tamper", ["adapter", "checkpoint", "reward", "receipt", "result"])
def test_curve_rejects_modified_cost_or_evaluation_artifact(tmp_path, monkeypatch, tamper):
    entry, directory = measured_fixture(tmp_path, monkeypatch)
    archive = directory / "policy/curve-checkpoints/step-30"
    if tamper == "adapter":
        core.atomic_json(archive / "adapter_model.safetensors", {"changed": True})
    elif tamper == "checkpoint":
        core.atomic_json(archive / "checkpoint_state.json", {"changed": True})
    elif tamper == "receipt":
        value = core.read(archive / "cost-receipt.json")
        value["event_id"] = "missing"
        core.atomic_json(archive / "cost-receipt.json", value)
    elif tamper == "reward":
        value = core.read(directory / "curve.json")
        value["points"]["30"]["reward"] = .9
        core.atomic_json(directory / "curve.json", value)
    else:
        core.atomic_json(directory / "result.json", {"changed": True})
    with pytest.raises(ValueError):
        gpu.measured_curve(entry, "selection_full")


def test_archive_copies_cost_receipt_with_checkpoint(tmp_path, monkeypatch):
    import selection_switch_curve_train as archive
    checkpoint = tmp_path / "checkpoint-000030"
    core.atomic_json(checkpoint / "adapter_model.safetensors", {"step": 30})
    core.atomic_json(checkpoint / "checkpoint_state.json", {"step": 30})
    core.atomic_json(checkpoint / "cost-receipt.json", {"event_id": "train", "time": 100})
    monkeypatch.setattr(archive, "KEEP", (*archive.KEEP, "cost-receipt.json"))
    archive.archive_then_remove(checkpoint)
    assert not checkpoint.exists()
    assert core.read(tmp_path / "curve-checkpoints/step-30/cost-receipt.json")["event_id"] == "train"


def test_real_checkpoint_writer_publishes_cost_and_archive_together(tmp_path, monkeypatch):
    import train_policy_grpo as checkpoints
    import selection_switch_curve_train as archive
    monkeypatch.setenv("OM_SELECTION_COST_train123", "1")
    monkeypatch.setattr(checkpoints, "compact_adapter", lambda _: None)
    monkeypatch.setattr(checkpoints, "_atomic_json", trainer.checkpoint_writer(checkpoints._atomic_json))
    class Model:
        def save_pretrained(self, path, **kwargs):
            core.atomic_json(path / "adapter_model.safetensors", {"fixture": True})
    class Optimizer:
        def state_dict(self):
            return {}
    for step in (30, 35):
        base.journal(tmp_path / "grpo_stats.jsonl", {"step": step})
        checkpoints._save_checkpoint(Model(), Optimizer(), tmp_path, step, 0, {})
        saved = tmp_path / f"checkpoint-{step:06d}"
        receipt = core.read(saved / "cost-receipt.json")
        assert receipt["step"] == step
        assert receipt["checkpoint_state_id"] == core.fingerprint(core.read(saved / "checkpoint_state.json"))
        monkeypatch.setattr(archive, "KEEP", (*archive.KEEP, "cost-receipt.json"))
        archive.archive_then_remove(saved)
        assert core.read(tmp_path / f"curve-checkpoints/step-{step}/cost-receipt.json") == receipt


def bootstrap_predecessor():
    hashes = gpu.code_hashes()
    hashes["src/selector_pair_gpu.py"] = "0d537c620cc778963bfb3dbd98387b9f3417b6dba43b9779b9fac05a810fe72a"
    hashes["scripts/run_selector_pair.sh"] = "cfd0d2273ddbf2cd945d6d677eff2407814efe7b1a4dfc45b3428cb22793cef7"
    assert core.fingerprint(hashes) == gpu.PRE_BOOTSTRAP_CODE
    return hashes


def defaults_predecessor():
    hashes = gpu.code_hashes()
    hashes["src/selector_pair_gpu.py"] = "6db840bcb82e8a9c2089c7e836ef5735a8b9e5a68e3707c397245f790b84e095"
    hashes["scripts/run_selector_pair.sh"] = "49ea934118bf4c88a451801fb225ed6b25cb7c9892a2ecf421a85a340dc5b3a2"
    assert core.fingerprint(hashes) == gpu.PRE_DEFAULTS_CODE
    return hashes


def test_init_creates_only_setup_and_preserves_edits(tmp_path, monkeypatch):
    monkeypatch.setenv("OM_WORK", str(tmp_path / "storage"))
    value = gpu.initialize(tmp_path)
    assert value["schema"] == gpu.BOOTSTRAP_SCHEMA
    assert value["configuration"]["target_reward"] == .35
    assert value["configuration"]["budget_gpu_seconds"] == 87120.
    assert value["configuration"]["prefix_source"] == str(tmp_path / "storage/runs/selection-switch-v1")
    assert "protocol_id" not in value and "branch_manifests" not in value
    value["configuration"]["target_reward"] = .42
    core.atomic_json(tmp_path / "pair.json", value)
    before = (tmp_path / "pair.json").read_bytes()
    assert gpu.initialize(tmp_path) == value
    assert (tmp_path / "pair.json").read_bytes() == before
    with pytest.raises(ValueError, match="setup template"):
        gpu.manifest(tmp_path)


@pytest.mark.parametrize("command", ["init", "status", "run", None])
def test_first_shell_launch_creates_pair_json_without_touching_gpus(tmp_path, command):
    root = tmp_path / "pair"
    process = subprocess.run(["bash", "scripts/run_selector_pair.sh", *([command] if command else [])], cwd=base.ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PAIR_ROOT": str(root),
             "OM_WORK": str(tmp_path / "storage"), "PAIR_PYTHON": sys.executable,
             "SWITCH_PREFIX_SOURCE": str(tmp_path / "missing-source")},
        capture_output=True, text=True, timeout=20)
    assert core.read(root / "pair.json")["schema"] == gpu.BOOTSTRAP_SCHEMA
    assert "Traceback" not in process.stderr and "FileNotFoundError" not in process.stderr
    assert "[node]" not in process.stdout and "nvidia-smi" not in process.stderr
    if command in ("run", None):
        assert process.returncode != 0 and "No GPU work started" in process.stderr
        assert "certified prefix source is missing" in process.stderr
    else:
        assert process.returncode == 0 and "ready_to_prepare" in process.stdout


@pytest.mark.parametrize("target", [None, .42, 0])
def test_old_unfrozen_setup_fills_only_empty_defaults(tmp_path, target):
    value = gpu.initialize(tmp_path)
    value["configuration"].update(target_reward=target, budget_gpu_seconds=None)
    core.atomic_json(tmp_path / "pair.json", value)
    result = gpu.initialize(tmp_path)
    assert result["configuration"] == {**value["configuration"],
        "target_reward": .35 if target is None else target, "budget_gpu_seconds": 87120.}
    before = (tmp_path / "pair.json").read_bytes()
    assert gpu.initialize(tmp_path) == result
    assert (tmp_path / "pair.json").read_bytes() == before


def test_defaults_never_change_setup_with_frozen_request(tmp_path):
    value = gpu.initialize(tmp_path)
    value["configuration"].update(target_reward=None, budget_gpu_seconds=None)
    core.atomic_json(tmp_path / "pair.json", value)
    core.atomic_json(tmp_path / "request.json", {"frozen": True})
    before = (tmp_path / "pair.json").read_bytes()
    assert gpu.initialize(tmp_path) == value
    assert (tmp_path / "pair.json").read_bytes() == before


def test_prepare_promotes_setup_only_after_real_preparation(tmp_path, monkeypatch):
    root, prefix = tmp_path / "pair", tmp_path / "prefix"
    core.atomic_json(prefix / "switch.json", {"fixture": True})
    value = gpu.initialize(root)
    value["configuration"].update(matrix=str(tmp_path / "matrix"), prefix_source=str(prefix))
    core.atomic_json(root / "pair.json", value)
    calls = []
    def prepare(options):
        assert core.read(root / "pair.json")["schema"] == gpu.BOOTSTRAP_SCHEMA
        assert options.budget_gpu_seconds == 87120.
        calls.append(options)
        core.atomic_json(options.root / "switch.json", {"selector": options.selector})
    monkeypatch.setattr(gpu.switch, "prepare", prepare)
    ready = gpu.ensure_prepared(root)
    assert ready["schema"] == pair.SCHEMA and len(calls) == 4
    assert ready["target_reward"] == .35
    assert ready["training_cap_gpu_seconds"] == 87120.
    assert gpu.ensure_prepared(root) == ready and len(calls) == 4


@pytest.mark.parametrize("legacy", [None, bootstrap_predecessor, defaults_predecessor])
def test_failed_prepare_keeps_placeholder_and_resumes_frozen_request(tmp_path, monkeypatch, legacy):
    root, prefix = tmp_path / "pair", tmp_path / "prefix"
    core.atomic_json(prefix / "switch.json", {"fixture": True})
    config = {**gpu.setup_config(), "matrix": str(tmp_path / "matrix"), "prefix_source": str(prefix),
              "target_reward": .35, "budget_gpu_seconds": 1000.}
    gpu.initialize(root, config)
    calls = []
    def prepare(options):
        calls.append(options.root.name)
        if len(calls) == 2:
            raise RuntimeError("interrupted prepare")
        core.atomic_json(options.root / "switch.json", {"selector": options.selector})
    monkeypatch.setattr(gpu.switch, "prepare", prepare)
    with pytest.raises(RuntimeError, match="interrupted prepare"):
        gpu.ensure_prepared(root)
    assert core.read(root / "pair.json")["status"] == "preparation_incomplete"
    if legacy:
        value = core.read(root / "request.json")
        value["code_hashes"] = legacy()
        core.atomic_json(root / "request.json", value)
    request = (root / "request.json").read_bytes()
    # Reproduce the predecessor's publication gap: request exists, pair absent.
    (root / "pair.json").unlink()
    draft = gpu.initialize(root)
    assert draft["configuration"] == config and draft["status"] == "preparation_incomplete"
    gpu.ensure_prepared(root)
    assert (root / "request.json").read_bytes() == request
    assert core.read(root / "pair.json")["schema"] == pair.SCHEMA


@pytest.mark.parametrize("predecessor", [bootstrap_predecessor, defaults_predecessor])
def test_startup_fix_preserves_predecessor_manifest_without_rebinding(tmp_path, predecessor):
    value = {"schema": pair.SCHEMA, "code_hashes": predecessor(), "branch_manifests": {}}
    value["protocol_id"] = core.fingerprint(value)
    core.atomic_json(tmp_path / "pair.json", value)
    before = (tmp_path / "pair.json").read_bytes()
    assert gpu.initialize(tmp_path) == value
    assert gpu.ensure_prepared(tmp_path) == value
    assert (tmp_path / "pair.json").read_bytes() == before
    assert core.read(tmp_path / "startup-runtime.json")["runtime_code_hashes"] == gpu.code_hashes()


def test_second_startup_upgrade_preserves_receipt_and_rejects_later_edits(tmp_path, monkeypatch):
    value = {"schema": pair.SCHEMA, "code_hashes": bootstrap_predecessor(), "branch_manifests": {}}
    value["protocol_id"] = core.fingerprint(value)
    core.atomic_json(tmp_path / "pair.json", value)
    with monkeypatch.context() as patch:
        previous = defaults_predecessor()
        patch.setattr(gpu, "code_hashes", lambda: previous)
        gpu.manifest(tmp_path)
    before = (tmp_path / "startup-runtime.json").read_bytes()
    assert gpu.manifest(tmp_path) == value
    assert (tmp_path / "startup-runtime.json").read_bytes() == before
    assert core.read(tmp_path / "startup-defaults-runtime.json")["runtime_code_hashes"] == gpu.code_hashes()
    changed = {**gpu.code_hashes(), "src/selector_pair_gpu.py": "unreviewed-later-change"}
    monkeypatch.setattr(gpu, "code_hashes", lambda: changed)
    with pytest.raises(ValueError, match="frozen contract changed"):
        gpu.manifest(tmp_path)


def test_first_startup_upgrade_does_not_allow_later_entrypoint_edits(tmp_path, monkeypatch):
    value = {"schema": pair.SCHEMA, "code_hashes": bootstrap_predecessor(), "branch_manifests": {}}
    value["protocol_id"] = core.fingerprint(value)
    core.atomic_json(tmp_path / "pair.json", value)
    gpu.manifest(tmp_path)
    changed = {**gpu.code_hashes(), "src/selector_pair_gpu.py": "unreviewed-later-change"}
    monkeypatch.setattr(gpu, "code_hashes", lambda: changed)
    with pytest.raises(ValueError, match="frozen contract changed"):
        gpu.manifest(tmp_path)


@pytest.mark.parametrize("filename", ["src/selector_pair.py", gpu.TRAINER, "src/selection_switch_gpu.py"])
def test_startup_migration_does_not_allow_scientific_changes(monkeypatch, filename):
    previous, current = bootstrap_predecessor(), gpu.code_hashes()
    current[filename] = "changed"
    monkeypatch.setattr(gpu, "code_hashes", lambda: current)
    assert not gpu.compatible_code(previous)


def test_unrecognized_dummy_is_preserved_not_blessed(tmp_path):
    core.atomic_json(tmp_path / "pair.json", {})
    with pytest.raises(ValueError, match="unrecognized"):
        gpu.initialize(tmp_path)
    assert core.read(tmp_path / "pair.json") == {}
