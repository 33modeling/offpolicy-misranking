"""Dispatch observability cannot mutate saved state or stall/change GPU work."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import selection_switch as rule

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/queue_dispatch_evidence.py"
SPEC = importlib.util.spec_from_file_location("queue_dispatch_evidence_test", SCRIPT)
evidence = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evidence)


def manifest(root, kind="switch"):
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{kind}.json").write_text(json.dumps({"dataset": "mbpp", "selector": "fresh_r",
                                                "accounting": "budget", "gate": "final"}))


@pytest.mark.parametrize("kind", ["switch", "mopps"])
def test_single_root_protocol_counts_saved_steps_and_blocked_reason(tmp_path, kind):
    manifest(tmp_path, kind)
    calls = []

    def snapshot(selected_kind, root):
        calls.append((selected_kind, root))
        return {"tasks": [{"kind": "branch", "status": name, "directory": "states/s0-t25/" + name,
                           "training_step": 125, "phase": "train", "reason": "checkpoint needs validation"}
                          for name in evidence.STATUSES], "training_published": 3, "gate_ready": True}

    output = "\n".join(evidence.describe(tmp_path, "abc123", snapshot_loader=snapshot))
    assert calls == [(kind, tmp_path.resolve())]
    assert f"root={tmp_path}" in output and "checkout=abc123" in output
    assert "frozen dataset=mbpp selector=fresh_r accounting=budget gate=final" in output
    for label in ("DONE", "EVAL", "RESUME", "REVIEW", "READY", "RUN", "FAIL"):
        assert f"{label}=1" in output
    assert "logged_step=125" in output and "reason=checkpoint needs validation" in output
    assert "checkpoint_step=?" in output  # Logged updates are not durable checkpoint proof.
    assert output.count("[dispatch-task]") == 3


def test_absent_or_invalid_manifest_does_not_start_snapshot_or_write(tmp_path):
    def forbidden(*args):
        raise AssertionError("status must not inspect an unprepared root")

    missing = tmp_path / "absent"
    output = "\n".join(evidence.describe(missing, "rev", snapshot_loader=forbidden))
    assert "manifest=absent" in output and not missing.exists()
    manifest(tmp_path)
    (tmp_path / "switch.json").write_text("not JSON")
    before = (tmp_path / "switch.json").read_bytes()
    output = "\n".join(evidence.describe(tmp_path, "rev", snapshot_loader=forbidden))
    assert "snapshot unavailable" in output
    assert (tmp_path / "switch.json").read_bytes() == before


def test_metadata_helper_does_not_read_model_or_optimizer_payloads(tmp_path, monkeypatch):
    manifest(tmp_path)
    for name in ("adapter_model.safetensors", "optimizer.pt"):
        (tmp_path / name).write_bytes(b"must not be read or changed")
    before = {path: path.read_bytes() for path in tmp_path.iterdir()}
    original = Path.open

    def guarded(path, *args, **kwargs):
        assert path.suffix not in {".safetensors", ".pt"}
        return original(path, *args, **kwargs)

    with monkeypatch.context() as guard:
        guard.setattr(Path, "open", guarded)
        list(evidence.describe(tmp_path, "rev", snapshot_loader=lambda *args: {"tasks": []}))
    assert before == {path: path.read_bytes() for path in tmp_path.iterdir()}


def test_snapshot_disables_node_and_gpu_inventory_and_restores_helpers(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("must not scan other roots or query GPUs")

    node_view = SimpleNamespace(launcher_nodes=forbidden, local_gpus=forbidden)

    def snapshot(root):
        assert node_view.launcher_nodes(root, [], now=1) == []
        assert node_view.local_gpus()["available"] is False
        return {"tasks": []}

    fake = SimpleNamespace(node_view=node_view, snapshot=snapshot)
    monkeypatch.setattr(evidence.importlib, "import_module", lambda name: fake)
    assert evidence.load_snapshot("switch", tmp_path) == {"tasks": []}
    assert node_view.launcher_nodes is forbidden and node_view.local_gpus is forbidden


def test_reason_redaction_and_single_line_bounds():
    value = evidence.clean("token=private password=hunter2 api_key=abc\nhttps://name:pass@host/path " + "x" * 300)
    assert "private" not in value and "hunter2" not in value and "name:pass" not in value
    assert "\n" not in value and len(value) <= 200


@pytest.mark.parametrize("helper", ["raise SystemExit(7)", "import time; time.sleep(30)"])
def test_controller_logging_is_bounded_and_preserves_actual_worker_exit(tmp_path, helper):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "queue_dispatch_evidence.py").write_text(helper + "\n")
    source = (ROOT / "scripts/run_experiments.sh").read_text()
    function = "dispatch_evidence() {" + source.split("dispatch_evidence() {", 1)[1].split("\n}", 1)[0] + "\n}\n"
    command = function + 'dispatch_evidence "$TEST_ROOT"\necho worker-started\nexit 42\n'
    result = subprocess.run(["bash", "-c", command], cwd=tmp_path,
                            env={**os.environ, "PY": sys.executable, "TEST_ROOT": str(tmp_path / "root")},
                            text=True, capture_output=True, timeout=7, check=False)
    assert result.returncode == 42
    assert "worker-started" in result.stdout
    assert "metadata logging rc=" in result.stdout


def test_helper_is_called_only_for_dispatched_root_and_mopps():
    source = (ROOT / "scripts/run_experiments.sh").read_text()
    assert 'run_switch_root() {\n  dispatch_evidence "$1"' in source
    assert 'dispatch_evidence "$MOPPS_ROOT"' in source
    assert "timeout -k 1 3 env CUDA_VISIBLE_DEVICES=''" in source


def test_real_status_metadata_snapshot_cli_is_read_only(tmp_path):
    manifest(tmp_path)
    path = tmp_path / "switch.json"
    record = json.loads(path.read_text())
    record["schema"] = rule.SCHEMA
    path.write_text(json.dumps(record))
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    result = subprocess.run([sys.executable, str(SCRIPT), "--root", str(tmp_path), "--checkout", "tested123"],
                            cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"},
                            capture_output=True, text=True, timeout=5, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "checkout=tested123" in result.stdout
    assert "branches=48" in result.stdout
    assert "snapshot unavailable" not in result.stdout
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


def test_checkpoint_step_is_separate_from_newer_uncheckpointed_log_step(tmp_path):
    manifest(tmp_path)
    data = {"tasks": [{"status": "RESUME", "training_step": 129,
                       "reason": "checkpoint step 125 present; trainer must validate hashes and contract before resume"}]}
    output = "\n".join(evidence.describe(tmp_path, "rev", snapshot_loader=lambda *args: data))
    assert "checkpoint_step=125 logged_step=129" in output
