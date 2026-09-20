"""Repair guards run on CPUs and never permit work on the 37 saved branches."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("repair_runtime_test", ROOT / "scripts/mbpp_repair_runtime.py")
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)
switch = adapter.switch


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    root, source = tmp_path / "repair", tmp_path / "original"
    for directory in (root, source):
        directory.mkdir()
        switch.core.atomic_json(directory / "switch.json", {
            "schema": switch.rule.SCHEMA, "dataset": "mbpp", "code_hashes": {"scientific": "frozen"}})
    rerun = sorted(adapter.branch_name(*item) for item in adapter.RERUN)
    dependent = sorted(adapter.branch_name(seed, step, "gated")
                       for seed in switch.rule.TEST_SEEDS for step in switch.rule.STEPS)
    all_branches = {adapter.branch_name(seed, step, arm)
                    for seeds, arms in ((switch.rule.DEV_SEEDS, switch.rule.DEV_ARMS),
                                        (switch.rule.TEST_SEEDS, switch.rule.TEST_ARMS))
                    for seed in seeds for step in switch.rule.STEPS for arm in arms}
    for seed in (*switch.rule.DEV_SEEDS, *switch.rule.TEST_SEEDS):
        for step in switch.rule.STEPS:
            switch.core.atomic_json(switch.prefix_dir(root, seed) / f"prefix-{step}.json", {})
            switch.core.atomic_json(switch.child_root(root, seed, step) / "net_protocol.json", {})
    for name in all_branches:
        (root / name).mkdir(parents=True)
    payload = {
        "schema": "mbpp-repair/v1", "source_root": str(source),
        "source_switch_sha256": switch.base.digest(source / "switch.json"),
        "rerun_branches": rerun, "dependent_branches": dependent,
        "reused_branches": sorted(all_branches - set(rerun) - set(dependent)),
        "snapshot_files": {"switch.json": switch.base.digest(root / "switch.json")}}
    switch.core.atomic_json(root / "repair.json", payload)
    monkeypatch.setattr(switch, "validate_code_hashes", lambda hashes: hashes)
    return root, source, payload


def test_source_manifest_is_strict_and_never_opens_a_write_lease(prepared, monkeypatch):
    root, source, _ = prepared
    original_bytes = {str(p): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    calls = []
    monkeypatch.setattr(switch, "manifest", lambda path: calls.append(path) or {"target": True})
    with adapter.activated(root):
        assert switch.manifest(source)["dataset"] == "mbpp"
        assert switch.manifest(root) == {"target": True}
        assert calls == [root]
        monkeypatch.setattr(switch, "validate_code_hashes", lambda _: (_ for _ in ()).throw(ValueError("code changed")))
        with pytest.raises(ValueError, match="code changed"):
            switch.manifest(source)
    assert original_bytes == {str(p): p.read_bytes() for p in source.rglob("*") if p.is_file()}


def test_all_37_reused_branches_refuse_training_and_11_allow(prepared, monkeypatch):
    root, _, p = prepared
    calls = []
    monkeypatch.setattr(switch.runtime, "run_arm", lambda *args: calls.append(args))
    with adapter.activated(root):
        for name in p["reused_branches"]:
            directory = root / name
            with pytest.raises(ValueError, match="reused"):
                switch.runtime.run_arm(directory.parent, {}, {}, directory.name, [], {})
        for name in p["rerun_branches"] + p["dependent_branches"]:
            directory = root / name
            switch.runtime.run_arm(directory.parent, {}, {}, directory.name, [], {})
    assert len(calls) == 11


def test_install_runtime_propagates_adapter_to_workers_and_restores(prepared):
    root, _, p = prepared
    previous = switch.HERE, switch.runtime.HERE, switch.runtime.measurement_worker
    callbacks = switch.base.verify, switch.base.train_command, switch.runtime.protocol
    point = (root / p["rerun_branches"][0]).parent
    with adapter.activated(point):
        switch.install_runtime()
        assert switch.HERE == adapter.HERE == switch.runtime.HERE
        for method in (switch.prepare, switch.publish_state, switch.build_prefix, switch.runtime.measurement_worker):
            with pytest.raises(ValueError, match="repair cannot"):
                method(root)
    assert (switch.HERE, switch.runtime.HERE, switch.runtime.measurement_worker) == previous
    assert (switch.base.verify, switch.base.train_command, switch.runtime.protocol) == callbacks


@pytest.mark.parametrize("kind", ["source", "snapshot", "allowlist", "prefix", "state"])
def test_changed_or_incomplete_repair_is_rejected(prepared, kind):
    root, source, p = prepared
    if kind == "source":
        (source / "switch.json").write_text("{}")
    elif kind == "snapshot":
        p["snapshot_files"]["switch.json"] = "bad"
    elif kind == "allowlist":
        p["rerun_branches"][0] = p["reused_branches"][0]
    elif kind == "prefix":
        (root / "prefixes/seed-0/prefix-25.json").unlink()
    else:
        (switch.child_root(root, 0, 25) / "net_protocol.json").unlink()
    switch.core.atomic_json(root / "repair.json", p)
    with pytest.raises(ValueError):
        adapter.install(root)


def test_worker_target_cannot_escape_via_symlink(prepared, monkeypatch):
    root, source, p = prepared
    directory = root / p["rerun_branches"][0]
    directory.rmdir()
    directory.symlink_to(source, target_is_directory=True)
    with adapter.activated(root):
        with pytest.raises(ValueError, match="outside"):
            switch.base.evaluate(directory.parent, directory.name, 0)


def test_curve_worker_reused_branch_is_rejected(prepared):
    root, _, p = prepared
    directory = root / p["reused_branches"][0]
    with adapter.activated(root):
        with pytest.raises(ValueError, match="reused"):
            switch.curve_evaluate(directory.parent, directory.name, 0, 0)


def test_queue_installs_repair_before_entering_switch_main(prepared, monkeypatch):
    from test_mbpp_node_queue import queue_worker
    root, source, _ = prepared
    monkeypatch.setattr(sys, "argv", ["queue_selection_switch_gpu.py", "run", "--root", str(root)])
    def main():
        assert switch.HERE.name == "mbpp_repair_runtime.py"
        assert switch.manifest(source)["dataset"] == "mbpp"
        return 17
    monkeypatch.setattr(switch, "main", main)
    assert queue_worker.run() == 17


def test_ordinary_root_does_not_change_runtime(tmp_path):
    original = switch.manifest, switch.runtime.run_arm, switch.HERE
    with adapter.activated(tmp_path):
        assert (switch.manifest, switch.runtime.run_arm, switch.HERE) == original


def test_launcher_routes_every_direct_switch_command_through_adapter():
    script = (ROOT / "scripts/run_selection_switch.sh").read_text()
    assert '[ ! -f "$OUT_ROOT/repair.json" ] || SWITCH_DRIVER=scripts/mbpp_repair_runtime.py' in script
    assert '"$PY" src/selection_switch_gpu.py' not in script
    assert 'WORKER=$SWITCH_DRIVER' in script


def test_real_check_code_process_preserves_original_bytes(prepared):
    root, source, p = prepared
    protocol = switch.core.read(root / "switch.json")
    protocol["code_hashes"] = switch.code_hashes()
    for directory in (root, source):
        switch.core.atomic_json(directory / "switch.json", protocol)
    p["source_switch_sha256"] = switch.base.digest(source / "switch.json")
    p["snapshot_files"]["switch.json"] = switch.base.digest(root / "switch.json")
    switch.core.atomic_json(root / "repair.json", p)
    before = {str(path): path.read_bytes() for path in source.rglob("*") if path.is_file()}
    result = subprocess.run([sys.executable, str(adapter.HERE), "check-code", "--root", str(root)],
                            cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": "",
                                           "PYTHONDONTWRITEBYTECODE": "1"},
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "exact"
    assert before == {str(path): path.read_bytes() for path in source.rglob("*") if path.is_file()}
