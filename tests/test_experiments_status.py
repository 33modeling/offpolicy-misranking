import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import selection_gate as core

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


combined = load("experiments_status")
sys.path.insert(0, str(ROOT / "tests"))
from test_mopps_comparison_status import fixture as mopps_fixture  # noqa: E402
from test_selection_switch_status import four_nodes  # noqa: E402


def test_one_screen_shows_both_experiments_and_this_node_once(tmp_path):
    switch_root, mopps_root = tmp_path / "switch", tmp_path / "mopps"
    now = time.time()
    four_nodes(switch_root, now)
    mopps_fixture(mopps_root, switch_root, now)
    data = combined.snapshot(switch_root, mopps_root, now=now)
    # The one screen is exactly the two views, nothing recomputed differently.
    assert data["selection_switch"] == combined.switch_status.snapshot(switch_root, now=now)
    assert data["mopps_comparison"] == combined.mopps_status.snapshot(mopps_root, now=now)
    assert data["selection_switch"]["active_nodes"] >= 1 and data["mopps_comparison"]["prepared"] is True
    output = combined.render(data, width=120)
    first, second = output.splitlines()[:2]
    assert first.startswith("EXPERIMENTS  ") and second.startswith("NODES  ") and " live  |  " in second
    assert output.index("NODES (every host") < output.index("SELECTION SWITCH") < output.index("MOPPS COMPARISON")
    assert output.count("NODES (every host") == 1
    assert output.count("IDLE  ") == 1 and output.index("IDLE  ") < output.index("NODES (every host")
    assert "CONTINUATIONS" in output and "PARENT PREFIXES" in output
    assert output.count("THIS NODE GPUS") == 1
    assert output.index("THIS NODE GPUS") > output.index("MOPPS COMPARISON")
    assert all(len(line) <= 120 for line in output.splitlines())


def test_nodes_training_a_sibling_root_are_counted_and_labelled(tmp_path):
    """A node running long or difficulty is silent on its console for hours; the
    combined view must still show it as RUN from that root's own heartbeat."""
    from test_selection_switch_status import point, prepared, running
    switch_root, mopps_root = tmp_path / "selection-switch-v1", tmp_path / "mopps-comparison-v1"
    now = time.time()
    four_nodes(switch_root, now)
    mopps_fixture(mopps_root, switch_root, now)
    long_root = tmp_path / "selection-switch-long-v1"
    prepared(long_root)
    running(point(long_root) / "random_reduced", "node-long", now=now, phase="train")
    data = combined.snapshot(switch_root, mopps_root, now=now)
    hosts = {node["host"]: node for node in data["nodes"]}
    assert hosts["node-long"]["state"] == "RUN" and hosts["node-long"]["phase"] == "train"
    assert hosts["node-long"]["task"].endswith("long: random_reduced")
    assert "node-long" in combined.render(data, width=120)
    assert data["selection_switch"] == combined.switch_status.snapshot(switch_root, now=now)


def test_unprepared_mopps_root_does_not_hide_the_switch(tmp_path):
    switch_root = tmp_path / "switch"
    four_nodes(switch_root, time.time())
    data = combined.snapshot(switch_root, tmp_path / "absent")
    output = combined.render(data)
    assert "Dev 1/18" in output and "NOT PREPARED" in output and output.count("THIS NODE GPUS") == 1


def test_both_launchers_status_show_one_screen_and_stay_read_only(tmp_path):
    switch_root, mopps_root = tmp_path / "switch", tmp_path / "mopps"
    now = time.time()
    four_nodes(switch_root, now)
    mopps_fixture(mopps_root, switch_root, now)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    env = {**os.environ, "SWITCH_ROOT": str(switch_root), "MOPPS_ROOT": str(mopps_root),
           "SWITCH_PYTHON": sys.executable, "OM_WORK": str(tmp_path / "absent-work")}
    for launcher in ("run_selection_switch.sh", "run_mopps_comparison.sh", "run_experiments.sh"):
        result = subprocess.run(["bash", f"scripts/{launcher}", "status"], cwd=ROOT, env=env,
                                capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, (launcher, result.stderr)
        assert "SELECTION SWITCH" in result.stdout and "MOPPS COMPARISON" in result.stdout, launcher
        assert result.stdout.count("THIS NODE GPUS") == 1, launcher
    result = subprocess.run(["bash", "scripts/run_mopps_comparison.sh", "status", "--json"], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert set(payload) == {"updated", "nodes", "node_summary", "selection_switch", "mopps_comparison"}
    single = subprocess.run(["bash", "scripts/run_selection_switch.sh", "status"], cwd=ROOT,
                            env={**env, "EXPERIMENTS_COMBINED": "0"}, capture_output=True, text=True, timeout=30)
    assert single.returncode == 0 and "MOPPS COMPARISON" not in single.stdout
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before
    assert not list(tmp_path.rglob("*.lock")) and not list(tmp_path.rglob("*-runtime.json"))


def test_why_from_any_launcher_writes_one_report_for_both_experiments(tmp_path):
    # Real layout: both roots under WORK/runs, next to the node launcher's runs/experiments/logs.
    switch_root, mopps_root = tmp_path / "work/runs/selection-switch-v1", tmp_path / "work/runs/mopps-comparison-v1"
    now = time.time()
    four_nodes(switch_root, now)
    mopps_fixture(mopps_root, switch_root, now)
    (tmp_path / "work/runs/experiments/logs").mkdir(parents=True)
    (tmp_path / "work/runs/experiments/logs/console.node-9_.log").write_text(
        "[node-launcher-start] host=node-9\n[holding] node retained (switch rc=75 node busy: lock held or GPUs occupied | mopps rc=0 nothing left to claim); next pass in 60s\n")
    # A node that never held (and an empty keepalive log) must not mark the report incomplete.
    (tmp_path / "work/runs/experiments/logs/console.node-8_.log").write_text("[node-launcher-start] host=node-8\n[pass 1] selection switch\n")
    (tmp_path / "work/runs/experiments/logs/keepalive.node-8_.log").write_text("")
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    env = {**os.environ, "SWITCH_ROOT": str(switch_root), "MOPPS_ROOT": str(mopps_root),
           "SWITCH_PYTHON": sys.executable, "OM_WORK": str(tmp_path / "work")}
    for launcher in ("run_selection_switch.sh", "run_mopps_comparison.sh", "run_experiments.sh"):
        result = subprocess.run(["bash", f"scripts/{launcher}", "why"], cwd=ROOT, env=env,
                                capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, (launcher, result.stdout + result.stderr)
        path = Path(result.stdout.strip().splitlines()[-1].removeprefix("[saved] "))
        assert path.parent == tmp_path / "work/reports/experiments", launcher
        report = path.read_text()
        assert report.startswith("EXPERIMENTS WHY")
        assert "NODES  " in report and "HOLD" in report
        assert "######## run_selection_switch.sh why ########" in report
        assert "######## run_mopps_comparison.sh why ########" in report
        assert "SELECTION SWITCH EXPERIMENT" in report and "MOPPS COMPARISON" in report
        assert "===== states/s3-t50/random_online/failure.json =====" in report
        assert "NODE LAUNCHER LOG" in report and "lock held or GPUs occupied" in report
        assert "incomplete" not in report
        assert report.count("[pass 1] selection switch") >= 2
    assert {path: path.read_bytes() for path in before} == before
