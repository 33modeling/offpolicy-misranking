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
    assert output.index("SELECTION SWITCH") < output.index("MOPPS COMPARISON")
    assert "CONTINUATIONS" in output and "PARENT PREFIXES" in output
    assert output.count("THIS NODE GPUS") == 1
    assert output.index("THIS NODE GPUS") > output.index("MOPPS COMPARISON")
    assert all(len(line) <= 120 for line in output.splitlines())


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
    assert set(payload) == {"updated", "selection_switch", "mopps_comparison"}
    single = subprocess.run(["bash", "scripts/run_selection_switch.sh", "status"], cwd=ROOT,
                            env={**env, "EXPERIMENTS_COMBINED": "0"}, capture_output=True, text=True, timeout=30)
    assert single.returncode == 0 and "MOPPS COMPARISON" not in single.stdout
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before
    assert not list(tmp_path.rglob("*.lock")) and not list(tmp_path.rglob("*-runtime.json"))
