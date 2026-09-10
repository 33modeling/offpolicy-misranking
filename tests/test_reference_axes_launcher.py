"""CPU-only checks using the existing reliability launcher's fake backend."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest
from test_reliability_budget import (
    launch_env as launch_env,  # noqa: PLC0414 - pytest fixture export
)

CODE = Path(__file__).resolve().parents[1]


@pytest.fixture
def axes(launch_env):
    env, _, _, _ = launch_env
    repo = Path(env["OM_REPO"])
    shutil.copy2(CODE / "scripts/run_reference_axes.sh", repo / "scripts/run_reference_axes.sh")
    env.update(REFERENCE_WORK=env["OM_WORK"], REFERENCE_VENV=env["VENV_DIR"],
               REFERENCE_MODEL=env["OM_OLMO3_MODEL_PATH"], REFERENCE_SOURCE_ROOT=env["OM_OLMO3_ROOT"])
    (Path(env["REFERENCE_MODEL"]) / "config.json").write_text("{}")
    return env, ["bash", str(repo / "scripts/run_reference_axes.sh"), "mbpp", "0"]


def test_existing_environment_and_isolated_outputs_despite_inherited_v2(axes):
    env, command = axes
    source = Path(env["REFERENCE_SOURCE_ROOT"])
    before = {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    for key in ("OM_REPO", "OM_WORK", "VENV_DIR", "OM_OLMO3_ROOT", "OM_OLMO3_MODEL_PATH",
                "OM_RLZERO_CONFIG", "PYTHONPATH", "PYTHONHOME", "HF_HOME", "TMPDIR", "RB_RUNS_ROOT"):
        env[key] = "/nonexistent/offpolicy-misranking-v2"
    env.update(OM_GEN_BATCH="999", GRADIENT_MICRO_BATCH="999", OM_THINKING="on")
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    root = Path(env["REFERENCE_WORK"]) / "runs/reference-axes"
    configs = [json.loads(p.read_text()) for p in root.glob("*/run_config.json")]
    assert {(c["fresh_k"], c["val_k"], c["seed"]) for c in configs} == {
        (32, 8, 101288), (64, 8, 102568), (128, 8, 105128),
        (32, 16, 101296), (32, 32, 101312)}
    assert len(list(root.glob("*/RB_DONE"))) == 5
    assert "gen_batch=16 grad_micro_batch=1" in result.stdout
    consoles = list((Path(env["REFERENCE_WORK"]) / "console-logs").glob("reference-*.log"))
    assert len(consoles) == 1
    console = consoles[0].read_text()
    assert console.count("[condition]") == 5
    assert "[reference-begin]" in console and "state=COMPLETE rc=0" in console
    state = next((Path(env["REFERENCE_WORK"]) / "reference-workers").glob("*.state"))
    assert state.read_text().startswith("COMPLETE\t")
    assert not (root.parent / "reliability-budget-v1").exists()
    assert before == {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    again = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60, check=False)
    assert again.returncode == 0, again.stdout + again.stderr
    assert again.stdout.count("effective contract matches") == 5


def test_plan_needs_no_python_or_gpu(axes):
    env, command = axes
    env["REFERENCE_VENV"] = "/nonexistent/python"
    result = subprocess.run(command + ["--plan"], env=env, capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("[condition]") == 5
    assert not (Path(env["REFERENCE_WORK"]) / "runs/reference-axes").exists()


def test_preflight_failure_is_saved_without_starting_conditions(axes):
    env, command = axes
    env["REFERENCE_VENV"] = "/nonexistent/python"
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == 1
    work = Path(env["REFERENCE_WORK"])
    console = next((work / "console-logs").glob("reference-*.log")).read_text()
    assert "[abort] existing Python environment is missing" in console
    assert "state=FAILED rc=1" in console
    assert "[condition]" not in console
    assert next((work / "reference-workers").glob("*.state")).read_text().startswith("FAILED\t")


def test_check_runs_no_gpu_stages(axes):
    env, command = axes
    result = subprocess.run(command + ["--check"], env=env, capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("nothing started") == 5
    assert not (Path(env["REFERENCE_WORK"]) / "runs/reference-axes").exists()


def test_failed_condition_does_not_block_remaining_conditions(axes):
    env, command = axes
    backend = Path(env["OM_REPO"]) / "scripts/run_reliability_budget.sh"
    backend.write_text('#!/bin/bash\necho "MOCK condition $2 $3"\n[ "$2" != 64 ]\n')
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == 1
    assert result.stdout.count("MOCK condition") == 5
    assert "continuing with remaining conditions" in result.stdout
    state = next((Path(env["REFERENCE_WORK"]) / "reference-workers").glob("*.state"))
    assert state.read_text().startswith("FAILED\t")


def test_output_override_cannot_write_primary_source(launch_env):
    env, command, _, _ = launch_env
    env["RB_RUNS_ROOT"] = env["OM_OLMO3_ROOT"]
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode != 0
    assert "cannot overwrite primary inputs" in result.stdout


def test_sigterm_stops_active_condition_and_no_next_condition_starts(axes):
    env, command = axes
    env["FAKE_BLOCK_CHILDREN"] = "1"
    root = Path(env["REFERENCE_WORK"]) / "runs/reference-axes"
    process = subprocess.Popen(command, env=env, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        deadline = time.monotonic() + 30
        records = []
        while time.monotonic() < deadline:
            records = list(root.glob("*/child-*.json"))
            if len(records) == 2:
                break
            time.sleep(0.05)
        assert len(records) == 2
        pids = [json.loads(p.read_text())["pid"] for p in records]
        process.terminate()
        assert process.wait(timeout=15) == 143
        state = next((Path(env["REFERENCE_WORK"]) / "reference-workers").glob("*.state"))
        assert state.read_text().startswith("INTERRUPTED\t")
        assert len(list(root.iterdir())) == 1
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            alive = []
            for pid in pids:
                try:
                    if Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z":
                        alive.append(pid)
                except FileNotFoundError:
                    pass
            if not alive:
                break
            time.sleep(0.05)
        assert not alive
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)
