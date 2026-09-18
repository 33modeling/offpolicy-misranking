"""Repeating the single queue command must leave healthy work untouched."""

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/run_experiments.sh"


@pytest.fixture
def controller(tmp_path):
    work = tmp_path / "work"
    logs = work / "runs/experiments/logs"
    logs.mkdir(parents=True)
    node = "safe-resume-fixture"
    env = {**os.environ, "OM_WORK": str(work), "EXPERIMENTS_NODE_ID": node,
           "CUDA_VISIBLE_DEVICES": "", "EXPERIMENTS_PULL": "0", "EXPERIMENTS_AUTO_PULL": "0",
           "EXPERIMENTS_KEEPALIVE": "0", "EXPERIMENTS_WATCHDOG": "0"}
    for name in ("EXPERIMENTS_DETACHED", "EXPERIMENTS_MBPP_SUITE", "MBPP_GUARD_PID", "OUT_ROOT"):
        env.pop(name, None)
    binaries = tmp_path / "bin"
    binaries.mkdir()
    gpu = binaries / "nvidia-smi"
    gpu.write_text("#!/usr/bin/env bash\nexit 0\n")
    gpu.chmod(0o755)
    env["PATH"] = str(binaries) + os.pathsep + env["PATH"]
    process = subprocess.Popen(
        ["bash", "-c", 'exec -a "bash scripts/run_experiments.sh run" sleep 300'],
        cwd=ROOT, env=env, start_new_session=True,
    )
    pid_file = logs / f"launcher.{node}_.pid"
    pid_file.write_text(str(process.pid))
    fault = work / f"runs/experiments/node-faults/{node}.json"
    fault.parent.mkdir(parents=True)
    fault.write_text('{"strikes":2,"phase":"train","time":1000}\n')
    policy = work / "runs/selection-switch-long-v1/states/s1-t50/points/view-50/random_reduced/policy"
    checkpoint = policy / "checkpoint-000150/adapter_model.safetensors"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"already saved checkpoint")
    ledger = policy.parent / "cost.jsonl"
    ledger.write_text('{"event_id":"saved","state":"started"}\n')
    try:
        yield process, env, pid_file, (fault, checkpoint, ledger)
    finally:
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=10)


@pytest.mark.parametrize("arguments", [[], ["run"]])
def test_repeating_queue_command_preserves_existing_controller_checkpoint_and_fault(controller, arguments):
    process, env, pid_file, saved = controller
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in (*saved, pid_file)}
    result = subprocess.run(["bash", str(LAUNCHER), *arguments], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[already running]" in result.stdout
    assert "[stop]" not in result.stdout and "[fault-reset]" not in result.stdout
    assert "[pass " not in result.stdout and "[clean]" not in result.stdout
    assert process.poll() is None, result.stdout
    assert before == {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in before}


@pytest.fixture
def mbpp_controller(controller):
    original, env, original_pid_file, saved = controller
    original.terminate()
    original.wait(timeout=5)
    original_pid_file.unlink()
    process = subprocess.Popen(
        ["bash", "-c", 'exec -a "python scripts/_mbpp_node_guard.py -- bash scripts/run_experiments.sh run" sleep 300'],
        cwd=ROOT, env={**env, "EXPERIMENTS_MBPP_SUITE": "all"}, start_new_session=True,
    )
    pid_file = original_pid_file.with_name(original_pid_file.name.replace("launcher.", "launcher.mbpp.", 1))
    pid_file.write_text(str(process.pid))
    try:
        yield process, env, pid_file, saved
    finally:
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=5)


@pytest.mark.parametrize("arguments,expected", [([], 0), (["run"], 0), (["restart"], 75), (["stop"], 75)])
def test_generic_commands_never_interrupt_verified_mbpp_owner(mbpp_controller, arguments, expected):
    process, env, pid_file, saved = mbpp_controller
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in (*saved, pid_file)}
    result = subprocess.run(["bash", str(LAUNCHER), *arguments], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == expected, result.stdout + result.stderr
    assert "MBPP" in result.stdout + result.stderr
    if expected == 0:
        assert "[already running]" in result.stdout
    assert "[stop]" not in result.stdout and "[clean]" not in result.stdout
    assert "[pass " not in result.stdout and "[fault-reset]" not in result.stdout
    assert process.poll() is None, result.stdout
    assert before == {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in before}


@pytest.mark.parametrize("mismatch", ["work", "node", "pid", "command"])
def test_read_only_owner_probe_rejects_unrelated_or_stale_pid(controller, mismatch):
    process, env, pid_file, _ = controller
    # Exercise the actual identity helper alone; a rejected owner must not start
    # any queue, admission probe or cleanup in this test.
    text = LAUNCHER.read_text()
    helper = text.split("launcher_pid_alive() {", 1)[1].split('\nif [ "$MODE" = progress ]', 1)[0]
    script = "launcher_pid_alive() {" + helper + "\nlauncher_pid_alive\n"
    check_env = {**env, "WORK": env["OM_WORK"], "PID_FILE": str(pid_file), "LOG_DIR": str(pid_file.parent)}
    bystander = None
    if mismatch == "work":
        check_env["WORK"] += "-different"
    elif mismatch == "node":
        check_env["EXPERIMENTS_NODE_ID"] += "-different"
    elif mismatch == "pid":
        pid_file.write_text("999999999")
    else:
        bystander = subprocess.Popen(["sleep", "300"], env=env, start_new_session=True)
        pid_file.write_text(str(bystander.pid))
    try:
        result = subprocess.run(["bash"], input=script, cwd=ROOT, env=check_env,
                                capture_output=True, text=True, timeout=5, check=False)
        assert result.returncode != 0, "unrelated PID was accepted as the queue controller"
        assert process.poll() is None
        if bystander is not None:
            assert bystander.poll() is None
    finally:
        if bystander is not None:
            bystander.terminate()
            bystander.wait(timeout=5)
