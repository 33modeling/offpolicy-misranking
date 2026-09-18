"""CPU regressions for node admission and cleanup; never signal GPU jobs."""

import os
from pathlib import Path
import re
import subprocess
import time

import pytest

from test_selection_nccl_preflight import (
    E802, admission, check, clear_fabric_env, fake_attempt,
)


ROOT = Path(__file__).resolve().parents[1]


def shell_function(name, script="run_experiments.sh"):
    source = (ROOT / "scripts" / script).read_text()
    return re.search(rf"^{name}\(\) \{{\n.*?^\}}", source, re.M | re.S)[0]


def wait_for_exec(process, command):
    # Popen may return before bash has exec'ed the process being inspected.
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        assert process.poll() is None, "process fixture exited before inspection"
        if Path(f"/proc/{process.pid}/cmdline").read_bytes().startswith(command.encode() + b"\0"):
            return
        time.sleep(.01)
    raise AssertionError("process fixture did not exec its expected command")


@pytest.mark.parametrize("failures", [
    [E802, "ncclUnhandledCudaError", None],
    ["ncclUnhandledCudaError", E802, None],
])
def test_fabric_and_legacy_host_workarounds_compose_without_losing_overrides(
    tmp_path, monkeypatch, failures,
):
    clear_fabric_env(monkeypatch)
    calls = fake_attempt(monkeypatch, failures)
    expected = {"NCCL_NVLS_ENABLE": "0", "NCCL_CUMEM_HOST_ENABLE": "0"}
    assert check.preflight(tmp_path) == expected
    assert len(calls) == 3
    assert all(calls[-1][key] == value for key, value in expected.items())
    value = admission(tmp_path)
    assert value["state"] == "passed"
    assert len({row["directory"] for row in value["attempts"]}) == 3


def test_mixed_nccl_failures_still_stop_after_the_bounded_probe_ladder(tmp_path, monkeypatch):
    clear_fabric_env(monkeypatch)
    calls = fake_attempt(monkeypatch, [E802, "ncclUnhandledCudaError", E802, E802, E802])
    with pytest.raises(RuntimeError, match="CUDA 802"):
        check.preflight(tmp_path)
    assert len(calls) == 5
    value = admission(tmp_path)
    assert value["state"] == "failed"
    assert len({row["directory"] for row in value["attempts"]}) == 5
    assert calls[-1]["NCCL_CUMEM_HOST_ENABLE"] == "0"


def test_gpu_cleanup_does_not_target_unrelated_or_other_allocation_processes(tmp_path):
    source = (ROOT / "scripts/run_experiments.sh").read_text()
    functions = source[source.index("ROOT_PROCESS_PATTERN="):source.index("gpu_memory_line()")]
    work = str(tmp_path / "work")
    env = {**os.environ, "OM_WORK": work, "EXPERIMENTS_NODE_ID": "audit-node"}
    processes = []
    try:
        for node in (None, "other-allocation", "audit-node"):
            child_env = {**env, "OUT_ROOT": work + "/runs/selection-switch-v1"}
            if node is None:
                for key in ("OM_WORK", "OUT_ROOT", "EXPERIMENTS_NODE_ID"):
                    child_env.pop(key, None)
            else:
                child_env["EXPERIMENTS_NODE_ID"] = node
            processes.append(subprocess.Popen(
                ["bash", "-c", 'exec -a "python src/train_policy_grpo.py" sleep 30'],
                env=child_env, start_new_session=True,
            ))
        for process in processes:
            wait_for_exec(process, "python src/train_policy_grpo.py")
        rows = "\n".join(f"{process.pid}, 1000" for process in processes)
        harness = functions + '''
nvidia-smi() { printf '%s\n' "$AUDIT_GPU_ROWS"; }
timeout() { shift 3; "$@"; }
gpu_holder_groups
'''
        result = subprocess.run(["bash", "-c", harness], env={
            **env, "WORK": work, "AUDIT_GPU_ROWS": rows,
        }, capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, result.stderr
        assert result.stdout.split() == [str(processes[-1].pid)], result.stderr
    finally:
        for process in processes:
            process.terminate()
            process.wait(timeout=5)


@pytest.mark.parametrize("output,code", [("", 0), ("", 124), ("0", 1), ("N/A", 0)])
def test_gpu_memory_query_fails_closed(output, code):
    harness = shell_function("gpus_free") + '''
nvidia-smi() { :; }
timeout() { printf '%s' "$AUDIT_OUTPUT"; return "$AUDIT_EXIT"; }
gpus_free
'''
    result = subprocess.run(["bash", "-c", harness], env={
        **os.environ, "AUDIT_OUTPUT": output, "AUDIT_EXIT": str(code),
    }, capture_output=True, text=True, timeout=5)
    assert result.returncode != 0


def test_node_identity_bounds_gpu_driver_probe(tmp_path):
    harness = '''
unset EXPERIMENTS_NODE_ID
nvidia-smi() { echo '[unbounded GPU query]' >> "$AUDIT_PROBE_LOG"; return 1; }
timeout() { printf '[bounded] %s\n' "$*" >> "$AUDIT_PROBE_LOG"; return 124; }
source scripts/_node_id.sh
printf '%s\n' "$EXPERIMENTS_NODE_ID"
'''
    log = tmp_path / "probe.log"
    result = subprocess.run(["bash", "-c", harness], cwd=ROOT,
                            env={**os.environ, "AUDIT_PROBE_LOG": str(log)},
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 0
    assert "[bounded] -k 2 10 nvidia-smi" in log.read_text()
    assert "[unbounded GPU query]" not in log.read_text()
    assert result.stdout.strip()


def test_missing_gpu_driver_command_is_not_a_free_gpu():
    result = subprocess.run(["bash", "-c", shell_function("gpus_free") + '''
command() { return 1; }
gpus_free
'''], capture_output=True, text=True, timeout=5)
    assert result.returncode != 0


@pytest.mark.parametrize("rc", [75, 78, 79])
def test_node_rejection_prevents_starting_the_second_queue(tmp_path, rc):
    source = (ROOT / "scripts/run_experiments.sh").read_text()
    block = source[source.index("  rc_mopps=0 why_mopps=skipped"):source.index('  reason="switch rc=')]
    (tmp_path / "mopps.json").write_text("{}")
    result = subprocess.run(["bash", "-c", '''
mopps_complete() { return 1; }
inner() { echo '[should-not-run]'; }
rc_reason() { :; }
pass=1
''' + block], env={**os.environ, "rc_switch": str(rc), "MOPPS_ROOT": str(tmp_path),
                  "EXPERIMENTS_SKIP_MOPPS": "0"}, capture_output=True, text=True, timeout=5)
    assert result.returncode == 0
    assert "[should-not-run]" not in result.stdout


@pytest.mark.parametrize("rc", [78, 79])
def test_mopps_fault_does_not_short_circuit_hold_for_claimable_work(rc):
    source = (ROOT / "scripts/run_experiments.sh").read_text()
    condition = next(line for line in source.splitlines() if '&& found=$(claimable_work); then' in line)
    harness = 'claimable_work() { echo available; }\n' + condition + "\necho bypassed\nfi"
    result = subprocess.run(["bash", "-c", harness],
                            env={**os.environ, "rc_switch": "0", "rc_mopps": str(rc)},
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 0
    assert "bypassed" not in result.stdout


@pytest.mark.parametrize("script", ["run_selection_switch.sh", "run_mopps_comparison.sh"])
@pytest.mark.parametrize("node,root,expected", [
    ("audit-node", "root.v1", 0),
    ("other-node", "root.v1", 1),
    ("audit-node", "root-v1", 1),
])
def test_root_worker_requires_exact_root_and_allocation(script, node, root, expected, tmp_path):
    env = {**os.environ, "EXPERIMENTS_NODE_ID": node, "OUT_ROOT": str(tmp_path / root)}
    worker = subprocess.Popen(
        ["bash", "-c", 'exec -a "python src/train_policy_grpo.py" sleep 30'],
        env=env, start_new_session=True,
    )
    try:
        wait_for_exec(worker, "python src/train_policy_grpo.py")
        result = subprocess.run(
            ["bash", "-c", shell_function("root_worker_cmdline", script) + f"\nroot_worker_cmdline {worker.pid}"],
            env={**env, "EXPERIMENTS_NODE_ID": "audit-node", "OUT_ROOT": str(tmp_path / "root.v1")},
            capture_output=True, text=True, timeout=5,
        )
        assert result.returncode == expected, result.stderr
    finally:
        worker.terminate()
        worker.wait(timeout=5)


@pytest.mark.parametrize("script", ["run_experiments.sh", "run_selection_switch.sh", "run_mopps_comparison.sh"])
@pytest.mark.parametrize("matching_command", [True, False])
def test_stale_pid_file_is_not_authority_to_signal(script, matching_command, tmp_path):
    env = {**os.environ, "EXPERIMENTS_NODE_ID": "audit-node", "OM_WORK": str(tmp_path),
           "WORK": str(tmp_path), "OUT_ROOT": str(tmp_path / "root"),
           "LAUNCHER_SELF": str(ROOT / "scripts" / script), "EXPERIMENTS_MBPP_SUITE": ""}
    command = f"bash scripts/{script} run" if matching_command else "unrelated-job"
    worker = subprocess.Popen(["bash", "-c", f'exec -a "{command}" sleep 30'],
                              env=env, start_new_session=True)
    pid_file = tmp_path / "launcher.pid"
    pid_file.write_text(str(worker.pid))
    try:
        wait_for_exec(worker, command)
        harness = shell_function("launcher_pid_alive", script)
        if script != "run_experiments.sh":
            harness += "\n" + shell_function("root_worker_cmdline", script)
        result = subprocess.run(["bash", "-c", harness + "\nlauncher_pid_alive"],
                                env={**env, "PID_FILE": str(pid_file)},
                                capture_output=True, text=True, timeout=5)
        assert result.returncode == (0 if matching_command else 1), result.stderr
        assert worker.poll() is None
    finally:
        worker.terminate()
        worker.wait(timeout=5)
