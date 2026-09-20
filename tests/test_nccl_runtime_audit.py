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


def test_gpu_cleanup_no_longer_sweeps_processes_by_pattern_or_gpu_holder(tmp_path):
    """6551160 removed the node-wide process/GPU sweep: a new controller never
    signals processes because their command, root or node ID matches, and never
    reads GPU holders in order to kill them. Cleanup is lease-checked cost
    recovery only, in both the MBPP and the shared launcher modes."""
    source = (ROOT / "scripts/run_experiments.sh").read_text()
    for marker in ("ROOT_PROCESS_PATTERN", "gpu_holder_groups", "leftover_groups", "query-compute-apps"):
        assert marker not in source, marker
    harness = shell_function("full_clean") + '''
HOST=audit-node
close_dead_events() { echo "[audit] close_dead_events mbpp=${EXPERIMENTS_MBPP_SUITE:-}"; }
kill() { echo "[audit] kill $*" >&2; return 1; }
pkill() { echo "[audit] pkill $*" >&2; return 1; }
nvidia-smi() { echo "[audit] nvidia-smi $*" >&2; return 1; }
full_clean
EXPERIMENTS_MBPP_SUITE=quality full_clean
'''
    result = subprocess.run(["bash", "-c", harness], env={**os.environ, "OM_WORK": str(tmp_path)},
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stderr == "", result.stderr
    assert result.stdout.count("[audit] close_dead_events mbpp=\n") == 1
    assert result.stdout.count("[audit] close_dead_events mbpp=quality\n") == 1
    assert result.stdout.count("no node-wide process/GPU sweep") == 2


def gpu_memory_admission_block():
    """The inner launcher's fail-closed GPU memory admission (run_selection_switch.sh):
    a failed, empty, malformed or incomplete query blocks admission; only four
    numeric rows at or under 4000 MiB admit. 6551160 removed the node launcher's
    separate gpus_free probe, so this block is the remaining guard."""
    source = (ROOT / "scripts/run_selection_switch.sh").read_text()
    start = source.index('if ! MEMORY=$(timeout -k 5 20 nvidia-smi --query-gpu=memory.used')
    end = source.index('# The allocation is reclaimed when its GPUs sit idle', start)
    return source[start:end]


def run_gpu_memory_admission(output, code, *, missing_driver=False):
    stubs = (
        "CUDA_VISIBLE_DEVICES=0,1,2,3\n"
        "GPU_ADMISSION_RC=78\n"
        "nvidia-smi() { printf '%s' \"$AUDIT_OUTPUT\"; return \"$AUDIT_EXIT\"; }\n"
        "timeout() { shift 3; \"$@\"; }\n"
    )
    if missing_driver:
        stubs += 'nvidia-smi() { echo "bash: nvidia-smi: command not found" >&2; return 127; }\n'
    harness = stubs + gpu_memory_admission_block() + '\necho admitted\n'
    return subprocess.run(["bash", "-c", harness], env={
        **os.environ, "AUDIT_OUTPUT": output, "AUDIT_EXIT": str(code),
    }, capture_output=True, text=True, timeout=5)


@pytest.mark.parametrize("output,code", [
    ("", 0), ("", 124), ("0", 1), ("N/A\n0\n0\n0", 0), ("0\n0\n0", 0), ("0\n0\n0\n0\n0", 0),
])
def test_gpu_memory_query_fails_closed(output, code):
    result = run_gpu_memory_admission(output, code)
    assert result.returncode == 78, result.stdout + result.stderr
    assert "[blocked]" in result.stdout and "admitted" not in result.stdout


def test_occupied_gpu_is_busy_not_blocked_and_free_gpus_admit():
    busy = run_gpu_memory_admission("0\n4001\n0\n0", 0)
    assert busy.returncode == 75 and "[busy]" in busy.stdout and "admitted" not in busy.stdout
    free = run_gpu_memory_admission(" 727\n727\n 4000 \n0", 0)
    assert free.returncode == 0 and free.stdout.strip().endswith("admitted"), free.stdout + free.stderr


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
    result = run_gpu_memory_admission("", 0, missing_driver=True)
    assert result.returncode == 78, result.stdout + result.stderr
    assert "[blocked]" in result.stdout and "admitted" not in result.stdout


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
