"""The combined node launcher: one pass per experiment, hold lines that say why."""
import os
from pathlib import Path
import stat
import subprocess
import sys
import time

import selection_gate as core

ROOT = Path(__file__).resolve().parents[1]


def fake_inner(tmp_path, switch_rc, mopps_rc):
    fake = tmp_path / "fake-inner.sh"
    fake.write_text("#!/usr/bin/env bash\n"
                    f'case "$1" in *run_selection_switch.sh) echo "[launcher-start] fake switch"; exit {switch_rc} ;; '
                    f'*run_mopps_comparison.sh) echo "[launcher-start] fake mopps"; exit {mopps_rc} ;; esac\nexit 9\n')
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    return fake


def environment(tmp_path, fake):
    work = tmp_path / "work"
    switch_root, mopps_root = work / "runs/selection-switch-v1", work / "runs/mopps-comparison-v1"
    core.atomic_json(mopps_root / "mopps.json", {"schema": "fake"})
    return {**os.environ, "OM_WORK": str(work), "SWITCH_ROOT": str(switch_root), "MOPPS_ROOT": str(mopps_root),
            "SWITCH_PYTHON": sys.executable, "EXPERIMENTS_DETACHED": "1", "EXPERIMENTS_KEEPALIVE": "0", "EXPERIMENTS_WATCHDOG": "0",
            "EXPERIMENTS_HOLD_SECONDS": "1", "EXPERIMENTS_INNER": str(fake), "CUDA_VISIBLE_DEVICES": "",
            "EXPERIMENTS_PULL": "0"}


def test_run_restarts_a_launcher_already_running_on_this_node(tmp_path):
    """One command per node: run stops the launcher already running here, then starts."""
    env = environment(tmp_path, fake_inner(tmp_path, 0, 0))
    env.pop("EXPERIMENTS_DETACHED")
    log_dir = Path(env["OM_WORK"]) / "runs/experiments/logs"
    log_dir.mkdir(parents=True)
    host = subprocess.check_output(["bash", "-c", "hostname | tr -c 'a-zA-Z0-9._-' '_'"], text=True).strip()
    # Reparented to init so its death is reaped there, not left as a zombie of this test.
    old_pid = int(subprocess.check_output(["bash", "-c", "setsid sleep 300 >/dev/null 2>&1 & echo $!"], text=True).strip())
    (log_dir / f"launcher.{host}.pid").write_text(str(old_pid))
    process = subprocess.Popen(["bash", "scripts/run_experiments.sh", "run"], cwd=ROOT, env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    lines, deadline = [], time.monotonic() + 90
    try:
        while time.monotonic() < deadline:
            line = process.stdout.readline()
            if not line:
                break
            lines.append(line.rstrip("\n"))
            if line.startswith("[holding]"):
                break
        os.killpg(process.pid, 15)
        process.wait(timeout=30)
        out = "\n".join(lines)
        assert f"[restart] host={host}: a node launcher is already running (pid {old_pid}); stopping it first" in out
        assert "[stop] host=" in out and "[pass 1]" in out
        assert subprocess.run(["kill", "-0", str(old_pid)], capture_output=True).returncode != 0
    finally:
        if process.poll() is None:
            process.kill()
        subprocess.run(["kill", "-9", str(old_pid)], capture_output=True)


def test_two_blocked_passes_release_the_node_and_every_hold_line_says_why(tmp_path):
    env = environment(tmp_path, fake_inner(tmp_path, 78, 78))
    result = subprocess.run(["bash", "scripts/run_experiments.sh", "run"], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 78, result.stdout + result.stderr
    out = result.stdout
    assert "[pass 1] selection switch ended: rc=78, admission failed: NCCL/CUDA probe" in out
    assert "[pass 1] MoPPS comparison ended: rc=78, admission failed: NCCL/CUDA probe" in out
    assert ("[hold] pass 1 ended (switch rc=78 admission failed: NCCL/CUDA probe | "
            "mopps rc=78 admission failed: NCCL/CUDA probe); keeping this node's GPUs") in out
    assert "[holding] node retained (switch rc=78 admission failed: NCCL/CUDA probe | mopps rc=78 admission failed: NCCL/CUDA probe); next pass in" in out
    assert "[blocked] node admission failed on two passes" in out
    assert "[node-launcher-exit]" in out and "rc=78" in out.splitlines()[-1]


def test_failed_and_idle_passes_hold_with_their_reasons_until_stopped(tmp_path):
    env = environment(tmp_path, fake_inner(tmp_path, 1, 0))
    process = subprocess.Popen(["bash", "scripts/run_experiments.sh", "run"], cwd=ROOT, env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    lines, deadline = [], time.monotonic() + 60
    try:
        while time.monotonic() < deadline:
            line = process.stdout.readline()
            if not line:
                break
            lines.append(line.rstrip("\n"))
            if line.startswith("[holding]"):
                break
        os.killpg(process.pid, 15)
        process.wait(timeout=30)
    finally:
        if process.poll() is None:
            process.kill()
    out = "\n".join(lines)
    assert "[pass 1] selection switch ended: rc=1, failed tasks, see [failed] lines above" in out
    assert "[pass 1] MoPPS comparison ended: rc=0, nothing left to claim" in out
    assert "[hold] pass 1 ended (switch rc=1 failed tasks, see [failed] lines above | mopps rc=0 nothing left to claim)" in out
    assert lines[-1].startswith("[holding] node retained (switch rc=1 failed tasks, see [failed] lines above | mopps rc=0 nothing left to claim); next pass in")
    # On a node this output is the console log; the status view reads the reason back from it.
    sys.path.insert(0, str(ROOT / "scripts"))
    import _node_view as view
    root = Path(env["SWITCH_ROOT"])
    console = view.node_launcher_logs(root) / "console.node-t_.log"
    console.parent.mkdir(parents=True, exist_ok=True)
    console.write_text(out + "\n")
    node = view.launcher_nodes(root, [], now=time.time())[0]
    assert node["host"] == "node-t" and node["state"] == "HOLD"
    assert node["reason"] == "switch rc=1 failed tasks, see [failed] lines above | mopps rc=0 nothing left to claim"
    assert view.render_summary([node]) == "NODES  1 live  |  HOLD 1"


def test_leftover_processes_of_either_root_are_stopped_before_the_first_pass(tmp_path):
    env = environment(tmp_path, fake_inner(tmp_path, 78, 78))
    # An orphaned keepalive of an earlier launcher: our marker, our command name, its own group.
    leftover = subprocess.Popen(["bash", "-c", 'exec -a "python scripts/_gpu_keepalive.py" sleep 300'],
                                env={**env, "OUT_ROOT": env["SWITCH_ROOT"]}, start_new_session=True)
    # A marked process that is not an experiment command must be left alone.
    bystander = subprocess.Popen(["bash", "-c", 'exec -a "python scripts/selection_switch_status.py" sleep 300'],
                                 env={**env, "OUT_ROOT": env["SWITCH_ROOT"]}, start_new_session=True)
    try:
        time.sleep(.3)
        result = subprocess.run(["bash", "scripts/run_experiments.sh", "run"], cwd=ROOT, env=env,
                                capture_output=True, text=True, timeout=120)
        assert result.returncode == 78, result.stdout + result.stderr
        assert f"[clean] leftover pid={leftover.pid}" in result.stdout + result.stderr
        assert f"pid={bystander.pid}" not in result.stdout + result.stderr
        assert result.stdout.index("[clean] host=") < result.stdout.index("[pass 1] selection switch")
        assert leftover.wait(timeout=10) != 0
        assert bystander.poll() is None
    finally:
        for proc in (leftover, bystander):
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)


def test_node_launcher_runs_the_stall_watchdog_for_its_life(tmp_path):
    env = {**environment(tmp_path, fake_inner(tmp_path, 78, 78)), "EXPERIMENTS_WATCHDOG": "1"}
    result = subprocess.run(["bash", "scripts/run_experiments.sh", "run"], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 78, result.stdout + result.stderr
    assert "[watchdog] pid=" in result.stdout
    log = Path(env["OM_WORK"]) / "runs/experiments/logs" / f"stall.{os.uname().nodename.replace('.', '_')}.log"
    logs = list((Path(env["OM_WORK"]) / "runs/experiments/logs").glob("stall.*.log"))
    assert logs and "[watchdog] pid=" in logs[0].read_text()
    time.sleep(1)
    pid = int(result.stdout.split("[watchdog] pid=")[1].split()[0])
    assert not Path(f"/proc/{pid}").exists(), "the watchdog dies with the launcher"


def test_recorded_gpu_fault_blocks_gpu_work_on_that_host(tmp_path):
    env = environment(tmp_path, fake_inner(tmp_path, 0, 0))
    faults = Path(env["OM_WORK"]) / "runs/experiments/node-faults"
    faults.mkdir(parents=True)
    (faults / f"{os.uname().nodename}.json").write_text('{"phase": "train"}\n')
    core.atomic_json(Path(env["SWITCH_ROOT"]) / "switch.json", {"schema": "fake"})
    inner = {**os.environ, "SWITCH_ROOT": env["SWITCH_ROOT"], "OM_WORK": env["OM_WORK"], "SWITCH_PYTHON": sys.executable,
             "SWITCH_FOREGROUND": "1", "SWITCH_HOLD_SECONDS": "0", "SWITCH_KEEPALIVE": "0", "SWITCH_RUNTIME_REPO": str(ROOT),
             "CUDA_VISIBLE_DEVICES": ""}
    result = subprocess.run(["bash", "scripts/run_selection_switch.sh", "run"], cwd=ROOT, env=inner,
                            capture_output=True, text=True, timeout=120)
    assert "[blocked] host=" in result.stdout + result.stderr, result.stdout + result.stderr
    assert result.returncode == 78
