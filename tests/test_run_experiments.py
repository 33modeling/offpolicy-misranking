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
            "SWITCH_PYTHON": sys.executable, "EXPERIMENTS_DETACHED": "1", "EXPERIMENTS_KEEPALIVE": "0",
            "EXPERIMENTS_HOLD_SECONDS": "1", "EXPERIMENTS_INNER": str(fake), "CUDA_VISIBLE_DEVICES": ""}


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
