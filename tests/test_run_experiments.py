"""The combined node launcher: one pass per experiment, hold lines that say why."""
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time

import pytest

import selection_gate as core

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def no_physical_gpu_cleanup(tmp_path, monkeypatch):
    """Node lifecycle tests must never inspect or terminate real GPU holders."""
    binaries = tmp_path / "bin"
    binaries.mkdir()
    nvidia = binaries / "nvidia-smi"
    nvidia.write_text('#!/usr/bin/env bash\ncase "$*" in *query-gpu=memory.used*) echo 0 ;; esac\nexit 0\n')
    nvidia.chmod(0o755)
    monkeypatch.setenv("PATH", str(binaries) + os.pathsep + os.environ["PATH"])


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
            "EXPERIMENTS_PULL": "0", "EXPERIMENTS_AUTO_PULL": "0", "EXPERIMENTS_NODE_ID": os.uname().nodename}


def test_explicit_restart_restarts_a_launcher_already_running_on_this_node(tmp_path):
    """Only explicit restart stops a running launcher and starts its replacement."""
    env = environment(tmp_path, fake_inner(tmp_path, 0, 0))
    env.pop("EXPERIMENTS_DETACHED")
    log_dir = Path(env["OM_WORK"]) / "runs/experiments/logs"
    log_dir.mkdir(parents=True)
    host = subprocess.check_output(["bash", "-c", "hostname | tr -c 'a-zA-Z0-9._-' '_'"], text=True).strip()
    # Reparented to init so its death is reaped there, not left as a zombie of this test.
    old_pid = int(subprocess.check_output(["bash", "-c",
        '''setsid bash -c 'exec -a "bash scripts/run_experiments.sh run" sleep 300' >/dev/null 2>&1 & echo $!'''],
        env=env, text=True).strip())
    (log_dir / f"launcher.{host}.pid").write_text(str(old_pid))
    fault = Path(env["OM_WORK"]) / "runs/experiments/node-faults" / f"{os.uname().nodename}.json"
    fault.parent.mkdir(parents=True)
    fault.write_text('{"phase": "train", "strikes": 2}\n')
    process = subprocess.Popen(["bash", "scripts/run_experiments.sh", "restart"], cwd=ROOT, env=env,
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
        assert "[stop] host=" in out and "[pass 1]" in out
        assert subprocess.run(["kill", "-0", str(old_pid)], capture_output=True).returncode != 0
        # The restart is the operator's second chance for this node: the fault record is cleared.
        assert f"[fault-reset] host={host}: cleared the GPU-fault record" in out and not fault.exists()
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
    assert "[pass 1] MoPPS comparison" not in out
    assert ("[hold] pass 1 ended (switch rc=78 admission failed: NCCL/CUDA probe | "
            "mopps rc=0 skipped: node busy, failed admission or cooling down); keeping this node's GPUs") in out
    assert "[holding] node retained (switch rc=78 admission failed: NCCL/CUDA probe | mopps rc=0 skipped: node busy, failed admission or cooling down); next pass in" in out
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


def test_start_never_stops_existing_processes_based_on_node_or_root_markers(tmp_path):
    env = environment(tmp_path, fake_inner(tmp_path, 78, 78))
    # Matching markers and command names do not establish that a process is orphaned.
    leftover = subprocess.Popen(["bash", "-c", 'exec -a "python scripts/_gpu_keepalive.py" sleep 300'],
                                env={**env, "OUT_ROOT": env["SWITCH_ROOT"]}, start_new_session=True)
    # A marked process that is not an experiment command must be left alone.
    bystander = subprocess.Popen(["bash", "-c", 'exec -a "python scripts/selection_switch_status.py" sleep 300'],
                                 env={**env, "OUT_ROOT": env["SWITCH_ROOT"]}, start_new_session=True)
    # A worker of another experiment root on this node is not ours to stop.
    other_root = str(Path(env["OM_WORK"]) / "runs/selection-switch-long-v1")
    other = subprocess.Popen(["bash", "-c", 'exec -a "python src/selection_switch_gpu.py run" sleep 300'],
                             env={**env, "OUT_ROOT": other_root}, start_new_session=True)
    try:
        time.sleep(.3)
        result = subprocess.run(["bash", "scripts/run_experiments.sh", "run"], cwd=ROOT, env=env,
                                capture_output=True, text=True, timeout=120)
        out = result.stdout + result.stderr
        assert result.returncode == 78, out
        assert "[orphans] stopping" not in out
        assert f"pid={bystander.pid}" not in out
        assert "no node-wide process/GPU sweep" in out
        assert leftover.poll() is None and other.poll() is None
        assert bystander.poll() is None
    finally:
        for proc in (leftover, bystander, other):
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)


def open_event(root, host, pid, *, age=100.):
    directory = root / "states/s0-t25/points/view-25/random_reduced"
    core.atomic_json(root / "switch.json", {"schema": "fake"})
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "cost.jsonl").open("a") as handle:
        handle.write(json.dumps({"event_id": "e1", "phase": "train", "ledger": "deployment", "gpus": 4, "gpu_type": "H100",
                                 "host": host, "state": "started", "time": time.time()-age, "pid": pid}) + "\n")
    return directory


def finished_rows(directory):
    return [json.loads(l) for l in (directory / "cost.jsonl").read_text().splitlines() if '"finished"' in l]


@pytest.mark.parametrize("mode", ["run", "stop"])
def test_start_and_stop_close_this_hosts_dead_cost_events_in_every_root(tmp_path, mode):
    import socket
    env = environment(tmp_path, fake_inner(tmp_path, 78, 78))
    work = Path(env["OM_WORK"])
    mine = open_event(work / "runs/selection-switch-long-v1", socket.gethostname(), 999999)
    # A killed node's attempt: closed once its heartbeat is three minutes old, not fifteen.
    dead_node = open_event(work / "runs/selection-switch-quality-v1", "some-dead-node", 4242, age=400.)
    live_node = open_event(work / "runs/selection-switch-difficulty-v1", "some-live-node", 4243, age=100.)
    result = subprocess.run(["bash", "scripts/run_experiments.sh", mode], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=120)
    out = result.stdout + result.stderr
    assert "[sweep selection-switch-long-v1] [recover-cost]" in out and "1 stale event(s) closed" in out
    assert len(finished_rows(mine)) == 1 and finished_rows(mine)[0]["event_id"] == "e1"
    assert len(finished_rows(dead_node)) == 1
    assert finished_rows(live_node) == []


def test_historical_gpu_fault_without_event_binding_preserves_costs_and_failure(tmp_path):
    from test_waive_stalled_attempts import branch as faulted_branch
    env = environment(tmp_path, fake_inner(tmp_path, 78, 78))
    root = Path(env["SWITCH_ROOT"])
    core.atomic_json(root / "switch.json", {"schema": "x"})
    directory = faulted_branch(root, "random_reduced")
    assert (directory / "failure.json").exists()
    # This shared fixture has an old append-only CUDA log, not an event-bound
    # watchdog/signal/stale-owner receipt. It cannot authorize an automatic refund.
    before_cost = (directory / "cost.jsonl").read_bytes()
    before_failure = (directory / "failure.json").read_bytes()
    result = subprocess.run(["bash", "scripts/run_experiments.sh", "run"], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=120)
    out = result.stdout + result.stderr
    assert "without event-bound infrastructure evidence" in out
    assert "GPU-s returned to the allocation" not in out
    assert (directory / "cost.jsonl").read_bytes() == before_cost
    assert (directory / "failure.json").read_bytes() == before_failure
    assert not (directory / "waivers/train1.json").exists()
    assert out.index("[auto-waive]") < out.index("[pass 1] selection switch")


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
             "CUDA_VISIBLE_DEVICES": "", "EXPERIMENTS_NODE_ID": os.uname().nodename}
    result = subprocess.run(["bash", "scripts/run_selection_switch.sh", "run"], cwd=ROOT, env=inner,
                            capture_output=True, text=True, timeout=120)
    # A fresh first strike is a cooldown (79), not the terminal admission failure (78).
    assert "[cooldown] host=" in result.stdout + result.stderr, result.stdout + result.stderr
    assert result.returncode == 79
    # A first strike expires after EXPERIMENTS_FAULT_TTL_SECONDS: the probe decides again.
    (faults / f"{os.uname().nodename}.json").write_text(json.dumps({"phase": "train", "time": time.time() - 4000, "strikes": 1}))
    result = subprocess.run(["bash", "scripts/run_selection_switch.sh", "run"], cwd=ROOT, env=inner,
                            capture_output=True, text=True, timeout=120)
    out = result.stdout + result.stderr
    assert "[fault-expired] host=" in out and "[blocked] host=" not in out and "[cooldown]" not in out, out
    assert result.returncode not in (78, 79), out
    # A second strike blocks until an operator restart clears the record.
    (faults / f"{os.uname().nodename}.json").write_text(json.dumps({"phase": "train", "time": time.time() - 4000, "strikes": 2}))
    result = subprocess.run(["bash", "scripts/run_selection_switch.sh", "run"], cwd=ROOT, env=inner,
                            capture_output=True, text=True, timeout=120)
    assert "[blocked] host=" in result.stdout + result.stderr and "strike 2" in result.stdout + result.stderr
    assert result.returncode == 78


def test_a_node_with_nothing_to_claim_works_sibling_roots_in_priority_order(tmp_path):
    """Own root out of claimable work (rc=1) -> the node takes v1, difficulty, long... siblings before holding."""
    calls = tmp_path / "calls.txt"
    fake = tmp_path / "fake-inner.sh"
    fake.write_text("#!/usr/bin/env bash\n"
                    f'case "$1" in *run_selection_switch.sh) echo "$SWITCH_ROOT" >> "{calls}"; '
                    'case "$SWITCH_ROOT" in *selection-switch-long-v1) exit 1 ;; *) exit 0 ;; esac ;; '
                    '*run_mopps_comparison.sh) exit 0 ;; esac\nexit 9\n')
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    env = environment(tmp_path, fake)
    work = Path(env["OM_WORK"])
    env["SWITCH_ROOT"] = str(work / "runs/selection-switch-long-v1")
    for name in ("selection-switch-long-v1", "selection-switch-difficulty-v1", "selection-switch-v1", "selection-switch-hard-v1"):
        core.atomic_json(work / "runs" / name / "switch.json", {"schema": "fake"})
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
    finally:
        if process.poll() is None:
            process.kill()
    out = "\n".join(lines)
    roots = [Path(l).name for l in calls.read_text().split()]
    assert roots == ["selection-switch-long-v1", "selection-switch-v1", "selection-switch-difficulty-v1", "selection-switch-hard-v1"], out
    assert "[pass 1] selection switch ended: rc=1, failed tasks, see [failed] lines above" in out
    assert "[pass 1] sibling selection-switch-v1 ended: rc=0, nothing left to claim" in out
    assert out.index("sibling selection-switch-v1") < out.index("sibling selection-switch-difficulty-v1") < out.index("sibling selection-switch-hard-v1")
    assert ("[hold] pass 1 ended (switch rc=1 failed tasks, see [failed] lines above | selection-switch-v1 rc=0 nothing left to claim"
            " | selection-switch-difficulty-v1 rc=0 nothing left to claim | selection-switch-hard-v1 rc=0 nothing left to claim"
            " | mopps rc=0 nothing left to claim)") in out
    # Sibling progress counts as progress: the hold is the base interval, not doubled.
    assert "next pass in 1s" in lines[-1]


def test_a_holding_node_resumes_as_soon_as_a_branch_becomes_claimable(tmp_path):
    """The hold polls the status snapshot; a READY branch ends the hold before the timer."""
    from test_selection_switch_status import completed_prefix, prepared
    env = environment(tmp_path, fake_inner(tmp_path, 1, 0))
    env.update({"EXPERIMENTS_HOLD_SECONDS": "40", "EXPERIMENTS_HOLD_POLL_SECONDS": "2", "EXPERIMENTS_HELP_SIBLINGS": "0"})
    root = Path(env["SWITCH_ROOT"])
    prepared(root)
    completed_prefix(root)  # seed 0 prefix 25 published: its two dev branches are READY
    process = subprocess.Popen(["bash", "scripts/run_experiments.sh", "run"], cwd=ROOT, env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    lines, deadline = [], time.monotonic() + 60
    try:
        while time.monotonic() < deadline:
            line = process.stdout.readline()
            if not line:
                break
            lines.append(line.rstrip("\n"))
            if line.startswith("[pass 2]"):
                break
        os.killpg(process.pid, 15)
        process.wait(timeout=30)
    finally:
        if process.poll() is None:
            process.kill()
    out = "\n".join(lines)
    assert "[hold] claimable work in selection-switch-v1; starting the next pass now" in out, out
    assert out.index("[holding]") < out.index("claimable work") < out.index("[pass 2]")
    assert out.count("[holding]") <= 3


def test_controller_finishes_owned_work_then_yields_primary_peer_wait_to_long_suite(tmp_path):
    calls = tmp_path / "queue-calls.jsonl"
    fake = tmp_path / "queue-inner.py"
    fake.write_text('''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
root = Path(os.environ["SWITCH_ROOT"])
with open(os.environ["QUEUE_CALLS"], "a") as handle:
    handle.write(json.dumps({"root": root.name, "queue": os.environ.get("SWITCH_QUEUE_PASS")}) + "\\n")
if root.name == "selection-switch-v1":
    if os.environ.get("SWITCH_QUEUE_PASS") != "1":
        sys.exit(78)
    # A claimed task finishes before the peer-wait callback can yield.
    Path(os.environ["OWNED_FINISHED"]).write_text("published")
    print("[queue-yield] peer-owned tasks; returning to shared queue", flush=True)
    sys.exit(0)
assert Path(os.environ["OWNED_FINISHED"]).read_text() == "published"
sys.exit(130)
''')
    fake.chmod(0o755)
    env = environment(tmp_path, fake)
    env.update({"EXPERIMENTS_CLEAN": "0", "QUEUE_CALLS": str(calls),
                "OWNED_FINISHED": str(tmp_path / "owned-finished")})
    for name in ("selection-switch-v1", "selection-switch-long-v1"):
        core.atomic_json(Path(env["OM_WORK"]) / "runs" / name / "switch.json", {"schema": "fake"})
    result = subprocess.run(["bash", "scripts/run_experiments.sh", "run"], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=20, check=False)
    assert result.returncode == 130, result.stdout + result.stderr
    rows = [json.loads(line) for line in calls.read_text().splitlines()]
    assert rows == [{"root": "selection-switch-v1", "queue": "1"},
                    {"root": "selection-switch-long-v1", "queue": "1"}]
    assert "[queue-yield]" in result.stdout and "sibling selection-switch-long-v1" in result.stdout


def test_controller_mopps_pass_has_no_internal_peer_or_prefix_wait(tmp_path):
    calls = tmp_path / "mopps-args.json"
    fake = tmp_path / "mopps-inner.py"
    fake.write_text('''#!/usr/bin/env python3
import json, os, sys
with open(os.environ["QUEUE_CALLS"], "w") as handle:
    json.dump({"args": sys.argv[1:], "queue": os.environ.get("SWITCH_QUEUE_PASS")}, handle)
sys.exit(130)
''')
    fake.chmod(0o755)
    env = environment(tmp_path, fake)
    env.update({"EXPERIMENTS_CLEAN": "0", "EXPERIMENTS_SKIP_SWITCH": "1",
                "EXPERIMENTS_HELP_SIBLINGS": "0", "QUEUE_CALLS": str(calls)})
    result = subprocess.run(["bash", "scripts/run_experiments.sh", "run"], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=20, check=False)
    assert result.returncode == 130, result.stdout + result.stderr
    assert json.loads(calls.read_text()) == {"args": ["scripts/run_mopps_comparison.sh"], "queue": "1"}


@pytest.mark.parametrize("queue", ["0", "1"])
def test_mopps_queue_yield_keeps_automatic_failed_branch_retry(tmp_path, queue):
    """Execute the real launcher loop with CPU-only worker callbacks."""
    root = tmp_path / "comparison"
    core.atomic_json(root / "states/s3-t25/random_online/failure.json", {"error": "interrupted"})
    checkpoint = root / "states/s3-t25/random_online/policy/checkpoint-000030/checkpoint_state.json"
    core.atomic_json(checkpoint, {"completed_steps": 30})
    before = checkpoint.read_bytes()
    calls = tmp_path / "worker-calls.txt"
    launcher = (ROOT / "scripts/run_mopps_comparison.sh").read_text()
    tail = "mopps_retry_failures()" + launcher.split("mopps_retry_failures()", 1)[1]
    script = 'set -euo pipefail\nrc=0\nselection_run_worker() { printf "%s\\n" "$*" >> "$QUEUE_CALLS"; }\n' + tail
    result = subprocess.run(["bash", "-c", script], cwd=ROOT, text=True, capture_output=True, timeout=10, check=False,
        env={**os.environ, "OUT_ROOT": str(root), "PY": sys.executable, "MODE": "run",
             "SWITCH_QUEUE_PASS": queue, "SWITCH_HOLD_SECONDS": "0", "SWITCH_AUTO_RECOVER": "0",
             "MOPPS_AUTO_RETRY": "1", "QUEUE_CALLS": str(calls)})
    assert result.returncode == 0, result.stdout + result.stderr
    rows = calls.read_text().splitlines()
    assert len(rows) == 2 and " retry --root " in rows[0] and " run --root " in rows[1]
    assert "--idle-timeout 0" in rows[0]
    assert ("--idle-timeout 0" in rows[1]) is (queue == "1")
    assert checkpoint.read_bytes() == before


def test_node_identity_tells_two_containers_with_one_hostname_apart(tmp_path):
    """The identity is the hostname plus a suffix from the node's GPUs and container; the env wins."""
    script = "source scripts/_node_id.sh; echo \"$EXPERIMENTS_NODE_ID\""
    plain = subprocess.run(["bash", "-c", script], cwd=ROOT, env={"PATH": os.environ["PATH"]}, capture_output=True, text=True, check=True).stdout.strip()
    assert plain.startswith(os.uname().nodename)
    forced = subprocess.run(["bash", "-c", script], cwd=ROOT, env={"PATH": os.environ["PATH"], "EXPERIMENTS_NODE_ID": "node-a-g1234"},
                            capture_output=True, text=True, check=True).stdout.strip()
    assert forced == "node-a-g1234"
    other = subprocess.run(["bash", "-c", script], cwd=ROOT, env={"PATH": os.environ["PATH"], "CUDA_VISIBLE_DEVICES": "4,5,6,7"},
                           capture_output=True, text=True, check=True).stdout.strip()
    assert other.startswith(os.uname().nodename) and other != plain
