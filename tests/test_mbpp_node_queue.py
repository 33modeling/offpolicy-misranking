"""Real node controllers with a CPU-only leased worker; no physical GPU access."""

import json
import importlib.util
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("mbpp_queue_worker", ROOT / "scripts/queue_selection_switch_gpu.py")
queue_worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(queue_worker)


def test_original_worker_yields_when_only_live_peer_tasks_remain(tmp_path, monkeypatch, capsys):
    from test_selection_switch_gpu import simulated_queue
    worker = queue_worker.worker
    calls, _ = simulated_queue(tmp_path, monkeypatch)
    original = worker.wait_for_peers
    def busy(*args, **kwargs):
        raise BlockingIOError("owned by a live peer")
    monkeypatch.setattr(worker.base, "lease", busy)
    monkeypatch.setattr(worker, "busy_task", lambda *args: {
        "task": "prefix", "active": True, "host": "healthy-peer", "pid": 123, "phase": "train"})
    monkeypatch.setattr(worker, "main", lambda: worker.work(tmp_path, idle_timeout=600))
    assert queue_worker.run() == 0
    assert calls == [] and "queue-yield" in capsys.readouterr().out
    assert worker.wait_for_peers is original


def test_queue_callback_is_restored_after_worker_failure(monkeypatch):
    worker = queue_worker.worker
    original = worker.wait_for_peers
    def fail():
        raise RuntimeError("node failure")
    monkeypatch.setattr(worker, "main", fail)
    with pytest.raises(RuntimeError, match="node failure"):
        queue_worker.run()
    assert worker.wait_for_peers is original


def wait_for(predicate, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.05)
    assert predicate(), "node simulation timed out"


def publish_prefixes(root):
    root.mkdir(parents=True, exist_ok=True)
    (root / "switch.json").write_text("{}")
    for seed in range(5):
        path = root / f"prefixes/seed-{seed}"
        path.mkdir(parents=True, exist_ok=True)
        for step in (25, 50, 100):
            (path / f"prefix-{step}.json").write_text("{}")


@pytest.fixture
def cluster(tmp_path):
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    for name in ("run_experiments.sh", "run_mbpp_experiments.sh", "_mbpp_experiments.sh", "_stall_watchdog.py", "_mbpp_node_guard.py"):
        shutil.copy(ROOT / "scripts" / name, scripts)
    (repo / "src").mkdir()
    shutil.copy(ROOT / "src/cleanup_run_processes.py", repo / "src")
    (scripts / "run_selection_switch.sh").write_text("#!/usr/bin/env bash\nexit 0\n")
    for name in ("recover_selection_switch_cost.py", "waive_stalled_attempts.py", "split_curve_ledger.py"):
        (scripts / name).write_text("pass\n")
    (scripts / "check_mbpp_experiments.py").write_text(
        'import os, sys\nfrom pathlib import Path\n'
        'assert os.environ["CUDA_VISIBLE_DEVICES"] == ""\n'
        'print("[fake-check] input check", flush=True)\n'
        'sys.exit(1 if os.environ.get("TEST_REQUIRE_INPUTS") and not Path(os.environ["OM_WORK"], "inputs-ready").exists() else 0)\n')
    (scripts / "selection_switch_status.py").write_text(
        'import json, sys\nfrom pathlib import Path\n'
        'root = Path(sys.argv[sys.argv.index("--root")+1])\n'
        'done = len(list(root.glob("tasks/*/result.json"))) == 3\n'
        'print(json.dumps({"development_done": 18 if done else 0, "test_done": 30 if done else 0, '
        '"tasks": [] if done else [{"status": "READY"}]}))\n')
    engine = scripts / "fake_engine.py"
    engine.write_text('''import fcntl, json, os, sys, time
from pathlib import Path
root = Path(os.environ["SWITCH_ROOT"])
node = os.environ["EXPERIMENTS_NODE_ID"]
root.mkdir(parents=True, exist_ok=True)
def event(kind, **fields):
    row = dict(kind=kind, node=node, root=root.name, selector=os.environ["SWITCH_SELECTOR"],
               accounting=os.environ["SWITCH_ACCOUNTING"], gate=os.environ["SWITCH_GATE"], **fields)
    with open(os.environ["TEST_EVENTS"], "a") as f:
        f.write(json.dumps(row) + "\\n")
event("pass")
if os.environ.get("TEST_FAIL_SUITE") and os.environ["TEST_FAIL_SUITE"] in root.name:
    sys.exit(int(os.environ.get("TEST_FAIL_RC", "78")))
(root / "switch.json").write_text("{}")
if root.name == "selection-switch-mbpp-v1" and not os.environ.get("TEST_DELAY_PREFIXES"):
    for seed in range(5):
        path = root / f"prefixes/seed-{seed}"
        path.mkdir(parents=True, exist_ok=True)
        for step in (25, 50, 100):
            (path / f"prefix-{step}.json").write_text("{}")
for index in range(3):
    task = root / "tasks" / str(index)
    task.mkdir(parents=True, exist_ok=True)
    with (task / ".lease").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            continue
        if (task / "result.json").exists():
            continue
        checkpoint = task / "checkpoint.json"
        previous = json.loads(checkpoint.read_text()) if checkpoint.exists() else None
        if previous is None:
            checkpoint.write_text(json.dumps({"node": node}))
        event("claim", task=index, resumed=previous)
        if os.environ.get("TEST_BLOCK_NODE") == node and root.name == "selection-switch-mbpp-v1" and index == 0:
            Path(os.environ["OM_WORK"], "node-blocked").write_text(node)
            time.sleep(300)
        time.sleep(.1)
        result = {"node": node, "resumed": previous}
        temporary = task / f"result.{os.getpid()}.tmp"
        temporary.write_text(json.dumps(result))
        temporary.replace(task / "result.json")
        event("finished", task=index)
        break
''')
    inner = scripts / "fake_inner.sh"
    inner.write_text('#!/usr/bin/env bash\nexec "$TEST_PYTHON" "$TEST_ENGINE" "$@"\n')
    inner.chmod(0o755)
    binaries = tmp_path / "bin"
    binaries.mkdir()
    nvidia = binaries / "nvidia-smi"
    nvidia.write_text('#!/usr/bin/env bash\ncase "$*" in *query-gpu=memory.used*) echo 0 ;; esac\nexit 0\n')
    nvidia.chmod(0o755)
    work = tmp_path / "shared work"
    work.mkdir()
    # An unfinished unrelated experiment must not block the MBPP completion check.
    unrelated = work / "runs/mopps-comparison-v1"
    unrelated.mkdir(parents=True)
    (unrelated / "mopps.json").write_text("{}")
    env = {**os.environ, "OM_WORK": str(work), "SWITCH_PYTHON": sys.executable,
           "TEST_PYTHON": sys.executable, "TEST_ENGINE": str(engine), "TEST_EVENTS": str(work / "events.jsonl"),
           "EXPERIMENTS_DETACHED": "1", "EXPERIMENTS_PULL": "0", "EXPERIMENTS_AUTO_PULL": "0",
           "EXPERIMENTS_KEEPALIVE": "0", "EXPERIMENTS_WATCHDOG": "0", "EXPERIMENTS_CLEAN": "0",
           "EXPERIMENTS_HOLD_SECONDS": "1", "EXPERIMENTS_HOLD_POLL_SECONDS": "1",
           "EXPERIMENTS_INNER": str(inner), "CUDA_VISIBLE_DEVICES": "", "PATH": str(binaries) + os.pathsep + os.environ["PATH"]}
    processes = []
    def start(node, mode="run", **overrides):
        log = work / f"{node}-{len(processes)}.log"
        handle = log.open("w")
        process = subprocess.Popen(["bash", "scripts/run_mbpp_experiments.sh", mode], cwd=repo,
                                   env={**env, "EXPERIMENTS_NODE_ID": node, **overrides},
                                   stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        processes.append((process, handle))
        return process, log
    yield work, start
    for process, handle in processes:
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=20)
        handle.close()
    # Even a failed test that deliberately killed the guard must not leak its
    # separately-sessioned children into other tests or the developer's node.
    import cleanup_run_processes as cleanup
    for receipt in (work / "runs/experiments/logs").glob("mbpp-controller.*.owner.json"):
        token = json.loads(receipt.read_text())["token"]
        cleanup.terminate("/unused-test-mbpp-scope", timeout=2, command_patterns=("",),
                          required_environment=(("OM_MBPP_CONTROLLER_TOKEN", token),), compact=True)


def events(work):
    path = work / "events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_two_nodes_and_a_replacement_resume_a_killed_owner_without_duplicate_results(cluster):
    work, start = cluster
    first, first_log = start("node-a", TEST_BLOCK_NODE="node-a")
    wait_for(lambda: (work / "node-blocked").exists())
    second, second_log = start("node-b")
    # Quality/difficulty must be claimable while fresh's task 0 is still running.
    wait_for(lambda: any(row["kind"] == "finished" and "difficulty" in row["root"] for row in events(work)))
    assert first.poll() is None
    # Kill the owner only. Its separately-sessioned child must be reclaimed by
    # the replacement on the SAME node, not mistaken for a healthy peer.
    os.kill(first.pid, signal.SIGKILL)
    first.wait(timeout=5)
    replacement, replacement_log = start("node-a")
    assert second.wait(timeout=30) == 0, second_log.read_text()
    assert replacement.wait(timeout=30) == 0, replacement_log.read_text()
    rows = [row for row in events(work) if row["kind"] == "finished"]
    assert len(rows) == len({(row["root"], row["task"]) for row in rows}) == 9
    recovered = json.loads((work / "runs/selection-switch-mbpp-v1/tasks/0/result.json").read_text())
    assert recovered["resumed"] == {"node": "node-a"}
    assert recovered["node"] in ("node-b", "node-a")
    expected = {"selection-switch-mbpp-v1": ("fresh_r", "budget", "final"),
                "selection-switch-mbpp-quality-v1": ("fresh_r", "matched", "convergence"),
                "selection-switch-mbpp-difficulty-v1": ("difficulty", "budget", "convergence")}
    for row in rows:
        assert (row["selector"], row["accounting"], row["gate"]) == expected[row["root"]]


def test_pending_variants_keep_node_alive_and_open_when_peer_publishes_prefixes(cluster):
    work, start = cluster
    process, log = start("node-pending", TEST_DELAY_PREFIXES="1")
    fresh = work / "runs/selection-switch-mbpp-v1"
    wait_for(lambda: len(list(fresh.glob("tasks/*/result.json"))) == 3)
    time.sleep(.3)
    assert process.poll() is None, log.read_text()
    assert "shared fresh prefixes are not ready" in log.read_text()
    publish_prefixes(fresh)
    assert process.wait(timeout=30) == 0, log.read_text()
    assert len([row for row in events(work) if row["kind"] == "finished"]) == 9


@pytest.mark.parametrize("code", ["78", "79"])
def test_sibling_gpu_fault_stops_other_work_and_repeated_admission_failure_releases_node(cluster, code):
    work, start = cluster
    process, log = start("node-fault", TEST_FAIL_SUITE="quality", TEST_FAIL_RC=code)
    if code == "78":
        assert process.wait(timeout=30) == 78, log.read_text()
        assert "node admission failed on two passes" in log.read_text()
    else:
        wait_for(lambda: "cooling down after a GPU fault" in log.read_text())
    assert not any("difficulty" in row["root"] for row in events(work))


def test_cleanup_precedes_input_check_and_missing_inputs_retry_instead_of_exiting(cluster):
    work, start = cluster
    process, log = start("node-inputs", EXPERIMENTS_CLEAN="1", TEST_REQUIRE_INPUTS="1")
    wait_for(lambda: "[holding]" in log.read_text())
    text = log.read_text()
    assert text.index("[clean] host=") < text.index("[fake-check]")
    assert process.poll() is None and not events(work)
    (work / "inputs-ready").write_text("ready")
    assert process.wait(timeout=30) == 0, log.read_text()


def test_watchdog_already_watches_variant_roots_before_they_are_prepared(cluster):
    work, start = cluster
    process, log = start("node-watch", EXPERIMENTS_WATCHDOG="1", TEST_REQUIRE_INPUTS="1")
    watchdog = work / "runs/experiments/logs/stall.node-watch_.log"
    wait_for(lambda: watchdog.exists() and "roots=" in watchdog.read_text())
    text = watchdog.read_text()
    assert "selection-switch-mbpp-quality-v1" in text and "selection-switch-mbpp-difficulty-v1" in text
    assert "mopps-comparison-v1" not in text
    assert not (work / "runs/selection-switch-mbpp-quality-v1").exists()
    os.killpg(process.pid, signal.SIGTERM)
    process.wait(timeout=10)


def test_peer_completion_releases_node_without_waiting_out_a_long_hold(cluster):
    work, start = cluster
    process, log = start("node-idle", TEST_REQUIRE_INPUTS="1", EXPERIMENTS_HOLD_SECONDS="40")
    wait_for(lambda: "[holding]" in log.read_text())
    for name in ("selection-switch-mbpp-v1", "selection-switch-mbpp-quality-v1", "selection-switch-mbpp-difficulty-v1"):
        root = work / "runs" / name
        root.mkdir(parents=True, exist_ok=True)
        (root / "switch.json").write_text("{}")
        for index in range(3):
            task = root / "tasks" / str(index)
            task.mkdir(parents=True)
            (task / "result.json").write_text("{}")
    assert process.wait(timeout=5) == 0, log.read_text()
    assert "peers completed every experiment" in log.read_text()


@pytest.mark.parametrize('code', ['1', '75'])
def test_ordinary_failure_and_busy_lock_backoff_are_capped_at_sixty_seconds(cluster, code):
    _work, start = cluster
    process, log = start('node-short-hold', TEST_FAIL_SUITE='mbpp-v1', TEST_FAIL_RC=code,
                         EXPERIMENTS_HOLD_SECONDS='40', TEST_DELAY_PREFIXES='1', EXPERIMENTS_HELP_SIBLINGS='0')
    wait_for(lambda: '[holding]' in log.read_text())
    assert 'next pass in 60s' in log.read_text()
    assert 'next pass in 80s' not in log.read_text()
    assert process.poll() is None


def test_gpu_cooldown_keeps_its_separate_backoff(cluster):
    _work, start = cluster
    process, log = start('node-cooldown', TEST_FAIL_SUITE='mbpp-v1', TEST_FAIL_RC='79',
                         EXPERIMENTS_HOLD_SECONDS='40')
    wait_for(lambda: '[holding]' in log.read_text())
    assert 'next pass in 80s' in log.read_text()
    assert process.poll() is None


def test_zero_poll_interval_is_rejected_instead_of_spinning_forever(cluster):
    work, start = cluster
    process, log = start('node-invalid-poll', EXPERIMENTS_HOLD_POLL_SECONDS='0')
    assert process.wait(timeout=10) == 2, log.read_text()
    assert 'must be a positive whole number' in log.read_text()
    assert not events(work)


def test_duplicate_node_launch_does_not_stop_or_duplicate_the_live_controller(cluster):
    work, start = cluster
    first, log = start("node-duplicate", TEST_BLOCK_NODE="node-duplicate")
    wait_for(lambda: (work / "node-blocked").exists())
    claims = [row for row in events(work) if row["kind"] == "claim"]
    duplicate, duplicate_log = start("node-duplicate")
    assert duplicate.wait(timeout=10) == 0, duplicate_log.read_text()
    assert "[already running]" in duplicate_log.read_text()
    assert first.poll() is None
    assert [row for row in events(work) if row["kind"] == "claim"] == claims


def test_busy_mbpp_pass_never_sweeps_other_experiment_processes(cluster):
    work, start = cluster
    # Looks like a real GPU worker, even under the same work volume. It is not
    # owned by this MBPP controller and must survive both initial and busy passes.
    unrelated = subprocess.Popen(["bash", "-c", 'exec -a "python src/selection_switch_gpu.py run" sleep 120'],
        env={**os.environ, "OUT_ROOT": str(work / "runs/selection-switch-math-v1")}, start_new_session=True)
    try:
        process, log = start("node-busy", EXPERIMENTS_CLEAN="1", TEST_FAIL_SUITE="mbpp-v1", TEST_FAIL_RC="75")
        wait_for(lambda: log.exists() and "[pass 2]" in log.read_text())
        assert process.poll() is None and unrelated.poll() is None
        assert "no node-wide process/GPU sweep" in log.read_text()
        assert "[clean] leftover pid=" not in log.read_text()
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)


def test_explicit_stop_reaps_guard_and_same_node_can_resume(cluster):
    work, start = cluster
    first, log = start("node-stop", TEST_BLOCK_NODE="node-stop")
    wait_for(lambda: (work / "node-blocked").exists())
    stopped, stop_log = start("node-stop", mode="stop")
    assert stopped.wait(timeout=20) == 0, stop_log.read_text()
    assert first.wait(timeout=20) == 143, log.read_text()
    replacement, replacement_log = start("node-stop")
    assert replacement.wait(timeout=30) == 0, replacement_log.read_text()
    result = json.loads((work / "runs/selection-switch-mbpp-v1/tasks/0/result.json").read_text())
    assert result["resumed"] == {"node": "node-stop"}


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("wrong_node", [False, True])
def test_stop_ignores_reused_pid_file_and_preserves_bystander(cluster, legacy, wrong_node):
    work, start = cluster
    env = {key: value for key, value in {**os.environ, "OM_WORK": str(work)}.items()
           if key != "EXPERIMENTS_MBPP_SUITE"}
    if wrong_node:
        env.update(EXPERIMENTS_MBPP_SUITE="all", EXPERIMENTS_NODE_ID="another-node")
    unrelated = subprocess.Popen(["bash", "-c", 'exec -a "bash scripts/run_experiments.sh run" sleep 120'],
        env=env, start_new_session=True)
    try:
        logs = work / "runs/experiments/logs"
        logs.mkdir(parents=True)
        (logs / ("launcher.node-stale_.pid" if legacy else "launcher.mbpp.node-stale_.pid")).write_text(str(unrelated.pid))
        stopped, stop_log = start("node-stale", mode="stop")
        assert stopped.wait(timeout=10) == 0, stop_log.read_text()
        assert "no live node launcher" in stop_log.read_text()
        assert unrelated.poll() is None
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)
