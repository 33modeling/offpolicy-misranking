"""Real node controllers with a CPU-only leased worker; no physical GPU access."""

import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

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
    for name in ("run_experiments.sh", "run_mbpp_experiments.sh", "_mbpp_experiments.sh", "_stall_watchdog.py", "_mbpp_node_guard.py", "node_fault_state.py"):
        shutil.copy(ROOT / "scripts" / name, scripts)
    (repo / "src").mkdir()
    shutil.copy(ROOT / "src/cleanup_run_processes.py", repo / "src")
    (scripts / "run_selection_switch.sh").write_text("#!/usr/bin/env bash\nexit 0\n")
    (scripts / "check_mbpp_storage.sh").write_text(
        '#!/usr/bin/env bash\nexit "${TEST_AUDIT_EXIT:-0}"\n')
    for name in ("recover_selection_switch_cost.py", "waive_stalled_attempts.py", "split_curve_ledger.py"):
        (scripts / name).write_text("pass\n")
    (scripts / "check_mbpp_experiments.py").write_text(
        'import os, sys\nfrom pathlib import Path\n'
        'assert os.environ["CUDA_VISIBLE_DEVICES"] == ""\n'
        'print("[fake-check] input check", flush=True)\n'
        'sys.exit(1 if os.environ.get("TEST_REQUIRE_INPUTS") and not Path(os.environ["OM_WORK"], "inputs-ready").exists() else 0)\n')
    (scripts / "selection_switch_status.py").write_text(
        'import json, os, sys\nfrom pathlib import Path\n'
        'root = Path(sys.argv[sys.argv.index("--root")+1])\n'
        'done = len(list(root.glob("tasks/*/result.json"))) == 3\n'
        'print(json.dumps({"development_done": 18 if done else 0, "test_done": 30 if done else 0, '
        '"tasks": [] if done else [{"status": os.environ.get("TEST_TASK_STATUS", "READY"), '
        '"retryable": os.environ.get("TEST_TASK_STATUS") in ("FAILED", "STALE")}]}))\n')
    engine = scripts / "fake_engine.py"
    engine.write_text('''import fcntl, json, os, subprocess, sys, time
from pathlib import Path
root = Path(os.environ["SWITCH_ROOT"])
node = os.environ["EXPERIMENTS_NODE_ID"]
root.mkdir(parents=True, exist_ok=True)
def event(kind, **fields):
    row = dict(kind=kind, node=node, root=root.name, selector=os.environ["SWITCH_SELECTOR"],
               accounting=os.environ["SWITCH_ACCOUNTING"], gate=os.environ["SWITCH_GATE"],
               budget=os.environ.get("SWITCH_BUDGET_GPU_SECONDS"), **fields)
    with open(os.environ["TEST_EVENTS"], "a") as f:
        f.write(json.dumps(row) + "\\n")
event("pass")
if os.environ.get("TEST_FAULT_CHECK"):
    receipt = Path(os.environ["OM_WORK"], "runs/experiments/node-faults", node + ".json")
    rc = subprocess.run([sys.executable, "scripts/node_fault_state.py", str(receipt)]).returncode
    if rc:
        sys.exit(rc)
    event("admission")
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
        if os.environ.get("TEST_BLOCK_NODE") == node and root.name == "selection-switch-mbpp-quality-v1" and index == 0:
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
    publish_prefixes(work / "runs/selection-switch-mbpp-v1")
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
    def start(node, mode="run", suite=None, **overrides):
        log = work / f"{node}-{len(processes)}.log"
        handle = log.open("w")
        arguments = ["bash", "scripts/run_mbpp_experiments.sh", mode]
        if suite is not None:
            arguments.append(suite)
        process = subprocess.Popen(arguments, cwd=repo,
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


@pytest.mark.parametrize("mode", ["run", "restart"])
def test_blocked_storage_audit_never_enters_node_controller(cluster, mode):
    work, start = cluster
    process, log = start("node-audit-blocked", mode=mode, TEST_AUDIT_EXIT="2")
    assert process.wait(timeout=10) == 2, log.read_text()
    assert 'no controller was started or stopped' in log.read_text()
    assert '[fake-check]' not in log.read_text()
    assert events(work) == []
    assert not (work / 'runs/experiments').exists()


def test_two_nodes_and_a_replacement_resume_a_killed_owner_without_duplicate_results(cluster):
    work, start = cluster
    first, _first_log = start("node-a", TEST_BLOCK_NODE="node-a")
    wait_for(lambda: (work / "node-blocked").exists())
    second, second_log = start("node-b")
    # Peer nodes can claim another quality task while task 0 remains owned.
    wait_for(lambda: any(row["kind"] == "finished" and "quality" in row["root"] for row in events(work)))
    assert first.poll() is None
    # Kill the owner only. Its separately-sessioned child must be reclaimed by
    # the replacement on the SAME node, not mistaken for a healthy peer.
    os.kill(first.pid, signal.SIGKILL)
    first.wait(timeout=5)
    replacement, replacement_log = start("node-a")
    assert second.wait(timeout=30) == 0, second_log.read_text()
    assert replacement.wait(timeout=30) == 0, replacement_log.read_text()
    rows = [row for row in events(work) if row["kind"] == "finished"]
    assert len(rows) == len({(row["root"], row["task"]) for row in rows}) == 3
    recovered = json.loads((work / "runs/selection-switch-mbpp-quality-v1/tasks/0/result.json").read_text())
    assert recovered["resumed"] == {"node": "node-a"}
    assert recovered["node"] in ("node-b", "node-a")
    expected = {"selection-switch-mbpp-quality-v1": ("fresh_r", "matched", "convergence")}
    for row in rows:
        assert (row["selector"], row["accounting"], row["gate"]) == expected[row["root"]]
        assert row["budget"] == ("87120" if row["root"].endswith("-long-v1") else None)


def test_pending_variants_keep_node_alive_and_open_when_peer_publishes_prefixes(cluster):
    work, start = cluster
    fresh = work / "runs/selection-switch-mbpp-v1"
    for path in fresh.glob("prefixes/seed-*/prefix-*.json"):
        path.unlink()
    process, log = start("node-pending", TEST_DELAY_PREFIXES="1")
    wait_for(lambda: "shared on-policy prefixes are not ready" in log.read_text())
    assert process.poll() is None, log.read_text()
    assert "no fresh continuation is started automatically" in log.read_text()
    assert events(work) == []
    publish_prefixes(fresh)
    assert process.wait(timeout=30) == 0, log.read_text()
    assert len([row for row in events(work) if row["kind"] == "finished"]) == 3
    assert all("quality" in row["root"] for row in events(work))


def test_default_queue_never_dispatches_or_rewrites_retained_variant_work(cluster):
    work, start = cluster
    roots = [work / "runs" / name for name in ("selection-switch-mbpp-v1", "selection-switch-mbpp-difficulty-v1", "selection-switch-mbpp-long-v1")]
    for root in roots:
        policy = root / "states/s0-t25/points/view-25/random_reduced/policy"
        policy.mkdir(parents=True)
        (policy / "optimizer.pt").write_bytes(b"historical optimizer")
        (policy.parent / "cost.jsonl").write_text('{"event_id":"existing-charge"}\n')
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for root in roots for path in root.rglob("*") if path.is_file()}
    process, log = start("node-default-parity")
    assert process.wait(timeout=30) == 0, log.read_text()
    rows = events(work)
    assert {row["root"] for row in rows} == {"selection-switch-mbpp-quality-v1"}
    assert before == {path: (path.read_bytes(), path.stat().st_mtime_ns) for root in roots for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("suite", ["fresh", "difficulty", "quality", "long"])
def test_explicit_profiles_can_run_without_dispatching_other_suites(cluster, suite):
    work, start = cluster
    publish_prefixes(work / "runs/selection-switch-mbpp-v1")
    process, log = start(f"node-explicit-{suite}", suite=suite)
    assert process.wait(timeout=30) == 0, log.read_text()
    rows = events(work)
    expected = "selection-switch-mbpp-v1" if suite == "fresh" else f"selection-switch-mbpp-{suite}-v1"
    assert rows and all(row["root"] == expected for row in rows)
    assert len([row for row in rows if row["kind"] == "finished"]) == 3
    assert all(row["budget"] == ("87120" if suite == "long" else None) for row in rows)


@pytest.mark.parametrize("suite", ["quality", "long"])
def test_explicit_profile_owner_is_recognized_and_preserved_on_duplicate_run(cluster, suite):
    work, start = cluster
    publish_prefixes(work / "runs/selection-switch-mbpp-v1")
    owner, log = start(f"node-owner-{suite}", suite=suite, TEST_REQUIRE_INPUTS="1")
    wait_for(lambda: "[holding]" in log.read_text())
    duplicate, duplicate_log = start(f"node-owner-{suite}", mode="logs", suite=suite, TEST_REQUIRE_INPUTS="1")
    assert duplicate.wait(timeout=10) == 0, duplicate_log.read_text()
    assert "already running" in duplicate_log.read_text()
    assert owner.poll() is None and events(work) == []


@pytest.mark.parametrize("code", ["78", "79"])
def test_sibling_gpu_fault_stops_other_work_and_repeated_admission_failure_releases_node(cluster, code):
    work, start = cluster
    process, log = start("node-fault", TEST_FAIL_SUITE="quality", TEST_FAIL_RC=code)
    if code == "78":
        assert process.wait(timeout=30) == 78, log.read_text()
        assert "no holding/retry loop" in log.read_text()
        assert '[holding]' not in log.read_text()
    else:
        wait_for(lambda: "cooling down after a GPU fault" in log.read_text())
    assert not any("long" in row["root"] for row in events(work))


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
    process, _log = start("node-watch", EXPERIMENTS_WATCHDOG="1", TEST_REQUIRE_INPUTS="1")
    watchdog = work / "runs/experiments/logs/stall.node-watch_.log"
    wait_for(lambda: watchdog.exists() and "roots=" in watchdog.read_text())
    text = watchdog.read_text()
    assert "selection-switch-mbpp-long-v1" not in text and "selection-switch-mbpp-difficulty-v1" not in text
    assert "selection-switch-mbpp-quality-v1" in text
    assert "mopps-comparison-v1" not in text
    assert not (work / "runs/selection-switch-mbpp-long-v1").exists()
    os.killpg(process.pid, signal.SIGTERM)
    process.wait(timeout=10)


def test_peer_completion_releases_node_without_waiting_out_a_long_hold(cluster):
    work, start = cluster
    process, log = start("node-idle", TEST_REQUIRE_INPUTS="1", EXPERIMENTS_HOLD_SECONDS="40")
    wait_for(lambda: "[holding]" in log.read_text())
    for name in ("selection-switch-mbpp-quality-v1",):
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
    process, log = start('node-short-hold', TEST_FAIL_SUITE='quality', TEST_FAIL_RC=code,
                         EXPERIMENTS_HOLD_SECONDS='40', TEST_DELAY_PREFIXES='1', EXPERIMENTS_HELP_SIBLINGS='0')
    if code == '75':
        assert process.wait(timeout=10) == 75, log.read_text()
        assert '[holding]' not in log.read_text()
        return
    wait_for(lambda: '[holding]' in log.read_text())
    assert 'next pass in 60s' in log.read_text()
    assert 'next pass in 80s' not in log.read_text()
    assert process.poll() is None


def test_gpu_cooldown_retry_is_bounded_without_bypassing_the_receipt(cluster):
    work, start = cluster
    fault = work / 'runs/experiments/node-faults/node-cooldown.json'
    fault.parent.mkdir(parents=True)
    fault.write_text(json.dumps({'time': time.time(), 'strikes': 1}))
    process, log = start('node-cooldown', TEST_FAIL_SUITE='quality', TEST_FAIL_RC='79',
                         EXPERIMENTS_HOLD_SECONDS='40', EXPERIMENTS_FAULT_TTL_SECONDS='1800')
    wait_for(lambda: '[holding]' in log.read_text())
    assert 'next pass in 60s' in log.read_text()
    assert 'next pass in 80s' not in log.read_text()
    assert process.poll() is None
    assert not any(row['kind'] == 'claim' for row in events(work))


@pytest.mark.parametrize('code', ['0', '1', '75', '78', '79'])
def test_inherited_six_hundred_second_hold_cannot_return_on_reload(cluster, code):
    work, start = cluster
    fault = work / 'runs/experiments/node-faults/node-inherited.json'
    fault.parent.mkdir(parents=True)
    fault.write_text(json.dumps({'time': time.time(), 'strikes': 1}))
    process, log = start('node-inherited', TEST_FAIL_SUITE='quality', TEST_FAIL_RC=code,
                         EXPERIMENTS_HOLD_SECONDS='600', EXPERIMENTS_HOLD_POLL_SECONDS='60',
                         EXPERIMENTS_FAULT_TTL_SECONDS='1800')
    if code in ('75', '78'):
        assert process.wait(timeout=10) == int(code), log.read_text()
        assert 'no holding/retry loop' in log.read_text()
        assert '[holding]' not in log.read_text()
        assert not any(row['kind'] == 'claim' for row in events(work))
        return
    wait_for(lambda: '[holding]' in log.read_text())
    text = log.read_text()
    assert 'idle=60s poll=5s maximum=60s' in text
    assert 'next pass in 60s' in text
    assert 'next pass in 600s' not in text and 'next pass in 1200s' not in text
    assert process.poll() is None


def test_failed_quality_primary_keeps_its_retry_backoff(cluster):
    _work, start = cluster
    process, log = start('node-pending-failure', TEST_FAIL_SUITE='quality', TEST_FAIL_RC='1',
                         EXPERIMENTS_HOLD_SECONDS='20')
    wait_for(lambda: '[holding]' in log.read_text())
    assert 'next pass in 40s' in log.read_text()
    assert process.poll() is None


@pytest.mark.parametrize('status', ['FAILED', 'STALE'])
def test_retryable_work_wakes_hold_without_needing_ready_status(cluster, status):
    work, start = cluster
    root = work / 'runs/selection-switch-mbpp-quality-v1'
    root.mkdir(parents=True)
    (root / 'switch.json').write_text('{}')
    process, log = start('node-retry', TEST_FAIL_SUITE='quality', TEST_FAIL_RC='1',
                         TEST_TASK_STATUS=status, EXPERIMENTS_HOLD_SECONDS='40',
                         EXPERIMENTS_HELP_SIBLINGS='0')
    wait_for(lambda: '[pass 2]' in log.read_text(), timeout=10)
    assert 'claimable work in selection-switch-mbpp-quality-v1' in log.read_text()
    assert process.poll() is None


def test_fault_expiry_resumes_through_admission_without_full_backoff_or_data_reset(cluster):
    work, start = cluster
    fault = work / 'runs/experiments/node-faults/node-expiry.json'
    fault.parent.mkdir(parents=True)
    record = json.dumps({'time': time.time(), 'strikes': 1})
    fault.write_text(record)
    process, log = start('node-expiry', TEST_FAULT_CHECK='1', EXPERIMENTS_FAULT_TTL_SECONDS='3',
                         EXPERIMENTS_HOLD_SECONDS='600')
    assert process.wait(timeout=25) == 0, log.read_text()
    assert '[cooldown]' in log.read_text() and '[fault-expired]' in log.read_text()
    cooldown_holds = [line for line in log.read_text().splitlines()
                      if line.startswith('[hold] pass') and 'rc=79' in line]
    assert cooldown_holds and all('next pass in 60s' not in line for line in cooldown_holds)
    assert fault.read_text() == record
    rows = events(work)
    assert len([row for row in rows if row['kind'] == 'finished']) == 3
    assert next(i for i, row in enumerate(rows) if row['kind'] == 'admission') < next(
        i for i, row in enumerate(rows) if row['kind'] == 'claim')


def test_corrupt_fault_releases_node_instead_of_repeating_cooldown(cluster):
    work, start = cluster
    fault = work / 'runs/experiments/node-faults/node-corrupt.json'
    fault.parent.mkdir(parents=True)
    fault.write_text('{')
    process, log = start('node-corrupt', TEST_FAULT_CHECK='1')
    assert process.wait(timeout=15) == 78, log.read_text()
    assert 'invalid GPU-fault record' in log.read_text()
    assert '[cooldown]' not in log.read_text()
    assert fault.read_text() == '{'
    assert not any(row['kind'] in ('admission', 'claim') for row in events(work))


def test_zero_poll_interval_is_rejected_instead_of_spinning_forever(cluster):
    work, start = cluster
    process, log = start('node-invalid-poll', EXPERIMENTS_HOLD_POLL_SECONDS='0')
    assert process.wait(timeout=10) == 2, log.read_text()
    assert 'must be a positive whole number' in log.read_text()
    assert not events(work)


def test_same_code_plain_run_restarts_controller_and_resumes_checkpoint(cluster):
    work, start = cluster
    first, _log = start("node-duplicate", TEST_BLOCK_NODE="node-duplicate")
    wait_for(lambda: (work / "node-blocked").exists())
    duplicate, duplicate_log = start("node-duplicate")
    assert duplicate.wait(timeout=30) == 0, duplicate_log.read_text()
    assert "even with unchanged code" in duplicate_log.read_text()
    assert first.wait(timeout=10) == 143
    result = json.loads((work / 'runs/selection-switch-mbpp-quality-v1/tasks/0/result.json').read_text())
    assert result['resumed'] == {'node': 'node-duplicate'}


def test_stop_shows_and_saves_live_cleanup_evidence_while_child_resists_term(cluster):
    work, start = cluster
    engine = work.parent / 'repo/scripts/fake_engine.py'
    engine.write_text('import signal\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\n' + engine.read_text())
    first, _ = start('node-stop-visible', TEST_BLOCK_NODE='node-stop-visible')
    wait_for(lambda: (work / 'node-blocked').exists())
    replacement, log = start('node-stop-visible')
    assert replacement.wait(timeout=30) == 0, log.read_text()
    assert first.wait(timeout=10) == 143
    text = log.read_text()
    assert '종료 대기' in text and '아직 재시작하지 않았습니다' in text
    assert 'owned processes remaining=' in text and 'used MiB' in text
    saved = work / 'runs/experiments/logs/cleanup.mbpp.node-stop-visible_.log'
    assert '[cleanup-status]' in saved.read_text()


def test_busy_mbpp_pass_never_sweeps_other_experiment_processes(cluster):
    work, start = cluster
    # Looks like a real GPU worker, even under the same work volume. It is not
    # owned by this MBPP controller and must survive both initial and busy passes.
    unrelated = subprocess.Popen(["bash", "-c", 'exec -a "python src/selection_switch_gpu.py run" sleep 120'],
        env={**os.environ, "OUT_ROOT": str(work / "runs/selection-switch-math-v1")}, start_new_session=True)
    try:
        process, log = start("node-busy", EXPERIMENTS_CLEAN="1", TEST_FAIL_SUITE="quality", TEST_FAIL_RC="75")
        assert process.wait(timeout=10) == 75, log.read_text()
        assert '[holding]' not in log.read_text()
        assert unrelated.poll() is None
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
    result = json.loads((work / "runs/selection-switch-mbpp-quality-v1/tasks/0/result.json").read_text())
    assert result["resumed"] == {"node": "node-stop"}


def test_one_command_restart_preserves_checkpoint_and_fault_receipt(cluster):
    work, start = cluster
    fault = work / 'runs/experiments/node-faults/node-restart.json'
    fault.parent.mkdir(parents=True)
    record = json.dumps({'strikes': 1, 'time': time.time() - 100})
    fault.write_text(record)
    first, first_log = start('node-restart', TEST_BLOCK_NODE='node-restart')
    wait_for(lambda: (work / 'node-blocked').exists())
    replacement, log = start('node-restart', mode='restart')
    assert replacement.wait(timeout=30) == 0, log.read_text()
    assert first.wait(timeout=10) == 143, first_log.read_text()
    result = json.loads((work / 'runs/selection-switch-mbpp-quality-v1/tasks/0/result.json').read_text())
    assert result['resumed'] == {'node': 'node-restart'}
    assert fault.read_text() == record


@pytest.mark.parametrize("change", ["code", "legacy", "bad-storage"])
def test_plain_command_automatically_reloads_changed_code_and_resumes(cluster, change):
    work, start = cluster
    first, first_log = start('node-auto-reload', TEST_BLOCK_NODE='node-auto-reload')
    wait_for(lambda: (work / 'node-blocked').exists())
    checkpoint = work / 'runs/selection-switch-mbpp-quality-v1/tasks/0/checkpoint.json'
    before = checkpoint.read_bytes()
    if change == "legacy":
        (work / 'runs/experiments/logs/mbpp-controller.node-auto-reload_.runtime.json').unlink()
    else:
        script = work.parent / 'repo/scripts/run_experiments.sh'
        script.write_text(script.read_text() + '\n# Simulated installed code update.\n')
    replacement, log = start('node-auto-reload', TEST_AUDIT_EXIT='2' if change == 'bad-storage' else '0')
    assert replacement.wait(timeout=30) == (2 if change == 'bad-storage' else 0), log.read_text()
    assert checkpoint.read_bytes() == before
    if change == 'bad-storage':
        assert first.poll() is None
        assert '[stop]' not in log.read_text()
    else:
        assert first.wait(timeout=10) == 143, first_log.read_text()
        assert '[reload]' in log.read_text()
        result = json.loads(checkpoint.with_name('result.json').read_text())
        assert result['resumed'] == {'node': 'node-auto-reload'}


def test_plain_command_pulls_before_following_live_owner_and_reenters_updated_wrapper(cluster):
    work, start = cluster
    git = work.parent / 'bin/git'
    git.write_text(f'''#!{sys.executable}
import os, sys
from pathlib import Path
work = Path(os.environ["OM_WORK"])
marker = work / "pulled"
if sys.argv[1] == "rev-parse":
    print("b" * 40 if marker.exists() else "a" * 40)
elif sys.argv[1] == "pull":
    with (work / "pull-calls").open("a") as f:
        f.write("pull\\n")
    if not marker.exists():
        script = Path("scripts/run_experiments.sh")
        script.write_text(script.read_text() + "\\n# Installed by simulated pull.\\n")
        marker.write_text("updated")
''')
    git.chmod(0o755)
    first, first_log = start('node-pull', TEST_BLOCK_NODE='node-pull')
    wait_for(lambda: (work / 'node-blocked').exists())
    replacement, log = start('node-pull', EXPERIMENTS_PULL='1')
    assert replacement.wait(timeout=30) == 0, log.read_text()
    assert first.wait(timeout=10) == 143, first_log.read_text()
    assert '[pull]' in log.read_text() and '[reload]' in log.read_text()
    assert (work / 'pull-calls').read_text().splitlines() == ['pull']
    result = json.loads((work / 'runs/selection-switch-mbpp-quality-v1/tasks/0/result.json').read_text())
    assert result['resumed'] == {'node': 'node-pull'}


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
