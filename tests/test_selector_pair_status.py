import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import selection_gate as core
import selector_pair as pair
import selector_pair_gpu as gpu

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pair_status", ROOT / "scripts/selector_pair_status.py")
status = importlib.util.module_from_spec(spec)
spec.loader.exec_module(status)


def prepared(root):
    branches = {}
    for name in gpu.BRANCHES:
        path = root / "branches" / name / "switch.json"
        core.atomic_json(path, {"branch": name})
        branches[name] = status.digest(path)
    p = {"schema": pair.SCHEMA, "target_reward": .35, "training_cap_gpu_seconds": 87120,
         "branch_manifests": branches, "code_hashes": gpu.code_hashes()}
    p["protocol_id"] = core.fingerprint(p)
    core.atomic_json(root / "pair.json", p)
    return p


def branch(root):
    return root / "branches/on_policy/states/s0-t25/points/view-25/selection_reduced"


def published(root):
    path = branch(root)
    core.atomic_json(path / "result.json", {"complete": True, "schema": status.display.switch_status.rule.SCHEMA})
    value = status.digest(path / "result.json")
    core.atomic_json(path / "result.sha256.json", {"sha256": value})
    core.atomic_json(path / "curve.json", {"schema": status.display.switch_status.rule.SCHEMA,
                                          "result_sha256": value, "points": {"25": {"reward": .1}}})
    return path


def test_missing_root_is_readonly_and_keeps_all_planned_slots(tmp_path):
    root = tmp_path / "missing"
    data = status.snapshot(root)
    assert len(data["tasks"]) == 42
    assert {task["status"] for task in data["tasks"]} == {"WAIT"}
    assert "계획 42개 | 완료 확인 0/42 | 남음 42개" in status.render(data)
    assert "기록 미확인 42개" in status.render(data)
    assert not root.exists()


@pytest.mark.parametrize("damage", [None, "receipt", "curve", "missing_curve", "budget"])
def test_only_published_result_and_bound_curve_count_as_done(tmp_path, damage):
    prepared(tmp_path)
    path = published(tmp_path)
    if damage == "receipt":
        core.atomic_json(path / "result.sha256.json", {"sha256": "wrong"})
    elif damage == "curve":
        core.atomic_json(path / "curve.json", {})
    elif damage == "missing_curve":
        (path / "curve.json").unlink()
    elif damage == "budget":
        for name in ("result.json", "result.sha256.json", "curve.json"):
            (path / name).unlink()
        core.atomic_json(path / "pair-attempt.json", {"error": "budget exhausted"})
    data = status.snapshot(tmp_path)
    assert sum(task["status"] == "DONE" for task in data["tasks"]) == (damage is None)
    assert sum(task["role"] == "development" for task in data["tasks"]) == 18
    assert sum(task["role"] == "test" for task in data["tasks"]) == 24
    assert all(task["status"] == "WAIT" for task in data["tasks"] if task["role"] == "test")


def test_nested_live_progress_full_host_and_stale_queue(tmp_path):
    p = prepared(tmp_path)
    host = "run284441-wts-59-full-node-name"
    core.atomic_json(branch(tmp_path) / "curve/50/progress.json", {
        "host": host, "state": "running", "updated": 995, "phase": "evaluate", "seconds": 5, "timeout": 100})
    core.atomic_json(tmp_path / "queue-workers/worker.json", {
        "host": host, "state": "RUN", "updated": 10, "protocol_id": p["protocol_id"], "task": "development/s0-t25"})
    data = status.snapshot(tmp_path, now=1000)
    assert data["tasks"][0]["status"] == "RUN"
    assert len(data["nodes"]) == 1 and data["nodes"][0]["current"]
    for width in (80, 120):
        rendered = status.render(data, width=width)
        assert host in rendered
        assert all(status.display.columns(line) <= width for line in rendered.splitlines())
    stale = status.snapshot(tmp_path, now=2000)
    assert stale["tasks"][0]["status"] == "WAIT"
    assert not stale["nodes"][0]["current"]


def test_frozen_adaptive_choice_counts_only_one_branch(tmp_path, monkeypatch):
    prepared(tmp_path)
    core.atomic_json(tmp_path / "test-decisions.json", {"saved": True})
    monkeypatch.setattr(gpu, "decisions", lambda *_: {f"s{s}-t{t}": {"selector": "cached"}
                                                    for s in pair.TEST_SEEDS for t in pair.STEPS})
    data = status.snapshot(tmp_path)
    adaptive = [task for task in data["tasks"] if task["name"] == "adaptive"]
    assert len(adaptive) == 6
    assert all("adaptive-cached" in task["directory"] for task in adaptive)
    assert all(task["status"] == "READY" for task in data["tasks"])


def test_bad_test_barrier_preserves_published_development(tmp_path):
    prepared(tmp_path)
    published(tmp_path)
    core.atomic_json(tmp_path / "test-decisions.json", {"invalid": True})
    data = status.snapshot(tmp_path)
    assert data["error"]
    assert data["tasks"][0]["status"] == "DONE"
    assert all(task["status"] == "WAIT" for task in data["tasks"] if task["role"] == "test")


def test_launcher_observes_locked_run_without_writes(tmp_path):
    prepared(tmp_path)
    published(tmp_path)
    with (tmp_path / ".pair.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
        process = subprocess.run(["bash", "scripts/run_selector_pair.sh", "status", "--json"],
                                 cwd=ROOT, env={**os.environ, "PAIR_ROOT": str(tmp_path), "PAIR_PYTHON": sys.executable},
                                 text=True, capture_output=True, timeout=15)
        assert process.returncode == 0, process.stderr
        assert len(json.loads(process.stdout)["tasks"]) == 42
        assert before == {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}


@pytest.mark.parametrize("interval", ["0", "-1", "nan", "inf", "1.5"])
def test_watch_rejects_invalid_interval(tmp_path, interval):
    process = subprocess.run(["bash", "scripts/run_selector_pair.sh", "status", "--watch", interval],
                             cwd=ROOT, env={**os.environ, "PAIR_ROOT": str(tmp_path / "missing")},
                             text=True, capture_output=True, timeout=15)
    assert process.returncode == 2
    assert "watch interval must be a positive integer" in process.stdout
    assert not (tmp_path / "missing").exists()


@pytest.mark.parametrize("options,interval", [(["--watch"], "15"), (["--watch", "1", "--all"], "1")])
def test_watch_refreshes_like_mbpp_without_gpu_work(tmp_path, options, interval):
    from test_mbpp_status_watch import two_frame_sleep
    root = tmp_path / "run"
    prepared(root)
    published(root)
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in root.rglob("*") if path.is_file()}
    extra = two_frame_sleep(tmp_path)
    process = subprocess.run(["bash", "scripts/run_selector_pair.sh", "status", *options],
                             cwd=ROOT, env={**os.environ, **extra, "PAIR_ROOT": str(root), "PAIR_PYTHON": sys.executable},
                             text=True, capture_output=True, timeout=15)
    assert process.returncode == 143, process.stderr
    assert process.stdout.count("SELECTOR PAIR EXPERIMENTS") == 2
    assert process.stdout.count("계획 42개 | 완료 확인 1/42 | 남음 41개") == 2
    assert json.loads(Path(extra["SLEEP_LOG"]).read_text()) == [[interval], [interval]]
    assert before == {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in root.rglob("*") if path.is_file()}


def test_pair_uses_mbpp_renderer_with_same_sections_columns_and_korean_labels(tmp_path, monkeypatch):
    prepared(tmp_path)
    published(tmp_path)
    data = status.snapshot(tmp_path)
    shared_render = status.display.render
    calls = []
    def render(adapted, **kwargs):
        calls.append(adapted)
        return shared_render(adapted, **kwargs)
    monkeypatch.setattr(status.display, "render", render)
    output = status.render(data, width=160)
    assert len(calls) == 1
    assert output == shared_render(status.dashboard_data(data), width=160)
    assert "총 계획 42개 | 완료 확인 1개 | 남음 41개" in output
    assert "진단·학습 시간 한도: 87120 GPU-seconds" in output
    summary = next(line.split() for line in output.splitlines() if line.startswith("Experiment"))
    assert summary == ["Experiment", "계획", "DONE", "남음", "Progress", "READY", "WAIT", "RUN", "Remarks"]
    matrix = next(line.split() for line in output.splitlines() if line.startswith("Seed / Step"))
    assert matrix == ["Seed", "/", "Step", "Role", "Prefix", "On-policy", "Cached", "Adaptive", "Random", "Remarks"]
    sections = ["FULL STATUS", "CURRENT RUN", "NODE ASSIGNMENTS", "작업 없는 노드:"]
    assert [output.index(section) for section in sections] == sorted(output.index(section) for section in sections)
    assert "개발" in output and "검증" in output and "ROOT " not in output
    assert "MBPP" not in output and "Full selection" not in output


def test_same_mbpp_numbered_node_table_and_idle_list(tmp_path):
    p = prepared(tmp_path)
    core.atomic_json(branch(tmp_path) / "progress.json", {
        "host": "run-active-01", "state": "running", "updated": 995, "phase": "train", "seconds": 5, "timeout": 100})
    for host, updated in (("run-idle-02", 998), ("run-old-03", 1)):
        core.atomic_json(tmp_path / "queue-workers" / f"{host}.json", {
            "host": host, "state": "WAIT", "updated": updated, "protocol_id": p["protocol_id"]})
    data = status.snapshot(tmp_path, now=1000)
    output = status.render(data, width=160)
    header = next(line.split() for line in output.splitlines() if line.startswith("# "))
    assert header == ["#", "Node", "Experiment", "Status", "Progress", "Remarks"]
    assert "CURRENT RUN 1" in output and "NODES 2 current" in output
    assert "작업 없는 노드: 1개 (배정 대기 확인)" in output
    assert "1. run-idle-02 | WAIT | 작업 배정 대기; 확인 2초 전" in output
    assert "run-old-03" not in output and "run-old-03" in status.render(data, width=160, all_tasks=True)
    assert "branches/" not in output


def test_previous_runtime_and_receipts_are_preserved(tmp_path, monkeypatch):
    from test_selector_pair_gpu import bootstrap_predecessor
    previous = gpu.code_hashes()
    previous.update({"src/selector_pair_gpu.py": "5e2c5ca5446a609dad0f999ad134fef39d17fc6ac490cdea0b66d2479292e84f",
                     "scripts/run_selector_pair.sh": "40a7df854a6b866118164198956225468351bca81f87309fa6700a2c51549092"})
    assert core.fingerprint(previous) == gpu.PRE_PAIR_STATUS_CODE
    assert gpu.compatible_code(previous)
    p = {"schema": pair.SCHEMA, "code_hashes": bootstrap_predecessor(), "branch_manifests": {}}
    p["protocol_id"] = core.fingerprint(p)
    core.atomic_json(tmp_path / "pair.json", p)
    with monkeypatch.context() as patch:
        patch.setattr(gpu, "code_hashes", lambda: previous)
        patch.setattr(gpu, "PRE_SHARED_RUNTIME_CODES", gpu.PRE_SHARED_RUNTIME_CODES - {gpu.PRE_PAIR_STATUS_CODE})
        gpu.bind_startup_runtime(tmp_path, p["code_hashes"])
    (tmp_path / "pair-status-runtime.json").unlink()
    (tmp_path / "pair-curve-progress-runtime.json").unlink(missing_ok=True)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert gpu.manifest(tmp_path) == p
    assert gpu.manifest(tmp_path) == p
    assert all(path.read_bytes() == raw for path, raw in before.items())
    assert core.read(tmp_path / "pair-status-runtime.json")["runtime_code_hashes"] == gpu.code_hashes()
    tampered = {**previous, "scripts/run_selector_pair.sh": "unreviewed"}
    assert not gpu.compatible_code(tampered)


@pytest.mark.parametrize('offset', [-3600, 3600])
def test_curve_meter_is_visible_with_clock_skew_and_live_lease(tmp_path, offset):
    prepared(tmp_path)
    directory = branch(tmp_path) / 'curve'
    core.atomic_json(directory / 'progress.json', {'host': 'curve-peer', 'state': 'running',
        'phase': 'curve', 'updated': 10000 + offset, 'seconds': 200, 'timeout': 14400, 'event_id': 'curve-event'})
    with (directory / '.cost.lock').open('w') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        data = status.snapshot(tmp_path, now=10000)
        assert data['tasks'][0]['status'] == 'RUN'
        assert data['tasks'][0]['owner_active'] and not data['tasks'][0]['heartbeat_fresh']
        assert data['nodes'][0]['current']
        output = status.render(data, width=160)
        assert 'CURRENT RUN 1' in output and 'curve-peer' in output
        assert '실행 신호 끊김' not in output
    assert status.snapshot(tmp_path, now=10000)['tasks'][0]['status'] == 'WAIT'


def test_curve_direct_read_survives_exhausted_recursive_scan(tmp_path, monkeypatch):
    prepared(tmp_path)
    directory = branch(tmp_path) / 'curve'
    core.atomic_json(directory / 'progress.json', {'host': 'curve-peer', 'state': 'running',
        'phase': 'curve', 'updated': 995, 'seconds': 200, 'timeout': 14400})
    # Force the recursive scan's two-second deadline to expire immediately.
    ticks = iter(range(0, 100000, 3))
    monkeypatch.setattr(gpu.time, 'monotonic', lambda: next(ticks))
    data = status.snapshot(tmp_path, now=1000)
    assert data['tasks'][0]['status'] == 'RUN'
    assert data['nodes'][0]['current']
