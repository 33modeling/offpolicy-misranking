import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

import selection_gate as core
import selection_switch as rule

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("switch_status", ROOT / "scripts/selection_switch_status.py")
status = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(status)


def prepared(root):
    core.atomic_json(root / "switch.json", {"schema": rule.SCHEMA, "code_hashes": {"old": "unchanged"}})


def test_exhausted_budget_is_distinct_from_retryable_failure(tmp_path):
    prepared(tmp_path)
    completed_prefix(tmp_path)
    core.atomic_json(point(tmp_path) / "selection_reduced/failure.json", {
        "error": "branch allocation exhausted before further GPU work: used=10; saved work preserved"})
    data = status.snapshot(tmp_path)
    task = next(t for t in data["tasks"] if t["directory"].endswith("view-25/selection_reduced"))
    assert task["status"] == "BUDGET" and task["retryable"] is False


def point(root, seed=0, step=25):
    return root / f"states/s{seed}-t{step}/points/view-{step}"


def prefix(root, seed=0, step=25):
    return root / f"prefixes/seed-{seed}/segment-{step}"


def completed_prefix(root, seed=0, step=25):
    core.atomic_json(prefix(root, seed, step).parent / f"prefix-{step}.json", {"schema": rule.SCHEMA})


def published(directory):
    core.atomic_json(directory / "result.json", {"schema": rule.SCHEMA, "complete": True})
    core.atomic_json(directory / "result.sha256.json", {"sha256": hashlib.sha256((directory / "result.json").read_bytes()).hexdigest()})


def running(directory, host, *, now, phase="prefix-train", pid=123):
    core.atomic_json(directory / "progress.json", {"state": "running", "host": host, "pid": pid,
        "phase": phase, "updated": now-2, "seconds": 1234., "timeout": 14400., "event_id": host})


def four_nodes(root, now):
    prepared(root)
    completed_prefix(root)
    running(point(root) / "selection_reduced", "node-1", now=now, phase="fresh-r-validation")
    core.atomic_json(point(root) / "selection_reduced/failure.json", {"error": "old failure", "time": now-1000})
    published(point(root) / "random_reduced")
    for seed in (1, 2, 3):
        running(prefix(root, seed), f"node-{seed+1}", now=now, pid=124+seed)
    core.atomic_json(prefix(root, 4) / "failure.json", {
        "error": "prefix-train worker failed: [1]\n[worker-log] prefix-train-0.log\nRuntimeError: actual training failure"})


def test_four_active_nodes_compact_matrix_and_real_failure(tmp_path):
    now = time.time()
    four_nodes(tmp_path, now)
    data = status.snapshot(tmp_path, now=now)
    assert data["active_nodes"] == 4 and data["stale_nodes"] == 0
    assert data["development_done"] == 1 and data["test_done"] == 0
    assert data["prefix_done"] == 1
    assert sum(data["branch_counts"].values()) == 48
    assert len([task for task in data["tasks"] if task["kind"] == "prefix"]) == 15
    output = status.render(data)
    assert "4 active" in output and "Dev 1/18" in output and "Test 0/30" in output
    assert "RuntimeError: actual training failure" in output and "old failure" not in output
    assert "CURRENT WORK" in output and "PREFIXES" in output and "CONTINUATIONS" in output
    assert "ALERTS  FAILED 1" in output
    assert "prefix 25 (fail)" in output
    print(output)


def test_stale_heartbeat_is_not_running_or_ready(tmp_path):
    prepared(tmp_path)
    now = time.time()
    running(prefix(tmp_path), "old-node", now=now-100)
    data = status.snapshot(tmp_path, now=now)
    assert data["active_nodes"] == 0 and data["stale_nodes"] == 1
    assert data["tasks"][0]["status"] == "STALE"
    assert data["tasks"][0]["retryable"]
    assert "old-node" in status.render(data)


def test_retryable_failure_never_wakes_for_a_live_peer_or_unmet_dependency(tmp_path):
    now = time.time()
    four_nodes(tmp_path, now)
    # Old failures do not override the current live peer, or a missing prefix.
    core.atomic_json(point(tmp_path, 4) / 'selection_reduced/failure.json', {'error': 'old'})
    tasks = status.snapshot(tmp_path, now=now)['tasks']
    retryable = [task for task in tasks if task['retryable']]
    assert len(retryable) == 1
    assert retryable[0]['kind'] == 'prefix' and retryable[0]['seed'] == 4


def test_held_out_controls_are_ready_before_the_gate_and_only_gate_arms_wait(tmp_path):
    prepared(tmp_path)
    completed_prefix(tmp_path, 3)
    data = status.snapshot(tmp_path)
    assert not data["gate_ready"]
    branches = {task["arm"]: task for task in data["tasks"]
                if task["kind"] == "branch" and task["seed"] == 3 and task["step"] == 25}
    assert branches["gated"]["status"] == "WAIT" and branches["gated"]["reason"] == "development gate"
    for arm in rule.TEST_ARMS:
        if arm != "gated":
            assert branches[arm]["status"] == "READY", arm
    output = status.render(data)
    assert "only the 6 GATE arms wait" in output


@pytest.mark.parametrize("seed,expected", [(0, "WAIT"), (3, "READY")])
def test_failed_diagnostic_blocks_dev_but_allows_held_out_fallback(tmp_path, seed, expected):
    prepared(tmp_path)
    completed_prefix(tmp_path, seed)
    core.atomic_json(tmp_path / "model.json", {"model_id": "published"})
    name = "measurement" if seed == 0 else "gate_measurement"
    core.atomic_json(point(tmp_path, seed) / name / "initial.json", {"status": "failed_no_retry"})
    data = status.snapshot(tmp_path)
    branches = [task for task in data["tasks"] if task["kind"] == "branch" and task["seed"] == seed and task["step"] == 25]
    assert all(task["status"] == expected for task in branches)
    assert not any(task['retryable'] for task in data['tasks'] if task['kind'] == 'diagnostic')


def test_running_diagnostic_has_owner_and_blocks_branch_barrier(tmp_path):
    prepared(tmp_path)
    completed_prefix(tmp_path)
    running(point(tmp_path) / "measurement", "diagnostic-node", now=time.time(), phase="diagnose")
    data = status.snapshot(tmp_path)
    assert data["active_nodes"] == 1
    branches = [task for task in data["tasks"] if task["kind"] == "branch" and task["seed"] == 0 and task["step"] == 25]
    assert all(task["reason"] == "diagnostic run" and task["status"] == "WAIT" for task in branches)


@pytest.mark.parametrize("receipt,expected", [(None, "SAVING"), ("wrong", "INVALID")])
def test_unfinished_or_invalid_result_publication_is_not_done(tmp_path, receipt, expected):
    prepared(tmp_path)
    directory = point(tmp_path) / "random_reduced"
    published(directory)
    path = directory / "result.sha256.json"
    if receipt is None:
        path.unlink()
    else:
        core.atomic_json(path, {"sha256": receipt})
    data = status.snapshot(tmp_path)
    assert data["development_done"] == 0
    assert next(task for task in data["tasks"] if task["directory"] == str(directory.relative_to(tmp_path)))["status"] == expected


def test_active_cost_is_not_flagged_as_interrupted_and_status_does_not_recover(tmp_path):
    prepared(tmp_path)
    directory = prefix(tmp_path)
    running(directory, "node-1", now=time.time())
    events = [{"event_id": event_id, "state": "started", "phase": "prefix-train", "ledger": "research",
               "gpus": 4, "gpu_type": "H100"} for event_id in ("old", "node-1")]
    (directory / "cost.jsonl").write_text("\n".join(json.dumps(row) for row in events) + "\n")
    core.atomic_json(directory / "cost-events/old.json", {**events[0], "state": "finished",
        "seconds": 2., "allocated_gpu_seconds": 8., "exit_code": 1})
    before = (directory / "cost.jsonl").read_bytes()
    data = status.snapshot(tmp_path)
    assert data["cost_pending"] == [{"directory": str(directory.relative_to(tmp_path)), "event_id": "old", "scope": "prefix research"}]
    assert (directory / "cost.jsonl").read_bytes() == before


def test_bad_progress_does_not_hide_other_nodes(tmp_path):
    now = time.time()
    four_nodes(tmp_path, now)
    (prefix(tmp_path, 1) / "progress.json").write_text("{invalid")
    data = status.snapshot(tmp_path, now=now)
    assert data["active_nodes"] == 3
    assert data["notices"] and "READ WARNING" in status.render(data)


def test_fresh_waiting_launcher_is_separate_from_active_and_old_logs(tmp_path):
    now = time.time()
    four_nodes(tmp_path, now)
    logs = tmp_path / "logs"
    logs.mkdir()
    for host in ("node-1", "node-5", "old-node"):
        path = logs / f"launcher.{host}_.log"
        path.write_text("[waiting] no claimable task; retry in 15s\n")
        stamp = now-100 if host == "old-node" else now
        os.utime(path, (stamp, stamp))
    data = status.snapshot(tmp_path, now=now)
    assert data["active_nodes"] == 4
    assert [node["host"] for node in data["waiting_nodes"]] == ["node-5"]


@pytest.mark.parametrize("width", [80, 100, 120])
def test_terminal_layout_has_bounded_lines(tmp_path, width):
    four_nodes(tmp_path, time.time())
    output = status.render(status.snapshot(tmp_path), width=width)
    assert all(len(line) <= width for line in output.splitlines())
    assert len(output.splitlines()) < 65


def test_status_launcher_is_read_only_and_does_not_migrate_runtime(tmp_path):
    four_nodes(tmp_path, time.time())
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    result = subprocess.run(["bash", "scripts/run_selection_switch.sh", "status", "--json"], cwd=ROOT,
        env={**os.environ, "SWITCH_ROOT": str(tmp_path), "MOPPS_ROOT": str(tmp_path.parent / "absent-mopps"),
             "SWITCH_PYTHON": sys.executable},
        capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["selection_switch"]["active_nodes"] == 4
    assert payload["mopps_comparison"]["prepared"] is False
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before
    assert not list(tmp_path.glob(".*.lock"))
    assert not list(tmp_path.glob("*-runtime.json"))


@pytest.mark.parametrize("mode", ["why", "export"])
def test_report_includes_admission_ranks_and_new_runtime_receipts_without_writes(tmp_path, mode):
    root = tmp_path / "run"
    rows = {
        "node-preflight/node-1-probe/admission.json": {"state": "failed", "failure_kind": "cuda_system_not_ready"},
        "node-preflight/node-1-probe/baseline/rank-0.json": {"rank": 0, "error": "Cuda failure 802"},
        "shutdown-runtime.json": {"schema": "shutdown-test"},
        "cache-guard-runtime.json": {"schema": "cache-guard-test"},
    }
    for name, value in rows.items():
        core.atomic_json(root / name, value)
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    result = subprocess.run(["bash", "scripts/run_selection_switch.sh", mode], cwd=ROOT,
        env={**os.environ, "SWITCH_ROOT": str(root), "OM_WORK": str(tmp_path / "work"), "SWITCH_PYTHON": sys.executable,
             "EXPERIMENTS_COMBINED": "0"},
        capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    report = Path(result.stdout.strip().removeprefix("[saved] ")).read_text()
    for name in rows:
        assert f"===== {name} =====" in report
    assert "cuda_system_not_ready" in report and "Cuda failure 802" in report
    assert {path: path.read_bytes() for path in root.rglob("*") if path.is_file()} == before


def test_watch_can_stop_without_touching_workers(tmp_path, monkeypatch, capsys):
    prepared(tmp_path)
    monkeypatch.setattr(sys, "argv", ["status", "--root", str(tmp_path), "--watch", "1"])
    monkeypatch.setattr(status.time, "sleep", lambda _: (_ for _ in ()).throw(KeyboardInterrupt()))
    assert status.main() == 130
    assert "PROGRESS" in capsys.readouterr().out


def test_unprepared_root_does_not_get_created(tmp_path):
    root = tmp_path / "not-started"
    assert status.render(status.snapshot(root)).startswith("NOT PREPARED")
    assert not root.exists()


def test_holding_node_is_visible_but_not_counted_as_training(tmp_path):
    prepared(tmp_path)
    log = tmp_path / 'logs/launcher.node-3_.log'
    log.parent.mkdir()
    log.write_text('[holding] node retained; next queue pass in 600s; no training active in this launcher\n')
    data = status.snapshot(tmp_path)
    assert data['active_nodes'] == 0
    assert data['waiting_nodes'] == [{'host': 'node-3', 'state': 'HOLD', 'reason': 'node retained between queue passes'}]
    rendered = status.render(data)
    assert 'HOLD' in rendered and 'between passes' in rendered
    old = time.time() - 90
    os.utime(log, (old, old))
    assert status.snapshot(tmp_path)['waiting_nodes'] == []


def test_training_step_uses_last_complete_log_row_without_claiming_publication(tmp_path):
    prepared(tmp_path)
    directory = prefix(tmp_path)
    running(directory, "node-1", now=time.time())
    stats = directory / "fresh_r/policy/grpo_stats.jsonl"
    stats.parent.mkdir(parents=True)
    stats.write_bytes(b'{"step": 17}\n{"step": 18}\n{"step":')
    data = status.snapshot(tmp_path)
    assert data["tasks"][0]["training_step"] == 18
    assert data["prefix_done"] == 0
    assert status.last_training_step(stats) == 18


def test_invalid_prefix_blocks_dependents_and_is_not_counted_complete(tmp_path):
    prepared(tmp_path)
    completed_prefix(tmp_path)
    (prefix(tmp_path).parent / "prefix-25.json").write_text("not json")
    data = status.snapshot(tmp_path)
    assert data["prefix_done"] == 0
    assert data["tasks"][0]["status"] == "INVALID"
    assert all(task["status"] == "WAIT" for task in data["tasks"] if task["kind"] == "branch" and task["seed"] == 0)


def test_gate_fit_failure_and_unpublished_state_are_reported(tmp_path):
    prepared(tmp_path)
    core.atomic_json(tmp_path / "gate-fit/failure.json", {"error": "development labels are invalid: seed 1 result hash changed", "time": time.time()})
    core.atomic_json(tmp_path / "states/s3-t50/failure.json", {"error": "source state differs from registered prefix", "time": time.time()})
    data = status.snapshot(tmp_path)
    assert data["gate_fit_failure"].startswith("development labels are invalid")
    output = status.render(data)
    assert "GATE  FIT FAILED: development labels are invalid" in output
    assert any("gate fit failed" in n["error"] for n in data["notices"])
    assert any("state not published/validated" in n["error"] and "s3-t50" in n["path"] for n in data["notices"])
    core.atomic_json(tmp_path / "model.json", {"model_id": "published"})
    assert status.snapshot(tmp_path)["gate_fit_failure"] == ""


def test_nodes_evaluating_reward_curves_are_running_not_gone(tmp_path):
    """Curve evaluations are metered in curve-parent and <arm>/curve/step-N, outside the branch
    directories; the node doing one must count as RUN, without inflating the branch counts."""
    now = time.time()
    root = tmp_path / "selection-switch-difficulty-v1"
    prepared(root)
    running(point(root) / "curve-parent", "node-curve", now=now, phase="curve")
    running(point(root) / "selection_full/curve/step-50", "node-curve2", now=now, phase="curve", pid=321)
    core.atomic_json(point(root) / "random_full/curve/step-75/progress.json",
                     {"state": "running", "host": "node-old", "pid": 9, "phase": "curve", "updated": now-900, "event_id": "x"})
    data = status.snapshot(root, now=now)
    phases = {task["arm"]: task for task in data["tasks"] if task["kind"] == "phase"}
    assert set(phases) == {"curve-parent", "selection_full/curve/step-50"}
    assert phases["curve-parent"]["status"] == "RUNNING" and phases["curve-parent"]["host"] == "node-curve"
    assert phases["selection_full/curve/step-50"]["seed"] == 0 and phases["selection_full/curve/step-50"]["step"] == 25
    assert data["active_nodes"] == 2 and data["branch_counts"].get("RUNNING", 0) == 0
    hosts = {node["host"]: node for node in data["nodes"]}
    assert hosts["node-curve"]["state"] == "RUN" and hosts["node-curve"]["task"] == "s0/t25 curve-parent"
    assert hosts["node-curve2"]["state"] == "RUN" and "node-old" not in hosts
    text = status.render(data, width=120)
    assert "curve-parent" in text and "node-curve2" in text


def convergence_root(root):
    core.atomic_json(root / "switch.json", {"schema": rule.SCHEMA, "gate": "convergence",
                                            "code_hashes": {"old": "unchanged"}})


def test_a_convergence_branch_without_its_curve_is_not_counted_done(tmp_path):
    """The worker keeps claiming a convergence branch until curve.json exists. Counting it
    DONE made the launcher call the root complete and release the node while the worker
    still re-ran the curve, so the operator saw DONE with nodes working on nothing."""
    now = time.time()
    convergence_root(tmp_path)
    completed_prefix(tmp_path)
    published(point(tmp_path) / "selection_reduced")
    published(point(tmp_path) / "random_reduced")
    core.atomic_json(point(tmp_path) / "random_reduced/curve.json", {"schema": rule.SCHEMA, "points": {}})
    data = status.snapshot(tmp_path, now=now)
    by_arm = {task["arm"]: task for task in data["tasks"] if task["kind"] == "branch" and task["seed"] == 0 and task["step"] == 25}
    assert by_arm["random_reduced"]["status"] == "DONE"
    assert by_arm["selection_reduced"]["status"] == "READY"
    assert by_arm["selection_reduced"]["reason"] == "result published; reward curve pending"
    assert data["development_done"] == 1
    # Its curve arrives: the branch is done and the root can be called complete.
    core.atomic_json(point(tmp_path) / "selection_reduced/curve.json", {"schema": rule.SCHEMA, "points": {}})
    assert status.snapshot(tmp_path, now=now)["development_done"] == 2


def test_a_final_gate_branch_is_done_on_its_result_alone(tmp_path):
    """Only a convergence root needs the curve; the primary experiment is unchanged."""
    now = time.time()
    prepared(tmp_path)
    completed_prefix(tmp_path)
    published(point(tmp_path) / "selection_reduced")
    assert status.snapshot(tmp_path, now=now)["development_done"] == 1
