"""pair-03: one SR-GC measurement is one work item for the meter's owner."""
import fcntl
import importlib.util
from pathlib import Path

import selection_gate as core
import selector_pair as pair
import selector_pair_gpu as gpu

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pair_status_fix_pair", ROOT / "scripts/selector_pair_status.py")
status = importlib.util.module_from_spec(spec)
spec.loader.exec_module(status)
display = status.display
NOW = 100000.


def prepared(root, *, srgc=True):
    branches = {}
    for name in gpu.BRANCHES:
        path = root / "branches" / name / "switch.json"
        core.atomic_json(path, {"branch": name})
        branches[name] = status.digest(path)
    p = {"schema": pair.SCHEMA, "target_reward": .35, "training_cap_gpu_seconds": 87120,
         "branch_manifests": branches, "code_hashes": gpu.code_hashes()}
    p["protocol_id"] = core.fingerprint(p)
    core.atomic_json(root / "pair.json", p)
    import selector_pair_parallel as parallel
    core.atomic_json(root / parallel.RECEIPT, parallel.receipt_value(root.resolve(), p))
    if srgc:
        import selector_pair_srgc
        selector_pair_srgc.activate(root, p)
    return p


def meter(directory, host, pid, event, phase, updated):
    # Exactly the fields base.meter writes; SR-GC freeze() has no worker_id.
    core.atomic_json(directory / "progress.json", {
        "event_id": event, "phase": phase, "ledger": "deployment", "gpus": [0, 1, 2, 3],
        "gpu_type": "NVIDIA H100 80GB HBM3", "host": host, "state": "running", "pid": pid,
        "updated": updated, "seconds": 120., "timeout": 20000.})


def branch_worker(root, p, worker, host, pid, task, directory, phase):
    meter(directory, host, pid, "e-" + worker, phase, NOW - 3)
    core.atomic_json(root / "queue-workers" / f"{worker}.json", {
        "worker": worker, "host": host, "pid": pid, "protocol_id": p["protocol_id"],
        "stage": task.split("/")[0], "state": "RUN", "task": task, "updated": NOW - 3})


def views(data):
    suite = status.dashboard_data(data)["suites"][0]
    work = display.current_work(suite)
    nodes = [node for node in display.node_assignments(status.dashboard_data(data)) if node["current"]]
    return suite, work, nodes


def adaptive(data, seed, step):
    return next(t for t in data["tasks"] if t["name"] == "adaptive" and (t["seed"], t["step"]) == (seed, step))


def node_rows(output):
    lines = output.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("#   Node"))
    end = next(i for i, line in enumerate(lines) if i > start and line.startswith("Progress는"))
    return [line for line in lines[start + 1:end] if line[:1].isdigit()]


def test_single_srgc_measurement_is_one_work_item_with_real_owner(tmp_path):
    prepared(tmp_path)
    meter(tmp_path / "sr-gc/s3-t50", "run284168-wts-11", 4242, "srgc-event", "sr-gc-candidate-a", NOW - 5)
    data = status.snapshot(tmp_path, now=NOW)
    task = adaptive(data, 3, 50)
    assert task["status"] == "RUN" and "SR-GC" in task["reason"]
    assert (task.get("host"), task.get("pid"), task.get("event_id"), task["directory"]) == (
        "run284168-wts-11", 4242, "srgc-event", "sr-gc/s3-t50")
    assert task.get("heartbeat_fresh") is True and task.get("owner_active") is False
    suite, work, nodes = views(data)
    assert len(work) == 1 and work[0][1] is False
    assert display.counts(suite)["states"]["RUN"] == 1
    assert [(node["host"], node.get("work_id")) for node in nodes] == [("run284168-wts-11", None)]
    output = status.render(data, width=160)
    assert "현재 실행: 분기 RUN 1개 | 공통 단계 RUN 0개 | 실행 작업 1개" in output
    assert "CURRENT RUN 1" in output and "NODES 1 current" in output
    assert "unknown-owner" not in output and "work-" not in output
    rows = node_rows(output)
    assert len(rows) == 1
    assert "run284168-wts-11" in rows[0] and "seed 3 / step 50 / Adaptive" in rows[0]


def test_srgc_measurement_beside_branch_workers_is_counted_once(tmp_path):
    p = prepared(tmp_path)
    states = tmp_path / "branches/{}/states/s{}-t{}/points/view-{}/{}"
    branch_worker(tmp_path, p, "w1", "run284441-wts-59", 111, "development/s1-t50/cached/selection_reduced",
                  Path(str(states).format("cached", 1, 50, 50, "selection_reduced")), "train")
    branch_worker(tmp_path, p, "w2", "run284168-wts-3", 222, "test/s3-t25/on_policy/selection_full",
                  Path(str(states).format("on_policy", 3, 25, 25, "selection_full")), "fresh-r-candidate")
    branch_worker(tmp_path, p, "w3", "run284168-wts-7", 333, "test/s4-t50/on_policy/random_full",
                  Path(str(states).format("on_policy", 4, 50, 50, "random_full")), "train")
    core.atomic_json(tmp_path / "queue-workers/w4.json", {
        "worker": "w4", "host": "run284168-wts-9", "pid": 444, "protocol_id": p["protocol_id"],
        "stage": "development", "state": "WAIT", "task": "development pass yielded; no task owned",
        "updated": NOW - 5})
    meter(tmp_path / "sr-gc/s3-t50", "run284168-wts-11", 555, "srgc-event", "sr-gc-candidate-a", NOW - 2)
    data = status.snapshot(tmp_path, now=NOW)
    running = [t for t in data["tasks"] if t["status"] == "RUN"]
    assert {(t["name"], t.get("host")) for t in running} == {
        ("cached", "run284441-wts-59"), ("on_policy", "run284168-wts-3"),
        ("random", "run284168-wts-7"), ("adaptive", "run284168-wts-11")}
    suite, work, nodes = views(data)
    assert len(work) == 4 and not any(shared for _, shared in work)
    assert display.counts(suite)["states"]["RUN"] == 4
    assert sorted(node["host"] for node in nodes) == sorted(
        ["run284441-wts-59", "run284168-wts-3", "run284168-wts-7", "run284168-wts-9", "run284168-wts-11"])
    output = status.render(data, width=160)
    assert "남음 42개 = RUN 4개" in output
    assert "현재 실행: 분기 RUN 4개 | 공통 단계 RUN 0개 | 실행 작업 4개" in output
    assert "CURRENT RUN 4" in output and "WORKERS 5 current" in output
    assert "unknown-owner" not in output and "WORK ITEMS" not in output
    rows = node_rows(output)
    assert len(rows) == 5
    assert sum("/ Adaptive" in row for row in rows) == 1
    assert any("run284168-wts-11" in row and "seed 3 / step 50 / Adaptive" in row for row in rows)


def test_owner_held_srgc_measurement_without_heartbeat_is_one_work_item(tmp_path):
    prepared(tmp_path)
    directory = tmp_path / "sr-gc/s3-t25"
    meter(directory, "srgc-peer", 77, "srgc-owned", "sr-gc-validation-b", NOW - 600)
    with (directory / ".cost.lock").open("w") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        data = status.snapshot(tmp_path, now=NOW)
        suite, work, nodes = views(data)
        output = status.render(data, width=160)
    task = adaptive(data, 3, 25)
    assert task["status"] == "RUN" and task["directory"] == "sr-gc/s3-t25"
    assert (task.get("host"), task.get("owner_active"), task.get("heartbeat_fresh")) == ("srgc-peer", True, False)
    assert len(work) == 1 and display.counts(suite)["states"]["RUN"] == 1
    assert [node["host"] for node in nodes] == ["srgc-peer"]
    assert "CURRENT RUN 1" in output and "공통 단계 RUN 0개" in output and "unknown-owner" not in output


def test_two_srgc_measurements_on_one_hostname_keep_separate_identities(tmp_path):
    prepared(tmp_path)
    meter(tmp_path / "sr-gc/s3-t25", "shared-name", 1, "a", "sr-gc-candidate-a", NOW - 2)
    meter(tmp_path / "sr-gc/s4-t100", "shared-name", 2, "b", "sr-gc-candidate-b", NOW - 2)
    data = status.snapshot(tmp_path, now=NOW)
    measured = [adaptive(data, 3, 25), adaptive(data, 4, 100)]
    assert all(t["status"] == "RUN" and t.get("host") == "shared-name" for t in measured)
    assert len({t.get("worker_id") for t in measured}) == 2 and all(t.get("worker_id") for t in measured)
    suite, work, nodes = views(data)
    assert len(work) == 2 and not any(shared for _, shared in work)
    assert len(nodes) == 2 and {node["host"] for node in nodes} == {"shared-name"}
    output = status.render(data, width=160)
    assert "현재 실행: 분기 RUN 2개 | 공통 단계 RUN 0개" in output
    assert "CURRENT RUN 2" in output and "unknown-owner" not in output


def test_legacy_srgc_meter_without_identity_still_counts_once(tmp_path):
    # Same meter shape as test_srgc_does_not_wait_for_development_target_and_shows_measurement.
    prepared(tmp_path)
    core.atomic_json(tmp_path / "sr-gc/s3-t25/progress.json", {
        "state": "running", "phase": "sr-gc-candidate-a", "host": "node-srgc", "updated": NOW})
    data = status.snapshot(tmp_path, now=NOW)
    assert adaptive(data, 3, 25).get("host") == "node-srgc"
    suite, work, nodes = views(data)
    assert len(work) == 1 and [node["host"] for node in nodes] == ["node-srgc"]
    output = status.render(data, width=160)
    assert "분기 RUN 1개 | 공통 단계 RUN 0개" in output and "CURRENT RUN 1" in output
    assert "unknown-owner" not in output


def test_stale_srgc_meter_leaves_adaptive_waiting_without_owner(tmp_path):
    prepared(tmp_path)
    meter(tmp_path / "sr-gc/s3-t50", "gone-node", 9, "old", "sr-gc-candidate-a", NOW - 600)
    data = status.snapshot(tmp_path, now=NOW)
    task = adaptive(data, 3, 50)
    assert task["status"] == "WAIT" and task["directory"] == "" and task.get("host") is None
    assert "CURRENT RUN 0" in status.render(data, width=160)
