"""CPU/flock regressions for interrupted Pair costs and runtime migration."""
import sys
from types import SimpleNamespace

import pytest

import queue_selector_pair_gpu as adapter
import selector_pair_cost_recovery as recovery

worker = adapter.worker
core, base = worker.core, worker.base


@pytest.fixture
def case(tmp_path):
    branches = {}
    for name in ("on_policy", "cached"):
        branch = tmp_path / "branches" / name
        core.atomic_json(branch / "switch.json", {
            "schema": worker.switch.rule.SCHEMA, "code_hashes": worker.switch.code_hashes(), "dataset": "gsm8k"})
        branches[name] = base.digest(branch / "switch.json")
    protocol = {"schema": worker.pair.SCHEMA, "code_hashes": worker.code_hashes(), "branch_manifests": branches}
    protocol["protocol_id"] = core.fingerprint(protocol)
    core.atomic_json(tmp_path / "pair.json", protocol)
    return SimpleNamespace(root=tmp_path, protocol=protocol)


def open_event(case, kind="main", *, name="on_policy", seed=0, arm="selection_reduced"):
    branch = case.root / "branches" / name
    out = branch / "states" / f"s{seed}-t25" / "points/view-25"
    directory = out / ("curve-parent" if kind == "parent" else arm)
    if kind == "curve":
        directory /= "curve"
    start = {"event_id": "interrupted", "state": "started", "phase": "train" if kind == "main" else "curve-eval",
             "ledger": "deployment" if kind == "main" else "reporting", "gpus": 4,
             "gpu_type": "H100", "host": "remote-stopped-worker", "time": 100.}
    base.journal(directory / "cost.jsonl", start)
    core.atomic_json(directory / "progress.json", {**start, "state": "running", "seconds": 12., "updated": 112.})
    return directory, start


def snapshot(path):
    return path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns


@pytest.mark.parametrize("kind", ["main", "curve", "parent"])
def test_stopped_pair_cost_recovers_once_without_refunding_saved_work(case, kind):
    directory, _ = open_event(case, kind)
    original = (directory / "cost.jsonl").read_bytes()
    saved = directory / "policy/checkpoint_state.json"
    core.atomic_json(saved, {"steps": 5, "budget": 17000})
    before = snapshot(saved)
    core.atomic_json(case.root / "development/s0-t25/state.json", {"state_id": "identity-only"})
    rows = recovery.recover(case.root, case.protocol, now=5000)
    assert len(rows) == 1 and rows[0]["status"] == "recovered"
    assert rows[0]["seconds"] == 72 and rows[0]["allocated_gpu_seconds"] == 288
    assert rows[0]["evidence"]["kind"] == "stale_owner_last_evidence"
    assert rows[0]["evidence"]["last_evidence_time"] == 112
    assert rows[0]["evidence"]["duration_is_estimate"] is True
    assert rows[0]["evidence"]["directly_measured"] is False
    assert "not a guaranteed upper bound" in rows[0]["evidence"]["reason"]
    assert (directory / "cost.jsonl").read_bytes().startswith(original)
    assert snapshot(saved) == before
    assert base.cost(directory)["incomplete_events"] == []
    closed = snapshot(directory / "cost.jsonl")
    assert recovery.recover(case.root, case.protocol, now=6000) == []
    assert snapshot(directory / "cost.jsonl") == closed


@pytest.mark.parametrize("kind,lock", [
    ("main", ".task.lock"), ("main", ".cost.lock"),
    ("curve", "../.task.lock"), ("curve", "step-50/.point.lock"),
    ("curve", "step-50/shard-0.lock"), ("parent", ".point.lock"),
    ("parent", "shard-3.lock"), ("parent", "../selection_reduced/.task.lock"),
])
def test_live_owner_cost_point_or_orphan_shard_is_never_closed(case, kind, lock):
    directory, _ = open_event(case, kind)
    before = snapshot(directory / "cost.jsonl")
    path = (directory / lock).resolve()
    with base.lease(path):
        lease = snapshot(path)
        rows = recovery.recover(case.root, case.protocol, now=5000)
        assert rows[0]["status"] == "active"
        assert snapshot(directory / "cost.jsonl") == before
        assert snapshot(path) == lease
    assert recovery.recover(case.root, case.protocol, now=5000)[0]["status"] == "recovered"


@pytest.mark.parametrize("lock", [".state.lock", "queue-branches/on_policy--selection_reduced.lock"])
def test_queue_and_state_publishers_are_not_interrupted(case, lock):
    directory, _ = open_event(case)
    before = snapshot(directory / "cost.jsonl")
    with base.lease(case.root / "development/s0-t25" / lock):
        rows = recovery.recover(case.root, case.protocol, now=5000)
    assert rows[0]["status"] == "active" and snapshot(directory / "cost.jsonl") == before


@pytest.mark.parametrize("kind,publication", [
    ("main", "arm"), ("curve", "branch"), ("parent", "branch"),
    ("main", "state"), ("curve", "state"), ("parent", "state"),
])
def test_published_cost_ledgers_are_immutable(case, kind, publication):
    directory, _ = open_event(case, kind)
    if publication == "arm":
        path = directory / "result.json"
    else:
        path = case.root / "development/s0-t25" / (
            "queue-branches/on_policy--selection_reduced.json" if publication == "branch" else "result.json")
    core.atomic_json(path, {"published": True})
    before, published = snapshot(directory / "cost.jsonl"), snapshot(path)
    rows = recovery.recover(case.root, case.protocol, now=5000)
    assert rows[0]["status"] == "blocked"
    assert snapshot(directory / "cost.jsonl") == before and snapshot(path) == published


def test_shared_parent_sealed_by_other_test_arm_is_preserved(case):
    directory, _ = open_event(case, "parent", seed=3, arm="selection_full")
    core.atomic_json(case.root / "test/s3-t25/queue-branches/on_policy--random_full.json", {"published": True})
    before = snapshot(directory / "cost.jsonl")
    rows = recovery.recover(case.root, case.protocol, now=5000)
    assert rows[0]["status"] == "blocked" and snapshot(directory / "cost.jsonl") == before


def test_bad_branch_runtime_receipt_does_not_block_independent_branch(case):
    invalid, _ = open_event(case)
    valid, _ = open_event(case, name="cached")
    before = snapshot(invalid / "cost.jsonl")
    core.atomic_json(case.root / "branches/on_policy/mbpp-branch-quarantine-runtime.json", {"bad": "receipt"})
    rows = recovery.recover(case.root, case.protocol, now=5000)
    assert [(row["branch"], row["status"]) for row in rows] == [("on_policy", "blocked"), ("cached", "recovered")]
    assert "frozen contract changed" in rows[0]["reason"]
    assert snapshot(invalid / "cost.jsonl") == before
    assert base.cost(valid)["incomplete_events"] == []


def test_atomic_finish_is_exact_and_recent_remote_heartbeat_is_preserved(case):
    directory, start = open_event(case)
    original = snapshot(directory / "cost.jsonl")
    assert recovery.recover(case.root, case.protocol, now=122)[0]["status"] == "skipped"
    assert snapshot(directory / "cost.jsonl") == original
    core.atomic_json(directory / "cost-events/interrupted.json", {
        **start, "state": "finished", "time": 115., "seconds": 15., "allocated_gpu_seconds": 60., "exit_code": 0})
    row = recovery.recover(case.root, case.protocol, now=122)[0]
    assert row["status"] == "recovered" and row["seconds"] == 15.
    assert row["evidence"]["kind"] == "atomic_finish_receipt"
    assert "duration_is_estimate" not in row["evidence"]


@pytest.mark.parametrize("name", ["cost.jsonl", "progress.json", ".cost.lock", ".task.lock",
                                 "cost-events/interrupted.json", "pending-costs/interrupted.json"])
def test_symlinked_cost_or_evidence_never_changes_its_target(case, name):
    directory, _ = open_event(case)
    alias = directory / name
    target = case.root / "branches/on_policy" / f"saved-{alias.name}"
    if alias.exists():
        alias.rename(target)
    else:
        target.write_text("{}")
    alias.parent.mkdir(parents=True, exist_ok=True)
    alias.symlink_to(target)
    before = snapshot(target)
    rows = recovery.recover(case.root, case.protocol, now=5000)
    assert rows[0]["status"] == "blocked" and snapshot(target) == before


def test_predecessor_guard_receipt_is_preserved_and_new_policy_is_bound(case):
    old = {**adapter.guard_receipt(case.root, case.protocol), "guard_sha256": adapter.PRE_COST_GUARD_SHA256}
    core.atomic_json(case.root / adapter.RECEIPT, old)
    before = snapshot(case.root / adapter.RECEIPT)
    adapter.validate_receipts(case.root, case.protocol)
    assert not (case.root / adapter.COST_RECEIPT).exists()
    adapter.bind_receipt(case.root, case.protocol)
    adapter.bind_recovery_receipt(case.root, case.protocol)
    assert snapshot(case.root / adapter.RECEIPT) == before
    receipt = core.read(case.root / adapter.COST_RECEIPT)
    assert receipt["curve_guard_receipt_sha256"] == base.digest(case.root / adapter.RECEIPT)
    assert set(receipt["runtime_code_hashes"]) == {
        "queue_selector_pair_gpu.py", "selector_pair_cost_recovery.py", "recover_selection_switch_cost.py",
        "_recovery_owners.py", "mbpp_storage_audit.py"}
    published = snapshot(case.root / adapter.COST_RECEIPT)
    adapter.validate_receipts(case.root, case.protocol)
    assert snapshot(case.root / adapter.COST_RECEIPT) == published


def test_symlink_loop_cannot_block_recovery_of_an_independent_branch(case):
    broken, _ = open_event(case)
    healthy, _ = open_event(case, name="cached")
    ledger = broken / "cost.jsonl"
    original = broken / "saved-cost.jsonl"
    ledger.rename(original)
    ledger.symlink_to(ledger.name)
    before = snapshot(original)
    rows = recovery.recover(case.root, case.protocol, now=5000)
    assert [(row["branch"], row["status"]) for row in rows] == [("on_policy", "blocked"), ("cached", "recovered")]
    assert snapshot(original) == before and base.cost(healthy)["incomplete_events"] == []


@pytest.mark.parametrize("kind", ["main", "curve", "parent"])
def test_reporting_overlap_is_accounted_but_invalid_main_order_is_unchanged(case, kind):
    directory, start = open_event(case, kind)
    later = {**start, "event_id": "later", "time": 200.}
    base.journal(directory / "cost.jsonl", later)
    base.journal(directory / "cost.jsonl", {
        **later, "state": "finished", "time": 215., "seconds": 15., "allocated_gpu_seconds": 60., "exit_code": 0})
    before = snapshot(directory / "cost.jsonl")
    rows = recovery.recover(case.root, case.protocol, now=5000)
    if kind == "main":
        assert rows[0]["status"] == "blocked" and "serial prefix" in rows[0]["reason"]
        assert snapshot(directory / "cost.jsonl") == before
    else:
        assert rows[0]["status"] == "recovered"
        assert (directory / "cost.jsonl").read_bytes().startswith(before[0])
        assert base.cost(directory)["complete"]
        assert base.cost(directory)["ledgers"]["reporting"]["gpu_seconds"] == 348


def test_main_closed_prefix_and_recovered_tail_pass_the_actual_pair_consumer(case):
    directory, start = open_event(case)
    tail = (directory / "cost.jsonl").read_bytes()
    previous = {**start, "event_id": "previous", "time": 1.}
    original_journal = directory / "cost.jsonl"
    original_journal.rename(directory / "saved-open.jsonl")
    base.journal(original_journal, previous)
    base.journal(original_journal, {**previous, "state": "finished", "time": 11., "seconds": 10.,
                                  "allocated_gpu_seconds": 40., "exit_code": 0})
    base.journal(original_journal, start)
    before = original_journal.read_bytes()
    assert tail in before
    assert recovery.recover(case.root, case.protocol, now=5000)[0]["status"] == "recovered"
    pairs = worker.pair.finished_events(base.read_cost_events(directory)[1])
    assert [finish["event_id"] for _, finish in pairs] == ["previous", "interrupted"]
    assert sum(finish["allocated_gpu_seconds"] for _, finish in pairs) == 328
    assert original_journal.read_bytes().startswith(before)


@pytest.mark.parametrize("receipt", [adapter.RECEIPT, adapter.COST_RECEIPT])
def test_unknown_operational_receipt_is_rejected_read_only(case, receipt):
    adapter.bind_receipt(case.root, case.protocol)
    adapter.bind_recovery_receipt(case.root, case.protocol)
    value = core.read(case.root / receipt)
    value["cost_policy"] = "unreviewed change"
    core.atomic_json(case.root / receipt, value)
    before = {path: snapshot(path) for path in case.root.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="frozen contract changed"):
        adapter.validate_receipts(case.root, case.protocol)
    assert before == {path: snapshot(path) for path in case.root.rglob("*") if path.is_file()}


def test_recovery_precedes_gpu_admission_once_and_bad_branch_stays_local(case, monkeypatch):
    invalid, _ = open_event(case)
    valid, _ = open_event(case, name="cached")
    before = snapshot(invalid / "cost.jsonl")
    core.atomic_json(case.root / "branches/on_policy/mbpp-branch-quarantine-runtime.json", {"bad": "receipt"})
    monkeypatch.setattr(sys, "argv", [adapter.__file__, "run", "--root", str(case.root)])
    calls = []
    original = recovery.recover

    def recover(root, protocol):
        calls.append("recovery")
        return original(root, protocol, now=5000)

    def admit(*args):
        assert base.cost(valid)["incomplete_events"] == []
        assert snapshot(invalid / "cost.jsonl") == before
        calls.append("admit")
        return []

    def main():
        worker.admit_node(case.root, case.protocol)
        worker.run_distributed(case.root, case.protocol, [], "run")
        worker.admit_node(case.root, case.protocol)

    monkeypatch.setattr(recovery, "recover", recover)
    monkeypatch.setattr(worker, "admit_node", admit)
    monkeypatch.setattr(worker, "run_distributed", lambda *args: calls.append("stage"))
    monkeypatch.setattr(worker, "main", main)
    adapter.run()
    assert calls == ["recovery", "admit", "stage", "admit"]
