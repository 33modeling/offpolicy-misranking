"""Reviewed MBPP quarantine upgrades preserve frozen experiments and receipts."""

from pathlib import Path

import pytest

import mopps_comparison_gpu as mopps
import selection_gate as core
import selection_gate_gpu as base
import selection_switch as rule
import selection_switch_gpu as switch
import selector_pair_gpu as pair
from test_checkpoint_retention_runtime import files, saved_work
from test_mopps_comparison_gpu import code_compat_predecessor, source
from test_selection_switch_gpu import initial_predecessor
from test_selector_pair_gpu import bootstrap_predecessor
from test_selector_pair_lock_migration import frozen_work
from test_selector_pair_wait_migration import PREVIOUS_RECEIPTS, distributed_chain


SWITCH_RECEIPTS = tuple(name + "-runtime.json" for name in (
    "kv-cache", "cost", "prefix-resume", "worker-logs", "code-compat", "shutdown",
    "cache-guard", "test-parallel", "fit-resilience", "variant-root", "dataset",
    "selector", "curve", "quality", "scoring-label", "curve-ledger", "node-id",
    "curve-wait", "publication", "allocation-guard", "resume-preservation",
    "saved-policy-recovery", "checkpoint-retention", "budget-stop-evaluation"))
PAIR_RECEIPTS = (*PREVIOUS_RECEIPTS, "pair-wait-guard-runtime.json")
SWITCH_9BE50A8 = "48d593db8a6abd8d3363a5fa2b8421cb0528fd4f6e8068e2330d141c5f4f4122"


def previous_switch():
    hashes = switch.code_hashes()
    hashes["src/selection_switch_gpu.py"] = SWITCH_9BE50A8
    assert core.fingerprint(hashes) == switch.PRE_MBPP_BRANCH_QUARANTINE_CODE
    return hashes


def previous_pair():
    hashes = {**pair.code_hashes(), **previous_switch()}
    hashes.update({
        "src/selector_pair_gpu.py": "bebe89f565d1d0f7560ef69327d4ae3e36591e9d5f80028f8000351383b6fe1b",
        "scripts/run_selector_pair.sh": "40a7df854a6b866118164198956225468351bca81f87309fa6700a2c51549092",
    })
    assert core.fingerprint(hashes) == pair.PRE_SHARED_MBPP_QUARANTINE_CODE
    return hashes


def previous_mopps():
    hashes = {**mopps.hashes(), **previous_switch()}
    hashes["src/mopps_comparison_gpu.py"] = "5f7722a3e448ed3a67d912d5667418f2448444a2266ec5885b60b736060d6bc3"
    assert core.fingerprint(hashes) == mopps.PRE_SHARED_MBPP_QUARANTINE_CODE
    return hashes


def switch_history(root, monkeypatch, migrated=True, dataset="mbpp"):
    previous = previous_switch()
    frozen = {"schema": rule.SCHEMA, "dataset": dataset,
              "code_hashes": initial_predecessor() if migrated else previous,
              "budget_gpu_seconds": 28380., "selector": "fresh_r", "accounting": "matched"}
    core.atomic_json(root / "switch.json", frozen)
    if migrated:
        with monkeypatch.context() as patch:
            patch.setattr(switch, "code_hashes", lambda: previous)
            switch.manifest(root)
        assert all((root / name).exists() for name in SWITCH_RECEIPTS)
    saved_work(root)
    return frozen


@pytest.mark.parametrize("migrated", [False, True])
@pytest.mark.parametrize("dataset", ["mbpp", "math500"])
def test_switch_quarantine_upgrade_preserves_all_prior_receipts_and_paid_work(tmp_path, monkeypatch, migrated, dataset):
    frozen = switch_history(tmp_path, monkeypatch, migrated, dataset)
    before = files(tmp_path)
    assert switch.check_code(tmp_path) == frozen
    assert files(tmp_path) == before
    assert switch.manifest(tmp_path) == frozen
    assert all(path.read_bytes() == raw for path, raw in before.items())
    receipt = core.read(tmp_path / "mbpp-branch-quarantine-runtime.json")
    assert receipt["runtime_code_hashes"] == switch.code_hashes()
    assert receipt["storage_audit_sha256"] == base.digest(base.ROOT / "scripts/mbpp_storage_audit.py")
    assert receipt["budget_stop_evaluation_runtime_sha256"] == base.digest(tmp_path / SWITCH_RECEIPTS[-1])
    if migrated:
        assert set(files(tmp_path)) - set(before) == {tmp_path / "mbpp-branch-quarantine-runtime.json"}
    after = files(tmp_path)
    assert switch.manifest(tmp_path) == frozen
    assert files(tmp_path) == after


@pytest.mark.parametrize("name", SWITCH_RECEIPTS)
def test_switch_quarantine_rejects_each_tampered_historical_receipt(tmp_path, monkeypatch, name):
    switch_history(tmp_path, monkeypatch)
    path = tmp_path / name
    core.atomic_json(path, {**core.read(path), "tampered": True})
    before = files(tmp_path)
    with pytest.raises(ValueError, match="frozen contract changed"):
        switch.manifest(tmp_path)
    assert files(tmp_path) == before


def test_current_frozen_switch_binds_helper_and_rejects_helper_change(tmp_path, monkeypatch):
    frozen = {"schema": rule.SCHEMA, "dataset": "mbpp", "code_hashes": switch.code_hashes()}
    core.atomic_json(tmp_path / "switch.json", frozen)
    switch.manifest(tmp_path)
    receipt = core.read(tmp_path / "mbpp-branch-quarantine-runtime.json")
    assert receipt["budget_stop_evaluation_runtime_sha256"] is None
    original = base.digest
    monkeypatch.setattr(base, "digest", lambda path: "unreviewed helper" if
                        Path(path) == base.ROOT / "scripts/mbpp_storage_audit.py" else original(path))
    before = files(tmp_path)
    with pytest.raises(ValueError, match="frozen contract changed"):
        switch.manifest(tmp_path)
    assert files(tmp_path) == before


@pytest.mark.parametrize("where", ["current", "recorded"])
@pytest.mark.parametrize("name", ["src/train_policy_grpo.py", "src/selection_switch_score.py", "src/data.py"])
def test_quarantine_does_not_relax_scientific_hashes(tmp_path, monkeypatch, where, name):
    previous, current = previous_switch(), switch.code_hashes()
    (previous if where == "recorded" else current)[name] = "unreviewed"
    core.atomic_json(tmp_path / "switch.json", {"schema": rule.SCHEMA, "code_hashes": previous})
    monkeypatch.setattr(switch, "code_hashes", lambda: current)
    before = files(tmp_path)
    with pytest.raises(ValueError, match="scientific code changed"):
        switch.manifest(tmp_path)
    assert files(tmp_path) == before


def pair_history(root, monkeypatch, migrated=True):
    previous = previous_pair()
    frozen = frozen_work(root, bootstrap_predecessor() if migrated else previous)
    if migrated:
        distributed_chain(root, frozen["code_hashes"], previous, inherited=True)
        with monkeypatch.context() as patch:
            patch.setattr(pair, "code_hashes", lambda: previous)
            patch.setattr(switch, "code_hashes", lambda: {name: previous[name] for name in switch.CODE})
            pair.manifest(root)
        assert all((root / name).exists() for name in PAIR_RECEIPTS)
    return frozen


@pytest.mark.parametrize("migrated", [False, True])
def test_pair_quarantine_compatibility_preserves_all_nine_receipts(tmp_path, monkeypatch, migrated):
    frozen = pair_history(tmp_path, monkeypatch, migrated)
    before = files(tmp_path)
    assert pair.manifest(tmp_path) == frozen
    assert all(path.read_bytes() == raw for path, raw in before.items())
    receipt = core.read(tmp_path / "shared-mbpp-quarantine-runtime.json")
    assert receipt["runtime_code_hashes"] == pair.code_hashes()
    assert receipt["wait_guard_runtime_sha256"] == base.digest(tmp_path / PAIR_RECEIPTS[-1])
    if migrated:
        assert set(files(tmp_path)) - set(before) == {
            tmp_path / "shared-mbpp-quarantine-runtime.json", tmp_path / "pair-status-runtime.json",
            tmp_path / "pair-curve-progress-runtime.json"}
        status_receipt = core.read(tmp_path / "pair-status-runtime.json")
        assert status_receipt["runtime_code_hashes"] == pair.code_hashes()
    after = files(tmp_path)
    assert pair.manifest(tmp_path) == frozen
    assert files(tmp_path) == after


@pytest.mark.parametrize("name", PAIR_RECEIPTS)
def test_pair_quarantine_rejects_each_tampered_historical_receipt(tmp_path, monkeypatch, name):
    pair_history(tmp_path, monkeypatch)
    path = tmp_path / name
    core.atomic_json(path, {**core.read(path), "tampered": True})
    before = files(tmp_path)
    with pytest.raises(ValueError, match="frozen contract changed"):
        pair.manifest(tmp_path)
    assert files(tmp_path) == before


def test_pair_status_stays_read_only_after_shared_upgrade(tmp_path, monkeypatch):
    pair_history(tmp_path, monkeypatch)
    before = files(tmp_path)
    pair.read_status(tmp_path)
    assert files(tmp_path) == before


@pytest.mark.parametrize("migrated", [False, True])
def test_mopps_quarantine_compatibility_preserves_all_frozen_work(tmp_path, monkeypatch, migrated):
    parent, parent_manifest = source(tmp_path)
    parent_manifest["code_hashes"] = previous_switch()
    core.atomic_json(parent / "switch.json", parent_manifest)
    root = tmp_path / "comparison"
    frozen = mopps.prepare(root, parent)
    previous = previous_mopps()
    frozen["code_hashes"] = code_compat_predecessor() if migrated else previous
    core.atomic_json(root / "mopps.json", frozen)
    if migrated:
        with monkeypatch.context() as patch:
            patch.setattr(mopps, "hashes", lambda: previous)
            mopps.protocol(root)
    saved_work(root)
    before = files(tmp_path)
    assert mopps.protocol(root) == frozen
    assert all(path.read_bytes() == raw for path, raw in before.items())
    receipt = core.read(root / "shared-mbpp-quarantine-runtime.json")
    assert receipt["runtime_code_hashes"] == mopps.hashes()
    if migrated:
        assert set(files(tmp_path)) - set(before) == {root / "shared-mbpp-quarantine-runtime.json"}
    after = files(tmp_path)
    assert mopps.protocol(root) == frozen
    assert files(tmp_path) == after
