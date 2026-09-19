"""Distributed pair scheduling must retain the complete released audit chain."""

from pathlib import Path

import pytest

import selection_gate as core
import selection_gate_gpu as base
import selector_pair_gpu as gpu
from test_selector_pair_gpu import bootstrap_predecessor
from test_selector_pair_lock_migration import (
    PRIOR_RECEIPTS,
    file_bytes,
    frozen_work,
    prior_receipts,
    released_operations_runtime,
)


HISTORICAL_RECEIPTS = (*PRIOR_RECEIPTS, "pair-lock-observation-runtime.json")


def released_lock_runtime():
    # SHA-256 of exact `git show 1aebf1d:path` bytes, not the dirty working tree.
    hashes = gpu.code_hashes()
    hashes.update({
        "src/selector_pair_gpu.py": "56c1320d453c9f0e189e27f98549fd243e0b2b22ef184d19ca6865aef1097da8",
        "scripts/run_selector_pair.sh": "805b9937bd418889fd3e4cb29ee110185788ed26892f6b595515266c86c6b4e5",
    })
    assert core.fingerprint(hashes) == gpu.PRE_PAIR_DISTRIBUTED_CODE
    return hashes


def released_chain(root, recorded, runtime, *, inherited=False):
    prior_receipts(root, recorded, released_operations_runtime() if inherited else runtime)
    core.atomic_json(root / "pair-lock-observation-runtime.json", {
        "schema": "offpolicy-selector-pair/lock-observation-runtime-v1",
        "frozen_code_hashes": recorded,
        "runtime_code_hashes": runtime,
        "operations_runtime_sha256": base.digest(root / "pair-operations-runtime.json"),
        "change": "observe an existing controller before setup/GPU admission; no duplicate worker",
        "cost_policy": "preserve targets, allocations, costs, results, decisions and all previous receipts",
    })


@pytest.mark.parametrize("history", ["frozen-at-release", "upgraded-at-release", "inherited-chain"])
def test_distributed_upgrade_preserves_manifest_and_every_previous_receipt(tmp_path, history):
    previous = released_lock_runtime()
    recorded = previous if history == "frozen-at-release" else bootstrap_predecessor()
    value = frozen_work(tmp_path, recorded)
    if history != "frozen-at-release":
        released_chain(tmp_path, recorded, previous, inherited=history == "inherited-chain")
    before = file_bytes(tmp_path)

    assert gpu.compatible_code(recorded)
    assert gpu.manifest(tmp_path) == value
    assert all((tmp_path / path).read_bytes() == raw for path, raw in before.items())
    receipt = core.read(tmp_path / "pair-distributed-runtime.json")
    assert receipt["frozen_code_hashes"] == recorded
    assert receipt["runtime_code_hashes"] == gpu.code_hashes()
    assert receipt["lock_observation_runtime_sha256"] == base.digest(tmp_path / HISTORICAL_RECEIPTS[-1])
    if history != "frozen-at-release":
        assert set(file_bytes(tmp_path)) - set(before) == {
            Path("pair-distributed-runtime.json"), Path("pair-wait-guard-runtime.json")}
        assert core.read(tmp_path / HISTORICAL_RECEIPTS[-1])["runtime_code_hashes"] == previous

    after = file_bytes(tmp_path)
    assert gpu.manifest(tmp_path) == value
    assert file_bytes(tmp_path) == after


@pytest.mark.parametrize("name", HISTORICAL_RECEIPTS)
def test_distributed_upgrade_rejects_tampered_receipts_without_overwrite(tmp_path, name):
    previous = released_lock_runtime()
    recorded = bootstrap_predecessor()
    frozen_work(tmp_path, recorded)
    released_chain(tmp_path, recorded, previous, inherited=True)
    path = tmp_path / name
    core.atomic_json(path, {**core.read(path), "tampered": True})
    before = file_bytes(tmp_path)
    with pytest.raises(ValueError, match="frozen contract changed"):
        gpu.manifest(tmp_path)
    assert file_bytes(tmp_path) == before
    assert not (tmp_path / "pair-distributed-runtime.json").exists()


def test_distributed_upgrade_rejects_unreviewed_lock_runtime(tmp_path):
    previous = released_lock_runtime()
    recorded = bootstrap_predecessor()
    frozen_work(tmp_path, recorded)
    released_chain(tmp_path, recorded, previous)
    path = tmp_path / HISTORICAL_RECEIPTS[-1]
    receipt = core.read(path)
    receipt["runtime_code_hashes"]["src/selector_pair_gpu.py"] = "unreviewed changes"
    core.atomic_json(path, receipt)
    before = file_bytes(tmp_path)
    with pytest.raises(ValueError, match="frozen contract changed"):
        gpu.manifest(tmp_path)
    assert file_bytes(tmp_path) == before


def test_distributed_migration_stays_readonly_during_status(tmp_path):
    previous = released_lock_runtime()
    recorded = bootstrap_predecessor()
    frozen_work(tmp_path, recorded)
    released_chain(tmp_path, recorded, previous, inherited=True)
    before = file_bytes(tmp_path)
    with base.lease(tmp_path / ".pair.lock"):
        gpu.read_status(tmp_path)
    after = file_bytes(tmp_path)
    after.pop(Path(".pair.lock"))
    assert after == before
    assert not (tmp_path / "pair-distributed-runtime.json").exists()
