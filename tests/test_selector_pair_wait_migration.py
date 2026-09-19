"""Bounded-wait upgrades append provenance without erasing distributed work."""

from pathlib import Path

import pytest

import selection_gate as core
import selection_gate_gpu as base
import selector_pair_gpu as gpu
from test_selector_pair_gpu import bootstrap_predecessor
from test_selector_pair_lock_migration import file_bytes, frozen_work
from test_selector_pair_queue_migration import (
    HISTORICAL_RECEIPTS,
    released_chain,
    released_lock_runtime,
)


PREVIOUS_RECEIPTS = (*HISTORICAL_RECEIPTS, "pair-distributed-runtime.json")


def released_distributed_runtime():
    # Exact SHA-256 of `git show 5fd2410:path`, independent of concurrent edits.
    hashes = gpu.code_hashes()
    hashes.update({
        "src/selection_switch_gpu.py": "48d593db8a6abd8d3363a5fa2b8421cb0528fd4f6e8068e2330d141c5f4f4122",
        "src/selector_pair_gpu.py": "5f961d578efcef717c6a0a32aaa96f210cbb145dee98c97668b0a588b0c6c002",
        "scripts/run_selector_pair.sh": "b6299c946160325e0e733500fc1d843aea0339752c778522486abc12eb6f2d66",
    })
    assert core.fingerprint(hashes) == gpu.PRE_PAIR_WAIT_GUARD_CODE
    return hashes


def distributed_chain(root, recorded, runtime, *, inherited=False):
    released_chain(root, recorded, released_lock_runtime() if inherited else runtime,
                   inherited=inherited)
    core.atomic_json(root / "pair-distributed-runtime.json", {
        "schema": "offpolicy-selector-pair/distributed-runtime-v1",
        "frozen_code_hashes": recorded,
        "runtime_code_hashes": runtime,
        "lock_observation_runtime_sha256": base.digest(root / "pair-lock-observation-runtime.json"),
        "change": "matched-state leases across nodes with shared preparation and fit/freeze barriers",
        "cost_policy": "preserve selectors, trainer, targets, caps, all saved work, costs, decisions and previous receipts; no refunds",
    })


@pytest.mark.parametrize("history", ["frozen-at-release", "upgraded-at-release", "inherited-chain"])
def test_wait_upgrade_preserves_work_caps_and_all_eight_historical_receipts(tmp_path, history):
    previous = released_distributed_runtime()
    recorded = previous if history == "frozen-at-release" else bootstrap_predecessor()
    value = frozen_work(tmp_path, recorded)
    core.atomic_json(tmp_path / "saved/result.json", {"complete": True, "reward": .42})
    core.atomic_json(tmp_path / "saved/result.sha256.json", {
        "sha256": base.digest(tmp_path / "saved/result.json")})
    if history != "frozen-at-release":
        distributed_chain(tmp_path, recorded, previous, inherited=history == "inherited-chain")
    before = file_bytes(tmp_path)

    assert gpu.compatible_code(recorded)
    assert gpu.manifest(tmp_path) == value
    assert all((tmp_path / path).read_bytes() == raw for path, raw in before.items())
    receipt = core.read(tmp_path / "pair-wait-guard-runtime.json")
    assert receipt["frozen_code_hashes"] == recorded
    assert receipt["runtime_code_hashes"] == gpu.code_hashes()
    assert receipt["distributed_runtime_sha256"] == base.digest(tmp_path / PREVIOUS_RECEIPTS[-1])
    if history != "frozen-at-release":
        assert set(file_bytes(tmp_path)) - set(before) == {
            Path("pair-wait-guard-runtime.json"), Path("shared-mbpp-quarantine-runtime.json")}
        assert core.read(tmp_path / PREVIOUS_RECEIPTS[-1])["runtime_code_hashes"] == previous

    after = file_bytes(tmp_path)
    assert gpu.manifest(tmp_path) == value
    assert file_bytes(tmp_path) == after


@pytest.mark.parametrize("name", PREVIOUS_RECEIPTS)
def test_wait_upgrade_rejects_tampered_history_without_overwrite(tmp_path, name):
    recorded = bootstrap_predecessor()
    frozen_work(tmp_path, recorded)
    distributed_chain(tmp_path, recorded, released_distributed_runtime(), inherited=True)
    path = tmp_path / name
    core.atomic_json(path, {**core.read(path), "tampered": True})
    before = file_bytes(tmp_path)
    with pytest.raises(ValueError, match="frozen contract changed"):
        gpu.manifest(tmp_path)
    assert file_bytes(tmp_path) == before
    assert not (tmp_path / "pair-wait-guard-runtime.json").exists()


def test_wait_upgrade_rejects_unreviewed_distributed_runtime(tmp_path):
    recorded = bootstrap_predecessor()
    frozen_work(tmp_path, recorded)
    distributed_chain(tmp_path, recorded, released_distributed_runtime())
    path = tmp_path / PREVIOUS_RECEIPTS[-1]
    receipt = core.read(path)
    receipt["runtime_code_hashes"]["src/selector_pair_gpu.py"] = "unreviewed queue changes"
    core.atomic_json(path, receipt)
    before = file_bytes(tmp_path)
    with pytest.raises(ValueError, match="frozen contract changed"):
        gpu.manifest(tmp_path)
    assert file_bytes(tmp_path) == before


def test_wait_migration_status_does_not_append_any_runtime_receipt(tmp_path):
    recorded = bootstrap_predecessor()
    frozen_work(tmp_path, recorded)
    distributed_chain(tmp_path, recorded, released_distributed_runtime(), inherited=True)
    with base.lease(tmp_path / ".pair.lock"):
        before = file_bytes(tmp_path)
        gpu.read_status(tmp_path)
        assert file_bytes(tmp_path) == before
    assert not (tmp_path / "pair-wait-guard-runtime.json").exists()
