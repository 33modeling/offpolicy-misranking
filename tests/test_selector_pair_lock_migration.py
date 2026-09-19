"""The pair-lock observation upgrade must preserve released runtime history."""

import pytest

import selection_gate as core
import selection_gate_gpu as base
import selector_pair as pair
import selector_pair_gpu as gpu
from test_selector_pair_gpu import (
    bootstrap_predecessor,
    defaults_predecessor,
    legacy_startup_receipt,
    resources_predecessor,
)


PRIOR_RECEIPTS = (
    "startup-runtime.json",
    "startup-defaults-runtime.json",
    "startup-resources-runtime.json",
    "shared-checkpoint-recovery-runtime.json",
    "budget-stop-evaluation-runtime.json",
    "pair-operations-runtime.json",
)


def released_operations_runtime():
    # Exact git-show bytes from 0caa8a0/75d1716, before the lock-observation fix.
    hashes = gpu.code_hashes()
    hashes.update({
        "src/selector_pair_gpu.py": "2ac7b5eafdcec149721940c0fd800528d6e5aeec2cfb8e2c8b80e42f2b3b6dcd",
        "scripts/run_selector_pair.sh": "6636fe98423e6e71b7a74dfe4a386bf6ef0d3def0378e4c44e90e391a8e65e90",
    })
    assert core.fingerprint(hashes) == gpu.PRE_PAIR_LOCK_OBSERVATION_CODE
    return hashes


def frozen_work(root, recorded):
    branch = root / "branches/on_policy/switch.json"
    core.atomic_json(branch, {"schema": "frozen-branch-fixture", "budget_gpu_seconds": 87120.})
    value = {
        "schema": pair.SCHEMA,
        "target_reward": .35,
        "code_hashes": recorded,
        "branch_manifests": {"on_policy": base.digest(branch)},
    }
    value["protocol_id"] = core.fingerprint(value)
    core.atomic_json(root / "pair.json", value)
    core.atomic_json(root / "request.json", {"target_reward": .35, "training_cap_gpu_seconds": 87120.})
    core.atomic_json(root / "decisions/s3-t25/decision.json", {"selector": "on_policy"})
    core.atomic_json(root / "saved/policy/checkpoint_state.json", {"completed_steps": 35})
    (root / "saved/policy/adapter_model.safetensors").write_bytes(b"preserve saved pair training")
    (root / "saved/cost.jsonl").write_text("all prior charges remain recorded\n")
    return value


def prior_receipts(root, recorded, runtime):
    """Recreate the released chain without invoking the upgraded receipt writer."""
    startup = legacy_startup_receipt(recorded, defaults_predecessor())
    core.atomic_json(root / "startup-runtime.json", startup)
    core.atomic_json(root / "startup-defaults-runtime.json", {
        **startup,
        "runtime_code_hashes": resources_predecessor(),
        "previous_receipt_sha256": base.digest(root / "startup-runtime.json"),
        "change": "no-argument launch and defaults for unfrozen setup only",
    })
    core.atomic_json(root / "startup-resources-runtime.json", {
        "schema": "offpolicy-selector-pair/resources-runtime-v1",
        "frozen_code_hashes": recorded,
        "runtime_code_hashes": runtime,
        "previous_receipt_sha256": {
            name: base.digest(root / name) for name in PRIOR_RECEIPTS[:2]
        },
        "cpu_environment": gpu.CPU_ENV,
        "change": "bound CPU thread pools and distinguish lock contention; actual elapsed costs retained",
    })
    core.atomic_json(root / "shared-checkpoint-recovery-runtime.json", {
        "schema": "offpolicy-selector-pair/shared-checkpoint-recovery-runtime-v1",
        "frozen_code_hashes": recorded,
        "runtime_code_hashes": runtime,
        "resources_runtime_sha256": base.digest(root / "startup-resources-runtime.json"),
        "change": "shared Switch runtime recovery and validated checkpoint retention only; pair design, selectors and trainer unchanged",
        "cost_policy": "preserve all protocols, receipts, policies, costs, choices and budgets; no refunds or parent restart",
    })
    core.atomic_json(root / "budget-stop-evaluation-runtime.json", {
        "schema": "offpolicy-selector-pair/budget-stop-evaluation-runtime-v1",
        "frozen_code_hashes": recorded,
        "runtime_code_hashes": runtime,
        "shared_checkpoint_recovery_runtime_sha256": base.digest(root / "shared-checkpoint-recovery-runtime.json"),
        "change": "shared Switch completed-policy evaluation resume only; pair design, selectors and trainer unchanged",
        "cost_policy": "preserve all protocols, receipts, policies, costs, choices and budgets; no refunds or retraining",
    })
    core.atomic_json(root / "pair-operations-runtime.json", {
        "schema": "offpolicy-selector-pair/operations-runtime-v1",
        "frozen_code_hashes": recorded,
        "runtime_code_hashes": runtime,
        "evaluation_runtime_sha256": base.digest(root / "budget-stop-evaluation-runtime.json"),
        "change": "read-only status, bounded branch retries with NCCL admission, exhausted-allocation guard",
        "cost_policy": "preserve selectors, trainer, target, caps, checkpoints, decisions and all prior costs",
    })


def file_bytes(root):
    return {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("already_migrated", [False, True])
def test_lock_upgrade_preserves_released_manifest_receipts_and_saved_work(tmp_path, already_migrated):
    previous = released_operations_runtime()
    recorded = bootstrap_predecessor() if already_migrated else previous
    value = frozen_work(tmp_path, recorded)
    if already_migrated:
        prior_receipts(tmp_path, recorded, previous)
    before = file_bytes(tmp_path)

    assert gpu.manifest(tmp_path) == value
    assert all((tmp_path / path).read_bytes() == contents for path, contents in before.items())
    observation = core.read(tmp_path / "pair-lock-observation-runtime.json")
    assert observation["frozen_code_hashes"] == recorded
    assert observation["runtime_code_hashes"] == gpu.code_hashes()
    if already_migrated:
        assert core.read(tmp_path / "pair-operations-runtime.json")["runtime_code_hashes"] == previous

    after = file_bytes(tmp_path)
    assert gpu.manifest(tmp_path) == value
    assert file_bytes(tmp_path) == after


@pytest.mark.parametrize("name", PRIOR_RECEIPTS)
def test_lock_upgrade_rejects_tampered_historical_receipt_without_overwrite(tmp_path, name):
    previous = released_operations_runtime()
    recorded = bootstrap_predecessor()
    frozen_work(tmp_path, recorded)
    prior_receipts(tmp_path, recorded, previous)
    path = tmp_path / name
    core.atomic_json(path, {**core.read(path), "tampered": True})
    before = file_bytes(tmp_path)
    with pytest.raises(ValueError, match="frozen contract changed"):
        gpu.manifest(tmp_path)
    assert file_bytes(tmp_path) == before
    assert not (tmp_path / "pair-lock-observation-runtime.json").exists()


def test_lock_upgrade_rejects_unreviewed_operations_runtime(tmp_path):
    previous = released_operations_runtime()
    recorded = bootstrap_predecessor()
    frozen_work(tmp_path, recorded)
    prior_receipts(tmp_path, recorded, previous)
    path = tmp_path / "pair-operations-runtime.json"
    value = core.read(path)
    value["runtime_code_hashes"]["src/selector_pair_gpu.py"] = "unreviewed-entrypoint"
    core.atomic_json(path, value)
    before = file_bytes(tmp_path)
    with pytest.raises(ValueError, match="frozen contract changed"):
        gpu.manifest(tmp_path)
    assert file_bytes(tmp_path) == before
