"""Reviewed retention-only upgrade preserves every frozen protocol and receipt."""

import pytest

import mopps_comparison_gpu as mopps_run
import net_gain_gate_gpu as net_run
import selection_gate as core
import selection_gate_gpu as base
import selection_switch as rule
import selection_switch_gpu as switch
from test_mopps_comparison_gpu import code_compat_predecessor, source
from test_net_gain_gate_gpu import protocol as net_protocol
from test_selection_switch_gpu import initial_predecessor

OLD_NET_RUN = "1ee7c1fa81d32a4a61627c12ee4956c8c0e2436f62a10907919291ce9140f423"
OLD_SWITCH_RUN = "955727844dd4a2d824ebdf5972a44fee8495c40ab51575594d9a0f1f7707a382"
OLD_MOPPS_RUN = "2421a7c546e9900146fd60b7afef6c76fbd1985469917a30cbc4ee68e44d0532"


def old_switch():
    hashes = switch.code_hashes()
    hashes.update({"src/train_policy_grpo.py": net_run.PRE_RETENTION_TRAINER,
                   "src/net_gain_gate_gpu.py": OLD_NET_RUN, "src/selection_switch_gpu.py": OLD_SWITCH_RUN})
    assert core.fingerprint(hashes) == switch.PRE_CHECKPOINT_RETENTION_CODE
    return hashes


def old_mopps():
    hashes = mopps_run.hashes()
    hashes.update(old_switch())
    hashes["src/mopps_comparison_gpu.py"] = OLD_MOPPS_RUN
    assert core.fingerprint(hashes) == mopps_run.PRE_CHECKPOINT_RETENTION_CODE
    return hashes


def files(root):
    return {path: path.read_bytes() for path in root.rglob("*") if path.is_file() and not path.name.endswith(".lock")}


def saved_work(root):
    directory = root / "states/s3-t25/points/view-25/random_full"
    core.atomic_json(directory / "result.json", {"complete": True, "completed_steps": 40})
    core.atomic_json(directory / "policy/checkpoint-000035/checkpoint_state.json", {"completed_steps": 35})
    (directory / "policy/checkpoint-000035/adapter_model.safetensors").write_bytes(b"saved adapter")
    (directory / "cost.jsonl").write_text("preserve every prior charge\n")


@pytest.mark.parametrize("dataset", ["mbpp", "math500"])
@pytest.mark.parametrize("migrated", [False, True])
def test_switch_retention_upgrade_keeps_protocol_results_and_all_previous_receipts(tmp_path, monkeypatch, dataset, migrated):
    previous = old_switch()
    frozen = {"schema": rule.SCHEMA, "dataset": dataset, "code_hashes": initial_predecessor() if migrated else previous}
    core.atomic_json(tmp_path / "switch.json", frozen)
    if migrated:
        with monkeypatch.context() as patch:
            patch.setattr(switch, "code_hashes", lambda: previous)
            switch.manifest(tmp_path)
        assert core.read(tmp_path / "saved-policy-recovery-runtime.json")["runtime_code_hashes"] == previous
        assert not (tmp_path / "checkpoint-retention-runtime.json").exists()
    saved_work(tmp_path)
    before = files(tmp_path)
    assert switch.check_code(tmp_path) == frozen
    assert files(tmp_path) == before
    for _ in range(2):
        assert switch.manifest(tmp_path) == frozen
        assert {path: path.read_bytes() for path in before} == before
    receipt = core.read(tmp_path / "checkpoint-retention-runtime.json")
    assert receipt["runtime_code_hashes"] == switch.code_hashes()
    assert receipt["saved_policy_recovery_runtime_sha256"] == base.digest(tmp_path / "saved-policy-recovery-runtime.json")
    receipt["cost_policy"] = "refund old training"
    core.atomic_json(tmp_path / "checkpoint-retention-runtime.json", receipt)
    with pytest.raises(ValueError, match="frozen contract changed"):
        switch.manifest(tmp_path)


@pytest.mark.parametrize("migrated", [False, True])
def test_mopps_retention_upgrade_preserves_parent_and_existing_receipts(tmp_path, monkeypatch, migrated):
    parent, _ = source(tmp_path)
    root = tmp_path / "comparison"
    frozen = mopps_run.prepare(root, parent)
    previous = old_mopps()
    frozen["code_hashes"] = code_compat_predecessor() if migrated else previous
    core.atomic_json(root / "mopps.json", frozen)
    if migrated:
        with monkeypatch.context() as patch:
            patch.setattr(mopps_run, "hashes", lambda: previous)
            mopps_run.protocol(root)
        assert core.read(root / "shared-recovery-runtime.json")["runtime_code_hashes"] == previous
        assert not (root / "checkpoint-retention-runtime.json").exists()
    saved_work(root)
    before = files(tmp_path)
    for _ in range(2):
        assert mopps_run.protocol(root) == frozen
        assert {path: path.read_bytes() for path in before} == before
    receipt = core.read(root / "checkpoint-retention-runtime.json")
    assert receipt["shared_recovery_runtime_sha256"] == base.digest(root / "shared-recovery-runtime.json")
    receipt["cost_policy"] = "discard prior policies"
    core.atomic_json(root / "checkpoint-retention-runtime.json", receipt)
    with pytest.raises(ValueError, match="frozen contract changed"):
        mopps_run.protocol(root)


def test_net_gain_retention_upgrade_preserves_frozen_protocol_and_saved_work(tmp_path):
    frozen = net_protocol()
    old = {name: base.digest(base.ROOT / name) for name in net_run.CODE_FILES}
    old.update({"src/train_policy_grpo.py": net_run.PRE_RETENTION_TRAINER, "src/net_gain_gate_gpu.py": OLD_NET_RUN})
    assert core.fingerprint(old) == net_run.PRE_CHECKPOINT_RETENTION_CODE
    frozen["code_hashes"] = old
    core.atomic_json(tmp_path / "net_protocol.json", frozen)
    saved_work(tmp_path)
    before = files(tmp_path)
    for _ in range(2):
        assert net_run.protocol(tmp_path) == frozen
        assert {path: path.read_bytes() for path in before} == before
    receipt = core.read(tmp_path / "checkpoint-retention-runtime.json")
    assert receipt["runtime_code_hashes"]["src/train_policy_grpo.py"] == net_run.RETENTION_TRAINER


@pytest.mark.parametrize("where", ["recorded", "current"])
def test_retention_pins_reject_unknown_trainer_hash_before_any_switch_writes(tmp_path, monkeypatch, where):
    previous, current = old_switch(), switch.code_hashes()
    (previous if where == "recorded" else current)["src/train_policy_grpo.py"] = "unreviewed"
    core.atomic_json(tmp_path / "switch.json", {"schema": rule.SCHEMA, "code_hashes": previous})
    monkeypatch.setattr(switch, "code_hashes", lambda: current)
    before = files(tmp_path)
    with pytest.raises(ValueError, match="scientific code changed"):
        switch.manifest(tmp_path)
    assert files(tmp_path) == before


def test_switch_retention_rejects_tampered_previous_terminal_receipt(tmp_path, monkeypatch):
    core.atomic_json(tmp_path / "switch.json", {"schema": rule.SCHEMA, "code_hashes": initial_predecessor()})
    previous = old_switch()
    with monkeypatch.context() as patch:
        patch.setattr(switch, "code_hashes", lambda: previous)
        switch.manifest(tmp_path)
    path = tmp_path / "saved-policy-recovery-runtime.json"
    receipt = core.read(path)
    receipt["cost_policy"] = "retrain all completed arms"
    core.atomic_json(path, receipt)
    before = files(tmp_path)
    with pytest.raises(ValueError, match="frozen contract changed"):
        switch.manifest(tmp_path)
    assert files(tmp_path) == before


def test_mopps_retention_rejects_tampered_previous_terminal_receipt(tmp_path, monkeypatch):
    parent, _ = source(tmp_path)
    root = tmp_path / "comparison"
    frozen = mopps_run.prepare(root, parent)
    frozen["code_hashes"] = code_compat_predecessor()
    core.atomic_json(root / "mopps.json", frozen)
    previous = old_mopps()
    with monkeypatch.context() as patch:
        patch.setattr(mopps_run, "hashes", lambda: previous)
        mopps_run.protocol(root)
    path = root / "shared-recovery-runtime.json"
    receipt = core.read(path)
    receipt["cost_policy"] = "reset previous ledger"
    core.atomic_json(path, receipt)
    before = files(tmp_path)
    with pytest.raises(ValueError, match="frozen contract changed"):
        mopps_run.protocol(root)
    assert files(tmp_path) == before
