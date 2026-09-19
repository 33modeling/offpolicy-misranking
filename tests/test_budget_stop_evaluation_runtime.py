"""Evaluation-only runtime upgrades preserve frozen science and paid work."""

import pytest

import mopps_comparison_gpu as mopps_run
import net_gain_gate_gpu as net_run
import selection_gate as core
import selection_gate_gpu as base
import selection_switch as rule
import selection_switch_gpu as switch
import selector_pair as pair
import selector_pair_gpu as pair_run
from test_checkpoint_retention_runtime import files, saved_work
from test_mopps_comparison_gpu import code_compat_predecessor, source
from test_net_gain_gate_gpu import protocol as net_protocol
from test_selection_switch_gpu import initial_predecessor
from test_selector_pair_gpu import bootstrap_predecessor

OLD_NET = "fe323c553277d6b5367dd85fd651481a2c982752871a21de3285e9ec7f9722ff"
OLD_SWITCH = "b8e44146eefbf48aa44a00643eb4e1abd28f36f5fb3992785ae3a1331ff11cc8"
OLD_MOPPS = "cf80095dfe9937336bbd0ccb4a42a56cc869d10cd0c1ac820bd7ae61cee3f916"
OLD_PAIR = "fb18b8d0b7ac69ef9fd71bb748337a4673b0da46a933454567d3bd2d7e2952b1"


def test_standalone_net_evaluation_upgrade_preserves_released_protocol_and_paid_work(tmp_path):
    previous = {name: base.digest(base.ROOT / name) for name in net_run.CODE_FILES}
    previous['src/net_gain_gate_gpu.py'] = OLD_NET
    assert core.fingerprint(previous) == net_run.PRE_EVALUATION_RESUME_CODE
    frozen = {**net_protocol(), 'code_hashes': previous}
    core.atomic_json(tmp_path / 'net_protocol.json', frozen)
    saved_work(tmp_path)
    before = files(tmp_path)
    for _ in range(2):
        assert net_run.protocol(tmp_path) == frozen
        assert {path: path.read_bytes() for path in before} == before
    receipt = core.read(tmp_path / 'evaluation-resume-runtime.json')
    assert receipt['original_code_hashes'] == previous
    assert receipt['runtime_code_hashes']['src/net_gain_gate_gpu.py'] == base.digest(base.ROOT / 'src/net_gain_gate_gpu.py')
    assert receipt['checkpoint_retention_runtime_sha256'] is None
    receipt['cost_policy'] = 'refund old training'
    core.atomic_json(tmp_path / 'evaluation-resume-runtime.json', receipt)
    before = files(tmp_path)
    with pytest.raises(ValueError, match='frozen contract changed'):
        net_run.protocol(tmp_path)
    assert files(tmp_path) == before


def previous_switch():
    hashes = switch.code_hashes()
    hashes.update({"src/net_gain_gate_gpu.py": OLD_NET, "src/selection_switch_gpu.py": OLD_SWITCH})
    assert core.fingerprint(hashes) == switch.PRE_BUDGET_STOP_EVALUATION_CODE
    return hashes


def previous_mopps():
    hashes = mopps_run.hashes()
    hashes.update(previous_switch())
    hashes["src/mopps_comparison_gpu.py"] = OLD_MOPPS
    assert core.fingerprint(hashes) == mopps_run.PRE_BUDGET_STOP_EVALUATION_CODE
    return hashes


def previous_pair():
    hashes = pair_run.code_hashes()
    hashes.update(previous_switch())
    hashes["src/selector_pair_gpu.py"] = OLD_PAIR
    hashes["scripts/run_selector_pair.sh"] = "25e3f9156caa200205f12d1501d82da48299947a649b0c7abcf7e815dfac628c"
    assert core.fingerprint(hashes) == pair_run.PRE_BUDGET_STOP_EVALUATION_CODE
    return hashes


def frozen_switch(root, monkeypatch, *, migrated, dataset="mbpp"):
    previous = previous_switch()
    frozen = {"schema": rule.SCHEMA, "dataset": dataset,
              "code_hashes": initial_predecessor() if migrated else previous}
    core.atomic_json(root / "switch.json", frozen)
    if migrated:
        with monkeypatch.context() as patch:
            patch.setattr(switch, "code_hashes", lambda: previous)
            switch.manifest(root)
        assert core.read(root / "checkpoint-retention-runtime.json")["runtime_code_hashes"] == previous
    assert not (root / "budget-stop-evaluation-runtime.json").exists()
    saved_work(root)
    return frozen


@pytest.mark.parametrize("dataset", ["mbpp", "math500"])
@pytest.mark.parametrize("migrated", [False, True])
def test_switch_evaluation_upgrade_preserves_every_prior_byte(tmp_path, monkeypatch, migrated, dataset):
    frozen = frozen_switch(tmp_path, monkeypatch, migrated=migrated, dataset=dataset)
    before = files(tmp_path)
    assert switch.check_code(tmp_path) == frozen
    assert files(tmp_path) == before
    for _ in range(2):
        assert switch.manifest(tmp_path) == frozen
        assert {path: path.read_bytes() for path in before} == before
    receipt = core.read(tmp_path / "budget-stop-evaluation-runtime.json")
    assert receipt["runtime_code_hashes"] == switch.code_hashes()
    assert receipt["checkpoint_retention_runtime_sha256"] == base.digest(tmp_path / "checkpoint-retention-runtime.json")


@pytest.mark.parametrize("receipt_name", ["kv-cache-runtime.json", "checkpoint-retention-runtime.json",
                                        "budget-stop-evaluation-runtime.json"])
@pytest.mark.parametrize("damage", ["cost_policy", "runtime_code_hashes"])
def test_switch_rejects_modified_old_and_new_receipts_without_rewriting(tmp_path, monkeypatch, receipt_name, damage):
    frozen_switch(tmp_path, monkeypatch, migrated=True)
    if receipt_name.startswith("budget-stop"):
        switch.manifest(tmp_path)
    path = tmp_path / receipt_name
    receipt = core.read(path)
    receipt[damage] = "unreviewed"
    core.atomic_json(path, receipt)
    before = files(tmp_path)
    with pytest.raises((ValueError, TypeError), match="frozen contract changed"):
        switch.manifest(tmp_path)
    assert files(tmp_path) == before


@pytest.mark.parametrize("where", ["recorded", "current"])
@pytest.mark.parametrize("name", ["src/net_gain_gate_gpu.py", "src/train_policy_grpo.py", "src/selection_switch_score.py"])
def test_switch_rejects_unreviewed_components_before_writes(tmp_path, monkeypatch, where, name):
    previous, current = previous_switch(), switch.code_hashes()
    (previous if where == "recorded" else current)[name] = "unreviewed"
    core.atomic_json(tmp_path / "switch.json", {"schema": rule.SCHEMA, "code_hashes": previous})
    monkeypatch.setattr(switch, "code_hashes", lambda: current)
    before = files(tmp_path)
    with pytest.raises(ValueError, match="scientific code changed"):
        switch.manifest(tmp_path)
    assert files(tmp_path) == before


def frozen_mopps(tmp_path, monkeypatch, *, migrated):
    previous = previous_mopps()
    parent, parent_manifest = source(tmp_path)
    parent_manifest["code_hashes"] = previous_switch()
    core.atomic_json(parent / "switch.json", parent_manifest)
    root = tmp_path / "comparison"
    frozen = mopps_run.prepare(root, parent)
    frozen["code_hashes"] = code_compat_predecessor() if migrated else previous
    core.atomic_json(root / "mopps.json", frozen)
    if migrated:
        with monkeypatch.context() as patch:
            patch.setattr(mopps_run, "hashes", lambda: previous)
            mopps_run.protocol(root)
        assert core.read(root / "checkpoint-retention-runtime.json")["runtime_code_hashes"] == previous
    assert not (root / "budget-stop-evaluation-runtime.json").exists()
    saved_work(root)
    return root, frozen


@pytest.mark.parametrize("migrated", [False, True])
def test_mopps_evaluation_upgrade_preserves_parent_and_all_paid_work(tmp_path, monkeypatch, migrated):
    root, frozen = frozen_mopps(tmp_path, monkeypatch, migrated=migrated)
    before = files(tmp_path)
    for _ in range(2):
        assert mopps_run.protocol(root) == frozen
        assert {path: path.read_bytes() for path in before} == before
    receipt = core.read(root / "budget-stop-evaluation-runtime.json")
    assert receipt["runtime_code_hashes"] == mopps_run.hashes()
    assert receipt["checkpoint_retention_runtime_sha256"] == base.digest(root / "checkpoint-retention-runtime.json")


@pytest.mark.parametrize("receipt_name", ["checkpoint-retention-runtime.json", "budget-stop-evaluation-runtime.json"])
def test_mopps_rejects_tampered_terminal_receipts_without_rewriting(tmp_path, monkeypatch, receipt_name):
    root, _ = frozen_mopps(tmp_path, monkeypatch, migrated=True)
    if receipt_name.startswith("budget-stop"):
        mopps_run.protocol(root)
    path = root / receipt_name
    receipt = core.read(path)
    receipt["cost_policy"] = "refund prior training"
    core.atomic_json(path, receipt)
    before = files(tmp_path)
    with pytest.raises(ValueError, match="frozen contract changed"):
        mopps_run.protocol(root)
    assert files(tmp_path) == before


def frozen_pair(root, monkeypatch, *, migrated):
    previous = previous_pair()
    frozen = {"schema": pair.SCHEMA, "branch_manifests": {},
              "code_hashes": bootstrap_predecessor() if migrated else previous}
    frozen["protocol_id"] = core.fingerprint(frozen)
    core.atomic_json(root / "pair.json", frozen)
    if migrated:
        shared_previous = {name: previous[name] for name in switch.CODE}
        with monkeypatch.context() as patch:
            patch.setattr(pair_run, "code_hashes", lambda: previous)
            patch.setattr(switch, "code_hashes", lambda: shared_previous)
            pair_run.manifest(root)
        assert core.read(root / "shared-checkpoint-recovery-runtime.json")["runtime_code_hashes"] == previous
    assert not (root / "budget-stop-evaluation-runtime.json").exists()
    saved_work(root)
    return frozen


@pytest.mark.parametrize("migrated", [False, True])
def test_pair_evaluation_upgrade_preserves_protocol_and_previous_receipts(tmp_path, monkeypatch, migrated):
    frozen = frozen_pair(tmp_path, monkeypatch, migrated=migrated)
    before = files(tmp_path)
    for _ in range(2):
        assert pair_run.manifest(tmp_path) == frozen
        assert {path: path.read_bytes() for path in before} == before
    receipt = core.read(tmp_path / "budget-stop-evaluation-runtime.json")
    assert receipt["runtime_code_hashes"] == pair_run.code_hashes()
    assert receipt["shared_checkpoint_recovery_runtime_sha256"] == base.digest(tmp_path / "shared-checkpoint-recovery-runtime.json")


@pytest.mark.parametrize("receipt_name", ["shared-checkpoint-recovery-runtime.json", "budget-stop-evaluation-runtime.json"])
def test_pair_rejects_tampered_terminal_receipts_without_rewriting(tmp_path, monkeypatch, receipt_name):
    frozen_pair(tmp_path, monkeypatch, migrated=True)
    if receipt_name.startswith("budget-stop"):
        pair_run.manifest(tmp_path)
    path = tmp_path / receipt_name
    receipt = core.read(path)
    receipt["cost_policy"] = "reset paid work"
    core.atomic_json(path, receipt)
    before = files(tmp_path)
    with pytest.raises(ValueError, match="frozen contract changed"):
        pair_run.manifest(tmp_path)
    assert files(tmp_path) == before


@pytest.mark.parametrize("name", ["src/net_gain_gate_gpu.py", "src/selector_pair.py", pair_run.TRAINER])
def test_pair_rejects_unknown_shared_or_pair_science(monkeypatch, name):
    previous, current = previous_pair(), pair_run.code_hashes()
    current[name] = "unreviewed"
    monkeypatch.setattr(pair_run, "code_hashes", lambda: current)
    assert not pair_run.compatible_code(previous)
