"""Reviewed shared-runtime upgrades must not replay or rewrite MoPPS work."""

import pytest

import mopps_comparison_gpu as run
import selection_gate as core
import selection_gate_gpu as base
import selection_switch_gpu as switch
from test_mopps_comparison_gpu import dataset_predecessor, snapshot, source


def historical_hashes(commit):
    value = run.hashes()
    value.update({
        "src/train_policy_grpo.py": "1560015999552b9481de78b69c58502543656e61216fda41ef210ad666090b42",
        "src/mopps_comparison_gpu.py": "206bed483696b1babc609f12bb399fa06b9486391b5336bce7eb9f47d0ccd7aa",
        "src/net_gain_gate_gpu.py": "3334c50158451751bf6cbbcf84afe67b9c85bcfaea8e6488a494032256b11df3",
        "src/selection_switch_gpu.py": "3886e97f49888d63e4683cc4222d6b7a7b1b1ecd3c1839b53e114f9d108ee616",
        "src/train_selection_gate_grpo.py": "c84f4a63cdeb40ee63feedffec4f3491089db35fb9fd9bbe243e1a2f09efbc9f",
    })
    expected = "1e57e00d07645d49e28cbacc693917d1c00a41f9d3b2df16e818c9756ce30419"
    if commit == "6345433":
        value.update({
            "src/selection_switch_gpu.py": "7e2e16eae01ba03194c9f8802e45c05a18995249eef58d679a1bbce8834c9e0e",
            "src/train_selection_gate_grpo.py": "9fe00567bf1b9e4637d0ef2d5d5aa5e6d8d76742878dd2587fc4de9d51bae997",
        })
        expected = "3b6ad4bb17b2e0c7cec64c535f86430bb5dd2af0f7940cec9b173634c9d462e6"
    assert core.fingerprint(value) == expected
    return value


def frozen_comparison(tmp_path, commit, *, migrated=False, monkeypatch):
    previous = historical_hashes(commit)
    parent, source_manifest = source(tmp_path)
    source_manifest["code_hashes"] = {name: previous[name] for name in switch.CODE}
    core.atomic_json(parent / "switch.json", source_manifest)
    root = tmp_path / "comparison"
    frozen = run.prepare(root, parent)
    frozen["code_hashes"] = dataset_predecessor() if migrated else previous
    core.atomic_json(root / "mopps.json", frozen)
    if migrated:
        with monkeypatch.context() as patch:
            patch.setattr(run, "hashes", lambda: previous)
            run.protocol(root)
        assert core.read(root / "dataset-runtime.json")["runtime_code_hashes"] == previous
        assert not (root / "shared-recovery-runtime.json").exists()
    directory = root / "states/s3-t25/random_online"
    core.atomic_json(directory / "result.json", {"complete": True, "completed_steps": 40})
    core.atomic_json(directory / "policy/policy_train.json", {"completed_steps": 40})
    (directory / "policy/adapter_model.safetensors").write_bytes(b"saved MoPPS control")
    (directory / "cost.jsonl").write_text("all historical charges preserved\n")
    return root, parent, frozen


@pytest.mark.parametrize("commit", ["2444e51", "6345433"])
@pytest.mark.parametrize("migrated", [False, True])
def test_shared_recovery_preserves_actual_frozen_runs_all_receipts_and_parent(tmp_path, monkeypatch, commit, migrated):
    root, parent, frozen = frozen_comparison(tmp_path, commit, migrated=migrated, monkeypatch=monkeypatch)
    before = snapshot(tmp_path)
    for _ in range(2):
        assert run.protocol(root) == frozen
        assert run.prepare(root, parent) == frozen
        assert {name: snapshot(tmp_path)[name] for name in before} == before
    receipt = core.read(root / "shared-recovery-runtime.json")
    assert receipt["schema"] == "mopps-shared-recovery-runtime/v1"
    assert receipt["runtime_code_hashes"] == run.hashes()
    assert receipt["dataset_runtime_sha256"] == base.digest(root / "dataset-runtime.json")
    assert all(name.startswith("comparison/") for name in snapshot(tmp_path).keys() - before.keys())


@pytest.mark.parametrize("name", ["src/net_gain_gate_gpu.py", "src/train_selection_gate_grpo.py", "src/train_mopps_grpo.py"])
def test_shared_recovery_rejects_unreviewed_current_scientific_hash_before_writes(tmp_path, monkeypatch, name):
    root, _, _ = frozen_comparison(tmp_path, "2444e51", monkeypatch=monkeypatch)
    before = snapshot(tmp_path)
    current = run.hashes()
    current[name] = "0" * 64
    monkeypatch.setattr(run, "hashes", lambda: current)
    with pytest.raises(ValueError, match="unreviewed|scientific code changed"):
        run.protocol(root)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("receipt", ["dataset-runtime.json", "code-compat-runtime.json"])
def test_shared_recovery_rejects_tampered_old_receipts(tmp_path, monkeypatch, receipt):
    root, _, _ = frozen_comparison(tmp_path, "2444e51", migrated=True, monkeypatch=monkeypatch)
    path = root / receipt
    value = core.read(path)
    value["cost_policy"] = "waive costs and reset saved policies"
    core.atomic_json(path, value)
    before = snapshot(tmp_path)
    with pytest.raises(ValueError, match="frozen contract changed"):
        run.protocol(root)
    assert snapshot(tmp_path) == before


def test_shared_recovery_receipt_is_bound_and_not_silently_rewritten(tmp_path, monkeypatch):
    root, _, _ = frozen_comparison(tmp_path, "2444e51", monkeypatch=monkeypatch)
    run.protocol(root)
    path = root / "shared-recovery-runtime.json"
    receipt = core.read(path)
    receipt["dataset_runtime_sha256"] = "0" * 64
    core.atomic_json(path, receipt)
    before = snapshot(tmp_path)
    with pytest.raises(ValueError, match="frozen contract changed"):
        run.protocol(root)
    assert snapshot(tmp_path) == before
