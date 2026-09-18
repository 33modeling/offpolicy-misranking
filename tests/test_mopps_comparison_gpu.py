import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

import mopps
import mopps_comparison_gpu as run
import selection_gate as core
import selection_gate_gpu as base
import selection_switch as rule
import selection_switch_gpu as switch


def source(tmp_path):
    parent = tmp_path / "parent"
    cfg = {"model": str(tmp_path / "model"), "seed": 3, "drift": 0,
           "max_new_tokens": 512, "prompt_format": "olmo_rlzero_math"}
    core.atomic_json(Path(cfg["model"]) / "config.json", {})
    pool = {"train": [{"question": f"q{i}", "answer": "1"} for i in range(100)], "val": []}
    initial = tmp_path / "initial"
    core.atomic_json(initial / "prompts.json", pool)
    item = {"config": cfg, "path": str(initial), "subset": {"train": pool["train"][:10]},
            "hashes": {"prompts.json": base.digest(initial / "prompts.json")}}
    p = {"schema": rule.SCHEMA, "sources": {"3": item}, "code_hashes": switch.code_hashes(),
         "test_seeds": list(rule.TEST_SEEDS), "steps": list(rule.STEPS),
         "gpu_type": "H100", "budget_gpu_seconds": 1000., "eval_timeout": 50.}
    core.atomic_json(parent / "switch.json", p)
    return parent, p


def imported_fixture(tmp_path, monkeypatch):
    parent, parent_p = source(tmp_path)
    root = tmp_path / "comparison"
    p = run.prepare(root, parent)
    origin = run.original_point(p, 3, 25)
    initial = Path(parent_p["sources"]["3"]["path"])
    c = {"source_run": str(initial), "config": {**parent_p["sources"]["3"]["config"], "drift": 25},
         "budget_gpu_seconds": 1000., "scope": {"gpu_type": "H100", "selector": "fresh_r",
                                                 "model": base.digest(tmp_path / "model/config.json")},
         "source_hashes": {"prompts.json": base.digest(initial / "prompts.json")},
         "evaluation": {"val": [{"question": "test"}], "provenance": {}}, "max_steps": 100000}
    core.atomic_json(origin / "contract.json", c)
    core.atomic_json(origin / "decisions-frozen.json", {})
    core.atomic_json(origin / "gate-frozen.json", {})
    monkeypatch.setattr(run, "verify_origin", lambda *args: copy.deepcopy(c))
    out = run.import_point(root, p, 3, 25)
    return root, parent, p, out


def snapshot(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_prepare_is_idempotent_and_does_not_mutate_live_parent(tmp_path):
    parent, _ = source(tmp_path)
    before = snapshot(parent)
    p = run.prepare(tmp_path / "comparison", parent)
    assert p == run.prepare(tmp_path / "comparison", parent)
    assert p == run.protocol(tmp_path / "comparison")
    assert p["seeds"] == [3, 4] and p["steps"] == [25, 50, 100]
    assert p["arms"] == ["mopps", "random_online"]
    assert snapshot(parent) == before
    assert not list(parent.rglob("*.lock"))


def original_shared_hashes():
    hashes = run.hashes()
    hashes["src/net_gain_gate_gpu.py"] = "f4604802211d5bac9f7c759e1a199957883d69c04fa7eb6b15f41e58d09637a3"
    hashes["src/train_selection_gate_grpo.py"] = "14b74afcdab6230d4706f26f823e64239e6d1caea627a6ef78acc3a2c68f1c2c"
    return hashes


def code_compat_predecessor():
    hashes = original_shared_hashes()
    hashes["src/net_gate_memory_worker.py"] = switch.PRE_CACHE_GUARD_WORKER
    hashes["src/selection_gate_gpu.py"] = switch.COST_METER
    hashes.update({"src/selection_switch_gpu.py": "f87119d0f40cc0166f9095049b234c0b25a6fbaf0b910cc13bef688ce494f753",
                   "src/mopps_comparison_gpu.py": "b036737e9d8a7318bcaec361505fe8b564628d34bfed34c858b6e88ff250fb77"})
    assert core.fingerprint(hashes) == run.PRE_CODE_COMPAT_CODE
    return hashes


def test_code_compat_keeps_existing_mopps_run_and_parent_unchanged(tmp_path):
    parent, _ = source(tmp_path)
    root = tmp_path / "comparison"
    p = run.prepare(root, parent)
    p["code_hashes"] = code_compat_predecessor()
    core.atomic_json(root / "mopps.json", p)
    base.journal(root / "states/s3-t25/mopps/cost.jsonl", {"state": "started", "event_id": "interrupted"})
    before = snapshot(tmp_path)
    assert run.protocol(root) == p
    assert run.protocol(root) == p
    assert run.prepare(root, parent) == p
    after = snapshot(tmp_path)
    assert {name: after[name] for name in before} == before
    assert all(name.startswith("comparison/") for name in after.keys() - before.keys())
    receipt = core.read(root / "code-compat-runtime.json")
    assert receipt["runtime_code_hashes"] == run.hashes()
    assert receipt["original_code_hashes"] == p["code_hashes"]
    with pytest.raises(ValueError, match="unknown cost"):
        base.spent(root / "states/s3-t25/mopps")
    receipt["cost_policy"] = "ignore costs"
    core.atomic_json(root / "code-compat-runtime.json", receipt)
    with pytest.raises(ValueError, match="frozen contract changed"):
        run.protocol(root)


@pytest.mark.parametrize("migrated", [False, True])
def test_lifecycle_upgrade_preserves_existing_mopps_manifest_and_receipt(tmp_path, monkeypatch, migrated):
    parent, _ = source(tmp_path)
    root = tmp_path / "comparison"
    p = run.prepare(root, parent)
    previous = original_shared_hashes()
    previous["src/net_gate_memory_worker.py"] = switch.PRE_CACHE_GUARD_WORKER
    previous.update({"src/selection_gate_gpu.py": switch.COST_METER,
        "src/selection_switch_gpu.py": "23faf38b352f31ee64a2f2989f3b2086cc47b54508c5bd5c89571a670b3e66e8",
        "src/mopps_comparison_gpu.py": "d55710f6909df21a17d45652c35b2e8acfd553960ee5dc492b86584f9bc983d4"})
    assert core.fingerprint(previous) == run.PRE_LIFECYCLE_CODE
    p["code_hashes"] = code_compat_predecessor() if migrated else previous
    core.atomic_json(root / "mopps.json", p)
    if migrated:
        core.atomic_json(root / "code-compat-runtime.json", {
            "schema": "mopps-code-compat-runtime/v1", "protocol_sha256": base.digest(root / "mopps.json"),
            "original_code_hashes": p["code_hashes"], "runtime_code_hashes": previous,
            "change": "switch frozen-runtime compatibility and diagnostics only",
            "cost_policy": "no change to selectors, training, frozen artifacts or budgets"})
    before = snapshot(tmp_path)
    assert run.protocol(root) == p
    assert run.protocol(root) == p
    assert {name: snapshot(tmp_path)[name] for name in before} == before
    assert core.read(root / "worker-lifecycle-runtime.json")["runtime_code_hashes"] == run.hashes()


@pytest.mark.parametrize("migrated", [False, True])
def test_queue_failure_upgrade_preserves_f258c46_manifests_and_receipts(tmp_path, monkeypatch, migrated):
    parent, _ = source(tmp_path)
    root = tmp_path / "comparison"
    p = run.prepare(root, parent)
    previous = original_shared_hashes()
    previous["src/selection_gate_gpu.py"] = switch.SHUTDOWN_METER
    previous.update({"src/net_gate_memory_worker.py": switch.PRE_CACHE_GUARD_WORKER,
                     "src/selection_switch_gpu.py": "05aa36a41197cca605933df9d62bba0e4482d6f592c17954c632b45e5cf51195"})
    previous["src/mopps_comparison_gpu.py"] = "de7f40dcc15ee9bea5812dbb5132313417868ce335dede161beedd894f5a9a4b"
    assert core.fingerprint(previous) == run.PRE_QUEUE_FAILURE_CODE
    p["code_hashes"] = code_compat_predecessor() if migrated else previous
    core.atomic_json(root / "mopps.json", p)
    if migrated:
        with monkeypatch.context() as patch:
            patch.setattr(run, "hashes", lambda: previous)
            run.protocol(root)
        (root / "queue-failure-runtime.json").unlink()
    before = snapshot(tmp_path)
    assert run.protocol(root) == p
    assert run.protocol(root) == p
    assert run.prepare(root, parent) == p
    after = snapshot(tmp_path)
    assert {name: after[name] for name in before} == before
    receipt_path = root / "queue-failure-runtime.json"
    assert core.read(receipt_path)["runtime_code_hashes"] == run.hashes()
    receipt = core.read(root / "worker-lifecycle-runtime.json")
    receipt["cost_policy"] = "waive failed deployment costs"
    core.atomic_json(root / "worker-lifecycle-runtime.json", receipt)
    with pytest.raises(ValueError, match="frozen contract changed"):
        run.protocol(root)


@pytest.mark.parametrize("migrated", [False, True])
def test_cache_guard_preserves_existing_mopps_parent_costs_and_receipts(tmp_path, monkeypatch, migrated):
    parent, _ = source(tmp_path)
    root = tmp_path / "comparison"
    p = run.prepare(root, parent)
    previous = original_shared_hashes()
    previous["src/selection_gate_gpu.py"] = switch.SHUTDOWN_METER
    previous.update({"src/net_gate_memory_worker.py": switch.PRE_CACHE_GUARD_WORKER,
        "src/selection_switch_gpu.py": "05aa36a41197cca605933df9d62bba0e4482d6f592c17954c632b45e5cf51195",
        "src/mopps_comparison_gpu.py": "e1f6c2021c8904484b185a482ca158be547a4776d97321d8550dd4e8398fc603"})
    assert core.fingerprint(previous) == run.PRE_CACHE_GUARD_CODE
    p["code_hashes"] = code_compat_predecessor() if migrated else previous
    core.atomic_json(root / "mopps.json", p)
    if migrated:
        with monkeypatch.context() as patch:
            patch.setattr(run, "hashes", lambda: previous)
            run.protocol(root)
    base.journal(root / "states/s3-t25/mopps/cost.jsonl", {"state": "started", "event_id": "still-unknown"})
    before = snapshot(tmp_path)
    assert run.protocol(root) == p
    assert run.prepare(root, parent) == p
    after = snapshot(tmp_path)
    assert {name: after[name] for name in before} == before
    assert all(name.startswith("comparison/") for name in after.keys() - before.keys())
    assert core.read(root / "cache-guard-runtime.json")["runtime_code_hashes"] == run.hashes()
    with pytest.raises(ValueError, match="unknown cost"):
        base.spent(root / "states/s3-t25/mopps")
    receipt = core.read(root / "cache-guard-runtime.json")
    receipt["cost_policy"] = "ignore previous costs"
    core.atomic_json(root / "cache-guard-runtime.json", receipt)
    with pytest.raises(ValueError, match="frozen contract changed"):
        run.protocol(root)


def parallel_predecessor():
    hashes = original_shared_hashes()
    hashes["src/selection_gate_gpu.py"] = switch.SHUTDOWN_METER
    hashes["src/selection_switch_gpu.py"] = "0c0ac3aed8c5c53378c91ae5c357bcfc4e0d11fd0ffb7d2a53e9874db3e4a0b6"
    hashes["src/mopps_comparison_gpu.py"] = "e7b55d8aac5a6a347f83651d080e1253857e41721c49c69563f69ac1d852f6dd"
    assert core.fingerprint(hashes) == run.PRE_TEST_PARALLEL_CODE
    return hashes


def fit_resilience_predecessor():
    hashes = original_shared_hashes()
    hashes["src/selection_gate_gpu.py"] = switch.SHUTDOWN_METER
    hashes["src/selection_switch_gpu.py"] = "af2aa2fe039da46d5f686fe80c0833aaca5cbdf4ed9e30ae38248e0c841524a9"
    hashes["src/mopps_comparison_gpu.py"] = "cb7051a21a39260753f6e5fd15612f6beab3c3cd77af423e117fb98a7dc60a32"
    assert core.fingerprint(hashes) == run.PRE_FIT_RESILIENCE_CODE
    return hashes


def variant_root_predecessor():
    hashes = original_shared_hashes()
    hashes["src/selection_gate_gpu.py"] = switch.SHUTDOWN_METER
    hashes["src/selection_switch_gpu.py"] = "a3001f512fa99a801e32783a60ff8983fb567005319e61e6df170b7865732fa8"
    hashes["src/mopps_comparison_gpu.py"] = "689771dfb643be2ed9b5d1037d98ce1cc36fa92da3668df577dfce1b8f46f653"
    assert core.fingerprint(hashes) == run.PRE_VARIANT_ROOT_CODE
    return hashes


def dataset_predecessor():
    hashes = original_shared_hashes()
    hashes["src/selection_gate_gpu.py"] = switch.SHUTDOWN_METER
    hashes["src/selection_switch_gpu.py"] = "8d9e94df8c3813e989b447ea918f60e1283c8587e872818ba8cb29fa2b2b3521"
    hashes["src/mopps_comparison_gpu.py"] = "88caa40bf7106a9b47039b280fbd80db0db47d7f41d48e0a808b548133b39a92"
    assert core.fingerprint(hashes) == run.PRE_DATASET_CODE
    return hashes


@pytest.mark.parametrize("migrated", [False, True])
def test_dataset_preserves_47339ca_manifest_receipts_and_costs(tmp_path, monkeypatch, migrated):
    parent, _ = source(tmp_path)
    root = tmp_path / "comparison"
    p = run.prepare(root, parent)
    previous = dataset_predecessor()
    recorded = variant_root_predecessor()
    p["code_hashes"] = recorded if migrated else previous
    core.atomic_json(root / "mopps.json", p)
    if migrated:
        with monkeypatch.context() as patch:
            patch.setattr(run, "hashes", lambda: previous)
            run.protocol(root)
        # The 47339ca runtime never wrote this receipt.
        (root / "dataset-runtime.json").unlink()
        assert core.read(root / "variant-root-runtime.json")["runtime_code_hashes"] == previous
    base.journal(root / "states/s3-t25/mopps/cost.jsonl", {"state": "started", "event_id": "unknown"})
    before = snapshot(tmp_path)
    assert run.protocol(root) == p
    assert run.prepare(root, parent) == p
    after = snapshot(tmp_path)
    assert {name: after[name] for name in before} == before
    receipt = core.read(root / "dataset-runtime.json")
    assert receipt["runtime_code_hashes"] == run.hashes()
    assert receipt["variant_root_runtime_sha256"] == base.digest(root / "variant-root-runtime.json")
    with pytest.raises(ValueError, match="unknown cost"):
        base.spent(root / "states/s3-t25/mopps")
    receipt["cost_policy"] = "ignore costs"
    core.atomic_json(root / "dataset-runtime.json", receipt)
    with pytest.raises(ValueError, match="frozen contract changed"):
        run.protocol(root)


@pytest.mark.parametrize("migrated", [False, True])
def test_variant_root_preserves_b8d90c0_manifest_receipts_and_costs(tmp_path, monkeypatch, migrated):
    parent, _ = source(tmp_path)
    root = tmp_path / "comparison"
    p = run.prepare(root, parent)
    previous = variant_root_predecessor()
    recorded = fit_resilience_predecessor()
    p["code_hashes"] = recorded if migrated else previous
    core.atomic_json(root / "mopps.json", p)
    if migrated:
        with monkeypatch.context() as patch:
            patch.setattr(run, "hashes", lambda: previous)
            run.protocol(root)
        # The b8d90c0 runtime never wrote these receipts.
        (root / "variant-root-runtime.json").unlink()
        (root / "dataset-runtime.json").unlink()
        assert core.read(root / "fit-resilience-runtime.json")["runtime_code_hashes"] == previous
    base.journal(root / "states/s3-t25/mopps/cost.jsonl", {"state": "started", "event_id": "unknown"})
    before = snapshot(tmp_path)
    assert run.protocol(root) == p
    assert run.prepare(root, parent) == p
    after = snapshot(tmp_path)
    assert {name: after[name] for name in before} == before
    receipt = core.read(root / "variant-root-runtime.json")
    assert receipt["runtime_code_hashes"] == run.hashes()
    assert receipt["fit_resilience_runtime_sha256"] == base.digest(root / "fit-resilience-runtime.json")
    with pytest.raises(ValueError, match="unknown cost"):
        base.spent(root / "states/s3-t25/mopps")
    receipt["cost_policy"] = "ignore costs"
    core.atomic_json(root / "variant-root-runtime.json", receipt)
    with pytest.raises(ValueError, match="frozen contract changed"):
        run.protocol(root)


@pytest.mark.parametrize("migrated", [False, True])
def test_fit_resilience_preserves_091ae20_manifest_receipts_and_costs(tmp_path, monkeypatch, migrated):
    parent, _ = source(tmp_path)
    root = tmp_path / "comparison"
    p = run.prepare(root, parent)
    previous = fit_resilience_predecessor()
    recorded = parallel_predecessor()
    p["code_hashes"] = recorded if migrated else previous
    core.atomic_json(root / "mopps.json", p)
    if migrated:
        with monkeypatch.context() as patch:
            patch.setattr(run, "hashes", lambda: previous)
            run.protocol(root)
        # The 091ae20 runtime never wrote these receipts.
        (root / "fit-resilience-runtime.json").unlink()
        (root / "variant-root-runtime.json").unlink()
        (root / "dataset-runtime.json").unlink()
        assert core.read(root / "test-parallel-runtime.json")["runtime_code_hashes"] == previous
    base.journal(root / "states/s3-t25/mopps/cost.jsonl", {"state": "started", "event_id": "unknown"})
    before = snapshot(tmp_path)
    assert run.protocol(root) == p
    assert run.prepare(root, parent) == p
    after = snapshot(tmp_path)
    assert {name: after[name] for name in before} == before
    receipt = core.read(root / "fit-resilience-runtime.json")
    assert receipt["runtime_code_hashes"] == run.hashes()
    assert receipt["test_parallel_runtime_sha256"] == base.digest(root / "test-parallel-runtime.json")
    with pytest.raises(ValueError, match="unknown cost"):
        base.spent(root / "states/s3-t25/mopps")
    receipt["cost_policy"] = "ignore costs"
    core.atomic_json(root / "fit-resilience-runtime.json", receipt)
    with pytest.raises(ValueError, match="frozen contract changed"):
        run.protocol(root)


@pytest.mark.parametrize("migrated", [False, True])
def test_test_parallel_preserves_89c26af_manifest_receipts_and_costs(tmp_path, monkeypatch, migrated):
    parent, _ = source(tmp_path)
    root = tmp_path / "comparison"
    p = run.prepare(root, parent)
    previous = parallel_predecessor()
    recorded = {**previous, "src/mopps_comparison_gpu.py": "8379a57ad00a9dae7324a4e3e5847f48b0f03da2f32c8f50ec61989107b04d11"}
    assert core.fingerprint(recorded) == run.PRE_NONBLOCKING_RETRY_CODE
    p["code_hashes"] = recorded if migrated else previous
    core.atomic_json(root / "mopps.json", p)
    if migrated:
        # Receipts as the 89c26af runtime left them; it never wrote the test-parallel receipt.
        with monkeypatch.context() as patch:
            patch.setattr(run, "hashes", lambda: previous)
            run.protocol(root)
        (root / "test-parallel-runtime.json").unlink()
        (root / "fit-resilience-runtime.json").unlink()
        (root / "variant-root-runtime.json").unlink()
        (root / "dataset-runtime.json").unlink()
        assert core.read(root / "nonblocking-retry-runtime.json")["runtime_code_hashes"] == previous
    base.journal(root / "states/s3-t25/mopps/cost.jsonl", {"state": "started", "event_id": "unknown"})
    before = snapshot(tmp_path)
    assert run.protocol(root) == p
    assert run.prepare(root, parent) == p
    after = snapshot(tmp_path)
    assert {name: after[name] for name in before} == before
    receipt_path = root / "test-parallel-runtime.json"
    receipt = core.read(receipt_path)
    assert receipt["runtime_code_hashes"] == run.hashes()
    assert receipt["retry_runtime_sha256"] == base.digest(root / "nonblocking-retry-runtime.json")
    if migrated:
        assert core.read(root / "nonblocking-retry-runtime.json")["runtime_code_hashes"] == previous
    with pytest.raises(ValueError, match="unknown cost"):
        base.spent(root / "states/s3-t25/mopps")
    receipt["cost_policy"] = "ignore costs"
    core.atomic_json(receipt_path, receipt)
    with pytest.raises(ValueError, match="frozen contract changed"):
        run.protocol(root)


@pytest.mark.parametrize("migrated", [False, True])
def test_nonblocking_retry_preserves_bdd727e_manifest_receipts_and_costs(tmp_path, monkeypatch, migrated):
    parent, _ = source(tmp_path)
    root = tmp_path / "comparison"
    p = run.prepare(root, parent)
    previous = parallel_predecessor()
    previous["src/mopps_comparison_gpu.py"] = "8379a57ad00a9dae7324a4e3e5847f48b0f03da2f32c8f50ec61989107b04d11"
    assert core.fingerprint(previous) == run.PRE_NONBLOCKING_RETRY_CODE
    recorded = {**previous, "src/net_gate_memory_worker.py": switch.PRE_CACHE_GUARD_WORKER,
        "src/selection_gate_gpu.py": switch.COST_METER,
        "src/selection_switch_gpu.py": "23faf38b352f31ee64a2f2989f3b2086cc47b54508c5bd5c89571a670b3e66e8",
        "src/mopps_comparison_gpu.py": "d55710f6909df21a17d45652c35b2e8acfd553960ee5dc492b86584f9bc983d4"}
    assert core.fingerprint(recorded) == run.PRE_LIFECYCLE_CODE
    p["code_hashes"] = recorded if migrated else previous
    core.atomic_json(root / "mopps.json", p)
    if migrated:
        # Reproduce the mixed receipt revisions in the operator's 10:06 export.
        lifecycle = {**previous, "src/net_gate_memory_worker.py": switch.PRE_CACHE_GUARD_WORKER,
            "src/selection_switch_gpu.py": "05aa36a41197cca605933df9d62bba0e4482d6f592c17954c632b45e5cf51195",
            "src/mopps_comparison_gpu.py": "de7f40dcc15ee9bea5812dbb5132313417868ce335dede161beedd894f5a9a4b"}
        queue = {**lifecycle, "src/mopps_comparison_gpu.py": "e1f6c2021c8904484b185a482ca158be547a4776d97321d8550dd4e8398fc603"}
        assert core.fingerprint(lifecycle) == run.PRE_QUEUE_FAILURE_CODE
        assert core.fingerprint(queue) == run.PRE_CACHE_GUARD_CODE
        with monkeypatch.context() as patch:
            patch.setattr(run, "hashes", lambda: lifecycle)
            run.protocol(root)
        (root / "queue-failure-runtime.json").unlink()
        with monkeypatch.context() as patch:
            patch.setattr(run, "hashes", lambda: queue)
            run.protocol(root)
        with monkeypatch.context() as patch:
            patch.setattr(run, "hashes", lambda: previous)
            run.protocol(root)
        (root / "nonblocking-retry-runtime.json").unlink()
        # Neither predecessor runtime wrote the later receipts.
        (root / "test-parallel-runtime.json").unlink()
        (root / "fit-resilience-runtime.json").unlink()
        (root / "variant-root-runtime.json").unlink()
        (root / "dataset-runtime.json").unlink()
    base.journal(root / "states/s3-t25/mopps/cost.jsonl", {"state": "started", "event_id": "unknown"})
    before = snapshot(tmp_path)
    assert run.protocol(root) == p
    assert run.prepare(root, parent) == p
    after = snapshot(tmp_path)
    assert {name: after[name] for name in before} == before
    assert all(name.startswith("comparison/") for name in after.keys() - before.keys())
    receipt_path = root / "nonblocking-retry-runtime.json"
    receipt = core.read(receipt_path)
    assert receipt["runtime_code_hashes"] == run.hashes()
    assert receipt["cache_guard_runtime_sha256"] == base.digest(root / "cache-guard-runtime.json")
    with pytest.raises(ValueError, match="unknown cost"):
        base.spent(root / "states/s3-t25/mopps")
    receipt["cost_policy"] = "ignore costs"
    core.atomic_json(receipt_path, receipt)
    with pytest.raises(ValueError, match="frozen contract changed"):
        run.protocol(root)


@pytest.mark.parametrize("name", ["src/mopps.py", "src/train_mopps_grpo.py", "src/grads.py", "src/selection_switch.py", "src/net_gate_memory_worker.py"])
@pytest.mark.parametrize("where", ["recorded", "current"])
def test_code_compat_rejects_mopps_scientific_changes(tmp_path, monkeypatch, name, where):
    parent, _ = source(tmp_path)
    root = tmp_path / "comparison"
    p = run.prepare(root, parent)
    p["code_hashes"] = code_compat_predecessor()
    if where == "recorded":
        p["code_hashes"][name] = "unreviewed"
    else:
        current = run.hashes()
        current[name] = "unreviewed"
        monkeypatch.setattr(run, "hashes", lambda: current)
    core.atomic_json(root / "mopps.json", p)
    before = snapshot(tmp_path)
    with pytest.raises(ValueError, match="unreviewed code hashes"):
        run.protocol(root)
    assert snapshot(tmp_path) == before


def test_code_compat_prepare_still_rejects_changed_frozen_design(tmp_path):
    parent, _ = source(tmp_path)
    root = tmp_path / "comparison"
    p = run.prepare(root, parent)
    p["code_hashes"] = code_compat_predecessor()
    p["sampling"] = "different sampling design"
    core.atomic_json(root / "mopps.json", p)
    before = snapshot(tmp_path)
    with pytest.raises(ValueError, match="frozen contract changed"):
        run.prepare(root, parent)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("target", ["parent", "parent/nested", "model", "initial", "."])
def test_output_overlap_rejected_before_any_output_write(tmp_path, target):
    parent, _ = source(tmp_path)
    before = snapshot(tmp_path)
    with pytest.raises(ValueError, match="separate"):
        run.prepare(tmp_path / target, parent)
    assert snapshot(tmp_path) == before


def test_import_keeps_full_pool_original_budget_and_is_read_only_on_source(tmp_path, monkeypatch):
    root, parent, p, out = imported_fixture(tmp_path, monkeypatch)
    before = snapshot(parent)
    c = run.verify(out)
    assert c["budget_gpu_seconds"] == p["budget_gpu_seconds"]
    assert c["scope"]["selector"] == "mopps_comparison"
    for arm in mopps.ARMS:
        assert len(core.read(out / "subsets" / f"subset-{arm}.json")["train"]) == 100
        assert core.read(out / arm / "selector.json") == mopps.specification(arm, 3, 25, 100)
    run.import_point(root, p, 3, 25)
    assert snapshot(parent) == before


@pytest.mark.parametrize("mutation", ["subset", "selector", "barrier", "parent_manifest", "code"])
def test_import_rejects_changed_input_or_runtime(tmp_path, monkeypatch, mutation):
    root, parent, p, out = imported_fixture(tmp_path, monkeypatch)
    if mutation == "subset": core.atomic_json(out / "subsets/subset-mopps.json", {"train": []})
    elif mutation == "selector": core.atomic_json(out / "mopps/selector.json", {})
    elif mutation == "barrier": core.atomic_json(run.original_point(p, 3, 25) / "decisions-frozen.json", {"changed": True})
    elif mutation == "parent_manifest": core.atomic_json(parent / "switch.json", {})
    else: monkeypatch.setattr(run, "hashes", lambda: {})
    with pytest.raises((ValueError, KeyError)):
        run.verify(out)


def test_worker_command_keeps_parent_optimizer_deadline_and_independent_driver(tmp_path, monkeypatch):
    out = tmp_path / "states/s3-t25"
    monkeypatch.setattr(base, "train_command", lambda *args: [sys.executable,
        str(base.ROOT / "src/train_selection_gate_grpo.py"), "--resume-optimizer", "original/optimizer.pt",
        "--wall-budget-deadline", "123", "--budget-save-reserve", "30"])
    command = run.train_command(out, {}, "mopps", 1000.)
    assert str(base.ROOT / "src/train_mopps_grpo.py") in command
    assert command[command.index("--resume-optimizer")+1] == "original/optimizer.pt"
    assert command[-2:] == ["--selector-config", str(out / "mopps/selector.json")]
    assert "--wall-budget-deadline" in command


def test_failed_training_cost_survives_and_unknown_cost_blocks_retry(tmp_path, monkeypatch):
    out = tmp_path / "states/s3-t25"
    c = {"budget_gpu_seconds": 1000., "config": {"drift": 25}, "max_steps": 100000}
    monkeypatch.setattr(run, "verify", lambda _: c)
    monkeypatch.setattr(run, "train_command", lambda *args: [sys.executable, "-c", "raise RuntimeError('inner worker cause')"])
    with pytest.raises(RuntimeError, match="inner worker cause"):
        run.run_arm(out, {"gpu_type": "H100"}, "mopps", list("0123"), {})
    directory = out / "mopps"
    assert base.spent(directory) > 0
    assert base.cost(directory)["ledgers"]["deployment"]["failed_events"] == 1
    base.journal(directory / "cost.jsonl", {"state": "started", "event_id": "interrupted"})
    before = snapshot(directory)
    with pytest.raises(ValueError, match="unknown cost"):
        run.run_arm(out, {"gpu_type": "H100"}, "mopps", list("0123"), {})
    assert snapshot(directory) == before


def test_status_and_errors_do_not_initialize_model_or_mutate_sources(tmp_path):
    parent, _ = source(tmp_path)
    root = tmp_path / "comparison"
    run.prepare(root, parent)
    directory = root / "states/s3-t25/mopps"
    core.atomic_json(directory / "failure.json", {"error": "train worker failed"})
    core.atomic_json(directory / "progress.json", {"phase": "train", "state": "failed"})
    (directory / "train-0.log").write_text("RuntimeError: original error\n")
    before = snapshot(tmp_path)
    env = {**os.environ, "MOPPS_ROOT": str(root), "MOPPS_PYTHON": sys.executable}
    for mode in ("status", "errors"):
        result = subprocess.run(["bash", "scripts/run_mopps_comparison.sh", mode], cwd=base.ROOT,
                                env=env, capture_output=True, text=True, timeout=20)
        assert result.returncode == 0, result.stdout + result.stderr
        assert ("FAILED" if mode == "status" else "RuntimeError: original error") in result.stdout
    assert snapshot(tmp_path) == before


def test_original_incomplete_cost_is_never_repaired_by_report(tmp_path, monkeypatch):
    out = tmp_path / "states/s3-t25/points/view-25"
    core.atomic_json(out.parent.parent / "net_protocol.json", {})
    base.journal(out / "selection_full/cost.jsonl", {"state": "started", "event_id": "open"})
    monkeypatch.setattr(base, "spent", lambda _: pytest.fail("source repair attempted"))
    before = snapshot(tmp_path)
    with pytest.raises(ValueError, match="source was not repaired"):
        run.original_result(out, "selection_full")
    assert snapshot(tmp_path) == before


def test_sidecar_cost_recovery_obeys_task_lock_and_actual_duration(tmp_path):
    sys.path.insert(0, str(base.ROOT / "scripts"))
    import recover_selection_switch_cost as recovery
    root = tmp_path / "comparison"
    core.atomic_json(root / "mopps.json", {})
    directory = root / "states/s3-t25/mopps"
    start = {"state": "started", "event_id": "stopped", "phase": "train", "ledger": "deployment",
             "gpus": 4, "gpu_type": "H100", "host": "stopped-node", "time": 100.}
    base.journal(directory / "cost.jsonl", start)
    core.atomic_json(directory / "progress.json", {**start, "state": "running", "seconds": 12.})
    with base.lease(directory / ".task.lock"), pytest.raises(BlockingIOError):
        recovery.recover(root, directory, "stopped", seconds=15., reason="termination log")
    with pytest.raises(ValueError, match="lower bound"):
        recovery.recover(root, directory, "stopped")
    recovery.recover(root, directory, "stopped", seconds=15., reason="termination log")
    assert base.spent(directory) == 60.


def certified_origin(tmp_path, monkeypatch, *, full_config=False):
    import evidence_downstream as ed
    import train_policy_grpo as trainer
    from test_selection_switch import development
    parent, p = source(tmp_path)
    p.update(eval_k=8, evaluation={"val": [{"question": "held-out", "answer": "1"}], "provenance": {}})
    if full_config:
        p["sources"]["3"]["config"].update(dataset="math500", grpo_world_size=4, behavior_k=8,
            grpo_group_size=8, grpo_epochs_per_batch=1, topk_frac=.1, temperature=1., top_p=1.,
            grpo_clip_epsilon=.2, grpo_learning_rate=1e-5, grpo_max_grad_norm=1.,
            grpo_advantage_epsilon=1e-6, grpo_lora_rank=4, grpo_lora_alpha=8,
            grpo_logprob_micro_batch=1, grpo_gradient_checkpointing=True)
        p["evaluation"]["val"] = [{"question": f"held-out {i}", "answer": "1"} for i in range(4)]
        monkeypatch.setattr(trainer, "validate_policy_manifest", lambda *args, **kwargs:
                            {"seed": 3, "prompt_format": "olmo_rlzero_math"})
    core.atomic_json(parent / "switch.json", p)
    item = p["sources"]["3"]
    initial = Path(item["path"])
    (initial / "rollouts_behavior_train.jsonl").write_text('{}\n')
    directory = switch.prefix_dir(parent, 3)
    core.atomic_json(directory / "subset.json", item["subset"])
    policy = directory / "policy_step_25"
    for name in ed.POLICY_FILES:
        core.atomic_json(policy / name, {"fixture": name})
    cert = {"schema": rule.SCHEMA, "seed": 3, "step": 25, "previous": 0, "selector": "fresh_r",
            "subset_sha256": base.digest(directory / "subset.json"), "source_sha256": core.fingerprint(item),
            "policy_hashes": {name: base.digest(policy / name) for name in ed.POLICY_FILES}}
    core.atomic_json(directory / "prefix-25.json", cert)
    view = directory / "view-25"
    core.atomic_json(view / "run_config.json", {**item["config"], "drift": 25})
    core.atomic_json(view / "selected-prefix.json", cert)
    (view / "policy_step_25").symlink_to(policy, target_is_directory=True)
    for name in ("prompts.json", "rollouts_behavior_train.jsonl"):
        (view / name).symlink_to(initial / name)
    source_hashes = {name: base.digest(view / name) for name in
                    ["run_config.json", "prompts.json", "selected-prefix.json", "rollouts_behavior_train.jsonl"]
                    + [f"policy_step_25/{name}" for name in ed.POLICY_FILES]}
    c = {"config": {**item["config"], "drift": 25}, "source_run": str(view), "source_hashes": source_hashes,
         "scope": {"selector": "fresh_r", "gpu_type": "H100", "model": base.digest(tmp_path / "model/config.json")},
         "selected_prefix": {"schema": rule.SCHEMA, "root": str(parent), "certificate_sha256": core.fingerprint(cert)},
         "budget_gpu_seconds": 1000., "role": "test", "n": 100, "eval_k": 8, "eval_seed": 704000012,
         "max_steps": 100000, "evaluation": p["evaluation"]}
    origin = switch.child_root(parent, 3, 25) / "points/view-25"
    core.atomic_json(origin / "contract.json", c)
    core.atomic_json(origin / "evaluation.json", c["evaluation"])
    model = rule.fit(development())
    core.atomic_json(parent / "model.json", model)
    net = {"schema": rule.SCHEMA, "mode": "test", "arms": list(rule.TEST_ARMS), "model": None,
           "code_hashes": p["code_hashes"]}
    core.atomic_json(origin.parent.parent / "net_protocol.json", net)
    gate = switch.gate_path(origin.parent.parent)
    core.atomic_json(gate, {"schema": rule.SCHEMA, "protocol_sha256": core.fingerprint(net),
                            "model_sha256": base.digest(parent / "model.json"), "model": model})
    for arm in rule.TEST_ARMS:
        core.atomic_json(origin / arm / "decision.json", {"action": "random"})
    controls = [arm for arm in rule.TEST_ARMS if arm != "gated"]
    core.atomic_json(origin / "decisions-frozen.json", {"protocol_sha256": core.fingerprint(net),
        "decisions": {arm: base.digest(origin / arm / "decision.json") for arm in controls}})
    core.atomic_json(origin / "gate-frozen.json", {"protocol_sha256": core.fingerprint(net),
        "gate_sha256": base.digest(gate), "controls_sha256": base.digest(origin / "decisions-frozen.json"),
        "decision": base.digest(origin / "gated/decision.json")})
    calls = []
    monkeypatch.setattr(trainer, "validate_policy_lineage", lambda path, **kwargs: calls.append((path, kwargs)))
    monkeypatch.setattr(ed, "_expected_config", lambda _: {})
    monkeypatch.setattr(ed, "independent_test", lambda prompts, evaluation: evaluation["val"])
    comparison = run.prepare(tmp_path / "comparison", parent)
    return parent, comparison, origin, calls


def test_actual_origin_validation_binds_policy_optimizer_prefix_and_frozen_gate(tmp_path, monkeypatch):
    parent, p, origin, calls = certified_origin(tmp_path, monkeypatch)
    before = snapshot(parent)
    assert run.verify_origin(p, 3, 25) == core.read(origin / "contract.json")
    assert len(calls) == 1
    assert calls[0][1]["require_complete_hashes"] is True
    assert calls[0][1]["expected_prompts"] == parent / "prefixes/seed-3/subset.json"
    assert snapshot(parent) == before


def test_prefix_only_import_starts_without_gate_and_preserves_parent(tmp_path, monkeypatch):
    import shutil
    parent, p, origin, _ = certified_origin(tmp_path, monkeypatch, full_config=True)
    root = tmp_path / "comparison"
    shutil.rmtree(parent / "states")
    shutil.rmtree(parent / "prefixes/seed-3/view-25")
    (parent / "model.json").unlink()
    before = snapshot(parent)
    assert run.ready(p, 3, 25)
    assert not run.ready(p, 3, 50)
    out = run.import_point(root, p, 3, 25)
    c = run.verify(out)
    assert c["comparison"]["source_kind"] == "certified_prefix"
    assert c["comparison"]["origin"] == str(origin)
    assert c["budget_gpu_seconds"] == p["budget_gpu_seconds"]
    assert c["eval_seed"] == 704000012 and c["eval_k"] == 8
    assert (Path(c["source_run"]) / "policy_step_25").resolve() == parent / "prefixes/seed-3/policy_step_25"
    assert len(core.read(out / "subsets/subset-mopps.json")["train"]) == 100
    imported = snapshot(out)
    run.import_point(root, p, 3, 25)
    assert {name: snapshot(out)[name] for name in imported if not name.startswith("import-cost/")} == {
        name: value for name, value in imported.items() if not name.startswith("import-cost/")}
    assert snapshot(parent) == before
    assert not (parent / "model.json").exists() and not (parent / "states").exists()
    with pytest.raises(FileNotFoundError):
        run.verify_origin(p, 3, 25)
    original_contract = (out / "contract.json").read_bytes()
    core.atomic_json(origin / "contract.json", {"later": "publication"})
    core.atomic_json(origin / "decisions-frozen.json", {"later": "publication"})
    later_parent = snapshot(parent)
    run.import_point(root, p, 3, 25)
    assert run.verify(out) == c
    assert (out / "contract.json").read_bytes() == original_contract
    assert snapshot(parent) == later_parent


@pytest.mark.parametrize("artifact", ["optimizer", "certificate", "view", "evaluation"])
def test_prefix_only_import_rejects_changed_frozen_input(tmp_path, monkeypatch, artifact):
    parent, p, origin, _ = certified_origin(tmp_path, monkeypatch, full_config=True)
    (origin / "decisions-frozen.json").unlink()
    out = run.import_point(tmp_path / "comparison", p, 3, 25)
    if artifact == "optimizer": path = parent / "prefixes/seed-3/policy_step_25/optimizer.pt"
    elif artifact == "certificate": path = parent / "prefixes/seed-3/prefix-25.json"
    elif artifact == "view": path = out / "source/run_config.json"
    else: path = out / "evaluation.json"
    core.atomic_json(path, {"changed": True})
    with pytest.raises((ValueError, KeyError)):
        run.verify(out)


def test_missing_prefix_is_named_in_wait_instead_of_claiming_gate_dependency(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace
    parent, _ = source(tmp_path)
    root = tmp_path / "comparison"
    run.prepare(root, parent)
    monkeypatch.setitem(sys.modules, "additive_experiment", SimpleNamespace(model_environment=lambda _: {}))
    monkeypatch.setattr(switch, "admitted_devices", lambda _: list("0123"))
    assert run.work(root, idle_timeout=0.) == 0
    output = capsys.readouterr().out
    assert "prefixes/seed-3/prefix-25.json missing (Gate not required)" in output
    assert "WAIT=12" in output
    assert not list(root.glob("states/*/*/cost.jsonl"))


@pytest.mark.parametrize("active", [False, True])
def test_nonblocking_retry_skips_locked_branch_then_runs_an_independent_one(tmp_path, monkeypatch, active):
    from types import SimpleNamespace
    parent, _ = source(tmp_path)
    root = tmp_path / "comparison"
    run.prepare(root, parent)
    out = run.point(root, 3, 25)
    core.atomic_json(out / "import.done.json", {})
    core.atomic_json(out / "contract.json", {"config": {}})
    for arm in mopps.ARMS:
        core.atomic_json(out / arm / "failure.json", {"error": "earlier training failure"})
    if active:
        core.atomic_json(out / "mopps/progress.json", {"state": "running", "updated": time.time(),
            "host": "other-node", "pid": 999, "phase": "train"})
    before = snapshot(out / "mopps")
    parent_before = snapshot(parent)
    monkeypatch.setitem(sys.modules, "additive_experiment", SimpleNamespace(model_environment=lambda _: {}))
    monkeypatch.setattr(switch, "admitted_devices", lambda _: list("0123"))
    monkeypatch.setattr(run, "ready", lambda *args: True)
    monkeypatch.setattr(run.time, "sleep", lambda _: pytest.fail("bulk retry must not sleep behind an active peer"))
    calls = []
    monkeypatch.setattr(run, "run_arm", lambda out, p, arm, *args: calls.append(arm))
    with base.lease(out / "mopps/.task.lock"):
        assert run.work(root, idle_timeout=0., only=(3, 25, "mopps")) == 0
        assert calls == []
        assert run.work(root, idle_timeout=0., only=(3, 25, "random_online")) == 0
    assert calls == ["random_online"]
    assert {name: snapshot(out / "mopps")[name] for name in before} == before
    assert snapshot(parent) == parent_before


def test_failed_prefixes_stop_all_waiting_without_launching_or_changing_parent(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace
    parent, _ = source(tmp_path)
    root = tmp_path / "comparison"
    run.prepare(root, parent)
    for seed in rule.TEST_SEEDS:
        core.atomic_json(parent / f"prefixes/seed-{seed}/segment-25/failure.json",
                         {"error": "prefix-train worker failed: ncclUnhandledCudaError"})
    before = snapshot(parent)
    monkeypatch.setitem(sys.modules, "additive_experiment", SimpleNamespace(model_environment=lambda _: {}))
    monkeypatch.setattr(switch, "admitted_devices", lambda _: list("0123"))
    monkeypatch.setattr(run.time, "sleep", lambda _: pytest.fail("failed prerequisite kept GPUs waiting"))
    monkeypatch.setattr(run, "run_arm", lambda *args: pytest.fail("launched without a valid prefix"))
    assert run.work(root) == 1
    output = capsys.readouterr().out
    assert "BLOCKED=12" in output
    assert "ncclUnhandledCudaError" in output
    assert "[waiting]" not in output
    assert snapshot(parent) == before
    assert not list(root.glob("states/*/*/cost.jsonl"))


def test_failed_earlier_segment_blocks_later_prefix_but_not_completed_prefix(tmp_path):
    parent, _ = source(tmp_path)
    p = run.prepare(tmp_path / "comparison", parent)
    directory = parent / "prefixes/seed-3"
    core.atomic_json(directory / "prefix-25.json", {})
    core.atomic_json(directory / "segment-50/failure.json", {"error": "NCCL failure"})
    before = snapshot(parent)
    assert run.prefix_dependency(p, 3, 25)["failure"] is None
    for step in (50, 100):
        dependency = run.prefix_dependency(p, 3, step)
        assert "segment-50/failure.json" in dependency["failure"]
        assert "NCCL failure" in dependency["failure"]
        assert not dependency["active"]
    assert snapshot(parent) == before


def test_active_prefix_retry_is_not_blocked_by_previous_failure(tmp_path):
    parent, _ = source(tmp_path)
    p = run.prepare(tmp_path / "comparison", parent)
    directory = parent / "prefixes/seed-3"
    core.atomic_json(directory / "segment-25/failure.json", {"error": "old NCCL failure"})
    with base.lease(directory / ".prefix.lock"):
        before = snapshot(parent)
        dependency = run.prefix_dependency(p, 3, 100)
        assert dependency["active"]
        assert dependency["failure"] is None
        assert snapshot(parent) == before
    assert run.prefix_dependency(p, 3, 100)["failure"] is not None


def test_waiting_queue_resumes_when_active_prefix_retry_publishes(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace
    parent, _ = source(tmp_path)
    root = tmp_path / "comparison"
    p = run.prepare(root, parent)
    for seed in rule.TEST_SEEDS:
        core.atomic_json(parent / f"prefixes/seed-{seed}/segment-25/failure.json", {"error": "old failure"})
    monkeypatch.setitem(sys.modules, "additive_experiment", SimpleNamespace(model_environment=lambda _: {}))
    monkeypatch.setattr(switch, "admitted_devices", lambda _: list("0123"))
    lease = base.lease(parent / "prefixes/seed-3/.prefix.lock")
    lease.__enter__()
    held = True
    sleeps, claims = [], []
    def publish_on_sleep(seconds):
        nonlocal held
        sleeps.append(seconds)
        assert held
        origin = run.original_point(p, 3, 25)
        core.atomic_json(origin / "contract.json", {})
        core.atomic_json(origin / "decisions-frozen.json", {})
        core.atomic_json(origin / "gate-frozen.json", {})
        lease.__exit__(None, None, None)
        held = False
    def import_ready(root, protocol, seed, step):
        out = run.point(root, seed, step)
        core.atomic_json(out / "contract.json", {"config": {}})
        core.atomic_json(out / "import.done.json", {})
        return out
    def train(out, protocol, arm, devices, env):
        claims.append((out.name, arm))
        core.atomic_json(out / arm / "result.json", {})
        core.atomic_json(out / arm / "result.sha256.json", {"sha256": base.digest(out / arm / "result.json")})
    monkeypatch.setattr(run.time, "sleep", publish_on_sleep)
    monkeypatch.setattr(run, "import_point", import_ready)
    monkeypatch.setattr(run, "run_arm", train)
    try:
        assert run.work(root) == 1
    finally:
        if held:
            lease.__exit__(None, None, None)
    assert sleeps == [15]
    assert set(claims) == {("s3-t25", arm) for arm in mopps.ARMS}
    assert "DONE=2" in capsys.readouterr().out


def test_four_waiting_controllers_exit_on_failed_prefix_without_gpu_claims(tmp_path):
    parent, _ = source(tmp_path)
    root = tmp_path / "comparison"
    run.prepare(root, parent)
    for seed in rule.TEST_SEEDS:
        core.atomic_json(parent / f"prefixes/seed-{seed}/segment-25/failure.json", {"error": "NCCL failure"})
    before = snapshot(tmp_path)
    script = """
import sys
from pathlib import Path
from types import SimpleNamespace
import mopps_comparison_gpu as m
sys.modules['additive_experiment'] = SimpleNamespace(model_environment=lambda _: {})
m.switch.admitted_devices = lambda _: list('0123')
def unexpected(*args):
    raise AssertionError('failed dependencies must not sleep or launch')
m.time.sleep = unexpected
m.run_arm = unexpected
raise SystemExit(m.work(Path(sys.argv[1])))
"""
    workers = [subprocess.Popen([sys.executable, "-c", script, str(root)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(4)]
    try:
        for worker in workers:
            stdout, stderr = worker.communicate(timeout=10)
            assert worker.returncode == 1, stdout + stderr
            assert "BLOCKED=12" in stdout
            assert "[claimed]" not in stdout and "[waiting]" not in stdout
            assert not stderr
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
                worker.communicate()
    assert snapshot(tmp_path) == before


def test_healthy_ready_arms_run_even_when_other_prefixes_failed(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace
    root, parent, p, out = imported_fixture(tmp_path, monkeypatch)
    core.atomic_json(parent / "prefixes/seed-4/segment-25/failure.json", {"error": "NCCL failure"})
    before = snapshot(parent)
    monkeypatch.setitem(sys.modules, "additive_experiment", SimpleNamespace(model_environment=lambda _: {}))
    monkeypatch.setattr(switch, "admitted_devices", lambda _: list("0123"))
    monkeypatch.setattr(run.time, "sleep", lambda _: pytest.fail("no remaining producer; do not keep waiting"))
    calls = []
    def train(point, protocol, arm, devices, env):
        assert point == out
        calls.append(arm)
        base.bind(point / arm / "result.json", {"complete": True})
        base.bind(point / arm / "result.sha256.json", {"sha256": base.digest(point / arm / "result.json")})
    monkeypatch.setattr(run, "run_arm", train)
    assert run.work(root) == 1
    assert sorted(calls) == sorted(mopps.ARMS)
    assert "DONE=2" in capsys.readouterr().out
    assert snapshot(parent) == before


@pytest.mark.parametrize("artifact", ["prefix", "optimizer", "decision", "model", "budget", "source"])
def test_origin_validation_rejects_mismatched_parent_or_decision(tmp_path, monkeypatch, artifact):
    parent, p, origin, _ = certified_origin(tmp_path, monkeypatch)
    if artifact == "prefix": path = parent / "prefixes/seed-3/prefix-25.json"
    elif artifact == "optimizer": path = parent / "prefixes/seed-3/policy_step_25/optimizer.pt"
    elif artifact == "decision": path = origin / "gated/decision.json"
    elif artifact == "model": path = parent / "model.json"
    else:
        path = origin / "contract.json"
        c = core.read(path)
        if artifact == "budget": c["budget_gpu_seconds"] += 1
        else: c["source_hashes"].pop("policy_step_25/optimizer.pt")
        core.atomic_json(path, c)
        path = None
    if path is not None: core.atomic_json(path, {"changed": True})
    with pytest.raises((ValueError, KeyError)):
        run.verify_origin(p, 3, 25)


def test_result_resume_recovers_receipt_only_after_full_validation(tmp_path, monkeypatch):
    out = tmp_path / "states/s3-t25"
    directory = out / "mopps"
    c = {"budget_gpu_seconds": 1000.}
    monkeypatch.setattr(run, "verify", lambda _: c)
    monkeypatch.setattr(base, "rewards", lambda *args: {"0": .5})
    for path in run.result_paths(out, "mopps"):
        core.atomic_json(path, {})
    stop = {"completed_steps": 26, "stop_reason": "no_block_fits"}
    core.atomic_json(directory / "policy/budget_stop.json", stop)
    result = {"schema": mopps.SCHEMA, "complete": True, "arm": "mopps", "rewards": {"0": .5},
              "budget_gpu_seconds": 1000., "used_gpu_seconds": 0., "cost": base.cost(directory), **stop,
              "artifact_hashes": {str(path.relative_to(out)): base.digest(path) for path in run.result_paths(out, "mopps")}}
    core.atomic_json(directory / "result.json", result)
    with pytest.raises(ValueError, match="receipt"):
        run.validate_result(out, "mopps")
    assert run.validate_result(out, "mopps", recover_receipt=True) == result
    assert core.read(directory / "result.sha256.json") == {"sha256": base.digest(directory / "result.json")}
    result["used_gpu_seconds"] = 1001.
    core.atomic_json(directory / "result.json", result)
    (directory / "result.sha256.json").unlink()
    with pytest.raises(ValueError, match="cost"):
        run.validate_result(out, "mopps", recover_receipt=True)
    assert not (directory / "result.sha256.json").exists()


def test_primary_is_executed_gate_minus_mopps_with_diagnostic_in_total_budget():
    gated = {"complete": True, "budget_gpu_seconds": 980., "measurement_gpu_seconds": 20.,
             "used_gpu_seconds": 900., "rewards": {"0": .4, "1": .6}, "action": "random"}
    reward = {"complete": True, "budget_gpu_seconds": 1000., "used_gpu_seconds": 950.,
              "rewards": {"0": .5, "1": .7}}
    result = run.primary_comparison(gated, reward, 1000., 3)
    assert result["gate_minus_mopps_pp"] == pytest.approx(-10.)
    assert result["gate_total_gpu_seconds"] == 920.
    assert result["mopps_total_gpu_seconds"] == 950.
    assert result["gate_executed_action"] == "random"
    assert result["conditional_prompt_ci95_pp"] == pytest.approx([-10., -10.])
    with pytest.raises(ValueError, match="allocations"):
        run.primary_comparison({**gated, "budget_gpu_seconds": 1000.}, reward, 1000., 3)
    with pytest.raises(ValueError, match="both executed"):
        run.primary_comparison({**gated, "complete": False}, reward, 1000., 3)


@pytest.mark.parametrize("missing_gate", [False, True])
def test_report_never_substitutes_winning_control_for_actual_gate(tmp_path, monkeypatch, missing_gate, capsys):
    p = {"seeds": [3, 4], "steps": [25, 50, 100], "arms": list(mopps.ARMS),
         "parent": str(tmp_path / "parent"), "budget_gpu_seconds": 1000.}
    monkeypatch.setattr(run, "protocol", lambda _: p)
    monkeypatch.setattr(run, "verify_origin", lambda *args: None)
    def result(arm):
        if arm == "gated" and missing_gate: raise FileNotFoundError("actual gate not finished")
        value = {"gated": .4, "mopps": .5, "selection_full": .99, "random_full": .3, "random_online": .2}[arm]
        return {"complete": True, "budget_gpu_seconds": 980. if arm == "gated" else 1000.,
                "measurement_gpu_seconds": 20. if arm == "gated" else 0., "used_gpu_seconds": 800.,
                "action": "random", "rewards": {"0": value, "1": value}}
    monkeypatch.setattr(run, "validate_result", lambda out, arm: result(arm))
    monkeypatch.setattr(run, "original_result", lambda out, arm: result(arm))
    for seed in p["seeds"]:
        for step in p["steps"]:
            for arm in p["arms"]:
                core.atomic_json(run.point(tmp_path, seed, step) / arm / "result.json", {})
    report = run.summarize(tmp_path)
    assert "PRIMARY: executed Gate vs MoPPS" in capsys.readouterr().out
    assert report["primary_complete"] is not missing_gate
    assert report["complete"] is not missing_gate
    if missing_gate:
        assert all("primary" not in row and "gated" in row["original_errors"] for row in report["states"])
    else:
        assert all(row["primary"]["gate_minus_mopps_pp"] == pytest.approx(-10.) for row in report["states"])
        assert report["per_seed_mean_contrasts_pp"]["3"]["gate_minus_mopps_pp"] == pytest.approx(-10.)


QUEUE_WORKER = '''
import json, os, sys, time
from pathlib import Path
from types import SimpleNamespace
import mopps_comparison_gpu as m
sys.modules['additive_experiment'] = SimpleNamespace(model_environment=lambda c: {})
root = Path(sys.argv[1])
m.protocol = lambda _: {'parent': str(root / 'parent'), 'seeds': [3, 4], 'steps': [25, 50, 100], 'arms': list(m.mopps.ARMS)}
m.switch.admitted_devices = lambda _: list('0123')
m.status = lambda _: None
def run(out, p, arm, devices, env):
    directory = out / arm
    with (directory / 'claim.json').open('x') as handle:
        json.dump({'pid': os.getpid(), 'started': time.monotonic()}, handle)
    (root / f'running-{os.getpid()}').touch()
    deadline = time.monotonic() + 5
    while len(list(root.glob('running-*'))) < 4:
        if time.monotonic() > deadline: raise RuntimeError('four ready tasks did not overlap')
        time.sleep(.01)
    time.sleep(.2)
    m.core.atomic_json(directory / 'result.json', {'pid': os.getpid(), 'finished': time.monotonic()})
    m.core.atomic_json(directory / 'result.sha256.json', {'sha256': m.base.digest(directory / 'result.json')})
m.run_arm = run
(root / f'ready-{os.getpid()}').touch()
deadline = time.monotonic()+10
while not (root / 'go').exists():
    if time.monotonic() > deadline: raise RuntimeError('start timeout')
    time.sleep(.01)
sleep = time.sleep
m.time.sleep = lambda seconds: sleep(.01 if seconds == 15 else seconds)
raise SystemExit(m.work(root, idle_timeout=1))
'''


def test_four_nodes_run_twelve_branches_without_duplicate_claims(tmp_path):
    for seed in rule.TEST_SEEDS:
        for step in rule.STEPS:
            core.atomic_json(switch.prefix_dir(tmp_path / "parent", seed) / f"prefix-{step}.json", {})
            out = run.point(tmp_path, seed, step)
            core.atomic_json(out / "import.done.json", {})
            core.atomic_json(out / "contract.json", {"config": {}})
    workers = [subprocess.Popen([sys.executable, "-c", QUEUE_WORKER, str(tmp_path)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(4)]
    try:
        deadline = time.monotonic()+10
        while len(list(tmp_path.glob("ready-*"))) < 4 and time.monotonic() < deadline:
            time.sleep(.01)
        assert len(list(tmp_path.glob("ready-*"))) == 4
        (tmp_path / "go").touch()
        for worker in workers:
            stdout, stderr = worker.communicate(timeout=20)
            assert worker.returncode == 0, stdout+stderr
        claims = {path.parent: core.read(path) for path in tmp_path.glob("states/*/*/claim.json")}
        results = {path.parent: core.read(path) for path in tmp_path.glob("states/*/*/result.json")}
        assert len(claims) == len(results) == 12
        assert len({row["pid"] for row in claims.values()}) == 4
        assert max(sum(claims[path]["started"] <= instant < results[path]["finished"] for path in claims)
                   for instant in (row["started"] for row in claims.values())) == 4
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
                worker.communicate()
