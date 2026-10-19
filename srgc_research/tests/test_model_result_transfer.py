"""Transfer exports stay small and still permit independent restart accounting."""

import copy
import json
import math
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from srgc_research.dispatch import model_results
from srgc_research.dispatch.model_result_transfer import SCHEMA, compact, records
from srgc_research.dispatch.restart_costs import reconcile
from srgc_research.dispatch.result_export import save_result
from srgc_research.dispatch.submission_time import read_json
from srgc_research.tests.test_model_results import (
    cohort as cohort,  # noqa: PLC0414 -- pytest fixture registration
)
from srgc_research.tests.test_model_results import read_result, saved
from srgc_research.tests.test_restart_costs import event, fixture
from srgc_research.tests.test_rollout_records import export_records


@pytest.mark.parametrize("unfinished", [False, True])
def test_transfer_receipts_reproduce_full_restart_audit_without_rounding(tmp_path, unfinished):
    fixture(tmp_path, step=275)
    # Fractional seconds must survive JSON serialization without rounding.
    event(tmp_path, "random", "replayed", 25, 18.1234567890123)
    directory = tmp_path / "cost-receipts/random"
    (directory / "copy.json").write_bytes((directory / "training-25.json").read_bytes())
    if unfinished:
        event(tmp_path, "random", "interrupted", 26, None, state="started")
    history = read_json(tmp_path / "random-progress.json")["history"]
    points = [{"step": row["completed_updates"], "checkpoint": row["checkpoint"],
               "selection_gpu_seconds": row["selection_gpu_seconds"],
               "training_gpu_seconds": row["training_gpu_seconds"]} for row in history]
    audit = reconcile(tmp_path, "random", points, start=25, stop=275, read=read_json, expected_gpus=4)
    original = {"datasets": {"math": {"tasks": [{"seed": 5, "arm": "random", "submission_timing": {
        "update_timers": points, "restart_accounting": audit}}]}}}
    before = copy.deepcopy(original)
    artifact = json.loads(json.dumps(compact(original), allow_nan=False))
    measured = artifact["datasets"]["math"]["tasks"][0]["submission_timing"]
    transferred = records(measured["restart_accounting"]["source_receipts"])
    replay = tmp_path / "reconstructed"
    for row in transferred:
        path = replay / "cost-receipts/random" / row.pop("source_file")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(row))
    rebuilt = reconcile(replay, "random", records(measured["update_timers"]),
                        start=25, stop=275, read=read_json, expected_gpus=4)
    for key in ("record", "recorded_before_gpu_seconds", "removed_gpu_seconds", "retained_gpu_seconds",
                "unfinished_receipts", "duplicate_receipt_copies", "removal_is_lower_bound"):
        assert rebuilt[key] == audit[key]
    assert rebuilt["removed_gpu_seconds"] == 22.1234567890123
    assert original == before
    assert compact(artifact) == artifact


@pytest.mark.parametrize("flags", [[], ["--json"]])
def test_public_default_is_small_but_explicit_full_export_keeps_details(cohort, capsys, flags):
    export_records(cohort)
    before = saved(cohort)
    args = [cohort.model, "all", "results", "--root", str(cohort.root), *flags]
    with patch.dict(os.environ, cohort.env, clear=True):
        assert model_results.main(args) == 0
    output = capsys.readouterr().out
    artifact = read_result(cohort)
    assert artifact["transfer_schema"] == SCHEMA
    assert "MATH FINAL REWARD" in output and "MBPP GPU TIME" in output and "JSON:" in output
    curve = next(row for row in artifact["datasets"]["math"]["reward_curves"] if row["arm"] == "on_policy")
    assert "rollout_groups" not in curve and "rollout_sources" not in curve
    assert curve["detail_records"]["rollout_groups"] == 4
    assert curve["training_rewards"][0]["reward"] == .5
    assert curve["training_rewards"][0]["kind"] == "selected_training_batch"
    assert curve["held_out_rewards"][0]["kind"] == "fixed_held_out_evaluation"
    assert curve["held_out_rewards"][0]["step"] == 275
    destination = cohort.work / "results" / f"{cohort.model}-results.json"
    small_bytes = destination.read_bytes()
    with patch.dict(os.environ, cohort.env, clear=True):
        assert model_results.main([*args, "--full"]) == 0
    full_path = cohort.work / "results" / f"{cohort.model}-full-results.json"
    detailed = json.loads(full_path.read_text())
    full_curve = next(row for row in detailed["datasets"]["math"]["reward_curves"] if row["arm"] == "on_policy")
    assert len(full_curve["rollout_groups"]) == 4
    assert f"JSON: {full_path}" in capsys.readouterr().out
    assert destination.read_bytes() == small_bytes
    assert saved(cohort) == before
    with patch.dict(os.environ, cohort.env, clear=True):
        assert model_results.publish(cohort.model, SimpleNamespace(plan_path=cohort.plans["math"])) == 0
    assert read_result(cohort)["transfer_schema"] == SCHEMA
    assert saved(cohort) == before


def production_result():
    """Two datasets, five seeds, four arms, 250 updates, responses and retries."""
    result = {"format": "compact", "model": "qwen35", "datasets": {}}
    for dataset in ("math", "mbpp"):
        snapshot = {"tasks": [], "reward_curves": []}
        for seed in range(5, 10):
            for arm in ("random", "sr", "on_policy", "switch"):
                points, sources, retained = [], [], []
                root = f"/group-volume/minsoo3.kim/offpolicy-misranking/srgc-rebuttal/qwen35/runs/{dataset}/seed-{seed}"
                for step in range(26, 276):
                    identity = f"{step:032x}"
                    seconds = 100.12345678901234 + step
                    points.append({"step": step, "checkpoint": step - 1,
                                   "selection_gpu_seconds": 0., "training_gpu_seconds": seconds,
                                   "timing_receipt_ids": {"training": identity}})
                    sources.append({"id": identity, "phase": "training", "checkpoint": step - 1,
                                    "gpu_count": 4, "state": "finished", "wall_seconds": seconds / 4,
                                    "gpu_seconds": seconds, "source_file": f"{root}/cost-receipts/{arm}/{identity}.json",
                                    "stages": {"generation": {"gpu_seconds": seconds / 2, "calls": 4}},
                                    "counts": {"prompts": 4, "responses": 32}})
                    retained.append({"step": step, "phase": "training", "receipt_id": identity,
                                     "gpu_seconds": seconds, "matching_receipts": 1})
                sources += [{**sources[0], "id": "discarded", "source_file": f"{root}/cost-receipts/{arm}/discarded.json"}]
                total = math.fsum(p["training_gpu_seconds"] for p in points)
                snapshot["tasks"].append({"seed": seed, "arm": arm, "submission_timing": {
                    "record": "COMPLETE", "update_timers": points, "gpu_seconds": {"primary": total},
                    "restart_accounting": {"record": "COMPLETE", "source_receipts": sources,
                        "source_files": [r["source_file"] for r in sources], "retained_receipts": retained,
                        "discarded_completed_attempts": [{"id": "discarded", "phase": "training", "checkpoint": 25,
                                                          "gpu_seconds": sources[0]["gpu_seconds"]}],
                        "recorded_before_gpu_seconds": total + sources[0]["gpu_seconds"],
                        "retained_gpu_seconds": total, "removed_gpu_seconds": sources[0]["gpu_seconds"]}}})
                snapshot["reward_curves"].append({"seed": seed, "arm": arm,
                    "training_rewards": [{"step": p["step"], "reward": .5, "reward_percent": 50.,
                                          "selector": arm, "kind": "selected_training_batch"} for p in points],
                    "training_windows_25": [{"end_step": step, "reward": .5} for step in range(50, 276, 25)],
                    "held_out_rewards": [{"step": step, "reward": .75, "kind": "fixed_held_out_evaluation"}
                                         for step in range(25, 276, 25)],
                    "rollout_groups": [{"update": step, "prompt_id": prompt,
                        "responses": [{"reward": .5, "completion_tokens": 2048, "response_sha256": "a" * 64}
                                      for _ in range(8)]} for step in range(26, 276) for prompt in range(4)],
                    "rollout_sources": [{"path": f"{root}/reward-curves/{arm}/rollouts/step-{step}.json.gz",
                                         "sha256": "b" * 64} for step in range(26, 276)]})
        result["datasets"][dataset] = snapshot
    return result


def test_production_sized_matrix_stays_small_with_all_update_and_restart_evidence(tmp_path):
    original = production_result()
    artifact = compact(original)
    destination = save_result("qwen35", artifact, work=tmp_path, quiet=True, dense=True)
    byte_count = destination.stat().st_size
    assert byte_count < 4 * 1024 * 1024
    # Compare with the previous serializer, including its response detail payload.
    full_bytes = len(json.dumps(original, indent=2, allow_nan=False).encode())
    assert byte_count < full_bytes / 20
    decoded = json.loads(destination.read_bytes())
    for dataset, snapshot in decoded["datasets"].items():
        for task, source in zip(snapshot["tasks"], original["datasets"][dataset]["tasks"], strict=True):
            measured = task["submission_timing"]
            assert records(measured["update_timers"]) == source["submission_timing"]["update_timers"]
            assert len(records(measured["restart_accounting"]["source_receipts"])) == 251
        assert all(len(curve["training_rewards"]) == 250 for curve in snapshot["reward_curves"])
        assert all(len(curve["held_out_rewards"]) == 11 for curve in snapshot["reward_curves"])
    print(f"Full matrix: {full_bytes} -> {byte_count} bytes")
