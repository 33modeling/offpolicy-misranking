"""Status reads measured stage costs directly, consistently across model runners."""

import copy
import importlib
import json
import os
import sys
from unittest.mock import patch

import pytest

from scripts.srgc_time_status import attach, from_arm, from_ledger, invalid, render
from srgc_rebuttal.cost_ledger import PhaseLedger
from srgc_research.tests.test_model_results import (
    cohort as cohort,  # noqa: PLC0414 -- re-export pytest fixture
)
from srgc_research.tests.test_model_results import finish, saved
from srgc_research.tests.test_model_status import RUNNERS, read_status


def record(directory, phase, stages, counts=None):
    ledger = PhaseLedger(directory)
    start = {"id": phase, "phase": phase, "checkpoint": 25, "gpu_count": 4, "state": "started"}
    ledger.record(start)
    seconds = sum(stages.values())
    ledger.record({**start, "state": "finished", "gpu_seconds": seconds, "wall_seconds": seconds / 4,
                   "stages": {name: {"gpu_seconds": value, "rank_wall_seconds": value, "calls": 4}
                              for name, value in stages.items()}, "counts": counts or {}})
    return ledger.totals()


def test_partial_stages_and_work_keep_exact_units_and_phase_sections(tmp_path):
    ledger = record(tmp_path, "selection", {"candidate.generation": 3600, "reference.forward": 1800,
                                          "gradient_projection": 1800},
                    {"candidate.responses": 320, "reference.responses": 400})
    before = copy.deepcopy(ledger)
    value = from_ledger(ledger, completed=False)
    assert value["record"] == "PARTIAL"
    assert value["exclusive_stages"] == before["exclusive_stages"]
    assert value["counts"] == before["counts"]
    assert value["exclusive_stage_gpu_hours"]["selection.candidate.generation"] == 1
    assert value["gpu_hours"]["primary"] == 2
    text = "\n".join(render({"tasks": [{"seed": 5, "arm": "switch", "timing": value}]}))
    assert "selection.candidate" in text and "selection.reference" in text
    assert ">=1.000" in text and ">=320" in text and ">=400" in text
    assert "mean +/-" not in text
    value["exclusive_stages"]["selection.candidate.generation"]["gpu_seconds"] = 0
    assert ledger == before


def test_legacy_complete_costs_do_not_invent_stage_or_runtime_measurements():
    report = attach({"output_root": "/unused", "tasks": []}, "/unused/plan.json", {"seeds": []})
    assert report["recorded_execution_settings"] == {}
    value = from_arm({"complete": True, "training_gpu_seconds": 3600,
                      "selection_training_preparation_gpu_seconds": 3600}, completed=True)
    assert "exclusive_stages" not in value and "counts" not in value
    text = "\n".join(render({**report, "tasks": [{"seed": 5, "arm": "sr", "timing": value}]}))
    assert "GPU STAGES" not in text and "RECORDED WORK" not in text
    assert "RECORDED PLAN SETTINGS" not in text


def test_recorded_zero_stage_is_displayed_and_invalid_details_are_masked(tmp_path):
    ledger = record(tmp_path, "training", {"generation": 0}, {"responses": 0})
    row = {"seed": 5, "arm": "sr", "timing": from_ledger(ledger, completed=True)}
    text = "\n".join(render({"tasks": [row]}))
    assert "GPU STAGES" in text and "0.000" in text and "RECORDED WORK" in text
    row["status"] = "invalid"
    text = "\n".join(render({"tasks": [row]}))
    assert "INVALID" in text and "GPU STAGES" not in text and "RECORDED WORK" not in text
    invalid(row)
    assert "exclusive_stages" not in row["timing"] and "counts" not in row["timing"]


@pytest.mark.parametrize("field,bad", [
    ("exclusive_stages", {"training.generation": {"gpu_seconds": float("nan")}}),
    ("exclusive_stages", {"training.generation": {"gpu_seconds": -1}}),
    ("exclusive_stages", {"training.generation": {"gpu_seconds": 1, "calls": -1}}),
    ("counts", {"training.responses": -1}),
    ("counts", {"training.responses": True}),
    ("counts", {"responses": 32}),
])
def test_corrupt_detail_is_rejected_without_inventing_a_valid_timing(field, bad):
    with pytest.raises((TypeError, ValueError)):
        from_arm({"complete": True, field: bad}, completed=True)


def test_public_status_reads_existing_receipts_without_results_or_model_packages(cohort, capsys):
    originals = {}
    for dataset in ("math", "mbpp"):
        folder = finish(cohort, dataset)
        receipts = folder / "cost-receipts/on_policy"
        record(receipts, "training", {"generation": 7200, "backward": 3600, "unattributed_and_wait": 3600},
               {"prompts": 4, "responses": 32, "generated_tokens": 1024, "backward_responses": 32})
        record(receipts, "selection", {"candidate.generation": 3600, "reference.forward": 1800,
                                       "gradient_projection": 1800},
               {"candidate.responses": 320, "reference.responses": 400})
        originals[dataset] = record(receipts, "evaluation", {"generation": 3600},
                                    {"prompts": 300, "responses": 2400})
        if dataset == "math":
            endpoint_path = folder / "on_policy-endpoint.json"
            endpoint = json.loads(endpoint_path.read_text())
            endpoint["cost_measurement_complete"] = True
            endpoint_path.write_text(json.dumps(endpoint))
            record(folder / "invocations/on_policy", "session", {"unattributed_and_wait": 28800})
    before = saved(cohort)
    identities = cohort.adapter.engine_digest(), cohort.adapter.adapter_digest()
    code, data = read_status(cohort, capsys)
    assert code == 0
    for dataset, report in zip(("math", "mbpp"), data["datasets"], strict=True):
        settings = report["recorded_execution_settings"]
        assert (settings["generation_micro_batch"], settings["logprob_micro_batch"], settings["logit_chunk_tokens"]) == (2, 1, 64)
        assert settings["attention"] == "eager"
        row = next(row for row in report["tasks"] if row["task"] == "seed-5.on_policy")
        value = row["timing"]
        assert value["record"] == ("COMPLETE" if dataset == "math" else "PARTIAL")
        assert value["exclusive_stages"] == originals[dataset]["exclusive_stages"]
        assert value["counts"] == originals[dataset]["counts"]
        assert value["exclusive_stage_gpu_hours"]["training.generation"] == 2
        assert value["gpu_hours"]["primary"] == 6
        assert value["gpu_hours"]["phases"] == 7
        assert value["counts"]["training.responses"] == 32
        assert value["counts"]["evaluation.responses"] == 2400
    runner = importlib.import_module("srgc_research.dispatch." + RUNNERS[cohort.model])
    with patch.dict(os.environ, cohort.env, clear=True), \
            patch.dict(sys.modules, {"torch": None, "transformers": None, "peft": None}):
        assert runner.main(["all", "status", "--root", str(cohort.root)]) == 0
    text = capsys.readouterr().out
    assert text.count("GPU STAGES") == 2 and text.count("RECORDED WORK") == 2
    assert "selection.candidate" in text and "selection.reference" in text
    assert "generation_micro_batch=2" in text and ">=2.000" in text
    assert "not measured GPU idle time" in text
    assert saved(cohort) == before and not (cohort.work / "results").exists()
    assert identities == (cohort.adapter.engine_digest(), cohort.adapter.adapter_digest())
