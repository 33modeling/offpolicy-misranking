"""Recover existing measured costs through results, without touching learners."""

import copy

from scripts import srgc_qwen35 as qwen
from srgc_rebuttal.cost_ledger import PhaseLedger
from srgc_rebuttal.runtime import code_digest
from srgc_research.dispatch import model_results
from srgc_research.tests.test_model_results import cohort, finish, read_result, saved


def test_all_model_exports_retain_real_stage_receipts_and_counts(cohort):
    folder = finish(cohort, "math")
    event = {"id": "measured", "phase": "training", "checkpoint": 25, "state": "finished",
             "wall_seconds": 3., "gpu_count": 4, "gpu_seconds": 12.,
             "stages": {"generation": {"gpu_seconds": 8., "rank_wall_seconds": 8., "calls": 4},
                        "unattributed_and_wait": {"gpu_seconds": 4., "rank_wall_seconds": 4., "calls": 4}},
             "counts": {"prompts": 4, "responses": 32, "generated_tokens": 1024}}
    ledger = PhaseLedger(folder / "cost-receipts/on_policy")
    ledger.record({key: event[key] for key in ("id", "phase", "checkpoint", "gpu_count")} | {"state": "started"})
    ledger.record(event)
    original = ledger.totals()
    before = saved(cohort)
    engine_before = code_digest(), qwen.adapter_digest(), qwen.engine_digest()
    assert model_results.export(cohort.model, "math", environment=cohort.env) == 0
    result = read_result(cohort)
    cost = next(row for row in result["datasets"]["math"]["costs"] if row["seed"] == 5)["arms"]["on_policy"]
    assert cost["exclusive_stages"] == original["exclusive_stages"]
    assert cost["counts"] == original["counts"]
    assert cost["exclusive_stages"]["training.generation"]["gpu_seconds"] == 8.
    assert cost["counts"]["training.generated_tokens"] == 1024
    # A stage measurement does not complete missing invocation/endpoint costs.
    assert cost["complete"] is False
    assert cost["selection_training_preparation_gpu_seconds"] is None
    assert saved(cohort) == before
    assert engine_before == (code_digest(), qwen.adapter_digest(), qwen.engine_digest())
    plan = result["datasets"]["math"]["plan"]
    assert (plan["generation_micro_batch"], plan["logprob_micro_batch"], plan["logit_chunk_tokens"]) == (2, 1, 64)
    assert plan["attention"] == "eager"


def test_shared_ledgers_and_endpoint_settings_survive_compaction():
    stages = {"selection.backward": {"gpu_seconds": 7., "calls": 1}}
    counts = {"selection.backward_responses": 8}
    ledger = {"complete": False, "total_gpu_seconds": None,
              "known_gpu_seconds": {"selection_gpu_seconds": 7.},
              "exclusive_stages": stages, "counts": counts, "recorded_phases": 1}
    original = {"datasets": {"math": {"tasks": [], "plan": {}, "costs": [{"seed": 5, "arms": {},
        "shared_prefix": ledger, "shared_all_task_startup_and_load": ledger}], "endpoints": [
            {"seed": 5, "arm": "sr", "endpoint": {"checkpoint_policy": {"attention": "sdpa"}}}]}}}
    before = copy.deepcopy(original)
    result = model_results.compact(original)
    cost = result["datasets"]["math"]["costs"][0]
    for key in ("shared_prefix", "shared_all_task_startup_and_load"):
        assert cost[key]["exclusive_stages"] == stages
        assert cost[key]["counts"] == counts
        assert cost[key]["total_gpu_seconds"] is None
    assert result["datasets"]["math"]["endpoints"][0]["endpoint"]["checkpoint_policy"] == {"attention": "sdpa"}
    assert model_results.compact(result) == result
    assert original == before


def test_unrecorded_stages_and_runtime_settings_are_not_invented():
    original = {"datasets": {"math": {"tasks": [], "costs": [{"seed": 5, "arms": {
        "sr": {"complete": False, "training_gpu_seconds": None}}}]}}}
    result = model_results.compact(original)["datasets"]["math"]
    assert "generation_micro_batch" not in result["plan"]
    assert "exclusive_stages" not in result["costs"][0]["arms"]["sr"]
    assert "counts" not in result["costs"][0]["arms"]["sr"]
    assert result["costs"][0]["arms"]["sr"]["training_gpu_seconds"] is None
