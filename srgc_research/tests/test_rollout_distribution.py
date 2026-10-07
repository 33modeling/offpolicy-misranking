import copy
import json

import pytest

from srgc_research.rollout_distribution import (
    endpoint_report,
    group_report,
    main,
    trace_report,
)


def endpoint(seed=5, arm="sr", total=8000, zero=416):
    return {"seed": seed, "arm": arm, "reward": 0.4, "total_updates": 275,
            "shared_prefix_updates": 25, "input_sha256": "inputs", "prefix_checkpoint_sha256": "prefix",
            "cost_receipts": {"counts": {"training.responses": total,
                "training.zero_advantage_responses": zero, "training.generated_tokens": 8000000}}}


def test_receipts_do_not_fabricate_traces_and_isolate_bad_endpoints():
    export = {"dataset": "mbpp", "source_results": {
        "seed-5/sr-endpoint.json": endpoint(),
        "seed-5/random-endpoint.json": endpoint(arm="random", total=8960),
        "seed-6/sr-endpoint.json": endpoint(seed=7),
        "seed-5/switch_fixed200-endpoint.json": endpoint(arm="switch_fixed200"),
        "seed-5/replicate-1/sr-endpoint.json": endpoint()}}
    report = endpoint_report(export, responses_per_update=32)
    assert len(report["rows"]) == 2 and len(report["issues"]) == 1
    assert report["raw_trace_available"] is False and report["groups"] is None
    assert report["all_correct_vs_all_wrong_available"] is False
    assert report["aggregates"]["sr"]["zero_advantage_response_pct"]["mean"] == 5.2
    random = next(r for r in report["rows"] if r["arm"] == "random")
    assert random["expected_responses_assuming_fixed_workload"] == 8000
    assert random["receipt_matches_assumed_workload"] is False


@pytest.mark.parametrize("field,value", [
    ("training.responses", 0), ("training.responses", True),
    ("training.zero_advantage_responses", 8001),
    ("training.generated_tokens", float("nan")),
    ("training.zero_advantage_responses", -1)])
def test_corrupt_counters_cannot_be_normal_results(field, value):
    row = endpoint()
    row["cost_receipts"]["counts"][field] = value
    with pytest.raises(ValueError, match="no valid"):
        endpoint_report({"dataset": "math", "source_results": {"seed-5/sr-endpoint.json": row}})


def test_missing_identity_prevents_certifying_a_paired_comparison():
    a, b = endpoint(), endpoint(arm="switch")
    b.pop("prefix_checkpoint_sha256")
    report = endpoint_report({"dataset": "math_train", "source_results": {
        "seed-5/sr-endpoint.json": a, "seed-5/switch-endpoint.json": b}})
    assert report["paired"][0]["reported_prefix_and_input_match"] is False
    assert report["responses_per_update_assumption"] is None


def trace_group(rewards, arm="sr", phase="training", update=100, prompt="p0", draw=0):
    return {"arm": arm, "phase": phase, "update": update, "draw": draw, "prompt_id": prompt,
            "responses": [{"reward": r, "token_ids": [1, 2, i + 3]} for i, r in enumerate(rewards)]}


def trace(groups):
    return {"schema": "srgc-rollout-trace-v1", "dataset": "math", "model": "test-model",
            "seed": 5, "groups": groups}


def test_group_outcomes_and_training_vs_scoring_are_separate():
    groups = [trace_group([1] * 8), trace_group([0] * 8, prompt="p1"),
              trace_group([0, 1] * 4, prompt="p2"),
              trace_group([1] * 8, phase="selection")]
    report = trace_report(trace(groups))
    training = next(r for r in report["rows"] if r["phase"] == "training")
    assert training["mixed_group_fraction"] == 1 / 3
    assert training["all_correct_groups"] == training["all_wrong_groups"] == 1
    assert training["distinct_prompts"] == 3
    assert training["success_count_histogram"] == {"0/8": 1, "4/8": 1, "8/8": 1}
    assert training["response_token_length"]["mean"] == 3
    assert report["semantic_diversity_measured"] is False


def test_repeated_prompt_exposure_is_not_coverage():
    groups = [trace_group([0, 1], draw=i) for i in range(4)]
    row = trace_report(trace(groups))["rows"][0]
    assert row["distinct_prompts"] == row["effective_prompt_count"] == 1
    assert row["prompt_exposure_hhi"] == 1
    with pytest.raises(ValueError, match="duplicate"):
        trace_report(trace([groups[0], copy.deepcopy(groups[0])]))


def test_cumulative_coverage_separates_repeated_and_distinct_prompts():
    groups = [trace_group([0, 1], update=step) for step in (0, 1, 2)]
    groups.append(trace_group([0, 1], update=2, prompt="p1"))
    row = trace_report(trace(groups))["cumulative_prompt_coverage"][0]
    assert row["distinct_prompts"] == 2 and row["prompt_group_exposures"] == 4
    assert row["prompt_exposure_hhi"] == 0.625
    assert 1 < row["effective_prompt_count"] < 2


def test_missing_tokens_and_stop_reasons_stay_unmeasured():
    group = trace_group([0, 1])
    for response in group["responses"]:
        response.pop("token_ids")
    row = group_report(group)
    assert row["response_token_lengths"] is None
    assert row["unique_token_sequence_fraction"] is None
    assert row["truncated_response_fraction"] is None
    group["responses"][0]["token_ids"] = [1, 2]
    with pytest.raises(ValueError, match="partial token"):
        group_report(group)


def test_lexical_duplicates_and_real_stop_reasons():
    group = trace_group([0, 1])
    for response in group["responses"]:
        response["token_ids"] = [1, 2, 3]
    group["responses"][0]["finish_reason"] = "length"
    group["responses"][1]["finish_reason"] = "eos"
    row = group_report(group)
    assert row["unique_token_sequence_fraction"] == 0.5
    assert row["mean_token_bigram_jaccard_distance"] == 0
    assert row["truncated_response_fraction"] == 0.5


@pytest.mark.parametrize("reward", [0.5, float("nan"), True, -1])
def test_nonbinary_rewards_cannot_enter_group_statistics(reward):
    with pytest.raises(ValueError):
        group_report(trace_group([0, reward]))


def test_cli_preserves_inputs_and_existing_output(tmp_path):
    path, output = tmp_path / "input.json", tmp_path / "output.json"
    path.write_text(json.dumps(trace([trace_group([0, 1])])) + "\n")
    original = path.read_bytes()
    main(["traces", str(path), "--output", str(output)])
    assert path.read_bytes() == original
    result = json.loads(output.read_text())
    assert len(result["reports"][0]["source"]["sha256"]) == 64
    saved = output.read_bytes()
    with pytest.raises(SystemExit) as exc:
        main(["traces", str(path), "--output", str(output)])
    assert exc.value.code == 2 and output.read_bytes() == saved
