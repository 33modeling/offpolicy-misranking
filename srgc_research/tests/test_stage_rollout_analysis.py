"""Only measured quantities enter the CPU reports; missing curves stay missing."""

import copy
import hashlib
import json

import pytest

from srgc_research.stage_rollout_analysis import Analysis, main

SOURCE = {"path": "fixture.json", "sha256": "e" * 64}
IDENTITY = {"plan_sha256": "a" * 64, "input_sha256": "b" * 64, "implementation_sha256": "c" * 64}


def group(successes, *, step=26, prompt="p", arm="switch", tokens=True):
    return {"arm": arm, "phase": "training", "update": step, "prompt_id": prompt,
            "cached_success_rate": .5, "responses": [
                {"reward": int(i < successes), **({"token_ids": [10, i]} if tokens else {})} for i in range(8)]}


def trace(groups):
    return {"schema": "srgc-rollout-trace-v1", "model": "fixture", "dataset": "math", "seed": 5,
            "arm": "switch", "switched_at": 25, "identity": IDENTITY, "total_updates": 275, "groups": groups}


def test_legacy_final_only_cannot_create_a_curve_or_difficulty_histogram():
    endpoint = {**IDENTITY, "seed": 5, "arm": "switch", "total_updates": 275,
                "switched_at": 50, "reward": .8, "checks": [{"step": 25, "d": -.1}]}
    document = {"dataset": "math", "source_results": {"seed-5/switch-endpoint.json": endpoint,
                "replicates/repeat/seed-5/switch-endpoint.json": {**endpoint, "reward": .1}}}
    analysis = Analysis(model="declared-fixture")
    analysis.document(document, SOURCE)
    report = analysis.result()
    assert not report["groups"] and not report["windows_25"]
    assert len(report["reward_points"]) == 1
    assert report["reward_points"][0]["kind"] == "held_out"
    row = report["coverage"][0]
    assert row["recorded_training_reward_steps"] == []
    assert row["recorded_held_out_steps"] == [275]
    assert row["missing_held_out_steps"] == list(range(25, 275, 25))
    assert row["switch_arm_first_sr_update"] == 51
    assert report["issues"] and report["switch_checks"][0]["d"] == -.1


def test_zero_to_eight_counts_cache_gap_and_unmeasured_tokens():
    analysis = Analysis()
    analysis.document(trace([group(0, prompt="hard"), group(4, prompt="mixed"), group(8, prompt="easy", tokens=False)]), SOURCE)
    report = analysis.result()
    row = report["windows_25"][0]
    assert (row["window_start"], row["window_end"]) == (26, 50)
    assert row["success_count_histogram"] == {str(i): int(i in (0, 4, 8)) for i in range(9)}
    assert row["mixed_group_fraction"] == pytest.approx(1 / 3)
    assert row["cache_gap"]["mean"] == 0
    assert row["absolute_cache_gap"]["mean"] == pytest.approx(1 / 3)
    assert row["token_length_coverage_groups"] == 2
    assert report["groups"][2]["response_token_lengths"] is None
    assert all(g["updates_relative_to_switch"] == 1 and g["switch_period"] == "post_switch" for g in report["groups"])


def test_training_and_held_out_are_separate_and_shared_prefix_deduplicated():
    prefix = {**group(4, step=25, arm="prefix"), "shared_prefix": True}
    snapshot = {"model": "fixture", "dataset": "math", "plan": {"total_updates": 275},
                "endpoints": [], "reward_curves": []}
    for arm in ("switch", "sr"):
        endpoint = {**IDENTITY, "seed": 5, "arm": arm, "total_updates": 275, "reward": .8}
        snapshot["endpoints"].append({"seed": 5, "arm": arm, "endpoint": endpoint})
        snapshot["reward_curves"].append({"seed": 5, "arm": arm, "rollout_groups": [prefix],
            "training_rewards": [{"step": 25, "reward": .5, "shared_prefix": True}, {"step": 26, "reward": .25}],
            "held_out_rewards": [{"step": 25, "reward": .75, "shared_prefix": True}]})
    analysis = Analysis()
    analysis.document(snapshot, SOURCE)
    report = analysis.result()
    assert report["shared_prefix_duplicates_removed"] == 1
    assert len(report["groups"]) == 1
    assert {(p["kind"], p["reward"]) for p in report["reward_points"] if p["arm"] == "prefix"} == {("training", .5), ("held_out", .75)}


def test_mechanism_training_is_branch_local_and_tokens_are_not_invented():
    analysis = Analysis(model="fixture", dataset="mbpp")
    analysis.document({"seed": 5, "arm": "stage_mechanism", "measurements": [
        {"stage": 100, "A": {"rewards": {"p": [0, 1] * 4}}, "B": {"rewards": {"p": [1] * 8}},
         "selected": {"sr": ["p"]}, "cached_success_rates": {"p": .5}}],
        "training_records": [{"stage": 100, "mode": "sr", "update": 1, "rewards": {"p": [0] * 8}}],
        "rows": [{"stage": 100, "mode": "sr", "updates": 25, "baseline_reward": .5, "reward": .6}]}, SOURCE)
    report = analysis.result()
    row = next(g for g in report["groups"] if g["phase"] == "training")
    assert row["anchor_step"] == 100 and row["update"] == 1
    assert row["response_token_lengths"] is None
    assert {p["step"] for p in report["reward_points"]} == {0, 25}
    assert not report["interpretation"]["mechanism_branch_updates_are_global_updates"]


@pytest.mark.parametrize("damage", ["duplicate", "reward", "partial_tokens", "step_zero"])
def test_invalid_groups_fail_instead_of_silently_changing_denominators(damage):
    document = trace([group(4)])
    if damage == "duplicate":
        document["groups"].append(copy.deepcopy(document["groups"][0]))
    elif damage == "reward":
        document["groups"][0]["responses"][0]["reward"] = .4
    elif damage == "partial_tokens":
        del document["groups"][0]["responses"][0]["token_ids"]
    else:
        document["groups"][0]["update"] = 0
    with pytest.raises(ValueError):
        Analysis().document(document, SOURCE)


def test_cli_preserves_inputs_and_refuses_overwrite(tmp_path):
    path, output = tmp_path / "trace.json", tmp_path / "report"
    path.write_text(json.dumps(trace([group(4)])))
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    main([str(path), "--output-dir", str(output)])
    report = json.loads((output / "analysis.json").read_text())
    assert report["sources"][0]["sha256"] == before
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    with pytest.raises(SystemExit) as error:
        main([str(path), "--output-dir", str(output)])
    assert error.value.code == 2
    assert (output / "groups.csv").exists() and (output / "held_out_rewards.csv").exists()
