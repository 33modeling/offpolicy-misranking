import json

import pytest

import selection_gate as core
import selection_gate_gpu as base
from selector_pair_online_check_cost import check_cost, single_reference_checks, STAGES
from test_selector_pair import allocation


def add_check(directory, seed=3, step=25, *, b_open=False, retry=False, receipt_only=False, scale=1):
    core.atomic_json(directory / "reference.json", {"config": {"seed": seed, "drift": step}})
    reference_sha = base.digest(directory / "reference.json")
    prefix = "sr-gc-" if step == 25 else "sr-gc-repeat-"
    for index, stage in enumerate(STAGES):
        if retry and index == 0:
            for event in allocation("retry", prefix + stage, 0, 5, rc=1):
                base.journal(directory / "cost.jsonl", event)
        events = allocation(stage, prefix + stage, 100 * index + 10, (10 + index * 20) * scale)
        for event in events:
            if receipt_only and index == 1 and event["state"] == "finished":
                core.atomic_json(directory / "cost-events" / f"{event['event_id']}.json", event)
            else:
                base.journal(directory / "cost.jsonl", event)
        for shard in range(4):
            payload = directory / f"{stage}-{shard}.json"
            core.atomic_json(payload, {"fixture": [1, 2, 3]})
            core.atomic_json(directory / f"{stage}-{shard}.done.json", {
                "reference_sha256": reference_sha, "stage": stage, "shard": shard,
                "sha256": base.digest(payload)})
    for stage, seconds in (("validation-b", 1000), ("candidate-b", 900), ("aggregate", 10), ("r-candidate", 3000)):
        events = allocation(stage, prefix + stage, 1000, seconds)
        for event in (events[:1] if b_open and stage == "candidate-b" else events):
            base.journal(directory / "cost.jsonl", event)


@pytest.mark.parametrize("step", [25, 50])
@pytest.mark.parametrize("b_open", [False, True])
def test_only_a_is_charged_even_with_asymmetric_or_unclosed_b(tmp_path, step, b_open):
    add_check(tmp_path, step=step, b_open=b_open)
    result = check_cost(tmp_path, 3, step)
    assert result["complete"] and result["gpu_seconds"] == (10 + 30) * 4
    assert result["reference"] == "A" and not result["issues"]
    assert len(result["projection_receipts"]) == 8
    assert all(event["phase"].endswith(("validation-a", "candidate-a")) for event in result["raw_a_events"])
    assert len(result["raw_a_events"]) == 4


def test_failed_a_attempt_and_atomic_finish_receipt_are_retained_read_only(tmp_path):
    add_check(tmp_path, retry=True, receipt_only=True)
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    result = check_cost(tmp_path, 3, 25)
    assert result["complete"] and result["gpu_seconds"] == (5 + 10 + 30) * 4
    assert result["phases"][0]["failed_events"] == 1
    assert len(result["receipt_recoveries"]) == 1
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


@pytest.mark.parametrize("change", ["unclosed_a", "missing_start", "missing_phase", "cost", "gpu", "reference", "projection", "binding"])
def test_incomplete_or_inconsistent_a_is_not_replaced_with_half_of_ab(tmp_path, change):
    add_check(tmp_path)
    path = tmp_path / "cost.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    if change == "unclosed_a":
        del events[3]
    elif change == "missing_start":
        del events[2]
    elif change == "missing_phase":
        del events[2:4]
    elif change == "cost":
        events[3]["allocated_gpu_seconds"] += 1
    elif change == "gpu":
        events[2]["gpus"] = events[3]["gpus"] = 1
        events[3]["allocated_gpu_seconds"] = events[3]["seconds"]
    elif change == "reference":
        core.atomic_json(tmp_path / "reference.json", {"config": {"seed": 4, "drift": 25}})
    elif change == "projection":
        (tmp_path / "candidate-a-0.json").unlink()
    else:
        core.atomic_json(tmp_path / "candidate-a-0.done.json", {"sha256": "wrong"})
    path.write_text("".join(json.dumps(event) + "\n" for event in events))
    result = check_cost(tmp_path, 3, 25)
    assert result["gpu_seconds"] is None and not result["complete"] and result["issues"]


def test_all_a_checkpoints_are_exported_but_only_pre_switch_work_is_charged(tmp_path):
    for step in range(25, 276, 25):
        directory = (tmp_path / "sr-gc/s3-t25" if step == 25 else
                     tmp_path / f"sr-gc-repeat/every-25/s3-t25/step-{step}")
        add_check(directory, step=step, scale=1 if step <= 125 else 100)
    result = single_reference_checks(tmp_path, 3, 125)
    assert result["complete"] and result["gpu_seconds"] == 5 * 160
    assert len(result["checks"]) == 11
    assert [c["step"] for c in result["checks"] if c["included_before_switch"]] == [25, 50, 75, 100, 125]
    assert all(c["reference"] == "A" for c in result["checks"])


def test_missing_a_checkpoint_keeps_known_subtotal_without_filling_it(tmp_path):
    add_check(tmp_path / "sr-gc/s3-t25")
    result = single_reference_checks(tmp_path, 3, 50, through=50)
    assert result["gpu_seconds"] is None and not result["complete"]
    assert result["known_gpu_seconds"] == 160 and result["unknown_steps"] == [50]
