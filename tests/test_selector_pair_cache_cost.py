import json
from datetime import datetime, timedelta

import pytest

from selector_pair_cache_cost import cache_creation_cost, digest


def source(tmp_path):
    run = tmp_path / "origin"
    (run / "logs").mkdir(parents=True)
    (run / "rollouts_behavior_train.jsonl").write_text("cached responses\n")
    (run / "run_config.json").write_text(json.dumps({"n_train": 400, "behavior_k": 8}))
    (run / "logs/main.log").write_text(
        "[2026-09-01 00:00:00] [progress] run  2/8 behavior-rollout 400x8 on 4 GPUs  +0min\n"
        "[2026-09-01 00:10:00] [progress] run  3/8 grpo skipped (d0 base policy)  +10min\n")
    return run, {"source_run": str(run), "source_hashes": {
        "rollouts_behavior_train.jsonl": digest(run / "rollouts_behavior_train.jsonl")}}


def test_stage_time_and_gpu_count_are_taken_from_the_original_log(tmp_path):
    run, contract = source(tmp_path)
    result = cache_creation_cost(contract)
    assert result["gpu_seconds"] == 600 * 4
    assert result["status"] == "reconstructed_stage_allocation"
    assert result["log_sha256"] == digest(run / "logs/main.log")


def test_reuse_is_followed_back_to_hash_matched_generation(tmp_path):
    run, contract = source(tmp_path)
    reused = tmp_path / "reused"
    reused.mkdir()
    (reused / "rollouts_behavior_train.jsonl").write_bytes((run / "rollouts_behavior_train.jsonl").read_bytes())
    (reused / "run_config.json").write_text(json.dumps({"behavior_source": str(run)}))
    contract["source_run"] = str(reused)
    assert cache_creation_cost(contract)["gpu_seconds"] == 2400
    (run / "rollouts_behavior_train.jsonl").write_text("different cache\n")
    assert cache_creation_cost(contract)["gpu_seconds"] is None


@pytest.mark.parametrize("intermediate", [False, True])
def test_selected_prefix_view_follows_cache_symlink_without_view_logs(tmp_path, intermediate):
    run, contract = source(tmp_path)
    origin = run
    if intermediate:
        origin = tmp_path / "reused"
        origin.mkdir()
        (origin / "rollouts_behavior_train.jsonl").write_bytes((run / "rollouts_behavior_train.jsonl").read_bytes())
        (origin / "run_config.json").write_text(json.dumps({"behavior_source": str(run)}))
    view = tmp_path / "branches/on_policy/prefixes/seed-3/view-25"
    view.mkdir(parents=True)
    (view / "rollouts_behavior_train.jsonl").symlink_to(origin / "rollouts_behavior_train.jsonl")
    (view / "run_config.json").write_text(json.dumps({"n_train": 400, "behavior_k": 8, "drift": 25}))
    contract["source_run"] = str(view)
    result = cache_creation_cost(contract)
    assert result["gpu_seconds"] == 2400
    assert result["source_run"] == str(run)
    assert result["log_path"] == str(run / "logs/main.log")
    assert result["source_trace"][0] == {"from": str(view), "to": str(origin), "via": "cache_symlink"}
    assert len(result["source_trace"]) == 1 + int(intermediate)
    assert len(result["progress_records"]) == 2


def test_selected_prefix_symlink_still_requires_the_contract_cache_hash(tmp_path):
    run, contract = source(tmp_path)
    view = tmp_path / "view-25"
    view.mkdir()
    (view / "rollouts_behavior_train.jsonl").symlink_to(run / "rollouts_behavior_train.jsonl")
    contract["source_run"] = str(view)
    contract["source_hashes"]["rollouts_behavior_train.jsonl"] = "wrong"
    result = cache_creation_cost(contract)
    assert result["gpu_seconds"] is None
    assert "differs from the experiment contract" in result["reason"]


def legacy_log(run, *, minute=0, failed=None):
    time = datetime(2026, 9, 1) + timedelta(minutes=minute)
    def stamp(seconds):
        return (time + timedelta(seconds=seconds)).strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"[{stamp(0)}] === RLVR point start: model -> {run} (4 GPUs) ==="]
    commands = [f"--stage rollout-behavior --run {run} --n-train 400 --behavior-k 8 --shard {i}:4"
                for i in range(4)]
    lines += [f"[{stamp(i)}] GPU{i} \u25b6 {command}" for i, command in enumerate(commands)]
    for i, command in enumerate(commands):
        seconds = 100 + i * 10
        end = f"\u2718 {command} rc=1" if failed == i else f"\u2714 --stage rollout-behavior ({seconds - i}s)"
        lines.append(f"[{stamp(seconds)}] GPU{i} {end}")
    return "\n".join(lines) + "\n"


def test_legacy_gpu_records_charge_the_full_parallel_stage_once(tmp_path):
    run, contract = source(tmp_path)
    (run / "logs/main.log").write_text(legacy_log(run))
    result = cache_creation_cost(contract)
    assert result["status"] == "reconstructed_legacy_stage_allocation"
    assert result["gpu_seconds"] == 130 * 4
    assert len(result["attempts"]) == 1
    assert len(result["attempts"][0]["workers"]) == 4
    assert len(result["raw_timing_records"]) == 9


def test_legacy_closed_failed_attempt_is_retained_but_idle_gap_is_not_charged(tmp_path):
    run, contract = source(tmp_path)
    (run / "logs/main.log").write_text(legacy_log(run, failed=0) + legacy_log(run, minute=60))
    result = cache_creation_cost(contract)
    assert result["gpu_seconds"] == 2 * 130 * 4
    assert [a["successful"] for a in result["attempts"]] == [False, True]


@pytest.mark.parametrize("change", ["missing_start", "missing_end", "retry_unclosed", "mismatch", "duplicate",
                                    "negative", "timer", "failed_final"])
def test_legacy_incomplete_or_inconsistent_records_remain_unknown(tmp_path, change):
    run, contract = source(tmp_path)
    lines = legacy_log(run).splitlines()
    if change == "missing_start":
        del lines[1]
    elif change == "missing_end":
        lines.pop()
    elif change == "retry_unclosed":
        lines = lines[:-1] + legacy_log(run, minute=60).splitlines()
    elif change == "mismatch":
        lines[1] = lines[1].replace("--n-train 400", "--n-train 399")
    elif change == "duplicate":
        lines.insert(2, lines[1])
    elif change == "negative":
        lines[-1] = lines[-1].replace("00:02:10", "00:00:00")
    elif change == "timer":
        lines[-1] = lines[-1].replace("(127s)", "(1s)")
    else:
        lines = legacy_log(run, failed=0).splitlines()
    (run / "logs/main.log").write_text("\n".join(lines) + "\n")
    result = cache_creation_cost(contract)
    assert result["gpu_seconds"] is None
    assert result["status"] == "unknown" and result["reason"]
    assert result["raw_timing_records"]


@pytest.mark.parametrize("change", ["missing", "unclosed", "retried", "reused", "mismatch", "cycle"])
def test_incomplete_or_ambiguous_provenance_is_not_a_zero_cost(tmp_path, change):
    run, contract = source(tmp_path)
    log = run / "logs/main.log"
    if change == "missing":
        log.unlink()
    elif change == "unclosed":
        log.write_text(log.read_text().splitlines()[0] + "\n")
    elif change == "retried":
        log.write_text(log.read_text() * 2)
    elif change == "reused":
        log.write_text(log.read_text().replace("400x8 on 4 GPUs", "reused"))
    elif change == "mismatch":
        contract["source_hashes"]["rollouts_behavior_train.jsonl"] = "wrong"
    else:
        (run / "run_config.json").write_text(json.dumps({"behavior_source": str(run)}))
    result = cache_creation_cost(contract)
    assert result["gpu_seconds"] is None
    assert result["status"] == "unknown" and result["reason"]
