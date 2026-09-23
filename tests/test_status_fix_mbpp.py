"""Evaluation/curve failures after saved work are visible, and discarded GPU views are not queried."""

import hashlib
import importlib.util
import os
from pathlib import Path
import sys

import selection_gate as core
import selection_switch as rule

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("mbpp_status_fix", ROOT / "scripts/mbpp_status.py")
dashboard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dashboard)
switch = dashboard.switch_status
NOW = 1_900_000_000.


def mbpp_root(root):
    core.atomic_json(root / "switch.json", {"schema": rule.SCHEMA, "dataset": "mbpp", "selector": "fresh_r",
                                            "accounting": "matched", "gate": "convergence"})
    for seed in (*rule.DEV_SEEDS, *rule.TEST_SEEDS):
        for step in rule.STEPS:
            core.atomic_json(root / f"prefixes/seed-{seed}/prefix-{step}.json", {"schema": rule.SCHEMA})
    return root


def branch(root, arm="random_full", seed=4, step=100):
    return root / f"states/s{seed}-t{step}/points/view-{step}/{arm}"


def published(directory):
    core.atomic_json(directory / "result.json", {"schema": rule.SCHEMA, "complete": True})
    core.atomic_json(directory / "result.sha256.json",
                     {"sha256": hashlib.sha256((directory / "result.json").read_bytes()).hexdigest()})


def final_policy(policy):
    policy.mkdir(parents=True, exist_ok=True)
    for name in ("adapter_config.json", "adapter_model.safetensors", "optimizer.pt", "grpo_stats.jsonl"):
        (policy / name).write_bytes(b"metadata-only candidate fixture")
    budget = {"completed_steps": 180, "requested_target_steps": 100000, "stop_reason": "budget_exhausted"}
    core.atomic_json(policy / "policy_train.json", {"schema": "offpolicy-rlvr-policy/v1", "start_step": 100,
        "completed_steps": 180, "training_budget": budget,
        "adapter_sha256": "a" * 64, "optimizer_sha256": "b" * 64, "grpo_stats_sha256": "c" * 64})
    core.atomic_json(policy / "budget_stop.json", {**budget, "use_parent_policy": False})


def failure(directory, error, *, older_than=None, time=NOW - 300):
    core.atomic_json(directory / "failure.json", {"error": error, "host": "node-9", "time": time})
    if older_than is not None:
        # An earlier attempt: the saved work it would contradict came later.
        stamp = older_than.stat().st_mtime - 3600
        os.utime(directory / "failure.json", (stamp, stamp))


def task_of(data, directory, root):
    return next(t for t in data["tasks"] if t["directory"] == str(directory.relative_to(root)))


CURVE_OOM = ("curve worker failed: [1, 0, 0, 0]; see curve-*.log\n[worker-log] ...\n"
             "torch.OutOfMemoryError: CUDA out of memory")
EVAL_CRASH = "evaluate worker failed: [0, 1, 0, 0]\nRuntimeError: CUDA error: unspecified launch failure"


def test_curve_failure_after_published_result_is_visible_everywhere(tmp_path):
    root = mbpp_root(tmp_path / "selection-switch-mbpp-quality-v1")
    directory = branch(root)
    published(directory)
    failure(directory, CURVE_OOM)
    task = task_of(switch.snapshot(root, now=NOW, local_gpus=False), directory, root)
    # Status and controller semantics are unchanged; only the evidence is kept.
    assert task["status"] == "EVAL" and task["retryable"] is True and task["training_published"] is True
    assert task["last_failure"] == "torch.OutOfMemoryError: CUDA out of memory"
    assert "최근 실패: torch.OutOfMemoryError: CUDA out of memory" in dashboard.remark(task)
    output = dashboard.render(dashboard.snapshot([root], now=NOW), width=200, all_tasks=True)
    assert "저장 후 실패 1개" in output
    matrix = next(line for line in output.splitlines() if line.startswith("4 / 100"))
    assert "Full random: 최종 평가 저장됨; 곡선 평가 남음 (재학습 없음); 최근 실패: torch.OutOfMemoryError" in matrix
    assert any(line.startswith("WAIT states/s4-t100/points/view-100/random_full") and "OutOfMemoryError" in line
               for line in output.splitlines())
    full = switch.render(switch.snapshot(root, now=NOW, local_gpus=False), width=160, local_gpus=False, all_tasks=True)
    assert "FAILED AFTER SAVE 1" in full
    assert "EVAL s4/t100 random_full: last failure: torch.OutOfMemoryError: CUDA out of memory" in full
    assert "  last failure: torch.OutOfMemoryError: CUDA out of memory" in full


def test_failure_older_than_published_result_stays_history(tmp_path):
    root = mbpp_root(tmp_path / "selection-switch-mbpp-quality-v1")
    directory = branch(root)
    published(directory)
    failure(directory, CURVE_OOM, older_than=directory / "result.sha256.json")
    task = task_of(switch.snapshot(root, now=NOW, local_gpus=False), directory, root)
    assert task["status"] == "EVAL" and "last_failure" not in task
    assert "실패" not in dashboard.remark(task)
    assert "저장 후 실패" not in dashboard.render(dashboard.snapshot([root], now=NOW), width=200)


def test_evaluation_failure_after_saved_final_policy_is_visible(tmp_path):
    root = mbpp_root(tmp_path / "selection-switch-mbpp-quality-v1")
    directory = branch(root, "selection_reduced")
    final_policy(directory / "policy")
    # The meter's final heartbeat precedes record_failure().
    core.atomic_json(directory / "progress.json", {"state": "failed", "phase": "evaluate", "host": "node-9",
                                                   "updated": NOW - 301, "event_id": "e9"})
    failure(directory, EVAL_CRASH)
    task = task_of(switch.snapshot(root, now=NOW, local_gpus=False), directory, root)
    assert task["status"] == "EVAL" and task["retryable"] is True
    assert task["last_failure"] == "RuntimeError: CUDA error: unspecified launch failure"
    assert dashboard.remark(task) == "평가·결과 저장 남음; 최근 실패: RuntimeError: CUDA error: unspecified launch failure"
    output = dashboard.render(dashboard.snapshot([root], now=NOW), width=200, all_tasks=True)
    assert "평가·결과 저장 남음 1개; 저장 후 실패 1개" in output
    assert any(line.startswith("WAIT states/s4-t100/points/view-100/selection_reduced")
               and "unspecified launch failure" in line for line in output.splitlines())


def test_failure_older_than_saved_final_policy_stays_history(tmp_path):
    root = mbpp_root(tmp_path / "selection-switch-mbpp-quality-v1")
    directory = branch(root, "selection_reduced")
    final_policy(directory / "policy")
    failure(directory, EVAL_CRASH, older_than=directory / "policy/policy_train.json")
    task = task_of(switch.snapshot(root, now=NOW, local_gpus=False), directory, root)
    assert task["status"] == "EVAL" and task["retryable"] is True and "last_failure" not in task
    assert dashboard.remark(task) == "평가·결과 저장 남음"
    full = switch.render(switch.snapshot(root, now=NOW, local_gpus=False), width=160, local_gpus=False)
    assert "FAILED AFTER SAVE" not in full and "last failure" not in full


def test_failure_recorded_before_a_later_attempt_stays_history(tmp_path):
    root = mbpp_root(tmp_path / "selection-switch-mbpp-quality-v1")
    directory = branch(root, "selection_reduced")
    final_policy(directory / "policy")
    failure(directory, EVAL_CRASH, time=NOW - 1000)
    core.atomic_json(directory / "progress.json", {"state": "finished", "phase": "train", "host": "node-3",
                                                   "updated": NOW - 300, "event_id": "e10"})
    task = task_of(switch.snapshot(root, now=NOW, local_gpus=False), directory, root)
    assert task["status"] == "EVAL" and "last_failure" not in task
    assert dashboard.remark(task) == "평가·결과 저장 남음"


def test_compact_cli_does_not_query_gpus_but_full_view_still_does(tmp_path, monkeypatch, capsys):
    root = mbpp_root(tmp_path / "selection-switch-mbpp-v1")
    calls = []
    view = {"host": "this-node", "available": True, "gpus": [], "processes": []}
    monkeypatch.setattr(switch.node_view, "local_gpus", lambda: calls.append(1) or view)
    monkeypatch.setenv("SWITCH_STATUS_COMPACT", "1")
    monkeypatch.setattr(sys, "argv", ["status", "--root", str(root)])
    assert switch.main() == 0
    assert "SUITE MBPP" in capsys.readouterr().out and calls == []
    monkeypatch.delenv("SWITCH_STATUS_COMPACT")
    assert switch.main() == 0
    assert "THIS NODE GPUS" in capsys.readouterr().out and calls == [1]
