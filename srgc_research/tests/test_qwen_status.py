"""Qwen status: real artifacts, current-attempt progress and immutable queues."""

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts import srgc_qwen35 as qwen
from srgc_rebuttal.plan import digest, input_path, load_plan
from srgc_rebuttal.runtime import atomic_json, code_digest, lease, run_root
from srgc_research.dispatch import qwen_status as status

REPO = Path(__file__).resolve().parents[2]


class ChatTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return "Qwen test chat: " + messages[0]["content"]


@dataclass
class Prepared:
    root: Path
    env: dict
    plans: dict

    def read(self, dataset="all"):
        return status.load_reports(dataset, self.root, self.env)


@pytest.fixture
def prepared(tmp_path):
    work = tmp_path / "work with spaces"
    root = work / "srgc-rebuttal/qwen35-9b-v2"
    plans = {name: qwen.prepare(name, REPO / "srgc_rebuttal/experiments" / filename, root, ChatTokenizer())
             for name, filename in (("math", "additional_seeds.json"), ("mbpp", "mbpp_seeds.json"))}
    env = {**os.environ, "GROUP_VOLUME": str(tmp_path), "OM_WORK": str(work),
           "SRGC_QWEN_ROOT": str(root), "QWEN_PYTHON": sys.executable}
    return Prepared(root, env, plans)


def files(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def cached_seed(prepared, dataset, seed):
    plan_path = prepared.plans[dataset]
    plan = load_plan(plan_path)
    bundle_path = input_path(plan_path, plan, seed)
    bundle = json.loads(bundle_path.read_text())
    bundle["cached_rewards"] = {rid: [0, 1] * 4 for rid in bundle["candidate_ids"]}
    bundle["provenance"]["cache"] = {"model": qwen.MODEL, "model_revision": qwen.REVISION,
        "responses": 8, "max_new_tokens": 2048, "cache_seed": seed,
        "verifier": plan["verifier"], "attention": "eager"}
    atomic_json(bundle_path, bundle)
    identity = {"seed": seed, "plan_sha256": digest(plan_path), "input_sha256": digest(bundle_path),
        "implementation_sha256": hashlib.sha256((code_digest() + qwen.adapter_digest()).encode()).hexdigest()}
    folder = run_root(plan_path, plan) / f"seed-{seed}"
    atomic_json(folder / "run.json", {**identity, "status": "running"})
    return plan, folder, identity


def running_prefix(prepared, *, finished_step=3):
    _, folder, identity = cached_seed(prepared, "math", 7)
    directory = folder.parent / ".queue"
    started = time.time() - 30
    atomic_json(directory / "tasks/seed-7.prefix.json", {"status": "running", "attempt": 1,
        "task": "seed-7.prefix", "attempt_id": "current", "started": started, "host": "h100-A"})
    atomic_json(folder / "cost-receipts/shared-prefix/current.json", {"phase": "training",
        "state": "started", "checkpoint": finished_step})
    for dataset in ("math", "mbpp"):
        root = prepared.root / "runs" / dataset
        atomic_json(root / ".queue/workers/shared-worker.json", {"host": "h100-A", "pid": 123,
            "worker_id": "shared-worker", "heartbeat": time.time(), "dataset": dataset,
            "status": "running" if dataset == "math" else "idle",
            "task": "seed-7.prefix" if dataset == "math" else None, "active_dataset": "math_train"})
    return folder, directory, identity


def test_all_shows_sixty_planned_tasks_without_gpu_imports_or_filesystem_writes(prepared):
    before = files(prepared.root.parent.parent)
    with patch.dict(sys.modules, {"torch": None, "transformers": None, "peft": None}):
        snapshots = prepared.read()
    assert [report["label"] for report in snapshots] == ["MATH", "MBPP"]
    assert all(len(report["tasks"]) == 30 and report["errors"] == [] for report in snapshots)
    text = status.render(snapshots, prepared.root, host="viewer", now=0)
    assert "0/60 completed" in text and "WAIT 50" in text and "READY 10" in text
    assert "cache 0/5" in text and "train 0/20" in text
    assert "On-policy" in text and "WAIT cache" in text and "WAIT prefix" in text
    assert files(prepared.root.parent.parent) == before
    assert not list(prepared.root.rglob("protocol.json"))
    assert not list(prepared.root.rglob("*.html"))


def test_one_command_renders_dashboard_and_shortcut_without_gpu_packages(prepared):
    wrapper = ("import runpy, sys; sys.modules.update(torch=None, transformers=None, peft=None); "
               "sys.argv=sys.argv[1:]; runpy.run_module('srgc_research.dispatch.qwen_status', run_name='__main__')")
    before = files(prepared.root.parent.parent)
    result = subprocess.run([sys.executable, "-c", wrapper, "qwen-status", "all", "status"],
                            cwd=REPO, env=prepared.env, capture_output=True, text=True, timeout=20, check=False)
    assert result.returncode == 0, result.stderr
    assert "Qwen3.5-9B | SRGC" in result.stdout and "0/60 completed" in result.stdout
    result = subprocess.run(["sh", str(REPO / "scripts/run_srgc_qwen35.sh"), "status"],
                            cwd="/tmp", env=prepared.env, capture_output=True, text=True, timeout=20, check=False)
    assert result.returncode == 0, result.stderr
    assert "On-policy" in result.stdout and "0/60 completed" in result.stdout
    assert files(prepared.root.parent.parent) == before


def test_running_prefix_counts_only_completed_updates_and_deduplicates_shared_node(prepared):
    _, directory, _ = running_prefix(prepared)
    before = files(prepared.root)
    with lease(directory / "leases/seed-7.prefix.lock"):
        snapshots = prepared.read()
        row = next(task for task in snapshots[0]["tasks"] if task["task"] == "seed-7.prefix")
        assert row["completed_steps"] == 3 and row["current_step"] == 4
        nodes = status.node_rows(snapshots)
        assert len(nodes) == 1 and nodes[0]["task"] == "seed-7.prefix"
        text = status.render(snapshots, prepared.root, host="h100-A")
        assert "1 active" in text and "RUN 3/25" in text
        assert "update 4/25 in progress" in text and "3/25 completed" in text
        assert "* h100-A" in text and "0/20" in text
    # The read must not change any receipts, checkpoints or inputs.
    after = files(prepared.root)
    assert {name: data for name, data in after.items() if not name.endswith(".lock")} == {
        name: data for name, data in before.items() if not name.endswith(".lock")}


def test_previous_attempt_high_step_does_not_override_current_resume(prepared):
    folder, directory, _ = running_prefix(prepared, finished_step=2)
    old = folder / "cost-receipts/shared-prefix/old.json"
    atomic_json(old, {"phase": "training", "state": "finished", "checkpoint": 24})
    modified = time.time() - 60
    os.utime(old, (modified, modified))
    with lease(directory / "leases/seed-7.prefix.lock"):
        row = next(task for task in prepared.read("math")[0]["tasks"] if task["task"] == "seed-7.prefix")
        assert status.cell(row) == "RUN 2/25"
        assert row["current_step"] == 3


def test_final_evaluation_at_275_is_running_until_endpoint_exists(prepared):
    _, folder, _ = cached_seed(prepared, "math", 7)
    directory = folder.parent / ".queue"
    atomic_json(directory / "tasks/seed-7.on_policy.json", {"status": "running", "attempt": 1,
        "task": "seed-7.on_policy", "started": time.time() - 5})
    atomic_json(folder / "cost-receipts/on_policy/eval.json", {"phase": "evaluation",
        "state": "started", "checkpoint": 275})
    with lease(directory / "leases/seed-7.on_policy.lock"):
        snapshot = prepared.read("math")[0]
        task = next(task for task in snapshot["tasks"] if task["arm"] == "on_policy" and task["seed"] == 7)
        assert status.cell(task) == "RUN 275/275"
        assert task["reward_percent"] is None and not snapshot["complete"]
        assert "evaluation" in status.progress(task)


def test_cache_partial_counts_are_prompt_counts_without_optimizer_steps(prepared):
    plan_path = prepared.plans["math"]
    plan = load_plan(plan_path)
    bundle = json.loads(input_path(plan_path, plan, 7).read_text())
    cache = input_path(plan_path, plan, 7).with_suffix(".cache")
    for rid in bundle["candidate_ids"][:13]:
        atomic_json(cache / f"{hashlib.sha256(rid.encode()).hexdigest()}.json", {"saved": True})
    with lease(cache / "execution.lock"):
        snapshot = prepared.read("math")[0]
        row = next(task for task in snapshot["tasks"] if task["task"] == "seed-7.cache")
        assert status.cell(row) == "RUN 13/400"
        assert "13/400 prompts" in status.progress(row) and "completed_steps" not in row
        assert "train 0/20" in status.render([snapshot], prepared.root)


def test_missing_one_dataset_does_not_hide_other_dataset(prepared):
    prepared.plans["math"].unlink()
    snapshots = prepared.read()
    assert not snapshots[0]["prepared"] and snapshots[1]["prepared"]
    assert len(snapshots[1]["tasks"]) == 30
    text = status.render(snapshots, prepared.root)
    assert "MATH  NOT PREPARED" in text and "MBPP" in text and "READ ERRORS" in text


def test_old_runtime_is_viewable_and_not_rewritten(prepared):
    plan_path = prepared.plans["math"]
    plan = load_plan(plan_path)
    plan.update(adapter_sha256="a" * 64, engine_sha256="b" * 64)
    atomic_json(plan_path, plan)
    queue_root = run_root(plan_path, plan) / ".queue"
    atomic_json(queue_root / "protocol.json", {"implementation_sha256": "c" * 64})
    before = files(prepared.root)
    report = prepared.read("math")[0]
    assert report["prepared"] and report["warnings"] and not report["errors"]
    assert files(prepared.root) == before


def test_bad_qwen_bundle_is_visible_and_never_reported_as_complete(prepared):
    plan_path = prepared.plans["math"]
    path = input_path(plan_path, load_plan(plan_path), 7)
    bundle = json.loads(path.read_text())
    bundle["provenance"]["model"] = "wrong-model"
    atomic_json(path, bundle)
    report = prepared.read("math")[0]
    assert report["errors"] and not report["complete"]
    assert all(row["status"] == "invalid" for row in report["tasks"] if row["seed"] == 7)
    assert "wrong" in " ".join(report["errors"]) or "pinned Qwen" in " ".join(report["errors"])


def test_failed_tasks_show_last_exception_and_log_path(prepared):
    root = prepared.root / "runs/math/.queue"
    atomic_json(root / "tasks/seed-7.cache.json", {"task": "seed-7.cache", "status": "failed", "attempt": 3})
    log = root / "logs/seed-7.cache.log"
    log.parent.mkdir(parents=True)
    log.write_text("Traceback (most recent call last):\ntorch.OutOfMemoryError: CUDA out of memory\n")
    text = status.render(prepared.read("math"), prepared.root)
    assert "ERROR x3" in text and "torch.OutOfMemoryError: CUDA out of memory" in text
    assert f"log: {log}" in text


def test_stale_worker_and_admission_failure_remain_visible(prepared):
    root = prepared.root / "runs/math/.queue/workers"
    atomic_json(root / "stale.json", {"worker_id": "stale", "host": "h100-old", "heartbeat": time.time() - 120,
        "status": "running", "task": "seed-5.cache"})
    atomic_json(root / "failed.json", {"worker_id": "failed", "host": "h100-bad", "heartbeat": time.time() - 10,
        "status": "failed", "task": None, "error": "four-GPU admission failed; inspect /shared/preflight.log"})
    text = status.render(prepared.read("math"), prepared.root)
    assert "0 active | 1 stale | 1 stopped" in text and "STALE" in text
    assert "h100-bad" in text and "/shared/preflight.log" in text


def test_json_uses_same_snapshot_and_bad_data_returns_nonzero(prepared, monkeypatch, capsys):
    read = status.load_reports
    monkeypatch.setattr(status, "load_reports", lambda dataset, root, env: read(dataset, root, prepared.env))
    assert status.main(["math", "status", "--root", str(prepared.root), "--json"]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["model"] == qwen.MODEL and len(value["datasets"]) == 1
    prepared.plans["math"].write_text("broken JSON")
    assert status.main(["math", "status", "--root", str(prepared.root)]) == 1
    assert "NOT PREPARED" in capsys.readouterr().out


def test_outside_group_root_is_rejected_without_creating_directories(prepared, tmp_path):
    outside = tmp_path.parent / "not-a-qwen-status-output"
    with pytest.raises(ValueError, match="below group storage"):
        status.load_reports("all", outside, prepared.env)
    assert not outside.exists()


def test_status_code_is_excluded_from_training_and_measurement_runtime_identities():
    from srgc_research.storage import runtime_files
    assert "srgc_research/dispatch/qwen_status.py" not in runtime_files()
    assert "scripts/run_srgc_qwen35.sh" not in qwen.ADAPTER_FILES


def test_new_idle_session_does_not_hide_a_verified_running_session(prepared):
    _, directory, _ = running_prefix(prepared)
    atomic_json(directory / "workers/new-idle.json", {
        "worker_id": "new-idle", "host": "h100-A", "heartbeat": time.time(),
        "status": "idle", "task": None,
    })
    with lease(directory / "leases/seed-7.prefix.lock"):
        nodes = status.node_rows(prepared.read())
        assert len(nodes) == 1 and nodes[0]["status"] == "running"
        assert nodes[0]["worker_id"] == "shared-worker"
        text = status.render(prepared.read(), prepared.root)
        assert "| RUN 1 | IDLE 0 | ADMISSION 0" in text


def test_dataset_only_view_marks_the_shared_worker_as_busy_elsewhere(prepared):
    running_prefix(prepared)
    nodes = status.node_rows(prepared.read("mbpp"))
    assert len(nodes) == 1 and nodes[0]["status"] == "serving_other_dataset"
    text = status.render(prepared.read("mbpp"), prepared.root)
    assert "RUN OTHER" in text and "math_train" in text
    assert "| RUN 1 | IDLE 0" in text
    assert "shared worker is running the other dataset" in text


def test_resume_first_wait_reason_uses_current_task_evidence_and_is_read_only(prepared):
    snapshots = prepared.read()
    marker = prepared.root / ".dispatch/resume-first.json"
    atomic_json(marker, {"schema": "qwen-resume-first-v1", "tasks": [
        [str(prepared.plans["math"]), "seed-7.cache"],
        [str(prepared.plans["mbpp"]), "seed-8.cache"],
    ]})
    # A completed task can remain in the marker until the next dispatch pass.
    next(row for row in snapshots[0]["tasks"] if row["task"] == "seed-7.cache")["status"] = "complete"
    next(row for row in snapshots[1]["tasks"] if row["task"] == "seed-8.cache")["status"] = "running"
    snapshots[0]["workers"] = [{"host": "h100-idle", "worker_id": "idle", "status": "idle",
                                "heartbeat_age_seconds": 1, "task": None}]
    before = files(prepared.root)
    with patch.dict(sys.modules, {"torch": None, "transformers": None, "peft": None}):
        text = status.render(snapshots, prepared.root)
    assert "resume-first: 1 unfinished, 1 running; fresh tasks blocked" in text
    assert "Wait reason" in text and "| RUN 0 | IDLE 1" in text
    assert files(prepared.root) == before


def test_resume_backlog_in_other_dataset_is_visible_in_dataset_only_view(prepared):
    atomic_json(prepared.root / ".dispatch/resume-first.json", {
        "schema": "qwen-resume-first-v1",
        "tasks": [[str(prepared.plans["mbpp"]), "seed-8.cache"]],
    })
    backlog, error = status.resume_backlog(prepared.read("math"), prepared.root)
    assert error is None and backlog == [{"status": "outside_view"}]


@pytest.mark.parametrize("entries", [None, ["bad-entry"], [["/another/experiments/math.json", "seed-8.cache"]]])
def test_invalid_resume_diagnostics_are_visible_without_breaking_status(prepared, entries):
    atomic_json(prepared.root / ".dispatch/resume-first.json", {
        "schema": "qwen-resume-first-v1", "tasks": entries,
    })
    text = status.render(prepared.read(), prepared.root)
    assert "resume-first status unavailable:" in text and "0/60 completed" in text


def test_rank_progress_does_not_infer_gpu_utilization_or_invent_rank_ids():
    task = {"arm": "on_policy", "progress": [
        {"stage": "generation", "updated": 980},
        {"stage": "gradient_scoring", "updated": 975, "rank": 3},
    ]}
    with patch.object(status.time, "time", return_value=1000):
        text = status.progress(task)
    assert "rank progress: generation (20s ago); r3 gradient_scoring (25s ago)" in text
    assert "r0" not in text and "busy" not in text


def test_status_only_fix_preserves_all_model_and_information_runtime_identities():
    from srgc_research.dispatch.gemma4 import adapter as gemma
    from srgc_research.dispatch.llama31 import adapter as llama
    from srgc_research.storage import runtime_files

    assert llama.adapter_digest() == "1a7ae6ecf3b7fd24643797c97ce56750d32a9e49678125d8ddd2eabb5b61b53a"
    assert llama.engine_digest() == "12cf5ef830ebfd92fa8a87ea62dc7df734cd9ceab57fbce18fc4b2548385f960"
    assert qwen.adapter_digest() == "20e90f784b19afa6602b839ff2f7934eb207f03868a4052896edc9355b7ffd8b"
    assert "srgc_research/dispatch/qwen_status.py" not in gemma.SHARED_FILES
    assert "srgc_research/dispatch/qwen_status.py" not in runtime_files()
