"""Saved MoPPS/random-online work must not be displayed as a fresh READY task."""

import hashlib

import pytest
from test_mopps_comparison_status import (
    prefix_done,
    prepared,
    published,
    running,
    status,
)
from test_status_saved_random_integration import policy

import selection_gate as core


@pytest.fixture
def state(tmp_path, monkeypatch):
    root, parent = tmp_path / "comparison", tmp_path / "parent"
    prepared(root, parent)
    prefix_done(parent, 3, 25)
    monkeypatch.setattr(status.node_view, "local_gpus", lambda: {
        "host": "fixture", "available": False, "gpus": [], "processes": []})
    return root, root / "states/s3-t25"


def task(root, arm):
    return next(t for t in status.snapshot(root, now=1000.)["tasks"]
                if t["seed"] == 3 and t["step"] == 25 and t["arm"] == arm)


def online_final(directory):
    policy(directory, 125, 25)
    path = directory / "policy"
    core.atomic_json(path / "selector_state.json", {"state": "saved"})
    manifest = core.read(path / "policy_train.json")
    manifest["selector_state_sha256"] = hashlib.sha256((path / "selector_state.json").read_bytes()).hexdigest()
    core.atomic_json(path / "policy_train.json", manifest)


@pytest.mark.parametrize("arm", ["mopps", "random_online"])
def test_intact_checkpoint_is_resume_not_ready_without_progress(state, arm):
    root, point = state
    policy(point / arm, 125, 25, checkpoint=True)
    observed = task(root, arm)
    assert observed["status"] == "RESUME", observed
    assert observed["resume_validation_required"] is True
    assert observed["retryable"] is True
    assert "RESUME" in status.render(status.snapshot(root, now=1000.), local_gpus=False)


@pytest.mark.parametrize("arm", ["mopps", "random_online"])
def test_saved_final_online_policy_is_eval_not_ready_or_done(state, arm):
    root, point = state
    online_final(point / arm)
    observed = task(root, arm)
    assert observed["status"] == "EVAL", observed
    assert observed["resume_validation_required"] is True
    assert observed["retryable"] is True


def test_missing_selector_state_blocks_optimistic_eval(state):
    root, point = state
    policy(point / "mopps", 125, 25)
    assert task(root, "mopps")["status"] == "REVIEW"


@pytest.mark.parametrize("saved", ["checkpoint", "final", "partial"])
def test_current_training_heartbeat_has_priority_over_saved_files(state, saved):
    root, point = state
    directory = point / "random_online"
    if saved == "checkpoint":
        policy(directory, 125, 25, checkpoint=True)
    elif saved == "final":
        online_final(directory)
    else:
        core.atomic_json(directory / "policy/grpo_stats.jsonl", {"step": 100})
    running(directory, "healthy-node", now=1000.)
    assert task(root, "random_online")["status"] == "RUNNING"
    assert task(root, "random_online")["retryable"] is False


@pytest.mark.parametrize("saved,want", [("result", "DONE"), ("checkpoint", "RESUME"), ("final", "EVAL")])
def test_archived_history_never_overrides_current_saved_work(state, saved, want):
    root, point = state
    directory = point / "random_online"
    if saved == "result":
        published(directory)
    elif saved == "checkpoint":
        policy(directory, 125, 25, checkpoint=True)
    else:
        online_final(directory)
    core.atomic_json(directory / "discarded/old/policy/grpo_stats.jsonl", {"step": 50})
    observed = task(root, "random_online")
    assert observed["status"] == want
    assert observed["archived_work"] and observed["history_warning"]


def test_partial_or_archived_only_policy_is_review_not_ready_and_status_is_read_only(state):
    root, point = state
    core.atomic_json(point / "random_online/policy/grpo_stats.jsonl", {"step": 75})
    core.atomic_json(point / "mopps/discarded/old/policy/grpo_stats.jsonl", {"step": 75})
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}
    assert task(root, "random_online")["status"] == "REVIEW"
    assert task(root, "mopps")["status"] == "REVIEW"
    status.render(status.snapshot(root, now=1000.), local_gpus=False)
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}
