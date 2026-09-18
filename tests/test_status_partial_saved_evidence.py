"""Partial saved evidence must not appear as permission to start a new branch."""

import pytest
from test_selection_switch_status import (
    completed_prefix,
    point,
    prepared,
    published,
    running,
    status,
)


def task_at(root, *, kind="branch"):
    return next(task for task in status.snapshot(root, now=1900000000.)["tasks"]
                if task["kind"] == kind and task["seed"] == 0 and task["step"] == 25
                and task["arm"] == ("random_reduced" if kind == "branch" else "prefix"))


@pytest.mark.parametrize("name", ["adapter_config.json", "checkpoint_state.json"])
def test_partial_policy_metadata_is_review_not_ready(tmp_path, name):
    prepared(tmp_path)
    completed_prefix(tmp_path)
    path = point(tmp_path) / "random_reduced/policy" / name
    path.parent.mkdir(parents=True)
    path.write_bytes(b"saved metadata\n")
    before = path.read_bytes(), path.stat().st_mtime_ns
    task = task_at(tmp_path)
    assert task["status"] == "REVIEW" and task["retryable"] is False
    assert task["training_published"] is False
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


@pytest.mark.parametrize("name", ["result.sha256.json", "result.json"])
@pytest.mark.parametrize("broken_link", [False, True])
def test_orphan_completion_artifact_is_review_without_a_readable_result(tmp_path, name, broken_link):
    prepared(tmp_path)
    completed_prefix(tmp_path)
    path = point(tmp_path) / "random_reduced" / name
    path.parent.mkdir(parents=True)
    if broken_link:
        path.symlink_to(tmp_path / "unavailable-storage-target")
    elif name == "result.json":
        path.mkdir()  # Unreadable completion record, not a missing/fresh branch.
    else:
        path.write_text('{"sha256":"old-completion-seal"}\n')
    before = path.lstat()
    task = task_at(tmp_path)
    assert task["status"] == "REVIEW" and task["retryable"] is False
    assert name in task["reason"]
    assert path.is_symlink() is broken_link
    assert path.lstat().st_mtime_ns == before.st_mtime_ns


@pytest.mark.parametrize("name", ["policy", "policy/policy_train.json", "policy/adapter_config.json",
                                   "policy/checkpoint_state.json"])
def test_broken_policy_links_are_review_and_preserved(tmp_path, name):
    prepared(tmp_path)
    completed_prefix(tmp_path)
    path = point(tmp_path) / "random_reduced" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(tmp_path / "unavailable-storage-target")
    before = path.readlink()
    task = task_at(tmp_path)
    assert task["status"] == "REVIEW" and task["retryable"] is False
    assert path.is_symlink() and path.readlink() == before


def test_valid_published_result_still_wins_over_partial_auxiliary_files(tmp_path):
    prepared(tmp_path)
    completed_prefix(tmp_path)
    directory = point(tmp_path) / "random_reduced"
    published(directory)
    (directory / "policy").mkdir()
    (directory / "policy/adapter_config.json").write_text("{}\n")
    assert task_at(tmp_path)["status"] == "DONE"


def test_fresh_active_worker_is_not_reclassified_as_idle_review(tmp_path):
    prepared(tmp_path)
    completed_prefix(tmp_path)
    directory = point(tmp_path) / "random_reduced"
    running(directory, "live-node", now=1900000000., phase="train")
    (directory / "policy").mkdir()
    (directory / "policy/adapter_config.json").write_text("{}\n")
    assert task_at(tmp_path)["status"] == "RUNNING"


def test_snapshot_can_skip_local_gpu_queries_without_changing_saved_counts(tmp_path, monkeypatch):
    prepared(tmp_path)
    completed_prefix(tmp_path)
    published(point(tmp_path) / "random_reduced")

    def forbidden_query():
        raise AssertionError("read-only dashboard must not wait for nvidia-smi")

    monkeypatch.setattr(status.node_view, "local_gpus", forbidden_query)
    data = status.snapshot(tmp_path, local_gpus=False)
    assert data["local_gpus"] == []
    assert data["branch_counts"]["DONE"] == 1


def test_default_snapshot_still_includes_local_gpu_view(tmp_path, monkeypatch):
    prepared(tmp_path)
    view = {"host": "fixture", "available": False, "gpus": [], "processes": []}
    monkeypatch.setattr(status.node_view, "local_gpus", lambda: view)
    assert status.snapshot(tmp_path)["local_gpus"] is view


def test_mbpp_dashboard_does_not_query_local_gpus(tmp_path, monkeypatch):
    import mbpp_status as dashboard

    prepared(tmp_path)
    completed_prefix(tmp_path)
    published(point(tmp_path) / "random_reduced")

    def forbidden_query():
        raise AssertionError("dashboard must not wait for nvidia-smi")

    monkeypatch.setattr(dashboard.switch_status.node_view, "local_gpus", forbidden_query)
    data = dashboard.snapshot([tmp_path])
    assert data["suites"][0]["branch_counts"]["DONE"] == 1
