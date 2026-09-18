"""Point resolution must not hide saved work behind a fabricated READY path."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("switch_state_point", ROOT / "scripts/_switch_state_point.py")
resolver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resolver)


def publish_suite(child, name="view-25"):
    out = child / "points" / name
    out.mkdir(parents=True)
    contract = out / "contract.json"
    contract.write_text('{"saved": true}\n')
    suite = {"schema": resolver.SUITE_SCHEMA,
             "points": [{"name": name, "sha256": hashlib.sha256(contract.read_bytes()).hexdigest()}]}
    (child / "suite.json").write_text(json.dumps(suite))
    return out


def test_unpublished_empty_state_uses_expected_future_point(tmp_path):
    assert resolver.resolve_state_point(tmp_path, 25) == (tmp_path / "points/view-25", "")


def test_single_legacy_point_retains_its_actual_path(tmp_path):
    out = tmp_path / "points/saved-view-25"
    out.mkdir(parents=True)
    assert resolver.resolve_state_point(tmp_path, 25) == (out, "")


@pytest.mark.parametrize("name", ["view-25", "saved-view-25"])
def test_bound_point_wins_over_unrelated_directories_like_worker(tmp_path, name):
    import selection_gate_gpu as worker

    out = publish_suite(tmp_path, name)
    (tmp_path / "points/unrelated").mkdir()
    actual, error = resolver.resolve_state_point(tmp_path, 25)
    assert not error
    assert actual == out == next(worker.entries(tmp_path))


@pytest.mark.parametrize("canonical_exists", [False, True])
def test_multiple_unbound_points_are_explicit_review_not_ready(tmp_path, canonical_exists):
    for name in ("view-25" if canonical_exists else "saved-view-25", "unrelated"):
        (tmp_path / "points" / name).mkdir(parents=True)
    actual, error = resolver.resolve_state_point(tmp_path, 25)
    assert actual == tmp_path / "points/view-25"
    assert "multiple state points without suite authority" in error


def test_tampered_contract_cannot_fall_back_to_another_point(tmp_path):
    out = publish_suite(tmp_path, "saved-view-25")
    (out / "contract.json").write_text('{"saved": false}\n')
    (tmp_path / "points/view-25").mkdir()
    actual, error = resolver.resolve_state_point(tmp_path, 25)
    assert actual == tmp_path / "points/view-25"
    assert error == "point contract changed"


@pytest.mark.parametrize("suite", [[], {}, {"schema": resolver.SUITE_SCHEMA, "points": []},
    {"schema": resolver.SUITE_SCHEMA, "points": [{"name": "../other"}]},
    {"schema": resolver.SUITE_SCHEMA, "points": [{"name": "."}]},
    {"schema": resolver.SUITE_SCHEMA, "points": [{"name": "a"}, {"name": "b"}]}])
def test_invalid_suite_is_explicit_review_even_with_one_point(tmp_path, suite):
    publish_suite(tmp_path)
    (tmp_path / "suite.json").write_text(json.dumps(suite))
    assert resolver.resolve_state_point(tmp_path, 25)[1]


def test_unreadable_suite_does_not_look_unprepared(tmp_path):
    (tmp_path / "suite.json").write_text("{")
    assert resolver.resolve_state_point(tmp_path, 25)[1]


@pytest.mark.parametrize("link", ["points", "points/saved-view-25"])
def test_unavailable_point_link_never_looks_like_empty_new_work(tmp_path, link):
    path = tmp_path / link
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(tmp_path / "unmounted-target", target_is_directory=True)
    assert "unavailable" in resolver.resolve_state_point(tmp_path, 25)[1]


def test_status_reads_published_random_from_authoritative_point_not_fabricated_ready(tmp_path):
    from test_selection_switch_status import (
        completed_prefix,
        prepared,
        published,
        status,
    )

    prepared(tmp_path)
    completed_prefix(tmp_path)
    child = tmp_path / "states/s0-t25"
    out = publish_suite(child, "saved-view-25")
    published(out / "random_reduced")
    (child / "points/unrelated").mkdir()
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()}
    data = status.snapshot(tmp_path)
    task = next(t for t in data["tasks"] if t["seed"] == 0 and t["step"] == 25 and t["arm"] == "random_reduced")
    assert task["status"] == "DONE"
    assert task["directory"] == "states/s0-t25/points/saved-view-25/random_reduced"
    assert data["development_done"] == 1
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}


def test_status_ambiguous_saved_paths_are_review_never_ready(tmp_path):
    from test_selection_switch_status import (
        completed_prefix,
        prepared,
        published,
        status,
    )

    prepared(tmp_path)
    completed_prefix(tmp_path)
    child = tmp_path / "states/s0-t25"
    published(child / "points/saved-view-25/random_reduced")
    (child / "points/unrelated").mkdir()
    data = status.snapshot(tmp_path)
    tasks = [t for t in data["tasks"] if t["seed"] == 0 and t["step"] == 25 and t["kind"] == "branch"]
    assert tasks and all(t["status"] == "REVIEW" and not t["retryable"] for t in tasks)
    assert all("state path unresolved" in t["reason"] for t in tasks)


@pytest.mark.parametrize("saved,expected", [("result", "DONE"), ("final", "EVAL"), ("checkpoint", "RESUME")])
def test_status_reads_bound_saved_work_with_multiple_point_directories(tmp_path, saved, expected):
    from test_selection_switch_status import (
        completed_prefix,
        core,
        final_policy_candidate,
        prepared,
        published,
        status,
    )

    prepared(tmp_path)
    completed_prefix(tmp_path)
    child = tmp_path / "states/s0-t25"
    out = publish_suite(child, "saved-view-25")
    directory = out / "random_reduced"
    if saved == "result":
        published(directory)
    elif saved == "final":
        final_policy_candidate(directory / "policy")
    else:
        checkpoint = directory / "policy/checkpoint-000030"
        checkpoint.mkdir(parents=True)
        for name in ("adapter_config.json", "adapter_model.safetensors", "optimizer.pt", "grpo_stats.jsonl"):
            (checkpoint / name).write_bytes(b"candidate fixture")
        core.atomic_json(checkpoint / "checkpoint_state.json", {"completed_steps": 30,
            "adapter_sha256": "a" * 64, "optimizer_sha256": "b" * 64, "grpo_stats_sha256": "c" * 64})
    (child / "points/unrelated").mkdir()
    before = {path: path.read_bytes() for path in child.rglob("*") if path.is_file()}
    task = next(task for task in status.snapshot(tmp_path)["tasks"]
                if task["kind"] == "branch" and task["seed"] == 0 and task["step"] == 25
                and task["arm"] == "random_reduced")
    assert task["directory"] == str(directory.relative_to(tmp_path))
    assert task["status"] == expected
    assert before == {path: path.read_bytes() for path in child.rglob("*") if path.is_file()}


def test_status_unbound_ambiguity_never_announces_ready_or_retry(tmp_path):
    from test_selection_switch_status import (
        completed_prefix,
        prepared,
        published,
        status,
    )

    prepared(tmp_path)
    completed_prefix(tmp_path)
    child = tmp_path / "states/s0-t25"
    published(child / "points/saved-view-25/random_reduced")
    (child / "points/unrelated").mkdir()
    branches = [task for task in status.snapshot(tmp_path)["tasks"]
                if task["kind"] == "branch" and task["seed"] == 0 and task["step"] == 25]
    assert branches
    assert all(task["status"] == "REVIEW" and task["retryable"] is False for task in branches)
