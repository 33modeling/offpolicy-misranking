"""Published results must not hide still-running work in alternate views."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _status_execution import current_tasks, execution_tasks
import experiments_progress as progress
import experiments_status as combined
import queue_dispatch_evidence as dispatch
import selection_switch_status as switch
import mbpp_status as mbpp


def task(**updates):
    return {"kind": "branch", "status": "DONE", "seed": 0, "step": 25,
            "arm": "selection_reduced", "directory": "states/s0-t25/selection_reduced",
            "role": "DEV", "host": "same-host", "pid": 123, "phase": "curve",
            "seconds": 10, "timeout": 20, "heartbeat_age": 1, "reason": "published",
            "training_step": 25, **updates}


@pytest.mark.parametrize("evidence", [{"heartbeat_fresh": True}, {"owner_active": True},
                                      {"task_lease_held": True}])
def test_projection_preserves_publication_and_covers_live_ownership(evidence):
    original = [task(**evidence)]
    projected = execution_tasks(original)
    assert projected[0]["status"] == "RUNNING"
    assert projected[0]["publication_status"] == "DONE"
    assert original[0]["status"] == "DONE"


def test_descendant_activity_has_path_boundary_and_never_mutates_snapshot():
    branch = task()
    child = task(kind="curve", directory=branch["directory"] + "/curve/step-25",
                 heartbeat_fresh=True)
    neighbor = task(directory=branch["directory"] + "-other")
    projected = execution_tasks([branch, child, neighbor])
    assert [item["status"] for item in projected] == ["RUNNING", "RUNNING", "DONE"]
    assert [item['directory'] for item in current_tasks(projected)] == [child['directory']]
    assert branch["status"] == "DONE"


@pytest.mark.parametrize("evidence", [{"heartbeat_fresh": True}, {"task_lease_held": True}])
def test_same_directory_phase_overrides_saved_publication(evidence):
    branch = task()
    phase = task(kind="phase", status="WAIT", **evidence)
    projected = execution_tasks([branch, phase])
    assert [item["status"] for item in projected] == ["RUNNING", "RUNNING"]
    assert projected[0]["publication_status"] == "DONE"
    assert projected[0]["execution_inferred"]
    assert current_tasks(projected) == [projected[1]]
    assert branch["status"] == "DONE"


@pytest.mark.parametrize("pids", [(123, 456), (123, 123)])
def test_current_nested_work_does_not_merge_duplicate_hosts_without_owner_id(pids):
    branch = task(status="RUNNING", heartbeat_fresh=True, pid=pids[0])
    child = task(status="RUNNING", heartbeat_fresh=True, pid=pids[1], kind="curve",
                 directory=branch["directory"] + "/curve/step-25")
    assert current_tasks([branch, child]) == [branch, child]


@pytest.mark.parametrize("identity", ["worker_id", "event_id"])
def test_current_nested_work_collapses_only_confirmed_same_owner(identity):
    branch = task(status="RUNNING", heartbeat_fresh=True, **{identity: "allocation-unique"})
    child = task(status="RUNNING", heartbeat_fresh=True, kind="curve",
                 directory=branch["directory"] + "/curve/step-25", **{identity: "allocation-unique"})
    assert current_tasks([branch, child]) == [child]


@pytest.mark.parametrize("nested", [False, True])
def test_compact_progress_and_dispatch_agree_on_active_saved_result(tmp_path, nested):
    branch = task(heartbeat_fresh=not nested)
    tasks = [branch]
    if nested:
        tasks.append(task(kind="curve", directory=branch["directory"] + "/curve/step-25",
                          heartbeat_fresh=True))
    data = {"root": str(tmp_path), "prepared": True, "tasks": tasks,
            "branch_counts": {"DONE": 1}, "training_published": 1}
    original = copy.deepcopy(data)
    compact = switch.render_compact(data)
    short = "\n".join(progress.render_root(tmp_path, data, width=200, kind="switch"))
    (tmp_path / "switch.json").write_text(json.dumps({}))
    evidence = "\n".join(dispatch.describe(tmp_path, "test", snapshot_loader=lambda *_: data))
    assert "DONE 0/1 branches" in compact and "RUNNING 1" in compact
    assert "CURRENT RUN 1" in compact
    assert "DONE 0/1" in short and "RUN 1" in short
    assert "DONE=0" in evidence and "RUN=1" in evidence
    assert "status=RUNNING" in evidence
    assert data == original


def test_full_view_current_matrix_and_details_use_execution_state(tmp_path):
    fixture_spec = importlib.util.spec_from_file_location("status_fixture", ROOT / "tests/test_selection_switch_status.py")
    fixture = importlib.util.module_from_spec(fixture_spec)
    fixture_spec.loader.exec_module(fixture)
    fixture.prepared(tmp_path)
    data = switch.snapshot(tmp_path, local_gpus=False)
    branch = next(item for item in data["tasks"] if item["kind"] == "branch")
    branch.update(task(directory=branch["directory"], heartbeat_fresh=True))
    data["branch_counts"]["DONE"] = 1
    original = copy.deepcopy(data)
    output = switch.render(data, all_tasks=True, nodes=False, local_gpus=False, width=180)
    assert "BRANCHES  RUNNING 1" in output
    assert f"RUNNING  {branch['directory']}" in output
    current = output.split("CURRENT WORK", 1)[1].split("PREFIXES", 1)[0]
    assert "RUN" in current and "DONE" not in current
    assert data == original


def test_sibling_published_live_task_is_in_node_input_and_counts(tmp_path, monkeypatch):
    sibling = tmp_path / "selection-switch-other-v1"
    sibling.mkdir()
    (sibling / "switch.json").write_text("{}")
    source = task(heartbeat_fresh=True)
    monkeypatch.setattr(combined.switch_status, "snapshot", lambda *a, **k: {
        "prepared": True, "tasks": [source]})
    tasks, summaries = combined.sibling_status(tmp_path / "primary", tmp_path / "mopps", now=1)
    assert len(tasks) == 1 and tasks[0]["status"] == "RUNNING"
    assert summaries[0]["branches"] == 48
    assert summaries[0]["branch_counts"] == {"RUNNING": 1, "WAIT": 47}
    assert source["status"] == "DONE"


@pytest.mark.parametrize("pids", [(123, 456), (123, 123)])
def test_same_hostname_independent_work_survives_even_with_same_pid(tmp_path, pids):
    first = task(pid=pids[0], heartbeat_fresh=True)
    second = task(pid=pids[1], heartbeat_fresh=True, arm="random_reduced",
                  directory="states/s0-t25/random_reduced")
    nested = task(pid=pids[0], heartbeat_fresh=True, kind="phase",
                  directory=first["directory"] + "/curve/step-25")
    data = {"suites": [{"root": str(tmp_path), "tasks": [first, second, nested]}]}
    original = copy.deepcopy(data)
    nodes = mbpp.node_assignments(data)
    assert len(nodes) == 2 and all(node["state"] == "RUN" for node in nodes)
    assert sorted(len(node["assignments"]) for node in nodes) == [1, 2]
    assert len({node["work_id"] for node in nodes}) == 2
    output = "\n".join(mbpp.render_nodes(data, width=400))
    assert "WORK ITEMS 2 current" in output
    assert all(node["work_id"] in output for node in nodes)
    assert data == original
