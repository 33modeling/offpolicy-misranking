"""Idle-node summaries need fresh waiting evidence, not merely absent work."""

import os
from copy import deepcopy

import pytest

from test_mbpp_status_dashboard import dashboard
from test_selection_switch_status import completed_prefix, point, prepared, running


NOW = 1900000000.0
GRACE = dashboard.switch_status.node_view.HEARTBEAT_GRACE


@pytest.fixture(autouse=True)
def no_gpu_query(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("idle-node status must not query a GPU")

    monkeypatch.setattr(dashboard.switch_status.node_view, "local_gpus", forbidden)


def report(nodes=(), tasks=(), retained=()):
    return {
        "updated": NOW,
        "suites": [{"root": "/unused/selection-switch-mbpp-quality-v1",
                    "prepared": False, "nodes": list(nodes), "tasks": list(tasks)}],
        "retained_suites": list(retained),
    }


def hosts(data):
    return [node["host"] for node in dashboard.idle_nodes(data)]


@pytest.mark.parametrize("state", ["WAIT", "HOLD"])
@pytest.mark.parametrize("age", [-5, 0, 5, GRACE - 0.1])
def test_only_fresh_explicit_waiting_is_confirmed_idle(state, age):
    data = report([{"host": "waiting-node", "state": state, "last_age": age}])
    before = deepcopy(data)
    nodes = dashboard.idle_nodes(data)
    assert len(nodes) == 1 and nodes[0]["host"] == "waiting-node"
    assert nodes[0]["current"] and not nodes[0]["assignments"]
    assert nodes[0]["evidence_age"] == age
    assert data == before


@pytest.mark.parametrize("state", ["ADMIT", "COOL", "LIVE", "UNKNOWN", "STALE", "EXITED", "GONE", "RUN", "-"])
def test_unassigned_is_not_the_same_as_idle(state):
    data = report([{"host": "unconfirmed-node", "state": state, "last_age": 5,
                    "launcher_alive": True}])
    assert not dashboard.idle_nodes(data)
    assert dashboard.render_idle_nodes(data)[0] == "작업 없는 노드: 0개 (배정 대기 확인)"


@pytest.mark.parametrize("age", [None, -5.1, GRACE, GRACE + 1, 900])
@pytest.mark.parametrize("launcher_alive", [None, True])
def test_old_or_unusable_waiting_evidence_is_not_confirmed_idle(age, launcher_alive):
    data = report([{"host": "old-wait-node", "state": "WAIT", "last_age": age,
                    "launcher_alive": launcher_alive}])
    assert not dashboard.idle_nodes(data)


def test_exited_launcher_is_not_idle_even_with_recent_wait_message():
    data = report([{"host": "exited-node", "state": "WAIT", "last_age": 2,
                    "launcher_alive": False}])
    assert not dashboard.idle_nodes(data)


def test_fresh_pid_does_not_refresh_an_old_wait_message():
    data = report([{"host": "new-launch-node", "state": "HOLD", "last_age": 900,
                    "pid_age": 1, "launcher_alive": True}])
    assert dashboard.node_assignments(data)[0]["current"]
    assert not dashboard.idle_nodes(data)


@pytest.mark.parametrize("status,kind,arm", [
    ("RUNNING", "branch", "random_reduced"),
    ("EVAL", "branch", "selection_reduced"),
    ("RUNNING", "phase", "random_reduced/curve/step-50"),
    ("RUNNING", "phase", "curve-parent"),
    ("DONE", "branch", "random_reduced"),
])
@pytest.mark.parametrize("retained", [False, True])
def test_live_assignment_in_any_observed_suite_prevents_idle(status, kind, arm, retained):
    task = {"host": "busy-node", "status": status, "heartbeat_fresh": True,
            "kind": kind, "seed": 0, "step": 25, "arm": arm,
            "directory": "states/s0-t25/points/view-25/" + arm}
    data = report([{"host": "busy-node", "state": "HOLD", "last_age": 1}])
    if retained:
        data["retained_suites"] = [{"root": "/unused/selection-switch-mbpp-v1", "tasks": [task]}]
    else:
        data["suites"][0]["tasks"] = [task]
    before = deepcopy(data)
    assert not dashboard.idle_nodes(data)
    assert data == before


def test_waiting_hosts_are_deduplicated_naturally_sorted_and_never_abbreviated():
    names = ["run284000-wts-10-g1234-full-allocation", "run284000-wts-2-g1234-full-allocation"]
    data = report([{"host": host, "state": "WAIT", "last_age": 4} for host in names])
    data["retained_suites"] = [{
        "root": "/unused/selection-switch-mbpp-v1",
        "nodes": [{"host": names[0] + "_", "state": "HOLD", "last_age": 3}],
    }]
    assert hosts(data) == list(reversed(names))
    lines = dashboard.render_idle_nodes(data)
    assert lines[0] == "작업 없는 노드: 2개 (배정 대기 확인)"
    assert lines[1].startswith(f"1. {names[1]} | WAIT | 작업 배정 대기")
    assert lines[2].startswith(f"2. {names[0]} | WAIT | 작업 배정 대기")
    assert sum("작업 배정 대기" in line for line in lines) == 2


@pytest.mark.parametrize("all_tasks", [False, True])
def test_summary_is_at_dashboard_bottom_with_zero_case_visible(all_tasks):
    data = report()
    lines = dashboard.render(data, all_tasks=all_tasks).splitlines()
    assert lines[0].startswith("MBPP EXPERIMENTS")
    assert lines[-2] == "작업 없는 노드: 0개 (배정 대기 확인)"
    assert lines.count("작업 없는 노드: 0개 (배정 대기 확인)") == 1
    assert "NODE ASSIGNMENTS" in lines


def test_snapshot_keeps_nested_and_retained_work_out_of_idle_list_read_only(tmp_path):
    quality = tmp_path / "runs/selection-switch-mbpp-quality-v1"
    retained = quality.with_name("selection-switch-mbpp-v1")
    for root in (quality, retained):
        prepared(root)
        completed_prefix(root)
    running(point(quality) / "random_reduced/curve/step-50", "nested-curve-node", now=NOW, phase="curve")
    running(point(retained) / "selection_reduced", "retained-node", now=NOW, phase="train")
    logs = quality.parent / "experiments/logs"
    logs.mkdir(parents=True)
    for host in ("idle-node", "nested-curve-node", "retained-node"):
        path = logs / f"console.mbpp.{host}_.log"
        path.write_text("[holding] node retained (peer work active)\n")
        os.utime(path, (NOW - 5, NOW - 5))

    def contents():
        return {path: (path.read_bytes(), path.stat().st_mtime_ns)
                for path in tmp_path.rglob("*") if path.is_file()}

    before = contents()
    data = dashboard.snapshot([quality], now=NOW, retained_roots=[retained])
    assert hosts(data) == ["idle-node"]
    output = dashboard.render(data)
    assert output.splitlines()[-3] == "작업 없는 노드: 1개 (배정 대기 확인)"
    assert "1. idle-node | WAIT | 작업 배정 대기" in output
    assert "nested-curve-node" in output and "retained-node" in output
    assert contents() == before


def test_plain_status_cli_prints_waiting_nodes_at_bottom_without_watch(monkeypatch, capsys):
    import sys

    data = report([{"host": "run284000-wts-2-g1234", "state": "WAIT", "last_age": 5}])
    monkeypatch.setattr(sys, "argv", ["mbpp_status.py", "--root", data["suites"][0]["root"]])
    monkeypatch.setattr(dashboard, "snapshot", lambda roots: data)
    assert dashboard.main() == 0
    text = capsys.readouterr().out
    footer = text[text.index("작업 없는 노드:"):]
    assert "1. run284000-wts-2-g1234 | WAIT | 작업 배정 대기" in footer
    assert text.index("NODE ASSIGNMENTS") < text.index("작업 없는 노드:")
    assert " ".join(text.split()).endswith(dashboard.render_idle_nodes(data)[-1])
