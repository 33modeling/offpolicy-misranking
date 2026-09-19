"""Every current MBPP node maps to its work without changing completion state."""

import os
import re
from copy import deepcopy
from pathlib import Path

import pytest
from test_mbpp_status_dashboard import dashboard
from test_selection_switch_status import (
    completed_prefix,
    convergence_root,
    point,
    prefix,
    prepared,
    published,
    running,
)

NOW = 1900000000.


@pytest.fixture(autouse=True)
def no_gpu_query(monkeypatch):
    def forbidden():
        pytest.fail("node assignment status must not query a GPU")

    monkeypatch.setattr(dashboard.switch_status.node_view, "local_gpus", forbidden)


def roots_at(tmp_path):
    return [tmp_path / "runs" / name for name in (
        "selection-switch-mbpp-v1", "selection-switch-mbpp-quality-v1", "selection-switch-mbpp-difficulty-v1")]


def contents(base):
    return {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in base.rglob("*") if path.is_file()}


def host_row(data, host):
    matches = [node for node in dashboard.node_assignments(data) if node["host"] == host]
    assert len(matches) == 1
    return matches[0]


@pytest.mark.parametrize("active_index", [2, 10])
@pytest.mark.parametrize("reverse", [False, True])
def test_server_order_is_natural_name_order_not_run_wait_priority(tmp_path, active_index, reverse):
    root = roots_at(tmp_path)[1]
    hosts = {index: f"run284000-wts-{index}-g1234" for index in (2, 10)}
    old_host = "run283999-wts-1-g0000"
    nodes = [{"host": host, "state": "WAIT", "last_age": 5} for host in hosts.values()]
    nodes.append({"host": old_host, "state": "EXITED", "last_age": 900})
    data = {"suites": [{"root": str(root), "nodes": nodes[::-1] if reverse else nodes,
        "tasks": [{"kind": "branch", "status": "RUNNING", "host": hosts[active_index],
                   "seed": 0, "step": 25, "arm": "random_reduced", "phase": "train",
                   "directory": "states/s0-t25/points/view-25/random_reduced"}]}]}
    before = deepcopy(data)
    assert [node["host"] for node in dashboard.node_assignments(data)] == [hosts[2], hosts[10], old_host]
    for show_history in (False, True):
        rows = [line for line in dashboard.render_nodes(data, width=120, all_nodes=show_history)
                if re.match(r"^\d+\. ", line)]
        assert rows[0].startswith(f"1. {hosts[2]} ->")
        assert rows[1].startswith(f"2. {hosts[10]} ->")
        assert len(rows) == (3 if show_history else 2)
        if show_history:
            assert rows[2].startswith(f"3. {old_host} ->")
        active_row = rows[0 if active_index == 2 else 1]
        idle_row = rows[1 if active_index == 2 else 0]
        assert "| RUN |" in active_row and "| WAIT |" in idle_row
    assert data == before


def test_live_curve_heartbeat_keeps_sealed_result_eval_and_node_visible(tmp_path):
    root = roots_at(tmp_path)[1]
    convergence_root(root)
    completed_prefix(root)
    directory = point(root) / "random_reduced"
    published(directory)
    running(directory, "eval-node", now=NOW, phase="curve")
    before = contents(tmp_path)
    data = dashboard.snapshot([root], now=NOW)
    suite = data["suites"][0]
    task = next(task for task in suite["tasks"] if task["directory"] == str(directory.relative_to(root)))
    assert task["status"] == "EVAL" and task["heartbeat_fresh"] is True
    assert suite["branch_counts"]["EVAL"] == 1 and suite["training_published"] == 1
    assert suite["active_nodes"] == 1
    node = host_row(data, "eval-node")
    assert node["state"] == "RUN"
    assert any(Path(suite_root) == root and assignment["status"] == "EVAL"
               for suite_root, assignment in node["assignments"])
    output = dashboard.render(data)
    assert "eval-node" in output and "NODE ASSIGNMENTS" in output
    assert re.search(r"NODES\s+1 current", output)
    assert before == contents(tmp_path)


def test_published_prefix_with_current_heartbeat_retains_done_and_node(tmp_path):
    root = roots_at(tmp_path)[0]
    prepared(root)
    completed_prefix(root)
    running(prefix(root), "prefix-publication-node", now=NOW, phase="prefix-train")
    data = dashboard.snapshot([root], now=NOW)
    task = next(task for task in data["suites"][0]["tasks"]
                if task["kind"] == "prefix" and task["seed"] == 0 and task["step"] == 25)
    assert task["status"] == "DONE" and task["heartbeat_fresh"] is True
    node = host_row(data, "prefix-publication-node")
    assert any(assignment["kind"] == "prefix" for _, assignment in node["assignments"])
    assert "prefix-publication-node" in dashboard.render(data)


@pytest.mark.parametrize("suffix", ["curve-parent", "random_reduced/curve", "selection_full/curve/step-50"])
def test_nested_curve_phase_node_is_mapped_even_outside_branch_task(tmp_path, suffix):
    root = roots_at(tmp_path)[1]
    prepared(root)
    completed_prefix(root)
    directory = point(root) / suffix
    running(directory, "curve-node", now=NOW, phase="curve")
    data = dashboard.snapshot([root], now=NOW)
    node = host_row(data, "curve-node")
    assert any(task["kind"] == "phase" and task["arm"] == suffix and task["heartbeat_fresh"]
               for _, task in node["assignments"])
    assert "curve-node" in dashboard.render(data)


@pytest.mark.parametrize("width", [80, 100, 120])
def test_all_twelve_long_node_names_and_task_assignments_are_visible(tmp_path, width):
    root = roots_at(tmp_path)[0]
    prepared(root)
    hosts = []
    for index in range(12):
        seed, step = index // 3, (25, 50, 100)[index % 3]
        completed_prefix(root, seed, step)
        host = f"gpu-node-{index:02d}-" + "allocation-name-that-must-not-be-clipped-" * 2
        hosts.append(host)
        running(point(root, seed, step) / "random_reduced", host, now=NOW, phase="train")
    data = dashboard.snapshot([root], now=NOW)
    output = dashboard.render(data, width=width)
    # Long node names wrap without interleaved table columns or truncation.
    node_section = output.split("NODE ASSIGNMENTS", 1)[1]
    node_column = re.sub(r"\s+", "", node_section)
    assert re.search(r"NODES\s+12 current", output)
    assert len([node for node in dashboard.node_assignments(data) if node["state"] == "RUN"]) == 12
    for host in hosts:
        assert host in node_column
        assert host_row(data, host)["assignments"]
    assert "... more" not in output
    assert all(len(line) <= width for line in output.splitlines())
    numbered = [line for line in dashboard.render_nodes(data, width=width) if re.match(r"^\d+\. ", line)]
    assert len(numbered) == 12
    for index, host in enumerate(hosts, 1):
        assert numbered[index - 1].startswith(f"{index}. {host} -> ")


def test_one_host_in_two_suites_retains_both_assignments_and_counts_once(tmp_path):
    roots = roots_at(tmp_path)[:2]
    for root in roots:
        prepared(root)
        completed_prefix(root)
        running(point(root) / "random_reduced", "shared-node", now=NOW, phase="train")
    data = dashboard.snapshot(roots, now=NOW)
    node = host_row(data, "shared-node")
    assert {Path(suite_root) for suite_root, _ in node["assignments"]} == set(roots)
    assert len(node["assignments"]) == 2
    output = dashboard.render(data)
    assert re.search(r"NODES\s+1 current", output)
    assignment_section = output.split("NODE ASSIGNMENTS", 1)[1]
    assert "On-policy · 선택비용 포함" in assignment_section and "On-policy · 선택비용 별도" in assignment_section
    assert assignment_section.count("1. shared-node ->") == 2
    assert "2. shared-node" not in assignment_section


@pytest.mark.parametrize("line,state", [
    ("[waiting] all shared prefixes pending", "WAIT"),
    ("[holding] node retained (peer work active)", "HOLD"),
    ("[recover-cost] open events inspected", "LIVE"),
    ("[nccl-preflight] probe running", "ADMIT"),
])
def test_shared_mbpp_controller_visible_without_prepared_suite_and_math_is_excluded(tmp_path, line, state):
    roots = roots_at(tmp_path)
    logs = roots[0].parent / "experiments/logs"
    logs.mkdir(parents=True)
    for name, text in (("console.mbpp.code-node_.log", line),
                       ("console.math-node_.log", "[holding] unrelated math"),
                       ("keepalive.math-ghost_.log", "[keepalive] pid=99 devices=[0]")):
        path = logs / name
        path.write_text(text + "\n")
        os.utime(path, (NOW - 5, NOW - 5))
    before = contents(tmp_path)
    data = dashboard.snapshot(roots, now=NOW)
    node = host_row(data, "code-node")
    assert node["state"] == state and not node["assignments"]
    assert node.get("source_root") is None
    output = dashboard.render(data)
    assert "code-node -> 배정 없음 | WAIT" in output
    assert re.search(r"NODES\s+1 current", output)
    assert "math-node" not in output and "math-ghost" not in output
    assert all(not root.exists() for root in roots)
    assert before == contents(tmp_path)


def test_old_node_history_hidden_by_default_but_all_preserves_it_read_only(tmp_path):
    root = roots_at(tmp_path)[0]
    prepared(root)
    logs = root.parent / "experiments/logs"
    logs.mkdir(parents=True)
    for host, age in (("current-code-node", 5), ("old-code-node", 900)):
        log = logs / f"console.mbpp.{host}_.log"
        log.write_text("[holding] waiting for prerequisites\n")
        os.utime(log, (NOW - age, NOW - age))
    before = contents(tmp_path)
    data = dashboard.snapshot([root], now=NOW)
    output = dashboard.render(data)
    detailed = dashboard.render(data, all_tasks=True)
    assert "current-code-node" in output and "old-code-node" not in output
    assert "old-code-node" in detailed and "오래된 실행 기록" in detailed
    assert "1. current-code-node ->" in output
    assert "1. current-code-node ->" in detailed and "2. old-code-node ->" in detailed
    assert before == contents(tmp_path)


def test_recent_stale_task_without_launcher_log_remains_visible_as_unconfirmed(tmp_path):
    root = roots_at(tmp_path)[0]
    prepared(root)
    completed_prefix(root)
    running(point(root) / "random_reduced", "recent-stale-node", now=NOW - 88, phase="train")
    data = dashboard.snapshot([root], now=NOW)
    node = host_row(data, "recent-stale-node")
    assert node["state"] == "STALE" and node["current"] is True
    assert node["evidence_age"] == 90 and not node["assignments"]
    output = dashboard.render(data)
    assert "recent-stale-node -> 배정 없음 | WAIT | - | 실행 신호 끊김" in output
    assert re.search(r"NODES\s+1 current", output)


@pytest.mark.parametrize("reverse", [False, True])
def test_old_assignment_never_hides_same_nodes_current_work_in_another_suite(tmp_path, reverse):
    roots = roots_at(tmp_path)[:2]
    for root in roots:
        prepared(root)
        completed_prefix(root)
    running(point(roots[0]) / "random_reduced", "reused-node", now=NOW - 900, phase="train")
    running(point(roots[1]) / "selection_reduced", "reused-node", now=NOW, phase="fresh-r-candidate")
    data = dashboard.snapshot(list(reversed(roots)) if reverse else roots, now=NOW)
    node = host_row(data, "reused-node")
    assert node["state"] == "RUN" and node["current"] is True
    assert len(node["assignments"]) == 1
    suite_root, task = node["assignments"][0]
    assert Path(suite_root) == roots[1] and task["arm"] == "selection_reduced"
    output = dashboard.render(data)
    assert "reused-node -> On-policy · 선택비용 별도 / seed 0 / step 25 / Selection" in " ".join(output.split())


def test_recent_pid_after_old_controller_log_is_unknown_current_not_running(tmp_path):
    root = roots_at(tmp_path)[0]
    logs = root.parent / "experiments/logs"
    logs.mkdir(parents=True)
    old_log = logs / "console.mbpp.pid-pending-node_.log"
    old_log.write_text("[holding] prior controller waited for prerequisites\n")
    os.utime(old_log, (NOW - 900, NOW - 900))
    pid_file = logs / "launcher.mbpp.pid-pending-node_.pid"
    pid_file.write_text("1234\n")
    os.utime(pid_file, (NOW - 5, NOW - 5))
    before = contents(tmp_path)
    data = dashboard.snapshot([root], now=NOW)
    node = host_row(data, "pid-pending-node")
    assert node["state"] == "UNKNOWN" and node["current"] is True
    assert not node["assignments"] and node["evidence_age"] == 5
    output = dashboard.render(data)
    assert "pid-pending-node -> 배정 없음 | WAIT | - | 배정 확인 안 됨" in output
    assert before == contents(tmp_path)


@pytest.mark.parametrize("all_tasks", [False, True])
def test_simple_mapping_shows_busy_and_two_unassigned_nodes_without_extra_diagnostics(tmp_path, all_tasks):
    roots = roots_at(tmp_path)
    for root in roots:
        prepared(root)
    running(point(roots[1]) / "random_reduced", "busy-node", now=NOW, phase="train")
    logs = roots[0].parent / "experiments/logs"
    logs.mkdir(parents=True)
    names = ["run1234-mbpp-2-long-cluster-allocation-g1234", "run1234-mbpp-10-long-cluster-allocation-g5678"]
    for host, state, reason in ((names[0], "holding", "peer work active"),
                                (names[1], "waiting", "shared prefixes pending"),
                                ("old-node", "holding", "old history")):
        path = logs / f"console.mbpp.{host}_.log"
        path.write_text(f"[{state}] {reason}\n")
        age = 900 if host == "old-node" else 5
        os.utime(path, (NOW - age, NOW - age))
    before = contents(tmp_path)
    output = dashboard.render(dashboard.snapshot(roots, now=NOW), width=80, all_tasks=all_tasks)
    mapping = output.split("NODE ASSIGNMENTS", 1)[1].split("ROOT ", 1)[0]
    joined = re.sub(r"\s+", " ", mapping)
    assert f"{names[0]} -> 배정 없음 | WAIT" in joined
    assert f"{names[1]} -> 배정 없음 | WAIT" in joined
    assert mapping.index(names[0]) < mapping.index(names[1])
    assert "busy-node -> On-policy · 선택비용 별도 / seed 0 / step 25 / Random" in joined
    assert ("old-node" in mapping) is all_tasks
    assert not any(text in mapping for text in ("PID", "AGE", "PHASE", "peer work active", "checking receipts"))
    assert all(len(line) <= 80 for line in output.splitlines())
    assert before == contents(tmp_path)


def test_unassigned_mapping_does_not_invent_a_suite_for_recovery_admission_or_unknown(tmp_path):
    root = roots_at(tmp_path)[0]
    logs = root.parent / "experiments/logs"
    logs.mkdir(parents=True)
    for host, line in (("wait-node", "[waiting] peers busy"),
                       ("probe-node", "[nccl-preflight] probing GPUs"),
                       ("recover-node", "[recover-cost] checking receipts")):
        path = logs / f"console.mbpp.{host}_.log"
        path.write_text(line + "\n")
        os.utime(path, (NOW - 5, NOW - 5))
    pid = logs / "launcher.mbpp.unknown-node_.pid"
    pid.write_text("1234\n")
    os.utime(pid, (NOW - 5, NOW - 5))
    mapping = "\n".join(dashboard.render_nodes(dashboard.snapshot([root], now=NOW), width=120))
    assert "wait-node -> 배정 없음 | WAIT | - | 작업 배정 대기" in mapping
    assert "probe-node -> 배정 없음 | WAIT | - | 장치 점검 중" in mapping
    assert "recover-node -> 배정 없음 | WAIT | - | 작업 배정 확인 중" in mapping
    assert "unknown-node -> 배정 없음 | WAIT | - | 배정 확인 안 됨" in mapping


def test_mapping_states_absence_of_evidence_when_no_nodes_are_observed(tmp_path):
    mapping = "\n".join(dashboard.render_nodes(dashboard.snapshot(roots_at(tmp_path), now=NOW), width=120))
    assert "NODES 0 current" in mapping
    assert "No current MBPP node evidence." in mapping


def test_parent_branch_and_curve_phase_share_one_numbered_experiment_row(tmp_path):
    root = roots_at(tmp_path)[1]
    convergence_root(root)
    completed_prefix(root)
    directory = point(root) / "random_reduced"
    published(directory)
    running(directory, "curve-worker", now=NOW, phase="curve")
    running(directory / "curve", "curve-worker", now=NOW, phase="curve-evaluation")
    before = contents(tmp_path)
    data = dashboard.snapshot([root], now=NOW)
    mapping = "\n".join(dashboard.render_nodes(data, width=120))
    assert mapping.count("1. curve-worker ->") == 1
    assert "On-policy · 선택비용 별도 / seed 0 / step 25 / Random | RUN |" in mapping
    assert '현재 단계 시간 한도 사용률' in mapping
    assert "단계: curve-evaluation" in mapping and "최종 평가 저장됨; 곡선 평가 남음 (재학습 없음)" in mapping
    assert "Random/curve" not in mapping
    assert "CURRENT RUN 1" in dashboard.render(data)
    suite = data["suites"][0]
    assert suite["branch_counts"]["EVAL"] == 1
    assert len([task for task in suite["tasks"] if task["kind"] == "branch"]) == 48
    assert before == contents(tmp_path)
