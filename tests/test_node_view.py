import importlib.util
import os
from pathlib import Path
import stat
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("node_view", ROOT / "scripts/_node_view.py")
view = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(view)


def logs(root, host, *, console=None, keepalive=None, pid=None, launcher=None):
    directory = root / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    if console is not None:
        (directory / f"console.{host}_.log").write_text(console)
    if launcher is not None:
        (directory / f"launcher.{host}_.log").write_text(launcher)
    if keepalive is not None:
        (directory / f"keepalive.{host}_.log").write_text(keepalive)
    if pid is not None:
        (directory / f"launcher.{host}_.pid").write_text(f"{pid}\n")


def test_launcher_nodes_classify_each_host_from_logs_pids_and_tasks(tmp_path, monkeypatch):
    now = time.time()
    monkeypatch.setattr(view.socket, "gethostname", lambda: "node-a")
    logs(tmp_path, "node-a", console="[launcher-start] x\n[hold] pass 2 ended rc=0\n[holding] node retained\n",
         keepalive="[keepalive] pid=555 parent=444 devices=[0, 1, 2, 3] period=0.25s\n", pid=os.getpid())
    logs(tmp_path, "node-b", console="[launcher-start] x\n[waiting] no claimable task; busy=...\n",
         keepalive="[keepalive] pid=1 parent=2 devices=[0]\n[keepalive] stopped\n", pid=1)
    logs(tmp_path, "node-c", launcher="[launcher-start] x\n[launcher-exit] pid=9 mode=run rc=78 utc=x\n")
    logs(tmp_path, "node-d", console="[claimed] host=node-d task=s1/t25/prefix\n")
    logs(tmp_path, "node-e", console="[launcher-start] x\n[launcher-exit] pid=3 mode=run rc=0 utc=x\n",
         keepalive="[keepalive] pid=77 parent=3 devices=[0]\n")
    tasks = [{"status": "RUNNING", "host": "node-d", "seed": 1, "step": 25, "arm": "prefix", "phase": "prefix-train"},
             {"status": "STALE", "host": "node-f_", "seed": 2, "step": 50, "arm": "random_reduced", "phase": "train"}]
    nodes = {item["host"]: item for item in view.launcher_nodes(tmp_path, tasks, now=now)}
    assert nodes["node-a"]["state"] == "HOLD" and nodes["node-a"]["launcher_alive"] is True and nodes["node-a"]["keepalive"] == "busy"
    assert nodes["node-b"]["state"] == "WAIT" and nodes["node-b"]["launcher_alive"] is None and nodes["node-b"]["keepalive"] == "stopped"
    assert nodes["node-c"]["state"] == "BLOCKED"
    assert nodes["node-d"]["state"] == "RUN" and nodes["node-d"]["task"] == "s1/t25 prefix" and nodes["node-d"]["phase"] == "prefix-train"
    assert nodes["node-e"]["state"] == "EXITED" and nodes["node-e"]["keepalive"] == "orphan?"
    assert nodes["node-f"]["state"] == "STALE" and nodes["node-f"]["task"] == "s2/t50 random_reduced"
    assert [item["host"] for item in view.launcher_nodes(tmp_path, tasks, now=now)] == sorted(nodes)


def test_launcher_nodes_treat_dead_pid_as_exited_on_this_host(tmp_path, monkeypatch):
    monkeypatch.setattr(view.socket, "gethostname", lambda: "node-a")
    logs(tmp_path, "node-a", console="[holding] node retained\n", pid=2**22-1)
    item = view.launcher_nodes(tmp_path, [], now=time.time())[0]
    assert item["launcher_alive"] is False and item["state"] == "EXITED"


def test_mbpp_guard_namespace_does_not_invent_another_node(tmp_path, monkeypatch):
    monkeypatch.setenv("EXPERIMENTS_NODE_ID", "node-mbpp")
    root = tmp_path / "runs/selection-switch-mbpp-v1"
    node_logs = view.node_launcher_logs(root)
    node_logs.mkdir(parents=True)
    (node_logs / "launcher.mbpp.node-mbpp_.pid").write_text(str(os.getpid()))
    (node_logs / "console.mbpp.node-mbpp_.log").write_text("[holding] node retained (node busy)\n")
    (node_logs / "keepalive.node-mbpp_.log").write_text("[keepalive] pid=12 devices=[0]\n")
    nodes = view.launcher_nodes(root, [])
    assert len(nodes) == 1 and nodes[0]["host"] == "node-mbpp"
    assert nodes[0]["state"] == "HOLD" and nodes[0]["launcher_alive"] is True
    assert nodes[0]["keepalive"] == "busy"
    (node_logs / "console.mbpp.node-mbpp_.log").write_text("[node-launcher-exit] pid=1 rc=143 owner=mbpp-guard\n")
    assert view.launcher_nodes(root, [])[0]["state"] == "EXITED"


def test_local_gpus_without_nvidia_smi_and_with_a_fake_one(tmp_path, monkeypatch):
    monkeypatch.setattr(view.shutil, "which", lambda name: None)
    assert view.local_gpus()["available"] is False
    fake = tmp_path / "nvidia-smi"
    fake.write_text('#!/usr/bin/env bash\ncase "$*" in *query-gpu*) printf "0, 727, 81559, 3\\n1, 727, 81559, 2\\n" ;; '
                    f'*query-compute-apps*) printf "{os.getpid()}, 727, GPU-x\\n{os.getpid()}, 727, GPU-y\\n" ;; esac\n')
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(view.shutil, "which", lambda name: str(fake))
    result = view.local_gpus()
    assert result["available"] is True
    assert [g["memory_used_mib"] for g in result["gpus"]] == [727, 727]
    assert len(result["processes"]) == 2 and all(p["memory_mib"] == 727 for p in result["processes"])
    assert all(p["role"] in {"other", "worker", "score", "train", "keepalive", "nccl-probe", "torchrun", "eval"} for p in result["processes"])


def test_renderers_fit_narrow_and_wide_terminals(tmp_path, monkeypatch):
    monkeypatch.setattr(view.socket, "gethostname", lambda: "node-a")
    logs(tmp_path, "node-a", console="[holding] node retained\n", keepalive="[keepalive] pid=5 parent=4 devices=[0]\n", pid=os.getpid())
    nodes = view.launcher_nodes(tmp_path, [], now=time.time())
    def table(headers, rows, widths):
        return ["  ".join(str(cell)[:w].ljust(w) for cell, w in zip(row, widths)).rstrip() for row in [headers, *rows]]
    for width in (80, 100, 120):
        for line in view.render_nodes(nodes, table, width):
            assert len(line) <= width, (width, line)
        gpus = {"host": "node-a", "available": True, "gpus": [{"index": 0, "memory_used_mib": 727, "memory_total_mib": 81559, "utilization": 3}],
                "processes": [{"pid": 5, "memory_mib": 727, "role": "keepalive", "cmdline": "python scripts/_gpu_keepalive.py"}]}
        rendered = view.render_local_gpus(gpus, table, width)
        assert any("keepalive" in line for line in rendered)
        assert all(len(line) <= width for line in rendered[1:-1])


def test_node_launcher_logs_next_to_the_roots_are_read_and_summarized(tmp_path, monkeypatch):
    now = time.time()
    monkeypatch.setattr(view.socket, "gethostname", lambda: "node-x")
    root = tmp_path / "runs" / "selection-switch-v1"
    node_logs = tmp_path / "runs" / "experiments" / "logs"
    node_logs.mkdir(parents=True)
    # Old per-root launcher log says EXITED; the fresher node launcher console says it is holding, and why.
    logs(root, "node-a", launcher="[launcher-start] x\n[launcher-exit] pid=9 mode=run rc=1 utc=x\n")
    os.utime(root / "logs" / "launcher.node-a_.log", (now-300, now-300))
    (node_logs / "console.node-a_.log").write_text(
        "[pass 3] MoPPS comparison\n[launcher-exit] pid=9 mode=run rc=1 utc=x\n"
        "[hold] pass 3 ended (switch rc=1: worker reported failed tasks | mopps rc=0: nothing left to claim); next pass in 600s\n"
        "[holding] node retained (switch rc=1: worker reported failed tasks | mopps rc=0: nothing left to claim); next pass in 585s\n")
    (node_logs / "launcher.node-a_.pid").write_text("1\n")
    # An inner launcher's exit inside the node console is just the end of a pass.
    (node_logs / "console.node-b_.log").write_text("[pass 1] selection switch\n[launcher-exit] pid=4 mode=run rc=0 utc=x\n")
    # A holding node whose console went silent is gone, not holding.
    (node_logs / "console.node-c_.log").write_text("[holding] node retained; next pass in 30s\n")
    os.utime(node_logs / "console.node-c_.log", (now-600, now-600))
    (node_logs / "console.node-d_.log").write_text("[node-launcher-exit] pid=3 rc=78 utc=x\n")
    (node_logs / "console.node-e_.log").write_text("[pass 2] selection switch\n[nccl-preflight] probing\n")
    # A host cooling down after a GPU fault, and one that just pulled new code and restarted in place.
    (node_logs / "console.node-f_.log").write_text("[pass 4] selection switch\n[cooldown] host=node-f: GPU fault recorded 120s ago\n")
    (node_logs / "console.node-g_.log").write_text("[pull] checkout moved abc1234 -> def5678; restarting this launcher with the new code\n")
    nodes = {item["host"]: item for item in view.launcher_nodes(root, [], now=now)}
    assert nodes["node-f"]["state"] == "COOL" and nodes["node-g"]["state"] == "LIVE"
    for host in ("node-f", "node-g"):
        del nodes[host]
    assert nodes["node-a"]["state"] == "HOLD" and nodes["node-a"]["launcher_pid"] == 1
    assert nodes["node-a"]["reason"] == "switch rc=1: worker reported failed tasks | mopps rc=0: nothing left to claim"
    assert nodes["node-b"]["state"] == "LIVE"
    assert nodes["node-c"]["state"] == "GONE"
    assert nodes["node-d"]["state"] == "BLOCKED"
    assert nodes["node-e"]["state"] == "ADMIT"
    summary = view.summarize(list(nodes.values()))
    assert summary["live"] == 3 and summary["counts"] == {"HOLD": 1, "LIVE": 1, "GONE": 1, "BLOCKED": 1, "ADMIT": 1}
    # Idle: live hosts with no task, oldest first; a training host is not idle.
    assert [item["host"] for item in view.idle(list(nodes.values()))] == ["node-a", "node-b", "node-e"]
    idle_line = view.render_idle(list(nodes.values()))
    assert idle_line.startswith("IDLE  3 node(s) with GPUs and no task: node-a (HOLD ") and "node-e (ADMIT" in idle_line
    assert view.render_idle([{**nodes["node-b"], "task": "s1/t25 gated"}]) == "IDLE  none: every live node has a task"
    assert view.render_summary(list(nodes.values())) == "NODES  3 live  |  ADMIT 1  HOLD 1  LIVE 1"
    assert "not live: BLOCKED 1  GONE 1" in view.render_summary(list(nodes.values()), all_nodes=True)
    assert view.render_summary([]) == "NODES  0 live  |  no launcher evidence yet"
    def table(headers, rows, widths):
        return ["  ".join(str(cell)[:w].ljust(w) for cell, w in zip(row, widths)).rstrip() for row in [headers, *rows]]
    wide = view.render_nodes(list(nodes.values()), table, 120)
    assert any("worker reported failed tasks" in line for line in wide)
    for width in (80, 100, 120):
        assert all(len(line) <= width for line in view.render_nodes(list(nodes.values()), table, width))


def test_old_claimed_lines_are_dead_hosts_not_running_nodes(tmp_path, monkeypatch):
    """Cluster hosts change with every job: nine nodes must never show as 15 RUN."""
    now = time.time()
    monkeypatch.setattr(view.socket, "gethostname", lambda: "elsewhere")
    for index in range(15):
        logs(tmp_path, f"old-{index:02d}", console="[pass 1] selection switch\n[claimed] host=x pid=1 task=s0/t25/prefix\n")
        os.utime(tmp_path / "logs" / f"console.old-{index:02d}_.log", (now-8*3600, now-8*3600))
    logs(tmp_path, "recent-dead", console="[claimed] host=x pid=1 task=s1/t25/prefix\n")
    os.utime(tmp_path / "logs" / "console.recent-dead_.log", (now-1200, now-1200))
    logs(tmp_path, "training", console="[claimed] host=training pid=1 task=s3/t25/selection_full\n")
    os.utime(tmp_path / "logs" / "console.training_.log", (now-7200, now-7200))
    logs(tmp_path, "starting", console="[pass 1] selection switch\n")
    tasks = [{"status": "RUNNING", "host": "training", "seed": 3, "step": 25, "arm": "selection_full", "phase": "train"}]
    nodes = view.launcher_nodes(tmp_path, tasks, now=now)
    summary = view.summarize(nodes)
    assert summary["live"] == 2 and summary["counts"]["RUN"] == 1 and summary["counts"]["LIVE"] == 1, summary
    assert summary["counts"]["GONE"] == 16
    assert view.render_summary(nodes) == "NODES  2 live  |  RUN 1  LIVE 1"
    assert [item["host"] for item in view.listed(nodes)] == ["training", "starting"]
    assert len(view.listed(nodes, all_nodes=True)) == 18
    def table(headers, rows, widths):
        return ["  ".join(str(cell)[:w].ljust(w) for cell, w in zip(row, widths)).rstrip() for row in [headers, *rows]]
    rendered = view.render_nodes(nodes, table, 120)
    assert rendered[-1] == "  16 inactive node(s) hidden; --all shows history."
    assert not any("old-0" in line for line in rendered)
    assert not any("recent-dead" in line for line in rendered)
    assert any("recent-dead" in line for line in view.render_nodes(nodes, table, 120, all_nodes=True))


def test_node_display_sorts_live_states_and_numbers_without_mutating_records():
    nodes = [{"host": host, "state": state} for host, state in
             [("node-10", "RUN"), ("node-1", "EXITED"), ("node-2", "RUN"), ("node-3", "HOLD")]]
    before = [dict(item) for item in nodes]
    assert [item["host"] for item in view.listed(nodes)] == ["node-2", "node-10", "node-3"]
    assert [item["host"] for item in view.listed(nodes, all_nodes=True)] == ["node-2", "node-10", "node-3", "node-1"]
    assert nodes == before


@pytest.mark.parametrize("reverse", [False, True])
def test_old_task_cannot_hide_a_current_live_task_on_the_same_node(tmp_path, reverse):
    tasks = [{"status": state, "host": "node-2", "seed": seed, "step": 25, "arm": "random_reduced"}
             for state, seed in [("RUNNING", 1), ("STALE", 0)]]
    if reverse:
        tasks.reverse()
    node = view.launcher_nodes(tmp_path, tasks)[0]
    assert node["state"] == "RUN" and node["task"] == "s1/t25 random_reduced"


def test_hiding_all_inactive_nodes_keeps_their_files(tmp_path):
    logs(tmp_path, "dead-node", console="[launcher-exit] rc=0\n", pid=999999999)
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in tmp_path.rglob("*") if path.is_file()}
    nodes = view.launcher_nodes(tmp_path, [])
    rendered = view.render_nodes(nodes, lambda *args: [], 80)
    assert rendered == ["No live nodes observed.", "  1 inactive node(s) hidden; --all shows history."]
    assert nodes[0]["host"] == "dead-node" and nodes[0]["state"] == "EXITED"
    assert before == {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in before}


def test_mbpp_namespace_excludes_unattributed_shared_math_and_legacy_logs(tmp_path):
    root = tmp_path / "runs/selection-switch-mbpp-v1"
    shared = view.node_launcher_logs(root)
    shared.mkdir(parents=True)
    (shared / "console.math-node_.log").write_text("[holding] math work\n")
    (shared / "launcher.math-node_.pid").write_text("1\n")
    (shared / "launcher.legacy-node_.log").write_text("[pass 1] selection switch\n")
    (shared / "console.mbpp.code-node_.log").write_text("[waiting] prefixes are not ready\n")
    (shared / "launcher.mbpp.code-node_.pid").write_text("2\n")
    (shared / "launcher.mbpp.second-code-node_.log").write_text("[nccl-preflight] probing\n")
    logs(root, "root-node", console="[recover-cost] closed interrupted event\n")
    scoped = {row["host"]: row for row in view.launcher_nodes(root, [], node_namespace="mbpp")}
    assert set(scoped) == {"code-node", "second-code-node", "root-node"}
    assert scoped["code-node"]["state"] == "WAIT" and scoped["code-node"]["launcher_pid"] == 2
    assert scoped["second-code-node"]["state"] == "ADMIT"
    assert scoped["root-node"]["source_root"] == str(root.resolve())
    unscoped = {row["host"] for row in view.launcher_nodes(root, [])}
    assert unscoped == set(scoped) | {"math-node", "legacy-node"}


@pytest.mark.parametrize("last,state", [
    ("[waiting] shared prefixes not ready", "WAIT"),
    ("[holding] node retained (peer work active)", "HOLD"),
    ("[nccl-preflight] probing", "ADMIT"),
    ("[recover-cost] open events inspected; queue continues", "LIVE"),
])
def test_mbpp_controller_visible_without_prepared_root_and_without_mutation(tmp_path, monkeypatch, last, state):
    root = tmp_path / "runs/selection-switch-mbpp-difficulty-v1"
    shared = view.node_launcher_logs(root)
    shared.mkdir(parents=True)
    console = shared / "console.mbpp.code-node_.log"
    console.write_text(last + "\n")
    stamp = console.stat().st_mtime
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()}
    monkeypatch.setattr(view, "local_gpus", lambda: pytest.fail("read-only node metadata must not query GPUs"))
    nodes = view.launcher_nodes(root, [], now=stamp + 10, node_namespace="mbpp")
    assert len(nodes) == 1
    assert nodes[0]["host"] == "code-node" and nodes[0]["state"] == state
    assert nodes[0]["source_root"] is None and nodes[0]["source_log"] == str(console)
    assert nodes[0]["last_age"] == 10
    assert not root.exists()
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()}


@pytest.mark.parametrize("root_is_newer", [False, True])
def test_node_provenance_tracks_only_the_freshest_selected_log(tmp_path, root_is_newer):
    root = tmp_path / "runs/selection-switch-mbpp-quality-v1"
    shared = view.node_launcher_logs(root)
    shared.mkdir(parents=True)
    local = root / "logs/console.code-node_.log"
    logs(root, "code-node", console="[nccl-preflight] probing quality\n")
    console = shared / "console.mbpp.code-node_.log"
    console.write_text("[holding] node retained (all suites waiting)\n")
    now = 1000
    newest, oldest = (local, console) if root_is_newer else (console, local)
    os.utime(newest, (now - 5, now - 5))
    os.utime(oldest, (now - 20, now - 20))
    node = view.launcher_nodes(root, [], now=now, node_namespace="mbpp")[0]
    assert node["source_log"] == str(newest)
    assert node["source_root"] == (str(root.resolve()) if root_is_newer else None)
    assert node["last_age"] == 5
    assert node["state"] == ("ADMIT" if root_is_newer else "HOLD")


def test_mbpp_shared_keepalive_requires_prior_controller_or_root_task_evidence(tmp_path):
    root = tmp_path / "runs/selection-switch-mbpp-v1"
    shared = view.node_launcher_logs(root)
    shared.mkdir(parents=True)
    (shared / "console.mbpp.code-node_.log").write_text("[holding] waiting for peer\n")
    for host in ("code-node", "task-node", "math-ghost"):
        (shared / f"keepalive.{host}_.log").write_text("[keepalive] pid=99 devices=[0]\n")
    tasks = [{"status": "RUNNING", "host": "task-node", "seed": 0, "step": 25, "arm": "random_full"}]
    nodes = {row["host"]: row for row in view.launcher_nodes(root, tasks, node_namespace="mbpp")}
    assert set(nodes) == {"code-node", "task-node"}
    assert all(row["keepalive"] == "busy" for row in nodes.values())
    assert nodes["task-node"]["state"] == "RUN"
    assert "math-ghost" in {row["host"] for row in view.launcher_nodes(root, tasks)}


def test_mbpp_shared_waiting_does_not_inherit_old_dispatched_suite(tmp_path):
    root = tmp_path / "runs/selection-switch-mbpp-v1"
    shared = view.node_launcher_logs(root)
    shared.mkdir(parents=True)
    detail = "[holding] " + "all suite prerequisites pending; " * 12
    (shared / "console.mbpp.code-node_.log").write_text(f"[dispatch] root={root} checkout=old\n{detail}\n")
    node = view.launcher_nodes(root, [], node_namespace="mbpp")[0]
    assert node["source_root"] is None and node["detail"] == detail
    assert node["state"] == "HOLD" and not node["task"]
    assert len(view.launcher_nodes(root, [])[0]["detail"]) == 160


def test_mbpp_namespace_rejects_unknown_filter_without_creating_paths(tmp_path):
    root = tmp_path / "not-created"
    with pytest.raises(ValueError, match="node_namespace"):
        view.launcher_nodes(root, [], node_namespace="typo")
    assert not root.exists()


def test_pid_only_starting_node_has_age_without_claiming_remote_liveness(tmp_path, monkeypatch):
    root = tmp_path / "runs/selection-switch-mbpp-v1"
    shared = view.node_launcher_logs(root)
    shared.mkdir(parents=True)
    pid_file = shared / "launcher.mbpp.starting-node_.pid"
    pid_file.write_text("42\n")
    os.utime(pid_file, (995, 995))
    monkeypatch.setattr(view, "node_id", lambda: "another-node")
    monkeypatch.setattr(view.os, "kill", lambda *_: pytest.fail("a remote PID is not locally verifiable"))
    node = view.launcher_nodes(root, [], now=1000, node_namespace="mbpp")[0]
    assert node["host"] == "starting-node" and node["launcher_pid"] == 42
    assert node["pid_age"] == 5 and node["last_age"] is None
    assert node["launcher_alive"] is None and node["state"] == "-"
    assert node["source_log"] is None and node["source_root"] is None


def test_pid_age_does_not_replace_selected_log_age(tmp_path):
    root = tmp_path / "runs/selection-switch-mbpp-v1"
    shared = view.node_launcher_logs(root)
    shared.mkdir(parents=True)
    pid_file = shared / "launcher.mbpp.code-node_.pid"
    console = shared / "console.mbpp.code-node_.log"
    pid_file.write_text("42\n")
    console.write_text("[waiting] for shared prefixes\n")
    os.utime(pid_file, (500, 500))
    os.utime(console, (990, 990))
    node = view.launcher_nodes(root, [], now=1000, node_namespace="mbpp")[0]
    assert node["pid_age"] == 500 and node["last_age"] == 10 and node["state"] == "WAIT"


@pytest.mark.parametrize("status", ["DONE", "EVAL", "STALE"])
def test_fresh_task_heartbeat_keeps_owner_visible_without_relabeling_saved_task(tmp_path, status):
    root = tmp_path / "runs/selection-switch-mbpp-v1"
    shared = view.node_launcher_logs(root)
    shared.mkdir(parents=True)
    (shared / "keepalive.curve-node_.log").write_text("[keepalive] pid=42 devices=[0]\n")
    task = {"status": status, "heartbeat_fresh": True, "host": "curve-node", "seed": 3,
            "step": 100, "arm": "random_reduced", "phase": "curve-evaluate"}
    node = view.launcher_nodes(root, [task], node_namespace="mbpp")[0]
    assert node["state"] == "RUN" and node["host"] == "curve-node"
    assert node["task"] == "s3/t100 random_reduced" and node["phase"] == "curve-evaluate"
    assert node["keepalive"] == "busy" and task["status"] == status


def test_saved_task_without_fresh_heartbeat_does_not_create_live_node(tmp_path):
    task = {"status": "EVAL", "heartbeat_fresh": False, "host": "old-node", "seed": 3,
            "step": 100, "arm": "random_reduced", "phase": "curve-evaluate"}
    assert view.launcher_nodes(tmp_path, [task], node_namespace="mbpp") == []
