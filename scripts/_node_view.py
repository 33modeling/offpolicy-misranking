#!/usr/bin/env python3
"""What each node's launcher is doing, and what this node's GPUs hold.

Shared by the switch and MoPPS status views. Everything is read-only: launcher
pid files, console/launcher/keepalive logs under ROOT/logs, /proc of this user's
processes, and nvidia-smi on the current node.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import socket
import subprocess
import textwrap
import time


def node_id():
    """EXPERIMENTS_NODE_ID (hostname plus GPU suffix) when the launcher set it, else the hostname."""
    return os.environ.get("EXPERIMENTS_NODE_ID") or socket.gethostname()

ROLES = (("_gpu_keepalive.py", "keepalive"), ("selection_nccl_preflight.py", "nccl-probe"),
         ("train_mopps_grpo.py", "train"), ("train_selection_gate_grpo.py", "train"),
         ("train_policy_grpo.py", "train"), ("torch.distributed.run", "torchrun"),
         ("selection_switch_score.py", "score"), ("experiment.py", "score"),
         ("mopps_comparison_gpu.py", "worker"), ("selection_switch_gpu.py", "worker"))


def _tail_lines(path, size=8192):
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            handle.seek(max(0, handle.tell()-size))
            return handle.read().decode("utf-8", errors="replace").splitlines()
    except OSError:
        return []


def _last(lines):
    return next((line for line in reversed(lines) if line.strip()), "")


def _host_of(path, prefix):
    return path.name[len(prefix):].rsplit(".", 1)[0].rstrip("_")


LIVE_STATES = ("RUN", "ADMIT", "WAIT", "HOLD", "COOL", "LIVE")
STATE_ORDER = ("RUN", "ADMIT", "WAIT", "HOLD", "COOL", "LIVE", "STALE", "BLOCKED", "STOPPING", "EXITED", "GONE", "QUIET", "-")
# A holding or waiting launcher prints every 15s; longer silence means it is gone.
# A launcher log alone never proves training: hosts change with every cluster
# job, so an old "[claimed]" line is a dead host unless a task heartbeat is fresh.
HEARTBEAT_GRACE = 180.
# Hosts whose only evidence is older than this are counted, not listed.
LISTING_AGE = 6*3600.


def node_launcher_logs(root):
    """The combined node launcher (scripts/run_experiments.sh) logs next to both roots."""
    return Path(root).resolve().parent / "experiments" / "logs"


def classify(last, *, node_launcher):
    """State from the last log line. In the node launcher's console an inner
    launcher's exit is just the end of one pass, not the node leaving."""
    if last.startswith("[node-launcher-exit]"):
        return "BLOCKED" if "rc=78" in last else "EXITED"
    if last.startswith("[launcher-exit]"):
        if node_launcher:
            return "LIVE"
        return "BLOCKED" if "rc=78" in last else "EXITED"
    if last.startswith(("[hold]", "[holding]")):
        return "HOLD"
    if last.startswith("[waiting]"):
        return "WAIT"
    if last.startswith("[blocked]"):
        return "BLOCKED"
    if last.startswith("[cooldown]"):
        # A GPU fault recorded on this host; the launcher holds until the record expires.
        return "COOL"
    if last.startswith("[done]"):
        return "EXITED"
    if last.startswith("[nccl-preflight]"):
        return "ADMIT"
    if last.startswith(("[retry]", "[claimed]", "[gate]", "[grpo]", "[fresh_r]", "[pass ", "[clean]", "[pull]",
                        "[fault-reset]", "[fault-expired]", "[auto-waive]", "[recover-cost]", "[sweep ", "[watchdog]",
                        "[keepalive]", "[restart]", "[node-launcher-start]", "[stall]", "[waive]", "[curve",
                        "[measurement]", "[selection]", "[evaluate]", "[difficulty", "[hard")):
        # Active launcher; RUN itself comes only from a fresh task heartbeat.
        return "LIVE"
    if last.startswith("[stopping]"):
        return "STOPPING"
    return None


def hold_reason(last):
    """The part of a [hold]/[holding] line that says why the last pass ended."""
    if "(" in last and ")" in last and last.index("(") < last.index(")"):
        return last[last.index("(")+1:last.index(")")]
    return ""


def launcher_nodes(root, tasks, *, now=None):
    """One row per host that has launcher evidence under ROOT/logs or the node launcher's logs."""
    now = time.time() if now is None else now
    here = node_id().rstrip("_")
    hosts = {}

    def row(host):
        return hosts.setdefault(host, {"host": host, "launcher_pid": None, "launcher_alive": None,
                                       "state": "-", "detail": "", "last_age": None, "keepalive": "-",
                                       "task": "", "phase": "", "reason": ""})
    sources = [(Path(root) / "logs", False), (node_launcher_logs(root), True)]
    for logs, node_launcher in sources:
        for path in logs.glob("launcher.*.pid"):
            host = _host_of(path, "launcher.")
            try:
                pid = int(path.read_text().strip())
            except (OSError, ValueError):
                continue
            item = row(host)
            if item["launcher_pid"] is not None and not node_launcher:
                continue
            item["launcher_pid"] = pid
            if host == here:
                try:
                    os.kill(pid, 0)
                    item["launcher_alive"] = True
                except ProcessLookupError:
                    item["launcher_alive"] = False
                except PermissionError:
                    item["launcher_alive"] = True
    for logs, node_launcher in sources:
        for path in list(logs.glob("console.*.log")) + list(logs.glob("launcher.*.log")):
            host = _host_of(path, "console." if path.name.startswith("console.") else "launcher.")
            item = row(host)
            try:
                age = now - path.stat().st_mtime
            except OSError:
                continue
            if item["last_age"] is not None and age > item["last_age"]:
                continue
            item["last_age"] = age
            last = _last(_tail_lines(path))
            item["detail"] = last[:160]
            state = classify(last, node_launcher=node_launcher)
            if state is None:
                state = "LIVE" if age < HEARTBEAT_GRACE else "QUIET"
            elif state in LIVE_STATES and age > HEARTBEAT_GRACE:
                state = "GONE"
            item["state"] = state
            item["reason"] = hold_reason(last) if state == "HOLD" else ""
    for logs, _ in sources:
        for path in logs.glob("keepalive.*.log"):
            host = _host_of(path, "keepalive.")
            last = _last(_tail_lines(path, 2048))
            item = row(host)
            if last.startswith("[keepalive] stopped"):
                item["keepalive"] = "stopped"
            elif "pid=" in last:
                item["keepalive"] = "busy"
            elif last:
                item["keepalive"] = "off"
    for task in tasks:
        if task.get("status") in {"RUNNING", "STALE"} and task.get("host"):
            item = row(str(task["host"]).rstrip("_"))
            item["state"] = "RUN" if task["status"] == "RUNNING" else "STALE"
            item["task"] = f"s{task['seed']}/t{task['step']} {task['arm']}"
            item["phase"] = task.get("phase", "")
    for item in hosts.values():
        if item["launcher_alive"] is False and item["state"] in {"HOLD", "COOL", "WAIT", "LIVE", "QUIET", "ADMIT"}:
            item["state"] = "EXITED"
        if item["state"] in {"EXITED", "GONE", "QUIET"} and item["keepalive"] == "busy":
            item["keepalive"] = "orphan?"
    return sorted(hosts.values(), key=lambda item: item["host"])


def summarize(nodes):
    counts = {}
    for item in nodes:
        counts[item["state"]] = counts.get(item["state"], 0) + 1
    return {"live": sum(counts.get(state, 0) for state in LIVE_STATES), "counts": counts}


def render_summary(nodes):
    """First line of a status screen: how many nodes are live and what they are doing."""
    summary = summarize(nodes)
    live = [f"{state} {summary['counts'][state]}" for state in LIVE_STATES if summary["counts"].get(state)]
    dead = [f"{state} {summary['counts'][state]}" for state in STATE_ORDER
            if state not in LIVE_STATES and summary["counts"].get(state)]
    if not live and not dead:
        return "NODES  0 live  |  no launcher evidence yet"
    line = f"NODES  {summary['live']} live  |  " + ("  ".join(live) if live else "none")
    return line + (f"  |  not live: " + "  ".join(dead) if dead else "")


def idle(nodes):
    """Live hosts with no running task: holding, waiting, cooling down, or an active launcher between claims."""
    return sorted((item for item in nodes if item["state"] in {"HOLD", "WAIT", "COOL", "LIVE", "ADMIT"} and not item["task"]),
                  key=lambda item: (-(item["last_age"] or 0), item["host"]))


def render_idle(nodes, *, limit=8):
    """One line naming the idle hosts (GPUs allocated, nothing training), oldest first."""
    hosts = idle(nodes)
    if not hosts:
        return "IDLE  none: every live node has a task"
    shown = [f"{item['host']} ({item['state']}" + (f" {int(item['last_age'])//60}m" if item["last_age"] is not None else "") + ")"
             for item in hosts[:limit]]
    more = f" +{len(hosts)-limit} more" if len(hosts) > limit else ""
    return f"IDLE  {len(hosts)} node(s) with GPUs and no task: " + ", ".join(shown) + more


def listed(nodes):
    """Live and stale hosts, plus dead ones whose evidence is recent enough to matter."""
    return [item for item in nodes if item["state"] in (*LIVE_STATES, "STALE", "STOPPING")
            or item["last_age"] is None or item["last_age"] <= LISTING_AGE]


def _role(pid):
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return "other process (not mine or gone)", ""
    for needle, role in ROLES:
        if needle in cmdline:
            if role == "worker" and "--shard" in cmdline:
                role = "eval"
            return role, cmdline[:80]
    return "other", cmdline[:80]


def local_gpus():
    """Per-GPU memory/utilisation and the compute processes on this node, with their roles."""
    if not shutil.which("nvidia-smi"):
        return {"host": node_id(), "available": False, "gpus": [], "processes": []}
    try:
        gpus = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory,gpu_uuid",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return {"host": node_id(), "available": False, "gpus": [], "processes": []}
    rows = []
    for line in gpus.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 4 and all(part.isdigit() for part in parts):
            rows.append({"index": int(parts[0]), "memory_used_mib": int(parts[1]),
                         "memory_total_mib": int(parts[2]), "utilization": int(parts[3])})
    processes = []
    for line in apps.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit():
            role, cmdline = _role(int(parts[0]))
            processes.append({"pid": int(parts[0]), "memory_mib": int(parts[1]) if parts[1].isdigit() else None,
                              "role": role, "cmdline": cmdline})
    return {"host": node_id(), "available": gpus.returncode == 0, "gpus": rows, "processes": processes}


def render_nodes(nodes, table, width):
    if not nodes:
        return ["No launcher evidence under logs/ yet."]
    hidden = len(nodes) - len(listed(nodes))
    nodes = listed(nodes)
    rows = [[item["host"], item["launcher_pid"] or "-",
             "yes" if item["launcher_alive"] else "no" if item["launcher_alive"] is False else "?",
             item["state"], item["task"] or "-", item["phase"] or "-", item["keepalive"],
             f"{int(item['last_age'])}s" if item["last_age"] is not None else "-"] for item in nodes]
    if width < 100:
        # 12+8+5+8+9+7 fixed columns and six separators leave width-61 for TASK.
        lines = table(["NODE", "LAUNCHER", "ALIVE", "STATE", "TASK", "KEEPALIVE", "LOG AGE"],
                      [row[:5] + row[6:] for row in rows], [12, 8, 5, 8, max(8, width-61), 9, 7])
    else:
        # 8+5+8+18+16+9+7 fixed columns and seven separators leave width-85 for NODE.
        lines = table(["NODE", "LAUNCHER", "ALIVE", "STATE", "TASK", "PHASE", "KEEPALIVE", "LOG AGE"], rows,
                      [max(12, min(22, width-85)), 8, 5, 8, 18, 16, 9, 7])
    # Why each holding node's last pass ended, from its [holding] line.
    for item in nodes:
        if item["state"] == "HOLD" and item.get("reason"):
            lines += textwrap.wrap(f"{item['host']} holds: {item['reason']}", width=width,
                                   initial_indent="  ", subsequent_indent="    ", break_long_words=True)
    if hidden:
        lines.append(f"  and {hidden} older host(s) with no live launcher (logs older than {LISTING_AGE/3600:.0f}h) not listed")
    return lines


def render_local_gpus(view, table, width):
    if not view["available"]:
        return [f"{view['host']}: nvidia-smi unavailable here; run status on a node to see its GPUs."]
    lines = [f"{view['host']}: " + "  ".join(f"gpu{g['index']} {g['memory_used_mib']}MiB {g['utilization']}%" for g in view["gpus"])]
    if view["processes"]:
        rows = [[p["pid"], f"{p['memory_mib']}MiB" if p["memory_mib"] is not None else "-", p["role"], p["cmdline"]]
                for p in view["processes"]]
        # 8+9+12 fixed columns and three separators leave width-35 for COMMAND.
        lines += table(["PID", "MEMORY", "ROLE", "COMMAND"], rows, [8, 9, 12, max(16, width-35)])
        if any(p["role"] == "keepalive" for p in view["processes"]):
            lines.append("keepalive = the launcher's tiny kernel that stops the cluster from reclaiming idle GPUs; it ends with 'stop'.")
    else:
        lines.append("no compute process on this node's GPUs")
    return lines
