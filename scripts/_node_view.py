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
import time

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


def launcher_nodes(root, tasks, *, now=None):
    """One row per host that has launcher evidence under ROOT/logs."""
    now = time.time() if now is None else now
    logs = Path(root) / "logs"
    here = socket.gethostname().rstrip("_")
    hosts = {}

    def row(host):
        return hosts.setdefault(host, {"host": host, "launcher_pid": None, "launcher_alive": None,
                                       "state": "-", "detail": "", "last_age": None, "keepalive": "-",
                                       "task": "", "phase": ""})
    for path in logs.glob("launcher.*.pid"):
        host = _host_of(path, "launcher.")
        try:
            pid = int(path.read_text().strip())
        except (OSError, ValueError):
            continue
        item = row(host)
        item["launcher_pid"] = pid
        if host == here:
            try:
                os.kill(pid, 0)
                item["launcher_alive"] = True
            except ProcessLookupError:
                item["launcher_alive"] = False
            except PermissionError:
                item["launcher_alive"] = True
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
        if last.startswith("[launcher-exit]"):
            item["state"] = "BLOCKED" if "rc=78" in last else "EXITED"
        elif last.startswith(("[hold]", "[holding]")):
            item["state"] = "HOLD"
        elif last.startswith("[waiting]"):
            item["state"] = "WAIT"
        elif last.startswith("[blocked]"):
            item["state"] = "BLOCKED"
        elif last.startswith("[nccl-preflight]"):
            item["state"] = "ADMIT"
        elif last.startswith(("[retry]", "[claimed]", "[gate]", "[grpo]", "[fresh_r]")):
            item["state"] = "RUN"
        elif last.startswith("[stopping]"):
            item["state"] = "STOPPING"
        elif item["state"] == "-":
            item["state"] = "LIVE" if age < 120 else "QUIET"
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
        if item["launcher_alive"] is False and item["state"] in {"HOLD", "WAIT", "LIVE", "QUIET"}:
            item["state"] = "EXITED"
        if item["state"] == "EXITED" and item["keepalive"] == "busy":
            item["keepalive"] = "orphan?"
    return sorted(hosts.values(), key=lambda item: item["host"])


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
        return {"host": socket.gethostname(), "available": False, "gpus": [], "processes": []}
    try:
        gpus = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory,gpu_uuid",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return {"host": socket.gethostname(), "available": False, "gpus": [], "processes": []}
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
    return {"host": socket.gethostname(), "available": gpus.returncode == 0, "gpus": rows, "processes": processes}


def render_nodes(nodes, table, width):
    if not nodes:
        return ["No launcher evidence under logs/ yet."]
    rows = [[item["host"], item["launcher_pid"] or "-",
             "yes" if item["launcher_alive"] else "no" if item["launcher_alive"] is False else "?",
             item["state"], item["task"] or "-", item["phase"] or "-", item["keepalive"],
             f"{int(item['last_age'])}s" if item["last_age"] is not None else "-"] for item in nodes]
    if width < 100:
        # 12+8+5+8+9+7 fixed columns and six separators leave width-61 for TASK.
        return table(["NODE", "LAUNCHER", "ALIVE", "STATE", "TASK", "KEEPALIVE", "LOG AGE"],
                     [row[:5] + row[6:] for row in rows], [12, 8, 5, 8, max(8, width-61), 9, 7])
    # 8+5+8+18+16+9+7 fixed columns and seven separators leave width-85 for NODE.
    return table(["NODE", "LAUNCHER", "ALIVE", "STATE", "TASK", "PHASE", "KEEPALIVE", "LOG AGE"], rows,
                 [max(12, min(22, width-85)), 8, 5, 8, 18, 16, 9, 7])


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
