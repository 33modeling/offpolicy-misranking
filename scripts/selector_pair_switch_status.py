"""Lightweight, read-only observations of Switch workers and saved progress."""
from __future__ import annotations

import json
import math
import time
import uuid
from pathlib import Path


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    temporary.write_text(value)
    temporary.replace(path)


def duration(seconds):
    if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def read_json(path, errors):
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise TypeError("expected JSON object")
        return value
    except FileNotFoundError:
        return None
    except (OSError, ValueError, TypeError) as exc:
        errors.append({"path": str(path), "error": str(exc)})
        return None


def tail(path, size=16384):
    try:
        with path.open("rb") as handle:
            end = handle.seek(0, 2)
            offset = max(0, end-size)
            handle.seek(offset)
            lines = handle.read(size).decode("utf-8", errors="replace").splitlines()
            return lines[1:] if offset else lines
    except OSError:
        return []


def policy_progress(policy, errors):
    logged = None
    for line in reversed(tail(policy / "grpo_stats.jsonl")):
        try:
            row = json.loads(line)
            if isinstance(row.get("step"), int):
                logged = row["step"]
                break
        except (ValueError, AttributeError):
            continue  # A running writer can leave the final JSONL row partial.
    saved = []
    for path in policy.glob("checkpoint-*"):
        suffix = path.name.removeprefix("checkpoint-")
        if not suffix.isdigit():
            continue
        state = read_json(path / "checkpoint_state.json", errors)
        if (state and state.get("completed_steps") == int(suffix)
                and all((path / name).is_file() for name in ("adapter_model.safetensors", "optimizer.pt", "grpo_stats.jsonl"))):
            saved.append(int(suffix))
    final = read_json(policy / "policy_train.json", errors)
    return {"logged_step": logged, "saved_step": max(saved, default=None),
            "published_step": final.get("completed_steps") if final else None,
            "checkpoint_count": len(saved)}


def snapshot(output, seeds, list_tasks):
    now, errors, rows, nodes = time.time(), [], [], {}
    paths = list(output.glob("s*/attempts/*/progress.json"))
    paths.extend(output.glob("node-preflight/*/*/progress.json"))
    paths.extend(output.glob("node-preflight/*/admission.json"))
    for path in paths:
        value = read_json(path, errors)
        if not value:
            continue
        try:
            updated = float(value.get("updated", path.stat().st_mtime))
            host = str(value.get("host", "unknown"))
            age = max(0., now-updated)
            observed = value.get("state", "unknown")
            state = "stale/unknown" if observed == "running" and age > 120 else observed
            logs = sorted(str(p) for p in path.parent.glob("*.log"))
            excerpt = value.get("error", "")
            if observed == "failed":
                excerpt += "\n".join("\n".join(tail(Path(p))[-20:]) for p in logs)
            row = {"host": host, "phase": value.get("phase", "NCCL admission"),
                   "state": state, "recorded_state": observed, "updated": updated,
                   "age_seconds": round(age, 1), "elapsed_seconds": value.get("seconds", value.get("wall_seconds")),
                   "record": str(path), "logs": logs, "error_excerpt": excerpt[-6000:]}
            if host not in nodes or updated > nodes[host]["updated"]:
                nodes[host] = row
        except (OSError, ValueError, TypeError) as exc:
            errors.append({"path": str(path), "error": str(exc)})
    for seed in seeds:
        directory = output / f"s{seed}"
        plan = read_json(directory / "plan.json", errors)
        row = {"seed": seed, "plan": "saved" if plan else "not available",
               "replay": policy_progress(directory / "replay/policy", errors),
               "suffix": policy_progress(directory / "policy", errors)}
        if plan:
            row.update(switch_step=plan.get("switch_step"), end_step=plan.get("end_step"),
                       resume_mode=plan.get("resume_mode", "exact_checkpoint"))
            try:
                reused = read_json(directory / "reused-curves.json", errors) or {}
                jobs = list_tasks(directory, plan, reused)
                pending, done, shards = [], 0, 0
                for arm, step in jobs:
                    present = sum((directory / f"evaluations/{arm}/step-{step}/shard-{i}.done.json").is_file()
                                  for i in range(4))
                    shards += present
                    if present == 4:
                        done += 1
                    else:
                        pending.append({"arm": arm, "step": step, "shards_present": present})
                row.update(evaluation_jobs=len(jobs), evaluations_with_all_receipts=done,
                           shard_receipts_present=shards, missing_evaluations=pending,
                           reused_points={arm: len(points) for arm, points in reused.items()})
            except (OSError, ValueError, KeyError, TypeError) as exc:
                errors.append({"path": str(directory), "error": str(exc)})
        rows.append(row)
    return {"created_at": now, "output": str(output), "seeds": rows,
            "nodes": sorted(nodes.values(), key=lambda item: item["host"]), "errors": errors,
            "scope": "Last saved records, not live remote process checks. File/receipt presence is not hash validation. "
                     "A stale heartbeat does not prove failure. Use results for verified rewards."}


def format_status(data):
    def show(value):
        return "not saved" if value is None else str(value)
    lines = ["SR-GC SWITCH STATUS", f"Root: {data['output']}"]
    for row in data["seeds"]:
        lines.append(f"Seed {row['seed']}: switch={show(row.get('switch_step'))}, final={show(row.get('end_step'))}, "
                     f"plan={row['plan']}")
        for key, label in (("replay", "On-policy replay"), ("suffix", "SR continuation")):
            if key == "replay" and row.get("resume_mode") == "exact_checkpoint":
                lines.append("  On-policy replay: not needed (saved trigger optimizer)")
                continue
            point = row[key]
            lines.append(f"  {label}: logged={show(point['logged_step'])}, saved={show(point['saved_step'])}, "
                         f"published={show(point['published_step'])}, checkpoints={point['checkpoint_count']}")
        if "evaluation_jobs" in row:
            lines.append(f"  Evaluation receipts: {row['evaluations_with_all_receipts']}/{row['evaluation_jobs']} points; "
                         f"{row['shard_receipts_present']}/{4*row['evaluation_jobs']} shards")
    lines.append("NODES (last recorded activity)")
    for node in data["nodes"]:
        lines.append(f"  {node['host']} | {node['phase']} | {node['state']} | "
                     f"elapsed {duration(node['elapsed_seconds'])} | record age {node['age_seconds']:.0f}s")
        if node["state"] in ("failed", "stale/unknown", "interrupted"):
            lines.append(f"    Evidence: {node['record']}")
            if node["error_excerpt"]:
                lines.extend("    " + line for line in node["error_excerpt"].splitlines()[-4:])
    if not data["nodes"]:
        lines.append("  No worker records yet")
    for error in data["errors"]:
        lines.append(f"READ ERROR: {error['path']}: {error['error']}")
    lines.append(data["scope"])
    return "\n".join(lines) + "\n"


def write_status(output, seeds, list_tasks, *, out=None, json_output=False):
    data = snapshot(output, seeds, list_tasks)
    formatted = format_status(data)
    serialized = json.dumps(data, allow_nan=False, indent=2)
    target = Path(out) if out else Path.home() / "selector-pair-switch-status.txt"
    atomic_text(target, formatted + "\nJSON\n" + serialized + "\n")
    print(serialized if json_output else formatted + f"[status] {target}", flush=True)
    return data
