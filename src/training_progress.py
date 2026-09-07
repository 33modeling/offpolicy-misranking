#!/usr/bin/env python3
"""Is training actually advancing? One line, from durable artifacts only.

2026-09-07: a Qwen3.5-9B launcher looked alive for 18 hours while no GRPO
step had ever run, and an OLMo family sat unowned for nine hours while its
worker was "running". Liveness (heartbeats, CPU, GPU duty, log lines) is not
progress. Progress is what lands on disk:

- points with a non-empty DONE,
- GRPO optimizer steps (lines of policy_step_*/grpo_stats.jsonl),
- rollout bytes (rollouts_*.jsonl and their .partial files grow per prompt),
- the newest durable artifact write (logs, keepalive and telemetry excluded).

Each probe appends its signature to <root>/.progress/history.jsonl, so the
next probe (from any node, launcher heartbeat or status) can say what changed
over the last window. The verdict is one word the operator can act on:

    TRAINING      something durable changed within --stall-minutes
    NOT TRAINING  nothing durable changed for longer than that
    NOT STARTED   no point directory exists yet
    DONE          every point has DONE

    python src/training_progress.py --root RUNS --total-points 40 --record
    python src/training_progress.py --root RUNS --watch --interval 600   # prints one line per interval
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

HISTORY_KEEP = 400
EXCLUDED_NAMES = {"keepalive.log", "keepalive.pid", "ALERTS.log", "status-history.log"}
EXCLUDED_PREFIXES = (".pipeline-activity", ".progress", ".workers", ".families", ".queue")
EXCLUDED_SUFFIXES = (".log", ".lock", ".tmp")


@dataclass
class Signature:
    points_total: int
    points_done: int
    grpo_steps: int
    rollout_bytes: int
    rollout_files: int
    last_write_epoch: float
    last_write_name: str

    def key(self) -> tuple:
        return (self.points_done, self.grpo_steps, self.rollout_bytes, self.rollout_files)

    def to_json(self) -> dict:
        return {
            "points_total": self.points_total,
            "points_done": self.points_done,
            "grpo_steps": self.grpo_steps,
            "rollout_bytes": self.rollout_bytes,
            "rollout_files": self.rollout_files,
            "last_write_epoch": self.last_write_epoch,
            "last_write_name": self.last_write_name,
        }


def _count_lines(path: Path) -> int:
    try:
        with path.open("rb") as stream:
            return sum(1 for _ in stream)
    except OSError:
        return 0


STAGE_LOG_PREFIXES = ("fresh-shard", "beta-shard", "ograds-shard", "score-shard", "val-grads", "grpo", "recovery-rollout")


def _durable(path: Path) -> bool:
    name = path.name
    if name in EXCLUDED_NAMES or name.startswith(EXCLUDED_PREFIXES):
        return False
    if "logs" in path.parts:
        # Stage logs are written per item by the compute process itself
        # ("oracle 65/134", "score 40/128", GRPO step lines): a gradient or
        # scoring stage publishes nothing else for 15-30 minutes. Supervisor,
        # watchdog and keepalive logs stay excluded.
        return name.endswith(".log") and name.startswith(STAGE_LOG_PREFIXES)
    return not name.endswith(EXCLUDED_SUFFIXES)


def point_dirs(root: Path) -> list[Path]:
    """Point directories: <root>/*/run_config.json or <root>/*/*/run_config.json
    (OLMo: root/family-x/point; additional matrices: runs/<id>/<model>/point)."""
    found: list[Path] = []
    for pattern in ("*/run_config.json", "*/*/run_config.json"):
        for config in root.glob(pattern):
            if config.parent not in found:
                found.append(config.parent)
    return sorted(found)


def probe(root: Path, total_points: int | None = None) -> Signature:
    points = point_dirs(root)
    done = 0
    steps = 0
    rollout_bytes = 0
    rollout_files = 0
    last_epoch = 0.0
    last_name = "-"
    for point in points:
        stamp = point / "DONE"
        try:
            if stamp.is_file() and stamp.stat().st_size > 0:
                done += 1
        except OSError:
            pass
        for stats in point.glob("policy_step_*/grpo_stats.jsonl"):
            steps += _count_lines(stats)
        for artifact in point.glob("rollouts_*"):
            try:
                if not artifact.is_file():
                    continue
                if artifact.suffix in (".jsonl", ".partial"):
                    rollout_bytes += artifact.stat().st_size
                    rollout_files += 1
            except OSError:
                continue
        try:
            for path in point.rglob("*"):
                if not path.is_file() or not _durable(path.relative_to(point)):
                    continue
                mtime = path.stat().st_mtime
                if mtime > last_epoch:
                    last_epoch, last_name = mtime, path.name
        except OSError:
            continue
    return Signature(
        points_total=total_points if total_points is not None else len(points),
        points_done=done,
        grpo_steps=steps,
        rollout_bytes=rollout_bytes,
        rollout_files=rollout_files,
        last_write_epoch=last_epoch,
        last_write_name=last_name,
    )


def history_path(root: Path) -> Path:
    return root / ".progress" / "history.jsonl"


def read_history(root: Path) -> list[dict]:
    try:
        lines = history_path(root).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    records = []
    for line in lines:
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict) and "epoch" in record:
            records.append(record)
    return records


def record(root: Path, signature: Signature, now: float) -> None:
    path = history_path(root)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"epoch": now, **signature.to_json()}
        records = read_history(root)
        records.append(entry)
        records = records[-HISTORY_KEEP:]
        temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        temporary.write_text(
            "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in records),
            encoding="utf-8",
        )
        temporary.replace(path)
    except OSError:
        pass  # a shared-volume hiccup must not break the reporter


def fmt_age(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    if seconds < 90:
        return f"{seconds}s"
    if seconds < 5400:
        return f"{seconds // 60}m"
    if seconds < 172800:
        return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"
    return f"{seconds // 86400}d"


def last_change_epoch(history: list[dict], current: Signature, now: float) -> float | None:
    """When the signature last changed: the newest history record whose key
    differs from the current one gives the change bound; the newest durable
    write is the other bound. Returns None with no evidence at all."""
    candidates = [current.last_write_epoch] if current.last_write_epoch > 0 else []
    key = current.key()
    for entry in reversed(history):
        entry_key = (
            entry.get("points_done"), entry.get("grpo_steps"),
            entry.get("rollout_bytes"), entry.get("rollout_files"),
        )
        if entry_key == key:
            continue
        candidates.append(float(entry["epoch"]))
        break
    return max(candidates) if candidates else None


def deltas(history: list[dict], current: Signature, now: float, window_seconds: float) -> dict | None:
    """Change since the newest record at least window_seconds old (None without one)."""
    baseline = None
    for entry in history:
        if now - float(entry["epoch"]) >= window_seconds:
            baseline = entry
        else:
            break
    if baseline is None:
        return None
    return {
        "window_seconds": now - float(baseline["epoch"]),
        "points": current.points_done - int(baseline.get("points_done", 0)),
        "steps": current.grpo_steps - int(baseline.get("grpo_steps", 0)),
        "rollout_mb": (current.rollout_bytes - int(baseline.get("rollout_bytes", 0))) / 1e6,
    }


def verdict(
    root: Path,
    *,
    total_points: int | None,
    stall_seconds: float,
    now: float | None = None,
    record_probe: bool = False,
    window_seconds: float = 1800.0,
) -> tuple[str, str, Signature]:
    """Returns (word, one-line summary, signature). word in
    {"TRAINING", "NOT TRAINING", "NOT STARTED", "DONE"}."""
    now = time.time() if now is None else now
    current = probe(root, total_points)
    history = read_history(root)
    if record_probe:
        record(root, current, now)
    total = current.points_total
    counts = f"points {current.points_done}/{total}  grpo {current.grpo_steps} steps  rollouts {current.rollout_bytes / 1e6:.0f} MB"
    if not point_dirs(root):
        return "NOT STARTED", f"NOT STARTED  no point directory under {root}", current
    if total and current.points_done >= total:
        return "DONE", f"DONE  {counts}", current
    changed_at = last_change_epoch(history, current, now)
    since = None if changed_at is None else now - changed_at
    change = deltas(history, current, now, window_seconds)
    delta_text = ""
    if change is not None:
        delta_text = (
            f"  last {fmt_age(change['window_seconds'])}: +{change['points']} points"
            f" +{change['steps']} grpo steps +{change['rollout_mb']:.0f} MB rollouts"
        )
    write_text = (
        f"  last write {fmt_age(now - current.last_write_epoch)} ago ({current.last_write_name})"
        if current.last_write_epoch > 0 else "  no durable write yet"
    )
    if since is not None and since <= stall_seconds:
        return "TRAINING", f"TRAINING  {counts}{delta_text}{write_text}", current
    quiet = fmt_age(since) if since is not None else "?"
    return (
        "NOT TRAINING",
        f"NOT TRAINING for {quiet}  {counts}{delta_text}{write_text}",
        current,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="one-line training progress verdict from durable artifacts")
    parser.add_argument("--root", type=Path, required=True, help="matrix root holding the point directories")
    parser.add_argument("--total-points", type=int, default=None)
    parser.add_argument("--stall-minutes", type=float, default=float(os.environ.get("OM_PROGRESS_STALL_MINUTES", "30")))
    parser.add_argument("--window-minutes", type=float, default=30.0, help="delta window for the +N summary")
    parser.add_argument("--record", action="store_true", help="append this probe to <root>/.progress/history.jsonl")
    parser.add_argument("--watch", action="store_true", help="repeat every --interval seconds (always records)")
    parser.add_argument("--interval", type=float, default=600.0)
    parser.add_argument("--tag", default="[progress]", help="prefix for --watch lines")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.stall_minutes <= 0 or args.interval <= 0:
        parser.error("--stall-minutes and --interval must be positive")

    def once() -> int:
        word, line, signature = verdict(
            args.root,
            total_points=args.total_points,
            stall_seconds=args.stall_minutes * 60,
            record_probe=args.record or args.watch,
            window_seconds=args.window_minutes * 60,
        )
        if args.json:
            print(json.dumps({"verdict": word, "line": line, **signature.to_json()}, sort_keys=True))
        elif args.watch:
            stamp = time.strftime("%H:%M:%SZ", time.gmtime())
            print(f"{args.tag} {stamp}  {line}", flush=True)
        else:
            print(line, flush=True)
        return 0 if word in ("TRAINING", "DONE") else 1

    if not args.watch:
        return once()
    while True:
        once()
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
