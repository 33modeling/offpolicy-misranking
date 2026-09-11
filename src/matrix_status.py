"""Full status of a flat run_matrix.sh matrix (Qwen3.5-9B and the other
additional matrices), in the shape of the OLMo `status h100` screen.

    python src/matrix_status.py --root "$OM_WORK/runs/<run_id>" \
        --config configs/qwen35_9b_grpo.json \
        --console-logs "$OM_WORK/console-logs" --log-glob 'additional-qwen35-*.log'

The OLMo launcher keeps one family per `family-<dataset>-s<seed>` folder with
owner records and worker heartbeats, which `rlzero_status.py` reads. The
additional matrices are run by `run_additional_experiments.sh` through one
`run_matrix.sh` per node: every point sits directly under
`runs/<run_id>/<model_key>/`, the family claim is a `flock` on
`.queue/<dataset>-s<seed>.lock`, and the only per-node record is the session
log under `console-logs/`. This module reads exactly those artifacts, read
only, and prints:

- one row per family (all 10, not just the six newest points): state, node,
  the four points, the current point, its stage, GRPO steps, last write, note;
- one row per launcher session log: node, pid, started, stage, family it is
  on, failures, exit code, liveness (only the local node's pid is verified);
- all registered points, including completed and not-yet-started points;
- an overall verdict line, and the KEY NUMBERS of every scored point.

Nothing here creates locks, files or processes; `flock` is only tested.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import socket
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import training_progress
from point_key_numbers import HEADER as KEY_NUMBERS_HEADER
from point_key_numbers import point_lines

# Same markers the Qwen status searched for (✘, [abort]) plus the OLMo error set.
ERROR_RE = re.compile(
    r"✘|\[abort\]|CUDA error|CUBLAS_STATUS|cuBLAS|CUDA out of memory|OutOfMemoryError|"
    r"device-side assert|unspecified launch failure|illegal memory access|Traceback|"
    r"RuntimeError|regime-hard-stall|config-abort|done-but-incomplete|repair-failed|"
    r"point-failed",
    re.IGNORECASE,
)
POINT_NAME = re.compile(r"-s(\d+)-([a-z0-9]+)-d(\d+)$")
PROGRESS_RE = re.compile(r"\[progress\]\s+(\S+)\s+(\d+/\d+)\s+(.*?)(?:\s+\+(\d+)min)?\s*$")
ATTEMPT_RE = re.compile(r"^regime-attempt-(\d+)\.log$")
LAUNCH_KV_RE = re.compile(r"(\w+)=(\S+)")
SESSION_FAMILY_RE = re.compile(r"\[progress\] family=(\S+) point=(d\d+)")
DEFAULT_DATASETS = ["math500", "mbpp"]
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
DEFAULT_DRIFTS = [0, 25, 100, 400]
STAGE_TOTAL = 8


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="runs/<run_id> (model-key folders below) or one matrix root")
    parser.add_argument("--config", type=Path, default=None, help="matrix config JSON (datasets/seeds/drifts); defaults to the 40-point design")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--drifts", nargs="*", type=int, default=None)
    parser.add_argument("--console-logs", type=Path, default=None, help="folder of launcher session logs")
    parser.add_argument("--log-glob", default="additional-*.log")
    parser.add_argument("--stall-seconds", type=int, default=2700, help="quiet family becomes a WARNING after this")
    parser.add_argument("--hung-seconds", type=int, default=10800, help="quiet family becomes HUNG after this")
    parser.add_argument("--launcher-live-seconds", type=int, default=1200, help="a remote session log this fresh counts as a live launcher")
    parser.add_argument("--verbose", action="store_true", help="add attempt details and the newest stage-log lines")
    parser.add_argument("--no-key-numbers", action="store_true")
    parser.add_argument("--now", type=float, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    for name in ("stall_seconds", "hung_seconds", "launcher_live_seconds"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


# ---------------------------------------------------------------- small helpers

def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _read_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _nonempty(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def fmt_age(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    seconds = max(0, int(seconds))
    if seconds < 90:
        return f"{seconds}s"
    if seconds < 5400:
        return f"{seconds // 60}m"
    if seconds < 172800:
        return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"
    return f"{seconds // 86400}d"


def elide(text: str, width: int) -> str:
    text = " ".join(text.split())
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    head = (width - 3) * 2 // 3
    return text[:head] + "..." + text[len(text) - (width - 3 - head):]


def elide_head(text: str, width: int) -> str:
    """Keep the beginning (stage labels carry their meaning up front)."""
    text = " ".join(text.split())
    return text if len(text) <= width else text[: max(0, width - 3)] + "..."


def short_host(name: str) -> str:
    """run279668-first-rlvr-1 -> rlvr-1 (the OLMo status shortens the same way)."""
    parts = [p for p in name.split("-") if p]
    return "-".join(parts[-2:]) if len(parts) >= 2 else name


def parse_point_name(name: str) -> tuple[str, int, int] | None:
    match = POINT_NAME.search(name)
    if not match:
        return None
    return match.group(2), int(match.group(1)), int(match.group(3))


def lock_held(path: Path) -> bool:
    """True when another process holds the flock. Never creates the file."""
    if not path.is_file():
        return False
    try:
        with path.open("r") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(stream, fcntl.LOCK_UN)
            return False
    except OSError:
        return True


def error_text_after(lines: list[str], index: int, width: int = 90) -> str:
    """The message that follows an error marker: the last non-empty line within
    eight lines that is not a tagged '[...]' line (the Qwen status rule)."""
    text = ""
    for line in lines[index + 1:index + 9]:
        stripped = line.strip()
        if stripped and not stripped.startswith("["):
            text = stripped
    if not text:
        text = lines[index].strip()
    return elide(text, width)


def scan_errors(path: Path) -> tuple[int, str, int]:
    """(count, text of the last error, line index of the last error or -1)."""
    lines = _read_lines(path)
    count = 0
    last = -1
    for index, line in enumerate(lines):
        if ERROR_RE.search(line):
            count += 1
            last = index
    if last < 0:
        return 0, "", -1
    return count, error_text_after(lines, last), last


# ---------------------------------------------------------------- design

def load_design(args: argparse.Namespace) -> tuple[list[str], list[int], list[int]]:
    datasets, seeds, drifts = None, None, None
    if args.config is not None:
        config = _load_json(args.config) or {}
        experiment = config.get("experiment") if isinstance(config, dict) else None
        if isinstance(experiment, dict):
            datasets = experiment.get("datasets")
            seeds = experiment.get("seeds")
            drifts = experiment.get("drifts")
    if args.datasets:
        datasets = args.datasets
    if args.seeds:
        seeds = args.seeds
    if args.drifts:
        drifts = args.drifts
    datasets = [str(d) for d in (datasets or DEFAULT_DATASETS)]
    seeds = [int(s) for s in (seeds or DEFAULT_SEEDS)]
    drifts = [int(d) for d in (drifts or DEFAULT_DRIFTS)]
    return datasets, seeds, drifts


@dataclass
class Matrix:
    root: Path
    points: dict[tuple[str, int, int], Path] = field(default_factory=dict)

    @property
    def queue(self) -> Path:
        return self.root / ".queue"


def discover_matrices(root: Path) -> list[Matrix]:
    """Point folders directly under `root` or under one level of model-key folders."""
    def points_in(folder: Path) -> dict[tuple[str, int, int], Path]:
        found: dict[tuple[str, int, int], Path] = {}
        try:
            children = sorted(p for p in folder.iterdir() if p.is_dir())
        except OSError:
            return found
        for child in children:
            parsed = parse_point_name(child.name)
            if parsed is not None and (parsed not in found):
                found[parsed] = child
        return found

    if not root.is_dir():
        return [Matrix(root)]
    own = points_in(root)
    if own:
        return [Matrix(root, own)]
    matrices = []
    try:
        subfolders = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    except OSError:
        subfolders = []
    for sub in subfolders:
        found = points_in(sub)
        if found or (sub / ".queue").is_dir():
            matrices.append(Matrix(sub, found))
    return matrices or [Matrix(root)]


# ---------------------------------------------------------------- points

@dataclass
class Point:
    dataset: str
    seed: int
    drift: int
    path: Path | None = None
    done: bool = False
    stage_k: str = ""
    stage_label: str = ""
    stage_minutes: int | None = None
    main_log_mtime: float = 0.0
    last_write: float = 0.0
    grpo_steps: int = 0
    current_error: str = ""
    earlier_error: str = ""
    error_count: int = 0
    telemetry: dict | None = None
    recovery: dict | None = None
    attempt: int = 0

    @property
    def key(self) -> tuple[str, int, int]:
        return self.dataset, self.seed, self.drift

    @property
    def started(self) -> bool:
        return self.path is not None

    def mark(self) -> str:
        """One cell of the d0/d25/d100/d400 grid."""
        if self.done:
            return "ok"
        if not self.started:
            return "-"
        cell = self.stage_k or "start"
        if self.current_error:
            cell = "!" + cell
        return cell


def newest_write(point: Path) -> float:
    latest = 0.0
    fallback = 0.0
    try:
        for path in point.rglob("*"):
            try:
                if not path.is_file():
                    continue
                relative = path.relative_to(point)
                mtime = path.stat().st_mtime
            except OSError:
                continue
            name = path.name
            if name == "keepalive.log" or name.startswith(".pipeline-activity"):
                continue
            fallback = max(fallback, mtime)
            if training_progress._durable(relative):
                latest = max(latest, mtime)
    except OSError:
        pass
    return latest or fallback


def count_lines(path: Path) -> int:
    try:
        with path.open("rb") as stream:
            return sum(1 for _ in stream)
    except OSError:
        return 0


def inspect_point(dataset: str, seed: int, drift: int, path: Path | None, drifts: list[int]) -> Point:
    point = Point(dataset, seed, drift, path)
    if path is None:
        return point
    point.done = _nonempty(path / "DONE")
    logs = path / "logs"
    main_log = logs / "main.log"
    lines = _read_lines(main_log)
    point.main_log_mtime = _mtime(main_log)
    last_progress = -1
    for index, line in enumerate(lines):
        match = PROGRESS_RE.search(line)
        if match:
            last_progress = index
            point.stage_k = match.group(2)
            point.stage_label = match.group(3).strip()
            point.stage_minutes = int(match.group(4)) if match.group(4) else None
    point.last_write = newest_write(path)
    stats = path / f"policy_step_{drift}" / "grpo_stats.jsonl"
    if drift and stats.is_file():
        previous = max([d for d in drifts if d < drift], default=0)
        point.grpo_steps = min(drift, previous + count_lines(stats))
    if point.done:
        return point
    # Errors. main.log first (the rule the short status used): an error after
    # the last [progress] line belongs to the current attempt, one before it is
    # history. Attempt logs fill the gaps: the newest is the current attempt.
    count = 0
    last = -1
    for index, line in enumerate(lines):
        if ERROR_RE.search(line):
            count += 1
            last = index
    if count:
        point.error_count += count
        text = error_text_after(lines, last)
        if last > last_progress:
            point.current_error = text
        else:
            point.earlier_error = text
    attempts: list[tuple[int, Path]] = []
    if logs.is_dir():
        for log in logs.iterdir():
            match = ATTEMPT_RE.match(log.name)
            if match:
                attempts.append((int(match.group(1)), log))
    attempts.sort()
    if attempts:
        point.attempt = attempts[-1][0]
        count, text, _ = scan_errors(attempts[-1][1])
        point.error_count += count
        if count:
            point.current_error = point.current_error or text
        for _, log in attempts[:-1]:
            count, text, _ = scan_errors(log)
            point.error_count += count
            if count:
                point.earlier_error = point.earlier_error or text
    supervisor = logs / "supervisor.log"
    if supervisor.is_file():
        last_failed = last_accepted = -1
        supervisor_lines = _read_lines(supervisor)
        for index, line in enumerate(supervisor_lines):
            if "[point-failed]" in line:
                last_failed = index
            elif "[point-accepted]" in line:
                last_accepted = index
        if last_failed > last_accepted:
            failed = supervisor_lines[last_failed]
            point.earlier_error = point.earlier_error or elide(failed.split("]", 2)[-1].strip(), 90)
    telemetry = _load_json(path / ".pipeline-activity.json")
    if isinstance(telemetry, dict):
        point.telemetry = telemetry
    recovery_log = path / "rollout_recovery.jsonl"
    if recovery_log.is_file():
        records = [_load_json_line(l) for l in _read_lines(recovery_log)]
        records = [r for r in records if isinstance(r, dict)]
        if records:
            point.recovery = records[-1]
    return point


def _load_json_line(line: str):
    try:
        return json.loads(line)
    except ValueError:
        return None


# ---------------------------------------------------------------- launchers

@dataclass
class Launcher:
    log: Path
    host: str = "?"
    pid: int | None = None
    profile: str = ""
    started: str = ""
    stage: str = ""
    family: str = ""
    point: str = ""
    fails: int = 0
    blocked: list[str] = field(default_factory=list)
    exit_rc: int | None = None
    waiting: str = ""
    mtime: float = 0.0
    alive_here: bool | None = None  # None = not this node, cannot verify
    last_error: str = ""

    def state(self, now: float, live_seconds: int) -> str:
        if self.exit_rc is not None:
            return f"EXITED rc={self.exit_rc}"
        if self.alive_here is True:
            return "ALIVE (this node)"
        if self.alive_here is False:
            return "GONE (pid dead, no exit record)"
        age = now - self.mtime
        if age <= live_seconds:
            return "RUNNING? (remote, log fresh)"
        return f"SILENT (remote, log {fmt_age(age)} old)"

    def live(self, now: float, live_seconds: int) -> bool:
        if self.exit_rc is not None:
            return False
        if self.alive_here is not None:
            return self.alive_here
        return now - self.mtime <= live_seconds


def inspect_launcher(log: Path, hostname: str) -> Launcher:
    launcher = Launcher(log=log, mtime=_mtime(log))
    lines = _read_lines(log)
    for line in lines:
        if line.startswith("[launch]"):
            fields = dict(LAUNCH_KV_RE.findall(line))
            launcher.host = fields.get("host", "?")
            launcher.profile = fields.get("profile", "")
            launcher.started = fields.get("utc", "")
            try:
                launcher.pid = int(fields.get("pid", ""))
            except ValueError:
                launcher.pid = None
        elif line.startswith("[stage]"):
            launcher.stage = line[len("[stage]"):].strip()
        elif line.startswith("[family-fail]"):
            launcher.fails += 1
        elif line.startswith("[family-blocked]"):
            match = re.search(r"failure: (\S+?);", line)
            launcher.blocked.append(match.group(1) if match else line)
        elif line.startswith("[exit]"):
            match = re.search(r"rc=(\d+)", line)
            launcher.exit_rc = int(match.group(1)) if match else -1
        elif line.startswith("[queue] waiting"):
            launcher.waiting = line[len("[queue]"):].strip()
        else:
            match = SESSION_FAMILY_RE.search(line)
            if match:
                launcher.family, launcher.point = match.group(1), match.group(2)
                launcher.waiting = ""
            elif ERROR_RE.search(line) and not line.startswith("[progress]"):
                launcher.last_error = elide(line.strip(), 90)
    if launcher.exit_rc is None and launcher.pid is not None and launcher.host == hostname:
        try:
            os.kill(launcher.pid, 0)
            launcher.alive_here = True
        except ProcessLookupError:
            launcher.alive_here = False
        except PermissionError:
            launcher.alive_here = True
    return launcher


def discover_launchers(folder: Path | None, pattern: str, hostname: str, now: float) -> list[Launcher]:
    if folder is None or not folder.is_dir():
        return []
    launchers = []
    for log in folder.glob(pattern):
        if not log.is_file():
            continue
        launcher = inspect_launcher(log, hostname)
        if launcher.exit_rc is not None and now - launcher.mtime > 3 * 86400:
            continue  # exited days ago: history, not status
        launchers.append(launcher)
    launchers.sort(key=lambda l: l.mtime, reverse=True)
    return launchers


# ---------------------------------------------------------------- families

@dataclass
class FamilyRow:
    dataset: str
    seed: int
    matrix: Matrix | None
    points: list[Point]
    state: str = "PENDING"
    verdict: str = ""
    host: str = ""
    current: Point | None = None
    last_write: float = 0.0
    note: str = ""
    blocked: bool = False

    @property
    def key(self) -> str:
        return f"{self.dataset}/s{self.seed}"


def family_row(
    dataset: str,
    seed: int,
    drifts: list[int],
    matrices: list[Matrix],
    launchers: list[Launcher],
    args: argparse.Namespace,
    now: float,
) -> FamilyRow:
    matrix = None
    points: list[Point] = []
    for drift in drifts:
        path = None
        for candidate in matrices:
            found = candidate.points.get((dataset, seed, drift))
            if found is not None:
                path, matrix = found, candidate
                break
        points.append(inspect_point(dataset, seed, drift, path, drifts))
    if matrix is None:
        matrix = matrices[0] if matrices else None
    row = FamilyRow(dataset, seed, matrix, points)
    key = row.key
    row.last_write = max((p.last_write for p in points), default=0.0)
    done = [p for p in points if p.done]
    started = [p for p in points if p.started]
    claimed = False
    if matrix is not None:
        claimed = lock_held(matrix.queue / f"{dataset}-s{seed}.lock") or lock_held(
            matrix.queue / f"{dataset}-s{seed}.control.lock"
        )
    live_launchers = [l for l in launchers if l.live(now, args.launcher_live_seconds)]
    # A remote session log without an [exit] record is unverified, not dead; but a
    # log untouched for days (node lost, launcher killed) must not keep every
    # family UNVERIFIED forever. Same three-day horizon as retained exit records.
    unverified_remote = any(
        l.exit_rc is None and l.alive_here is None and now - l.mtime <= 3 * 86400 for l in launchers
    )
    on_family = [l for l in launchers if l.family == key and l.exit_rc is None]
    for launcher in on_family:
        if launcher.live(now, args.launcher_live_seconds):
            row.host = short_host(launcher.host)
            break
    row.blocked = any(key in l.blocked for l in launchers if l.exit_rc is None)
    # current point: newest non-done point by main.log write, else the next drift
    active = [p for p in points if p.started and not p.done]
    if active:
        row.current = max(active, key=lambda p: (p.main_log_mtime, p.last_write))
    age = (now - row.last_write) if row.last_write else None
    write_age = fmt_age(age)
    if len(done) == len(points):
        row.state, row.verdict, row.note = "COMPLETE", "COMPLETE", "all points DONE"
        return row
    if claimed:
        row.state = "CLAIMED"
        telemetry = None
        if row.current is not None and row.current.telemetry:
            telemetry = row.current.telemetry
            observed = telemetry.get("observed_at_epoch")
            try:
                fresh = now - float(observed) <= 600
            except (TypeError, ValueError):
                fresh = False
            if not fresh:
                telemetry = None
        err = row.current.current_error if row.current is not None else ""
        if age is not None and age >= args.hung_seconds:
            row.verdict = "HUNG"
            row.note = f"NEEDS YOU: claimed but nothing written for {write_age}; inspect that node and stage logs before interrupting it"
        elif telemetry is not None and str(telemetry.get("state", "")).startswith("idle"):
            row.verdict = "IDLE"
            row.note = f"AUTO: watchdog reports idle for {telemetry.get('idle_seconds', '?')}s; it kills and resumes the point by itself"
        elif age is not None and age >= args.stall_seconds:
            row.verdict = "QUIET"
            row.note = f"WARNING: nothing written for {write_age}; a 2048-token rollout stage can be quiet this long, check again in 30 min"
        elif telemetry is not None and str(telemetry.get("state", "")) in {"computing", "output-progress"}:
            row.verdict = "COMPUTING"
            row.note = "ok"
        else:
            row.verdict = "PROGRESSING"
            row.note = "ok"
        if err:
            row.note = f"ERROR (current): {err}" if row.verdict in {"PROGRESSING", "COMPUTING"} else f"{row.note} | error: {err}"
        elif row.current is not None and row.current.recovery is not None:
            recovery = row.current.recovery
            status = recovery.get("status")
            if status not in (None, "recovered", "completed"):
                row.note = f"AUTO: CUDA recovery {status} ({recovery.get('failure_kind') or recovery.get('stage') or 'unknown-cause'})"
            elif row.verdict in {"PROGRESSING", "COMPUTING"}:
                row.note = "ok (recovered from a CUDA error earlier in this point)"
        elif row.current is not None and row.current.earlier_error and row.verdict in {"PROGRESSING", "COMPUTING"}:
            row.note = f"ok (earlier attempt failed: {row.current.earlier_error})"
        return row
    if started:
        row.state = "PARTIAL"
        current_error = earlier_error = ""
        if row.current is not None:
            current_error = row.current.current_error
            earlier_error = row.current.earlier_error
        if row.blocked:
            row.verdict = "BLOCKED"
            row.note = "NEEDS YOU: permanent contract failure reported by a launcher; fix the reported contract, then rerun"
        elif live_launchers:
            row.verdict = "QUEUED"
            row.note = f"QUEUED: no launcher on it for {write_age}; the next free launcher resumes it from the artifacts"
        elif unverified_remote:
            row.verdict = "UNVERIFIED"
            row.note = "remote session has no exit record; verify its node and the shared work path before restarting"
        else:
            row.verdict = "STOPPED"
            row.note = f"NEEDS YOU: no launcher anywhere for {write_age} -> start one: bash scripts/run_qwen35_9b.sh"
        # An error after the last progress line is the current attempt's, whoever
        # holds the family now; name it first, as the short status did.
        if current_error:
            row.note = f"ERROR (current): {current_error} | {row.note}"
        elif earlier_error:
            row.note += f" | last error: {earlier_error}"
        return row
    row.state = "PENDING"
    row.verdict = "QUEUED" if live_launchers else ("UNVERIFIED" if unverified_remote else "NOT_STARTED")
    row.note = "waits for a free launcher" if live_launchers else ("remote launcher unverified" if unverified_remote else "no launcher running")
    return row


# ---------------------------------------------------------------- rendering

def render(args: argparse.Namespace) -> tuple[list[str], str]:
    now = args.now if args.now is not None else time.time()
    hostname = socket.gethostname()
    datasets, seeds, drifts = load_design(args)
    matrices = discover_matrices(args.root)
    launchers = discover_launchers(args.console_logs, args.log_glob, hostname, now)
    rows = [
        family_row(dataset, seed, drifts, matrices, launchers, args, now)
        for seed in seeds
        for dataset in datasets
    ]
    total_points = len(rows) * len(drifts)
    points_done = sum(1 for row in rows for p in row.points if p.done)
    points_started = sum(1 for row in rows for p in row.points if p.started)
    live_launchers = [l for l in launchers if l.live(now, args.launcher_live_seconds)]
    out: list[str] = []

    for matrix in matrices:
        generation = matrix.queue / "generation.git"
        generation_git = generation.read_text(encoding="utf-8").strip()[:12] if generation.is_file() else "not-started"
        out.append(f"matrix   {matrix.root}   generation {generation_git or 'empty'}")
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.verdict] = counts.get(row.verdict, 0) + 1
    summary = "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    out.append(
        f"families {len(rows)}: {summary}   points DONE {points_done}/{total_points}, started {points_started}   "
        f"launchers live {len(live_launchers)}/{len(launchers)} (remote liveness is inferred from log age)"
    )
    out.append("")
    grid = " ".join(f"{'d' + str(d):<6}" for d in drifts)
    out.append(f" {'family':<11} {'state':<11} {'node':<10} {grid} {'now':<5} {'stage':<34} {'grpo':<8} {'write':<7} note")
    for row in rows:
        cells = " ".join(f"{p.mark():<6}" for p in row.points)
        current = row.current
        if current is not None:
            now_col = f"d{current.drift}"
            stage = f"{current.stage_k} {current.stage_label}".strip() if current.stage_k else "starting"
            if current.stage_minutes is not None:
                stage += f" +{current.stage_minutes}m"
            grpo = f"{current.grpo_steps}/{current.drift}" if current.drift else "-"
        else:
            missing = [p for p in row.points if not p.done]
            now_col = f"d{missing[0].drift}" if missing and row.state != "COMPLETE" else "-"
            stage = "not started" if row.state == "PENDING" else ("next" if missing else "-")
            grpo = "-"
        write = fmt_age(now - row.last_write) if row.last_write else "-"
        out.append(
            f" {row.key:<11} {row.verdict:<11} {elide_head(row.host or '-', 10):<10} {cells} {now_col:<5} "
            f"{elide_head(stage, 34):<34} {grpo:<8} {write:<7} {row.note}"
        )
    out.append(" cells: ok = DONE   k/8 = stage of the point pipeline (1 prep 2 behavior-rollout 3 grpo 4 fresh-rollout 5 gradients 6 scores 7 merge+report 8 DONE)   - = not started   ! = error in the current attempt")
    out.append(" state: PROGRESSING/COMPUTING = claimed with recent output; QUIET/HUNG = prolonged silence, verify the node before interrupting; QUEUED = launcher activity observed; UNVERIFIED = remote process not checked")
    out.append("")
    out.append(f"ALL POINTS ({total_points})")
    out.append(f" {'point':<18} {'state':<13} {'stage':<30} {'grpo':<8} write")
    for row in rows:
        for point in row.points:
            if point.done:
                state, stage = "DONE", "8/8 DONE"
            elif not point.started:
                state, stage = "NOT_STARTED", "-"
            else:
                state = row.verdict if point is row.current else "PENDING"
                stage = f"{point.stage_k} {point.stage_label}".strip() or "starting"
            grpo = f"{point.grpo_steps}/{point.drift}" if point.started and point.drift else "-"
            write = fmt_age(now - point.last_write) if point.last_write else "-"
            out.append(f" {row.key + '/d' + str(point.drift):<18} {state:<13} {elide_head(stage, 30):<30} {grpo:<8} {write}")
    out.append("")
    out.append("LAUNCHERS (all open sessions and exits from the last three days)")
    if launchers:
        out.append(f" {'launcher log':<34} {'node':<10} {'pid':<8} {'started':<21} {'state':<30} {'on':<14} {'fails':<5} stage")
        for launcher in launchers:
            on = f"{launcher.family} {launcher.point}".strip() or (launcher.waiting[:14] if launcher.waiting else "-")
            out.append(
                f" {elide(launcher.log.name, 34):<34} {short_host(launcher.host):<9} {launcher.pid or '-'!s:<7} "
                f"{launcher.started[:20]:<21} {launcher.state(now, args.launcher_live_seconds):<30} {elide(on, 14):<14} "
                f"{launcher.fails:<5} {elide(launcher.stage, 40)}"
            )
            if launcher.waiting:
                out.append(f"   {launcher.waiting}")
            if launcher.blocked:
                out.append(f"   blocked families: {', '.join(launcher.blocked)}")
            if launcher.exit_rc not in (None, 0) and launcher.last_error:
                out.append(f"   last error: {launcher.last_error}")
    else:
        out.append(" launchers: no session log found" + (f" under {args.console_logs}" if args.console_logs else ""))
    out.append("")

    if args.verbose:
        out.append(" per point (started, not DONE):")
        for row in rows:
            for point in row.points:
                if not point.started or point.done or point.path is None:
                    continue
                stage = f"{point.stage_k} {point.stage_label}".strip() or "starting"
                write = fmt_age(now - point.last_write) if point.last_write else "-"
                errors = ""
                if point.current_error:
                    errors = f" ERROR (current): {point.current_error}"
                elif point.earlier_error:
                    errors = f" earlier: {point.earlier_error}"
                out.append(f"  {row.key}/d{point.drift}  attempt {point.attempt or '-'}  {elide_head(stage, 60)}  write {write}  errors {point.error_count}{errors}")
                if point.telemetry:
                    t = point.telemetry
                    out.append(f"    telemetry state={t.get('state')} cpu_delta={t.get('cpu_delta_seconds')} gpu_peak={t.get('gpu_peak_percent')} idle={t.get('idle_seconds')}s")
                logs = point.path / "logs"
                if logs.is_dir():
                    stage_logs = [p for p in logs.glob("*.log") if p.name not in {"keepalive.log", "main.log", "supervisor.log"} and not p.name.startswith("regime-attempt")]
                    if stage_logs:
                        newest = max(stage_logs, key=_mtime)
                        tail = [l for l in _read_lines(newest) if l.strip()][-2:]
                        for line in tail:
                            out.append(f"    {newest.name}: {elide(line, 150)}")
        out.append("")

    if not args.no_key_numbers:
        out.append(KEY_NUMBERS_HEADER)
        printed = 0
        for row in rows:
            for point in row.points:
                if point.path is None:
                    continue
                for line in point_lines(point.path, point.dataset, point.seed, point.drift):
                    out.append(line)
                    printed += 1
        if not printed:
            out.append(" (no scored point yet)")
        out.append("")

    # ---- overall verdict, in the OLMo status vocabulary ----
    verdicts = {row.verdict for row in rows}
    complete = all(row.verdict == "COMPLETE" for row in rows)
    current_errors = [row for row in rows if row.note.startswith("ERROR (current)")]
    if complete:
        overall, action = "COMPLETE", "none"
    elif "BLOCKED" in verdicts:
        overall, action = "BLOCKED", "fix_the_reported_contract_then_rerun"
    elif "HUNG" in verdicts and not ({"PROGRESSING", "COMPUTING"} & verdicts):
        overall, action = "HUNG", "verify_HUNG_node_and_stage_logs_before_interrupting"
    elif {"PROGRESSING", "COMPUTING", "QUIET", "IDLE"} & verdicts:
        degraded = bool({"HUNG", "STOPPED", "QUIET", "IDLE"} & verdicts) or bool(current_errors)
        overall = "DEGRADED" if degraded else "RUNNING"
        if "HUNG" in verdicts:
            action = "verify_HUNG_node_and_stage_logs_before_interrupting"
        elif current_errors:
            action = "read_the_ERROR_rows__if_the_same_error_repeats_next_status_fix_it"
        elif "STOPPED" in verdicts:
            action = "start_a_launcher_on_a_free_node"
        elif "QUIET" in verdicts or "IDLE" in verdicts:
            action = "check_again_in_30_min"
        else:
            action = "none"
    elif live_launchers:
        silent = all(now - launcher.mtime >= args.stall_seconds for launcher in live_launchers)
        if current_errors:
            overall, action = "DEGRADED", "read_the_ERROR_rows__if_the_same_error_repeats_next_status_fix_it"
        elif silent:
            overall, action = "DEGRADED", "inspect_silent_launcher_preflight_or_queue__pid_liveness_is_not_progress"
        else:
            overall, action = "STARTING", "wait_for_launcher_preflight_or_queue_claim"
    elif "UNVERIFIED" in verdicts:
        overall, action = "UNVERIFIED", "verify_remote_launcher_and_shared_work_path"
    elif "STOPPED" in verdicts:
        overall, action = "STOPPED", "start_launchers_after_node_cleanup"
    elif points_done:
        overall, action = "INCOMPLETE", "start_launchers"
    else:
        overall, action = "NOT_STARTED", "start_launchers"
    decisions = {
        "COMPLETE": f"DONE: all {total_points} registered points have nonempty DONE records.",
        "RUNNING": "NO ERROR: the matrix is progressing. See every point and launcher below.",
        "STARTING": "NO ERROR: launcher activity observed; preparing or waiting for a family.",
        "DEGRADED": "WARNING: errors or quiet work need inspection; do not stop healthy workers.",
        "HUNG": "WARNING: prolonged output silence; verify the affected node and stage logs before interrupting.",
        "BLOCKED": "ERROR: a reported contract failure needs inspection; do not restart healthy workers.",
        "UNVERIFIED": "WARNING: remote session liveness is unverified, not confirmed dead. Check its node and work path.",
        "STOPPED": "WARNING: incomplete matrix with no observed launcher activity. Verify nodes before relaunching.",
        "INCOMPLETE": "WARNING: some registered points are unfinished; see the full matrix below.",
        "NOT_STARTED": "NOT STARTED: no registered point or live session found. Check the printed work path.",
    }
    out[0:0] = [
        f"DECISION {decisions[overall]}",
        (f"points   {points_done} done / {points_started} started / {total_points} in matrix   "
         f"family failures in shown sessions: {sum(l.fails for l in launchers)}"),
    ]
    out.append(f"overall_verdict={overall}")
    out.append(f"recommended_action={action}")
    return out, overall


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    lines, _ = render(args)
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
