"""Terminate stale processes belonging to one experiment run namespace."""

from __future__ import annotations

import argparse
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Process:
    pid: int
    ppid: int
    command: str
    environ: dict[str, str]
    open_files: frozenset[str]


def _read_process(pid: int) -> Process | None:
    proc = Path("/proc") / str(pid)
    try:
        if proc.stat().st_uid != os.getuid():
            return None
        stat_tail = (proc / "stat").read_text().rsplit(") ", 1)[1].split()
        command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(
            errors="replace"
        )
        raw_environment = (proc / "environ").read_bytes().split(b"\0")
        open_files = frozenset(
            os.readlink(entry)
            for entry in (proc / "fd").iterdir()
            if entry.is_symlink()
        )
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None

    environ: dict[str, str] = {}
    for item in raw_environment:
        if b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        environ[key.decode(errors="replace")] = value.decode(errors="replace")
    return Process(
        pid=pid,
        ppid=int(stat_tail[1]),
        command=command,
        environ=environ,
        open_files=open_files,
    )


def _snapshot() -> dict[int, Process]:
    processes: dict[int, Process] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        process = _read_process(int(entry.name))
        if process is not None:
            processes[process.pid] = process
    return processes


def _protected_ancestors(processes: dict[int, Process]) -> set[int]:
    protected: set[int] = set()
    pid = os.getpid()
    while pid > 1 and pid not in protected:
        protected.add(pid)
        process = processes.get(pid)
        if process is None:
            break
        pid = process.ppid
    return protected


def matching_processes(
    run_prefix: str,
    command_patterns: tuple[str, ...] = (),
    required_environment: tuple[tuple[str, str], ...] = (),
    open_files: tuple[str, ...] = (),
) -> dict[int, Process]:
    processes = _snapshot()
    protected = _protected_ancestors(processes)
    targets: set[int] = set()

    for pid, process in processes.items():
        environment_paths = (
            process.environ.get("OUT_ROOT", ""),
            process.environ.get("RUN_BASE", ""),
            process.environ.get("RUN_BASE_SMOKE", ""),
            process.environ.get("REGIME_ROOT", ""),
        )
        is_v4_worker = process.environ.get("RUN_LABEL", "").startswith("v4-")
        is_v4_launcher = "scripts/go_v4.sh" in process.command
        matches_scope = (
            any(path.startswith(run_prefix) for path in environment_paths)
            or any(pattern in process.command for pattern in command_patterns)
            or any(path in process.open_files for path in open_files)
            or is_v4_worker
            or is_v4_launcher
        )
        matches_environment = all(
            process.environ.get(key) == value
            for key, value in required_environment
        )
        if matches_scope and matches_environment:
            targets.add(pid)

    # Never walk down from this cleanup command or its caller. Otherwise a
    # protected launcher that matches a broad command pattern would cause a
    # sibling such as tee to be selected as its descendant.
    targets.difference_update(protected)

    # Include descendants so launchers cannot leave CUDA children behind.
    changed = True
    while changed:
        changed = False
        for pid, process in processes.items():
            if process.ppid in targets and pid not in targets:
                targets.add(pid)
                changed = True

    return {
        pid: processes[pid]
        for pid in targets - protected
        if pid in processes
    }


def terminate(
    run_prefix: str,
    timeout: float,
    command_patterns: tuple[str, ...] = (),
    required_environment: tuple[tuple[str, str], ...] = (),
    open_files: tuple[str, ...] = (),
) -> list[Process]:
    targets = matching_processes(
        run_prefix, command_patterns, required_environment, open_files
    )
    if not targets:
        return []

    # Stop launchers first so they cannot retry while children are terminating.
    ordered = sorted(targets.values(), key=lambda process: process.pid)
    for process in ordered:
        try:
            os.kill(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        remaining = matching_processes(
            run_prefix, command_patterns, required_environment, open_files
        )
        if not remaining:
            return ordered
        time.sleep(0.5)

    for pid in matching_processes(
        run_prefix, command_patterns, required_environment, open_files
    ):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return ordered


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--command-pattern", action="append", default=[])
    parser.add_argument("--require-environment", action="append", default=[])
    parser.add_argument("--open-file", action="append", default=[])
    args = parser.parse_args()
    required_environment = []
    for item in args.require_environment:
        if "=" not in item:
            parser.error("--require-environment must be KEY=VALUE")
        required_environment.append(tuple(item.split("=", 1)))
    terminated = terminate(
        args.run_prefix,
        args.timeout,
        tuple(args.command_pattern),
        tuple(required_environment),
        tuple(str(Path(path).resolve()) for path in args.open_file),
    )
    if terminated:
        print(f"[startup-cleanup] terminated {len(terminated)} stale processes")
    else:
        print("[startup-cleanup] no stale processes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
