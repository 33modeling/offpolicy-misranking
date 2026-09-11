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
    start_time: int = 0


def _read_process(pid: int) -> Process | None:
    proc = Path("/proc") / str(pid)
    try:
        if proc.stat().st_uid != os.getuid():
            return None
        stat_tail = (proc / "stat").read_text().rsplit(") ", 1)[1].split()
        if stat_tail[0] == "Z":
            return None
        command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(
            errors="replace"
        )
        raw_environment = (proc / "environ").read_bytes().split(b"\0")
        descriptors = []
        for entry in (proc / "fd").iterdir():
            try:
                descriptors.append(os.readlink(entry))
            except OSError:
                # A closing descriptor must not hide the live lock owner.
                continue
        open_files = frozenset(descriptors)
    except OSError:
        # An unreadable process must not abort recovery for the whole node.
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
        start_time=int(stat_tail[19]),
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
    launcher_environment_from_child: bool = False,
    session_log_prefix: str = "",
) -> dict[int, Process]:
    processes = _snapshot()
    protected = _protected_ancestors(processes)
    targets: set[int] = set()
    environment_witnesses: set[int] = set()
    if launcher_environment_from_child and required_environment:
        # Linux exposes the shell's initial environment, not exports added by
        # setup_env.sh. Its exec'ed children do expose those exports. Only an
        # explicitly named launcher may use a descendant as a work-root witness.
        for child in processes.values():
            if all(child.environ.get(key) == value for key, value in required_environment):
                pid = child.ppid
                visited: set[int] = set()
                while pid in processes and pid not in visited:
                    visited.add(pid)
                    environment_witnesses.add(pid)
                    pid = processes[pid].ppid

    for pid, process in processes.items():
        environment_paths = (
            process.environ.get("OUT_ROOT", ""),
            process.environ.get("RUN_BASE", ""),
            process.environ.get("RUN_BASE_SMOKE", ""),
            process.environ.get("REGIME_ROOT", ""),
        )
        # The v4 heuristics matched any RUN_LABEL=v4-* process on the node, even
        # for an unrelated --run-prefix. Apply them only inside a v4 scope.
        v4_scope = "v4" in run_prefix
        is_v4_worker = v4_scope and process.environ.get("RUN_LABEL", "").startswith("v4-")
        is_v4_launcher = v4_scope and "scripts/go_v4.sh" in process.command
        matches_scope = (
            any(path == run_prefix or path.startswith(run_prefix.rstrip("/") + "/") or path.startswith(run_prefix + "-") for path in environment_paths)
            or any(pattern in process.command for pattern in command_patterns)
            or any(path in process.open_files for path in open_files)
            or bool(session_log_prefix and (
                process.environ.get("SESSION_LOG", "").startswith(session_log_prefix)
                or any(path.startswith(session_log_prefix) for path in process.open_files)
            ))
            or is_v4_worker
            or is_v4_launcher
        )
        matches_environment = all(
            process.environ.get(key) == value
            for key, value in required_environment
        )
        if (not matches_environment and pid in environment_witnesses
                and any(pattern in process.command for pattern in command_patterns)
                and all(key not in process.environ or process.environ[key] == value
                        for key, value in required_environment)):
            matches_environment = True
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


def list_processes(
    run_prefix: str,
    command_patterns: tuple[str, ...] = (),
    required_environment: tuple[tuple[str, str], ...] = (),
    open_files: tuple[str, ...] = (),
    launcher_environment_from_child: bool = False,
    session_log_prefix: str = "",
) -> list[Process]:
    """The processes terminate() would stop, in pid order; nothing is signalled."""
    targets = matching_processes(
        run_prefix, command_patterns, required_environment, open_files,
        launcher_environment_from_child, session_log_prefix,
    )
    return sorted(targets.values(), key=lambda process: process.pid)


def terminate(
    run_prefix: str,
    timeout: float,
    command_patterns: tuple[str, ...] = (),
    required_environment: tuple[tuple[str, str], ...] = (),
    open_files: tuple[str, ...] = (),
    launcher_environment_from_child: bool = False,
    session_log_prefix: str = "",
) -> list[Process]:
    targets = matching_processes(
        run_prefix, command_patterns, required_environment, open_files,
        launcher_environment_from_child, session_log_prefix,
    )
    if not targets:
        return []

    # Stop launchers first so they cannot retry while children are terminating.
    def alive(process: Process) -> bool:
        try:
            fields = (Path("/proc") / str(process.pid) / "stat").read_text().rsplit(") ", 1)[1].split()
            return fields[0] != "Z" and int(fields[19]) == process.start_time
        except (OSError, ValueError, IndexError):
            return False

    def send(process: Process, sig: int) -> None:
        if not alive(process):
            return
        try:
            os.kill(process.pid, sig)
        except ProcessLookupError:
            pass

    for process in sorted(targets.values(), key=lambda process: process.pid):
        print(f"[cleanup-target] pid={process.pid} {process.command.strip()}", flush=True)
        send(process, signal.SIGTERM)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for pid, process in matching_processes(
            run_prefix, command_patterns, required_environment, open_files,
            launcher_environment_from_child, session_log_prefix,
        ).items():
            if pid not in targets or targets[pid].start_time != process.start_time:
                targets[pid] = process
                send(process, signal.SIGTERM)
        if not any(alive(process) for process in targets.values()):
            return sorted(targets.values(), key=lambda process: process.pid)
        time.sleep(0.1)

    # Keep tracking already selected children after their parent exits. An
    # orphaned tee/sleep may have no scope marker but still hold inherited locks.
    for process in targets.values():
        send(process, signal.SIGKILL)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if not any(alive(process) for process in targets.values()):
            return sorted(targets.values(), key=lambda process: process.pid)
        time.sleep(0.1)
    remaining = [p.pid for p in targets.values() if alive(p)]
    raise RuntimeError(f"selected processes did not exit after KILL: {remaining}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--command-pattern", action="append", default=[])
    parser.add_argument("--require-environment", action="append", default=[])
    parser.add_argument("--open-file", action="append", default=[])
    parser.add_argument("--launcher-environment-from-child", action="store_true",
                        help="allow named launchers to prove missing initial environment through children")
    parser.add_argument("--session-log-prefix", default="",
                        help="also select this exact work/profile session-log namespace")
    parser.add_argument(
        "--list", action="store_true",
        help="print the matching processes (pid<TAB>command) and stop nothing",
    )
    args = parser.parse_args()
    required_environment = []
    for item in args.require_environment:
        if "=" not in item:
            parser.error("--require-environment must be KEY=VALUE")
        required_environment.append(tuple(item.split("=", 1)))
    if args.list:
        for process in list_processes(
            args.run_prefix,
            tuple(args.command_pattern),
            tuple(required_environment),
            tuple(str(Path(path).resolve()) for path in args.open_file),
            args.launcher_environment_from_child,
            args.session_log_prefix,
        ):
            print(f"{process.pid}\t{process.command.strip()}")
        return 0
    try:
        terminated = terminate(
            args.run_prefix,
            args.timeout,
            tuple(args.command_pattern),
            tuple(required_environment),
            tuple(str(Path(path).resolve()) for path in args.open_file),
            args.launcher_environment_from_child,
            args.session_log_prefix,
        )
    except RuntimeError as exc:
        print(f"[abort] {exc}", flush=True)
        return 1
    if terminated:
        print(f"[startup-cleanup] terminated {len(terminated)} stale processes")
    else:
        print("[startup-cleanup] no stale processes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
