"""Observe a shared RL-Zero run and classify its live health."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

ERROR_RE = re.compile(
    r"CUDA error|CUBLAS_STATUS|cuBLAS|CUDA out of memory|device-side assert|"
    r"unspecified launch failure|illegal memory access|Traceback|RuntimeError|"
    r"regime-hard-stall|logs?.*GPU.*CPU.*(?:stopped|정지)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Family:
    dataset: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.dataset}/s{self.seed}"

    @property
    def file_key(self) -> str:
        return f"{self.dataset}-s{self.seed}"


@dataclass
class Snapshot:
    state: str
    owner: dict
    files: dict[str, tuple[int, int]]
    latest_activity_ns: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--drifts", nargs="+", type=int, required=True)
    parser.add_argument("--probe-seconds", type=int, default=20)
    parser.add_argument("--stuck-seconds", type=int, default=1800)
    parser.add_argument("--worker-stale-seconds", type=int, default=180)
    parser.add_argument("--expected-workers", type=int, default=3)
    parser.add_argument("--log-lines", type=int, default=20)
    parser.add_argument("--error-lines", type=int, default=6)
    args = parser.parse_args()
    for name in (
        "probe_seconds",
        "stuck_seconds",
        "worker_stale_seconds",
        "expected_workers",
        "log_lines",
        "error_lines",
    ):
        value = getattr(args, name)
        minimum = 0 if name == "probe_seconds" else 1
        if value < minimum:
            parser.error(f"--{name.replace('_', '-')} must be >= {minimum}")
    return args


def family_root(args: argparse.Namespace, family: Family) -> Path:
    return args.root / f"family-{family.dataset}-s{family.seed}"


def run_dir(args: argparse.Namespace, family: Family, drift: int) -> Path:
    return family_root(args, family) / (
        f"{args.model_tag}-s{family.seed}-{family.dataset}-d{drift}"
    )


def owner_path(args: argparse.Namespace, family: Family) -> Path:
    return args.root / ".families" / f"{family.file_key}.owner.json"


def lock_path(args: argparse.Namespace, family: Family) -> Path:
    return args.root / ".families" / f"{family.file_key}.lock"


def read_owner(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {"invalid": True}
    except (OSError, json.JSONDecodeError):
        return {"invalid": True}


def lock_held(path: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(stream, fcntl.LOCK_UN)
        return False


def family_state(args: argparse.Namespace, family: Family) -> tuple[str, dict]:
    root = family_root(args, family)
    owner_file = owner_path(args, family)
    complete_stamp = root / ".family-complete"
    points_complete = all(
        (run_dir(args, family, drift) / "DONE").is_file()
        and (run_dir(args, family, drift) / "DONE").stat().st_size
        for drift in args.drifts
    )
    if complete_stamp.is_file() and complete_stamp.stat().st_size and points_complete:
        return "complete", {}
    if owner_file.is_file() and owner_file.stat().st_size:
        owner = read_owner(owner_file)
        return (
            "claimed" if lock_held(lock_path(args, family)) else "stale-owner"
        ), owner
    if root.is_dir():
        return "partial", {}
    return "pending", {}


def worker_log(args: argparse.Namespace, owner: dict) -> Path | None:
    worker = owner.get("worker")
    if not isinstance(worker, str) or not worker:
        return None
    path = args.root / "logs" / f"{worker}.log"
    return path if path.is_file() else None


def file_metadata(root: Path) -> dict[str, tuple[int, int]]:
    files: dict[str, tuple[int, int]] = {}
    if not root.is_dir():
        return files
    for base, dirs, names in os.walk(root):
        dirs.sort()
        names.sort()
        for name in names:
            path = Path(base) / name
            try:
                stat = path.stat()
            except OSError:
                continue
            files[str(path)] = (stat.st_size, stat.st_mtime_ns)
    return files


def take_snapshot(args: argparse.Namespace, family: Family) -> Snapshot:
    state, owner = family_state(args, family)
    files = file_metadata(family_root(args, family))
    for extra in (owner_path(args, family), worker_log(args, owner)):
        if extra is None or not extra.is_file():
            continue
        try:
            stat = extra.stat()
        except OSError:
            continue
        files[str(extra)] = (stat.st_size, stat.st_mtime_ns)
    latest = max((metadata[1] for metadata in files.values()), default=0)
    return Snapshot(state=state, owner=owner, files=files, latest_activity_ns=latest)


def changed_files(before: Snapshot, after: Snapshot) -> list[str]:
    changed = []
    for path in sorted(set(before.files) | set(after.files)):
        if before.files.get(path) != after.files.get(path):
            old_size = before.files.get(path, (0, 0))[0]
            new_size = after.files.get(path, (0, 0))[0]
            changed.append(f"{Path(path).name}:{old_size}->{new_size}B")
    return changed


def age_seconds(timestamp_ns: int) -> int | None:
    if timestamp_ns <= 0:
        return None
    return max(0, int(time.time() - timestamp_ns / 1_000_000_000))


def count_lines(path: Path) -> int:
    try:
        with path.open("rb") as stream:
            return sum(
                chunk.count(b"\n")
                for chunk in iter(lambda: stream.read(1024 * 1024), b"")
            )
    except OSError:
        return 0


def rollout_rows(run: Path, base: str) -> int:
    merged = run / f"{base}.jsonl"
    if merged.is_file():
        return count_lines(merged)
    rows = 0
    for pattern in (f"{base}.shard*.jsonl", f"{base}.shard*.partial"):
        rows += sum(count_lines(path) for path in run.glob(pattern))
    return rows


def log_stage(path: Path | None) -> str:
    if path is None:
        return "initialized"
    name = path.name
    patterns = (
        (("rollout-behavior", "beta-shard"), "behavior-rollout"),
        (("rollout-fresh", "fresh-shard"), "fresh-rollout"),
        (("grpo.log",), "grpo"),
        (("val-grads.log",), "validation-gradients"),
        (("ograds-shard",), "oracle-gradients"),
        (("score-shard",), "scoring"),
        (("merge.log",), "merge"),
        (("report.log",), "report"),
        (("regime-recovery-",), "cuda-recovery"),
    )
    for needles, stage in patterns:
        if any(needle in name for needle in needles):
            return stage
    return "pipeline"


def log_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    paths = []
    for path in root.rglob("*.log"):
        try:
            path.stat()
        except OSError:
            continue
        paths.append(path)
    return sorted(paths, key=lambda path: path.stat().st_mtime_ns, reverse=True)


def latest_log(root: Path) -> Path | None:
    paths = log_files(root)
    return paths[0] if paths else None


def last_nonempty_line(path: Path) -> str:
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            end = stream.tell()
            block = b""
            position = end
            while position > 0 and block.count(b"\n") < 2:
                size = min(8192, position)
                position -= size
                stream.seek(position)
                block = stream.read(size) + block
        lines = [
            line
            for line in block.decode("utf-8", errors="replace").splitlines()
            if line.strip()
        ]
        return lines[-1] if lines else ""
    except OSError:
        return ""


def tail_lines(path: Path, count: int) -> list[str]:
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            position = stream.tell()
            block = b""
            while position > 0 and block.count(b"\n") <= count:
                size = min(65536, position)
                position -= size
                stream.seek(position)
                block = stream.read(size) + block
        return block.decode("utf-8", errors="replace").splitlines()[-count:]
    except OSError:
        return []


def scan_errors(paths: list[Path], keep: int) -> tuple[int, list[tuple[Path, str]]]:
    total = 0
    matches: list[tuple[int, int, Path, str]] = []
    for path in paths:
        try:
            mtime = path.stat().st_mtime_ns
            with path.open(encoding="utf-8", errors="replace") as stream:
                for line_number, line in enumerate(stream, 1):
                    if ERROR_RE.search(line):
                        total += 1
                        matches.append((mtime, line_number, path, line.rstrip()))
        except OSError:
            continue
    matches.sort(key=lambda item: (item[0], item[1]))
    return total, [(path, line) for _, _, path, line in matches[-keep:]]


def last_json(path: Path) -> dict | None:
    line = last_nonempty_line(path)
    if not line:
        return None
    try:
        value = json.loads(line)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None


def point_status(args: argparse.Namespace, family: Family, drift: int) -> str:
    run = run_dir(args, family, drift)
    if (run / "DONE").is_file() and (run / "DONE").stat().st_size:
        return f"  d{drift} stage=complete"
    if not run.is_dir():
        return f"  d{drift} stage=pending"
    log = latest_log(run)
    stage = log_stage(log)
    if stage in {"pipeline", "initialized"}:
        if (run / "rollouts_fresh_train.manifest.json").is_file():
            stage = "post-rollout"
        elif drift > 0 and (run / f"policy_step_{drift}/policy_train.json").is_file():
            stage = "grpo-complete"
        elif (run / "rollouts_behavior_train.manifest.json").is_file():
            stage = "behavior-ready"
        elif (run / "prompts.json").is_file():
            stage = "prepared"
    fields = [
        f"  d{drift}",
        f"stage={stage}",
        f"behavior_rows={rollout_rows(run, 'rollouts_behavior_train')}",
        f"fresh_rows={rollout_rows(run, 'rollouts_fresh_train')}",
    ]
    stats_path = run / f"policy_step_{drift}/grpo_stats.jsonl"
    stats = last_json(stats_path) if stats_path.is_file() else None
    if stats is not None:
        steps = count_lines(stats_path)
        try:
            active = int(stats["nonzero_advantage_groups"])
            groups = int(stats["groups"])
            gradient = float(stats["grad_norm"])
            signal = (
                "no-mixed-reward"
                if active == 0
                else ("update" if gradient > 0 else "ERROR-zero-grad")
            )
            fields.extend(
                (
                    f"grpo_steps={steps}/{drift}",
                    f"reward={float(stats['reward_mean']):.3f}",
                    f"active_groups={active}/{groups}",
                    f"loss={float(stats['loss']):.3e}",
                    f"grad_norm={gradient:.3e}",
                    f"ratio={float(stats['mean_ratio']):.6f}",
                    f"learning_signal={signal}",
                )
            )
        except (KeyError, TypeError, ValueError):
            fields.append("metrics=invalid")
    attempts = sorted(
        (run / "logs").glob("regime-attempt-*.log") if (run / "logs").is_dir() else [],
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    if attempts:
        fields.append(f"attempt={attempts[0].stem}")
    recovery = last_json(run / "rollout_recovery.jsonl")
    if recovery is not None:
        if "recovery_generation_batch" in recovery:
            fields.append(f"recovery_batch={recovery['recovery_generation_batch']}")
        if "status" in recovery:
            fields.append(f"recovery_status={recovery['status']}")
    return " ".join(fields)


def classify(
    before: Snapshot,
    after: Snapshot,
    changes: list[str],
    stuck_seconds: int,
) -> tuple[str, str]:
    if after.state == "complete":
        if before.state != "complete":
            return "COMPLETE", "family_completed_during_probe"
        return "COMPLETE", "all_registered_points_complete"
    if after.state == "stale-owner":
        return "DEAD", "family_lock_released_but_owner_record_remains"
    if after.state == "pending":
        return "PENDING", "not_claimed"
    if after.state == "partial":
        age = age_seconds(after.latest_activity_ns)
        if age is not None and age <= stuck_seconds:
            return "RETRYING", "partial_artifacts_waiting_for_next_claim"
        return "STOPPED", "partial_artifacts_exist_without_a_live_family_lock"
    if changes:
        return "PROGRESSING", "artifact_or_log_changed_during_probe"
    age = age_seconds(after.latest_activity_ns)
    if age is None:
        return "STUCK", "lock_held_but_no_log_or_artifact_exists"
    if age > stuck_seconds:
        return "STUCK", f"lock_held_without_activity_for_{age}s"
    return "ALIVE", "lock_held_and_recent_activity_but_no_change_during_probe"


def owner_display(owner: dict) -> str:
    return json.dumps(owner, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def recent_workers(
    args: argparse.Namespace, snapshots: dict[Family, Snapshot]
) -> set[str]:
    workers = {
        str(snapshot.owner["worker"])
        for snapshot in snapshots.values()
        if snapshot.state == "claimed" and snapshot.owner.get("worker")
    }
    logs_root = args.root / "logs"
    now = time.time_ns()
    if logs_root.is_dir():
        for path in logs_root.glob("*.log"):
            if path.name.endswith("-keepalive.log"):
                continue
            try:
                age = (now - path.stat().st_mtime_ns) / 1_000_000_000
            except OSError:
                continue
            if age <= args.worker_stale_seconds:
                workers.add(path.stem)
    return workers


def main() -> None:
    args = parse_args()
    families = [
        Family(dataset, seed) for seed in args.seeds for dataset in args.datasets
    ]
    before = {family: take_snapshot(args, family) for family in families}
    active = [
        family
        for family, snapshot in before.items()
        if snapshot.state in {"claimed", "partial"}
    ]
    if args.probe_seconds and active:
        print(
            f"[status] checking {len(active)} active/partial families for "
            f"{args.probe_seconds}s; logs, rollouts, and checkpoints will be compared",
            flush=True,
        )
        time.sleep(args.probe_seconds)
    after = {family: take_snapshot(args, family) for family in families}

    print(f"profile={args.profile}")
    print(f"experiment_root={args.root}")
    print(f"status_probe_seconds={args.probe_seconds}")
    print(f"stuck_after_seconds={args.stuck_seconds}")
    print(f"log_tail_lines={args.log_lines}")
    generation = args.root / ".queue/generation.git"
    print(
        "generation_git="
        + (
            generation.read_text(encoding="utf-8").strip()
            if generation.is_file()
            else "not-started"
        )
    )

    workers = recent_workers(args, after)
    claims_by_worker: dict[str, list[str]] = {}
    for family, snapshot in after.items():
        worker = snapshot.owner.get("worker")
        if snapshot.state == "claimed" and isinstance(worker, str) and worker:
            claims_by_worker.setdefault(worker, []).append(family.key)
    print("== worker diagnostics ==")
    for worker in sorted(workers):
        log = args.root / "logs" / f"{worker}.log"
        log_age = age_seconds(log.stat().st_mtime_ns) if log.is_file() else None
        worker_errors, _ = scan_errors([log] if log.is_file() else [], args.error_lines)
        state = "CLAIMED" if worker in claims_by_worker else "RECENT_LOG_ONLY"
        claims = ",".join(claims_by_worker.get(worker, [])) or "none"
        last_line = last_nonempty_line(log) if log.is_file() else ""
        print(
            f"worker={worker} state={state} claims={claims} "
            f"log_age={'none' if log_age is None else f'{log_age}s'} "
            f"error_matches={worker_errors}"
        )
        if last_line:
            print(f"  last_log_line={last_line}")
    if not workers:
        print("worker=none state=NOT_OBSERVED")

    verdict_counts: dict[str, int] = {}
    for family in families:
        snapshot = after[family]
        changes = changed_files(before[family], snapshot)
        verdict, reason = classify(
            before[family], snapshot, changes, args.stuck_seconds
        )
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        suffix = f" {owner_display(snapshot.owner)}" if snapshot.owner else ""
        print(f"{family.key} {snapshot.state}{suffix}")
        age = age_seconds(snapshot.latest_activity_ns)
        age_text = "none" if age is None else f"{age}s"
        family_logs = log_files(family_root(args, family))
        owner_log = worker_log(args, snapshot.owner)
        checked_logs = list(family_logs)
        if owner_log is not None and owner_log not in checked_logs:
            checked_logs.append(owner_log)
        error_count, errors = scan_errors(checked_logs, args.error_lines)
        if error_count == 0:
            error_assessment = "none"
        elif verdict in {"PROGRESSING", "ALIVE", "COMPLETE"}:
            error_assessment = "history_present_but_not_blocking_current_progress"
        elif verdict in {"STUCK", "DEAD", "STOPPED"}:
            error_assessment = "fatal_evidence_correlates_with_current_failure"
        else:
            error_assessment = "history_present_during_retry"
        print(
            f"  verdict={verdict} reason={reason} activity_age={age_text} "
            f"logs_checked={len(checked_logs)} error_matches={error_count} "
            f"error_assessment={error_assessment}"
        )
        if changes:
            print("  observed_changes=" + ", ".join(changes[:8]))
        for drift in args.drifts:
            print(point_status(args, family, drift))
        if snapshot.state not in {"complete", "pending"}:
            latest = family_logs[0] if family_logs else None
            if latest is not None:
                log_age = age_seconds(latest.stat().st_mtime_ns)
                print(
                    f"  latest_log={latest} age={log_age}s (last {args.log_lines} lines)"
                )
                for line in tail_lines(latest, args.log_lines):
                    print(f"    | {line}")
            if errors:
                print("  error_evidence_from_all_checked_logs:")
                for path, line in errors:
                    print(f"    ! {path}: {line}")
            if owner_log is not None and owner_log != latest:
                log_age = age_seconds(owner_log.stat().st_mtime_ns)
                print(
                    f"  worker_log={owner_log} age={log_age}s (last {args.log_lines} lines)"
                )
                for line in tail_lines(owner_log, args.log_lines):
                    print(f"    | {line}")

    complete = verdict_counts.get("COMPLETE", 0)
    progressing = verdict_counts.get("PROGRESSING", 0)
    alive = verdict_counts.get("ALIVE", 0)
    stuck = verdict_counts.get("STUCK", 0)
    dead = verdict_counts.get("DEAD", 0)
    stopped = verdict_counts.get("STOPPED", 0)
    pending = verdict_counts.get("PENDING", 0)
    retrying = verdict_counts.get("RETRYING", 0)
    degraded = stuck + dead + stopped > 0 or len(workers) < args.expected_workers
    if complete == len(families):
        overall = "COMPLETE"
        action = "none"
    elif progressing or alive:
        overall = "DEGRADED" if degraded else "RUNNING"
        action = (
            "inspect_STUCK_DEAD_families_and_missing_workers" if degraded else "none"
        )
    elif retrying and workers:
        overall = "RECOVERING"
        action = "wait_for_automatic_retry"
    elif any((stuck, dead, stopped, retrying)):
        overall = "STOPPED"
        action = "restart_missing_workers_after_node_cleanup"
    else:
        overall = "NOT_STARTED"
        action = "start_workers"
    print("== diagnosis ==")
    print(
        f"workers_observed={len(workers)}/{args.expected_workers} "
        f"worker_ids={','.join(sorted(workers)) or 'none'}"
    )
    print(
        f"families_total={len(families)} complete={complete} progressing={progressing} "
        f"alive={alive} retrying={retrying} stuck={stuck} dead={dead} "
        f"stopped={stopped} pending={pending}"
    )
    print(f"overall_verdict={overall}")
    print(f"recommended_action={action}")
    report = args.results / "FINAL_REPORT.md"
    if report.is_file() and report.stat().st_size:
        print(f"report={report}")


if __name__ == "__main__":
    main()
