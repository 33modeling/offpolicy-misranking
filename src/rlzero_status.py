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
    r"CUDA error|CUBLAS_STATUS|cuBLAS|CUDA out of memory|OutOfMemoryError|device-side assert|"
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
    artifact_activity_ns: int  # family-root files only (no worker log, no owner record)
    pipeline_activity: dict | None
    pipeline_activity_path: Path | None


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
    parser.add_argument("--heartbeat-stale-seconds", type=int, default=90)
    parser.add_argument("--expected-workers", type=int, default=3)
    parser.add_argument("--config-sha", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--generation-batch", type=int, required=True)
    parser.add_argument("--gradient-micro-batch", type=int, required=True)
    parser.add_argument("--logprob-micro-batch", type=int, required=True)
    parser.add_argument("--min-recovery-generation-batch", type=int, required=True)
    parser.add_argument("--log-lines", type=int, default=20)
    parser.add_argument("--error-lines", type=int, default=6)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="after the one-screen summary, print per-point detail, telemetry and log tails",
    )
    args = parser.parse_args()
    for name in (
        "probe_seconds",
        "stuck_seconds",
        "worker_stale_seconds",
        "heartbeat_stale_seconds",
        "expected_workers",
        "generation_batch",
        "gradient_micro_batch",
        "logprob_micro_batch",
        "min_recovery_generation_batch",
        "log_lines",
        "error_lines",
    ):
        value = getattr(args, name)
        minimum = 0 if name == "probe_seconds" else 1
        if value < minimum:
            parser.error(f"--{name.replace('_', '-')} must be >= {minimum}")
    if args.min_recovery_generation_batch > args.generation_batch:
        parser.error(
            "--min-recovery-generation-batch cannot exceed --generation-batch"
        )
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


def heartbeat_path(args: argparse.Namespace, worker: str) -> Path:
    return args.root / ".workers" / f"{worker}.json"


def read_owner(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {"invalid": True}
    except (OSError, json.JSONDecodeError):
        return {"invalid": True}


def expected_family_stamp(
    args: argparse.Namespace, family: Family
) -> str | None:
    generation = args.root / ".queue/generation.git"
    try:
        generation_git = generation.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not generation_git:
        return None
    return (
        f"{generation_git} {args.config_sha} {args.model_revision} "
        f"{family.dataset} {family.seed}"
    )


def lock_held(path: Path) -> bool:
    # A diagnostic must not create lock files or crash on a read-only queue.
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
        return True  # cannot test it from here: assume the owner still holds it


def family_state(args: argparse.Namespace, family: Family) -> tuple[str, dict]:
    root = family_root(args, family)
    owner_file = owner_path(args, family)
    complete_stamp = root / ".family-complete"
    points_complete = all(
        (run_dir(args, family, drift) / "DONE").is_file()
        and (run_dir(args, family, drift) / "DONE").stat().st_size
        for drift in args.drifts
    )
    expected_stamp = expected_family_stamp(args, family)
    if complete_stamp.is_file() and points_complete and expected_stamp is not None:
        try:
            if complete_stamp.read_text(encoding="utf-8").strip() == expected_stamp:
                return "complete", {}
        except OSError:
            pass
    loop_marker = args.root / ".families" / f"{family.file_key}.loop"
    if loop_marker.is_file() and loop_marker.stat().st_size:
        try:
            info = dict(
                line.split("=", 1) for line in loop_marker.read_text(encoding="utf-8").splitlines() if "=" in line
            )
        except OSError:
            info = {}
        return "looping", {"loop": info}
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
            if name == ".pipeline-activity.json" or ".pipeline-activity.json.tmp." in name:
                continue
            if name == "keepalive.log":
                # Written by the point's own GPU keepalive; never evidence of progress.
                continue
            path = Path(base) / name
            try:
                stat = path.stat()
            except OSError:
                continue
            files[str(path)] = (stat.st_size, stat.st_mtime_ns)
    return files


def latest_pipeline_activity(root: Path) -> tuple[Path | None, dict | None]:
    candidates: list[tuple[int, Path, dict]] = []
    if not root.is_dir():
        return None, None
    for path in root.rglob(".pipeline-activity.json"):
        record = read_owner(path)
        try:
            observed = int(record.get("observed_at_epoch", 0))
            fallback = path.stat().st_mtime_ns // 1_000_000_000
        except (OSError, TypeError, ValueError):
            continue
        candidates.append((observed or fallback, path, record))
    if not candidates:
        return None, None
    _, path, record = max(candidates, key=lambda item: item[0])
    return path, record


def take_snapshot(args: argparse.Namespace, family: Family) -> Snapshot:
    state, owner = family_state(args, family)
    files = file_metadata(family_root(args, family))
    artifact_latest = max((metadata[1] for metadata in files.values()), default=0)
    for extra in (owner_path(args, family), worker_log(args, owner)):
        if extra is None or not extra.is_file():
            continue
        try:
            stat = extra.stat()
        except OSError:
            continue
        files[str(extra)] = (stat.st_size, stat.st_mtime_ns)
    latest = max((metadata[1] for metadata in files.values()), default=0)
    activity_path, activity = latest_pipeline_activity(family_root(args, family))
    return Snapshot(
        state=state,
        owner=owner,
        files=files,
        latest_activity_ns=latest,
        artifact_activity_ns=artifact_latest,
        pipeline_activity=activity,
        pipeline_activity_path=activity_path,
    )


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


def record_age_seconds(record: dict | None, key: str, scale: int = 1) -> int | None:
    if record is None:
        return None
    try:
        timestamp = int(record[key]) / scale
    except (KeyError, TypeError, ValueError):
        return None
    return max(0, int(time.time() - timestamp))


def count_lines(path: Path) -> int:
    try:
        with path.open("rb") as stream:
            lines = 0
            last = b""
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                lines += chunk.count(b"\n")
                last = chunk[-1:]
            return lines + int(bool(last) and last != b"\n")
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
        (("regime-recovery-", "recovery-rollout-"), "cuda-recovery"),
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


def latest_stage_log(root: Path) -> Path | None:
    paths = [path for path in log_files(root) if log_stage(path) != "pipeline"]
    return paths[0] if paths else latest_log(root)


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


def current_attempt_errors(
    root: Path, keep: int
) -> tuple[bool, int, list[tuple[Path, str]], Path | None]:
    manifests: list[tuple[int, Path, dict]] = []
    for path in root.rglob("regime-attempt-*.log.start.json"):
        record = read_owner(path)
        try:
            started = int(record["started_at_ns"])
        except (KeyError, TypeError, ValueError):
            continue
        if (
            record.get("schema") == "offpolicy-pipeline-attempt/v1"
            and isinstance(record.get("log_offsets"), dict)
            and isinstance(record.get("attempt_log"), str)
        ):
            manifests.append((started, path, record))
    if not manifests:
        return False, 0, [], None

    _, manifest, record = max(manifests, key=lambda item: item[0])
    logs_root = manifest.parent
    attempt_name = record["attempt_log"]
    raw_offsets = record["log_offsets"]
    total = 0
    matches: list[tuple[int, int, Path, str]] = []
    for path in logs_root.rglob("*.log"):
        try:
            relative = path.relative_to(logs_root).as_posix()
            offset = 0 if relative == attempt_name else int(raw_offsets.get(relative, 0))
            stat = path.stat()
            if offset < 0:
                return False, 0, [], manifest
            if stat.st_size < offset:
                offset = 0
            if stat.st_size == offset:
                continue
            with path.open("rb") as stream:
                stream.seek(offset)
                lines = stream.read().decode("utf-8", errors="replace").splitlines()
            for line_number, line in enumerate(lines, 1):
                if ERROR_RE.search(line):
                    total += 1
                    matches.append((stat.st_mtime_ns, line_number, path, line))
        except (OSError, TypeError, ValueError):
            return False, 0, [], manifest
    matches.sort(key=lambda item: (item[0], item[1]))
    evidence = [(path, line) for _, _, path, line in matches[-keep:]]
    return True, total, evidence, manifest


def last_json(path: Path) -> dict | None:
    line = last_nonempty_line(path)
    if not line:
        return None
    try:
        value = json.loads(line)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None


def runtime_contract_issues(
    args: argparse.Namespace, run: Path, recovery: dict | None
) -> tuple[list[str], dict]:
    config_path = run / "run_config.json"
    config = read_owner(config_path) if config_path.is_file() else {}
    issues: list[str] = []
    expected = {
        "gen_batch": args.generation_batch,
        "gradient_micro_batch": args.gradient_micro_batch,
        "grpo_logprob_micro_batch": args.logprob_micro_batch,
    }
    if config.get("invalid"):
        issues.append("invalid_run_config")
    elif config:
        for key, wanted in expected.items():
            try:
                actual = int(config[key])
            except (KeyError, TypeError, ValueError):
                issues.append(f"missing_{key}")
                continue
            if actual != wanted:
                issues.append(f"{key}:{actual}!={wanted}")
    if recovery is not None and "recovery_generation_batch" in recovery:
        try:
            recovery_batch = int(recovery["recovery_generation_batch"])
            if recovery_batch < args.min_recovery_generation_batch:
                issues.append(
                    "recovery_batch_below_floor:"
                    f"{recovery_batch}<{args.min_recovery_generation_batch}"
                )
        except (TypeError, ValueError):
            issues.append("invalid_recovery_generation_batch")
    return issues, config


def point_status(
    args: argparse.Namespace, family: Family, drift: int
) -> tuple[str, list[str]]:
    run = run_dir(args, family, drift)
    if not run.is_dir():
        return f"  d{drift} stage=pending", []
    recovery = last_json(run / "rollout_recovery.jsonl")
    issues, config = runtime_contract_issues(args, run, recovery)
    done = (run / "DONE").is_file() and (run / "DONE").stat().st_size
    log = latest_stage_log(run)
    stage = log_stage(log)
    if done:
        stage = "complete"
    elif stage in {"pipeline", "initialized"}:
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
        f"generation_batch={config.get('gen_batch', 'missing')}/{args.generation_batch}",
        (
            "gradient_batch="
            f"{config.get('gradient_micro_batch', 'missing')}/{args.gradient_micro_batch}"
        ),
        (
            "logprob_batch="
            f"{config.get('grpo_logprob_micro_batch', 'missing')}/{args.logprob_micro_batch}"
        ),
    ]
    stats_path = run / f"policy_step_{drift}/grpo_stats.jsonl"
    stats = last_json(stats_path) if stats_path.is_file() else None
    if stats is not None:
        previous_drift = max([d for d in args.drifts if d < drift], default=0)
        steps = min(drift, previous_drift + count_lines(stats_path))
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
    if recovery is not None:
        if "recovery_generation_batch" in recovery:
            fields.append(f"recovery_batch={recovery['recovery_generation_batch']}")
        if "status" in recovery:
            fields.append(f"recovery_status={recovery['status']}")
        if "failure_kind" in recovery:
            fields.append(f"recovery_reason={recovery['failure_kind']}")
    if issues:
        fields.append("contract_errors=" + ",".join(issues))
    return " ".join(fields), issues


def classify(
    before: Snapshot,
    after: Snapshot,
    changes: list[str],
    stuck_seconds: int,
    heartbeat_fresh: bool,
    heartbeat_age: int | None,
    telemetry_stale_seconds: int,
) -> tuple[str, str]:
    if after.state == "complete":
        if before.state != "complete":
            return "COMPLETE", "family_completed_during_probe"
        return "COMPLETE", "all_registered_points_complete"
    if after.state == "looping":
        return "LOOPING", "same_failure_repeated_until_the_supervisor_stopped_retrying"
    if after.state == "stale-owner":
        # flock held on another node is invisible over NFS, so from here the
        # family looks unowned. Fresh watchdog telemetry, a fresh worker
        # heartbeat, or a recent write proves the owner is alive: fall through
        # to the telemetry/artifact logic (HUNG / COMPUTING) instead of DEAD.
        telemetry_age = record_age_seconds(after.pipeline_activity, "observed_at_epoch")
        artifact_age = age_seconds(after.artifact_activity_ns)
        owner_alive = heartbeat_fresh or (
            telemetry_age is not None and telemetry_age <= telemetry_stale_seconds
        )
        if not owner_alive:
            return "DEAD", "family_lock_released_but_owner_record_remains"
    if after.state == "pending":
        return "PENDING", "not_claimed"
    if after.state == "partial":
        age = age_seconds(after.latest_activity_ns)
        if age is not None and age <= stuck_seconds:
            return "RETRYING", "partial_artifacts_waiting_for_next_claim"
        return "STOPPED", "partial_artifacts_exist_without_a_live_family_lock"
    telemetry = after.pipeline_activity
    telemetry_age = record_age_seconds(telemetry, "observed_at_epoch")
    fresh_telemetry_state = None
    if telemetry is not None and telemetry_age is not None and telemetry_age <= telemetry_stale_seconds:
        if telemetry.get("schema") != "offpolicy-pipeline-activity/v1":
            return "UNKNOWN", "pipeline_telemetry_schema_invalid"
        fresh_telemetry_state = str(telemetry.get("state", "invalid"))
    # The supervisor's own "[regime-watchdog] ... idle" line lands in the worker
    # log and counted as a change, turning a confirmed-idle pipeline into
    # PROGRESSING. Measured idleness wins over that echo.
    if fresh_telemetry_state in {"idle-suspected", "terminating-idle"}:
        idle = telemetry.get("idle_seconds", "unknown")
        if fresh_telemetry_state == "idle-suspected":
            return "IDLE", f"pipeline_idle_suspected_for_{idle}s"
        return "STUCK", f"pipeline_confirmed_idle_for_{idle}s"
    if changes:
        return "PROGRESSING", "artifact_or_log_changed_during_probe"
    if telemetry is not None and telemetry_age is not None:
        telemetry_state = str(telemetry.get("state", "invalid"))
        if telemetry_age <= telemetry_stale_seconds:
            if telemetry.get("schema") != "offpolicy-pipeline-activity/v1":
                return "UNKNOWN", "pipeline_telemetry_schema_invalid"
            if telemetry_state in {"output-progress", "cpu-active", "gpu-active"}:
                # Supervisors pinned before 2026-09-06 counted the pipeline's own
                # GPU keepalive as compute, so a hung point reports gpu-active for
                # days while writing nothing. Telemetry "activity" without any
                # artifact or log change for far longer than a stage takes is a
                # hang, not computing.
                artifact_age = age_seconds(after.artifact_activity_ns)
                hang_after = max(6 * 3600, 8 * stuck_seconds)
                if (
                    telemetry_state != "output-progress"
                    and artifact_age is not None
                    and artifact_age > hang_after
                ):
                    return (
                        "HUNG",
                        f"telemetry_{telemetry_state}_but_no_artifact_or_log_change_for_{artifact_age}s",
                    )
                return "COMPUTING", f"pipeline_telemetry_{telemetry_state}"
            if telemetry_state == "idle-suspected":
                idle = telemetry.get("idle_seconds", "unknown")
                return "IDLE", f"pipeline_idle_suspected_for_{idle}s"
            if telemetry_state == "terminating-idle":
                idle = telemetry.get("idle_seconds", "unknown")
                return "STUCK", f"pipeline_confirmed_idle_for_{idle}s"
            if telemetry_state == "telemetry-error":
                return "UNKNOWN", "pipeline_activity_probe_failed_kill_suppressed"
            if telemetry_state == "exited":
                return "RETRYING", "pipeline_attempt_exited_under_live_family_lock"
            if telemetry_state in {"starting", "process-alive"}:
                return "ALIVE", f"pipeline_telemetry_{telemetry_state}"
            return "UNKNOWN", f"pipeline_telemetry_state_invalid:{telemetry_state}"
    if heartbeat_fresh:
        artifact_age = age_seconds(after.artifact_activity_ns)
        if artifact_age is not None and artifact_age > max(6 * 3600, 8 * stuck_seconds):
            return "HUNG", f"worker_heartbeat_fresh_but_no_artifact_or_log_change_for_{artifact_age}s"
        return "ALIVE", "worker_heartbeat_fresh_but_pipeline_progress_unobserved"
    age = age_seconds(after.latest_activity_ns)
    if age is None:
        return "UNKNOWN", "lock_held_but_no_worker_or_pipeline_telemetry_exists"
    if age > stuck_seconds:
        heartbeat = "none" if heartbeat_age is None else f"{heartbeat_age}s_old"
        return (
            "UNKNOWN",
            f"shared_activity_{age}s_old_and_worker_heartbeat_{heartbeat}",
        )
    return "ALIVE", "recent_shared_activity_without_current_pipeline_telemetry"


def owner_display(owner: dict) -> str:
    return json.dumps(owner, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def worker_heartbeat(
    args: argparse.Namespace, worker: str
) -> tuple[dict | None, int | None, bool]:
    path = heartbeat_path(args, worker)
    if not path.is_file():
        return None, None, False
    record = read_owner(path)
    age = record_age_seconds(record, "heartbeat_at_ns", 1_000_000_000)
    fresh = (
        record.get("schema") == "offpolicy-worker-heartbeat/v1"
        and record.get("worker") == worker
        and record.get("state") == "running"
        and age is not None
        and age <= args.heartbeat_stale_seconds
    )
    return record, age, fresh


def recent_workers(
    args: argparse.Namespace, snapshots: dict[Family, Snapshot]
) -> set[str]:
    claimed_workers = {
        str(snapshot.owner["worker"])
        for snapshot in snapshots.values()
        if snapshot.state == "claimed" and snapshot.owner.get("worker")
    }
    workers = {
        worker for worker in claimed_workers if worker_heartbeat(args, worker)[2]
    }
    workers_root = args.root / ".workers"
    if workers_root.is_dir():
        for path in workers_root.glob("*.json"):
            record = read_owner(path)
            worker = record.get("worker")
            if isinstance(worker, str) and worker and worker_heartbeat(args, worker)[2]:
                workers.add(worker)
    logs_root = args.root / "logs"
    now = time.time_ns()
    if logs_root.is_dir():
        for path in logs_root.glob("*.log"):
            if (
                path.name.endswith("-keepalive.log")
                or path.name.startswith("status-")
                or path.name in {"ALERTS.log"}
            ):
                continue
            try:
                age = (now - path.stat().st_mtime_ns) / 1_000_000_000
            except OSError:
                continue
            if age <= args.worker_stale_seconds:
                workers.add(path.stem)
    return workers


def fmt_age(seconds: int | None) -> str:
    if seconds is None:
        return "-"
    if seconds < 90:
        return f"{seconds}s"
    if seconds < 5400:
        return f"{seconds // 60}m"
    if seconds < 172800:
        return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}"
    return f"{seconds // 86400}d"


def last_progress(run: Path) -> str:
    """The pipeline's own `[progress] <run>  k/8 <stage>  +<min>` line, minus the run name."""
    main_log = run / "logs/main.log"
    if not main_log.is_file():
        return ""
    text = ""
    for line in tail_lines(main_log, 400):
        if "[progress]" in line:
            text = line
    if not text:
        return ""
    text = text.split("[progress]", 1)[1].strip()
    # "<run>  k/8 <label> (detail)  +12min" -> "k/8 <label> +12min"
    fields = [field.strip() for field in re.split(r"\s{2,}", text) if field.strip()]
    if len(fields) >= 2:
        fields = fields[1:]
    stage = re.sub(r"\s*\(.*?\)", "", fields[0]).strip() if fields else text
    words = stage.split()
    if len(words) >= 2 and re.fullmatch(r"\d+/\d+", words[0]):
        stage = f"{words[0]} {words[1]}"
    elapsed = fields[-1] if len(fields) > 1 and fields[-1].startswith("+") else ""
    return f"{stage} {elapsed}".strip()


def current_point(args: argparse.Namespace, family: Family) -> tuple[int | None, Path | None, str, list[int]]:
    """Count every DONE point and locate the most recently active unfinished one.

    The launcher can defer d0 evaluation until after d25/d100/d400 training.
    An unfinished earlier drift therefore cannot terminate the completion scan.
    """
    done: list[int] = []
    unfinished: list[tuple[int, Path, str]] = []
    for drift in args.drifts:
        run = run_dir(args, family, drift)
        if not run.is_dir():
            unfinished.append((drift, run, "not-started"))
            continue
        stamp = run / "DONE"
        if stamp.is_file() and stamp.stat().st_size:
            done.append(drift)
            continue
        unfinished.append((drift, run, "active"))
    active = [point for point in unfinished if point[2] == "active"]
    if active:
        def activity(point: tuple[int, Path, str]) -> int:
            root = point[1]
            latest = max((mtime for _, mtime in file_metadata(root).values()), default=0)
            path, record = latest_pipeline_activity(root)
            if path is not None and record.get("schema") == "offpolicy-pipeline-activity/v1":
                try:
                    latest = max(latest, path.stat().st_mtime_ns)
                except OSError:
                    pass
            return latest
        drift, run, kind = max(active, key=activity)
        return drift, run, kind, done
    if unfinished:
        drift, run, kind = unfinished[0]
        return drift, run, kind, done
    return None, None, "all-done", done


def short_error(errors: list[tuple[Path, str]], width: int = 70) -> str:
    if not errors:
        return ""
    _, line = errors[-1]
    line = re.sub(r"\s+", " ", line).strip()
    return line if len(line) <= width else line[: width - 3] + "..."


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
            f"[status] probing {len(active)} active families for {args.probe_seconds}s ...",
            flush=True,
        )
        time.sleep(args.probe_seconds)
    after = {family: take_snapshot(args, family) for family in families}

    generation = args.root / ".queue/generation.git"
    generation_git = (
        generation.read_text(encoding="utf-8").strip() if generation.is_file() else "not-started"
    )

    workers = recent_workers(args, after)
    heartbeat_workers: set[str] = set()
    workers_root = args.root / ".workers"
    if workers_root.is_dir():
        for path in workers_root.glob("*.json"):
            worker = read_owner(path).get("worker")
            if isinstance(worker, str) and worker:
                heartbeat_workers.add(worker)
    claims_by_worker: dict[str, list[str]] = {}
    for family, snapshot in after.items():
        worker = snapshot.owner.get("worker")
        if snapshot.state == "claimed" and isinstance(worker, str) and worker:
            claims_by_worker.setdefault(worker, []).append(family.key)
    diagnostic_workers = workers | heartbeat_workers | set(claims_by_worker)
    worker_rows: list[dict] = []
    for worker in sorted(diagnostic_workers):
        log = args.root / "logs" / f"{worker}.log"
        log_age = age_seconds(log.stat().st_mtime_ns) if log.is_file() else None
        worker_errors, _ = scan_errors([log] if log.is_file() else [], args.error_lines)
        if worker in claims_by_worker:
            state = "CLAIMED"
        elif worker in workers:
            state = "AVAILABLE"
        else:
            state = "STALE"
        heartbeat, heartbeat_age, heartbeat_fresh = worker_heartbeat(args, worker)
        if heartbeat_fresh:
            evidence = "heartbeat"
        elif log_age is not None and log_age <= args.worker_stale_seconds:
            evidence = "recent-log"
        elif heartbeat is not None:
            evidence = "stale-heartbeat"
        else:
            evidence = "lock-only"
        worker_rows.append(
            {
                "worker": worker,
                "state": state,
                "claims": claims_by_worker.get(worker, []),
                "log": log,
                "log_age": log_age,
                "heartbeat_age": heartbeat_age,
                "evidence": evidence,
                "errors": worker_errors,
                "last_line": last_nonempty_line(log) if log.is_file() else "",
            }
        )

    # ---- per-family classification (once) ----
    rows: list[dict] = []
    verdict_counts: dict[str, int] = {}
    contract_errors: list[str] = []
    points_done = 0
    for family in families:
        snapshot = after[family]
        changes = changed_files(before[family], snapshot)
        worker = snapshot.owner.get("worker")
        heartbeat_age = None
        heartbeat_fresh = False
        if isinstance(worker, str) and worker:
            _, heartbeat_age, heartbeat_fresh = worker_heartbeat(args, worker)
        verdict, reason = classify(
            before[family],
            snapshot,
            changes,
            args.stuck_seconds,
            heartbeat_fresh,
            heartbeat_age,
            args.heartbeat_stale_seconds,
        )
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        family_logs = log_files(family_root(args, family))
        owner_log = worker_log(args, snapshot.owner)
        checked_logs = list(family_logs)
        if owner_log is not None and owner_log not in checked_logs:
            checked_logs.append(owner_log)
        error_count, errors = scan_errors(checked_logs, args.error_lines)
        boundary, current_error_count, current_errors, attempt_manifest = (
            current_attempt_errors(family_root(args, family), args.error_lines)
        )
        points = []
        for drift in args.drifts:
            point, issues = point_status(args, family, drift)
            points.append(point)
            contract_errors.extend(f"{family.key}/d{drift}:{issue}" for issue in issues)
        drift, run, kind, done_drifts = current_point(args, family)
        points_done += len(done_drifts)
        stage = "-"
        grpo = "-"
        if kind == "active" and run is not None:
            stage = last_progress(run)
            if not stage:
                stage = log_stage(latest_stage_log(run))
            stats_path = run / f"policy_step_{drift}/grpo_stats.jsonl"
            if drift and stats_path.is_file():
                # the stats file of point d_k covers steps (d_{k-1}, d_k]; show the
                # cumulative step so 300 rows in d400 reads 400/400, not 300/400
                previous = max([d for d in args.drifts if d < drift], default=0)
                grpo = f"{min(drift, previous + count_lines(stats_path))}/{drift}"
        elif kind == "not-started":
            stage = "not started" if not done_drifts else "next"
        note = ""
        recovery = last_json(run / "rollout_recovery.jsonl") if run is not None and run.is_dir() else None
        write_age = fmt_age(age_seconds(snapshot.artifact_activity_ns))
        err_text = short_error(current_errors, 60) if current_errors else ""
        if verdict == "LOOPING":
            info = snapshot.owner.get("loop", {}) if isinstance(snapshot.owner, dict) else {}
            note = (f"NEEDS YOU: failed {info.get('consecutive_failures', '?')} times in a row, retries stopped. "
                    f"last error: {str(info.get('last_error', ''))[:110]} -> fix, then relaunch with OM_RLZERO_CLEAR_LOOPS=1")
        elif verdict == "HUNG":
            note = f"NEEDS YOU: alive but nothing written for {write_age} -> Ctrl-C this worker, git pull, run h100"
            if err_text:
                note += f" | last error: {err_text}"
        elif verdict in {"DEAD", "STOPPED"}:
            if workers:
                note = f"QUEUED: no worker on it for {write_age}; the next free worker picks it up where it stopped"
            else:
                note = f"NEEDS YOU: no worker anywhere for {write_age} -> start one: run h100"
            if err_text:
                note += f" | last error: {err_text}"
        elif verdict == "STUCK":
            note = f"AUTO: confirmed idle for {write_age}; the watchdog kills and resumes the point by itself"
        elif verdict == "RETRYING":
            note = "AUTO: failed attempt, retry scheduled by the supervisor"
            if err_text:
                note += f" | error: {err_text}"
        elif verdict == "UNKNOWN":
            note = f"CHECK: {reason.replace('_', ' ')}"
        elif current_errors:
            note = f"ERROR in current attempt but still moving: {err_text}"
        elif recovery is not None and recovery.get("status") not in (None, "recovered", "completed"):
            recovery_kind = recovery.get("failure_kind") or recovery.get("stage") or "unknown-cause"
            rec_logs = sorted((run / "logs").glob("regime-recovery-*.log"), key=lambda q: q.stat().st_mtime_ns) if run is not None and (run / "logs").is_dir() else []
            _, rec_errors = scan_errors(rec_logs[-1:], 3) if rec_logs else (0, [])
            tail = f" | {short_error(rec_errors, 70)}" if rec_errors else ""
            if recovery.get("status") == "failed":
                note = f"AUTO: CUDA recovery failed once ({recovery_kind}, batch {recovery.get('recovery_generation_batch', '?')}); supervisor retries the point. If this row still says failed next time, that node has a CUDA problem{tail}"
            else:
                note = f"AUTO: CUDA recovery {recovery.get('status')} ({recovery_kind}, batch {recovery.get('recovery_generation_batch', '?')}){tail}"
        elif recovery is not None and recovery.get("status") == "completed" and verdict in {"PROGRESSING", "COMPUTING", "ALIVE"}:
            note = "ok (recovered from a CUDA error earlier in this point)"
        elif errors and verdict in {"PROGRESSING", "COMPUTING", "ALIVE"}:
            note = "ok (old errors in earlier attempts, current attempt clean)"
        elif verdict in {"PROGRESSING", "COMPUTING", "ALIVE"}:
            note = "ok"
        rows.append(
            {
                "family": family,
                "snapshot": snapshot,
                "changes": changes,
                "verdict": verdict,
                "reason": reason,
                "worker": worker if isinstance(worker, str) else "",
                "drift": drift,
                "kind": kind,
                "done_drifts": done_drifts,
                "stage": stage,
                "grpo": grpo,
                "age": age_seconds(snapshot.artifact_activity_ns),
                "note": note,
                "points": points,
                "checked_logs": checked_logs,
                "errors": errors,
                "error_count": error_count,
                "current_errors": current_errors,
                "current_error_count": current_error_count,
                "boundary": boundary,
                "attempt_manifest": attempt_manifest,
                "family_logs": family_logs,
                "owner_log": owner_log,
            }
        )

    complete = verdict_counts.get("COMPLETE", 0)
    progressing = verdict_counts.get("PROGRESSING", 0)
    computing = verdict_counts.get("COMPUTING", 0)
    alive = verdict_counts.get("ALIVE", 0)
    idle = verdict_counts.get("IDLE", 0)
    unknown = verdict_counts.get("UNKNOWN", 0)
    stuck = verdict_counts.get("STUCK", 0)
    dead = verdict_counts.get("DEAD", 0)
    stopped = verdict_counts.get("STOPPED", 0)
    pending = verdict_counts.get("PENDING", 0)
    retrying = verdict_counts.get("RETRYING", 0)
    hung = verdict_counts.get("HUNG", 0)
    missing_workers = len(workers) < args.expected_workers
    degraded = stuck + dead + stopped + idle + unknown + hung > 0 or missing_workers
    if contract_errors:
        overall = "INVALID"
        action = "fix_runtime_contract_before_continuing"
    elif complete == len(families):
        overall = "COMPLETE"
        action = "none"
    elif hung and not (progressing or computing):
        overall = "HUNG"
        action = "Ctrl-C_that_worker__git_pull__relaunch_run_h100__partials_resume"
    elif progressing or computing or alive or idle:
        overall = "DEGRADED" if degraded else "RUNNING"
        if hung:
            action = "Ctrl-C_the_HUNG_family_worker__git_pull__relaunch_run_h100"
        elif stuck + dead + stopped > 0 or missing_workers:
            action = "inspect_STUCK_DEAD_families_and_missing_workers"
        elif idle:
            action = "wait_for_next_watchdog_confirmation"
        elif unknown:
            action = "inspect_node_telemetry_before_restarting_any_worker"
        else:
            action = "none"
    elif retrying and workers:
        overall = "RECOVERING"
        action = "wait_for_automatic_retry"
    elif any((stuck, dead, stopped, retrying)):
        overall = "STOPPED"
        action = "restart_missing_workers_after_node_cleanup"
    elif unknown:
        overall = "UNKNOWN"
        action = "inspect_node_telemetry_before_restarting_any_worker"
    elif workers:
        overall = "STARTING" if complete == 0 else "RUNNING"
        action = "wait_for_worker_preflight_or_queue_claim"
    elif complete:
        overall = "INCOMPLETE"
        action = "start_workers"
    else:
        overall = "NOT_STARTED"
        action = "start_workers"

    # ---- one screen ----
    total_points = len(families) * len(args.drifts)
    now = time.strftime("%Y-%m-%d %H:%MZ", time.gmtime())
    started = None
    for family in families:
        for drift in args.drifts:
            cfg = run_dir(args, family, drift) / "run_config.json"
            if cfg.is_file():
                mtime = cfg.stat().st_mtime
                started = mtime if started is None else min(started, mtime)
    eta = ""
    if started is not None and points_done >= 2:
        days = max((time.time() - started) / 86400.0, 1e-6)
        rate = points_done / days
        remaining = total_points - points_done
        eta = f"   ~{remaining / rate:.0f} days left ({rate:.1f} points/day, 3 nodes assumed busy)"
    action_text = {
        "none": "nothing to do",
        "inspect_STUCK_DEAD_families_and_missing_workers": "look at the X rows: Ctrl-C the worker on that node, git pull, run h100 again (partials resume)",
        "Ctrl-C_that_worker__git_pull__relaunch_run_h100__partials_resume": "Ctrl-C the worker on the X node, git pull, run h100 again (partials resume)",
        "Ctrl-C_the_HUNG_family_worker__git_pull__relaunch_run_h100": "Ctrl-C the worker on the X node, git pull, run h100 again (partials resume)",
        "wait_for_next_watchdog_confirmation": "wait: the watchdog is confirming idleness before it restarts the point",
        "inspect_node_telemetry_before_restarting_any_worker": "no telemetry: look at that node before restarting anything",
        "wait_for_automatic_retry": "a failed family retries by itself: wait",
        "restart_missing_workers_after_node_cleanup": "start a worker on each idle node: bash scripts/run_olmo3_rlzero.sh run h100",
        "wait_for_worker_preflight_or_queue_claim": "workers are starting: wait",
        "start_workers": "start the workers: bash scripts/run_olmo3_rlzero.sh run h100",
        "fix_runtime_contract_before_continuing": "config mismatch: do not continue, see the ! contract lines",
    }.get(action, action.replace("_", " "))
    verdict_word = {
        "RUNNING": "RUNNING - all good",
        "DEGRADED": "DEGRADED - something needs a look",
        "HUNG": "HUNG - alive but not working",
        "STOPPED": "STOPPED",
        "RECOVERING": "RECOVERING",
        "COMPLETE": "COMPLETE",
        "INVALID": "INVALID",
        "UNKNOWN": "UNKNOWN",
        "STARTING": "STARTING",
        "INCOMPLETE": "INCOMPLETE",
        "NOT_STARTED": "NOT STARTED",
    }.get(overall, overall)
    worker_ids = ", ".join(sorted(workers)) or "none"
    needs_you = [
        r["family"].key for r in rows
        if r["verdict"] in {"HUNG", "LOOPING"} or (r["verdict"] in {"DEAD", "STOPPED"} and not workers)
    ]
    looping = [r["family"].key for r in rows if r["verdict"] == "LOOPING"]
    queued = [r["family"].key for r in rows if r["verdict"] in {"DEAD", "STOPPED"} and workers]
    idle_workers = {w["worker"] for w in worker_rows if w["state"] == "AVAILABLE"}
    auto = [r["family"].key for r in rows if r["verdict"] in {"STUCK", "RETRYING"}]
    check = [r["family"].key for r in rows if r["verdict"] == "UNKNOWN"]
    errored = [r["family"].key for r in rows if r["current_error_count"]]
    dead_workers = []
    workers_dir = args.root / ".workers"
    if workers_dir.is_dir():
        for entry in sorted(workers_dir.glob("*.json")):
            rec = read_owner(entry)
            beat = record_age_seconds(rec, "heartbeat_at_ns", 1_000_000_000)
            if beat is not None and beat > 86400:
                continue  # a day-old record: already acted on or replaced; not an alarm
            if rec.get("state") in {"launcher-missing", "crashed"} or (rec.get("state") == "running" and beat is not None and beat > args.heartbeat_stale_seconds):
                dead_workers.append(f"{rec.get('worker', entry.stem)} on {rec.get('host', '?')} (last seen {fmt_age(beat) if beat is not None else '?'} ago)")
    if contract_errors:
        decision = "ERROR: config/contract mismatch. Do not restart; fix the ! contract lines first."
    elif dead_workers and len(workers) < args.expected_workers:
        decision = (f"WORKER DEAD: {'; '.join(dead_workers)}. Progress continues on {len(workers)} worker(s). "
                    "Start a worker on that host again: bash scripts/run_olmo3_rlzero.sh run h100")
    elif complete == len(families):
        decision = "DONE: every family is complete."
    elif looping:
        decision = (f"ERROR: {', '.join(looping)} keep(s) failing with the same error; retries were stopped. "
                    "Read the note on that row, fix the cause, then relaunch with OM_RLZERO_CLEAR_LOOPS=1.")
    elif needs_you and workers:
        decision = (f"ERROR: {len(needs_you)} family(ies) hung (alive, writing nothing): {', '.join(needs_you)}. "
                    "RESTART NEEDED on the node showing that family: Ctrl-C, git pull --ff-only, run h100 (finished work resumes).")
    elif needs_you:
        decision = (f"ERROR: {len(needs_you)} family(ies) stopped and no worker is running: {', '.join(needs_you)}. "
                    "START workers: bash scripts/run_olmo3_rlzero.sh run h100 on each node (finished work resumes).")
    elif missing_workers and len(workers) == 0:
        decision = "ERROR: no worker is running anywhere. Start one per node: bash scripts/run_olmo3_rlzero.sh run h100"
    elif missing_workers:
        decision = f"WARNING: only {len(workers)}/{args.expected_workers} workers. Progress continues but slower; start a worker on the idle node(s)."
    elif auto or check:
        parts = []
        if auto:
            parts.append(f"{', '.join(auto)} recovering by itself")
        if check:
            parts.append(f"{', '.join(check)} unknown (no telemetry)")
        decision = "NO ACTION NOW: " + "; ".join(parts) + ". Check again in 30 min; if the same rows are still not * then, restart that worker."
    elif errored:
        decision = f"NO ERROR blocking: {', '.join(errored)} logged an error but the current attempt is progressing. Nothing to do."
    elif queued and idle_workers:
        decision = (f"ERROR: {', '.join(queued)} unclaimed while worker(s) {', '.join(sorted(idle_workers))} sit idle: "
                    "a stale family lock or a worker stuck in preflight. Read that worker's last log line below.")
    elif queued and len(workers) >= args.expected_workers:
        decision = f"NO ERROR. Workers busy; {', '.join(queued)} waiting in the queue. Nothing to do."
    else:
        decision = "NO ERROR. Everything is progressing. Nothing to do."
    print(f"OLMo-3 RL-Zero {args.profile}   {now}   code {generation_git[:8]}   probe {args.probe_seconds}s")
    print(f"DECISION {decision}")
    print(f"STATE   {verdict_word}")
    print(f"        workers {len(workers)}/{args.expected_workers} ({worker_ids})   families {complete}/{len(families)} done   points {points_done}/{total_points} done{eta}")
    print(f"ACTION  {action_text}")
    if contract_errors:
        for issue in contract_errors[:6]:
            print(f"  ! contract: {issue}")
    print()

    problem = {"HUNG", "STUCK", "DEAD", "STOPPED", "LOOPING"}
    moving = {"PROGRESSING", "COMPUTING", "ALIVE", "IDLE", "RETRYING"}

    def glyphs(row: dict) -> str:
        out = []
        for drift in args.drifts:
            if drift in row["done_drifts"]:
                out.append("+")
            elif drift == row["drift"] and row["kind"] == "active":
                if row["verdict"] in problem:
                    out.append("X")
                elif row["verdict"] == "UNKNOWN":
                    out.append("?")
                else:
                    out.append("*")
            else:
                out.append(".")
        return "".join(out)

    def rank(row: dict) -> int:
        if row["verdict"] in problem:
            return 0
        if row["verdict"] == "UNKNOWN":
            return 1
        if row["verdict"] in moving:
            return 2
        if row["verdict"] == "COMPLETE":
            return 3
        return 4

    ordered = sorted(rows, key=lambda r: (rank(r), r["family"].dataset, r["family"].seed))
    shown = [r for r in ordered if r["verdict"] != "PENDING"]
    waiting = [r["family"].key for r in ordered if r["verdict"] == "PENDING"]
    print(" points column, one char per point d0 d25 d100 d400:   + = done   * = running   X = hung/stuck/dead   ? = unknown   . = waiting")
    header = f" {'family':<11} {'points':<{len(args.drifts) + 1}} {'now':<40} {'last write':<10} {'worker':<12} note"
    print(header)
    for row in shown:
        if row["verdict"] == "COMPLETE":
            now_text = "done"
        elif row["kind"] == "active" and row["drift"] is not None:
            bits = [f"d{row['drift']}", row["stage"]]
            if row["grpo"] != "-":
                bits.append(f"step {row['grpo']}")
            if row["verdict"] in problem or row["verdict"] == "UNKNOWN":
                bits.append(row["verdict"])
            now_text = " ".join(b for b in bits if b and b != "-")
        else:
            now_text = row["stage"] if row["stage"] != "-" else row["verdict"].lower()
        print(
            f" {row['family'].key:<11} {glyphs(row):<{len(args.drifts) + 1}} {now_text[:40]:<40} "
            f"{fmt_age(row['age']):<10} {row['worker'][:12]:<12} {row['note']}"
        )
    if waiting:
        print(f" waiting     {'.' * len(args.drifts):<{len(args.drifts) + 1}} {', '.join(waiting)}")
    print()
    alerts = args.root / "logs" / "ALERTS.log"
    if alerts.is_file() and alerts.stat().st_size:
        print()
        print(" recent alerts (logs/ALERTS.log):")
        for line in tail_lines(alerts, 3):
            print(f"  {line[:150]}")
    if worker_rows:
        print()
        print(" worker            log age  claims          last log line")
        for w in worker_rows:
            claims = ",".join(w["claims"]) or "-"
            print(f" {w['worker'][:17]:<17} {fmt_age(w['log_age']):<8} {claims[:15]:<15} {w['last_line'][:95]}")
    print(" family = one dataset x seed = 4 chained points d0 -> d25 -> d100 -> d400 on one node (each GRPO point resumes the previous checkpoint)")
    print(" last write = time since this family wrote any file.  note: NEEDS YOU = you act, AUTO = supervisor handles it, QUEUED = waits for a free worker, ok = fine")
    stale_workers = [w["worker"] for w in worker_rows if w["state"] == "STALE"]
    if stale_workers:
        print(f" stale worker records (no claim, no fresh log): {', '.join(stale_workers)}")
    report = args.results / "FINAL_REPORT.md"
    if report.is_file() and report.stat().st_size:
        print(f" report  {report}")
    print(f" logs    {args.root / 'logs'}      detail: status {args.profile} verbose")
    if not args.verbose:
        print(f"overall_verdict={overall}")
        print(f"recommended_action={action}")
        return

    # ---- verbose: the full evidence dump ----
    print()
    print(f"profile={args.profile}")
    print(f"experiment_root={args.root}")
    print(f"status_probe_seconds={args.probe_seconds}")
    print(f"stuck_after_seconds={args.stuck_seconds}")
    print(f"heartbeat_stale_seconds={args.heartbeat_stale_seconds}")
    print(f"log_tail_lines={args.log_lines}")
    print(
        f"runtime_contract generation_batch={args.generation_batch} "
        f"gradient_micro_batch={args.gradient_micro_batch} "
        f"logprob_micro_batch={args.logprob_micro_batch} "
        f"min_recovery_generation_batch={args.min_recovery_generation_batch}"
    )
    print("generation_git=" + generation_git)
    print("== worker diagnostics ==")
    for w in worker_rows:
        claims = ",".join(w["claims"]) or "none"
        print(
            f"worker={w['worker']} state={w['state']} claims={claims} "
            f"log_age={'none' if w['log_age'] is None else f'{w["log_age"]}s'} "
            f"heartbeat_age={'none' if w['heartbeat_age'] is None else f'{w["heartbeat_age"]}s'} "
            f"liveness_evidence={w['evidence']} "
            f"error_matches={w['errors']}"
        )
        if w["last_line"]:
            print(f"  last_log_line={w['last_line']}")
    if not workers:
        print("worker=none state=NOT_OBSERVED")
    for row in rows:
        family = row["family"]
        snapshot = row["snapshot"]
        verdict, reason, changes = row["verdict"], row["reason"], row["changes"]
        suffix = f" {owner_display(snapshot.owner)}" if snapshot.owner else ""
        print(f"{family.key} {snapshot.state}{suffix}")
        age = row["age"]
        age_text = "none" if age is None else f"{age}s"
        error_count, errors = row["error_count"], row["errors"]
        current_error_count, current_errors = row["current_error_count"], row["current_errors"]
        boundary, attempt_manifest = row["boundary"], row["attempt_manifest"]
        if current_error_count and verdict in {"PROGRESSING", "COMPUTING", "ALIVE", "COMPLETE"}:
            error_assessment = "current_attempt_errors_present_but_activity_continues"
        elif current_error_count:
            error_assessment = "current_attempt_error_evidence_present"
        elif error_count == 0:
            error_assessment = "none"
        elif boundary:
            error_assessment = "historical_only_not_current_attempt"
        elif verdict in {"PROGRESSING", "COMPUTING", "ALIVE", "COMPLETE"}:
            error_assessment = "history_present_but_not_blocking_current_progress"
        else:
            error_assessment = "attempt_boundary_unavailable_history_not_attributed"
        print(
            f"  verdict={verdict} reason={reason} activity_age={age_text} "
            f"logs_checked={len(row['checked_logs'])} error_matches={error_count} "
            f"current_attempt_error_matches={current_error_count} "
            f"error_assessment={error_assessment}"
        )
        if attempt_manifest is not None:
            print(f"  current_attempt_boundary={attempt_manifest}")
        if changes:
            print("  observed_changes=" + ", ".join(changes[:8]))
        if snapshot.pipeline_activity is not None:
            telemetry_age = record_age_seconds(snapshot.pipeline_activity, "observed_at_epoch")
            print(
                f"  pipeline_telemetry={snapshot.pipeline_activity_path} "
                f"state={snapshot.pipeline_activity.get('state', 'invalid')} "
                f"age={'none' if telemetry_age is None else f'{telemetry_age}s'} "
                f"cpu_delta={snapshot.pipeline_activity.get('cpu_delta_seconds', 'unknown')} "
                f"gpu_peak={snapshot.pipeline_activity.get('gpu_peak_percent', 'unknown')} "
                f"idle={snapshot.pipeline_activity.get('idle_seconds', 'unknown')}s"
            )
        for point in row["points"]:
            print(point)
        if snapshot.state not in {"complete", "pending"}:
            family_logs, owner_log = row["family_logs"], row["owner_log"]
            latest = family_logs[0] if family_logs else None
            if latest is not None:
                log_age = age_seconds(latest.stat().st_mtime_ns)
                print(f"  latest_log={latest} age={log_age}s (last {args.log_lines} lines)")
                for line in tail_lines(latest, args.log_lines):
                    print(f"    | {line}")
            if errors:
                print("  error_evidence_from_all_checked_logs:")
                for path, line in errors:
                    print(f"    ! {path}: {line}")
            if current_errors:
                print("  current_attempt_error_evidence:")
                for path, line in current_errors:
                    print(f"    ! {path}: {line}")
            if owner_log is not None and owner_log != latest:
                log_age = age_seconds(owner_log.stat().st_mtime_ns)
                print(f"  worker_log={owner_log} age={log_age}s (last {args.log_lines} lines)")
                for line in tail_lines(owner_log, args.log_lines):
                    print(f"    | {line}")
    print("== diagnosis ==")
    print(
        f"workers_observed={len(workers)}/{args.expected_workers} "
        f"worker_ids={','.join(sorted(workers)) or 'none'}"
    )
    print(
        f"families_total={len(families)} complete={complete} progressing={progressing} "
        f"computing={computing} alive={alive} idle={idle} unknown={unknown} "
        f"retrying={retrying} stuck={stuck} dead={dead} stopped={stopped} "
        f"pending={pending}"
    )
    print(f"runtime_contract_errors={len(contract_errors)}")
    for issue in contract_errors[:20]:
        print(f"  ! {issue}")
    print(f"overall_verdict={overall}")
    print(f"recommended_action={action}")
    if report.is_file() and report.stat().st_size:
        print(f"report={report}")


if __name__ == "__main__":
    main()
