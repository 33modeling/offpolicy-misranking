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
    expected_stamp = expected_family_stamp(args, family)
    if complete_stamp.is_file() and points_complete and expected_stamp is not None:
        try:
            if complete_stamp.read_text(encoding="utf-8").strip() == expected_stamp:
                return "complete", {}
        except OSError:
            pass
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
    telemetry = after.pipeline_activity
    telemetry_age = record_age_seconds(telemetry, "observed_at_epoch")
    if telemetry is not None and telemetry_age is not None:
        telemetry_state = str(telemetry.get("state", "invalid"))
        if telemetry_age <= telemetry_stale_seconds:
            if telemetry.get("schema") != "offpolicy-pipeline-activity/v1":
                return "UNKNOWN", "pipeline_telemetry_schema_invalid"
            if telemetry_state in {"output-progress", "cpu-active", "gpu-active"}:
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
    print(f"heartbeat_stale_seconds={args.heartbeat_stale_seconds}")
    print(f"log_tail_lines={args.log_lines}")
    print(
        f"runtime_contract generation_batch={args.generation_batch} "
        f"gradient_micro_batch={args.gradient_micro_batch} "
        f"logprob_micro_batch={args.logprob_micro_batch} "
        f"min_recovery_generation_batch={args.min_recovery_generation_batch}"
    )
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
    print("== worker diagnostics ==")
    diagnostic_workers = workers | heartbeat_workers | set(claims_by_worker)
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
        claims = ",".join(claims_by_worker.get(worker, [])) or "none"
        last_line = last_nonempty_line(log) if log.is_file() else ""
        heartbeat, heartbeat_age, heartbeat_fresh = worker_heartbeat(args, worker)
        if heartbeat_fresh:
            evidence = "heartbeat"
        elif log_age is not None and log_age <= args.worker_stale_seconds:
            evidence = "recent-log"
        elif heartbeat is not None:
            evidence = "stale-heartbeat"
        else:
            evidence = "lock-only"
        heartbeat_age_text = "none" if heartbeat_age is None else f"{heartbeat_age}s"
        print(
            f"worker={worker} state={state} claims={claims} "
            f"log_age={'none' if log_age is None else f'{log_age}s'} "
            f"heartbeat_age={heartbeat_age_text} "
            f"liveness_evidence={evidence} "
            f"error_matches={worker_errors}"
        )
        if last_line:
            print(f"  last_log_line={last_line}")
    if not workers:
        print("worker=none state=NOT_OBSERVED")

    verdict_counts: dict[str, int] = {}
    contract_errors: list[str] = []
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
        boundary, current_error_count, current_errors, attempt_manifest = (
            current_attempt_errors(family_root(args, family), args.error_lines)
        )
        if current_error_count and verdict in {
            "PROGRESSING",
            "COMPUTING",
            "ALIVE",
            "COMPLETE",
        }:
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
            f"logs_checked={len(checked_logs)} error_matches={error_count} "
            f"current_attempt_error_matches={current_error_count} "
            f"error_assessment={error_assessment}"
        )
        if attempt_manifest is not None:
            print(f"  current_attempt_boundary={attempt_manifest}")
        if changes:
            print("  observed_changes=" + ", ".join(changes[:8]))
        if snapshot.pipeline_activity is not None:
            telemetry_age = record_age_seconds(
                snapshot.pipeline_activity, "observed_at_epoch"
            )
            print(
                f"  pipeline_telemetry={snapshot.pipeline_activity_path} "
                f"state={snapshot.pipeline_activity.get('state', 'invalid')} "
                f"age={'none' if telemetry_age is None else f'{telemetry_age}s'} "
                f"cpu_delta={snapshot.pipeline_activity.get('cpu_delta_seconds', 'unknown')} "
                f"gpu_peak={snapshot.pipeline_activity.get('gpu_peak_percent', 'unknown')} "
                f"idle={snapshot.pipeline_activity.get('idle_seconds', 'unknown')}s"
            )
        for drift in args.drifts:
            point, issues = point_status(args, family, drift)
            print(point)
            contract_errors.extend(f"{family.key}/d{drift}:{issue}" for issue in issues)
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
            if current_errors:
                print("  current_attempt_error_evidence:")
                for path, line in current_errors:
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
    computing = verdict_counts.get("COMPUTING", 0)
    alive = verdict_counts.get("ALIVE", 0)
    idle = verdict_counts.get("IDLE", 0)
    unknown = verdict_counts.get("UNKNOWN", 0)
    stuck = verdict_counts.get("STUCK", 0)
    dead = verdict_counts.get("DEAD", 0)
    stopped = verdict_counts.get("STOPPED", 0)
    pending = verdict_counts.get("PENDING", 0)
    retrying = verdict_counts.get("RETRYING", 0)
    missing_workers = len(workers) < args.expected_workers
    degraded = stuck + dead + stopped + idle + unknown > 0 or missing_workers
    if contract_errors:
        overall = "INVALID"
        action = "fix_runtime_contract_before_continuing"
    elif complete == len(families):
        overall = "COMPLETE"
        action = "none"
    elif progressing or computing or alive or idle:
        overall = "DEGRADED" if degraded else "RUNNING"
        if stuck + dead + stopped > 0 or missing_workers:
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
    report = args.results / "FINAL_REPORT.md"
    if report.is_file() and report.stat().st_size:
        print(f"report={report}")


if __name__ == "__main__":
    main()
