"""Choose a bounded rollout recovery batch from current-attempt evidence."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

OOM_PATTERN = re.compile(r"CUDA out of memory|(?:torch\.)?OutOfMemoryError", re.IGNORECASE)
RUNTIME_PATTERN = re.compile(
    r"CUDA error|CUBLAS_STATUS|cuBLAS|device-side assert|"
    r"unspecified launch failure|illegal memory access",
    re.IGNORECASE,
)
ATTEMPT_MANIFEST_SCHEMA = "offpolicy-pipeline-attempt/v1"
STAGE_LOG_PATTERNS = {
    "rollout-behavior": ("beta-shard", "rollout-behavior"),
    "rollout-fresh": ("fresh-shard", "rollout-fresh"),
}


def classify_cuda_failure(text: str) -> str | None:
    if OOM_PATTERN.search(text):
        return "oom"
    if RUNTIME_PATTERN.search(text):
        return "runtime"
    return None


def attempt_log_offsets(path: Path) -> dict[str, int]:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid attempt manifest: {path}") from exc
    if record.get("schema") != ATTEMPT_MANIFEST_SCHEMA:
        raise ValueError(f"invalid attempt manifest schema: {path}")
    offsets = record.get("log_offsets")
    if not isinstance(offsets, dict):
        raise TypeError(f"attempt manifest has no log_offsets: {path}")
    parsed: dict[str, int] = {}
    for name, value in offsets.items():
        try:
            offset = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid log offset for {name!r}") from exc
        if not isinstance(name, str) or not name or offset < 0:
            raise ValueError(f"invalid log offset for {name!r}")
        parsed[name] = offset
    return parsed


def current_attempt_evidence(
    attempt_log: Path,
    attempt_manifest: Path,
    logs_root: Path,
    stage: str,
) -> str:
    """Read only bytes written by this attempt for the failed rollout stage."""
    patterns = STAGE_LOG_PATTERNS.get(stage)
    if patterns is None:
        raise ValueError(f"unsupported rollout stage: {stage!r}")
    offsets = attempt_log_offsets(attempt_manifest)
    candidates = [attempt_log]
    if logs_root.is_dir():
        candidates.extend(
            path
            for path in logs_root.rglob("*.log")
            if path != attempt_log and any(pattern in path.name for pattern in patterns)
        )
    chunks: list[str] = []
    for path in candidates:
        try:
            relative = path.relative_to(logs_root).as_posix()
            size = path.stat().st_size
            offset = 0 if path == attempt_log else offsets.get(relative, 0)
            # A truncated log contains only current-attempt data.
            if size < offset:
                offset = 0
            if size == offset:
                continue
            with path.open("rb") as stream:
                stream.seek(offset)
                chunks.append(stream.read().decode("utf-8", errors="replace"))
        except (OSError, ValueError):
            continue
    return "\n".join(chunks)


def configured_generation_batch(config: dict) -> int:
    try:
        batch = int(config["gen_batch"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("run_config.json has no positive gen_batch") from exc
    if batch <= 0:
        raise ValueError("run_config.json has no positive gen_batch")
    return batch


def failed_oom_batches(path: Path, stage: str) -> list[int]:
    if not path.is_file():
        return []
    batches: list[int] = []
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            try:
                record = json.loads(line)
                if (
                    record.get("status") == "failed"
                    and record.get("stage") == stage
                    and record.get("failure_kind") == "oom"
                ):
                    batches.append(int(record["recovery_generation_batch"]))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
    return [batch for batch in batches if batch > 0]


def select_recovery_batch(
    kind: str,
    configured: int,
    minimum: int,
    failed_batches: list[int] | tuple[int, ...] = (),
) -> int:
    if configured <= 0 or minimum <= 0:
        raise ValueError("generation batches must be positive")
    if minimum > configured:
        raise ValueError("minimum recovery batch cannot exceed configured batch")
    if kind == "oom":
        previous = [batch for batch in failed_batches if minimum <= batch <= configured]
        smallest_failed = min(previous, default=configured)
        if smallest_failed <= minimum:
            raise ValueError(
                f"OOM recovery exhausted at minimum batch {minimum}; refusing batch 1"
            )
        return max(minimum, smallest_failed // 2)
    if kind == "runtime":
        return configured
    raise ValueError(f"unsupported CUDA failure kind: {kind!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--attempt-log", type=Path, required=True)
    parser.add_argument("--attempt-manifest", type=Path, required=True)
    parser.add_argument("--logs-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("rollout-behavior", "rollout-fresh"), required=True)
    parser.add_argument("--recovery-log", type=Path, required=True)
    parser.add_argument("--min-oom-batch", type=int, required=True)
    args = parser.parse_args()

    try:
        evidence = current_attempt_evidence(
            args.attempt_log,
            args.attempt_manifest,
            args.logs_root,
            args.stage,
        )
    except (TypeError, ValueError) as exc:
        print(f"[recovery-abort] {exc}", file=sys.stderr)
        return 2
    kind = classify_cuda_failure(evidence)
    if kind is None:
        return 1
    config = json.loads(args.run_config.read_text(encoding="utf-8"))
    configured = configured_generation_batch(config)
    try:
        selected = select_recovery_batch(
            kind,
            configured,
            args.min_oom_batch,
            failed_oom_batches(args.recovery_log, args.stage),
        )
    except ValueError as exc:
        print(f"[recovery-abort] {exc}", file=sys.stderr)
        return 2
    print(kind, selected, configured)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
