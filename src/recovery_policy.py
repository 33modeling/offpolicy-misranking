"""Choose a rollout recovery batch from the current failure, not stale logs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

OOM_PATTERN = re.compile(r"CUDA out of memory", re.IGNORECASE)
RUNTIME_PATTERN = re.compile(
    r"CUDA error|CUBLAS_STATUS|cuBLAS|device-side assert|"
    r"unspecified launch failure|illegal memory access",
    re.IGNORECASE,
)


def classify_cuda_failure(text: str) -> str | None:
    if OOM_PATTERN.search(text):
        return "oom"
    if RUNTIME_PATTERN.search(text):
        return "runtime"
    return None


def configured_generation_batch(config: dict) -> int:
    try:
        batch = int(config["gen_batch"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("run_config.json has no positive gen_batch") from exc
    if batch <= 0:
        raise ValueError("run_config.json has no positive gen_batch")
    return batch


def select_recovery_batch(kind: str, configured: int, oom_batch: int) -> int:
    if configured <= 0 or oom_batch <= 0:
        raise ValueError("generation batches must be positive")
    if kind == "oom":
        return min(configured, oom_batch)
    if kind == "runtime":
        return configured
    raise ValueError(f"unsupported CUDA failure kind: {kind!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--attempt-log", type=Path, required=True)
    parser.add_argument("--oom-batch", type=int, required=True)
    args = parser.parse_args()

    kind = classify_cuda_failure(
        args.attempt_log.read_text(encoding="utf-8", errors="replace")
    )
    if kind is None:
        return 1
    config = json.loads(args.run_config.read_text(encoding="utf-8"))
    configured = configured_generation_batch(config)
    selected = select_recovery_batch(kind, configured, args.oom_batch)
    print(kind, selected, configured)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
