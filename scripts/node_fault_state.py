"""Read a GPU-fault receipt without renewing its cooldown or discarding evidence."""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path


def inspect(path: Path, ttl: float, *, now: float | None = None) -> tuple[str, int, str]:
    now = time.time() if now is None else now
    try:
        if not math.isfinite(ttl) or ttl <= 0:
            raise ValueError("fault TTL must be finite and positive")
        try:
            stamp = path.stat().st_mtime
        except FileNotFoundError:
            return "ready", 0, "no GPU-fault record"
        record = json.loads(path.read_text())
        if not isinstance(record, dict) or not record:
            raise ValueError("fault record must be a nonempty object")
        strikes = record.get("strikes", 1)
        if type(strikes) is not int or strikes < 1:
            raise ValueError("invalid strike count")
        if strikes >= 2:
            return "blocked", 0, f"strike {strikes}; repeated GPU faults"
        # Legacy receipts lacked `time`. Their mtime is stable; substituting
        # time.time() on every read renewed their cooldown forever.
        recorded = float(record.get("time", stamp))
        if not math.isfinite(recorded) or recorded <= 0 or recorded > now + 300:
            raise ValueError("invalid or future fault timestamp")
        remaining = math.ceil(max(0., min(ttl, recorded + ttl - now)))
        if remaining:
            return "cooldown", remaining, f"strike {strikes}; recheck in {remaining}s"
        return "expired", 0, f"strike {strikes}; cooldown expired, admission probe required"
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        # A broken receipt is a visible admission failure, never an endless
        # fresh cooldown and never permission to bypass the GPU health probe.
        return "blocked", 0, f"invalid GPU-fault record: {exc}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("--remaining", action="store_true")
    parser.add_argument("--ttl", type=float, default=os.environ.get(
        "EXPERIMENTS_FAULT_TTL_SECONDS", "60" if os.environ.get("EXPERIMENTS_MBPP_SUITE") else "1800"))
    args = parser.parse_args()
    state, remaining, reason = inspect(args.record, args.ttl)
    if args.remaining:
        print(remaining)
        return 0
    if state != "ready":
        label = "fault-expired" if state == "expired" else state
        print(f"[{label}] host={os.environ.get('EXPERIMENTS_NODE_ID', 'unknown')}: "
              f"{reason} ({args.record}; receipt retained)", flush=True)
    return {"blocked": 78, "cooldown": 79}.get(state, 0)


if __name__ == "__main__":
    raise SystemExit(main())
