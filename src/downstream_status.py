"""Detailed E5 progress without changing the frozen experiment implementation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def _tail(path: Path, width: int = 110) -> str:
    try:
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 8192))
            lines = [line for line in stream.read().decode(errors="replace").splitlines() if line.strip()]
    except OSError:
        return ""
    return lines[-1][-width:] if lines else ""


def _age(path: Path) -> str:
    try:
        seconds = max(0, int(time.time() - path.stat().st_mtime))
    except OSError:
        return "-"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m" if seconds >= 3600 else f"{seconds // 60}m{seconds % 60:02d}s"


def row_count(path: Path) -> int:
    with path.open() as stream:
        return sum(1 for line in stream if line.strip())


def shard_states(out: Path, arm: str) -> list[str]:
    """One line per evaluation shard with its response count and log tail."""
    contract = read(out / "experiment.json")
    n, k = contract["eval_prompts"], contract["eval_k"]
    lines = []
    for shard in range(4):
        expected = (n * (shard + 1) // 4 - n * shard // 4) * k
        target = out / arm / "evaluation"
        log = out / "logs" / f"eval-{arm}-{shard}.log"
        if (target / f"shard-{shard}.done.json").is_file():
            lines.append(f"    shard {shard}: done ({expected} responses)")
            continue
        partial = target / f"shard-{shard}.jsonl.partial"
        if partial.is_file():
            lines.append(f"    shard {shard}: {row_count(partial)}/{expected} responses, last write {_age(partial)} ago | {_tail(log)}")
        elif log.is_file():
            lines.append(f"    shard {shard}: started, no responses yet, log {_age(log)} old | {_tail(log)}")
        else:
            lines.append(f"    shard {shard}: not started")
    return lines


def train_state(out: Path, arm: str) -> str:
    contract = read(out / "experiment.json")
    policy = out / arm / "policy"
    log = out / "logs" / f"train-{arm}.log"
    if (policy / "policy_train.json").is_file():
        return "trained"
    steps = [int(path.name.split("-")[1]) for path in policy.glob("checkpoint-*") if path.name.split("-")[1].isdigit()]
    stats = policy / "grpo_stats.jsonl"
    done = row_count(stats) if stats.is_file() else 0
    if not policy.is_dir() and not log.is_file():
        return "not started"
    return (f"training: {done}/{contract['steps']} updates logged, checkpoint at step "
            f"{max(steps) if steps else contract['drift']}, log {_age(log)} old | {_tail(log)}")


def arm_state(out: Path, arm: str) -> str:
    """Display artifact progress; full result validation remains separate."""
    done = sum((out / arm / "evaluation" / f"shard-{s}.done.json").is_file() for s in range(4))
    if done == 4:
        return "done"
    if done or any((out / arm / "evaluation" / f"shard-{s}.jsonl.partial").is_file() for s in range(4)):
        return f"evaluating ({done}/4 shards done)"
    if arm == "before":
        return "not evaluated"
    state = train_state(out, arm)
    return "trained, not evaluated" if state == "trained" else state


def print_status(out: Path) -> None:
    contract = read(out / "experiment.json")
    print(f"seed {contract['seed']} d{contract['drift']} steps={contract['steps']} eval_k={contract['eval_k']} test={contract['eval_prompts']}")
    for arm in ["before", *contract["selectors"]]:
        state = arm_state(out, arm)
        print(f"  {arm:14s} {state}")
        if state.startswith("evaluating") or (arm == "before" and state == "not evaluated" and (out / "logs").is_dir()
                                                and any((out / "logs" / f"eval-before-{s}.log").is_file() for s in range(4))):
            for line in shard_states(out, arm):
                print(line)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    out = parser.parse_args().out.resolve()
    if not (out / "experiment.json").is_file():
        print("not prepared")
        return 0
    try:
        print_status(out)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"[status-error] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
