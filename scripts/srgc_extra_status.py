#!/usr/bin/env python3
"""Read extra-arm progress without claiming jobs, touching locks or loading GPUs."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.srgc_replicate_worker import SCOPES, signature, tasks_for
from srgc_rebuttal.plan import load_plan


def read_object(path):
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def inspect(task, *, now=None):
    now = time.time() if now is None else now
    plan = load_plan(task.plan)
    out = task.out
    row = dict(task=task.key, dataset=task.dataset, seed=task.seed, arm=task.name,
               plan=str(task.plan), output=str(out), state="waiting", step=None,
               total=plan["total_updates"], attempt=0, host="-", age_seconds=None, log=None)
    mechanism = task.arm == "stage_mechanism"
    if mechanism:
        from scripts.srgc_stage_mechanism import TOTAL_WORK
        row["total"] = TOTAL_WORK
    try:
        queue = read_object(task.receipt)
        if type(queue.get("attempt", 0)) is not int or queue.get("attempt", 0) < 0:
            raise ValueError(f"{task.receipt}: invalid attempt count")
        progress_path = out / f"{task.arm}-progress.json"
        progress = read_object(progress_path)
        if progress:
            if progress.get("seed") != task.seed or progress.get("arm") != task.arm:
                raise ValueError(f"{progress_path}: seed or arm mismatch")
            step = progress.get("step")
            minimum = 0 if mechanism else plan["shared_prefix_updates"]
            if type(step) is not int or not minimum <= step <= row["total"]:
                raise ValueError(f"{progress_path}: invalid completed step")
            row.update(step=step, age_seconds=max(0, int(now - progress_path.stat().st_mtime)))
            if mechanism:
                row.update(stage=progress.get("stage"), mode=progress.get("mode"),
                           branch_updates=progress.get("branch_updates"), phase=progress.get("phase"))
        worker_paths = list((out / "launches" / task.arm).glob("*/worker.json"))
        latest = max(worker_paths, key=lambda p: p.stat().st_mtime) if worker_paths else None
        worker = read_object(latest) if latest else {}
        row.update(attempt=queue.get("attempt", 0), host=worker.get("host", queue.get("host", "-")),
                   log=worker.get("log", str(latest.parent / "task.log") if latest else None))
        endpoint = (out / f"{task.arm}-endpoint.json").exists()
        fingerprint = signature(task) if endpoint else None
        if (queue.get("status") == "complete" and fingerprint is not None
                and queue.get("verified_files") == fingerprint):
            row.update(state="complete", step=row["total"])
        elif endpoint:
            row["state"] = "endpoint_unverified"
        else:
            stamp = worker.get("heartbeat", worker.get("finished", worker.get("started", 0)))
            if worker and stamp >= queue.get("started", 0):
                state = worker.get("status")
                if state in ("running", "preflight"):
                    row["state"] = f"reported_{state}" if now - stamp <= 180 else "stale_worker"
                else:
                    row["state"] = "failed" if state == "failed" else "interrupted"
            elif queue.get("status") in ("failed", "interrupted"):
                row["state"] = queue["status"]
            elif queue.get("status") == "running":
                row["state"] = "claimed_unconfirmed"
            elif progress:
                row["state"] = "checkpoint_saved"
            elif not all((task.folder / p).is_file() for p in ("prefix.pt", "prefix-ready.json")):
                row["state"] = "waiting_prefix"
    except (OSError, ValueError, TypeError, KeyError) as exc:
        row.update(state="invalid", error=str(exc))
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("all", "math", "mbpp"), required=True)
    parser.add_argument("--scope", choices=SCOPES, default="all")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    rows, errors = [], []
    for dataset in (("math", "mbpp") if args.dataset == "all" else (args.dataset,)):
        try:
            for task in tasks_for(dataset, args.scope):
                rows.append(inspect(task))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            errors.append(f"{dataset}: {exc}")
    counts = dict(Counter(row["state"] for row in rows))
    if args.json:
        print(json.dumps(dict(scope=args.scope, counts=counts, rows=rows, errors=errors), indent=2))
    else:
        print(f"EXTRAS status scope={args.scope} (read-only; worker states are reported, not live lock checks)")
        print(f"{'dataset':7} {'seed':>4} {'arm':28} {'state':23} {'step':>7} {'try':>3} {'age(s)':>7} host")
        for row in rows:
            step = f"{row['step']}/{row['total']}" if row['step'] is not None else "-"
            age = str(row['age_seconds']) if row['age_seconds'] is not None else "-"
            print(f"{row['dataset']:7} {row['seed']:4} {row['arm']:28} {row['state']:23} "
                  f"{step:>7} {row['attempt']:>3} {age:>7} {row['host']}")
            if row.get("error"):
                print(f"  ERROR: {row['error']}")
            if row["arm"] == "stage_mechanism":
                print(f"  physical work updates; policy stage={row.get('stage')} mode={row.get('mode')} "
                      f"branch_updates={row.get('branch_updates')} phase={row.get('phase')}")
            if row.get("log") and row['state'] in ("failed", "invalid", "stale_worker"):
                print(f"  log: {row['log']}")
        print("TOTAL " + " ".join(f"{key}={value}" for key, value in sorted(counts.items())))
        print("endpoint_unverified = result file present, not validated by this read-only command")
        print("age(s) = time since the last saved progress; '-' = no saved progress")
        for error in errors:
            print(f"ERROR: {error}")
    return 1 if errors or any(row["state"] == "invalid" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
