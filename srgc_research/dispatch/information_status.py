"""Audit all initial information measurements without GPU work or file writes."""

import argparse
import fcntl
import json
import os
from collections import Counter
from pathlib import Path

from srgc_research.information_cli import status
from srgc_research.information_report import PHASES, read_object

from .information_queue import SEEDS


def owned(paths):
    for path in paths:
        try:
            fd = os.open(path, os.O_RDONLY)
        except FileNotFoundError:
            continue
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
    return False


def inspect(root, dataset, seed):
    key = f"{dataset}.seed-{seed}.t0"
    output = root / dataset / f"seed-{seed}" / "t0"
    receipt = root / ".queue" / f"{key}.json"
    row = {"task": key, "dataset": dataset, "seed": seed, "stage": 0,
           "state": "ready", "phases": 0, "host": "-", "error": None}
    try:
        if (output / "endpoint.json").exists() and not (output / "manifest.json").exists():
            raise ValueError("endpoint has no measurement manifest")
        measured = status(output)
        identity = measured.get("identity")
        if identity is not None and any(identity.get(k) != v for k, v in
                                        (("dataset", dataset), ("seed", seed), ("stage", 0))):
            raise ValueError("measurement identity differs from its dataset/seed directory")
        row["phases"] = sum(value == "saved" for value in measured["phases"].values())
        saved = read_object(receipt) if receipt.is_file() else {}
        if saved and saved.get("task") != key:
            raise ValueError("queue receipt belongs to a different measurement")
        row["host"] = saved.get("host", "-")
        if measured["status"] == "complete":
            row["state"] = "complete"
        elif owned((receipt.with_suffix(".lock"), output / ".dispatch.lock", output / ".execution.lock")):
            row["state"] = "running"
        elif saved.get("status") == "complete":
            raise ValueError("completed queue receipt has no verified endpoint")
        elif saved.get("status") == "failed":
            row.update(state="failed", error=saved.get("error") or f"collector exit={saved.get('exit_code', '?')}")
        elif saved.get("status") in {"running", "interrupted"} or measured["status"] == "partial":
            row["state"] = "resume"
    except (OSError, ValueError, KeyError, TypeError) as error:
        row.update(state="invalid", error=str(error))
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("math", "mbpp", "all"))
    parser.add_argument("action", choices=("status",))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    work = Path(os.environ.get("OM_WORK", str(Path(os.environ.get("GROUP_VOLUME", "/group-volume")) /
        os.environ.get("OM_USER", "minsoo3.kim") / "offpolicy-misranking")))
    root = work / "selection-information"
    names = ("math", "mbpp") if args.dataset == "all" else (args.dataset,)
    rows = [inspect(root, name, seed) for name in names for seed in SEEDS]
    counts = dict(Counter(row["state"] for row in rows))
    complete = counts.get("complete", 0) == len(rows)
    result = {"complete": complete, "counts": counts, "rows": rows, "output": str(root)}
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"INFORMATION {counts.get('complete', 0)}/{len(rows)} complete | "
              f"RUN {counts.get('running', 0)} | RESUME {counts.get('resume', 0)} | "
              f"READY {counts.get('ready', 0)} | FAILED {counts.get('failed', 0)} | INVALID {counts.get('invalid', 0)}")
        print(f"{'task':22} {'state':10} {'phases':8} host")
        for row in rows:
            print(f"{row['task']:22} {row['state']:10} {row['phases']}/{len(PHASES):<6} {row['host']}")
            if row["error"]:
                print(f"  {row['error']}")
        print("COMPLETE: all requested information measurements" if complete else "INCOMPLETE: measurements remain")
    return int(bool(counts.get("invalid") or counts.get("failed")))


if __name__ == "__main__":
    raise SystemExit(main())
