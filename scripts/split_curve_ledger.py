#!/usr/bin/env python3
"""Move a published branch's curve-evaluation cost rows out of its sealed ledger.

Before curve-ledger-runtime, curve evaluations of archived checkpoints were
metered in the branch's own cost.jsonl after result.json had sealed that
ledger (result.cost == ledger). A curve retry after a failed shard then failed
result validation with "cost ledger changed" on every pass. This operator
action moves the reporting-ledger 'curve' rows to curve/cost.jsonl, where the
current runtime meters them, writes a receipt under curve/, and removes a
failure.json that says "cost ledger changed". Branches without a result, and
branches a worker holds, are left alone. Nothing is deleted.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import selection_gate as core  # noqa: E402
import selection_gate_gpu as base  # noqa: E402

SCHEMA = "selection-switch-curve-ledger-split/v1"


def curve_rows(events):
    return [e for e in events if e.get("ledger") == "reporting" and e.get("phase") == "curve"]


def candidates(root):
    out = []
    for result in sorted(root.glob("states/**/result.json")):
        directory = result.parent
        if "discarded" in directory.parts or not (directory / "cost.jsonl").exists():
            continue
        try:
            events = [json.loads(l) for l in (directory / "cost.jsonl").read_text().splitlines() if l.strip()]
        except ValueError:
            continue
        if curve_rows(events):
            out.append(directory)
    return out


def split(root, directory, *, apply):
    rel = str(directory.relative_to(root))
    path = directory / "cost.jsonl"
    events = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    moving = curve_rows(events)
    keep = [e for e in events if e not in moving]
    if not moving:
        return f"[curve-ledger] {rel}: nothing to split"
    if not apply:
        return f"[curve-ledger] {rel}: would move {len(moving)} curve row(s) to curve/cost.jsonl"
    try:
        with base.lease(directory / ".task.lock"), base.lease(directory / ".cost.lock"):
            core.cost_summary(keep)
            target = directory / "curve" / "cost.jsonl"
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a") as handle:
                for e in moving:
                    handle.write(json.dumps(e, allow_nan=False) + "\n")
            temporary = path.with_name("cost.jsonl.splitting")
            temporary.write_text("".join(json.dumps(e, allow_nan=False) + "\n" for e in keep))
            temporary.replace(path)
            core.atomic_json(directory / "curve" / "ledger-split.json", {
                "schema": SCHEMA, "directory": rel, "moved_event_ids": sorted({e["event_id"] for e in moving}),
                "rows": len(moving), "time": time.time(),
                "reason": "curve rows were metered in the branch ledger after result.json sealed it; "
                          "the current runtime meters them in curve/cost.jsonl"})
            result = core.read(directory / "result.json")
            sealed = result.get("cost") == base.cost(directory)
            failure = directory / "failure.json"
            cleared = False
            if failure.exists() and str(core.read(failure).get("error", "")).startswith("cost ledger changed") and sealed:
                failure.unlink()
                cleared = True
    except BlockingIOError:
        return f"[curve-ledger] {rel}: skipped, a worker holds this branch right now"
    return (f"[curve-ledger] {rel}: moved {len(moving)} curve row(s) to curve/cost.jsonl; "
            + ("ledger matches the result again" if sealed else "ledger still differs from the result (other rows changed)")
            + ("; failure cleared, the queue resumes the curve next pass" if cleared else ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if not (root / "switch.json").is_file():
        parser.error("root must contain switch.json")
    dirs = candidates(root)
    if not dirs:
        print(f"[curve-ledger] {root.name}: no published branch carries curve rows in its ledger")
        return 0
    for directory in dirs:
        print(split(root, directory, apply=args.apply), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
