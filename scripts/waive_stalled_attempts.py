#!/usr/bin/env python3
"""Waive attempts that stalled after a GPU fault and consumed a branch allocation.

A rank that dies with a CUDA fault can leave the trainer hung until the meter's
allocation limit; the attempt is then charged in full and every retry ends with
"branch allocation exhausted before a valid checkpoint". This operator action
moves that attempt's ledger lines to cost-waived.jsonl, writes a receipt under
waivers/ with the fault evidence, and removes failure.json so the queue retries
the branch with its allocation restored. Only attempts whose phase log shows a
CUDA/NCCL fault signature qualify; live branches and published results are
never touched. Nothing is deleted: the waived lines and the phase logs remain.
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

EXHAUSTED = "branch allocation exhausted before a valid checkpoint"
FAULT_SIGNATURES = ("unspecified launch failure", "Cuda failure", "CUDA error", "illegal memory access",
                    "uncorrectable ECC error", "NCCL error")
SCHEMA = "selection-switch-cost-waiver/v1"
RESET_SCHEMA = "selection-switch-branch-reset/v1"
# Everything an attempt writes that a retry would otherwise resume from or reuse.
ATTEMPT_OUTPUTS = ("policy", "evaluation", "result.json", "result.sha256.json", "progress.json",
                   "failure.json", "stalled.json")


def discard_outputs(directory, tag):
    """Move an attempt's training outputs aside so the retry starts from the state's parent policy.

    The trainer resumes from checkpoints found in its output directory; a waived
    attempt's checkpoints would otherwise give the retry the discarded attempt's
    updates on top of a fresh allocation.
    """
    target = directory / "discarded" / tag
    moved = []
    for name in ATTEMPT_OUTPUTS:
        path = directory / name
        if path.exists() or path.is_symlink():
            target.mkdir(parents=True, exist_ok=True)
            path.rename(target / name)
            moved.append(name)
    return moved


def fault_excerpt(directory, phase):
    """First fault line of the phase's worker logs, or None when no fault is recorded."""
    for path in sorted(directory.glob(f"{phase}-*.log")):
        try:
            for line in path.read_text(errors="replace").splitlines():
                if any(sig in line for sig in FAULT_SIGNATURES):
                    return {"log": path.name, "line": line.strip()[:300]}
        except OSError:
            continue
    return None


def stalled_attempts(directory):
    """Finished events that failed (nonzero exit) in a phase whose log shows a GPU fault."""
    path = directory / "cost.jsonl"
    if not path.exists():
        return [], []
    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    finished = {e["event_id"]: e for e in events if e.get("state") == "finished"}
    found = []
    for event_id, fin in finished.items():
        if fin.get("exit_code") in (0, None) or fin.get("ledger") == "reporting":
            continue
        excerpt = fault_excerpt(directory, fin.get("phase", ""))
        if excerpt is None:
            continue
        found.append({"event_id": event_id, "phase": fin.get("phase"), "host": fin.get("host"),
                      "seconds": fin.get("seconds"), "allocated_gpu_seconds": fin.get("allocated_gpu_seconds"),
                      "exit_code": fin.get("exit_code"), "fault": excerpt})
    return events, found


def candidates(root):
    """Branch directories whose recorded failure is an exhausted allocation."""
    out = []
    for failure in sorted(root.glob("states/**/failure.json")):
        if "discarded" in failure.parts:
            continue
        try:
            error = str(core.read(failure).get("error", ""))
        except (OSError, ValueError):
            continue
        if error.startswith(EXHAUSTED):
            out.append(failure.parent)
    return out


def waive(root, directory, *, apply):
    rel = str(directory.relative_to(root))
    if (directory / "result.json").exists():
        return f"[waive] {rel}: skipped, result already published"
    events, found = stalled_attempts(directory)
    if not found:
        return f"[waive] {rel}: skipped, no failed attempt with a GPU-fault signature in its log"
    if not apply:
        return f"[waive] {rel}: would waive " + ", ".join(f"{f['phase']} {f['event_id'][:8]} on {f['host']} ({f['allocated_gpu_seconds']:.0f} GPU-s)" for f in found)
    try:
        with base.lease(directory / ".task.lock"), base.lease(directory / ".cost.lock"):
            waived_ids = {f["event_id"] for f in found}
            keep = [e for e in events if e["event_id"] not in waived_ids]
            moved = [e for e in events if e["event_id"] in waived_ids]
            core.cost_summary(keep)  # the remaining ledger must stay well formed
            path = directory / "cost.jsonl"
            temporary = path.with_name("cost.jsonl.waiving")
            temporary.write_text("".join(json.dumps(e, allow_nan=False) + "\n" for e in keep))
            with (directory / "cost-waived.jsonl").open("a") as handle:
                for e in moved:
                    handle.write(json.dumps(e, allow_nan=False) + "\n")
            tag = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            moved = discard_outputs(directory, tag)
            for f in found:
                core.atomic_json(directory / "waivers" / f"{f['event_id']}.json", {
                    "schema": SCHEMA, "directory": rel, "attempt": f, "waived_at": time.time(),
                    "discarded_outputs": moved, "discarded_to": f"discarded/{tag}",
                    "reason": "attempt stalled after a GPU fault and was terminated at the allocation limit; "
                              "the fault is infrastructure, not selector cost",
                    "operator": "run_selection_switch.sh waive"})
            temporary.replace(path)
            (directory / "failure.json").unlink(missing_ok=True)
    except BlockingIOError:
        return f"[waive] {rel}: skipped, a worker holds this branch right now"
    restored = sum(f["allocated_gpu_seconds"] or 0 for f in found)
    return (f"[waive] {rel}: waived " + ", ".join(f"{f['phase']} {f['event_id'][:8]} on {f['host']}" for f in found)
            + f"; {restored:.0f} GPU-s returned to the allocation; failure cleared, the queue retries it next pass")


def resumed_after_waiver(root):
    """Branches waived before outputs were discarded: their retry resumed the discarded checkpoints."""
    out = []
    for waivers in sorted(root.glob("states/**/waivers")):
        directory = waivers.parent
        if not (directory / "discarded").exists():
            out.append(directory)
    return out


def reset_branch(root, directory, *, apply):
    """Discard a contaminated attempt entirely: its ledger lines, outputs and result.

    Used for branches whose retry resumed from a waived attempt's checkpoints, so
    one allocation bought the discarded attempt's updates plus its own. The
    frozen decision, execution record and subset stay; the queue reruns the
    branch from the state's parent policy with a full allocation.
    """
    rel = str(directory.relative_to(root))
    if not apply:
        return f"[reset] {rel}: would discard the resumed attempt's ledger and outputs and rerun from the parent policy"
    try:
        with base.lease(directory / ".task.lock"), base.lease(directory / ".cost.lock"):
            tag = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            path = directory / "cost.jsonl"
            lines = [l for l in path.read_text().splitlines() if l.strip()] if path.exists() else []
            with (directory / "cost-discarded.jsonl").open("a") as handle:
                for line in lines:
                    handle.write(line + "\n")
            moved = discard_outputs(directory, tag)
            core.atomic_json(directory / "discards" / f"{tag}.json", {
                "schema": RESET_SCHEMA, "directory": rel, "reset_at": time.time(),
                "ledger_lines_discarded": len(lines), "discarded_outputs": moved, "discarded_to": f"discarded/{tag}",
                "reason": "the retry after a waiver resumed the discarded attempt's checkpoints, so one allocation "
                          "bought two attempts' updates; the branch reruns from the state's parent policy",
                "operator": "run_selection_switch.sh reset-waived"})
            if path.exists():
                path.write_text("")
    except BlockingIOError:
        return f"[reset] {rel}: skipped, a worker holds this branch; stop that node first"
    return f"[reset] {rel}: discarded {len(lines)} ledger line(s) and {', '.join(moved) or 'no outputs'}; the queue reruns it from the parent policy"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="write the waivers; without it, only report")
    parser.add_argument("--reset-waived", action="store_true",
                        help="reset branches whose retry resumed a waived attempt's checkpoints (rerun from scratch)")
    args = parser.parse_args()
    root = args.root.resolve()
    if not any((root / name).is_file() for name in ("switch.json", "mopps.json")):
        parser.error("root must contain switch.json or mopps.json")
    if args.reset_waived:
        dirs = resumed_after_waiver(root)
        if not dirs:
            print(f"[reset] {root.name}: no waived branch resumed a discarded attempt")
            return 0
        for directory in dirs:
            print(reset_branch(root, directory, apply=args.apply), flush=True)
        if not args.apply:
            print("[reset] dry run; the launcher's 'reset-waived' mode applies these")
        return 0
    dirs = candidates(root)
    if not dirs:
        print(f"[waive] {root.name}: no branch failed with an exhausted allocation")
        return 0
    for directory in dirs:
        print(waive(root, directory, apply=args.apply), flush=True)
    if not args.apply:
        print("[waive] dry run; the launcher's 'waive' mode applies these")
    return 0


if __name__ == "__main__":
    sys.exit(main())
