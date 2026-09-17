#!/usr/bin/env python3
"""Waive attempts that stalled after a GPU fault and consumed a branch allocation.

A rank that dies with a CUDA fault can leave the trainer hung until the meter's
allocation limit; the attempt is then charged in full and every retry ends with
"branch allocation exhausted before a valid checkpoint" (or dies at once with
"train exceeded Ns allocation limit" when only a sliver is left). This operator
action moves the wasted attempts' ledger lines to cost-waived.jsonl, writes a
receipt under waivers/ with the evidence, discards the attempts' outputs and
removes failure.json so the queue reruns the branch from its parent policy with
the allocation restored. An attempt qualifies when its phase log shows a
CUDA/NCCL fault signature, the stall watchdog stopped it (stalled.json), a
signal killed it, or the branch never reached a checkpoint (the attempt bought
no training). A branch that has trained to a checkpoint keeps any failed attempt
without such evidence and is left to an operator, as is a branch already waived
MAX_ROUNDS times. Live branches and published results are never touched.
Nothing is deleted: the waived lines and the phase logs remain.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import selection_gate as core  # noqa: E402
import selection_gate_gpu as base  # noqa: E402

EXHAUSTED = "branch allocation exhausted before a valid checkpoint"
TIMEOUT = re.compile(r"^[\w-]+ exceeded \d+s allocation limit")
# Waiver rounds per branch before it is left to an operator: a branch that keeps
# failing without a checkpoint is not an infrastructure loss to return forever.
MAX_ROUNDS = 6
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


def has_checkpoint(directory):
    """True when an attempt of this branch reached a checkpoint or a published policy."""
    policy = directory / "policy"
    if any(policy.glob("checkpoint-*/adapter_model.safetensors")):
        return True
    return (policy / "adapter_model.safetensors").is_file() or (policy / "budget_stop.json").is_file()


def stall_records(directory):
    """Event ids the stall watchdog stopped in this branch (stalled.json holds the last one)."""
    path = directory / "stalled.json"
    try:
        record = json.loads(path.read_text()) if path.is_file() else None
    except ValueError:
        return {}
    return {record["event_id"]: record} if isinstance(record, dict) and record.get("event_id") else {}


def evidence(directory, fin, *, stalls, progressed):
    """Why a failed attempt is infrastructure loss, or None when it is not known to be."""
    excerpt = fault_excerpt(directory, fin.get("phase", ""))
    if excerpt is not None:
        return {"kind": "gpu-fault", **excerpt}
    stall = stalls.get(fin["event_id"])
    if stall is not None:
        return {"kind": "stall-watchdog", "silent_seconds": stall.get("silent_seconds"), "host": stall.get("host")}
    code = fin.get("exit_code")
    if isinstance(code, int) and code < 0:
        return {"kind": "killed", "signal": -code}
    recovery = fin.get("recovery")
    if isinstance(recovery, dict) and recovery.get("kind"):
        # Closed from evidence after its owner vanished (a reclaimed or dead node), not by the meter.
        return {"kind": "stale-closed", "recovery": recovery.get("kind"), "silent_seconds": recovery.get("silent_seconds")}
    if not progressed:
        return {"kind": "no-progress", "note": "the branch never reached a checkpoint, so the attempt bought no training"}
    return None


def stalled_attempts(directory):
    """Finished failed (nonzero exit) deployment events with infrastructure-loss evidence.

    Returns (events, found, unattributed): ``found`` are the waivable attempts and
    ``unattributed`` the failed attempts without evidence, which block the waiver.
    """
    path = directory / "cost.jsonl"
    if not path.exists():
        return [], [], []
    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    finished = {e["event_id"]: e for e in events if e.get("state") == "finished"}
    stalls, progressed = stall_records(directory), has_checkpoint(directory)
    found, unattributed = [], []
    for event_id, fin in finished.items():
        if fin.get("exit_code") in (0, None) or fin.get("ledger") == "reporting":
            continue
        row = {"event_id": event_id, "phase": fin.get("phase"), "host": fin.get("host"),
               "seconds": fin.get("seconds"), "allocated_gpu_seconds": fin.get("allocated_gpu_seconds"),
               "exit_code": fin.get("exit_code")}
        why = evidence(directory, fin, stalls=stalls, progressed=progressed)
        if why is None:
            unattributed.append(row)
        else:
            found.append({**row, "fault": why})
    return events, found, unattributed


def rounds_waived(directory):
    """Waiver rounds already applied to this branch (one discard tag per round)."""
    tags = set()
    for receipt in (directory / "waivers").glob("*.json"):
        try:
            tags.add(json.loads(receipt.read_text()).get("discarded_to"))
        except (OSError, ValueError):
            continue
    return len(tags)


def candidates(root):
    """Branch directories with a recorded failure and no published result.

    Every failed attempt is examined at once, not only after the branch's
    allocation is exhausted: an attempt lost to a faulty node must not eat into
    the allocation of the retry, or a branch that meets three or four such
    nodes ends exhausted before it ever trains.
    """
    out = []
    for failure in sorted(root.glob("states/**/failure.json")):
        if "discarded" in failure.parts or (failure.parent / "result.json").exists():
            continue
        try:
            core.read(failure)
        except (OSError, ValueError):
            continue
        out.append(failure.parent)
    return out


def waive(root, directory, *, apply):
    rel = str(directory.relative_to(root))
    if (directory / "result.json").exists():
        return f"[waive] {rel}: skipped, result already published"
    events, found, unattributed = stalled_attempts(directory)
    if unattributed:
        return (f"[waive] {rel}: skipped, failed attempt(s) without fault evidence after training progress: "
                + ", ".join(f"{u['phase']} {u['event_id'][:8]} exit {u['exit_code']}" for u in unattributed)
                + "; needs an operator")
    if not found:
        return f"[waive] {rel}: skipped, no failed attempt with fault evidence (log signature, stall watchdog, signal, stale close, or no checkpoint); the queue retries it as is"
    rounds = rounds_waived(directory)
    if rounds >= MAX_ROUNDS:
        return f"[waive] {rel}: skipped, {rounds} waiver rounds already; the branch keeps failing, needs an operator"
    if not apply:
        return f"[waive] {rel}: would waive " + ", ".join(f"{f['phase']} {f['event_id'][:8]} on {f['host']} ({f['allocated_gpu_seconds']:.0f} GPU-s) [{f['fault']['kind']}]" for f in found)
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
                    "reason": f"attempt lost to infrastructure ({f['fault']['kind']}): a GPU fault, a stall the "
                              "watchdog stopped, a kill, or an attempt that bought no training before the "
                              "allocation ran out; not selector cost", "round": rounds+1,
                    "operator": "run_selection_switch.sh waive"})
            temporary.replace(path)
            (directory / "failure.json").unlink(missing_ok=True)
    except BlockingIOError:
        return f"[waive] {rel}: skipped, a worker holds this branch right now"
    restored = sum(f["allocated_gpu_seconds"] or 0 for f in found)
    return (f"[waive] {rel}: waived " + ", ".join(f"{f['phase']} {f['event_id'][:8]} on {f['host']} ({f['fault']['kind']})" for f in found)
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
        print(f"[waive] {root.name}: no failed branch to waive")
        return 0
    for directory in dirs:
        print(waive(root, directory, apply=args.apply), flush=True)
    if not args.apply:
        print("[waive] dry run; the launcher's 'waive' mode applies these")
    return 0


if __name__ == "__main__":
    sys.exit(main())
