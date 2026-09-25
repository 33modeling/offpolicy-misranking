"""Read separately metered reference-A stages without running or changing experiments."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import selection_gate as core
import selection_gate_gpu as base

STAGES = ("validation-a", "candidate-a")
SCOPE = (
    "Reference A only: allocated GPU-seconds for its validation and candidate stages, "
    "including model loading, response generation, reward verification, gradient projection, "
    "waiting for workers and recorded retries. B, reused initial ranking, training, cache "
    "creation and the joint A/B CPU aggregation are excluded. This is the logged one-reference "
    "GPU-stage workload on the recorded checkpoints, not a newly executed A-only controller "
    "or an isolated CUDA-kernel timer. No division of A/B totals or step-count extrapolation."
)


def read_json(path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def check_cost(directory, seed, step):
    directory = Path(directory)
    prefix = "sr-gc-" if step == 25 else "sr-gc-repeat-"
    phases = {prefix + stage for stage in STAGES}
    result = {"step": step, "reference": "A", "directory": str(directory),
              "gpu_seconds": None, "known_gpu_seconds": 0., "complete": False,
              "phases": [], "raw_a_events": [], "receipt_recoveries": [],
              "projection_receipts": [], "issues": []}
    try:
        reference, reference_sha = read_json(directory / "reference.json")
        result.update(reference_sha256=reference_sha, reference_record=reference)
        if (reference["config"]["seed"] != seed or reference["config"]["drift"] != step):
            raise ValueError("reference seed/checkpoint does not match the requested A check")
        # Read-only receipt replay: a missing B finish must not hide completed A work.
        raw, all_events = base.read_cost_events(directory)
        result["ledger_sha256"] = hashlib.sha256(raw).hexdigest()
        result["raw_a_events"] = events = [e for e in all_events if e.get("phase") in phases]
        summary = core.cost_summary(events)
        for event_id in summary["incomplete_events"]:
            if Path(event_id).name != event_id or event_id in (".", ".."):
                raise ValueError("invalid A cost event identity")
            receipt = directory / "cost-events" / f"{event_id}.json"
            if receipt.is_file():
                finish, receipt_sha = read_json(receipt)
                if finish.get("event_id") != event_id or finish.get("state") != "finished":
                    raise ValueError("atomic A cost receipt identity mismatch")
                core.cost_summary([*events, finish])
                events.append(finish)
                result["receipt_recoveries"].append({"path": str(receipt), "sha256": receipt_sha})
        for stage in STAGES:
            phase = prefix + stage
            selected = [e for e in events if e["phase"] == phase]
            cost = core.cost_summary(selected)
            finishes = {e["event_id"]: e for e in selected if e["state"] == "finished"}
            starts = {e["event_id"]: e for e in selected if e["state"] == "started"}
            if any(e["gpus"] != 4 for e in selected):
                raise ValueError("A reference stages require the recorded four-GPU allocation")
            closed = [e for event_id, e in finishes.items() if event_id in starts]
            known = sum(e["allocated_gpu_seconds"] for e in closed)
            complete = bool(selected and cost["complete"] and closed and closed[-1]["exit_code"] == 0)
            detail = {"stage": stage, "phase": phase, "known_gpu_seconds": known,
                      "gpu_seconds": known if complete else None, "complete": complete,
                      "event_ids": sorted(starts.keys() | finishes.keys()),
                      "incomplete_events": cost["incomplete_events"],
                      "missing_starts": cost["missing_starts"],
                      "failed_events": sum(e["exit_code"] != 0 for e in closed)}
            result["phases"].append(detail)
            result["known_gpu_seconds"] += known
            if not complete:
                result["issues"].append(f"{stage}: missing, unclosed or unsuccessful A stage")
            for shard in range(4):
                payload = directory / f"{stage}-{shard}.json"
                receipt = directory / f"{stage}-{shard}.done.json"
                saved, receipt_sha = read_json(receipt)
                payload_sha = hashlib.sha256(payload.read_bytes()).hexdigest()
                expected = {"reference_sha256": reference_sha, "stage": stage,
                            "shard": shard, "sha256": payload_sha}
                if saved != expected:
                    raise ValueError(f"{stage} shard {shard}: projection receipt binding differs")
                result["projection_receipts"].append({"path": str(receipt), "sha256": receipt_sha,
                                                       "record": saved})
        result["complete"] = not result["issues"]
        if result["complete"]:
            result["gpu_seconds"] = result["known_gpu_seconds"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result["issues"].append(str(exc))
    return result


def single_reference_checks(root, seed, trigger, *, through=275):
    if type(trigger) is not int or not 25 <= trigger <= through or trigger % 25:
        raise ValueError("invalid recorded Switch horizon")
    root = Path(root)
    checks = []
    for step in range(25, through + 1, 25):
        directory = (root / f"sr-gc/s{seed}-t25" if step == 25 else
                     root / f"sr-gc-repeat/every-25/s{seed}-t25/step-{step}")
        check = check_cost(directory, seed, step)
        check["included_before_switch"] = step <= trigger
        checks.append(check)
    included = [c for c in checks if c["included_before_switch"]]
    missing = [c["step"] for c in included if not c["complete"]]
    known = sum(c["known_gpu_seconds"] for c in included)
    return {"reference": "A", "through_step": trigger, "complete": not missing,
            "gpu_seconds": known if not missing else None, "known_gpu_seconds": known,
            "unknown_steps": missing, "checks": checks, "scope": SCOPE,
            "horizon_scope": "Costs through the recorded Switch trigger; the experiment's "
            "transition, rewards and A/B diagnostic results are not changed by A-only cost accounting."}
