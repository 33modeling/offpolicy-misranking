"""Read-only allocated-GPU accounting, deduplicated by recorded operation identity.

Phase costs and inclusive invocation costs are parallel views, never additive.
Unknown interrupted durations stay unknown. No utilization or kernel-time claim
is derived from allocated GPU-seconds. Original receipts are never rewritten.
"""

import copy
import hashlib
import json
import math
from pathlib import Path

from srgc_rebuttal.plan import digest
from srgc_rebuttal.runtime import matches

SCHEMA = "srgc-allocated-gpu-accounting-v2"
PHASES = frozenset(("selection", "training", "startup", "preparation", "checkpoint_save",
                    "checkpoint_load", "evaluation", "cache_generation", "cache_export", "session"))
MEASUREMENTS = ("phase", "checkpoint", "gpu_count", "wall_seconds", "gpu_seconds", "stages", "counts", "rank_timings")


def finite(value):
    if type(value) not in (int, float):
        raise ValueError("GPU timing must be finite nonnegative numeric evidence")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError("GPU timing exceeds finite numeric range") from error
    if not math.isfinite(number) or number < 0:
        raise ValueError("GPU timing must be finite nonnegative numeric evidence")
    return number


def total(values):
    try:
        return finite(math.fsum(values))
    except OverflowError as error:
        raise ValueError("GPU timing sum exceeds finite numeric range") from error


def validate(row):
    if not isinstance(row, dict):
        raise TypeError("GPU timing receipt must be an object")
    if not isinstance(row.get("id"), str) or not row["id"]:
        raise ValueError("GPU timing receipt lacks an operation ID")
    if row.get("phase") not in PHASES or row.get("state") not in {"started", "finished"}:
        raise ValueError("invalid GPU timing phase or state")
    count = row.get("gpu_count")
    if count is not None and (type(count) is not int or count < 0):
        raise ValueError("allocated GPU count must be a nonnegative integer")
    if count is not None:
        finite(count)
    if row["state"] == "started":
        return
    seconds = finite(row.get("gpu_seconds"))
    if "wall_seconds" in row:
        wall = finite(row["wall_seconds"])
        if count is None or not math.isclose(seconds, wall * count, rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError("GPU-seconds differ from wall seconds times allocated GPUs")
    stages = row.get("stages", {})
    counts = row.get("counts", {})
    if not isinstance(stages, dict) or not isinstance(counts, dict):
        raise TypeError("GPU timing stages and counters must be objects")
    for name, stage in stages.items():
        if not isinstance(name, str) or not isinstance(stage, dict):
            raise TypeError("invalid GPU stage measurement")
        finite(stage.get("gpu_seconds"))
        for field in ("rank_wall_seconds", "calls"):
            if field in stage:
                finite(stage[field])
    if stages and not math.isclose(total(s["gpu_seconds"] for s in stages.values()),
                                   seconds, rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError("exclusive GPU stage costs do not reconcile with their phase")
    for name, value in counts.items():
        if not isinstance(name, str):
            raise TypeError("GPU timing counter name must be a string")
        finite(value)
    if "rank_timings" in row:
        ranks = row["rank_timings"]
        if not isinstance(ranks, list) or not ranks:
            raise ValueError("missing recorded rank timings")
        for rank in ranks:
            if not isinstance(rank, dict) or type(rank.get("local_gpu_count")) is not int or rank["local_gpu_count"] < 0:
                raise ValueError("invalid rank GPU allocation")
            finite(rank.get("wall_seconds"))
        if count is None or sum(r["local_gpu_count"] for r in ranks) != count:
            raise ValueError("rank GPU allocations differ from the phase allocation")
        if "wall_seconds" in row and not math.isclose(max(r["wall_seconds"] for r in ranks),
                                                       row["wall_seconds"], rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError("rank durations differ from the recorded synchronized phase")


class ReceiptLedger:
    """Canonical reader; a start and finish name one operation, not two charges."""

    def __init__(self, directory):
        self.directory = Path(directory)

    def totals(self):
        paths = sorted(self.directory.glob("*.json"))
        events, copies, raw_seconds = {}, 0, []
        sources = []
        for path in paths:
            before = path.stat()
            row = json.loads(path.read_text())
            after = path.stat()
            if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
                raise ValueError("GPU timing receipt changed during inspection; rerun status/results")
            validate(row)
            sources.append(str(path))
            if row["state"] == "finished":
                raw_seconds.append(row["gpu_seconds"])
            previous = events.get(row["id"])
            if previous is not None:
                if any(previous.get(k) != row.get(k) for k in ("phase", "checkpoint", "gpu_count")):
                    raise ValueError("one GPU operation ID names different operations")
                if previous["state"] == row["state"] == "finished":
                    if any(previous.get(k) != row.get(k) for k in MEASUREMENTS):
                        raise ValueError("conflicting completed copies of a GPU operation")
                    copies += 1
                    continue
                if previous["state"] == "finished":
                    continue
            events[row["id"]] = row
        finished = [row for row in events.values() if row["state"] == "finished"]
        sessions_only = bool(events) and all(row["phase"] == "session" for row in events.values())
        selected = finished if sessions_only else [r for r in finished if r["phase"] != "session"]
        totals = {"selection_gpu_seconds": 0.0, "training_gpu_seconds": 0.0}
        wall, stages, counts = {}, {}, {}
        for phase in sorted({r["phase"] for r in selected}):
            rows = [r for r in selected if r["phase"] == phase]
            totals[phase + "_gpu_seconds"] = total(r["gpu_seconds"] for r in rows)
            wall[phase] = (total(r["wall_seconds"] for r in rows)
                           if all("wall_seconds" in r for r in rows) else None)
            for row in rows:
                for name, value in row.get("stages", {}).items():
                    target = stages.setdefault(phase + "." + name, {"gpu_seconds": 0.0, "rank_wall_seconds": 0.0, "calls": 0})
                    for key in target:
                        target[key] += value.get(key, 0)
                for name, value in row.get("counts", {}).items():
                    key = phase + "." + name
                    counts[key] = counts.get(key, 0) + value
        unfinished = [{k: row.get(k) for k in ("id", "phase", "checkpoint")}
                      for row in events.values() if row["state"] == "started"]
        unique_seconds = total(row["gpu_seconds"] for row in finished)
        container_seconds = total(row["gpu_seconds"] for row in finished if row["phase"] == "session")
        return {"schema": SCHEMA, "known_gpu_seconds": totals, "complete": bool(events) and not unfinished,
                "unfinished_phases": unfinished, "recorded_phases": len(events),
                "phase_wall_seconds": wall, "exclusive_stages": stages, "counts": counts,
                "recorded_phase_names": sorted({r["phase"] for r in events.values()}),
                "finished_phases": len(selected),
                "completed_phase_counts": {phase: sum(r["phase"] == phase for r in selected)
                                           for phase in {r["phase"] for r in selected}},
                "total_gpu_seconds": total(totals.values()) if events and not unfinished else None,
                "known_completed_gpu_seconds": total(totals.values()),
                "inclusive_session_gpu_seconds": container_seconds if any(r["phase"] == "session" for r in finished) else None,
                "duplicate_receipt_copies": copies,
                "duplicate_copy_gpu_seconds_excluded": total(raw_seconds) - unique_seconds,
                "source_files": sources,
                "accounting": "unique operation IDs; inclusive sessions never added to contained phases"}


def measured(directory, *, kind=None):
    report = ReceiptLedger(directory).totals()
    names = report["recorded_phase_names"]
    if report["recorded_phases"] and (kind == "session" and names != ["session"]
                                     or kind == "phase" and "session" in names):
        raise ValueError("inclusive invocation and exclusive phase receipts are in the wrong scope")
    if not report["recorded_phases"]:
        report["complete"] = False
        report["total_gpu_seconds"] = None
    return report


def seed_costs(folder, bundle_path, arms):
    scopes = {p.name: measured(p, kind="phase") for p in sorted((folder / "cost-receipts").glob("*")) if p.is_dir()}
    sessions = {p.name: measured(p, kind="session") for p in sorted((folder / "invocations").glob("*")) if p.is_dir()}
    rows = {}
    marker_path = folder / "run.json"
    marker = json.loads(marker_path.read_text()) if marker_path.exists() else None
    for arm in arms:
        ledger = scopes.get(arm)
        if ledger is None:
            ledger = measured(folder / "cost-receipts" / arm, kind="phase")
        endpoint_path = folder / f"{arm}-endpoint.json"
        endpoint = json.loads(endpoint_path.read_text()) if endpoint_path.exists() else {}
        expected = {k: marker[k] for k in ("seed", "plan_sha256", "input_sha256", "implementation_sha256")} if marker else {}
        if endpoint and marker is not None and not matches(endpoint, expected):
            raise ValueError("cost endpoint belongs to a different experiment identity")
        declared_complete = endpoint.get("cost_measurement_complete", False)
        if type(declared_complete) is not bool:
            raise ValueError("cost measurement completeness must be a boolean")
        complete = ledger["complete"] and declared_complete
        relevant_sessions = [sessions[s] for s in (arm, "all") if s in sessions]
        complete = complete and bool(relevant_sessions) and all(s["complete"] for s in relevant_sessions)
        values = ledger["known_gpu_seconds"]
        if ledger["complete"] and relevant_sessions and all(s["complete"] for s in relevant_sessions):
            phase_seconds = total(values.values())
            allocated_seconds = total(s["known_completed_gpu_seconds"] for s in relevant_sessions)
            if phase_seconds > allocated_seconds + max(0.01, allocated_seconds * 1e-6):
                raise ValueError("arm phase GPU time exceeds its inclusive invocation cost")
        rows[arm] = {
            "complete": complete,
            "accounting_schema": SCHEMA,
            "recorded_phases": ledger["recorded_phases"],
            "finished_phases": ledger["finished_phases"],
            "completed_phase_counts": ledger["completed_phase_counts"],
            "duplicate_receipt_copies": ledger["duplicate_receipt_copies"],
            "duplicate_copy_gpu_seconds_excluded": ledger["duplicate_copy_gpu_seconds_excluded"],
            "selection_gpu_seconds": values.get("selection_gpu_seconds", 0.0),
            "training_gpu_seconds": values.get("training_gpu_seconds", 0.0),
            "preparation_gpu_seconds": values.get("preparation_gpu_seconds", 0.0),
            "evaluation_gpu_seconds": values.get("evaluation_gpu_seconds", 0.0),
            "checkpoint_gpu_seconds": sum(values.get(f"{p}_gpu_seconds", 0.0)
                                          for p in ("checkpoint_save", "checkpoint_load")),
            "startup_gpu_seconds": values.get("startup_gpu_seconds", 0.0),
            "continuation_phases_gpu_seconds": ledger["total_gpu_seconds"] if complete else None,
            "selection_training_preparation_gpu_seconds": sum(values.get(f"{p}_gpu_seconds", 0.0)
                for p in ("selection", "training", "preparation")) if complete else None,
            "exclusive_stages": ledger["exclusive_stages"], "counts": ledger["counts"],
            "unfinished_phases": ledger["unfinished_phases"]}
        rows[arm]["known_completed_phase_gpu_seconds"] = values
        if not complete:
            for key in ("selection_gpu_seconds", "training_gpu_seconds", "preparation_gpu_seconds",
                        "evaluation_gpu_seconds", "checkpoint_gpu_seconds", "startup_gpu_seconds"):
                rows[arm][key] = None
    cache_path = bundle_path.with_suffix(".cache") / "cost-summary.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else None
    if cache is not None:
        cache = copy.deepcopy(cache)
        declared_complete = cache.get("complete") is True
        directory = bundle_path.with_suffix(".cache")
        if any((directory / "invocations").glob("*.json")):
            cache["invocations"] = measured(directory / "invocations", kind="session")
        if any((directory / "cost-receipts").glob("*.json")):
            rebuilt = measured(directory / "cost-receipts", kind="phase")
            for key in ("known_gpu_seconds", "total_gpu_seconds", "complete", "recorded_phases",
                        "unfinished_phases", "exclusive_stages", "counts"):
                cache[key] = rebuilt[key]
        cache["complete"] = (declared_complete and cache.get("complete") is True
                             and cache.get("invocations", {}).get("complete") is True)
        if not cache["complete"]:
            cache["total_gpu_seconds"] = None
    live_costs = [json.loads(path.read_text()) for path in
                  sorted((bundle_path.with_suffix(".cache") / "live-costs").glob("*.json"))]
    if cache is not None and cache["bundle_sha256"] != digest(bundle_path):
        raise ValueError("cache timings belong to a different input bundle")
    all_complete = (bool(scopes) and bool(sessions) and "shared-prefix" in scopes and
                    all(r["complete"] for r in [*scopes.values(), *sessions.values(), *rows.values()]))
    phase_sum = total(value for r in scopes.values() for value in r["known_gpu_seconds"].values())
    session_sum = total(value for r in sessions.values() for value in r["known_gpu_seconds"].values())
    difference = session_sum - phase_sum
    if all_complete and difference < -max(0.01, session_sum * 1e-6):
        raise ValueError("phase totals exceed inclusive invocation costs")
    cache_complete = cache is not None and cache["complete"]
    cache_total = cache["invocations"]["total_gpu_seconds"] if cache_complete else None
    if cache_complete:
        cache_total = finite(cache_total)
    return {"accounting_schema": SCHEMA,
            "accounting_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "arms": rows, "shared_prefix": scopes.get("shared-prefix"),
            "shared_all_task_startup_and_load": scopes.get("all"),
            "cache_build": cache, "cache_live_rank_costs": live_costs, "invocations": sessions,
            "experiment_accounting": {
                "complete": all_complete,
                "phase_gpu_seconds": phase_sum,
                "inclusive_invocation_gpu_seconds": session_sum if all_complete else None,
                "cache_inclusive_gpu_seconds": cache_total,
                "experiment_including_cache_gpu_seconds": total((session_sum, cache_total))
                    if all_complete and cache_complete else None,
                "orchestration_and_timer_gpu_seconds": max(0.0, difference) if all_complete else None,
                "known_completed_phase_gpu_seconds": phase_sum,
                "known_completed_invocation_gpu_seconds": session_sum,
                "phase_and_session_totals_are_nonadditive": True,
                "queue_wait_gpu_seconds": None,
                "queue_wait_note": "not measured; task/worker age is never converted into GPU time",
                "duplicate_receipt_copies": sum(r["duplicate_receipt_copies"] for r in [*scopes.values(), *sessions.values()]),
                "duplicate_copy_gpu_seconds_excluded": total(r["duplicate_copy_gpu_seconds_excluded"] for r in [*scopes.values(), *sessions.values()])}}
