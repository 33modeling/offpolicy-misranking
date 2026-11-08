"""E6 cost ledger: append-only interval events, multi-GPU union, four cost views.

Event schema (plan Code §7.2) plus `component` tags. Whether an event is
algorithm-required depends on the LOGICAL method it is charged to (the same On
trajectory serves On-continued, every fixed-tau and every SR-GC rule), so views
are computed from components with a per-method requirement map.

Units: seconds and GPU-seconds internally; GPU-hours only when reporting.
Missing or interrupted costs are null/flagged, never zero.
"""

from __future__ import annotations

import math
import socket
import time

from .common import JsonlLog, new_id, read_jsonl

PHASES = ("model_init", "checkpoint_load", "prefix_train", "initial_cache_generation", "cache_verification",
          "cache_commit", "candidate_rollout", "candidate_gradient", "validation_gradient",
          "sr_comparison_gradient", "projection_and_ranking", "cheap_signal", "subset_preparation",
          "train_rollout", "train_backward", "optimizer_step", "checkpoint_save", "external_evaluation",
          "synchronization", "interrupted_work", "cache_refresh", "selection", "training", "other")

# component -> which logical methods require it
COMPONENTS = ("on_selection", "sr_comparison", "sr_preparation", "training", "cache_build", "cache_refresh",
              "cheap_signal", "checkpoint", "startup", "evaluation", "prefix")

# code-repo meter phase -> (v8 phase, component)
METER_PHASES = {"selection": ("selection", "on_selection"), "training": ("training", "training"),
                "startup": ("model_init", "startup"), "preparation": ("subset_preparation", "sr_preparation"),
                "checkpoint_save": ("checkpoint_save", "checkpoint"),
                "checkpoint_load": ("checkpoint_load", "checkpoint"),
                "evaluation": ("external_evaluation", "evaluation"),
                "cache_generation": ("initial_cache_generation", "cache_build"),
                "cache_export": ("cache_commit", "cache_build"), "session": ("other", "startup"),
                "cache_refresh": ("cache_refresh", "cache_refresh")}


def stage_phase(stage_key: str, parent_phase: str):
    """Map an exclusive stage of a code-repo phase to (v8 phase, component)."""
    k = stage_key
    if k.startswith("sr_comparison"):
        return "sr_comparison_gradient", "sr_comparison"
    if parent_phase == "selection":
        if k.startswith("validation"):
            return "validation_gradient", "on_selection"
        if "cosine_ranking" in k:
            return "projection_and_ranking", "on_selection"
        if "sr_gc_check" in k:
            return "projection_and_ranking", "sr_comparison"
        if "generation" in k or "rollout" in k or "tokenization" in k or "decode" in k or "verification" in k:
            return "candidate_rollout", "on_selection"
        if "backward" in k or "projection" in k or "forward" in k:
            return "candidate_gradient", "on_selection"
        return "selection", "on_selection"
    if parent_phase == "training":
        if "candidate_sampling_and_ranking" in k:
            return "subset_preparation", "training"
        if "backward" in k:
            return "train_backward", "training"
        if "optimizer" in k or "gradient_reduction" in k:
            return "optimizer_step", "training"
        if "generation" in k or "rollout" in k:
            return "train_rollout", "training"
        return "training", "training"
    return METER_PHASES.get(parent_phase, ("other", "startup"))


class Ledger:
    KEY = ("run_id", "attempt_id", "event_id")

    def __init__(self, path, *, run_id, attempt_id, gpu_type=None, host=None, measurement_status="measured",
                 gpu_ids=None, stage_split=True, starts_path=None):
        self.log = JsonlLog(path, self.KEY)
        # Durable phase starts (plan Code 6.6: interrupted attempt cost stays in actual expenditure).
        # Kept apart from the interval events so every consumer of events.jsonl is unchanged; a start
        # without a finished/interrupted event of the same id is work killed mid-phase (SIGKILL).
        self.starts = JsonlLog(starts_path, self.KEY) if starts_path is not None else None
        self.run_id, self.attempt_id = run_id, attempt_id
        self.gpu_type, self.host = gpu_type, host or socket.gethostname()
        self.measurement_status = measurement_status
        self.gpu_ids = gpu_ids
        self.stage_split = stage_split
        self._open = {}

    def record(self, **fields):
        event = {"event_id": fields.pop("event_id", None) or new_id(), "run_id": self.run_id,
                 "attempt_id": self.attempt_id, "gpu_type": self.gpu_type,
                 "measurement_status": self.measurement_status, "retained": True,
                 "shared_artifact_id": None, "overlap_group": None, "source_checkpoint_hash": None,
                 "algorithm_required": None, "attribution": "interval", "parent_event_id": None}
        event.update(fields)
        if event["phase"] not in PHASES:
            raise ValueError(f"unknown ledger phase {event['phase']}")
        if event.get("component") not in COMPONENTS:
            raise ValueError(f"unknown cost component {event.get('component')}")
        for key in ("wall_seconds", "gpu_seconds"):
            value = event.get(key)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError("cost values must be finite and nonnegative")
        start, end = event.get("start_monotonic_ns"), event.get("end_monotonic_ns")
        if start is not None and end is not None and end < start:
            raise ValueError("event end precedes start")
        self.log.append(event)
        return event["event_id"]

    # ---- bridge from srgc_rebuttal.timing.CostMeter / PhaseLedger events ----
    def on_meter_event(self, event: dict):
        if event["state"] == "started":
            self._open[event["id"]] = (time.monotonic_ns(), time.time_ns(), event)
            if self.starts is not None:
                phase, component = METER_PHASES.get(event["phase"], ("other", "startup"))
                gpu_count = int(event.get("gpu_count") or 0)
                self.starts.append({
                    "event_id": event["id"], "run_id": self.run_id, "attempt_id": self.attempt_id,
                    "state": "started", "phase": phase, "source_phase": event["phase"], "component": component,
                    "completed_update": event.get("checkpoint"), "start_monotonic_ns": self._open[event["id"]][0],
                    "start_epoch_ns": self._open[event["id"]][1], "gpu_count": gpu_count, "host": self.host,
                    "gpu_type": self.gpu_type, "gpu_ids": self.gpu_ids if self.gpu_ids is not None else
                    [f"{self.host}:{i}" for i in range(gpu_count)]})
            return
        started = self._open.pop(event["id"], None)
        end_ns = time.monotonic_ns()
        phase, component = METER_PHASES.get(event["phase"], ("other", "startup"))
        gpu_count = int(event.get("gpu_count") or 0)
        gpu_ids = self.gpu_ids if self.gpu_ids is not None else [f"{self.host}:{i}" for i in range(gpu_count)]
        parent = self.record(event_id=event["id"], phase=phase, source_phase=event["phase"], component=component,
                             completed_update=event.get("checkpoint"),
                             start_monotonic_ns=started[0] if started else None, end_monotonic_ns=end_ns,
                             start_epoch_ns=started[1] if started else None,
                             wall_seconds=event.get("wall_seconds"), gpu_seconds=event.get("gpu_seconds"),
                             gpu_ids=gpu_ids, rank_ids=list(range(len(event.get("rank_timings", [])) or 1)),
                             gpu_count=gpu_count)
        if self.stage_split:
            for key, row in (event.get("stages") or {}).items():
                sphase, scomp = stage_phase(key, event["phase"])
                self.record(event_id=f"{event['id']}:{key}", phase=sphase, source_phase=f"{event['phase']}.{key}",
                            component=scomp, completed_update=event.get("checkpoint"),
                            start_monotonic_ns=None, end_monotonic_ns=None, wall_seconds=None,
                            gpu_seconds=row.get("gpu_seconds"), gpu_ids=gpu_ids, rank_ids=[],
                            gpu_count=gpu_count, attribution="exclusive_stage", parent_event_id=parent)
        return parent

    def close_interrupted(self):
        """Phases that started but never finished: unknown cost, not zero."""
        for _, (start, _, event) in list(self._open.items()):
            self.record(event_id=event["id"], phase="interrupted_work", source_phase=event["phase"],
                        component=METER_PHASES.get(event["phase"], ("other", "startup"))[1],
                        completed_update=event.get("checkpoint"), start_monotonic_ns=start,
                        end_monotonic_ns=None, wall_seconds=None, gpu_seconds=None,
                        measurement_status="interrupted", gpu_ids=[], rank_ids=[])
        self._open.clear()


def union_seconds(intervals):
    total, current = 0, None
    for start, end in sorted(intervals):
        if current is None or start > current[1]:
            if current:
                total += current[1] - current[0]
            current = [start, end]
        else:
            current[1] = max(current[1], end)
    if current:
        total += current[1] - current[0]
    return total


def unfinished_starts(events, starts):
    """Durable phase starts with no finished or interrupted event of the same id (killed mid-phase)."""
    ended = {(e["run_id"], e["attempt_id"], e["event_id"]) for e in events}
    return [s for s in starts or [] if (s["run_id"], s["attempt_id"], s["event_id"]) not in ended]


def physical_gpu_seconds(events, starts=None):
    """Union per GPU of interval events, summed across GPUs; dedupe by event_id.

    Exclusive-stage sub-events are attributions inside an interval and are not added.
    Interval events lacking timestamps contribute their gpu_seconds once (flagged).
    `starts` (Ledger durable phase starts): a start that never finished is unknown cost, never zero.
    """
    per_gpu, untimed, missing, seen = {}, 0.0, [], set()
    killed = unfinished_starts(events, starts)
    missing.extend(s["event_id"] for s in killed)
    for e in events:
        key = (e["run_id"], e["attempt_id"], e["event_id"])
        if key in seen or e.get("attribution") == "exclusive_stage":
            continue
        seen.add(key)
        start, end = e.get("start_monotonic_ns"), e.get("end_monotonic_ns")
        host_epoch = e.get("start_epoch_ns")
        if e.get("measurement_status") in ("interrupted", "missing"):
            missing.append(e["event_id"])  # unknown cost is never zero
            continue
        if start is not None and end is not None and e.get("gpu_ids"):
            offset = (host_epoch - start) if host_epoch is not None else 0  # align processes on one host
            for gpu in e["gpu_ids"]:
                per_gpu.setdefault(gpu, []).append((start + offset, end + offset))
        elif e.get("gpu_seconds") is not None:
            untimed += float(e["gpu_seconds"])
        elif (start is not None and end is not None and int(e.get("gpu_count") or 0) == 0
              and e.get("gpu_ids") == []):
            pass  # measured interval on zero allocated GPUs (CPU-only work): 0 GPU-seconds is known
        else:
            missing.append(e["event_id"])
    known = sum(union_seconds(v) for v in per_gpu.values()) / 1e9 + untimed
    complete = not missing
    return {"gpu_seconds": known if complete else None, "known_lower_bound_gpu_seconds": known,
            "untimed_gpu_seconds": untimed, "missing_event_ids": missing, "complete": complete,
            "killed_phase_starts": len(killed), "gpus": sorted(per_gpu)}


def component_totals(events, *, min_update=None, max_update=None, include_selection_at=None, keep=None):
    """Sum exclusive costs by component for events in an update window.

    Selection at boundary u is charged with checkpoint u; training with checkpoint u
    is update u+1. Window semantics: updates (min_update, max_update], plus the
    selection phase at `include_selection_at` (the triggering check).
    Phase events with stage sub-events are split by stage; otherwise the whole
    phase goes to its component. `keep(component, update)` False drops that cost (e.g. SR-GC checks a
    sparser-cadence policy never runs); a dropped stage is still subtracted from its parent phase.
    """
    by_parent = {}
    for e in events:
        if e.get("attribution") == "exclusive_stage" and e.get("parent_event_id"):
            by_parent.setdefault(e["parent_event_id"], []).append(e)
    totals, missing = {}, []
    for e in events:
        if e.get("attribution") == "exclusive_stage":
            continue
        u = e.get("completed_update")
        sel_boundary = e.get("source_phase") == "selection" and include_selection_at is not None and u == include_selection_at
        if not sel_boundary and u is not None:
            if min_update is not None and u < min_update:
                continue
            if max_update is not None and u >= max_update:
                continue
        if e.get("gpu_seconds") is None:
            missing.append({"event_id": e["event_id"], "component": e.get("component"),
                            "status": e.get("measurement_status")})
            continue
        children = by_parent.get(e["event_id"], [])
        kept = (lambda comp: True) if keep is None else (lambda comp: keep(comp, u))
        if children:
            attributed = 0.0
            for c in children:
                if c.get("gpu_seconds") is None:
                    continue
                attributed += c["gpu_seconds"]
                if kept(c["component"]):
                    totals[c["component"]] = totals.get(c["component"], 0.0) + c["gpu_seconds"]
            rest = max(0.0, e["gpu_seconds"] - attributed)
            if rest and kept(e["component"]):
                totals[e["component"]] = totals.get(e["component"], 0.0) + rest
        elif kept(e["component"]):
            totals[e["component"]] = totals.get(e["component"], 0.0) + e["gpu_seconds"]
    return totals, missing


def check_schedule_filter(first, interval):
    """keep() for component_totals: SR-GC comparison cost only at the policy's own check boundaries."""
    def keep(component, update):
        return component != "sr_comparison" or update is None or (
            update >= first and (update - first) % interval == 0)
    return keep


# Which components each method family requires (algorithm-required).
REQUIRED = {
    "on_continued": {"on_selection", "training", "prefix"},
    "fixed_sr": {"on_selection", "training", "sr_preparation", "cache_build", "prefix"},
    "sr_early": {"training", "sr_preparation", "cache_build", "prefix"},
    "uniform_early": {"training", "prefix"},
    # cache_refresh: only srgc_current_refreshed_k* runs record it (refresh after the transition)
    "srgc": {"on_selection", "sr_comparison", "training", "sr_preparation", "cache_build", "cache_refresh", "prefix"},
    "srgc_uniform_destination": {"on_selection", "sr_comparison", "training", "cache_build", "prefix"},
    "plateau_sr": {"on_selection", "cheap_signal", "training", "sr_preparation", "cache_build", "prefix"},
    "sampled_time_sr": {"on_selection", "training", "sr_preparation", "cache_build", "prefix"},
    "sr_refreshed": {"training", "sr_preparation", "cache_build", "cache_refresh", "prefix"},
    "fixed_uniform": {"on_selection", "training", "prefix"},
}


def views(component_gpu_seconds: dict, family: str, *, prefix_gpu_seconds=None, cache_gpu_seconds=None,
          physical=None, amortize_cache_over=None):
    """Four cost views (GPU-seconds). None + reason when a required part is unmeasured."""
    required = REQUIRED[family]
    continuation = sum(v for k, v in component_gpu_seconds.items() if k in required and k not in ("prefix", "cache_build"))
    missing = []
    cold = continuation
    if "prefix" in required:
        if prefix_gpu_seconds is None:
            missing.append("prefix")
        else:
            cold += prefix_gpu_seconds
    if "cache_build" in required:
        if cache_gpu_seconds is None:
            missing.append("cache_build")
        else:
            cold += cache_gpu_seconds
    amortized = amortized_lb = None
    if amortize_cache_over:
        value = continuation + (prefix_gpu_seconds or 0.0) + (
            cache_gpu_seconds / amortize_cache_over if "cache_build" in required and cache_gpu_seconds is not None else 0.0)
        if missing:
            amortized_lb = value  # explicitly a lower bound: some required component is unmeasured
        else:
            amortized = value
    return {"continuation_algorithm": continuation,
            "cold_start_algorithm": None if missing else cold,
            "cold_start_lower_bound": cold if missing else None,
            "cold_start_missing": missing, "physical_expenditure": physical,
            "declared_amortized": amortized, "declared_amortized_lower_bound": amortized_lb,
            "declared_amortized_missing": list(missing) if amortize_cache_over else [],
            "family": family}


def lineage_events(events, attempts):
    """Keep, per completed update, only events of the attempt whose state lineage survived.

    attempts: chronological [(attempt_id, start_update)]; attempt i owns updates
    [start_i, start_{i+1}). Work of an interrupted attempt beyond the resume point was
    redone: it stays in physical expenditure but not in the algorithm views.
    """
    order = [(a, int(s)) for a, s in attempts]
    def owner(u):
        own = None
        for a, start in order:
            if start <= u:
                own = a
        return own
    keep = []
    for e in events:
        u = e.get("completed_update")
        if u is None or not order or owner(u) == e["attempt_id"]:
            keep.append(e)
    return keep


def load_events(paths):
    rows = []
    for p in paths:
        rows.extend(read_jsonl(p))
    return rows


def gpu_hours(seconds):
    return None if seconds is None else seconds / 3600.0
