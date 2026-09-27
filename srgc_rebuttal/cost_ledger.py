"""Durable phase receipts: charge retries once and expose interrupted timers."""

import json
import math
from pathlib import Path

from .runtime import atomic_json


class PhaseLedger:
    def __init__(self, directory: Path):
        self.directory = directory

    def record(self, event: dict):
        if event["phase"] not in {"selection", "training", "startup", "preparation",
                                  "checkpoint_save", "checkpoint_load", "evaluation",
                                  "cache_generation", "cache_export", "session"}:
            raise ValueError("unknown metered phase")
        path = self.directory / f"{event['id']}.json"
        if event["state"] == "finished":
            start = json.loads(path.read_text())
            if start["state"] != "started" or any(start[k] != event[k] for k in ("id", "phase", "checkpoint", "gpu_count")):
                raise ValueError("phase completion differs from its start")
            if not math.isfinite(event["gpu_seconds"]) or event["gpu_seconds"] < 0:
                raise ValueError("invalid measured cost")
            if "wall_seconds" in event and (not math.isfinite(event["wall_seconds"]) or
                    event["wall_seconds"] < 0 or not math.isclose(event["gpu_seconds"],
                        event["wall_seconds"] * event["gpu_count"], rel_tol=1e-9, abs_tol=1e-6)):
                raise ValueError("wall time and allocated GPU time disagree")
            if "stages" in event:
                values = [s["gpu_seconds"] for s in event["stages"].values()]
                if any(not math.isfinite(v) or v < 0 for v in values) or not math.isclose(
                        sum(values), event["gpu_seconds"], rel_tol=1e-9, abs_tol=1e-6):
                    raise ValueError("exclusive stage costs do not reconcile")
        elif event["state"] != "started" or path.exists():
            raise ValueError("invalid or repeated phase start")
        atomic_json(path, event)

    def totals(self) -> dict:
        totals = {"selection_gpu_seconds": 0.0, "training_gpu_seconds": 0.0}
        unfinished, phases, stages, counts = [], 0, {}, {}
        wall = {}
        for path in sorted(self.directory.glob("*.json")):
            event = json.loads(path.read_text())
            phases += 1
            if event["state"] != "finished":
                unfinished.append({"id": event["id"], "phase": event["phase"],
                                   "checkpoint": event["checkpoint"]})
                continue
            value = event["gpu_seconds"]
            if not math.isfinite(value) or value < 0:
                raise ValueError("invalid recorded GPU time")
            key = f"{event['phase']}_gpu_seconds"
            totals[key] = totals.get(key, 0.0) + value
            phase = event["phase"]
            wall[phase] = wall.get(phase, 0.0) + event.get("wall_seconds", 0.0)
            for name, row in event.get("stages", {}).items():
                target = stages.setdefault(f"{phase}.{name}",
                    {"gpu_seconds": 0.0, "rank_wall_seconds": 0.0, "calls": 0})
                for metric in target:
                    target[metric] += row[metric]
            for name, count in event.get("counts", {}).items():
                key = f"{phase}.{name}"
                counts[key] = counts.get(key, 0) + count
        return {"known_gpu_seconds": totals, "complete": not unfinished,
                "unfinished_phases": unfinished, "recorded_phases": phases,
                "phase_wall_seconds": wall, "exclusive_stages": stages, "counts": counts,
                "total_gpu_seconds": sum(totals.values()) if not unfinished else None}
