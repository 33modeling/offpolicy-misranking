"""Add diagnostic cost receipts without changing frozen measurement code."""

import json
import math
from unittest.mock import patch

from srgc_rebuttal import cost_ledger
from srgc_rebuttal.runtime import atomic_json


class InformationPhaseLedger(cost_ledger.PhaseLedger):
    def record(self, event):
        if event["phase"] != "diagnostic":
            return super().record(event)
        path = self.directory / f"{event['id']}.json"
        if event["state"] == "finished":
            start = json.loads(path.read_text())
            if start["state"] != "started" or any(start[k] != event[k] for k in (
                    "id", "phase", "checkpoint", "gpu_count")):
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


def diagnostic_costs():
    return patch.object(cost_ledger, "PhaseLedger", InformationPhaseLedger)
