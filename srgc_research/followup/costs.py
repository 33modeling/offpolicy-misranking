"""Use the frozen cost ledger with separate labels for new diagnostic phases."""

from srgc_rebuttal.cost_ledger import PhaseLedger

from .common import read


class DiagnosticLedger(PhaseLedger):
    def record(self, event):
        label = event["phase"]
        phase = {"source_snapshot": "preparation", "n12_panel_inference": "evaluation",
                 "n13_scoring_A": "selection", "n13_scoring_B": "selection"}.get(label, label)
        path = self.directory / f"{event['id']}.json"
        if event["state"] == "finished" and read(path).get("diagnostic_phase") != label:
            raise ValueError("diagnostic phase completion differs from its start")
        super().record({**event, "phase": phase, "diagnostic_phase": label})

    def totals(self):
        totals = super().totals()
        labels = {}
        for path in sorted(self.directory.glob("*.json")):
            event = read(path)
            label = event["diagnostic_phase"]
            row = labels.setdefault(label, {"known_gpu_seconds": 0., "known_wall_seconds": 0., "complete": True})
            if event["state"] != "finished":
                row["complete"] = False
            else:
                row["known_gpu_seconds"] += event["gpu_seconds"]
                row["known_wall_seconds"] += event["wall_seconds"]
        return {**totals, "diagnostic_phases": labels, "diagnostic_phases_additive_to_phase_totals": False}
