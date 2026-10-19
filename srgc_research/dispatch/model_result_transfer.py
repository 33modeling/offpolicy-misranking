"""Small, lossless timing evidence for handing model results to a reviewer.

Per-response rollout archives stay in the run directory. Scalar reward curves,
retained update timers and every restart receipt remain in the transfer file.
Tables declare their columns once; null cells mean unrecorded, never zero.
"""

from pathlib import Path

SCHEMA = "srgc-model-transfer-v1"


def table(rows, columns):
    return {"columns": list(columns), "rows": [[row.get(key) for key in columns] for row in rows]}


def records(value):
    """Read either a transfer table or the original object rows."""
    if isinstance(value, list):
        return value
    columns = value["columns"]
    return [{key: item for key, item in zip(columns, row, strict=True) if item is not None}
            for row in value["rows"]]


def timing(value):
    result = dict(value)
    if "update_timers" in value:
        result["update_timers"] = table(value["update_timers"], (
            "step", "checkpoint", "selection_gpu_seconds", "training_gpu_seconds", "timing_receipt_ids"))
    if "restart_accounting" not in value:
        return result
    original = value["restart_accounting"]
    audit = {key: item for key, item in original.items() if key != "source_files"}
    if "source_receipts" in original:
        # The seed/arm and source root are already bound by the containing task.
        # Keep filenames to distinguish copied receipts without repeating paths.
        sources = [{**row, "source_file": Path(row["source_file"]).name}
                   for row in original["source_receipts"]]
        audit["source_receipts"] = table(sources, (
            "id", "phase", "checkpoint", "gpu_count", "state", "wall_seconds", "gpu_seconds", "source_file"))
    for key, columns in (
        ("retained_receipts", ("step", "phase", "receipt_id", "gpu_seconds", "matching_receipts")),
        ("discarded_completed_attempts", ("id", "phase", "checkpoint", "gpu_seconds")),
    ):
        if key in original:
            audit[key] = table(original[key], columns)
    result["restart_accounting"] = audit
    return result


def compact(result):
    if result.get("transfer_schema") == SCHEMA:
        return result
    datasets = {}
    for name, snapshot in result["datasets"].items():
        row = dict(snapshot)
        row["tasks"] = [{**task, **({"submission_timing": timing(task["submission_timing"])}
                        if "submission_timing" in task else {})} for task in snapshot["tasks"]]
        row["reward_curves"] = []
        for curve in snapshot.get("reward_curves", []):
            # Keep measured scalar training and fixed held-out rewards separate,
            # including every recorded update and explicit missing evaluations.
            entry = {key: value for key, value in curve.items() if key not in {
                "rollout_groups", "rollout_sources", "rollout_recording_costs", "diagnostic_costs"}}
            entry["detail_records"] = {
                "rollout_groups": len(curve.get("rollout_groups", [])),
                "rollout_sources": len(curve.get("rollout_sources", [])),
                "location": "source_root/runs/<dataset>/seed-<seed>/reward-curves/<arm>",
            }
            row["reward_curves"].append(entry)
        datasets[name] = row
    return {**result, "datasets": datasets, "transfer_schema": SCHEMA,
            "transfer_notes": {
                "tables": "columns declare row positions; null means unrecorded; original numeric precision retained",
                "receipt_files": "source_root/runs/<dataset>/seed-<seed>/cost-receipts/<arm>/<source_file>",
                "detailed_records": "original rollout archives, stage receipts and checkpoints remain in source_root; --full exports details",
            }}
