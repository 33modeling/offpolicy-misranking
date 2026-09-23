"""Read-only inventory of saved t25 SR-GC projections at every 25-step check."""
from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path

import selector_pair_srgc_repeat as repeat
import selector_pair_srgc_score as score
from paper_result_text import write_export
from selector_pair_results import srgc_results


def scan_state(root, initial, interval=25):
    seed, start = initial["seed"], initial["step"]
    checkpoints = repeat.inventory(root, seed, start)
    end = max(checkpoints, default=start)
    points = [{"step": start, "status": "measured", "d": initial["d"],
               "selector": initial["selector"], "missing_files": []}]
    for step in range(start + interval, end + 1, interval):
        point = {"step": step, "status": None, "d": None, "selector": None,
                 "missing_files": []}
        checkpoint = checkpoints.get(step)
        if checkpoint is None:
            point["status"] = "checkpoint_missing"
        else:
            directory = repeat.output_dir(root, seed, start, interval) / f"step-{step}"
            try:
                expected, _ = repeat.checkpoint_reference(
                    root, seed, start, step, checkpoint, initial, interval)
                names = ["reference.json"] + [
                    f"{stage}-{shard}{suffix}"
                    for stage in score.STAGES for shard in range(4)
                    for suffix in (".json", ".done.json")]
                point["missing_files"] = [name for name in names
                                          if not (directory / name).is_file()]
                if point["missing_files"]:
                    point["status"] = "projection_missing"
                else:
                    value = repeat.check_projections(directory, root, expected)
                    point.update(status="measured", d=value["d"],
                                 selector=value["selector"])
            except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                point.update(status="invalid", error=str(exc))
        points.append(point)
    return {"state": f"s{seed}-t{start}", "last_saved_checkpoint": end,
            "points": points}


def collect(root, initial, interval=25):
    repeat.positive_int(interval)
    rows = [scan_state(root, item, interval)
            for item in initial.get("decisions", []) if item["step"] == 25]
    return {"schema": "offpolicy-selector-pair/sr-gc-inventory-v1",
            "scope": "Read-only saved t25 On-policy checkpoint diagnostics; later values do not imply an executed switch.",
            "interval": interval, "initial_status": initial["status"],
            "trajectories": sorted(rows, key=lambda row: row["state"])}


def table(report):
    output = io.StringIO()
    output.write("SR-GC t25 SAVED CHECKPOINT INVENTORY (read-only)\n")
    writer = csv.writer(output)
    writer.writerow(("state", "step", "status", "D", "selector", "missing_count", "first_missing", "error"))
    for row in report["trajectories"]:
        for point in row["points"]:
            missing = point["missing_files"]
            writer.writerow((row["state"], point["step"], point["status"],
                             point["d"], point["selector"], len(missing),
                             missing[0] if missing else "", point.get("error", "")))
    return output.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=25)
    args = parser.parse_args()
    root = args.root.resolve()
    report = collect(root, srgc_results(root), args.interval)
    write_export("selector-pair-srgc-inventory", report, table(report), args.out)
    if not report["trajectories"]:
        raise SystemExit("No saved t25 SR-GC decisions found")


if __name__ == "__main__":
    main()
