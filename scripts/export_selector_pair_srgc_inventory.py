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
               "selector": initial["selector"], "missing_files": [],
               "source_path": f"sr-gc/s{seed}-t{start}/decision.json"}]
    for step in range(start + interval, end + 1, interval):
        point = {"step": step, "status": None, "d": None, "selector": None,
                 "missing_files": [], "source_path": str(
                     (repeat.output_dir(root, seed, start, interval) / f"step-{step}/reference.json")
                     .relative_to(root))}
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
    initial_by_state = {(item["seed"], item["step"]): item
                        for item in initial.get("decisions", [])}
    initial_decisions = [
        {"state": f"s{item['seed']}-t{item['step']}", "step": item["step"],
         "d_a": item["d_a"], "d_b": item["d_b"], "d": item["d"],
         "source_path": f"sr-gc/s{item['seed']}-t{item['step']}/decision.json"}
        for item in initial.get("decisions", [])]
    stored_references = []
    for path in sorted((root / "sr-gc-repeat").rglob("reference.json")):
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("SR-GC reference escaped Pair root")
        receipt_count = sum((path.parent / f"{stage}-{shard}.done.json").is_file()
                            for stage in score.STAGES for shard in range(4))
        item = {"path": str(path.relative_to(root)), "done_receipts": receipt_count,
                "state": None, "step": None, "d_a": None, "d_b": None,
                "d": None, "status": None}
        try:
            saved = repeat.read(path, root)
            metadata = saved["repeat"]
            seed, start, step, check_interval = (
                metadata[key] for key in ("seed", "start_step", "step", "interval"))
            item.update(state=f"s{seed}-t{start}", step=step)
            initial_point = initial_by_state[(seed, start)]
            checkpoint = repeat.inventory(root, seed, start).get(step)
            if checkpoint is None:
                raise ValueError("matching On-policy checkpoint not saved")
            expected, _ = repeat.checkpoint_reference(
                root, seed, start, step, checkpoint, initial_point, check_interval)
            value = repeat.check_projections(path.parent, root, expected)
            item.update(status="measured", d_a=value["d_a"], d_b=value["d_b"], d=value["d"])
        except FileNotFoundError as exc:
            item.update(status="projection_missing", error=str(exc.filename))
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            item.update(status="invalid", error=str(exc))
        stored_references.append(item)
    return {"schema": "offpolicy-selector-pair/sr-gc-inventory-v1",
            "scope": "Read-only saved t25 On-policy checkpoint diagnostics; later values do not imply an executed switch.",
            "pair_root": str(root), "interval": interval, "initial_status": initial["status"],
            "initial_decisions": sorted(initial_decisions, key=lambda item: item["state"]),
            "stored_repeat_references": stored_references,
            "trajectories": sorted(rows, key=lambda row: row["state"])}


def table(report):
    output = io.StringIO()
    output.write(f"Pair root: {report['pair_root']}\n\n")
    output.write("PAIR START-STATE D (separate starting states, not one trajectory)\n")
    writer = csv.writer(output)
    writer.writerow(("state", "step", "d_a", "d_b", "d", "source_path"))
    for item in report["initial_decisions"]:
        writer.writerow(tuple(item[key] for key in
                              ("state", "step", "d_a", "d_b", "d", "source_path")))
    output.write("\n")
    output.write("SR-GC t25 SAVED CHECKPOINT INVENTORY (read-only)\n")
    writer.writerow(("state", "step", "status", "D", "selector", "missing_count",
                     "first_missing", "source_path", "error"))
    for row in report["trajectories"]:
        for point in row["points"]:
            missing = point["missing_files"]
            writer.writerow((row["state"], point["step"], point["status"],
                             point["d"], point["selector"], len(missing),
                             missing[0] if missing else "", point["source_path"],
                             point.get("error", "")))
    output.write("\nDISCOVERED REPEAT REFERENCES (all stored locations)\n")
    writer.writerow(("path", "state", "step", "status", "d_a", "d_b", "d",
                     "done_receipts", "required_receipts", "error"))
    for item in report["stored_repeat_references"]:
        writer.writerow((item["path"], item["state"], item["step"], item["status"],
                         item["d_a"], item["d_b"], item["d"], item["done_receipts"],
                         16, item.get("error", "")))
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
