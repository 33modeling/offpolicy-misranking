"""Export both datasets, seeds and saved stages into one information result."""

import argparse
import time
from unittest.mock import patch

from srgc_research import information_report as report

from .information_queue import SEEDS
from .result_export import export_lock, save_result, work_root


def compact(result):
    """Keep the information needed to compare selections and actual updates."""
    measurements, identity_fields = [], set()
    for item in result.get("measurements", []):
        endpoint = item.get("endpoint", {})
        identity = endpoint.get("identity", item.get("identity", {}))
        identity_fields.update(identity)
        entry = {key: item[key] for key in ("dataset", "seed", "stage", "source") if key in item}
        entry.update(identity=identity, configuration=endpoint.get("configuration", item.get("configuration", {})))
        costs = item.get("costs", endpoint.get("cost_receipts"))
        if costs:
            entry["costs"] = {key: costs[key] for key in ("complete", "total_gpu_seconds", "known_gpu_seconds") if key in costs}
            entry["costs"]["unfinished_phase_count"] = costs.get("unfinished_phase_count", len(costs.get("unfinished_phases", [])))
        measurements.append(entry)
    identity_fields.difference_update({"dataset", "seed", "stage"})

    def row_data(row, *, candidate=False):
        excluded = identity_fields | {"prompt", "raw_samples", "raw_response_source", "weight_tensor_source"}
        if candidate:
            excluded |= {"question", "answer"}
        value = {key: value for key, value in row.items() if key not in excluded}
        if not candidate and "question" in row and not value.get("question"):
            value["question"] = row.get("prompt")
        if "raw_samples" in row:
            value["response_updates"] = [{key: sample[key] for key in (
                "response", "reward", "completion_tokens", "mean_logp_change") if key in sample}
                for sample in row["raw_samples"]]
        return value

    value = {key: item for key, item in result.items() if key not in {
        "measurements", "selected_problems", "batch_updates", "candidate_information", "parameter_updates"}}
    value.update(export_schema="srgc-information-results-v2", format="compact", measurements=measurements,
                 selected_problems=[row_data(row) for row in result.get("selected_problems", [])],
                 batch_updates=[row_data(row) for row in result.get("batch_updates", [])],
                 candidate_information=[row_data(row, candidate=True) for row in result.get("candidate_information", [])])
    return value


def export(dataset="all", *, work=None):
    work = work_root() if work is None else work.resolve()
    with export_lock("information", work):
        return _export(dataset, work=work)


def _export(dataset, *, work):
    root = work / "selection-information"
    names = ("math", "mbpp") if dataset == "all" else (dataset,)
    expected = {(name, seed, 0) for name in names for seed in SEEDS}
    locations = {(name, seed, 0): root / name / f"seed-{seed}" / "t0" for name, seed, _ in expected}
    for name in names:
        for folder in sorted((root / name).glob("seed-*/t*")):
            seed_text, stage_text = folder.parent.name.removeprefix("seed-"), folder.name.removeprefix("t")
            if not folder.is_dir() or not seed_text.isdigit() or not stage_text.isdigit():
                continue
            if not folder.resolve().is_relative_to(root.resolve()):
                raise ValueError("measurement path escapes its source directory")
            locations[name, int(seed_text), int(stage_text)] = folder
    result = {"export_schema": "srgc-information-results-v1", "protocol": report.PROTOCOL,
              "experiment": "information", "generated": time.time(), "complete": False,
              "datasets": {}, "measurements": [], "selected_problems": [], "batch_updates": [],
              "candidate_information": [], "parameter_updates": [], "information_summary": [],
              "pending": [], "errors": [],
              "interpretation": "same-state observations; response samples and methods are not independent training seeds"}
    completed = set()
    for (name, seed, stage), folder in sorted(locations.items()):
        task = {"dataset": name, "seed": seed, "stage": stage, "source": str(folder)}
        if not (folder / "endpoint.json").is_file():
            result["pending"].append({**task, "status": "partial" if (folder / "manifest.json").exists() else "not-started"})
            continue
        try:
            print(f"READ information {name} seed={seed} stage={stage}", flush=True)
            endpoint, phases = report.read_measurement(folder)
            if any(endpoint["identity"].get(k) != v for k, v in (("dataset", name), ("seed", seed), ("stage", stage))):
                raise ValueError("measurement identity differs from its dataset/seed/stage directory")
            # Reuse the fully audited in-memory phase data for the flattened
            # tables, avoiding a second read of large tensor artifacts.
            with patch.object(report, "read_measurement", return_value=(endpoint, phases)):
                selected, batches, candidates, parameters = report.measurement_rows(folder)
            manifest = report.read_object(folder / "manifest.json")
            measurement = {**task, "manifest": manifest, "endpoint": endpoint, "phases": phases}
            costs = folder / "costs.json"
            if costs.is_file():
                measurement["costs"] = report.read_object(costs)
                report.finite_tree(measurement["costs"])
            result["measurements"].append(measurement)
            for field, rows in (("selected_problems", selected), ("batch_updates", batches),
                                ("candidate_information", candidates), ("parameter_updates", parameters)):
                result[field].extend(rows)
            completed.add((name, seed, stage))
        except (OSError, ValueError, KeyError, TypeError) as error:
            result["errors"].append({**task, "error": str(error)})
    result["information_summary"] = report.information_summary(result["candidate_information"])
    result["coverage"] = {"completed_initial_measurements": len(expected & completed),
                          "planned_initial_measurements": len(expected),
                          "completed_saved_measurements": len(completed)}
    result["complete"] = expected <= completed and not result["pending"] and not result["errors"]
    for name in names:
        done = sorted(seed for dataset_name, seed, stage in completed if dataset_name == name and stage == 0)
        result["datasets"][name] = {"completed_initial_seeds": done, "planned_initial_seeds": sorted(SEEDS),
                                    "stages": sorted({stage for dataset_name, _, stage in locations if dataset_name == name})}
    save_result("information", compact(result), work=work)
    coverage = result["coverage"]
    print(f"INFORMATION {coverage['completed_initial_measurements']}/{coverage['planned_initial_measurements']} "
          f"initial measurements; saved stages={len(completed)}; "
          f"{'COMPLETE' if result['complete'] else 'INCOMPLETE'}", flush=True)
    for error in result["errors"]:
        print(f"ERROR {error['dataset']}.seed-{error['seed']}.t{error['stage']}: {error['error']}", flush=True)
    return int(bool(result["errors"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("math", "mbpp", "all"))
    parser.add_argument("action", choices=("results",))
    args = parser.parse_args(argv)
    try:
        return export(args.dataset)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"ERROR: {error}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
