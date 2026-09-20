"""Export validated RLOO measurements without requiring the matrix to finish.

Kept outside src so reporting updates do not change frozen training code hashes.
Partial exports never publish the worker's canonical results.json.
"""

from __future__ import annotations

import argparse
from contextvars import ContextVar
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import re
import subprocess

import numpy as np

import rloo_experiment as frozen_experiment
from paper_result_text import write_export


EXPORTER_VERSION = "rloo-paper-results/v2"
# This standalone Qwen status entry point is never imported by the scientific
# modules. Keep the exception report-only and fail closed if that changes.
REPORT_DISPLAY_FILES = frozenset({"src/matrix_status.py"})
_display_isolation = ContextVar("rloo_report_display_isolation", default=None)


def display_is_isolated(name):
    module = Path(name).stem
    for source in (frozen_experiment.ROOT / "src").rglob("*.py"):
        if source.relative_to(frozen_experiment.ROOT).as_posix() == name:
            continue
        if re.search(rf"\b{re.escape(module)}\b", source.read_text()):
            return False
    return True


def display_changes(recorded):
    changes = {}
    for name in REPORT_DISPLAY_FILES:
        previous = recorded.get(name)
        if not isinstance(previous, str) or not re.fullmatch(r"[0-9a-f]{64}", previous):
            continue
        current = frozen_experiment.ed.digest(frozen_experiment.ROOT / name)
        if previous == current:
            continue
        isolation = _display_isolation.get()
        if isolation is None:
            isolated = display_is_isolated(name)
        else:
            key = (str(frozen_experiment.ROOT.resolve()), name)
            if key not in isolation:
                isolation[key] = display_is_isolated(name)
            isolated = isolation[key]
        if isolated:
            changes[name] = {"frozen_sha256": previous, "runtime_sha256": current}
    return changes


def exporter_provenance():
    root = frozen_experiment.ROOT
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                                capture_output=True, text=True, timeout=5, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {"version": EXPORTER_VERSION, "git_commit": commit,
            "script_sha256": frozen_experiment.ed.digest(Path(__file__)),
            "validation_module_sha256": frozen_experiment.ed.digest(Path(frozen_experiment.__file__)),
            "report_only_display_files": sorted(REPORT_DISPLAY_FILES)}


def code_diagnostics(out):
    ed = frozen_experiment.ed
    try:
        contract = ed.read(out / "experiment.json")
        recorded = contract["code_hashes"]
        if not isinstance(recorded, dict):
            raise ValueError("code_hashes is not a mapping")
        mismatches = {}
        for name, previous in recorded.items():
            item = {"frozen_sha256": previous}
            try:
                path = frozen_experiment.ROOT / name
                if path.resolve().is_relative_to(frozen_experiment.ROOT.resolve()):
                    item["runtime_sha256"] = ed.digest(path)
                else:
                    raise ValueError("code path is outside the repository")
            except (OSError, ValueError) as exc:
                item.update(runtime_sha256=None, error=str(exc))
            if item["runtime_sha256"] != previous:
                mismatches[name] = item
        return {"mismatches": mismatches,
                "scope": "Recorded versus current code hashes; not a validation override."}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"error": str(exc)}


def reporting_experiment():
    # Isolate the report's compatibility rule: importing this exporter must
    # never relax the validator used by training in the same Python process.
    spec = importlib.util.spec_from_file_location(
        "_rloo_report_validation", frozen_experiment.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def reviewed_code_changes(recorded):
        display = display_changes(recorded)
        return frozen_experiment.reviewed_code_changes(
            {name: digest for name, digest in recorded.items() if name not in display})

    module.reviewed_code_changes = reviewed_code_changes
    return module


experiment = reporting_experiment()


def point_report(out):
    # Nested policy/shard validation retains every scientific hash check but
    # only needs one display-dependency scan per point, not one per check.
    token = _display_isolation.set({})
    try:
        return _point_report(out)
    finally:
        _display_isolation.reset(token)


def _point_report(out):
    c, _ = experiment.validate(out)
    values, evaluations = {}, []
    for arm in ("before", *experiment.ARMS):
        rewards = [[] for _ in range(c["eval_n"])]
        completed_shards = []
        for shard in range(4):
            if not (out / arm / "evaluation" / f"shard-{shard}.done.json").is_file():
                continue
            # Only sealed shards count; live rollout files are not evidence.
            for row in experiment.checked_rows(out, arm, shard):
                rewards[row["prompt_idx"]].append(row["reward"])
            completed_shards.append(shard)
        measured = {str(i): float(np.mean(r)) for i, r in enumerate(rewards) if r}
        complete = len(completed_shards) == 4
        if complete:
            values[arm] = np.array([measured[str(i)] for i in range(c["eval_n"])])
        cost_path = out / arm / "cost.jsonl"
        costs = {"status": "missing", "events": []}
        if cost_path.is_file():
            try:
                costs = {"status": "snapshot", "events": [json.loads(line)
                         for line in cost_path.read_text().splitlines() if line.strip()],
                         "note": "Raw metered events, not a certified total; open events have unknown final cost."}
            except (OSError, ValueError) as exc:
                costs = {"status": "unreadable", "error": str(exc), "events": []}
        evaluations.append({
            "arm": arm, "complete": complete, "completed_shards": completed_shards,
            "missing_shards": [s for s in range(4) if s not in completed_shards],
            "measured_prompts": len(measured), "expected_prompts": c["eval_n"],
            "prompt_rewards": measured,
            "observed_mean_reward": float(np.mean(list(measured.values()))) if measured else None,
            "cost_ledger": costs,
        })
    rows = []
    for arm in experiment.ARMS:
        if arm not in values:
            continue
        row = {"arm": arm, "mean_reward": float(values[arm].mean()), "missing_references": []}
        for reference in ("before", "random", "passrate_beta"):
            if reference not in values:
                row["missing_references"].append(reference)
                continue
            delta = values[arm] - values[reference]
            lo, hi = experiment.ed.paired_interval(delta, c["source"]["seed"])
            row["vs_" + reference] = {"mean": float(delta.mean()), "lower": lo, "upper": hi}
        rows.append(row)
    missing = [e["arm"] for e in evaluations if not e["complete"]]
    return {"seed": c["source"]["seed"], "drift": c["source"]["drift"],
            "experiment_sha256": experiment.ed.digest(out / "experiment.json"),
            "report_display_code_changes": display_changes(c.get("code_hashes", {})),
            "status": "incomplete" if missing else "complete", "missing_arms": missing,
            "rows": rows, "evaluations": evaluations}


def report(root):
    if not root.is_dir():
        raise ValueError(f"no RLOO root: {root}")
    points = []
    for drift, seed in experiment.POINTS:
        out = root / f"math500-d{drift}" / f"s{seed}"
        point = {"seed": seed, "drift": drift, "path": str(out), "rows": []}
        if not (out / "experiment.json").is_file():
            point["status"] = "unprepared"
        else:
            try:
                point.update(point_report(out))
            except (ValueError, OSError, KeyError) as exc:
                point.update(status="invalid", error=str(exc), code_diagnostics=code_diagnostics(out))
        points.append(point)
    return {"schema": "rloo-progress-report/v1", "scope": experiment.SCOPE,
            "exporter": exporter_provenance(),
            "created_at": datetime.now(timezone.utc).isoformat(), "root": str(root),
            "complete": all(p["status"] == "complete" for p in points), "points": points,
            "comparison_scope": "Only fully evaluated arms are compared on the complete frozen prompt set. "
                                "Partial shard means are descriptive, not final benchmark scores.",
            "interval_scope": "Paired prompt bootstrap, conditional on this training seed; "
                              "not across-seed uncertainty."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    value = report(root)
    lines = ["drift\tseed\tstatus\tarm\tmean_reward\tvs_cached\tvs_cached_lower\tvs_cached_upper"]
    for point in value["points"]:
        for row in point["rows"]:
            cached = row.get("vs_passrate_beta", {})
            lines.append("\t".join(str(v) for v in (
                point["drift"], point["seed"], point["status"], row["arm"], row["mean_reward"],
                cached.get("mean", "NA"), cached.get("lower", "NA"), cached.get("upper", "NA"))))
    write_export("rloo", value, "\n".join(lines), args.out)
    if any(p["status"] == "invalid" for p in value["points"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
