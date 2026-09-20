"""Export strict paired results and separately labeled saved branch measurements."""

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess

from paper_result_text import write_export


BRANCH_SCOPE = (
    "Independent saved branch measurements, not completed paired comparisons. "
    "Result seal and curve/result binding are checked when present; checkpoint lineage, "
    "prompt coverage, protocol eligibility, and cost ledgers are not independently certified. "
    "Recorded used_gpu_seconds is deployment accounting, not measured cost to target. "
    "No H, crossing, censoring cost, adaptive decision, or paired completion is inferred."
)


def read_source(path, root):
    if not path.resolve().is_relative_to(root):
        raise ValueError("source path escapes experiment root")
    raw = path.read_bytes()
    def reject_constant(value):
        raise ValueError(f"non-finite JSON constant: {value}")
    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"non-finite JSON number: {value}")
        return number
    value = json.loads(raw, parse_constant=reject_constant, parse_float=finite_float)
    if not isinstance(value, dict):
        raise ValueError("source JSON must be an object")
    return value, hashlib.sha256(raw).hexdigest()


def saved_branch_measurements(root):
    """Read only the frozen layout, without invoking recovery or GPU-capable helpers."""
    rows, errors = [], []
    for path in sorted(root.glob("branches/*/states/s*-t*/points/view-*/*/result.json")):
        relative = path.relative_to(root)
        _, selector, _, state, _, view, arm, _ = relative.parts
        match = re.fullmatch(r"s([0-4])-t(25|50|100)", state)
        if not match or view != f"view-{match[2]}":
            continue
        seed, step = map(int, match.groups())
        if selector not in {"on_policy", "cached", "adaptive-on_policy", "adaptive-cached"}:
            continue
        if seed < 3:
            if selector.startswith("adaptive-") or arm != "selection_reduced":
                continue
        elif arm != "selection_full" and not (selector == "on_policy" and arm == "random_full"):
            continue
        try:
            result, result_hash = read_source(path, root)
        except (OSError, ValueError) as exc:
            errors.append({"path": str(relative), "error": str(exc)})
            continue
        row = {"role": "development" if seed < 3 else "test", "seed": seed,
               "prefix_updates": step, "selector_branch": selector, "arm": arm,
               "path": str(path.parent), "validation_scope": BRANCH_SCOPE,
               "eligible_for_paired_comparison": False, "independently_certified": False,
               "source_result": result, "source_result_sha256": result_hash,
               "mean_reward": None, "updates": None, "curve_points": [], "issues": []}
        try:
            seal, seal_hash = read_source(path.with_name("result.sha256.json"), root)
            row["source_result_seal"] = seal
            row["source_result_seal_sha256"] = seal_hash
            if seal != {"sha256": result_hash}:
                raise ValueError("result seal does not match result bytes")
            rewards = result.get("rewards")
            stop = result.get("completed_steps")
            if (result.get("complete") is not True or not isinstance(rewards, dict) or not rewards
                    or any(isinstance(v, bool) or not isinstance(v, (int, float))
                           or not math.isfinite(v) or not 0 <= v <= 1 for v in rewards.values())
                    or isinstance(stop, bool) or not isinstance(stop, int) or stop < step):
                raise ValueError("incomplete or malformed endpoint measurement")
            row.update(mean_reward=statistics.fmean(rewards.values()), updates=stop-step,
                       question_count=len(rewards))
        except (OSError, ValueError) as exc:
            row["issues"].append(str(exc))
        curve_path = path.with_name("curve.json")
        if curve_path.exists():
            try:
                curve, curve_hash = read_source(curve_path, root)
                row.update(source_curve=curve, source_curve_sha256=curve_hash)
                if row["mean_reward"] is None or curve.get("result_sha256") != result_hash:
                    raise ValueError("curve is not bound to an accepted saved endpoint")
                points = curve.get("points")
                if not isinstance(points, dict):
                    raise ValueError("curve points must be an object")
                validated = []
                for checkpoint, point in sorted(points.items(), key=lambda item: int(item[0])):
                    checkpoint = int(checkpoint)
                    if (not isinstance(point, dict) or type(point.get("updates")) is not int
                            or point['updates'] != checkpoint-step or not step <= checkpoint <= result['completed_steps']
                            or type(point.get("reward")) not in (int, float)
                            or not math.isfinite(point['reward']) or not 0 <= point['reward'] <= 1):
                        raise ValueError("malformed saved curve point")
                    if point.get('final') and (checkpoint != result['completed_steps']
                            or not math.isclose(point['reward'], row['mean_reward'], abs_tol=1e-12)):
                        raise ValueError("final curve point differs from saved endpoint")
                    validated.append({"checkpoint_step": checkpoint, **point})
                row["curve_points"] = validated
            except (OSError, ValueError, TypeError) as exc:
                row["issues"].append(str(exc))
        row['status'] = 'saved_branch_measurement' if row['mean_reward'] is not None else 'unverified_source_only'
        rows.append(row)
    return rows, errors


def branch_table(rows):
    output = io.StringIO()
    output.write("\nINDEPENDENT BRANCH MEASUREMENTS\n" + BRANCH_SCOPE + "\n")
    writer = csv.writer(output)
    writer.writerow(['role', 'seed', 'prefix_updates', 'selector_branch', 'arm', 'status',
                     'updates', 'mean_reward_fraction', 'question_count', 'issues'])
    for row in rows:
        writer.writerow([row.get(key) for key in ['role', 'seed', 'prefix_updates', 'selector_branch',
                         'arm', 'status', 'updates', 'mean_reward', 'question_count']] + ['; '.join(row['issues'])])
    output.write("\nSAVED BRANCH CURVE POINTS (reward fractions; checkpoint costs unmeasured here)\n")
    writer.writerow(['role', 'seed', 'prefix_updates', 'selector_branch', 'arm', 'updates', 'reward', 'gpu_seconds'])
    for row in rows:
        for point in row['curve_points']:
            writer.writerow([row[key] for key in ['role', 'seed', 'prefix_updates', 'selector_branch', 'arm']]
                            + [point['updates'], point['reward'], None])
    return output.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    repo = Path(__file__).resolve().parents[1]
    # Regenerate first. Never package a stale report after validation failed.
    result = subprocess.run(["bash", "scripts/run_selector_pair.sh", "report"], cwd=repo,
                            env={**os.environ, "PAIR_ROOT": str(root), "CUDA_VISIBLE_DEVICES": ""})
    if result.returncode:
        raise SystemExit(result.returncode)
    data = json.loads((root / "report.json").read_text())
    data["source_root"] = str(root)
    data["complete"] = not (data["missing_states"] or data["missing_development_states"])
    rows, errors = saved_branch_measurements(root)
    data.update(branch_measurements=rows, branch_measurement_errors=errors,
                branch_measurement_scope=BRANCH_SCOPE)
    write_export("selector-pair", data, (root / "curves.csv").read_text() + branch_table(rows), args.out)


if __name__ == "__main__":
    main()
