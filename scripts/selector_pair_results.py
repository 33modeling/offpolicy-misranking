"""Export strict paired results and separately labeled saved branch measurements."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import uuid

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


def exporter_metadata(repo):
    try:
        git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=repo,
                             capture_output=True, text=True, timeout=3)
        commit = git.stdout.strip() if git.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        commit = None
    return {'version': 'selector-pair-results/v2', 'git_commit': commit,
            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'created_at': datetime.now(timezone.utc).isoformat(), 'export_id': uuid.uuid4().hex}


def run_report(root, repo, timeout):
    """Bound only this export's CPU report process group, never existing workers."""
    command = ['bash', 'scripts/run_selector_pair.sh', 'report']
    process = subprocess.Popen(command, cwd=repo,
                               env={**os.environ, 'PAIR_ROOT': str(root), 'CUDA_VISIBLE_DEVICES': ''},
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                               start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = process.communicate()
        return subprocess.CompletedProcess(command, 124, stdout,
                                           f'report timed out after {timeout:g} seconds\n{stderr}')


def empty_paired_report():
    return {'rows': [], 'development_rows': [], 'complete': False,
            'missing_states': [f's{s}-t{t}' for s in (3, 4) for t in (25, 50, 100)],
            'missing_development_states': [f's{s}-t{t}' for s in (0, 1, 2) for t in (25, 50, 100)],
            'summary': None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--report-timeout", type=float, default=60,
                        help="Maximum seconds for strict paired validation (default: 60)")
    args = parser.parse_args()
    if not math.isfinite(args.report_timeout) or args.report_timeout <= 0:
        parser.error('--report-timeout must be finite and positive')
    root = args.root.resolve()
    repo = Path(__file__).resolve().parents[1]
    data, curves, exit_code = empty_paired_report(), '', 0
    validation = {'status': 'not_run_no_published_paired_results', 'error': None}
    published = [root / role / f's{s}-t{t}/result.json'
                 for role, seeds in [('development', (0, 1, 2)), ('test', (3, 4))]
                 for s in seeds for t in (25, 50, 100)]
    if not root.is_dir():
        exit_code = 2
        validation = {'status': 'failed', 'error': 'experiment root does not exist'}
    elif any(path.is_file() for path in published):
        try:
            result = run_report(root, repo, args.report_timeout)
            if result.returncode:
                exit_code = result.returncode if result.returncode > 0 else 1
                validation = {'status': 'failed', 'returncode': result.returncode,
                              'error': f'strict paired report failed (rc={result.returncode})',
                              'stdout_tail': result.stdout[-8000:], 'stderr_tail': result.stderr[-8000:]}
            else:
                fresh, _ = read_source(root / 'report.json', root)
                if any(not isinstance(fresh.get(key), list) for key in
                       ('rows', 'missing_states', 'missing_development_states')):
                    raise ValueError('malformed strict paired report')
                curves = (root / 'curves.csv').read_text()
                data = fresh
                data['complete'] = not (data['missing_states'] or data['missing_development_states'])
                validation = {'status': 'validated', 'error': None, 'returncode': 0}
        except (OSError, ValueError) as exc:
            data, curves, exit_code = empty_paired_report(), '', 2
            validation = {'status': 'failed', 'error': str(exc)}
    data["source_root"] = str(root)
    rows, errors = saved_branch_measurements(root)
    data.update(branch_measurements=rows, branch_measurement_errors=errors,
                branch_measurement_scope=BRANCH_SCOPE, paired_validation=validation,
                exporter=exporter_metadata(repo))
    if not exit_code and (errors or any(row['issues'] for row in rows)):
        exit_code = 2
    data['export_exit_code'] = exit_code
    header = 'EXPORTER ' + json.dumps(data['exporter'], sort_keys=True) + '\n'
    header += 'PAIRED VALIDATION ' + json.dumps(validation, sort_keys=True) + '\n'
    header += f'CURRENT SAVED BRANCHES {len(rows)}; ERRORS {len(errors)}\n'
    write_export("selector-pair", data, header + curves + branch_table(rows), args.out)
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
