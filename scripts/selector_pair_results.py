"""Export strict paired results and separately labeled saved branch measurements."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import uuid

from paper_result_text import write_export


BRANCH_SCOPE = (
    "Independent saved branch measurements, not completed paired comparisons. "
    "Result seal and curve/result binding are checked when present; checkpoint lineage, "
    "prompt coverage, protocol eligibility, and cost ledgers are not independently certified. "
    "Recorded used_gpu_seconds is deployment accounting, not measured cost to target. "
    "No H, crossing, censoring cost, adaptive decision, or paired completion is inferred."
)

OBSERVATION_LIMIT = 196608
OBSERVATION_SCOPE = (
    "Read-only, unverified execution metadata, separate from scientific results. "
    "Recorded RUN/DONE and timestamps are not proof of current process liveness or paired completion. "
    "Clocks may differ between servers; no heartbeat age or current-live classification is inferred. "
    "File presence and shard receipts are not independent seal/lineage validation. "
    "Adaptive storage candidates may include both selectors, but only one is a planned test branch."
)


def execution_observations(root, measurements):
    """Bounded metadata only: never recover work, load a model or probe a GPU."""
    data = {'scope': OBSERVATION_SCOPE, 'independently_certified': False,
            'planned_branches': {'total': 42, 'development': 18, 'test': 24},
            'saved_endpoint_files': 0,
            'accepted_endpoint_measurements': sum(row['mean_reward'] is not None for row in measurements),
            'saved_curve_files': 0, 'workers': [], 'progress': [], 'branches': [],
            'limits': {'json_read_bytes': 65536, 'worker_files_scanned': 2048,
                       'worker_records': 32, 'progress_records': 64,
                       'branch_records': 48, 'checkpoint_directories_per_branch': 64,
                       'section_bytes': OBSERVATION_LIMIT},
            'omitted_counts_are_lower_bounds': True,
            'omitted': {'workers': 0, 'progress': 0, 'checkpoint_directories': 0}, 'errors': []}

    def read(path):
        try:
            if not path.resolve().is_relative_to(root):
                raise ValueError('metadata path escapes experiment root')
            with path.open('rb') as handle:
                raw = handle.read(65537)
            if len(raw) > 65536:
                raise ValueError('metadata exceeds 65536 bytes')
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError('metadata is not an object')
            return value
        except FileNotFoundError:
            return {}
        except (OSError, ValueError, RuntimeError) as exc:
            if len(data['errors']) < 16:
                data['errors'].append({'path': str(path.relative_to(root))[:384], 'error': str(exc)[:256]})
            return {}

    def scalars(value, keys):
        return {key: item[:512] if isinstance(item, str) else item
                for key in keys if (item := value.get(key)) is not None
                and (type(item) in (str, bool, int) or type(item) is float and math.isfinite(item))}

    def present(path):
        try:
            return path.resolve().is_relative_to(root) and path.is_file()
        except (OSError, RuntimeError):
            return False

    workers = []
    try:
        with os.scandir(root / 'queue-workers') as entries:
            for index, entry in enumerate(entries):
                if index >= 2048:
                    data['worker_scan_truncated'] = True
                    break
                if entry.name.endswith('.json') and not entry.is_symlink() and entry.is_file():
                    workers.append((entry.stat().st_mtime_ns, Path(entry.path)))
    except OSError:
        pass
    for _, path in sorted(workers, reverse=True)[:32]:
        value = read(path)
        row = {'path': str(path.relative_to(root)), **scalars(value, (
            'worker', 'host', 'pid', 'stage', 'state', 'task', 'updated', 'protocol_id',
            'verified_states', 'total_states', 'verified_branches', 'total_branches',
            'queue_wait_wall_seconds'))}
        failures = value.get('failures')
        if isinstance(failures, list):
            row['recorded_failure_count'] = len(failures)
            row['failures'] = [scalars(item, ('task', 'state', 'error'))
                               for item in failures[:3] if isinstance(item, dict)]
        data['workers'].append(row)
    data['omitted']['workers'] = max(0, len(workers) - 32)
    try:
        source = str(Path(__file__).resolve().parents[1] / 'src')
        if source not in sys.path:
            sys.path.insert(0, source)
        from selector_pair_gpu import pair_progress
        progress = pair_progress(root)
        for _, path, value in progress[:64]:
            data['progress'].append({'path': str(path.relative_to(root))[:384], **scalars(value, (
                'state', 'phase', 'event_id', 'host', 'pid', 'worker_id', 'updated',
                'seconds', 'timeout', 'exit_code', 'gpus'))})
        data['omitted']['progress'] = max(0, len(progress) - 64)
        data['progress_scan_bounded'] = True
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        data['errors'].append({'path': 'progress', 'error': str(exc)[:256]})

    def shards(directory):
        return [shard for shard in range(4) if present(directory / f'shard-{shard}.done.json')]

    for seed in range(5):
        for step in (25, 50, 100):
            candidates = [('on_policy', 'selection_reduced'), ('cached', 'selection_reduced')]
            if seed >= 3:
                candidates = [(name, 'selection_full') for name in
                              ('on_policy', 'cached', 'adaptive-on_policy', 'adaptive-cached')]
                candidates.append(('on_policy', 'random_full'))
            for selector, arm in candidates:
                point = root / f'branches/{selector}/states/s{seed}-t{step}/points/view-{step}'
                directory = point / arm
                try:
                    contained = directory.is_dir() and directory.resolve().is_relative_to(root)
                except (OSError, RuntimeError):
                    contained = False
                if not contained:
                    continue
                row = {'path': str(directory.relative_to(root)),
                       'result_file': present(directory / 'result.json'),
                       'curve_file': present(directory / 'curve.json'),
                       'endpoint_shards_done': shards(directory / 'evaluation'),
                       'parent_shards_done': shards(point / 'curve-parent'), 'curve_checkpoints': []}
                data['saved_curve_files'] += row['curve_file']
                data['saved_endpoint_files'] += row['result_file']
                policy = read(directory / 'policy/policy_train.json')
                row['policy'] = scalars(policy, ('completed_steps', 'start_step', 'training_objective'))
                for name in ('pair-attempt.json', 'failure.json'):
                    value = read(directory / name)
                    if value:
                        row[name] = scalars(value, ('error', 'state', 'host', 'pid', 'time', 'updated'))
                checkpoint_dirs = list(itertools.islice((directory / 'curve').glob('step-*'), 65))
                data['omitted']['checkpoint_directories'] += max(0, len(checkpoint_dirs) - 64)
                for checkpoint in sorted(checkpoint_dirs[:64]):
                    if re.fullmatch(r'step-[0-9]+', checkpoint.name):
                        row['curve_checkpoints'].append({'step': int(checkpoint.name[5:]),
                                                        'shards_done': shards(checkpoint)})
                row['curve_shards_done_count'] = sum(len(item['shards_done']) for item in row['curve_checkpoints'])
                archived = list(itertools.islice((directory / 'policy/curve-checkpoints').glob('step-*'), 65))
                row['archived_checkpoint_steps'] = sorted(int(path.name[5:]) for path in archived[:64]
                    if re.fullmatch(r'step-[0-9]+', path.name) and present(path / 'adapter_model.safetensors'))
                data['omitted']['checkpoint_directories'] += max(0, len(archived) - 64)
                data['branches'].append(row)
    # Keep the diagnostic section small even with unusually long metadata fields.
    while len(json.dumps(data, ensure_ascii=True, separators=(',', ':')).encode()) > OBSERVATION_LIMIT:
        key = next(key for key in ('progress', 'workers', 'branches') if data[key])
        data[key].pop()
        data['omitted'][key] = data['omitted'].get(key, 0) + 1
    return data


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
    return {'version': 'selector-pair-results/v3', 'git_commit': commit,
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
                exporter=exporter_metadata(repo), execution_observations=execution_observations(root, rows))
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
