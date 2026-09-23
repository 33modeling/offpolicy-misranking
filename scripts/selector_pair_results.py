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
import stat
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
RESULT_SCHEMA = 'offpolicy-selected-prefix-switch/v1'
OBSERVATION_SCOPE = (
    "Read-only, unverified execution metadata, separate from scientific results. "
    "Recorded RUN/DONE and timestamps are not proof of current process liveness or paired completion. "
    "Clocks may differ between servers; no heartbeat age or current-live classification is inferred. "
    "File presence and shard receipts are not independent seal/lineage validation. "
    "Adaptive storage candidates may include both selectors, but only one is a planned test branch."
)


def schedule_provenance(root):
    """Export the explicit scheduling amendment, separate from measurements."""
    name = 'pair-parallel-controls-runtime.json'
    data = {'status': 'not_recorded', 'path': name, 'independently_certified': False,
            'scope': 'Operational schedule amendment: 18 fixed held-out controls may run alongside '
                     'development; six adaptive branches still require frozen development decisions. '
                     'This receipt does not certify measurements, infer adaptive choices, or establish paired completion.',
            'source_receipt': None, 'source_receipt_sha256': None, 'error': None}
    path = root / name
    if not path.exists() and not path.is_symlink():
        return data
    data['status'] = 'unverified'
    def reject_constant(value):
        raise ValueError(f'non-finite schedule metadata: {value}')
    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            reject_constant(value)
        return number
    try:
        records = []
        for source in (path, root / 'pair.json'):
            if source.resolve() != source:
                raise ValueError('schedule provenance refuses symlinked metadata')
            with os.fdopen(os.open(source, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW), 'rb') as handle:
                if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                    raise ValueError('schedule metadata is not a regular file')
                raw = handle.read(65537)
            if len(raw) > 65536:
                raise ValueError('schedule metadata exceeds 64 KiB read limit')
            value = json.loads(raw, parse_constant=reject_constant, parse_float=finite_float)
            if not isinstance(value, dict):
                raise ValueError('schedule metadata is not an object')
            records.append(value)
            if source == path:
                data.update(source_receipt=value, source_receipt_sha256=hashlib.sha256(raw).hexdigest())
        from selector_pair_parallel import validate_receipt
        validate_receipt(root, records[1])
        _, latest_digest = read_source(path, root)
        if latest_digest != data['source_receipt_sha256']:
            raise ValueError('schedule receipt changed during export')
        data['status'] = 'validated_current_runtime_receipt'
        if (root / 'pair-sr-gc-runtime.json').is_file():
            data['scope'] = ('Operational fixed-control schedule retained; the SR-GC amendment replaces '
                             'the legacy development-label barrier. Adaptive requires frozen current-policy '
                             'SR-GC decisions, not fitted H labels. Neither receipt establishes paired completion.')
    except (ImportError, OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError) as exc:
        data['error'] = str(exc)
    return data


def cost_provenance(root):
    """Expose recovery annotations without certifying or changing any cost."""
    data = {'scope': 'Read-only ledger annotations, not independent cost certification. '
            'A recovery without an atomic finish receipt is reconstructed accounting, '
            'not a directly measured finish. Do not describe affected totals as wholly measured.',
            'recovered_events': [], 'reconstructed_events': 0, 'inspection_complete': True,
            'omitted_events': 0, 'errors': []}
    paths = set()
    paths.update(root / 'sr-gc' / f's{seed}-t{step}' / 'cost.jsonl'
                 for seed in (3, 4) for step in (25, 50, 100))
    for name in ('on_policy', 'cached', 'adaptive-on_policy', 'adaptive-cached'):
        for seed in range(5):
            for step in (25, 50, 100):
                point = root / f'branches/{name}/states/s{seed}-t{step}/points/view-{step}'
                paths.add(point / 'curve-parent/cost.jsonl')
                for arm in ('selection_reduced', 'selection_full', 'random_full'):
                    paths.update((point / arm / 'cost.jsonl', point / arm / 'curve/cost.jsonl'))
    for path in sorted(paths):
        try:
            if not path.resolve().is_relative_to(root):
                raise ValueError('cost ledger escapes experiment root')
            with os.fdopen(os.open(path, os.O_RDONLY | os.O_NONBLOCK), 'rb') as handle:
                if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                    raise ValueError('cost ledger is not a regular file')
                raw = handle.read(1048577)
            if len(raw) > 1048576:
                raise ValueError('cost ledger exceeds 1 MiB provenance read limit')
            seen = set()
            for line in raw.splitlines():
                if not line.strip():
                    continue
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise ValueError('cost event is not an object')
                recovery = event.get('recovery')
                if event.get('state') != 'finished' or not recovery:
                    continue
                if not isinstance(recovery, dict) or not isinstance(event.get('event_id'), str):
                    raise ValueError('invalid recovery annotation')
                if event['event_id'] in seen:
                    continue
                seen.add(event['event_id'])
                kind = recovery.get('kind')
                reconstructed = kind != 'atomic_finish_receipt'
                data['reconstructed_events'] += reconstructed
                if len(data['recovered_events']) >= 128:
                    data['omitted_events'] += 1
                    data['inspection_complete'] = False
                    continue
                data['recovered_events'].append({
                    'path': str(path.relative_to(root)), 'event_id': event['event_id'][:128],
                    'ledger_sha256': hashlib.sha256(raw).hexdigest(),
                    'evidence_kind': str(kind)[:128], 'reconstructed': reconstructed})
        except FileNotFoundError:
            continue
        except (OSError, ValueError, RuntimeError) as exc:
            data['inspection_complete'] = False
            if len(data['errors']) < 16:
                data['errors'].append({'path': str(path.relative_to(root)), 'error': str(exc)[:256]})
    return data


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
            with os.fdopen(os.open(path, os.O_RDONLY | os.O_NONBLOCK), 'rb') as handle:
                if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                    raise ValueError('metadata is not a regular file')
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
            'queue_wait_wall_seconds', 'schedule', 'fixed_control_branches'))}
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
    try:
        resolved = path.resolve()
    except RuntimeError as exc:
        raise ValueError('source path contains a symlink loop') from exc
    if not resolved.is_relative_to(root):
        raise ValueError("source path escapes experiment root")
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NONBLOCK), 'rb') as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise ValueError('source metadata is not a regular file')
        raw = handle.read(8 * 1024 * 1024 + 1)
    if len(raw) > 8 * 1024 * 1024:
        raise ValueError('source metadata exceeds 8 MiB read limit')
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
            if (result.get('schema') != RESULT_SCHEMA or result.get("complete") is not True
                    or not isinstance(rewards, dict) or not rewards
                    or any(isinstance(v, bool) or not isinstance(v, (int, float))
                           or not math.isfinite(v) or not 0 <= v <= 1 for v in rewards.values())
                    or isinstance(stop, bool) or not isinstance(stop, int) or stop < step):
                raise ValueError("incomplete or malformed endpoint measurement")
            row.update(mean_reward=statistics.fmean(rewards.values()), updates=stop-step,
                       question_count=len(rewards))
        except (OSError, ValueError) as exc:
            row["issues"].append(str(exc))
        curve_path = path.with_name("curve.json")
        if curve_path.exists() or curve_path.is_symlink():
            try:
                curve, curve_hash = read_source(curve_path, root)
                row.update(source_curve=curve, source_curve_sha256=curve_hash)
                if (row["mean_reward"] is None or curve.get('schema') != RESULT_SCHEMA
                        or curve.get("result_sha256") != result_hash):
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
        # Raw diagnostic payloads are not measurements. Keep their provenance
        # without allowing one oversized field to suppress every partial row.
        for field, source_path in (('source_result', path), ('source_curve', curve_path),
                                   ('source_result_seal', path.with_name('result.sha256.json'))):
            if field not in row:
                continue
            encoded = json.dumps(row[field], ensure_ascii=True, separators=(',', ':')).encode()
            if len(encoded) <= 8192:
                continue
            try:
                source_bytes = source_path.stat().st_size
            except OSError:
                source_bytes = None
            row[field + '_reference'] = {'path': str(source_path.relative_to(root)),
                'sha256': row[field + '_sha256'], 'source_bytes': source_bytes,
                'raw_omitted': True, 'reason': 'oversized unvalidated source metadata'}
            del row[field]
            if field == 'source_result' and row['mean_reward'] is not None:
                row['question_rewards'] = result['rewards']
            if field == 'source_curve':
                row['curve_points'] = [{key: value for key, value in point.items()
                    if key in {'checkpoint_step', 'updates', 'reward'}
                    or (type(value) in (int, float, bool) or value is None)}
                    for point in row['curve_points']]
        rows.append(row)
    return rows, errors


COMPLETION_GROUPS = (
    ('development', 'development',
     tuple((selector, 'selection_reduced', seed, step) for seed in (0, 1, 2) for step in (25, 50, 100)
           for selector in ('on_policy', 'cached'))),
    ('test_fixed_controls', 'test fixed controls',
     tuple((selector, arm, seed, step) for seed in (3, 4) for step in (25, 50, 100)
           for selector, arm in (('on_policy', 'selection_full'), ('cached', 'selection_full'),
                                 ('on_policy', 'random_full')))),
    ('test_adaptive', 'test adaptive',
     tuple(('adaptive', 'selection_full', seed, step) for seed in (3, 4) for step in (25, 50, 100))),
)


def completion_label(selector, arm, seed, step):
    name = 'random' if arm == 'random_full' else selector
    return f'{name} s{seed}-t{step}'


BUDGET_FAILURE_MARKERS = ('allocation exhausted', 'original allocation exceeded')


def attempt_failures(observations):
    """Most recent recorded worker failure text per branch path.

    execution_observations lists workers newest first by file mtime, so the first
    record seen for a task wins. Only the exported failures of the newest 32
    workers are visible, so this is a lower bound, not a complete history.
    """
    failures = {}
    for worker in (observations or {}).get('workers') or []:
        for failure in reversed(worker.get('failures') or []):
            task, error = failure.get('task'), failure.get('error')
            if isinstance(task, str) and isinstance(error, str):
                failures.setdefault(task, error)
    return failures


def branch_completion(rows, observations=None, srgc=None):
    """Designed-branch progress from saved endpoints only; paired validation is separate."""
    chosen = {item.get('state'): item.get('selector') for item in (srgc or {}).get('decisions') or []
              if item.get('selector') in ('on_policy', 'cached')}
    failures = attempt_failures(observations)
    saved, not_chosen = {}, []
    for row in rows:
        selector = row['selector_branch']
        state = f"s{row['seed']}-t{row['prefix_updates']}"
        if selector.startswith('adaptive-'):
            if state in chosen and selector != 'adaptive-' + chosen[state]:
                not_chosen.append(f"{selector} {state}")
                continue
            selector = 'adaptive'
        key = (selector, row['arm'], row['seed'], row['prefix_updates'])
        clean = not row['issues'] and row['mean_reward'] is not None
        previous = saved.get(key)
        if previous is None or (clean and (previous['issues'] or previous['mean_reward'] is None)):
            saved[key] = row
    groups, planned_total, endpoint_total = {}, 0, 0
    for key, _, planned in COMPLETION_GROUPS:
        group = {'planned': len(planned), 'endpoints': 0, 'curves': 0, 'remaining': [],
                 'budget_exhausted_needs_review': [], 'failed_attempt': [],
                 'endpoint_without_curve': [], 'endpoint_with_issues': []}
        for branch in planned:
            selector, arm, seed, step = branch
            label, row = completion_label(*branch), saved.get(branch)
            if row is None:
                if selector == 'adaptive':
                    names = ([f"adaptive-{chosen[f's{seed}-t{step}']}"] if f's{seed}-t{step}' in chosen
                             else ['adaptive-on_policy', 'adaptive-cached'])
                else:
                    names = [selector]
                paths = [f"branches/{name}/states/s{seed}-t{step}/points/view-{step}/{arm}" for name in names]
                error = next((failures[path] for path in paths if path in failures), None)
                if error and any(marker in error for marker in BUDGET_FAILURE_MARKERS):
                    group['budget_exhausted_needs_review'].append(label)
                elif error:
                    group['failed_attempt'].append(label)
                else:
                    group['remaining'].append(label)
            elif row['issues'] or row['mean_reward'] is None:
                group['endpoint_with_issues'].append(label)
            else:
                group['endpoints'] += 1
                if row['curve_points']:
                    group['curves'] += 1
                else:
                    group['endpoint_without_curve'].append(label)
        groups[key] = group
        planned_total += group['planned']
        endpoint_total += group['endpoints']
    return {'scope': 'Saved endpoints per designed branch, independent of paired validation. '
                     'An endpoint is not a paired comparison, an H value or a checkpoint cost. '
                     'Worker failures are a lower bound; omitted worker records are not inspected.',
            'planned': planned_total, 'endpoints': endpoint_total, 'groups': groups,
            'adaptive_decisions_used': sorted(chosen), 'non_chosen_adaptive_endpoints': not_chosen}


def completion_table(completion, validation):
    lines = ['BRANCH COMPLETION (saved endpoints; independent of paired validation)']
    for key, title, _ in COMPLETION_GROUPS:
        group = completion['groups'][key]
        line = f"{title}: endpoints {group['endpoints']}/{group['planned']}; curves {group['curves']}/{group['endpoints']}"
        for field, name in (('remaining', 'remaining'),
                            ('budget_exhausted_needs_review', 'budget exhausted, needs review'),
                            ('failed_attempt', 'failed attempt'), ('endpoint_without_curve', 'no curve yet'),
                            ('endpoint_with_issues', 'endpoint with issues')):
            if group[field]:
                line += f"; {name}: " + ', '.join(group[field])
        lines.append(line)
    lines.append(f"total endpoints {completion['endpoints']}/{completion['planned']}")
    if completion['non_chosen_adaptive_endpoints']:
        lines.append('adaptive endpoints outside the frozen SR-GC choice (not counted): '
                     + ', '.join(completion['non_chosen_adaptive_endpoints']))
    if validation.get('status') == 'failed':
        lines.append('PAIRED STATUS: UNVERIFIED because strict paired validation failed. Paired rows, H and '
                     'checkpoint costs are not exported, and paired state lists are unknown (null), not missing. '
                     'The branch counts above are unaffected.')
    return '\n'.join(lines) + '\n\n'


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


def budget_recovery_measurements(root):
    """Export separately sealed saved-checkpoint evaluations, never paired wins."""
    data = {'rows': [], 'errors': [], 'scope':
            'Saved-checkpoint evaluations after the original allocation was exhausted. '
            'Result seals and plan bindings checked; full lineage and ledgers not independently certified. '
            'Original training and separate recovery evaluation costs retained. '
            'Not budget-compliant Pair completion, H labels, or SR-GC prediction accuracy.'}
    for seed in range(5):
        for step in (25, 50, 100):
            tasks = [('on_policy', 'selection_reduced'), ('cached', 'selection_reduced')]
            if seed >= 3:
                tasks = [(name, 'selection_full') for name in
                         ('on_policy', 'cached', 'adaptive-on_policy', 'adaptive-cached')]
                tasks.append(('on_policy', 'random_full'))
            for selector, arm in tasks:
                path = root / f'branches/{selector}/states/s{seed}-t{step}/points/view-{step}/{arm}/budget-recovery/result.json'
                if not path.exists() and not path.is_symlink():
                    continue
                try:
                    result, digest = read_source(path, root)
                    seal, _ = read_source(path.with_suffix('.sha256.json'), root)
                    plan, plan_hash = read_source(path.with_name('plan.json'), root)
                    schema = 'selector-pair-budget-recovery/v1'
                    if (seal != {'sha256': digest} or result.get('schema') != schema
                            or plan.get('schema') != schema or result.get('plan_sha256') != plan_hash
                            or result.get('evaluation_complete') is not True
                            or result.get('canonical_complete') is not False
                            or plan.get('canonical_complete') is not False
                            or plan.get('start_step') != step or plan.get('arm') != arm):
                        raise ValueError('recovery seal, schema, or plan binding changed')
                    stop = plan.get('completed_steps')
                    if type(stop) is not int or stop <= step:
                        raise ValueError('invalid recovery endpoint step')
                    costs = {}
                    for key in ('budget_gpu_seconds', 'used_gpu_seconds', 'over_budget_gpu_seconds'):
                        value = result.get(key)
                        if (type(value) not in (int, float) or not math.isfinite(value)
                                or value < 0 or value != plan.get(key)):
                            raise ValueError('invalid recovery allocation')
                        costs[key] = value
                    if (costs['used_gpu_seconds'] < costs['budget_gpu_seconds']
                            or not math.isclose(costs['over_budget_gpu_seconds'],
                                                costs['used_gpu_seconds'] - costs['budget_gpu_seconds'],
                                                abs_tol=1e-9)):
                        raise ValueError('inconsistent recovery allocation')
                    points, planned = result.get('points'), plan.get('points')
                    if not isinstance(points, list) or not points or not isinstance(planned, list) or len(points) != len(planned):
                        raise ValueError('missing recovery checkpoint points')
                    curves = []
                    for index, (point, source) in enumerate(zip(points, planned)):
                        if not isinstance(point, dict) or not isinstance(source, dict):
                            raise ValueError('malformed recovery checkpoint point')
                        checkpoint, rewards = point.get('step'), point.get('rewards')
                        if (type(checkpoint) is not int or not step <= checkpoint <= stop
                                or any(point.get(key) != source.get(key) for key in ('step', 'k', 'final'))
                                or point.get('final') is not (index == len(points) - 1)
                                or type(point.get('k')) is not int or point['k'] <= 0
                                or not isinstance(rewards, dict) or not rewards
                                or any(type(v) not in (int, float) or not math.isfinite(v)
                                       or not 0 <= v <= 1 for v in rewards.values())
                                or (curves and checkpoint < curves[-1]['checkpoint_step'])
                                or (point['final'] and checkpoint != stop)):
                            raise ValueError('invalid recovery checkpoint measurement')
                        curves.append({'checkpoint_step': checkpoint, 'updates': checkpoint - step,
                                       'reward': statistics.fmean(rewards.values()), 'k': point['k'],
                                       'final': point['final'], 'question_count': len(rewards)})
                    data['rows'].append({
                        'role': 'development' if seed < 3 else 'test', 'seed': seed,
                        'prefix_updates': step, 'selector_branch': selector, 'arm': arm,
                        'path': str(path.relative_to(root)), 'source_result_sha256': digest,
                        'source_plan_sha256': plan_hash, 'canonical_complete': False,
                        'eligible_for_paired_comparison': False, 'independently_certified': False,
                        'mean_reward': curves[-1]['reward'], 'updates': stop-step,
                        'curve_points': curves, 'original_cost': result.get('original_cost'),
                        'recovery_cost': result.get('recovery_cost'), **costs})
                except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
                    data['errors'].append({'path': str(path.relative_to(root)), 'error': str(exc)})
    return data


def recovery_table(data):
    output = io.StringIO()
    output.write('\nSAVED-CHECKPOINT BUDGET RECOVERY\n' + data['scope'] + '\n')
    writer = csv.writer(output)
    keys = ('role', 'seed', 'prefix_updates', 'selector_branch', 'arm', 'updates',
            'mean_reward', 'used_gpu_seconds', 'over_budget_gpu_seconds')
    writer.writerow(keys)
    for row in data['rows']:
        writer.writerow([row[key] for key in keys])
    return output.getvalue()


def exporter_metadata(repo):
    try:
        git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=repo,
                             capture_output=True, text=True, timeout=3)
        commit = git.stdout.strip() if git.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        commit = None
    return {'version': 'selector-pair-results/v8', 'git_commit': commit,
            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'created_at': datetime.now(timezone.utc).isoformat(), 'export_id': uuid.uuid4().hex}


def run_report(root, repo, timeout):
    """Bound only this export's CPU report process group, never existing workers."""
    command = ['bash', 'scripts/run_selector_pair.sh', 'report']
    if (root / 'pair-sr-gc-runtime.json').is_file():
        python = os.environ.get('PAIR_PYTHON') or str(repo / '.venv-cu126/bin/python')
        if not Path(python).is_file():
            python = sys.executable
        if not (repo / '.pair-runtime.json').is_file():
            from selector_pair_deploy import stage_runtime
            repo = stage_runtime(repo)
        command = [python, 'scripts/report_selector_pair_srgc.py', '--root', str(root)]
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


def srgc_results(root):
    """Export frozen decisions before endpoints arrive; never run a measurement."""
    data = {'status': 'not_recorded', 'method': None, 'decisions': [], 'pending_states': [],
            'errors': [], 'decision_barrier_frozen': False,
            'scope': 'Current-parent SR-GC decisions. Outcomes and crossing times are separate evidence; '
                     'a decision is not proof of an executed switch or a successful prediction.'}
    receipt = root / 'pair-sr-gc-runtime.json'
    if not receipt.exists() and not receipt.is_symlink():
        return data
    data.update(status='partial', method='SR-GC')
    try:
        import selector_pair_srgc as srgc
        protocol, _ = read_source(root / 'pair.json', root)
        _, data['runtime_sha256'] = read_source(receipt, root)
        srgc.validate(root, protocol)
        barrier = root / 'test-decisions.json'
        if barrier.exists() or barrier.is_symlink():
            read_source(barrier, root)
            srgc.decisions(root, protocol)
            data['decision_barrier_frozen'] = True
        for seed in (3, 4):
            for step in (25, 50, 100):
                name = f's{seed}-t{step}'
                path = root / 'sr-gc' / name / 'decision.json'
                if not path.exists() and not path.is_symlink():
                    data['pending_states'].append(name)
                    continue
                try:
                    saved, digest = read_source(path, root)
                    validated = srgc.validate_choice(root, protocol, seed, step)
                    if saved != validated:
                        raise ValueError('SR-GC decision changed during export')
                    data['decisions'].append({**saved, 'state': name, 'source_sha256': digest})
                except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                    data['errors'].append({'state': name, 'error': str(exc)})
        data['status'] = ('invalid' if data['errors'] else 'validated' if data['decision_barrier_frozen'] else 'partial')
    except (ImportError, OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        data.update(status='invalid', errors=[{'error': str(exc)}])
    return data


def srgc_table(data):
    if data['method'] is None:
        return ''
    output = io.StringIO()
    output.write('\nSR-GC DECISIONS: ' + data['status'] + '\n')
    output.write(data['scope'] + '\n')
    writer = csv.writer(output)
    keys = ('state', 'd_a', 'd_b', 'd', 'selector', 'new_measurement_gpu_seconds',
            'reused_ranking_gpu_seconds', 'diagnosis_gpu_seconds')
    writer.writerow(keys)
    for row in data['decisions']:
        writer.writerow([row[key] for key in keys])
    return output.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--report-timeout", type=float, default=60,
                        help="Maximum seconds for strict paired validation (default: 60)")
    parser.add_argument("--srgc-interval", type=int, default=25,
                        help="Recheck saved On checkpoints at this update interval; retain SR after first SR choice")
    args = parser.parse_args()
    if not math.isfinite(args.report_timeout) or args.report_timeout <= 0:
        parser.error('--report-timeout must be finite and positive')
    if args.srgc_interval <= 0:
        parser.error('--srgc-interval must be positive')
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
        except (OSError, ValueError, RuntimeError) as exc:
            data, curves, exit_code = empty_paired_report(), '', 2
            validation = {'status': 'failed', 'error': str(exc)}
    if validation.get('status') == 'failed':
        # A failed check says nothing about which states are missing.
        data.update(missing_states=None, missing_development_states=None, paired_status='unverified')
    data["source_root"] = str(root)
    rows, errors = saved_branch_measurements(root)
    data.update(branch_measurements=rows, branch_measurement_errors=errors,
                branch_measurement_scope=BRANCH_SCOPE, paired_validation=validation,
                exporter=exporter_metadata(repo), execution_observations=execution_observations(root, rows),
                cost_provenance=cost_provenance(root), schedule_provenance=schedule_provenance(root))
    data['srgc'] = srgc_results(root)
    import selector_pair_srgc_repeat as repeat
    data['srgc_repeated'] = repeat.collect(root, data['srgc'], args.srgc_interval)
    data['branch_completion'] = branch_completion(rows, data['execution_observations'], data['srgc'])
    data['budget_recovery_measurements'] = budget_recovery_measurements(root)
    if data['srgc']['method']:
        data['adaptive_method'] = data['srgc']['method']
    if data['srgc']['errors'] and not exit_code:
        exit_code = 2
    if data['srgc_repeated']['status'] == 'invalid' and not exit_code:
        exit_code = 2
    if not exit_code and (errors or any(row['issues'] for row in rows)
                          or data['budget_recovery_measurements']['errors']):
        exit_code = 2
    data['export_exit_code'] = exit_code
    header = completion_table(data['branch_completion'], validation)
    header += 'EXPORTER ' + json.dumps(data['exporter'], sort_keys=True) + '\n'
    header += 'PAIRED VALIDATION ' + json.dumps(validation, sort_keys=True) + '\n'
    header += ('SCHEDULE PROVENANCE: ' + data['schedule_provenance']['status'] + '. '
               + data['schedule_provenance']['scope'] + '\n')
    header += ('COST PROVENANCE: reconstructed events='
               f"{data['cost_provenance']['reconstructed_events']}; "
               f"inspection_complete={data['cost_provenance']['inspection_complete']}. "
               + data['cost_provenance']['scope'] + '\n')
    header += f'CURRENT SAVED BRANCHES {len(rows)}; ERRORS {len(errors)}\n'
    write_export("selector-pair", data, header + srgc_table(data['srgc'])
                 + repeat.table(data['srgc_repeated']) + curves + branch_table(rows)
                 + recovery_table(data['budget_recovery_measurements']), args.out)
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
