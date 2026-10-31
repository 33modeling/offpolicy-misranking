"""Reconcile retained update timers against all recorded restart attempts.

This is a read-only accounting view. It never edits receipts/checkpoints, infers
unclosed durations, divides measured GPU time by world size, or removes real
operations within the retained update. A replay is excluded from the matched
update comparison, while its actual consumed cost stays in the original ledger.
"""

import math
from pathlib import Path


def finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError('restart costs require finite nonnegative GPU-seconds')
    return float(value)


def reconcile(folder, arm, points, *, start, stop, read, expected_gpus=None):
    paths = sorted((Path(folder) / 'cost-receipts' / arm).glob('*.json'))
    if not paths:
        return {'record': 'MISSING', 'recorded_before_gpu_seconds': None, 'removed_gpu_seconds': None,
                'retained_gpu_seconds': None, 'unfinished_receipts': 0, 'duplicate_receipt_copies': 0}
    events, copies, duplicate_seconds, unfinished, sources = {}, 0, 0., [], []
    for path in paths:
        row = read(path)
        if row is None:
            raise ValueError('cost receipts changed during restart accounting; rerun results')
        if not isinstance(row, dict):
            raise TypeError('restart receipt must be an object')
        phase = row.get('phase')
        if phase not in {'selection', 'training'}:
            continue
        checkpoint = row.get('checkpoint')
        identity = row.get('id')
        if type(checkpoint) is not int or not isinstance(identity, str) or not identity:
            raise ValueError('restart receipt lacks phase checkpoint or identity')
        if not start <= checkpoint < stop:
            continue
        source = {key: row[key] for key in ('id', 'phase', 'checkpoint', 'gpu_count', 'state',
                                            'wall_seconds', 'gpu_seconds', 'stages', 'counts') if key in row}
        source['source_file'] = str(path)
        if 'rank_timings' in row:
            source['rank_timings'] = [{key: part[key] for key in ('wall_seconds', 'local_gpu_count') if key in part}
                                      for part in row['rank_timings']]
        sources.append(source)
        state = row.get('state')
        if state not in {'started', 'finished'}:
            raise ValueError('invalid restart receipt state')
        if state == 'finished':
            finite(row['gpu_seconds'])
            if 'wall_seconds' in row:
                wall, count = finite(row['wall_seconds']), row.get('gpu_count')
                if type(count) is not int or count < 0 or not math.isclose(
                        row['gpu_seconds'], wall * count, rel_tol=1e-9, abs_tol=1e-6):
                    raise ValueError('restart receipt GPU time differs from wall time times allocated GPUs')
                if expected_gpus is not None and count != expected_gpus:
                    raise ValueError('restart receipt GPU count differs from the experiment plan')
        previous = events.get(identity)
        if previous is not None:
            for field in ('phase', 'checkpoint', 'gpu_count'):
                if previous.get(field) != row.get(field):
                    raise ValueError('one restart receipt ID names different operations')
            if previous['state'] == row['state'] == 'finished':
                if previous != row:
                    raise ValueError('conflicting completed copies of one restart receipt')
                copies += 1
                duplicate_seconds += finite(row['gpu_seconds'])
                continue
            if previous['state'] == 'finished':
                continue
        events[identity] = row

    finished = {}
    for identity, row in events.items():
        if row['state'] == 'finished':
            finished[identity] = row
        else:
            unfinished.append({'id': identity, 'phase': row['phase'], 'checkpoint': row['checkpoint']})
    used, matches, missing = set(), [], []
    for point in points:
        for phase in ('selection', 'training'):
            cost = point.get(phase + '_gpu_seconds')
            if cost is None:
                missing.append({'step': point['step'], 'phase': phase, 'reason': 'missing update timer'})
                continue
            # A non-refresh update has no selection phase. Explicit receipt IDs
            # also allow matching a genuine zero-cost CPU test phase.
            receipt_id = point.get('timing_receipt_ids', {}).get(phase)
            if cost == 0 and receipt_id is None:
                continue
            candidates = [identity for identity, event in finished.items()
                          if identity not in used and event['phase'] == phase
                          and event['checkpoint'] == point['checkpoint']
                          and math.isclose(event['gpu_seconds'], cost, rel_tol=1e-9, abs_tol=1e-6)]
            if receipt_id is not None:
                if receipt_id not in candidates:
                    raise ValueError('retained update receipt ID or measured duration differs')
                chosen = receipt_id
            elif candidates:
                chosen = min(candidates)
            else:
                missing.append({'step': point['step'], 'phase': phase, 'reason': 'no matching finished receipt'})
                continue
            used.add(chosen)
            matches.append({'step': point['step'], 'phase': phase, 'receipt_id': chosen,
                            'gpu_seconds': cost, 'matching_receipts': len(candidates)})
    covered = len(points) == stop - start and not missing
    before = math.fsum(event['gpu_seconds'] for event in finished.values()) + duplicate_seconds
    retained = math.fsum(match['gpu_seconds'] for match in matches)
    replayed = [{'id': identity, 'phase': event['phase'], 'checkpoint': event['checkpoint'],
                 'gpu_seconds': event['gpu_seconds']} for identity, event in finished.items() if identity not in used]
    removed = math.fsum(event['gpu_seconds'] for event in replayed) + duplicate_seconds if covered else None
    if covered and not math.isclose(before, retained + removed, rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError('restart before-minus-removed does not reconcile with retained updates')
    return {'schema': 'srgc-restart-accounting-v2',
            'unique_recorded_gpu_seconds': math.fsum(event['gpu_seconds'] for event in finished.values()),
            'replayed_gpu_seconds': math.fsum(event['gpu_seconds'] for event in replayed) if covered else None,
            'raw_receipt_sum_gpu_seconds': before,
            'duplicate_copy_gpu_seconds_excluded': duplicate_seconds,
            'record': 'COMPLETE' if covered and not unfinished else 'PARTIAL',
            'recorded_before_gpu_seconds': before, 'removed_gpu_seconds': removed,
            'retained_gpu_seconds': retained if covered else None,
            'removal_is_lower_bound': bool(unfinished), 'unfinished_receipts': len(unfinished),
            'unfinished': unfinished, 'duplicate_receipt_copies': copies,
            'duplicate_receipt_gpu_seconds': duplicate_seconds,
            'discarded_completed_attempts': replayed, 'retained_receipts': matches,
            'unmatched_retained_updates': missing,
            'source_receipts': sources,
            'source_files': [str(path) for path in paths],
            'identity': 'bound seed/arm run folder; phase checkpoint, duration and recorded receipt ID when available',
            'scope': 'selection and training only; preparation, evaluation, checkpoints and startup remain separate'}
