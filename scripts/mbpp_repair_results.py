"""Read-only, single-file MBPP repair results with separated cost provenance."""

import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics

from paper_result_text import write_export
import switch_results


LIMIT = 1024 * 1024
READ_LIMIT = 8 * LIMIT
RESULT_SCHEMA = 'offpolicy-selected-prefix-switch/v1'
GROUPS = ('reused_branches', 'rerun_branches', 'dependent_branches')
PATH = re.compile(r'states/s([0-4])-t(25|50|100)/points/view-(25|50|100)/'
                  r'(selection_reduced|random_reduced|selection_full|random_full|gated)')
NOTES = [
    'The 37 reused branches are saved original measurements, not 37 newly trained branches.',
    'Reward unit: fraction. Costs: GPU-seconds. Missing measurements are null, never estimated.',
    'Original-source branch costs and new repair branch costs are separate and are NOT pooled.',
    'Finished-event costs include failed attempts; incomplete ledgers have no known total.',
    'Shared parent/diagnostic, prefix, archived and job-billing costs are excluded from branch totals.',
    'Snapshot hashes and result seals are checked; policy/optimizer lineage is not independently certified.',
    'Repair results are a separate follow-up run, not an unchanged continuation of the original budget.',
]


def payload(root, relative):
    path = root / relative
    if not path.resolve().is_relative_to(root):
        raise ValueError('metadata path escapes its experiment root')
    with path.open('rb') as handle:
        raw = handle.read(READ_LIMIT + 1)
    if len(raw) > READ_LIMIT:
        raise ValueError('metadata exceeds bounded read limit')
    return raw


def object_at(root, relative):
    value = json.loads(payload(root, relative))
    if not isinstance(value, dict):
        raise ValueError('metadata must be a JSON object')
    return value


def number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def cost(root, relative, snapshots=None):
    phases, events, issues = defaultdict(float), {}, []
    found = False
    for suffix in ('cost.jsonl', 'curve/cost.jsonl'):
        try:
            raw = payload(root, relative + '/' + suffix)
            if snapshots is not None and snapshots.get(relative + '/' + suffix) != hashlib.sha256(raw).hexdigest():
                raise ValueError('original cost ledger differs from frozen repair snapshot')
        except FileNotFoundError:
            if suffix == 'cost.jsonl':
                issues.append('branch ledger missing')
            continue
        except (OSError, ValueError, RuntimeError) as exc:
            issues.append(str(exc)[:256])
            continue
        found = True
        for line in raw.splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if (not isinstance(row, dict) or not isinstance(row.get('event_id'), str)
                        or not row['event_id'] or row.get('state') not in {'started', 'finished'}):
                    raise ValueError('invalid cost event')
                event = events.setdefault(row['event_id'], {})
                if row['state'] in event:
                    raise ValueError('duplicate cost event')
                event[row['state']] = row
                if row['state'] == 'finished':
                    if (not number(row.get('allocated_gpu_seconds'))
                            or not isinstance(row.get('phase'), str) or not isinstance(row.get('ledger'), str)):
                        raise ValueError('invalid finished cost event')
                    phases[(row['ledger'], row['phase'])] += row['allocated_gpu_seconds']
            except (ValueError, TypeError, KeyError) as exc:
                if len(issues) < 8:
                    issues.append(str(exc)[:256])
    for event in events.values():
        if set(event) != {'started', 'finished'}:
            issues.append('open or missing cost event')
            break
        if any(event['started'].get(key) != event['finished'].get(key) for key in ('ledger', 'phase')):
            issues.append('cost event identity mismatch')
            break
    subtotal = sum(phases.values()) if phases else None
    if subtotal is not None and not number(subtotal):
        issues.append('cost subtotal overflows finite numeric range')
        subtotal = None
        phases = {key: value for key, value in phases.items() if number(value)}
    closed = bool(events) and not issues
    return {'unit': 'GPU-seconds', 'finished_events_subtotal': subtotal,
            'total': subtotal if closed else None,
            'coverage': 'closed events' if closed else 'unknown',
            'phases': [{'ledger': ledger, 'phase': phase, 'gpu_seconds': value}
                       for (ledger, phase), value in sorted(phases.items())],
            'issues': issues[:8] or ([] if found and events else ['no recorded cost events'])}


def measurement(root, relative, step, snapshots=None):
    row = {'status': 'unmeasured', 'mean_reward': None, 'questions': None,
           'curve_complete': False, 'question_rewards': None,
           'updates': None, 'curve': [], 'issues': []}
    try:
        raw = payload(root, relative + '/result.json')
    except FileNotFoundError:
        return row
    try:
        digest = hashlib.sha256(raw).hexdigest()
        result = json.loads(raw)
        seal_relative = relative + '/result.sha256.json'
        seal_raw = payload(root, seal_relative)
        seal = json.loads(seal_raw)
        if snapshots is not None:
            for name, content in ((relative + '/result.json', raw), (seal_relative, seal_raw)):
                if snapshots.get(name) != hashlib.sha256(content).hexdigest():
                    raise ValueError('reused result differs from its frozen repair snapshot')
        if seal != {'sha256': digest}:
            raise ValueError('result seal does not match endpoint')
        rewards = result.get('rewards') if isinstance(result, dict) else None
        stop = result.get('completed_steps') if isinstance(result, dict) else None
        if (not isinstance(result, dict) or result.get('schema') != RESULT_SCHEMA
                or result.get('complete') is not True
                or not isinstance(rewards, dict) or not rewards
                or any(not number(value) or value > 1 for value in rewards.values())
                or type(stop) is not int or stop < step):
            raise ValueError('incomplete or malformed endpoint measurement')
        row.update(status='saved_measurement', mean_reward=statistics.fmean(rewards.values()),
                   questions=len(rewards), updates=stop-step, result_sha256=digest,
                   question_rewards=rewards,
                   question_ids_sha256=hashlib.sha256(json.dumps(sorted(rewards)).encode()).hexdigest())
        try:
            curve_raw = payload(root, relative + '/curve.json')
        except FileNotFoundError:
            return row
        if snapshots is not None and snapshots.get(relative + '/curve.json') != hashlib.sha256(curve_raw).hexdigest():
            raise ValueError('reused curve differs from its frozen repair snapshot')
        curve = json.loads(curve_raw)
        if (not isinstance(curve, dict) or curve.get('schema') != RESULT_SCHEMA
                or curve.get('result_sha256') != digest):
            raise ValueError('curve is not bound to the accepted endpoint')
        points = curve.get('points')
        if not isinstance(points, dict):
            raise ValueError('curve points must be an object')
        accepted = []
        for checkpoint, point in sorted(points.items(), key=lambda item: int(item[0])):
            if (not isinstance(point, dict) or checkpoint != str(int(checkpoint))
                    or type(point.get('updates')) is not int or point['updates'] != int(checkpoint)-step
                    or not step <= int(checkpoint) <= stop
                    or not number(point.get('reward')) or point['reward'] > 1):
                raise ValueError('malformed saved curve point')
            if point.get('final') and (int(checkpoint) != stop or
                    not math.isclose(point['reward'], row['mean_reward'], rel_tol=0, abs_tol=1e-12)):
                raise ValueError('final curve point disagrees with endpoint')
            accepted.append({'updates': point['updates'], 'reward': point['reward']})
        row['curve'] = accepted
        row['curve_complete'] = bool(accepted) and any(
            point.get('final') is True and int(checkpoint) == stop for checkpoint, point in points.items())
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        row['issues'].append(str(exc)[:512])
        if row['mean_reward'] is None:
            row['status'] = 'unverified'
    return row


def export(root):
    root = root.resolve()
    data = {'schema': 'mbpp-repair-results/v1', 'repair_root': str(root),
            'notes': NOTES, 'branches': [], 'errors': [], 'complete': False}
    try:
        meta = object_at(root, 'repair.json')
        if meta.get('schema') != 'mbpp-repair/v1':
            raise ValueError('unsupported repair metadata schema')
        source = Path(meta['source_root'])
        if not source.is_absolute() or source.resolve() == root:
            raise ValueError('repair source must be a distinct absolute root')
        source = source.resolve()
        snapshots = meta.get('snapshot_files')
        if not isinstance(snapshots, dict):
            raise ValueError('repair snapshot hash inventory missing')
        data.update(source_root=str(source), source_switch_sha256=meta.get('source_switch_sha256'),
                    repair_metadata_sha256=hashlib.sha256(payload(root, 'repair.json')).hexdigest())
        classifications = {}
        for group in GROUPS:
            values = meta.get(group)
            if not isinstance(values, list):
                raise ValueError('repair branch inventory missing: ' + group)
            for relative in values:
                match = PATH.fullmatch(relative) if isinstance(relative, str) else None
                if (not match or match[2] != match[3] or relative in classifications
                        or (int(match[1]) < 3 and match[4] not in {'selection_reduced', 'random_reduced'})):
                    raise ValueError('invalid or overlapping repair branch inventory')
                classifications[relative] = (group, int(match[2]))
        if len(classifications) != 48:
            raise ValueError('repair inventory must contain all 48 planned branches')
        data['planned'] = {group: len(meta[group]) for group in GROUPS}
        if data['planned'] != dict(zip(GROUPS, (37, 5, 6))):
            raise ValueError('repair inventory must preserve the frozen 37 reused, 5 rerun, 6 dependent plan')
        if any((group == 'dependent_branches') != relative.endswith('/gated')
               for relative, (group, _) in classifications.items()):
            raise ValueError('dependent repair inventory must contain exactly the six gated branches')
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        data['errors'].append(str(exc)[:512])
        return data
    try:
        data['source_manifest_matches_snapshot'] = (hashlib.sha256(payload(source, 'switch.json')).hexdigest()
                                                    == meta['source_switch_sha256'])
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        data['source_manifest_matches_snapshot'] = None
        data['errors'].append('original source manifest: ' + str(exc)[:512])
    if data['source_manifest_matches_snapshot'] is False:
        data['errors'].append('original source manifest changed since repair preparation')
    for relative, (group, step) in sorted(classifications.items()):
        reused = group == 'reused_branches'
        try:
            observed = measurement(root, relative, step, snapshots if reused else None)
        except (OSError, ValueError, RuntimeError) as exc:
            observed = {'status': 'unverified', 'mean_reward': None, 'questions': None, 'curve_complete': False,
                        'updates': None, 'curve': [], 'issues': [str(exc)[:512]]}
        original = 'original-attempts/' + relative
        frozen_cost = group == 'rerun_branches' and (root / original).is_dir()
        data['branches'].append({'path': relative, 'origin': group,
            'measurement': observed,
            'original_source_cost': cost(root, original, snapshots) if frozen_cost else cost(source, relative),
            'new_repair_cost': None if reused else cost(root, relative),
            'original_source_cost_scope': ('frozen original-attempts repair snapshot' if frozen_cost else
                                            'current source ledger, not a frozen cost snapshot')})
    data['measured'] = {group: sum(row['origin'] == group and row['measurement']['mean_reward'] is not None
                                 for row in data['branches']) for group in GROUPS}
    data['endpoint_coverage_complete'] = all(row['measurement']['mean_reward'] is not None for row in data['branches'])
    data['complete'] = (data['endpoint_coverage_complete']
                        and all(row['measurement']['curve_complete'] for row in data['branches'])
                        and not data['errors'])
    try:
        switch_results.report(root)
        data['auxiliary_report'] = {'status': 'generated', 'included': False,
            'reason': 'Unpartitioned report costs are not new repair costs; use the origin-separated rows.'}
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError) as exc:
        data['auxiliary_report'] = {'status': 'failed', 'error': str(exc)[:1024]}
    return data


def table(data):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['branch', 'origin', 'reward_fraction', 'questions', 'updates',
                     'original_source_gpu_seconds', 'new_repair_gpu_seconds'])
    for row in data['branches']:
        measured = row['measurement']
        writer.writerow([row['path'], row['origin'], measured['mean_reward'], measured['questions'],
                         measured['updates'], row['original_source_cost']['total'],
                         row['new_repair_cost']['total'] if row['new_repair_cost'] else None])
    return output.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    data = export(args.root)
    rendered = table(data)
    size = len(json.dumps(data, ensure_ascii=True, separators=(',', ':'), allow_nan=False).encode())
    if size + len(rendered.encode()) + 1024 > LIMIT:
        for row in data['branches']:
            curve = row['measurement']['curve']
            row['measurement']['curve_points_omitted_for_size'] = len(curve)
            row['measurement']['curve'] = []
        data['errors'].append('Curve points omitted to preserve the single 1 MiB export limit; endpoints retained.')
    size = len(json.dumps(data, ensure_ascii=True, separators=(',', ':'), allow_nan=False).encode())
    if size + len(rendered.encode()) + 1024 > LIMIT:
        for row in data['branches']:
            rewards = row['measurement'].get('question_rewards')
            row['measurement']['question_rewards_omitted_for_size'] = len(rewards) if rewards else 0
            row['measurement']['question_rewards'] = None
        data['errors'].append('Per-question rewards omitted for size; endpoint means and source hashes retained.')
    size = len(json.dumps(data, ensure_ascii=True, separators=(',', ':'), allow_nan=False).encode())
    if size + len(rendered.encode()) + 1024 > LIMIT:
        data = {'schema': data['schema'], 'repair_root': data['repair_root'], 'complete': False,
                'errors': ['Repair metadata exceeds the 1 MiB export limit; no scientific values emitted.']}
        rendered = ''
    write_export('mbpp-repair', data, rendered, args.out)
    if data.get('errors') or any(row['measurement']['issues'] for row in data.get('branches', [])):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
