"""Provenance, validation and one-file export for MBPP off-policy calibration."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import tempfile

ESTIMATORS = ('g00', 'g10', 'g01', 'g11')
INPUTS = ('run_config.json', 'rollouts_behavior_train.jsonl', 'val_groups.pt',
          'scores_oracle.json', 'scores_offpolicy.json', 'scores_splithalf.json')
BINDING = 'mbpp_offpolicy_inputs.json'
SCHEMA = 'mbpp-offpolicy-calibration/v1'


def read_json(path):
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f'not a regular file: {path}')
    with path.open() as handle:
        return json.load(handle)


def digest(path):
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f'not a regular file: {path}')
    with path.open('rb') as handle:
        value = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
        return value.hexdigest()


def points(root, tag):
    return [(s, d, root / f'family-mbpp-s{s}' / f'{tag}-s{s}-mbpp-d{d}')
            for d in (0, 400) for s in range(3)]


def fingerprint(run, seed, drift):
    if not (run / 'DONE').is_file():
        raise ValueError('original matrix point is not complete; no training is started')
    config = read_json(run / 'run_config.json')
    if (config.get('dataset'), config.get('seed'), config.get('drift')) != ('mbpp', seed, drift):
        raise ValueError('source dataset/seed/checkpoint mismatch')
    if not isinstance(config.get('prompt_format'), str) or not config['prompt_format']:
        raise ValueError('saved prompt format is missing; refusing a default prompt format')
    ids = set(read_json(run / 'scores_oracle.json'))
    if ids != set(map(str, range(512))):
        raise ValueError('expected the registered 512 candidate indices')
    responses = {key: set() for key in ids}
    rollout_path = run / 'rollouts_behavior_train.jsonl'
    if not stat.S_ISREG(rollout_path.stat().st_mode):
        raise ValueError('behavior responses are not a regular file')
    with rollout_path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key, index = str(row['prompt_idx']), row['rollout_idx']
            if key not in responses or type(index) is not int or index not in range(8) or index in responses[key]:
                raise ValueError('missing, duplicate or unexpected behavior response index')
            if row['reward'] not in (0, 1):
                raise ValueError('expected binary execution rewards')
            responses[key].add(index)
    if any(len(group) != 8 for group in responses.values()):
        raise ValueError('expected eight stored responses per candidate; no new generation is started')
    files = list(INPUTS)
    if drift:
        adapter = run / f'policy_step_{drift}'
        weights = list(adapter.glob('adapter_model.*'))
        if not weights or not (adapter / 'adapter_config.json').is_file():
            raise ValueError('saved adapter is missing; refusing to initialize a new policy')
        files += [str(p.relative_to(run)) for p in sorted(weights)]
        files.append(f'policy_step_{drift}/adapter_config.json')
    source = Path(__file__).resolve().parent
    return {'schema': SCHEMA, 'dataset': 'mbpp', 'seed': seed, 'drift': drift,
            'files': {name: digest(run / name) for name in files},
            'scoring_code_sha256': {name: digest(source / name) for name in
                                   ('stale_splithalf.py', 'grads.py', 'rollout.py')}}


def validate_binding(run, seed, drift):
    current = fingerprint(run, seed, drift)
    if read_json(run / BINDING) != current:
        raise ValueError('source inputs changed since preparation; old scores are not reused')
    return current


def prepare(root, tag):
    # Validate every source before creating any binding or launching a GPU.
    records = [(run, fingerprint(run, s, d)) for s, d, run in points(root, tag)]
    for run, binding in records:
        path = run / BINDING
        if path.exists():
            if read_json(path) != binding:
                raise ValueError(f'source binding changed: {run}')
        elif list(run.glob('scores_stale_splithalf*.json')):
            raise ValueError(f'unbound pre-existing scores; preserved without reuse: {run}')
    for run, binding in records:
        path = run / BINDING
        fd, temporary = tempfile.mkstemp(prefix=f'.{BINDING}.', dir=run)
        try:
            with os.fdopen(fd, 'w') as handle:
                json.dump(binding, handle, sort_keys=True, indent=2)
                handle.write('\n')
            os.link(temporary, path)
        except FileExistsError:
            if read_json(path) != binding:
                raise ValueError(f'concurrent preparation mismatch: {run}')
        finally:
            os.unlink(temporary)
    print('[mbpp-offpolicy] prepared 6 existing points; no training or response generation')


def validate_scores(run):
    scores = read_json(run / 'scores_stale_splithalf.json')
    protocol = read_json(run / 'scores_stale_splithalf.protocol.json')
    ids = set(read_json(run / 'scores_oracle.json'))
    if len(ids) != 512:
        raise ValueError('expected the registered 512 MBPP candidates')
    if protocol.get('schema') != 'offpolicy-stale-splithalf/v1' or protocol.get('prompts') != len(ids):
        raise ValueError('missing or inconsistent scoring protocol')
    config = read_json(run / 'run_config.json')
    expected = {'proj_dim': int(config.get('proj_dim', 4096)),
                'grad_layers': int(config.get('grad_layers', 4)),
                'clip_cap': float(config.get('clip_cap', 10.0)),
                'micro_batch': int(config.get('micro_batch', 2)),
                'adapter': str(run / f"policy_step_{config['drift']}") if config['drift'] else None}
    if protocol.get('shards') != 4 or protocol.get('parameters') != expected:
        raise ValueError('scoring parameters differ from the registered source')
    if set(scores) != set(ESTIMATORS):
        raise ValueError('missing off-policy estimator')
    for est in ESTIMATORS:
        if set(scores[est]) != ids:
            raise ValueError(f'{est}: incomplete candidate coverage')
        for value in scores[est].values():
            if set(value) != {'a', 'b'} or not all(
                    isinstance(v, (int, float)) and not isinstance(v, bool) and
                    math.isfinite(v) and abs(v) <= 1.000001 for v in value.values()):
                raise ValueError(f'{est}: invalid half-score')
        check = protocol.get('full_score_check', {}).get(est, {})
        difference = check.get('max_abs_difference')
        if (check.get('prompts', 0) != 16 or not isinstance(difference, (int, float)) or
                not math.isfinite(difference) or difference < 0):
            raise ValueError(f'{est}: full-score consistency check missing')
    return scores, protocol


def active(run):
    path = run / '.stale-splithalf.lock'
    if not path.exists():
        return False
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f'not a regular lock file: {path}')
    with path.open('r') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle, fcntl.LOCK_UN)
    return False


def estimator_rows(run, cache):
    """Calibration rows for the four stored-response estimators only.

    gain_vs_reliability.py stays byte-identical to the version the RLOO contract
    froze, so its module-level signal list is narrowed for this call only.
    """
    import gain_vs_reliability as gain
    original = gain.SIGNALS
    gain.SIGNALS = ESTIMATORS
    try:
        return gain.point_rows(run, .1, cache)
    finally:
        gain.SIGNALS = original


def collect(root, tag, include_scores=False):
    records, cache = [], {}
    for seed, drift, run in points(root, tag):
        record = {'seed': seed, 'drift': drift, 'path': str(run), 'status': 'BLOCKED', 'rows': []}
        try:
            if active(run):
                record['status'] = 'RUN'
            else:
                record['inputs'] = validate_binding(run, seed, drift)
                if (run / 'scores_stale_splithalf.json').exists():
                    scores, protocol = validate_scores(run)
                    rows = estimator_rows(run, cache)
                    if len(rows) != 4:
                        raise ValueError('expected four estimator rows')
                    record.update(status='DONE', rows=rows, scoring_protocol=protocol)
                    if include_scores:
                        record['half_scores'] = scores
                else:
                    record['status'] = 'READY'
        except (OSError, ValueError, KeyError, TypeError) as exc:
            record['error'] = str(exc)
        records.append(record)
    return {'schema': SCHEMA, 'expected_points': 6, 'expected_estimator_rows': 24,
            'completed_points': sum(r['status'] == 'DONE' for r in records), 'points': records,
            'scope': 'Score calibration only; no policy training, benchmark reward, or H target-cost measurement.',
            'full_score_check': 'Recorded numerical differences are exported, not silently thresholded into equivalence.'}


def render(data):
    lines = [f"MBPP off-policy calibration: {data['completed_points']}/6 complete (24 planned estimator rows)"]
    for point in data['points']:
        lines.append(f"s{point['seed']}/d{point['drift']} {point['status']} {point.get('error', '')}")
        for row in point['rows']:
            fields = ' '.join(f'{name}={row[name]}' for name in ('rho_half', 'gain', 'predicted'))
            lines.append(f"  {row['signal']} {fields} {row.get('note', '')}")
    return '\n'.join(lines)


def export(data, path):
    text = render(data) + '\n\nDATA_JSON\n' + json.dumps(data, allow_nan=False, separators=(',', ':')) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as handle:
            handle.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'status', 'results'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    args = parser.parse_args(argv)
    if args.mode == 'prepare':
        try:
            prepare(args.root, args.tag)
            return 0
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f'[mbpp-offpolicy] blocked: {exc}')
            return 2
    data = collect(args.root, args.tag, include_scores=args.mode == 'results')
    print(render(data))
    if args.mode == 'results':
        path = Path.home() / 'mbpp-offpolicy-results.txt'
        export(data, path)
        print(f'[mbpp-offpolicy] written: {path}')
    return 0 if data['completed_points'] == 6 else 1


if __name__ == '__main__':
    raise SystemExit(main())
