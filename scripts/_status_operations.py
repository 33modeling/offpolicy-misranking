"""Read-only discovery of operational meters outside experiment branches."""

from pathlib import Path


def progress_paths(root):
    # Fixed-depth paths avoid traversing rollout/model/cost-event trees.
    root = Path(root)
    yield from (root / 'node-preflight').glob('*/*/progress.json')
    yield from (root / 'decisions').glob('s*-t*/progress.json')


def task(root, directory, progress, *, fresh, owned, age):
    import re
    relative = str(directory.relative_to(root))
    match = re.search(r'(?:^|/)s(\d+)-t(\d+)(?:/|$)', relative)
    admission = relative.startswith('node-preflight/')
    return {**progress, 'kind': 'phase', 'arm': 'admission' if admission else 'decision',
            'host': progress.get('host', ''), 'pid': progress.get('pid'), 'reason': '',
            'phase': progress.get('phase', ''), 'seconds': progress.get('seconds', 0),
            'timeout': progress.get('timeout', 0),
            'seed': int(match[1]) if match else '-', 'step': int(match[2]) if match else '-',
            'directory': relative, 'status': 'RUNNING', 'heartbeat_fresh': fresh,
            'owner_active': owned, 'heartbeat_age': age, 'training_step': None,
            'shared_operation': True}


def curve_lease_tasks(root, tasks, probe, *, prefix=''):
    """Discover surviving curve children without inventing their host identity."""
    import re
    from _status_execution import active

    root = Path(root)
    arms = ('selection_reduced', 'selection_full', 'random_reduced', 'random_full', 'gated')
    candidates = list((root / 'states').glob('s*-t*/points/*/curve-parent'))
    for arm in arms:
        candidates.extend((root / 'states').glob(f's*-t*/points/*/{arm}/curve/step-*'))
    found = []
    for directory in sorted(candidates):
        relative = directory.relative_to(root)
        match = re.fullmatch(r's([0-4])-t(25|50|100)', relative.parts[1])
        if not match:
            continue
        try:
            if not directory.resolve().is_relative_to(root.resolve()):
                continue
        except (OSError, RuntimeError):
            continue
        name = prefix + str(relative)
        ledger = name if directory.name == 'curve-parent' else prefix + str(relative.parent)
        if any(active(row) and (row.get('directory') in {name, ledger}
               or row.get('phase') == 'curve' and row.get('directory')
               and name.startswith(row['directory'].rstrip('/') + '/')) for row in tasks):
            continue
        if not any(probe(directory, lock) for lock in ('.point.lock', *(f'shard-{i}.lock' for i in range(4)))):
            continue
        seed, step = map(int, match.groups())
        found.append({'kind': 'phase', 'seed': seed, 'step': step,
                      'role': 'DEV' if seed < 3 else 'TEST',
                      'directory': name, 'arm': 'curve-parent' if directory.name == 'curve-parent'
                      else '/'.join(relative.parts[4:]), 'phase': 'curve',
                      'status': 'RUNNING', 'reason': 'curve evaluation lease held; worker identity unconfirmed',
                      'host': None, 'pid': None, 'worker_id': None, 'event_id': None,
                      'heartbeat_fresh': False, 'owner_active': False, 'task_lease_held': True,
                      'activity_identity_unconfirmed': True, 'seconds': None, 'timeout': None})
    return found
