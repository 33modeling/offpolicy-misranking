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
