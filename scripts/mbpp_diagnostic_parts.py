"""Read-only MBPP metadata export, split by default or one uncapped TXT on request."""

import json
import fcntl
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import mbpp_failure_summary as summary

# Overleaf: 2 MB per editable file and 7 MB editable material per project.
# Decimal bytes keep each file below either interpretation of those limits.
PART_BYTES = 1_900_000
MAX_PARTS = 3
LOG_BYTES = 8 * 1024
ADMISSION_LIMIT = 8
ADMISSION_BYTES = 16 * 1024
ARMS = {'selection_reduced', 'random_reduced', 'selection_full', 'random_full', 'gated'}
POLICY_FILES = ('adapter_config.json', 'adapter_model.safetensors', 'optimizer.pt',
                'grpo_stats.jsonl', 'checkpoint_state.json', 'policy_train.json', 'budget_stop.json')


def metadata(root, path, keys=None):
    value = summary.record(root, path)
    if keys is not None and '_read_error' not in value:
        value = {key: value[key] for key in keys if key in value}
    return f'FILE {path.relative_to(root)}\n{json.dumps(value, ensure_ascii=False, sort_keys=True)}\n'


def policy_inventory(root, policy):
    yield f'POLICY {policy.relative_to(root)} (presence only; hashes NOT validated)\n'
    try:
        summary.checked(root, policy)
        candidates = [policy, *sorted(policy.glob('checkpoint-*')), *sorted(policy.glob('.checkpoint-*.tmp')),
                      *sorted((policy / 'curve-checkpoints').glob('step-*'))]
        for candidate in candidates:
            summary.checked(root, candidate)
            files = {}
            for name in POLICY_FILES:
                path = summary.checked(root, candidate / name)
                files[name] = path.stat().st_size if path.is_file() else None
            yield f'INVENTORY {candidate.relative_to(root)} bytes={json.dumps(files, sort_keys=True)}\n'
            for name in ('checkpoint_state.json', 'policy_train.json', 'budget_stop.json'):
                path = candidate / name
                if path.exists():
                    yield metadata(root, path, ('schema', 'completed_steps', 'start_step', 'target_steps',
                        'requested_target_steps', 'stop_reason', 'use_parent_policy', 'training_objective',
                        'adapter_sha256', 'optimizer_sha256', 'grpo_stats_sha256'))
    except (OSError, ValueError) as exc:
        yield f'INVENTORY ERROR {exc}\n'


def log_excerpt(root, path):
    yield f'LOG {path.relative_to(root)} (last {LOG_BYTES} bytes at most)\n'
    try:
        summary.checked(root, path)
        with path.open('rb') as handle:
            size = handle.seek(0, 2)
            handle.seek(max(0, size - LOG_BYTES))
            data = handle.read(LOG_BYTES)
        if size > LOG_BYTES:
            yield '[Earlier log bytes not collected; metadata records below are exported separately.]\n'
        lines = data.decode('utf-8', errors='replace').splitlines()
        holding = [line for line in lines if line.startswith('[holding]')]
        lines = [line for line in lines if not line.startswith('[holding]')]
        if holding:
            lines += [f'[Collapsed {len(holding)} holding lines; last follows]', holding[-1]]
        yield '\n'.join(lines) + '\n'
    except (OSError, ValueError) as exc:
        yield f'LOG ERROR {exc}\n'


def lease_record(root, path):
    try:
        summary.checked(root, path)
        with path.open('rb') as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                state = 'held'
            else:
                state = 'free-at-probe'
                fcntl.flock(handle, fcntl.LOCK_UN)
    except FileNotFoundError:
        state = 'missing'
    except (OSError, ValueError, RuntimeError) as exc:
        state = f'unknown: {exc}'
    return f'LEASE {path.relative_to(root)} {state} (observation only, not owner identity)\n'


def phase_logs(root, directory):
    summary.checked(root, directory)
    progress = summary.record(root, directory / 'progress.json')
    failure = summary.record(root, directory / 'failure.json')
    phase = progress.get('phase') or failure.get('phase')
    pattern = f'{phase}-*.log' if isinstance(phase, str) and re.fullmatch(r'[\w-]+', phase) else '*.log'
    return [(root, path) for path in summary.recent(directory, (pattern,))[:4]]


def sections(work, roots, *, single_file=False):
    inventories, worker_logs, admissions = [], [], []
    upload_scope = ('One TXT file; no total output-size cap. Per-record/log safety bounds still apply.\n'
                    if single_file else
                    'At most 3 files of 1,900,000 bytes; any total-limit omission is explicitly marked.\n'
                    'Existing project text also counts toward the Overleaf 7 MB total limit.\n')
    yield ('MBPP DIAGNOSTIC DETAILS\nREAD-ONLY. No training, repair or lock creation.\n'
           f'UTC {datetime.now(timezone.utc).isoformat(timespec="seconds")}\n'
           'No model/optimizer/rollout payloads. Presence is NOT hash/lineage validation.\n'
           'Recent controller exits first, then all branch blockers; historical admissions are sampled last.\n'
           'JSON reads limited to 1 MiB each; oversize/unreadable records show errors.\n'
           'Log excerpts are bounded; this is not an atomic snapshot of live workers.\n'
           + upload_scope)
    yield '\nRECENT CONTROLLER AND CLEANUP LOGS\n'
    for pattern in ('runs/experiments/logs/console.mbpp.*.log', 'runs/experiments/logs/cleanup.mbpp.*.log'):
        logs = summary.recent(work, (pattern,))
        exits = {path: summary.controller_exit(work, path) for path in logs[:20]}
        failed = next((path for path in logs[:20]
                       if exits[path] and not re.match(r'rc=(?:0|130|143):', exits[path])), None)
        selected = [failed] if failed else []
        selected += [path for path in logs if path not in selected][:12-len(selected)]
        yield f'LOG SET {pattern}: total={len(logs)}; at most 12 recent logs, including latest failure among newest 20\n'
        for path in selected:
            if exits.get(path):
                yield f'CONTROLLER EXIT {path.name} {summary.clipped(exits[path], 650)}\n'
            yield from log_excerpt(work, path)
    for root in roots:
        yield f'\nROOT {root}\n'
        if not root.is_dir():
            yield 'ROOT MISSING\n'
            continue
        yield metadata(root, root / 'switch.json', ('dataset', 'selector', 'accounting', 'gate'))
        yield f'gate model present={int((root / "model.json").is_file())}\n'
        for name in ('.fit.lock', 'gate-fit/.task.lock', 'gate-fit/.cost.lock'):
            yield lease_record(root, root / name)
        for path in sorted((root / 'gate-fit').glob('failure.json')):
            yield metadata(root, path)
        for directory in sorted(root.glob('states/*/points/*/*')):
            if directory.name not in ARMS or not directory.is_dir():
                continue
            yield f'\nBRANCH {directory.relative_to(root)}\n'
            try:
                summary.checked(root, directory)
                yield f'charged deployment GPU-s={summary.charged(directory)}\n'
                path = directory / 'decision.json'
                yield metadata(root, path, ('binding', 'action', 'reason', 'budget_gpu_seconds',
                                            'measurement_gpu_seconds')) if path.exists() else f'MISSING {path.relative_to(root)}\n'
                for name in ('failure.json', 'progress.json',
                             'budget-recovery/review.json', 'budget-recovery/failure.json',
                             'budget-recovery/progress.json'):
                    path = directory / name
                    yield metadata(root, path) if path.exists() else f'MISSING {path.relative_to(root)}\n'
                yield lease_record(root, directory / '.task.lock')
                yield lease_record(root, directory / '.cost.lock')
                nested = {path.parent for name in ('progress.json', 'failure.json')
                          for path in directory.glob(f'**/{name}')}
                for phase_directory in sorted(nested):
                    if phase_directory == directory or any(
                        part.startswith('discarded') for part in phase_directory.relative_to(directory).parts
                    ):
                        continue
                    try:
                        summary.checked(root, phase_directory)
                    except (OSError, ValueError) as exc:
                        yield f'PHASE ERROR {phase_directory.relative_to(root)}: {exc}\n'
                        continue
                    for name in ('progress.json', 'failure.json'):
                        path = phase_directory / name
                        if path.exists():
                            yield metadata(root, path)
                    yield lease_record(root, phase_directory / '.cost.lock')
                    worker_logs.extend(phase_logs(root, phase_directory))
                for name, keys in (
                    ('result.json', ('complete', 'completed_steps', 'stop_reason', 'used_gpu_seconds', 'budget_gpu_seconds')),
                    ('curve.json', ('result_sha256',)),
                    ('budget-recovery/plan.json', ('schema', 'canonical_complete', 'purpose', 'completed_steps',
                                                  'start_step', 'used_gpu_seconds', 'budget_gpu_seconds', 'over_budget_gpu_seconds')),
                    ('budget-recovery/result.json', ('schema', 'canonical_complete', 'evaluation_complete',
                                                    'plan_sha256', 'used_gpu_seconds', 'budget_gpu_seconds', 'over_budget_gpu_seconds'))):
                    path = directory / name
                    yield metadata(root, path, keys) if path.exists() else f'MISSING {path.relative_to(root)}\n'
                inventories.append((root, directory / 'policy'))
                worker_logs.extend(phase_logs(root, directory))
            except (OSError, ValueError) as exc:
                yield f'BRANCH ERROR {exc}\n'
        for path in sorted(root.glob('prefixes/seed-*/segment-*/failure.json')):
            yield metadata(root, path)
        for path in sorted(root.glob('states/*/failure.json')):
            yield metadata(root, path)
        paths = summary.recent(root, ('node-preflight/*/admission.json',))
        yield f'ADMISSION HISTORY total={len(paths)}; latest {ADMISSION_LIMIT} records only; older records omitted\n'
        admissions.extend((root, path) for path in paths[:ADMISSION_LIMIT])
    yield '\nWORKER ERROR EXCERPTS\n'
    for root, path in worker_logs:
        yield f'ROOT {root}\nERROR EXCERPT {path.relative_to(root)} (bounded log context)\n'
        yield summary.error_excerpt(root, path) + '\n'
    yield '\nCHECKPOINT INVENTORIES\n'
    for root, policy in inventories:
        yield f'ROOT {root}\n'
        yield from policy_inventory(root, policy)
    yield '\nRECENT ADMISSIONS (bounded historical sample)\n'
    for root, path in admissions:
        yield f'ROOT {root}\n'
        yield summary.clipped(metadata(root, path), ADMISSION_BYTES) + '\n'


def write_single(chunks, destination, *, prefix='mbpp-why'):
    """Stream all diagnostic sections into one atomically published UTF-8 file."""
    if not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', prefix):
        raise ValueError('invalid diagnostic filename prefix')
    destination.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix=prefix + '-', dir=destination))
    path = folder / f'{prefix}-single.txt'
    temporary = folder / f'.{prefix}-single.tmp'
    try:
        with temporary.open('x', encoding='utf-8') as handle:
            handle.write(f'{prefix.upper().replace("-", " ")} single file; set={folder.name}\n'
                         'No total output-size cap.\n\n')
            for chunk in chunks:
                handle.write(chunk)
        temporary.rename(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        folder.rmdir()
        raise
    return [path]


def write_parts(chunks, destination):
    """Bound file count and bytes; explicitly mark any omitted trailing content."""
    destination.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix='mbpp-why-', dir=destination))
    paths, buffer = [], b''
    # Reserve room for both the part header and the final truncation notice.
    capacity = PART_BYTES - 512

    def save(data):
        path = folder / f'mbpp-why-{len(paths) + 1:03d}.txt'
        header = (f'MBPP WHY part {len(paths) + 1}; set={folder.name}\n'
                  'Parts concatenate in numeric order; content may continue from the preceding part.\n\n').encode()
        if len(header) > 256:
            raise ValueError('diagnostic part header exceeds reserved space')
        path.write_bytes(header + data)
        paths.append(path)

    for chunk in chunks:
        buffer += chunk.encode('utf-8')
        while len(buffer) > capacity:
            end = capacity
            while end < len(buffer) and buffer[end] & 0xC0 == 0x80:
                end -= 1
            if len(paths) == MAX_PARTS - 1:
                save(buffer[:end] + b'\n[EXPORT LIMIT: remaining content omitted to keep at most 3 TXT files '
                     b'below 2 MB each. Source files unchanged; this is NOT a complete export.]\n')
                return paths
            save(buffer[:end])
            buffer = buffer[end:]
    if buffer:
        save(buffer)
    return paths
