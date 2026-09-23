"""Read-only selector-pair lock evidence; never stop a process or edit a run.

Standard library only. A small report is saved outside the experiment so this
can inspect old controllers without importing or migrating their frozen code.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import stat
import subprocess
import tempfile
import time

MAX_BYTES = 4096
QUEUE_REPORT_BYTES = 64 * 1024
COST_REPORT_BYTES = 1024 * 1024 - MAX_BYTES - QUEUE_REPORT_BYTES - 2


def bounded(text):
    raw = text.encode()
    if len(raw) <= MAX_BYTES:
        return text
    footer = b'\n[Further observations omitted; report capped at 4 KiB.]\n'
    return raw[:MAX_BYTES - len(footer)].decode(errors='ignore') + footer.decode()


def read_small(path, limit=65536):
    with path.open('rb') as handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise ValueError('metadata read limit exceeded')
    return data


def local_owners(lock, proc, *, kind='WRITE'):
    stat = lock.stat()
    identity = (os.major(stat.st_dev), os.minor(stat.st_dev), stat.st_ino)
    owners = []
    for row in read_small(proc / 'locks', 1048576).decode(errors='replace').splitlines():
        fields = row.split()
        if len(fields) < 8 or '->' in fields or fields[3] != kind:
            continue
        try:
            major, minor, inode = fields[5].split(':')
            pid = int(fields[4])
            match = (int(major, 16), int(minor, 16), int(inode)) == identity
        except ValueError:
            continue
        if match and pid > 0:
            owners.append(pid)
    return sorted(set(owners))[:4]


def process_label(proc, pid):
    # Do not copy full argv or environ: either could contain unrelated secrets.
    words = read_small(proc / str(pid) / 'cmdline').decode(errors='replace').split('\0')
    for index, word in enumerate(words):
        if Path(word).name in {'selector_pair_gpu.py', 'queue_selector_pair_gpu.py', 'run_selector_pair.sh'}:
            following = words[index + 1] if index + 1 < len(words) else ''
            modes = {'run', 'develop', 'test', 'freeze', 'prepare', 'ensure-prepared',
                     'init', 'fit', 'report', 'status', 'check-code', 'check-running'}
            return f'{Path(word).name} mode={following if following in modes else "unknown"}'
    return 'unrecognized process; do not assume it belongs to selector_pair'


def observations(root, limit=8):
    """Bounded metadata traversal, without reading tensors, logs or ledgers."""
    root = Path(root)
    resolved = root.resolve()
    pending = [(root / name, 0) for name in ('node-preflight', 'queue-workers', 'branches')]
    seen, rows, recorded = 0, [], set()

    def remember(path):
        if path in recorded or path.is_symlink():
            return
        try:
            if not path.resolve().is_relative_to(resolved):
                return
            value = json.loads(read_small(path))
            updated = float(value.get('updated', 0))
            if not math.isfinite(updated):
                return
            rows.append((updated, path.relative_to(root), value))
            recorded.add(path)
        except (OSError, ValueError, TypeError, AttributeError):
            return

    # Fixed meter paths must not compete with historical admission receipts or
    # thousands of per-rollout/cost files for the fallback traversal's budget.
    for seed in (3, 4):
        for step in (25, 50, 100):
            remember(root / 'sr-gc' / f's{seed}-t{step}' / 'progress.json')
    for branch in ('on_policy', 'cached', 'adaptive-on_policy', 'adaptive-cached'):
        for seed in range(5):
            for step in (25, 50, 100):
                point = root / 'branches' / branch / 'states' / f's{seed}-t{step}' / 'points' / f'view-{step}'
                for name in ('curve-parent', 'measurement', 'gate_measurement',
                             'selection_reduced', 'selection_full', 'random_full'):
                    directory = point / name
                    remember(directory / 'progress.json')
                    remember(directory / 'curve/progress.json')
    deadline = time.monotonic() + 2
    while pending and seen < 2048 and time.monotonic() < deadline:
        directory, depth = pending.pop()
        if directory.is_symlink():
            continue
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    seen += 1
                    if seen > 2048 or time.monotonic() >= deadline:
                        break
                    if entry.is_symlink():
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        if depth < 8 and entry.name not in {'policy', 'curve-checkpoints', 'selector-work',
                                                          'discarded', 'cost-events', 'rollouts', 'rollout'}:
                            pending.append((Path(entry.path), depth + 1))
                    elif entry.name == 'progress.json' or (directory.name == 'queue-workers' and entry.name.endswith('.json')):
                        remember(Path(entry.path))
        except OSError:
            continue
    ordered = sorted(rows, key=lambda row: row[0], reverse=True)
    return ordered if limit is None else ordered[:limit]


def collect(root, proc=Path('/proc'), *, uncapped=False):
    root = Path(root)
    host = socket.gethostname()
    lines = ['SELECTOR PAIR LOCK DIAGNOSTIC', f'LOCAL_HOST {host}', f'ROOT {root}',
             'READ-ONLY. No process stopped; no locks/results/checkpoints/costs changed.',
             'Checkout revision is NOT the revision of an already-running Python process.']
    try:
        revision = subprocess.run(['git', 'rev-parse', '--short=12', 'HEAD'], capture_output=True,
                                  text=True, timeout=3, check=False)
        lines.append(f'CHECKOUT {revision.stdout.strip() if revision.returncode == 0 else "unknown"}')
    except (OSError, subprocess.TimeoutExpired):
        lines.append('CHECKOUT unknown')
    lock = root / '.pair.lock'
    try:
        with lock.open('rb') as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                lines.append('ROOT_LOCK blocked: an exclusive holder prevents new queue workers')
            else:
                lines.append('ROOT_LOCK shared-compatible at this instant; no exclusive holder detected')
                fcntl.flock(handle, fcntl.LOCK_UN)
    except FileNotFoundError:
        lines.append('ROOT_LOCK missing: no lock/root was created; verify mounted storage and configured root')
    except OSError as exc:
        lines.append(f'ROOT_LOCK unreadable: {type(exc).__name__} errno={exc.errno}')
    try:
        owners = local_owners(lock, Path(proc))
    except (OSError, ValueError):
        owners = []
    for pid in owners:
        try:
            label = process_label(Path(proc), pid)
        except (OSError, ValueError):
            label = 'process details unavailable'
        lines.append(f'OWNER confirmed-local kernel snapshot host={host} pid={pid} {label}')
    if not owners:
        lines.append('OWNER not visible locally; remote/NFS/namespace owner may exist. NOT proof of a dead owner.')
    now = time.time()
    for updated, path, value in observations(root, limit=None if uncapped else 8):
        age = now - updated
        fresh = value.get('state') == 'running' and -5 <= age < 60
        # These fields are observations only, never an authorization to signal.
        text = (f'OBSERVED host={value.get("host", "?")} state={value.get("state", "?")} '
                f'phase={value.get("phase", value.get("stage", "?"))} age={age:.0f}s path={path}; '
                + ('recent progress' if fresh else 'not confirmed running')
                + '; NOT confirmed lock ownership')
        lines.append(text[:700].replace('\n', ' ').replace('\r', ' '))
    lines.append('Do not delete .pair.lock: unlinking does not release an existing kernel lock.')
    output = '\n'.join(lines) + '\n'
    return output if uncapped else bounded(output)


def queue_report(root, *, uncapped=False):
    """Inspect existing state/branch leases without creating or breaking locks."""
    root = Path(root).resolve()
    lines = ['SELECTOR PAIR TASK WAIT EVIDENCE',
             'Lock probes are momentary observations, not permission to stop an owner.',
             'Progress and worker records may be historical; timestamps do not prove ownership.']

    def contained(path):
        if not path.resolve().is_relative_to(root):
            raise ValueError('metadata escapes Pair root')

    def lock_record(path):
        try:
            contained(path)
            with path.open('rb') as handle:
                try:
                    fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
                except BlockingIOError:
                    return 'exclusive-holder'
                fcntl.flock(handle, fcntl.LOCK_UN)
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return 'shared-holder-or-owner-changed'
                fcntl.flock(handle, fcntl.LOCK_UN)
                return 'free-at-probe'
        except FileNotFoundError:
            return 'missing'
        except (OSError, ValueError, RuntimeError) as exc:
            return f'unknown:{type(exc).__name__}'

    def observe(path, keys):
        try:
            contained(path)
            value = json.loads(read_small(path))
            if not isinstance(value, dict):
                raise ValueError('expected object')
            value = {key: value[key] for key in keys if key in value}
            lines.append(f'RECORD {path.relative_to(root)} ' + json.dumps(value, ensure_ascii=True))
        except FileNotFoundError:
            pass
        except (OSError, ValueError, RuntimeError) as exc:
            lines.append(f'UNREADABLE {path.relative_to(root)}: {type(exc).__name__}')

    def metadata(path):
        relative = path.relative_to(root)
        try:
            contained(path)
            if not stat.S_ISREG(path.stat().st_mode):
                raise ValueError('metadata is not a regular file')
            raw = read_small(path)
            lines.append(f'FILE {relative} bytes={len(raw)} sha256={hashlib.sha256(raw).hexdigest()}')
            lines.append(raw.decode('utf-8'))
        except FileNotFoundError:
            lines.append(f'MISSING {relative}')
        except (OSError, ValueError, RuntimeError) as exc:
            lines.append(f'UNREADABLE {relative}: {type(exc).__name__}: {str(exc)[:256]}')

    for name in ('.pair.lock', '.pair-runtime.lock', '.pair-barrier.lock', '.fit.lock',
                 'gate-fit/.task.lock', 'gate-fit/.cost.lock'):
        lines.append(f'GLOBAL_LOCK {name} {lock_record(root / name)}')
    for branch in ('on_policy', 'cached', 'adaptive-on_policy', 'adaptive-cached'):
        for name in ('.fit.lock', 'gate-fit/.task.lock', 'gate-fit/.cost.lock'):
            path = root / 'branches' / branch / name
            lines.append(f'GLOBAL_LOCK {path.relative_to(root)} {lock_record(path)}')

    if uncapped:
        lines += ['SELECTOR PAIR SAVED RUNTIME AND QUEUE EVIDENCE',
                  'Raw metadata and byte hashes, not validation or current process ownership.',
                  'Each metadata file is limited to 64 KiB; omitted/unreadable files are explicit.',
                  'Queue timestamps may differ across servers; ordering is not a liveness test.']
        for name in ('pair.json', 'model.json', 'test-decisions.json', 'gate-fit/failure.json',
                     'gate-fit/progress.json'):
            metadata(root / name)
        for directory in (root, *(root / 'branches' / branch for branch in
                                  ('on_policy', 'cached', 'adaptive-on_policy', 'adaptive-cached'))):
            if directory != root:
                for name in ('switch.json', 'model.json', 'gate-fit/failure.json', 'gate-fit/progress.json'):
                    metadata(directory / name)
            # Always expose the receipt implicated in historical Pair errors,
            # even when it is now missing or hidden behind a broken link.
            metadata(directory / 'mbpp-branch-quarantine-runtime.json')
            try:
                contained(directory)
                with os.scandir(directory) as entries:
                    paths = []
                    for index, entry in enumerate(entries):
                        if index >= 2048:
                            lines.append(f'RUNTIME_SCAN {directory.relative_to(root)} truncated at 2048 entries')
                            break
                        if entry.name.endswith('-runtime.json') and entry.name != 'mbpp-branch-quarantine-runtime.json':
                            paths.append(Path(entry.path))
                for path in sorted(paths)[:128]:
                    metadata(path)
                if len(paths) > 128:
                    lines.append(f'RUNTIME_SCAN {directory.relative_to(root)} omitted={len(paths)-128} records')
            except FileNotFoundError:
                pass
            except (OSError, ValueError, RuntimeError) as exc:
                lines.append(f'UNREADABLE {directory.relative_to(root)} runtime scan: {type(exc).__name__}')
        workers = []
        try:
            directory = root / 'queue-workers'
            contained(directory)
            with os.scandir(directory) as entries:
                for index, entry in enumerate(entries):
                    if index >= 2048:
                        lines.append('QUEUE_SCAN truncated at 2048 entries; omitted count is a lower bound')
                        break
                    if entry.name.endswith('.json'):
                        try:
                            workers.append((entry.stat(follow_symlinks=False).st_mtime_ns, Path(entry.path)))
                        except OSError as exc:
                            lines.append(f'UNREADABLE queue-workers/{entry.name}: {type(exc).__name__}')
            lines.append(f'QUEUE_RECORDS found={len(workers)} exported={min(len(workers), 256)} '
                         f'omitted={max(0, len(workers)-256)}; no live-owner claim')
            for _, path in sorted(workers, reverse=True)[:256]:
                metadata(path)
        except FileNotFoundError:
            lines.append('QUEUE_RECORDS missing; no ownership inference')
        except (OSError, ValueError, RuntimeError) as exc:
            lines.append(f'UNREADABLE queue-workers: {type(exc).__name__}')

    for stage, seeds in (('development', range(3)), ('test', range(3, 5))):
        for seed in seeds:
            for step in (25, 50, 100):
                state = f's{seed}-t{step}'
                folder = root / stage / state
                lines.append(f'STATE {stage}/{state} state-lock={lock_record(folder / ".state.lock")} '
                             f'prepare-lock={lock_record(folder / ".prepare-state.lock")}')
                if uncapped:
                    metadata(folder / 'result.json')
                branches = (('on_policy', 'selection_reduced'), ('cached', 'selection_reduced')) if stage == 'development' else (
                    ('on_policy', 'selection_full'), ('cached', 'selection_full'),
                    ('adaptive-on_policy', 'selection_full'), ('adaptive-cached', 'selection_full'),
                    ('on_policy', 'random_full'))
                for branch, arm in branches:
                    point = root / 'branches' / branch / 'states' / state / 'points' / f'view-{step}'
                    directory = point / arm
                    lock = folder / 'queue-branches' / f'{branch}--{arm}.lock'
                    if uncapped:
                        metadata(lock.with_suffix('.json'))
                    lines.append(f'TASK {stage}/{state}/{branch}/{arm} branch-lock={lock_record(lock)} '
                                 f'task-lock={lock_record(directory / ".task.lock")} '
                                 f'parent-curve-lock={lock_record(point / "curve-parent/.point.lock")}')
                    if uncapped:
                        for location in (directory, directory / 'curve', point / 'curve-parent'):
                            path = location / '.cost.lock'
                            lines.append(f'METER_LOCK {path.relative_to(root)} {lock_record(path)}')
                        for name in ('result.json', 'result.sha256.json', 'curve.json'):
                            metadata(directory / name)
                    for path in (directory / 'pair-attempt.json', directory / 'failure.json',
                                 directory / 'progress.json', directory / 'curve/progress.json',
                                 point / 'curve-parent/progress.json'):
                        observe(path, ('state', 'phase', 'event_id', 'seconds', 'updated', 'host',
                                       'pid', 'attempt', 'error', 'reason'))
    raw = ('\n'.join(lines) + '\n').encode()
    if uncapped or len(raw) <= QUEUE_REPORT_BYTES:
        return raw.decode()
    footer = b'\n[Further task observations omitted; task evidence capped at 64 KiB.]\n'
    return raw[:QUEUE_REPORT_BYTES - len(footer)].decode(errors='ignore') + footer.decode()


def cost_report(root, *, uncapped=False):
    """Export interrupted-event evidence, never estimate or close an event."""
    root = Path(root).resolve()
    lines = ['SELECTOR PAIR COST EVIDENCE', f'ROOT {root}',
             'READ-ONLY. No cost repair, inferred durations, GPU work or process termination.',
             'Saved metadata only, not independent scientific validation or proof an owner stopped.',
             'Only ledgers with unmatched events, parse errors or saved cost failures are expanded.']
    examined = expanded = 0

    def checked(path, limit):
        if not path.resolve().is_relative_to(root):
            raise ValueError('metadata escapes Pair root')
        return read_small(path, limit)

    def metadata(path):
        relative = path.relative_to(root)
        try:
            raw = checked(path, 65536)
            lines.append(f'FILE {relative} bytes={len(raw)} sha256={hashlib.sha256(raw).hexdigest()}')
            lines.append(raw.decode('utf-8'))
        except FileNotFoundError:
            lines.append(f'MISSING {relative}')
        except (OSError, ValueError, UnicodeError) as exc:
            lines.append(f'UNREADABLE {relative}: {type(exc).__name__}: {exc}')

    for branch in ('on_policy', 'cached', 'adaptive-on_policy', 'adaptive-cached'):
        for seed in range(5):
            for step in (25, 50, 100):
                point = root / 'branches' / branch / 'states' / f's{seed}-t{step}' / 'points' / f'view-{step}'
                for arm in ('selection_reduced', 'selection_full', 'random_full',
                            'measurement', 'gate_measurement', 'curve-parent'):
                    for directory in (point / arm, point / arm / 'curve'):
                        path = directory / 'cost.jsonl'
                        try:
                            raw = checked(path, 2 * 1024 * 1024)
                        except FileNotFoundError:
                            continue
                        except (OSError, ValueError) as exc:
                            lines.append(f'UNREADABLE {path.relative_to(root)}: {type(exc).__name__}: {exc}')
                            continue
                        examined += 1
                        events, errors = [], []
                        for index, row in enumerate(raw.splitlines(), 1):
                            if not row.strip():
                                continue
                            try:
                                value = json.loads(row)
                                if not isinstance(value, dict):
                                    raise ValueError('event is not an object')
                                events.append(value)
                            except (ValueError, UnicodeError) as exc:
                                errors.append(f'line {index}: {exc}; bytes_hex={row[:4096].hex()}')
                        starts = {str(e.get('event_id')) for e in events if e.get('state') == 'started'}
                        finishes = {str(e.get('event_id')) for e in events if e.get('state') == 'finished'}
                        pending = sorted(starts - finishes)
                        failures = []
                        for name in ('pair-attempt.json', 'failure.json'):
                            try:
                                saved = json.loads(checked(directory / name, 65536))
                                error = str(saved.get('error', '')).lower()
                                if any(word in error for word in ('cost', 'budget', 'allocation')):
                                    failures.append(name)
                            except (OSError, ValueError, AttributeError):
                                pass
                        if not pending and not errors and not finishes - starts and not failures:
                            continue
                        expanded += 1
                        lines.append(f'\nLEDGER {path.relative_to(root)} bytes={len(raw)} sha256={hashlib.sha256(raw).hexdigest()}')
                        lines.append(json.dumps({'events': len(events), 'open_event_ids': pending,
                                                 'missing_start_ids': sorted(finishes - starts), 'parse_errors': errors}))
                        for event in events:
                            if str(event.get('event_id')) in pending:
                                lines.append('OPEN_EVENT ' + json.dumps(event, sort_keys=True))
                        for event in pending:
                            if not event or Path(event).name != event or event in {'.', '..'}:
                                lines.append('INVALID event id; receipt paths not traversed')
                                continue
                            metadata(directory / 'cost-events' / f'{event}.json')
                            metadata(directory / 'pending-costs' / f'{event}.json')
                        metadata(directory / 'progress.json')
                        metadata(directory / 'decision.json')
                        metadata(directory / 'budget-recovery/result.json')
                        for name in failures:
                            metadata(directory / name)
                        policy = directory / 'policy'
                        if policy.resolve().is_relative_to(root):
                            candidates = [policy, *sorted(policy.glob('checkpoint-*')),
                                          *sorted(policy.glob('.checkpoint-*.tmp')),
                                          *sorted((policy / 'curve-checkpoints').glob('step-*'))]
                            for candidate in candidates[:128]:
                                if not candidate.resolve().is_relative_to(root):
                                    lines.append('POLICY path escapes root; skipped')
                                    continue
                                inventory = {}
                                for name in ('adapter_config.json', 'adapter_model.safetensors', 'optimizer.pt',
                                             'grpo_stats.jsonl', 'checkpoint_state.json', 'policy_train.json', 'budget_stop.json'):
                                    artifact = candidate / name
                                    try:
                                        if not artifact.resolve().is_relative_to(root):
                                            inventory[name] = 'outside-root; not inspected'
                                        else:
                                            inventory[name] = artifact.stat().st_size if artifact.is_file() else None
                                    except OSError as exc:
                                        inventory[name] = 'unreadable: ' + type(exc).__name__
                                lines.append(f'POLICY {candidate.relative_to(root)} bytes=' + json.dumps(inventory, sort_keys=True))
                            if len(candidates) > 128:
                                lines.append('Additional policy inventory candidates omitted (128 limit).')
    lines.append(f'LEDGERS examined={examined} expanded={expanded}; file presence is not hash/lineage certification.')
    raw = ('\n'.join(lines) + '\n').encode()
    if uncapped or len(raw) <= COST_REPORT_BYTES:
        return raw.decode()
    footer = b'\n[COST EVIDENCE OMITTED: single TXT capped at 1 MiB including lock summary.]\n'
    return raw[:COST_REPORT_BYTES - len(footer)].decode(errors='ignore') + footer.decode()


def main():
    work = Path(os.environ.get('OM_WORK', f'/group-volume/{os.environ.get("OM_USER", "minsoo3.kim")}/offpolicy-misranking'))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(os.environ.get('PAIR_ROOT', str(work / 'runs/selector-pair-v1'))))
    parser.add_argument('--report-dir', type=Path, default=Path.home())
    parser.add_argument('--costs', action='store_true', help='include interrupted-event evidence in one upload-sized TXT')
    args = parser.parse_args()
    output = collect(args.root)
    if args.costs:
        output += '\n' + queue_report(args.root) + '\n' + cost_report(args.root)
    print(output, end='')
    try:
        with tempfile.NamedTemporaryFile(prefix='selector-pair-cost-' if args.costs else 'selector-pair-lock-', suffix='.txt',
                                         dir=args.report_dir, mode='wb', delete=False) as handle:
            handle.write(output.encode())
            saved = Path(handle.name).resolve()
    except OSError as exc:
        print(f'[report-save-failed] {exc}')
        return 2
    print(f'[saved] {saved} ({len(output.encode())} bytes; send this TXT)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
