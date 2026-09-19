"""Read-only selector-pair lock evidence; never stop a process or edit a run.

Standard library only. A small report is saved outside the experiment so this
can inspect old controllers without importing or migrating their frozen code.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time

MAX_BYTES = 4096


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


def local_owners(lock, proc):
    stat = lock.stat()
    identity = (os.major(stat.st_dev), os.minor(stat.st_dev), stat.st_ino)
    owners = []
    for row in read_small(proc / 'locks', 1048576).decode(errors='replace').splitlines():
        fields = row.split()
        if len(fields) < 8 or '->' in fields or fields[3] != 'WRITE':
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
        if Path(word).name in {'selector_pair_gpu.py', 'run_selector_pair.sh'}:
            following = words[index + 1] if index + 1 < len(words) else ''
            modes = {'run', 'develop', 'test', 'freeze', 'prepare', 'ensure-prepared',
                     'init', 'fit', 'report', 'status', 'check-code', 'check-running'}
            return f'{Path(word).name} mode={following if following in modes else "unknown"}'
    return 'unrecognized process; do not assume it belongs to selector_pair'


def observations(root, limit=8):
    """Bounded metadata traversal, without reading tensors, logs or ledgers."""
    pending = [(root / name, 0) for name in ('branches', 'queue-workers', 'node-preflight')]
    seen, rows = 0, []
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
                        if depth < 8 and entry.name not in {'policy', 'curve-checkpoints', 'selector-work', 'discarded'}:
                            pending.append((Path(entry.path), depth + 1))
                    elif entry.name == 'progress.json' or (directory.name == 'queue-workers' and entry.name.endswith('.json')):
                        try:
                            value = json.loads(read_small(Path(entry.path)))
                            updated = float(value.get('updated', 0))
                            if not math.isfinite(updated):
                                continue
                            rows.append((updated, Path(entry.path).relative_to(root), value))
                        except (OSError, ValueError, TypeError, AttributeError):
                            continue
        except OSError:
            continue
    ordered = sorted(rows, key=lambda row: row[0], reverse=True)
    return ordered if limit is None else ordered[:limit]


def collect(root, proc=Path('/proc')):
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
    for updated, path, value in observations(root):
        age = now - updated
        fresh = value.get('state') == 'running' and -5 <= age < 60
        # These fields are observations only, never an authorization to signal.
        text = (f'OBSERVED host={value.get("host", "?")} state={value.get("state", "?")} '
                f'phase={value.get("phase", value.get("stage", "?"))} age={age:.0f}s path={path}; '
                + ('recent progress' if fresh else 'not confirmed running')
                + '; NOT confirmed lock ownership')
        lines.append(text[:700].replace('\n', ' ').replace('\r', ' '))
    lines.append('Do not delete .pair.lock: unlinking does not release an existing kernel lock.')
    return bounded('\n'.join(lines) + '\n')


def main():
    work = Path(os.environ.get('OM_WORK', f'/group-volume/{os.environ.get("OM_USER", "minsoo3.kim")}/offpolicy-misranking'))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(os.environ.get('PAIR_ROOT', str(work / 'runs/selector-pair-v1'))))
    parser.add_argument('--report-dir', type=Path, default=Path.home())
    args = parser.parse_args()
    output = collect(args.root)
    print(output, end='')
    try:
        with tempfile.NamedTemporaryFile(prefix='selector-pair-lock-', suffix='.txt',
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
