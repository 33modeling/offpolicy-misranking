"""One bounded MBPP diagnostic attachment; never starts or repairs experiments."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from selection_switch_errors import log_tail, utc_time

MAX_BYTES = 16 * 1024
ROOT_BYTES = 4000


def clipped(value, limit):
    data = str(value).encode('utf-8', errors='replace')
    if len(data) <= limit:
        return data.decode('utf-8')
    marker = b'\n[... omitted ...]\n'
    head = (limit - len(marker)) // 2
    tail = limit - len(marker) - head
    return data[:head].decode('utf-8', errors='ignore') + marker.decode() + data[-tail:].decode('utf-8', errors='ignore')


def checked(root, path):
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('file points outside the requested root')
    return path


def record(root, path):
    try:
        checked(root, path)
        if not path.exists():
            return {}
        with path.open('rb') as handle:
            data = handle.read(1024 * 1024 + 1)
        if len(data) > 1024 * 1024:
            raise ValueError('JSON exceeds the 1 MiB diagnostic read limit')
        result = json.loads(data)
        if not isinstance(result, dict):
            raise TypeError('expected a JSON object')
        return result
    except (OSError, ValueError, TypeError) as exc:
        return {'_read_error': str(exc)}


def recent(root, patterns):
    found = {}
    for pattern in patterns:
        for path in root.glob(pattern):
            try:
                checked(root, path)
                if path.is_file():
                    found[path] = path.stat().st_mtime_ns
            except (OSError, ValueError):
                continue
    return sorted(found, key=found.get, reverse=True)


def tail(root, path, limit=900):
    try:
        checked(root, path)
        return clipped(log_tail(path, 16), limit)
    except (OSError, ValueError) as exc:
        return f'[log unavailable] {exc}'


def error_excerpt(root, path):
    try:
        checked(root, path)
        lines = log_tail(path, 250).splitlines()
        for pattern in (r'\bcuda (?:failure|error)\b|out of memory|illegal memory',
                        r'cuda.*(?:failed)|nccl.*(?:warn|unhandled)'):
            for index, line in enumerate(lines):
                if re.search(pattern, line, re.IGNORECASE):
                    # Keep the original CUDA warning even if torchrun's generic
                    # ChildFailedError and shutdown messages follow it much later.
                    return clipped('\n'.join(lines[max(0, index-3):index+9]), 900)
        return clipped('\n'.join(lines[-16:]), 900)
    except (OSError, ValueError) as exc:
        return f'[log unavailable] {exc}'


def root_summary(root):
    lines = [f'\nROOT {root.name}']
    manifest = record(root, root / 'switch.json')
    lines.append('protocol: ' + str({key: manifest.get(key) for key in ('dataset', 'selector', 'accounting', 'gate')}))
    if '_read_error' in manifest:
        lines.append('manifest unreadable: ' + manifest['_read_error'])
    failures = recent(root, ('prefixes/seed-*/segment-*/failure.json',
        'states/*/points/*/*/failure.json', 'states/*/failure.json', 'gate-fit/failure.json'))
    lines.append(f'recorded failures: {len(failures)}; newest two below (historical, not proof of a live failure)')
    for path in failures[:2]:
        directory = path.parent
        failure = record(root, path)
        progress = record(root, directory / 'progress.json')
        decision = record(root, directory / 'decision.json')
        lines.append(f'\nFAIL {directory.relative_to(root)}')
        lines.append(f"host={failure.get('host', '?')} utc={utc_time(failure.get('time'))}")
        lines.append(clipped(f"last progress: phase={progress.get('phase', '?')} state={progress.get('state', '?')} "
            f"host={progress.get('host', '?')} utc={utc_time(progress.get('updated'))}", 250))
        artifacts = ('fresh-r/selected.sha256.json', 'cached-select/selection.sha256.json',
                     'execution.sha256.json', 'policy/budget_stop.json', 'result.sha256.json')
        lines.append('files present (not hash-validated): ' + ', '.join(name for name in artifacts
            if (directory / name).is_file()))
        lines.append(f"branch budget GPU-s={decision.get('budget_gpu_seconds', '?')}")
        lines.append(clipped(failure.get('error', failure.get('_read_error', 'unknown failure')), 800))
        phase = progress.get('phase')
        logs = recent(directory, (f'{phase}-*.log',)) if isinstance(phase, str) and re.fullmatch(r'[\w-]+', phase) else []
        if logs:
            # Prefer a shard tail containing the underlying exception, not an idle sibling.
            excerpts = [(path, error_excerpt(root, path)) for path in logs[:4]]
            path, text = next(((p, t) for p, t in excerpts if re.search(r'Error:|Exception:|Traceback|cuda failure|nccl.*warn', t, re.IGNORECASE)), excerpts[0])
            lines.append(f'LOG {path.relative_to(root)}\n{text}')
    admissions = recent(root, ('node-preflight/*/admission.json',))
    if admissions:
        value = record(root, admissions[0])
        lines.append(clipped(f"last admission: host={value.get('host', '?')} state={value.get('state', '?')} "
            f"runtime={value.get('runtime_commit', '?')} diagnosis={value.get('diagnosis', value.get('_read_error', ''))}", 400))
        lines.append(clipped(value.get('error', ''), 350))
        attempts = value.get('attempts', [])
        if isinstance(attempts, list):
            for attempt in attempts[-2:]:
                if not isinstance(attempt, dict):
                    continue
                lines.append(clipped(f"probe {attempt.get('name', '?')}: {attempt.get('error') or 'passed'}", 550))
                ranks = attempt.get('ranks', [])
                if isinstance(ranks, list):
                    for rank in [r for r in ranks if isinstance(r, dict) and r.get('error')][:1]:
                        lines.append(clipped(f"rank={rank.get('rank')} torch={rank.get('torch')} cuda={rank.get('cuda_runtime')} "
                            f"nccl={rank.get('nccl')}: {rank.get('error')}", 700))
    if not failures:
        lines.append('No saved branch failure found; missing logs do not prove success.')
        logs = recent(root, ('logs/launcher.*.log', 'logs/console.*.log'))
        if logs:
            lines.append(f'LOG {logs[0].relative_to(root)}\n{tail(root, logs[0], 1800)}')
    return clipped('\n'.join(lines), ROOT_BYTES)


def report(work, roots):
    try:
        commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=3, check=False).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        commit = ''
    sections = [(f'MBPP FAILURE SUMMARY\nUTC {datetime.now(timezone.utc).isoformat(timespec="seconds")}\n'
        f'checkout={commit or "unknown"} (running workers may use an older snapshot)\n'
        'Limit: 16 KiB. Only latest failures and short log tails; no rollouts, model data or full cost ledgers.')]
    sections.extend(root_summary(root) for root in roots)
    logs = recent(work, ('runs/experiments/logs/console.mbpp.*.log',))
    for path in logs[:2]:
        sections.append(f'\nNODE {path.name}\n{tail(work, path, 1400)}')
    return clipped('\n'.join(sections) + '\n', MAX_BYTES)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--root', type=Path, action='append', required=True)
    args = parser.parse_args()
    if len(args.root) > 3:
        parser.error('at most three MBPP suite roots')
    text = report(args.work, args.root)
    destination = args.work / 'reports/selection-switch'
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', prefix='mbpp-why-', suffix='.txt',
                                     dir=destination, delete=False) as handle:
        handle.write(text)
    print(f'[size] {len(text.encode("utf-8"))} bytes (maximum {MAX_BYTES})')
    print(f'[saved] {handle.name}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
