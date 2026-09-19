"""One bounded MBPP diagnostic attachment; never starts or repairs experiments."""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from selection_switch_errors import log_tail, utc_time
from _status_summary import accounting_label, gate_label, mbpp_suite_label, selector_label

MAX_BYTES = 16 * 1024
ROOT_BYTES = 4000
STORAGE_BYTES = 4096


def charged(directory):
    """Small ledger-only audit; never open a model, rollout, or worker log."""
    path = directory / 'cost.jsonl'
    if not path.exists():
        return 0.
    try:
        checked(directory, path)
        if path.stat().st_size > 2 * 1024 * 1024:
            return 'unknown (ledger >2 MiB)'
        events = {}
        with path.open('rb') as handle:
            data = handle.read(2 * 1024 * 1024 + 1)
        if len(data) > 2 * 1024 * 1024:
            return 'unknown (ledger grew >2 MiB)'
        for line in data.splitlines():
            row = json.loads(line)
            state, key = row['state'], row['event_id']
            if state not in ('started', 'finished'):
                raise ValueError('unknown event state')
            pair = events.setdefault(key, {})
            if state in pair and pair[state] != row:
                raise ValueError('conflicting cost events')
            pair[state] = row
        total = 0.
        for pair in events.values():
            if set(pair) != {'started', 'finished'}:
                return 'unknown (open/missing cost event)'
            end = pair['finished']
            value = end['allocated_gpu_seconds']
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError('invalid charge')
            if end.get('ledger') != pair['started'].get('ledger'):
                raise ValueError('ledger mismatch')
            if end.get('ledger') != 'reporting':
                total += value
        return round(total, 3)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return 'unknown (' + str(exc)[:100] + ')'


def saved_inventory(directory):
    """Presence counts only; explicitly not checkpoint/hash certification."""
    policy = directory / 'policy'
    checkpoints = sorted(policy.glob('checkpoint-*/adapter_model.safetensors'))
    latest = max((p.parent.name for p in checkpoints),
                 key=lambda name: int(name.split('-')[-1]) if name.split('-')[-1].isdigit() else -1,
                 default='none')
    scoring = directory / 'fresh-r'
    return (f"policy={int((policy / 'adapter_model.safetensors').is_file())} "
            f'checkpoints={len(checkpoints)} latest={latest} '
            f"rollouts={len(list(scoring.glob('*.jsonl')))} partial={len(list(scoring.glob('*.partial')))} "
            f"gradients={len(list(scoring.glob('*/prompt-*.json')))} "
            f"shards={len(list(scoring.glob('*.done.json')))}")


def storage_report(roots):
    sections = [('MBPP SAVED-WORK AUDIT (read-only; <=4 KiB)\n'
                'Presence only, NOT hash/resume validation. Archived/waived work is NOT free to reuse.\n'
                'No files moved, deleted, restored or training started.')]
    per_root = min(1230, (STORAGE_BYTES - len(sections[0].encode('utf-8')) - len(roots) - 1) // max(1, len(roots)))
    for root in roots:
        manifest = record(root, root / 'switch.json')
        lines = [f'\nEXPERIMENT {mbpp_suite_label(root, manifest)}', f'ROOT {root.name}']
        if not root.is_dir():
            sections.append('\n'.join(lines + ['ROOT MISSING here; cannot determine remote data loss.']))
            continue
        branches = sorted(p for p in root.glob('states/*/points/*/*') if p.is_dir() and p.name in {
            'selection_reduced', 'random_reduced', 'selection_full', 'random_full', 'gated'})
        prefixes = sorted(root.glob('prefixes/seed-*/segment-*/fresh_r'))
        training = branches + prefixes
        active = [p for p in training if (p / 'policy/adapter_model.safetensors').is_file()
                  or any((p / 'policy').glob('checkpoint-*/adapter_model.safetensors'))]
        archived = [(p, a) for p in branches for a in sorted((p / 'discarded').glob('*')) if a.is_dir()]
        lines.append(f'active training dirs with adapter/checkpoint={len(active)}; '
                     f'archived attempt dirs={len(archived)}; branches={len(branches)}')
        for p in active[:2]:
            lines.append(f'SAVED {p.relative_to(root)}: {saved_inventory(p)}')
        failed = [p for p in branches if (p / 'failure.json').is_file()]
        lines.append(f'failed branches={len(failed)}; showing first two')
        for p in failed[:2]:
            decision = record(root, p / 'decision.json')
            cost, cap = charged(p), decision.get('budget_gpu_seconds', '?')
            remaining = round(cap-cost, 3) if type(cap) in (int, float) and math.isfinite(cap) and type(cost) in (int, float) else 'unknown'
            lines.append(f'BRANCH {p.relative_to(root)}: GPU-s used={cost} cap={cap} remaining={remaining}')
            lines.append('  active ' + saved_inventory(p))
            archives = [a for parent, a in archived if parent == p]
            lines.append(f"  waivers={len(list((p / 'waivers').glob('*.json')))} archived={len(archives)}")
            for a in archives[-1:]:
                lines.append(f'  ARCHIVE {a.relative_to(root)}: {saved_inventory(a)}')
        if not failed:
            for _, a in archived[-1:]:
                lines.append(f'ARCHIVE {a.relative_to(root)}: {saved_inventory(a)}')
        sections.append(clipped('\n'.join(lines), per_root))
    return clipped('\n'.join(sections) + '\n', STORAGE_BYTES)


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
    manifest = record(root, root / 'switch.json')
    lines = [f'\nEXPERIMENT {mbpp_suite_label(root, manifest)}', f'ROOT {root.name}',
             f"SELECTOR {selector_label(manifest.get('selector', '?'))}  "
             f"ACCOUNTING {accounting_label(manifest.get('accounting', '?'))}  "
             f"GATE {gate_label(manifest.get('gate', '?'))}"]
    lines.append('protocol (raw audit): ' + str({key: manifest.get(key) for key in ('dataset', 'selector', 'accounting', 'gate')}))
    if '_read_error' in manifest:
        lines.append('manifest unreadable: ' + manifest['_read_error'])
    elif not (root / 'switch.json').is_file():
        lines.append('manifest missing: switch.json; protocol values are unknown')
    elif manifest.get('dataset') not in (None, 'mbpp'):
        lines.append('manifest dataset is not MBPP; shown as stored, not relabeled')
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
    logs = recent(work, ('runs/experiments/logs/console.mbpp.*.log',))
    nodes = [f'\nNODE {path.name}\n{tail(work, path, 1400)}' for path in logs[:2]]
    reserved = len(('\n'.join([sections[0], *nodes]) + '\n').encode('utf-8'))
    per_root = min(ROOT_BYTES, (MAX_BYTES - reserved - len(roots)) // max(1, len(roots)))
    sections.extend(clipped(root_summary(root), per_root) for root in roots)
    sections.extend(nodes)
    return clipped('\n'.join(sections) + '\n', MAX_BYTES)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--root', type=Path, action='append', required=True)
    parser.add_argument('--storage', action='store_true', help='read-only saved-work audit to stdout, <=4 KiB')
    args = parser.parse_args()
    if len(args.root) > 4:
        parser.error('at most four MBPP suite roots (including retained legacy work)')
    if args.storage:
        print(storage_report(args.root), end='')
        return 0
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
