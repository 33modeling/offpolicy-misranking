"""One tiny all-suite random-only inventory; never modifies experiment files."""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

from mbpp_storage_audit import json_bytes, present, worker_active

LIMIT = 4096
ARMS = {'random_full': 'RF', 'random_reduced': 'RR', 'random_online': 'RO'}


def read(path):
    return json_bytes(path)[0] if present(path) else {}


def inventory(work):
    runs = Path(work) / 'runs'
    reports, problems = [], []
    if not runs.is_dir():
        return [], [f'STORAGE_UNAVAILABLE {runs} (not proof of deletion)']
    roots = sorted({p.parent for name in ('switch.json', 'mopps.json') for p in runs.glob(f'*/{name}')})
    for root in roots:
        rows = []
        directories = {p for pattern in ('states/*/points/*/random_*', 'points/*/random_*', 'states/*/random_online')
                       for p in root.glob(pattern) if p.name in ARMS and p.is_dir()}
        for directory in sorted(directories):
            rel = directory.relative_to(root)
            state = rel.parts[1]
            key = f'{state}/{ARMS[directory.name]}'
            policy = directory / 'policy'
            code, step = 'N', '?'
            flags = []
            try:
                result = read(directory / 'result.json')
                seal = read(directory / 'result.sha256.json')
                final = read(policy / 'policy_train.json')
                stop = read(policy / 'budget_stop.json')
                progress = read(directory / 'progress.json')
                if result:
                    step = result.get('completed_steps', '?')
                    if result.get('complete') is not True:
                        code = 'X'
                    elif seal.get('sha256') == hashlib.sha256((directory / 'result.json').read_bytes()).hexdigest():
                        code = 'D'
                    else:
                        code = 'U'
                elif final:
                    code, step = 'P', final.get('completed_steps', '?')
                elif any(policy.glob('checkpoint-*')):
                    code = 'C'
                elif present(policy):
                    code = 'T'
                if final and not stop:
                    flags.append('FINAL_STOP_MISSING')
                if seal and not result:
                    flags.append('RESULT_MISSING_SEAL_REMAINS')
                if stop.get('use_parent_policy') is True and final:
                    flags.append('PARENT_STOP_WITH_SAVED_POLICY')
                archives = [p for p in (directory / 'discarded').glob('*') if any(
                    present(p / name) for name in ('policy', 'result.json', 'evaluation'))]
                if archives:
                    flags.append('ARCHIVE=' + ','.join(p.name for p in archives[-2:]))
                active = worker_active(directory / '.task.lock')
                phase = str(progress.get('phase', '?'))
                if active:
                    flags.append(f'LIVE:{phase}@{progress.get("host", "?")}')
                failed = read(directory / 'failure.json')
                if failed and code != 'D' and not active:
                    error = str(failed.get('error', '?')).splitlines()
                    flags.append('FAIL:' + (error[-1] if error else '?')[:140])
                waivers = sorted((directory / 'waivers').glob('*.json'), key=lambda p: p.stat().st_mtime)
                if waivers:
                    receipt = read(waivers[-1])
                    attempt = receipt.get('attempt', {})
                    flags.append(f'waivers={len(waivers)} last={attempt.get("phase", "?")}:{attempt.get("fault", {}).get("kind", "?")}')
            except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
                code = 'X'
                flags.append(str(exc)[:160])
            rows.append({'arm': ARMS[directory.name], 'key': key, 'code': code, 'step': step, 'flags': flags})
        reports.append({'root': str(root), 'rows': rows})
    return reports, problems


def render(work, reports, problems, revision):
    lines = ['RANDOM STORAGE: ALL DISCOVERED SUITES; read-only', f'WORK {work}', f'checkout={revision}',
             'RF=random_full RR=random_reduced RO=random_online. D=sealed result P=final manifest C=checkpoint candidate',
             'T=other policy files U=unsealed/invalid result X=unreadable N=no saved training found.',
             'Presence only; no tensor/lineage validation. LIVE=lock owner, not proof of healthy training.',
             'On-policy is a display name; existing storage paths are unchanged.']
    details, completed = [], []
    for index, report in enumerate(reports, 1):
        lines.append(f'R{index} {Path(report["root"]).name}')
        parts = []
        for arm in ('RF', 'RR', 'RO'):
            counts = Counter(row['code'] for row in report['rows'] if row['arm'] == arm)
            parts.append(arm + ':' + (','.join(f'{key}={counts[key]}' for key in sorted(counts)) or 'no branch directories'))
        lines.append('  ' + ' '.join(parts))
        for row in report['rows']:
            item = f'R{index} {row["key"]} {row["code"]} step={row["step"]}'
            if row['flags'] or row['code'] != 'D':
                flags = ' '.join(row['flags'])
                priority = 0 if any(x in flags for x in ('MISSING', 'ARCHIVE=', 'PARENT_STOP', 'FAIL:')) else 1
                details.append((priority, item + ' ' + flags))
            if row['code'] == 'D':
                completed.append(f'R{index}:{row["key"]}@{row["step"]}')
    lines += problems
    if not reports:
        lines.append('No switch/mopps manifests discovered; cannot infer deletion or initialize new runs.')
    lines += [item for priority, item in details if priority == 0]
    lines.append('SEALED RANDOM RESULTS: ' + (' '.join(completed) or 'none found in inspected paths'))
    lines += [item for priority, item in details if priority != 0]
    raw = ('\n'.join(lines) + '\n').encode()
    if len(raw) <= LIMIT:
        return raw.decode()
    footer = b'\n[Details omitted at 4 KiB; inspect per-root random counts above.]\n'
    return raw[:LIMIT - len(footer)].decode(errors='ignore') + footer.decode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path, default=Path.home())
    args = parser.parse_args()
    reports, problems = inventory(args.work)
    revision = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], text=True,
                              capture_output=True, check=False, timeout=5).stdout.strip() or 'unknown'
    output = render(args.work, reports, problems, revision)
    with tempfile.NamedTemporaryFile(prefix='random-storage-', suffix='.txt', dir=args.report_dir,
                                     mode='wb', delete=False) as handle:
        handle.write(output.encode())
        saved = handle.name
    print(output, end='')
    print(f'[saved] {saved} ({len(output.encode())} bytes; send this TXT)')
    return 2 if problems else 0


if __name__ == '__main__':
    raise SystemExit(main())
