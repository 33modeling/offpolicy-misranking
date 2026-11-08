#!/usr/bin/env python3
"""V8 queue status in the same layout as `sh scripts/run_srgc.sh <dataset> status` (read-only).

    sh scripts/v8_status.sh            # one snapshot of olmo_math and qwen_mbpp
    sh scripts/v8_status.sh --watch    # refresh every 60 s until Ctrl-C (--watch 30 for 30 s)

No locks, signals, Torch imports or queue writes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v8_probe import active_roots, last_row, read, root, tail  # noqa: E402

COHORTS = ('olmo_math', 'qwen_mbpp')
ATTENTION = {'failed', 'interrupted', 'stopped', 'unconfirmed'}
STALE_SECONDS = 180  # workers heartbeat every 10 s while a child runs (srgc_v8_nodes.Runtime.process)


def worker_seen(queue, worker, now):
    """Seconds since the worker's heartbeat file (workers/<sha256(canonical id)>.json) was written, or None."""
    key = hashlib.sha256(json.dumps(worker, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    beat = read(Path(queue) / 'workers' / (key + '.json'))
    return now - beat['at'] if beat and isinstance(beat.get('at'), (int, float)) else None


def age_text(seconds):
    if seconds is None:
        return '?'
    return f"{seconds / 60:.0f} min" if seconds >= 120 else f"{max(0, seconds):.0f}s"


def gpu_rows(run=subprocess.run):
    """[(index, util %, MiB used)] from nvidia-smi, or None when unavailable."""
    try:
        out = run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
                  capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    rows = []
    for line in out.splitlines():
        parts = [x.strip() for x in line.split(',')]
        if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
            rows.append((parts[0], int(parts[1]), int(parts[2])))
    return rows


def last_error_line(lines):
    for line in reversed(lines):
        stripped = line.strip()
        if re.search(r"(Error|Exception|error:|out of memory|Killed|Traceback)", stripped) and not stripped.startswith("File "):
            return stripped[:200]
    return None


def collect(base, cohort, now):
    directory = Path(base) / cohort
    queue = directory / '.multi-node'
    state = read(queue / 'state.json') or {}
    rows = []
    for path in sorted((queue / 'jobs').glob('*.json')):
        job = read(path)
        if not job:
            continue
        task = job.get('task') or {}
        attempts = job.get('attempts') or []
        attempt = attempts[-1] if attempts else {}
        row = dict(task=str(task.get('task_id') or task.get('run_id') or path.stem),
                   arm=str(task.get('method_id') or task.get('stage') or task.get('kind') or '-'),
                   seed=task.get('training_seed', '-'), status=job.get('state', 'pending'),
                   total=task.get('horizon') or task.get('endpoint') or task.get('total_updates'),
                   attempt=len(attempts), host=str(attempt.get('worker') or '').rsplit('.', 1)[0] or None, worker=attempt.get('worker'), done=None)
        rid = task.get('run_id')
        if rid:
            manifest = read(directory / 'runs' / rid / 'manifest.json')
            if manifest and manifest.get('attempts'):
                last = last_row(directory / 'runs' / rid / 'attempts' / manifest['attempts'][-1]['attempt_id'] / 'updates.jsonl')
                row['done'] = last.get('completed_updates') if last else None
            result = read(directory / 'runs' / rid / 'evaluation.json') or read(directory / 'runs' / rid / 'eval.json')
            if result and isinstance(result.get('reward_fraction'), (int, float)):
                row['reward_percent'] = 100 * result['reward_fraction']
        if row['status'] == 'running' and attempt.get('worker'):
            seen = worker_seen(queue, attempt['worker'], now)
            if seen is None or seen > STALE_SECONDS:
                # Admission, long I/O or a lost worker can all stop heartbeats. This is not proof
                # that ranks/leases are gone, especially on a remote host with fresh work logs.
                row['status'] = 'unconfirmed'
                row['error'] = (f"worker {attempt['worker']} heartbeat age {age_text(seen)}" if seen is not None
                                else f"worker {attempt['worker']} has no heartbeat") + '; process liveness unverified'
        if attempt.get('id'):
            log = queue / 'logs' / (str(attempt['id']) + '.log')
            row['log'] = str(log)
            try:
                row['log_age'] = now - log.stat().st_mtime
            except OSError:
                row['log_age'] = None
            lines = [x for x in tail(log, 32768) if x.strip()]
            row['last_line'] = lines[-1].strip()[:160] if lines else None
            row['error'] = row.get('error') or last_error_line(lines) or (str(attempt.get('error'))[:200] if attempt.get('error') else None)
        if row['status'] == 'pending':
            row['ready'] = task_ready(task)
        rows.append(row)
    return state, rows


def task_ready(task):
    """Same rule as srgc_v8_nodes.Runtime.ready: evaluation needs its snapshot, training its saved parent."""
    try:
        if task.get('kind') == 'evaluate':
            return Path(task['snapshot_path']).is_file()
        parent = task.get('parent') or {}
        return bool(parent.get('path')) and Path(parent['path']).is_file()
    except (KeyError, TypeError, OSError):
        return False


def worker_records(queue, now):
    """Heartbeat files written by srgc_v8_nodes workers (workers/<digest>.json: worker, pid, at, task)."""
    records = []
    for path in sorted(Path(queue).glob('workers/*.json')):
        beat = read(path)
        if not beat or not isinstance(beat.get('at'), (int, float)):
            continue
        worker = str(beat.get('worker') or path.stem)
        age = now - beat['at']
        task = beat.get('task')
        if age > STALE_SECONDS:
            status = 'stale'
        elif task:
            status = 'running'
        else:
            status = 'idle'
        records.append(dict(worker=worker, host=worker.rsplit('.', 1)[0], status=status, task=task, age=age))
    return records


def node_lines(workers, rows):
    active = [w for w in workers if w['status'] != 'stale']
    hidden = len(workers) - len(active)
    lines = []
    if active or hidden:
        lines += ['', 'nodes:']
        for w in sorted(active, key=lambda w: w['worker']):
            job = next((r for r in rows if r['status'] in ('running', 'unconfirmed') and r.get('worker') == w['worker']), None)
            if job is not None:
                w['status'] = 'running'
                what = f"running {job['task']}" + (f" · {job['done']}/{job['total']} done" if job['done'] is not None and job['total'] else '')
            elif w['status'] == 'running':
                what = 'running' if str(w['task']).startswith('child:') else f"running {w['task']}"
            else:
                what = 'idle (nothing claimable)'
            lines.append(f"  {w['worker']:<24} {what} · heartbeat {age_text(w['age'])} ago")
        if hidden:
            lines.append(f"  ({hidden} node record{'s' if hidden != 1 else ''} without a heartbeat in the last "
                         f"{STALE_SECONDS // 60} min not shown; not proof the process stopped)")
    running_workers = sum(w['status'] == 'running' for w in active)
    idle_workers = sum(w['status'] == 'idle' for w in active)
    ready = sum(1 for r in rows if r['status'] == 'pending' and r.get('ready'))
    running_tasks = sum(r['status'] == 'running' for r in rows)
    wanted = running_tasks + ready
    lines.append(f"workers: {len(active)} active ({running_workers} running · {idle_workers} idle) · "
                 f"work available now for {wanted} worker{'s' if wanted != 1 else ''} "
                 f"({running_tasks} running + {ready} ready to start)"
                 + (f" → {max(0, wanted - len(active))} more 4-GPU worker{'s' if wanted - len(active) != 1 else ''} could be added"
                    if wanted > len(active) else ''))
    return lines


def cell(row):
    status, total = row['status'], row['total']
    if status == 'done':
        reward = row.get('reward_percent')
        return 'done' if reward is None else f'done {reward:.1f}%'
    if status == 'running':
        return f"running {row['done']}/{total}" if row['done'] is not None and total else 'running'
    if status == 'pending':
        return 'ready' if row.get('ready') else 'waiting'
    return status.replace('_', ' ')


def running_detail(row):
    parts = []
    if row['done'] is not None:
        parts.append(f"{row['done']}/{row['total'] or '?'} done")
    if row.get('log_age') is not None:
        parts.append(f"log {age_text(row['log_age'])} ago")
    if row.get('host'):
        parts.append(f"node {row['host']}")
    if row.get('last_line'):
        parts.append(f"last: {row['last_line']}")
    return ' · '.join(parts) or 'starting'


def attention_detail(row):
    parts = [row['status'].replace('_', ' ')]
    if row.get('error'):
        parts.append(f"last error: {row['error']}")
    if row.get('attempt'):
        parts.append(f"{row['attempt']} attempt{'s' if row['attempt'] != 1 else ''}")
    if row['done'] is not None:
        parts.append(f"last step {row['done']}/{row['total'] or '?'}")
    if row.get('host'):
        parts.append(f"node {row['host']}")
    if row.get('log'):
        parts.append(f"log {row['log']}")
    return ' · '.join(parts)


def this_node_lines(rows, gpus, host):
    mine = [r for r in rows if r['status'] == 'running' and r.get('host') == host]
    if mine:
        row = mine[0]
        what = f"running {row['task']}" + (f" · {row['done']}/{row['total']} done" if row['done'] is not None and row['total'] else '')
        line = f"this node ({host}): {what}" + (f" · log {age_text(row['log_age'])} ago" if row.get('log_age') is not None else '')
    else:
        uncertain=[r for r in rows if r['status']=='unconfirmed' and r.get('host')==host]
        line = (f"this node ({host}): task activity unconfirmed" if uncertain
                else f"this node ({host}): no recorded running task of this queue")
    lines = [line]
    if gpus:
        lines.append('gpu memory used: ' + ' '.join(f'{i}:{m}MiB' for i, _, m in gpus)
                     + ' · util ' + ' '.join(f'{i}:{u}%' for i, u, _ in gpus))
    return lines


def render_cohort(base, cohort, now, gpus, host):
    state, rows = collect(base, cohort, now)
    generated = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))
    phase = state.get('phase', 'not started')
    lines = [f"V8 {cohort} · {generated} · phase {phase}", str(Path(base) / cohort), *this_node_lines(rows, gpus, host), '']
    if state.get('failure'):
        lines.append(f"ERROR queue: {str(state['failure'])[:300]}")
    counts = {'done': 0, 'running': 0, 'waiting': 0, 'attention': 0, 'ready': 0}
    for row in rows:
        key = ('done' if row['status'] == 'done' else 'running' if row['status'] == 'running'
               else 'attention' if row['status'] in ATTENTION else 'ready' if row.get('ready') else 'waiting')
        counts[key] += 1
    summary = ' · '.join(f'{name} {count}' for name, count in counts.items() if count)
    lines += [summary.replace('attention', 'needs attention') or 'no tasks', '']
    if rows:
        arms, seeds = [], []
        for row in rows:
            arms += [row['arm']] if row['arm'] not in arms else []
            seeds += [row['seed']] if row['seed'] not in seeds else []
        table = {(r['seed'], r['arm']): cell(r) for r in rows}
        widths = {arm: max(len(arm), *(len(table.get((seed, arm), '-')) for seed in seeds)) for arm in arms}
        lines.append('seed  ' + '  '.join(f'{arm:<{widths[arm]}}' for arm in arms))
        for seed in seeds:
            lines.append(f'{seed!s:>4}  ' + '  '.join(f"{table.get((seed, arm), '-'):<{widths[arm]}}" for arm in arms))
    running = [r for r in rows if r['status'] == 'running']
    if running:
        lines += ['', 'running now:'] + [f"  {r['task']:<16} {running_detail(r)}" for r in running]
    attention = [r for r in rows if r['status'] in ATTENTION]
    if attention:
        lines += ['', 'needs attention:'] + [f"  {r['task']:<16} {attention_detail(r)}" for r in attention]
    lines += node_lines(worker_records(Path(base) / cohort / '.multi-node', now), rows)
    idle = gpus is not None and not any(u >= 10 for _, u, _ in gpus)
    for r in running:
        if r.get('host') == host and idle and (r.get('log_age') or 0) > 1800:
            lines.append(f"WARNING {r['task']}: no log output for {age_text(r['log_age'])} and this node's GPUs are idle "
                         f"(stack dumps: {Path(base) / cohort / '.multi-node' / 'stacks'})")
    return '\n'.join(lines) + '\n'


def render(bases, now=None, gpus=None, host=None):
    now = time.time() if now is None else now
    gpus = gpu_rows() if gpus is None else gpus
    host = host or socket.gethostname()
    return '\n'.join(render_cohort(base, cohort, now, gpus, host) for base in bases for cohort in COHORTS)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--watch', nargs='?', const=60.0, type=float, default=None, metavar='SECONDS',
                        help='refresh every SECONDS (default 60) until Ctrl-C')
    args = parser.parse_args(argv)
    bases = active_roots() or [root(os.environ)]
    if args.watch is None:
        sys.stdout.write(render(bases))
        return 0
    try:
        while True:
            sys.stdout.write('\033[H\033[2J' + render(bases) + f"\n(refresh every {int(args.watch)}s · Ctrl-C to stop)\n")
            sys.stdout.flush()
            time.sleep(max(5.0, args.watch))
    except KeyboardInterrupt:
        return 0


if __name__ == '__main__':
    sys.exit(main())
