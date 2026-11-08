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
# Clean view liveness (user: 'idle 아닌데 자꾸 idle 이래 이미 삭제한 노드인데', 'status 업데이트 더 빠르게'):
ALIVE_SECONDS = 30       # idle workers write a heartbeat every poll (3 s), running children every 5-10 s
DEAD_FREE_SECONDS = 60   # task assigned, its lease lock is free and no heartbeat: the process is gone
HUNG_SECONDS = 600       # task assigned, lock still held, no heartbeat at all: hung (or crashed, lock not released)
GAVE_UP_SECONDS = 600    # a worker silent in one cohort this long while alive in the other no longer serves it
CPU_PHASES = {'prepare', 'freeze', 'finalize'}
_COLLECT_CACHE = None    # set by render_clean: collect() once per (root, cohort) per refresh


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
    # torchrun appends ChildFailedError after the actual rank traceback. Show the
    # specific exception from existing logs before falling back to that wrapper.
    for line in reversed(lines):
        stripped=line.strip();plain=re.sub(r'^\[rank\d+\]:\s*','',stripped)
        if plain.startswith('RANK FAILURE'):return stripped[:600]
        if (re.match(r'^(?:[\w.]+(?:Error|Exception)|SystemExit):',plain)
                and 'ChildFailedError' not in plain.split(':',1)[0]):return stripped[:600]
        if re.search(r'Signal \d+ \(SIG(?:KILL|SEGV|BUS|ABRT)\)',plain):return stripped[:600]
    for line in reversed(lines):
        stripped = line.strip()
        if re.search(r"(Error|Exception|error:|out of memory|Killed|Traceback)", stripped) and not stripped.startswith("File "):
            return stripped[:200]
    return None


def progress_events(base, queue, attempt_id, host, now):
    """Work records the frozen trainer writes while a task runs (SRGC_PROGRESS_DIR/rank-N.json):
    first the attempt's own directory (.multi-node/progress/<attempt>), else this host's most recent
    launch directory (<V8_ROOT>/.node-progress/<host>/<launch>). Returns [(rank, stage, details, age)]."""
    candidates = [Path(queue) / 'progress' / str(attempt_id)] if attempt_id else []
    if host:
        try:
            launches = sorted((Path(base) / '.node-progress' / host).iterdir(), key=lambda d: d.stat().st_mtime, reverse=True)
        except OSError:
            launches = []
        candidates += launches[:3]
    for directory in candidates:
        events = []
        try:
            files = sorted(directory.glob('rank-*.json'))
        except OSError:
            files = []
        for path in files:
            event = read(path)
            if isinstance(event, dict) and isinstance(event.get('updated'), (int, float)):
                rank = path.stem.split('-', 1)[-1]
                events.append((rank, str(event.get('stage', '?')), event, now - event['updated']))
        if events:
            return events
    return []


def progress_text(events):
    if not events:
        return None, None
    try:
        from v8_progress_text import describe
    except ImportError:  # older bundle
        def describe(stage, details):
            return stage
    rank, stage, details, age = min(events, key=lambda e: e[3])
    text = describe(stage, details)
    if len(events) > 1:
        ages = [e[3] for e in events]
        text += f" · {len(events)} GPU processes, last record {age_text(min(ages))}-{age_text(max(ages))} ago"
    else:
        text += f" · last record {age_text(age)} ago"
    return text, age


def collect(base, cohort, now):
    if _COLLECT_CACHE is not None:
        key = (str(base), cohort, now)
        if key not in _COLLECT_CACHE:
            _COLLECT_CACHE[key] = _collect(base, cohort, now)
        return _COLLECT_CACHE[key]
    return _collect(base, cohort, now)


def _collect(base, cohort, now):
    directory = Path(base) / cohort
    queue = directory / '.multi-node'
    state = read(queue / 'state.json') or {}
    rows = []
    for path in sorted((queue / 'jobs').glob('*.json')):
        job = read(path)
        if not isinstance(job, dict):
            state.setdefault('_unreadable_jobs', []).append(path.name)
            continue
        task = job.get('task') or {}
        attempts = job.get('attempts') or []
        attempt = attempts[-1] if attempts else {}
        row = dict(key=path.stem, task=str(task.get('task_id') or task.get('run_id') or path.stem),
                   arm=str(task.get('method_id') or task.get('stage') or task.get('kind') or '-'),
                   seed=task.get('training_seed', '-'), status=job.get('state', 'pending'),
                   total=task.get('horizon') or task.get('endpoint') or task.get('total_updates'),
                   attempt=len(attempts), started=attempt.get('started_at'), run_id=task.get('run_id'), kind=task.get('kind'), phase=job.get('phase'), host=str(attempt.get('worker') or '').rsplit('.', 1)[0] or None, worker=attempt.get('worker'), done=None)
        rid = task.get('run_id')
        if rid:
            manifest = read(directory / 'runs' / rid / 'manifest.json')
            if manifest and manifest.get('attempts'):
                run_attempt = manifest['attempts'][-1]
                last = last_row(directory / 'runs' / rid / 'attempts' / run_attempt['attempt_id'] / 'updates.jsonl')
                row['done'] = last.get('completed_updates') if last else None
                # Written only after the runner has loaded and validated the
                # actual parent/latest checkpoint, before the first new update.
                # A planned parent.completed_update alone proves no load.
                start = run_attempt.get('start_update') if task.get('kind') == 'train' else None
                if type(start) is int and start >= 0:
                    row['done'] = max(start,row['done']) if type(row['done']) is int else start
            result = read(directory / 'runs' / rid / 'evaluation.json') or read(directory / 'runs' / rid / 'eval.json')
            if result and isinstance(result.get('reward_fraction'), (int, float)):
                row['reward_percent'] = 100 * result['reward_fraction']
        if row['status'] == 'running' and attempt.get('worker'):
            seen = worker_seen(queue, attempt['worker'], now)
            row['beat_age'] = seen
            if seen is None or seen > ALIVE_SECONDS:
                # The task's lease lock, not heartbeat age, is authoritative (srgc_v8_nodes claim rule):
                # the worker holds it for the whole execution, including admission and model loading.
                row['lease'] = lease_held(queue, path.stem)
            quiet = seen is None or seen > ALIVE_SECONDS
            dead = quiet and ((row.get('lease') is False and (seen is None or seen > DEAD_FREE_SECONDS))
                              or (row.get('lease') is None and (seen is None or seen > STALE_SECONDS)))
            if dead:
                # Admission, long I/O or a lost worker can all stop heartbeats. This is not proof
                # that ranks/leases are gone, especially on a remote host with fresh work logs.
                row['status'] = 'unconfirmed'
                row['error'] = (f"worker {attempt['worker']} heartbeat age {age_text(seen)}" if seen is not None
                                else f"worker {attempt['worker']} has no heartbeat") + \
                               ('; task lock is free (process gone)' if row.get('lease') is False else '; process liveness unverified')
            elif quiet:
                row['quiet'] = seen
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
            failures=(attempt.get('child_failure') or {}).get('failures') or []
            if failures:
                row['error']='; '.join(f"rank {f.get('rank')}: {f.get('message')}" for f in failures)[:600]
        if row['status'] in ('running', 'unconfirmed') and attempt.get('id'):
            events = progress_events(base, queue, attempt.get('id'), row.get('host'), now)
            row['progress'], progress_age = progress_text(events)
            row['_events'] = events
            ages = [a for a in (row.get('log_age'), progress_age) if a is not None]
            row['activity_age'] = min(ages) if ages else None
        else:
            row['activity_age'] = row.get('log_age')
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
        records.append(dict(worker=worker, host=worker.rsplit('.', 1)[0], status=status, task=task, age=age,
                            controller_sha256=beat.get('controller_sha256'),dispatch_policy=beat.get('dispatch_policy')))
    return records


def node_lines(workers, rows):
    expected=hashlib.sha256(Path(__file__).with_name('srgc_v8_nodes.py').read_bytes()).hexdigest()
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
            version=w.get('controller_sha256')
            controller=('current '+version[:12] if version==expected else 'different '+version[:12] if version else 'unverified')
            lines.append(f"  {w['worker']:<24} {what} · heartbeat {age_text(w['age'])} ago · dispatch {controller}")
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


def access_problems(queue):
    """Queue paths that exist but this account cannot read (status would otherwise look empty)."""
    import pwd
    problems = []
    for path in (queue, queue / 'state.json', queue / 'binding.json', queue / 'jobs', queue / 'workers',
                 queue / 'logs', queue.parent / 'runs'):
        try:
            if not path.exists():
                continue
            mode = os.R_OK | (os.X_OK if path.is_dir() else 0)
            if not os.access(path, mode):
                try:
                    owner = pwd.getpwuid(path.stat().st_uid).pw_name
                except (KeyError, OSError):
                    owner = '?'
                problems.append(f"{path} (owner {owner})")
        except OSError as exc:
            problems.append(f"{path} ({exc.strerror or exc})")
    return problems


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


def column(row):
    """Grid column: a budget extension keeps its branch's method_id and every evaluation has no method_id,
    so they get their own columns instead of overwriting the training cell."""
    if row.get('phase') == 'evaluation' or row['arm'] == 'evaluate':
        return 'eval'
    return row['arm'] + ' ext' if row.get('phase') == 'extension' else row['arm']


def grid_cell(rows):
    """One task: its state. Several tasks in one cell: counts per state, problems first."""
    if len(rows) == 1:
        return cell(rows[0])
    counts = {}
    for row in rows:
        counts[cell(row).split(' ')[0]] = counts.get(cell(row).split(' ')[0], 0) + 1
    order = sorted(counts, key=lambda name: (name in ('done', 'running', 'ready', 'waiting'), name))
    return ','.join(f'{counts[name]} {name}' for name in order)


def running_detail(row):
    parts = []
    if row['done'] is not None:
        parts.append(f"{row['done']}/{row['total'] or '?'} done")
    if row.get('activity_age') is not None:
        parts.append(f"activity {age_text(row['activity_age'])} ago")
    if row.get('host'):
        parts.append(f"node {row['host']}")
    if row.get('progress'):
        parts.append(f"now: {row['progress']}")
    if row.get('last_line'):
        parts.append(f"log: {row['last_line']}")
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
        line = f"this node ({host}): {what}" + (f" · activity {age_text(row['activity_age'])} ago" if row.get('activity_age') is not None else '')
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
    problems = access_problems(Path(base) / cohort / '.multi-node')
    if state.get('_unreadable_jobs'):
        problems.append(f"{len(state['_unreadable_jobs'])} job record(s) unreadable or corrupt: " + ', '.join(state['_unreadable_jobs'][:5]))
    if problems:
        import getpass
        lines.append(f"ERROR some queue records could not be read by {getpass.getuser()}: " + '; '.join(problems[:4])
                     + " — counts below are incomplete (permission or partial write); run status as the account "
                       "that runs the workers, or inspect the named files")
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
            row['column'] = column(row)
            arms += [row['column']] if row['column'] not in arms else []
            seeds += [row['seed']] if row['seed'] not in seeds else []
        arms.sort(key=lambda name: (name == 'eval', name.endswith(' ext')))  # training, extensions, evaluations
        groups = {}
        for row in rows:
            groups.setdefault((row['seed'], row['column']), []).append(row)
        table = {key: grid_cell(group) for key, group in groups.items()}
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
        if r.get('host') == host and idle and (r.get('activity_age') or 0) > 1800:
            lines.append(f"WARNING {r['task']}: no log or progress record for {age_text(r['activity_age'])} and this node's GPUs are idle "
                         f"(stack dumps: {Path(base) / cohort / '.multi-node' / 'stacks'})")
    return summary_at_bottom('\n'.join(lines))


# ---- Same screen as `sh scripts/run_srgc.sh <dataset> status` -------------------------------------
# The original renderer (scripts/srgc_live_status.py render) is reused unchanged; this file only turns
# the V8 queue into the report rows that renderer reads. If it cannot be imported (no original
# checkout next to the bundle), the simplified V8 view below is printed instead.

SRGC_STATUS = {'done': 'complete', 'running': 'running', 'failed': 'failed', 'interrupted': 'interrupted',
               'unconfirmed': 'recoverable', 'stopped': 'recoverable'}


def code_roots():
    here = Path(__file__).resolve().parent
    seen = []
    for candidate in [os.environ.get('V8_CODE_ROOT'), here.parent, *here.parents]:
        if candidate and (Path(candidate) / 'srgc_rebuttal' / 'plan.py').is_file() and Path(candidate) not in seen:
            seen.append(Path(candidate))
    return seen


def original_renderer():
    """srgc_live_status.render with the GPU-time section disabled (V8 costs are in the V8 ledger)."""
    import importlib
    import types
    for root in code_roots():
        for path in (str(root), str(root / 'scripts'), str(Path(__file__).resolve().parent)):
            if path not in sys.path:
                sys.path.append(path)
        try:
            if 'scripts' not in sys.modules:
                package = types.ModuleType('scripts')
                package.__path__ = [str(root / 'scripts')]
                sys.modules['scripts'] = package
            times = types.ModuleType('scripts.srgc_time_status')
            times.render = lambda report, **_: []
            sys.modules['scripts.srgc_time_status'] = times
            live = importlib.import_module('srgc_live_status')
            return live
        except Exception:  # noqa: BLE001 - fall back to the simplified view
            continue
    return None


def srgc_report(base, cohort, now, host):
    state, rows = collect(base, cohort, now)
    queue = Path(base) / cohort / '.multi-node'
    tasks = []
    trained = {r.get('run_id'): r['arm'] for r in rows if r.get('kind') == 'train' and r.get('phase') != 'extension'}
    for r in rows:
        status = SRGC_STATUS.get(r['status']) or ('ready' if r.get('ready') else 'waiting_for_parent')
        # One column per distinct job kind so evaluations and budget extensions never hide each other.
        arm = r['arm']
        if r.get('kind') == 'evaluate':
            arm = 'eval ' + str(trained.get(r.get('run_id')) or r.get('run_id') or arm)
        elif r.get('phase') == 'extension':
            arm = arm + ' ext'
        task = dict(task=r['task'], arm=arm, seed=r['seed'], status=status, total_steps=r['total'],
                    completed_steps=r['done'], last_observed_step=r['done'], host=r.get('host'),
                    log=r.get('log') or '', attempt=r.get('attempt'))
        if r.get('reward_percent') is not None:
            task['reward_percent'] = r['reward_percent']
        if status in ('running', 'recoverable'):
            if r['done'] is not None and r['total'] and r['done'] < r['total']:
                task['current_step'] = r['done'] + 1
            events = r.get('_events') or []
            if events:
                rank, stage, details, age = min(events, key=lambda e: e[3])
                task['phase_stage'] = stage
                task['progress_age_seconds'] = age
                task['progress'] = [{'updated': now - e[3], 'stage': e[1]} for e in events]
        tasks.append(task)
    # Column order like the original screen: training arms (On first), then extensions, then evaluations.
    def order(task):
        arm = str(task['arm'])
        group = 2 if arm.startswith('eval ') else 1 if arm.endswith(' ext') else 0
        base_arm = arm[5:] if group == 2 else arm[:-4] if group == 1 else arm
        return (group, 0 if base_arm.startswith('on') else 1, base_arm, str(task['seed']))
    tasks.sort(key=order)
    workers = []
    for w in worker_records(queue, now):
        job = next((r for r in rows if r['status'] in ('running', 'unconfirmed') and r.get('worker') == w['worker']), None)
        status = 'heartbeat_stale' if w['status'] == 'stale' else 'running' if job or w['status'] == 'running' else 'idle'
        workers.append(dict(host=w['host'], status=status, heartbeat_age_seconds=w['age'],
                            task=job['task'] if job else (None if str(w.get('task') or '').startswith('child:') else w.get('task'))))
    errors = []
    problems = access_problems(queue)
    if state.get('_unreadable_jobs'):
        problems.append(f"{len(state['_unreadable_jobs'])} job record(s) unreadable or corrupt: " + ', '.join(state['_unreadable_jobs'][:5]))
    if problems:
        errors.append('some queue records could not be read (counts incomplete): ' + '; '.join(problems[:4]))
    if state.get('failure'):
        errors.append('queue: ' + str(state['failure'])[:300])
    return dict(dataset=f"V8 {cohort}", phase=state.get('phase', 'not started'), generated=now,
                output_root=str(Path(base) / cohort), tasks=tasks, workers=workers, warnings=[], errors=errors), rows


def summary_at_bottom(text, extra=()):
    """Details first, summary last (header, root, this node, gpu, counts, workers) so the summary is
    what stays visible at the bottom of the terminal."""
    lines = text.rstrip('\n').split('\n')
    try:
        first_blank = lines.index('')
    except ValueError:
        return '\n'.join([*lines, *extra]) + '\n'
    head = lines[:first_blank]                       # header, root, this node, gpu memory
    rest = lines[first_blank + 1:]
    counts = []
    while rest and rest[0] and not rest[0].startswith('seed'):
        counts.append(rest[0])                       # 'done 1 · running 1 · ...' / 'no tasks' (+ ERROR lines)
        rest = rest[1:]
    while rest and rest[0] == '':
        rest = rest[1:]
    details = [line for line in rest if not line.startswith('workers: ')]
    extra = [*[line for line in rest if line.startswith('workers: ')], *extra]
    while details and details[-1] == '':
        details.pop()
    body = [*details, ''] if details else []
    return '\n'.join([*body, *head, *counts, *extra]) + '\n'


def render_srgc(live, base, cohort, now, host):
    if not (Path(base) / cohort / '.multi-node').exists():
        return f"SRGC V8 {cohort} · {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))}\n{Path(base) / cohort}\nno queue yet (not started)\n"
    report, rows = srgc_report(base, cohort, now, host)
    head, _, rest = live.render(report).partition('\n')
    text = f"{head} · phase {report['phase']}\n{rest}"
    demand = node_lines(worker_records(Path(base) / cohort / '.multi-node', now), rows)[-1:]
    return summary_at_bottom(text, demand)


# ---- Clean view (default) ------------------------------------------------------------------------
# One compact block per cohort: short column names, one symbol per cell, one line per running task
# and per problem, one line for all nodes; the summary (counts, workers, this node) is at the bottom.

LEGEND = "✓ done   ▶n running (updates done)   ○ ready   · waiting   ✗ failed   ⏸ interrupted   ? no heartbeat"
COHORT_LABEL = {'olmo_math': 'OLMo MATH', 'qwen_mbpp': 'Qwen MBPP'}
PHASE_LABEL = {'prepare': 'prepare', 'train': 'training', 'freeze': 'budget freeze', 'extension': 'budget extension',
               'evaluation': 'evaluation', 'finalize': 'analysis', 'complete': 'complete'}
STAGE_SHORT = {'generation_start': 'answering', 'generation': 'answering', 'generation_finished': 'answers done',
               'generation_oom_retry': 'answering (OOM retry)', 'rollout': 'checking answers',
               'gradient_scoring': 'scoring questions', 'policy_update': 'optimizer step', 'update': 'update done',
               'model_ready': 'model loaded', 'checkpoint_load': 'loading checkpoint', 'checkpoint_save': 'saving checkpoint',
               'selection': 'selecting questions', 'training': 'training step', 'preparation': 'preparing',
               'startup': 'starting', 'cache_generation': 'building reward cache', 'cache_refresh': 'refreshing cache',
               'cache_export': 'saving cache', 'evaluation': 'evaluating'}


def short_arm(arm):
    arm = str(arm)
    if arm in ('on_monitored', 'on', 'on_policy'):
        return 'On'
    m = re.fullmatch(r'bank_(sr|uniform|random)_t0*(\d+)', arm)
    if m:
        return {'sr': 'SR', 'uniform': 'U', 'random': 'R'}[m.group(1)] + m.group(2)
    return arm.replace('bank_', '')[:10]


def symbol(row):
    status = row['status']
    if status == 'done':
        return '✓'
    if status == 'running':
        return f"▶{row['done']}" if row['done'] is not None else '▶'
    if status == 'pending':
        return '○' if row.get('ready') else '·'
    return {'failed': '✗', 'interrupted': '⏸', 'unconfirmed': '?', 'stopped': '?'}.get(status, '?')


def short_progress(row, now):
    events = row.get('_events') or []
    if not events:
        return None
    rank, stage, details, age = min(events, key=lambda e: e[3])
    text = STAGE_SHORT.get(stage, stage.replace('_', ' '))
    if details.get('question_index') and details.get('questions_total'):
        text += f" {details['question_index']}/{details['questions_total']}"
    if details.get('response_tokens') is not None:
        text += f", {details['response_tokens']} tok"
    return f"{text} ({age_text(age)} ago)"


def clip(line, width):
    return line if len(line) <= width else line[:max(10, width - 1)] + '…'


def clean_block(base, cohort, now, host, width):
    queue = Path(base) / cohort / '.multi-node'
    label = COHORT_LABEL.get(cohort, cohort)
    if not queue.exists():
        return [], f"{label:<10} not started", None
    state, rows = collect(base, cohort, now)
    trained = {r.get('run_id'): r['arm'] for r in rows if r.get('kind') == 'train' and r.get('phase') != 'extension'}
    train_cols, seeds, cells, ext, ev = [], [], {}, {}, {}
    for r in rows:
        seed = r['seed']
        if seed not in seeds:
            seeds.append(seed)
        if r.get('kind') == 'evaluate':
            ev.setdefault(seed, []).append(r)
        elif r.get('phase') == 'extension':
            ext.setdefault(seed, []).append(r)
        else:
            col = short_arm(r['arm'])
            if col not in train_cols:
                train_cols.append(col)
            cells[(seed, col)] = symbol(r)
    def col_key(c):
        kind = 0 if c == 'On' else 1 if c.startswith('SR') else 2 if c.startswith('U') else 3
        num = int(re.sub(r'\D', '', c) or 0)
        return (kind, num, c)
    train_cols.sort(key=col_key)
    seeds.sort(key=lambda x: (not isinstance(x, int), x if isinstance(x, int) else str(x)))
    def count_cell(items):
        if not items:
            return '-'
        done = sum(i['status'] == 'done' for i in items)
        bad = sum(i['status'] in ATTENTION for i in items)
        run = sum(i['status'] == 'running' for i in items)
        if done == len(items):
            return '✓'
        return f"{done}/{len(items)}" + ('✗' if bad else '▶' if run else '')
    cols = [*train_cols, *(['ext'] if ext else []), *(['eval'] if ev else [])]
    table = {}
    for seed in seeds:
        for c in train_cols:
            table[(seed, c)] = cells.get((seed, c), '-')
        if ext:
            table[(seed, 'ext')] = count_cell(ext.get(seed, []))
        if ev:
            table[(seed, 'eval')] = count_cell(ev.get(seed, []))
    widths = {c: max(len(c), *(len(table[(s, c)]) for s in seeds)) for c in cols} if seeds else {}
    lines = [f"━━ {label} ({cohort}) " + '━' * max(3, min(width, 100) - len(label) - len(cohort) - 7)]
    if seeds:
        lines.append('seed  ' + '  '.join(f'{c:<{widths[c]}}' for c in cols))
        for seed in seeds:
            lines.append(f'{seed!s:>4}  ' + '  '.join(f'{table[(seed, c)]:<{widths[c]}}' for c in cols))
    else:
        lines.append('  (no tasks yet)')
    problems = [r for r in rows if r['status'] in ATTENTION]
    unreadable = access_problems(queue)
    if state.get('_unreadable_jobs'):
        unreadable.append(f"{len(state['_unreadable_jobs'])} unreadable job record(s)")
    if problems or unreadable or state.get('failure'):
        lines.append('needs attention')
        if state.get('failure'):
            lines.append(clip(f"  ✗ queue: {state['failure']}", width))
        for r in problems:
            reason = r.get('error') or r['status']
            lines.append(clip(f"  {symbol(r)} s{r['seed']} {short_arm(r['arm'])}: {reason}", width))
        for u in unreadable[:3]:
            lines.append(clip(f"  ! cannot read {u}", width))
    workers = worker_records(queue, now)
    live = [w for w in workers if w['status'] != 'stale']
    counts = {k: 0 for k in ('done', 'running', 'ready', 'waiting', 'attention', 'dead')}
    for r in rows:
        key = ('done' if r['status'] == 'done' else 'running' if r['status'] == 'running'
               else 'dead' if r['status'] in ('unconfirmed', 'stopped')
               else 'attention' if r['status'] in ATTENTION else 'ready' if r.get('ready') else 'waiting')
        counts[key] += 1
    phase = PHASE_LABEL.get(state.get('phase'), state.get('phase', 'not started'))
    total = len(rows)
    # Training progress in optimizer updates (what 'how far along' means for a run), plus task counts.
    planned = done_updates = 0
    for r in rows:
        if r.get('kind') == 'train' and isinstance(r.get('total'), int):
            planned += r['total']
            done_updates += r['total'] if r['status'] == 'done' else (r['done'] or 0)
    pct = round(100 * counts['done'] / total) if total else 0
    filled = round(12 * counts['done'] / total) if total else 0
    bar = '█' * filled + '░' * (12 - filled)
    summary = (f"{label:<10} {phase:<13} {bar} {counts['done']}/{total} done ({pct}%)  running {counts['running']}  "
               f"ready {counts['ready']}  waiting {counts['waiting']}  failed {counts['attention']}"
               + (f"  on dead node {counts['dead']}" if counts['dead'] else '')
               + (f"  updates {done_updates:,}/{planned:,}" if planned else ''))
    wanted = counts['running'] + counts['ready']
    nodes = (len(live), sum(1 for w in live if any(r.get('worker') == w['worker'] and r['status'] in ('running', 'unconfirmed') for r in rows)))
    return lines, summary, (nodes, wanted)


def lease_held(queue, key):
    """True if a process still holds the job's lease lock, False if free, None if unknown.
    Read-only probe (no file is created); a claim racing this probe just retries on its next scan."""
    import fcntl
    try:
        fd = os.open(Path(queue) / 'leases' / f'{key}.lock', os.O_RDONLY)
    except OSError:
        return None
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        except OSError:
            return None
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def span(seconds):
    seconds = max(0, int(seconds))
    if seconds < 3600:
        return f"{seconds // 60}m"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


def gpu_text(group):
    """'0,1,2,3' -> '0-3' (contiguous), otherwise unchanged."""
    try:
        ids = [int(x) for x in str(group).split(',')]
    except ValueError:
        return str(group)
    if len(ids) > 1 and ids == list(range(ids[0], ids[0] + len(ids))):
        return f"{ids[0]}-{ids[-1]}"
    return str(group)


def gpu_groups(queues):
    """host -> GPU groups that passed admission ('.multi-node/admission/<host>-<g0>-<g1>-...-<uuid8>')."""
    groups = {}
    for queue in queues:
        try:
            names = [p.name for p in (Path(queue) / 'admission').iterdir()]
        except OSError:
            continue
        for name in names:
            m = re.fullmatch(r'(.+?)((?:-\d+)+)-[0-9a-f]{8}', name)
            if m:
                groups.setdefault(m.group(1), set()).add(m.group(2)[1:].replace('-', ','))
    return groups


def local_gpus(pid):
    """GPU list of a worker on this host, from its command line (--gpus) or environment."""
    try:
        argv = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    except OSError:
        return None
    args = [a.decode(errors='replace') for a in argv]
    for i, arg in enumerate(args):
        if arg == '--gpus' and i + 1 < len(args):
            return args[i + 1]
        if arg.startswith('--gpus='):
            return arg.split('=', 1)[1]
    try:
        env = Path(f'/proc/{pid}/environ').read_bytes().split(b'\0')
    except OSError:
        return '0,1,2,3'
    for item in env:
        if item.startswith(b'CUDA_VISIBLE_DEVICES='):
            return item.split(b'=', 1)[1].decode(errors='replace') or '0,1,2,3'
    return '0,1,2,3'


def natural(text):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', str(text))]


def cpu_log_age(queue, now):
    try:
        ages = [now - p.stat().st_mtime for p in (Path(queue) / 'logs').glob('cpu-*.log')]
    except OSError:
        return None
    return min(ages) if ages else None


def node_table(bases, cohorts, now, this_host=None):
    """One row per worker process (worker id = '<host>.<pid>'), over both cohorts.

    A worker writes one heartbeat file per cohort it serves; while it scans, claims or runs a CPU stage
    of one cohort only that cohort's file stops, and during a GPU task only the task's cohort file moves.
    running  a task is assigned and (heartbeat fresh or its lease lock is held)
    busy     no task, heartbeat fresh in one cohort but silent in another that is still open (CPU stage /
             checking / claiming there)
    starting loading the runtime / checking the node        idle   heartbeat fresh, no task
    DEAD     task assigned, lock free and no heartbeat >60 s, or lock held and no heartbeat >10 min
    gone     no heartbeat >30 s, nothing assigned            finished  same after its queue completed"""
    this_host = this_host or socket.gethostname()
    nodes, queues, info = {}, [], {}
    def entry(worker, host):
        return nodes.setdefault(worker, dict(worker=worker, host=host, beats={}, jobs=[], complete=True))
    for base in bases:
        for cohort in cohorts:
            queue = Path(base) / cohort / '.multi-node'
            if not queue.exists():
                continue
            queues.append(queue)
            state, rows = collect(base, cohort, now)
            phase = state.get('phase')
            info[cohort] = dict(phase=phase, open=phase != 'complete' and not state.get('failure'), queue=queue,
                                ready=sum(1 for r in rows if r['status'] == 'pending' and r.get('ready')))
            for w in worker_records(queue, now):
                e = entry(w['worker'], w['host'])
                e['beats'][cohort] = (w['age'], w['task'])
                e['complete'] = e['complete'] and phase == 'complete'
            for r in rows:
                if r['status'] in ('running', 'unconfirmed') and r.get('worker'):
                    e = entry(r['worker'], r.get('host') or str(r['worker']).rsplit('.', 1)[0])
                    e['complete'] = e['complete'] and phase == 'complete'
                    e['jobs'].append((cohort, r))
    groups = gpu_groups(queues)
    def fresh(e):
        return {c: v for c, v in e['beats'].items() if v[0] <= ALIVE_SECONDS}
    idle_alive = sum(1 for e in nodes.values() if fresh(e) and not e['jobs'])
    out = []
    for e in nodes.values():
        pid = e['worker'][len(e['host']) + 1:] if e['worker'].startswith(e['host'] + '.') else ''
        gpus = local_gpus(pid) if e['host'] == this_host and pid.isdigit() else None
        if gpus:
            gpus = gpu_text(gpus)
        else:
            known = sorted(gpu_text(g) for g in groups.get(e['host'], ()))
            gpus = known[0] if len(known) == 1 else ' or '.join(known) if known else '?'
        ages = [v[0] for v in e['beats'].values()]
        newest = min(ages) if ages else None
        beat = f"{age_text(newest)} ago" if newest is not None else 'never'
        local_alive = e['host'] == this_host and pid.isdigit() and Path(f'/proc/{pid}').exists()
        base = dict(host=e['host'], pid=pid, gpus=gpus, beat=beat, experiment='-', task='-', updates='-', took='-')
        if e['jobs']:
            for cohort, r in e['jobs']:
                task = f"s{r['seed']} " + ('eval ' if r.get('kind') == 'evaluate' else '') + short_arm(r['arm']) + \
                       (' ext' if r.get('phase') == 'extension' else '')
                updates = f"{r['done']}/{r['total']}" if r['done'] is not None and r['total'] else \
                          ('?/' + str(r['total']) if r['total'] and r.get('kind') != 'evaluate' else '-')
                took = span(now - r['started']) if isinstance(r.get('started'), (int, float)) else '-'
                row = dict(base, experiment=COHORT_LABEL.get(cohort, cohort), task=task, updates=updates, took=took)
                if r['status'] == 'running':
                    hung = r.get('quiet') is not None and r['quiet'] > HUNG_SECONDS and r.get('lease') is True
                    if hung:
                        row.update(state='DEAD', doing=(f"process {pid} alive here but silent {age_text(r['quiet'])}: hung? kill {pid}, rerun"
                                                        if local_alive else
                                                        f"no heartbeat {age_text(r['quiet'])}, lock still held (hung, or crashed node's lock not released): kill it there, rerun"))
                    elif r.get('quiet') is not None:
                        row.update(state='running', doing=(short_progress(r, now) or
                                   f"no heartbeat {age_text(r['quiet'])}, task lock held (admission / model load / checkpoint)"))
                    else:
                        row.update(state='running', doing=short_progress(r, now) or
                                   (f"log written {age_text(r['activity_age'])} ago" if r.get('activity_age') is not None
                                    else 'started, no output yet'))
                else:
                    if local_alive:
                        doing = f"process {pid} still alive here but silent: hung? kill {pid}, rerun same command"
                    elif r.get('lease') is False:
                        doing = 'task is free, ' + ('an idle node takes it over' if idle_alive else 'no idle node: start one')
                    else:
                        doing = 'lock state unknown: restart that node (same command)'
                    row.update(state='DEAD', doing=doing)
                out.append(row)
            continue
        live = fresh(e)
        if not live:
            out.append(dict(base, state='finished' if e['complete'] else 'gone',
                            doing='queue complete' if e['complete'] else 'stopped, crashed or removed; nothing waits for it'))
            continue
        if any(task == 'runtime_initialization' for _, task in live.values()):
            out.append(dict(base, state='starting', doing='loading runtime / checking node'))
            continue
        silent = {c: v[0] for c, v in e['beats'].items() if c not in live and c in info and info[c]['open']}
        busy = {c: a for c, a in silent.items() if a <= GAVE_UP_SECONDS}
        if busy:
            cohort, age = min(busy.items(), key=lambda x: x[1])
            label = COHORT_LABEL.get(cohort, cohort)
            if info[cohort]['phase'] in CPU_PHASES:
                log = cpu_log_age(info[cohort]['queue'], now)
                doing = f"CPU stage {PHASE_LABEL.get(info[cohort]['phase'], info[cohort]['phase'])}" + \
                        (f", log {age_text(log)} ago" if log is not None else '')
            else:
                doing = f"checking / claiming tasks ({age_text(age)})"
            out.append(dict(base, state='busy', experiment=label, doing=doing))
            continue
        ready = sum(info[c]['ready'] for c in live if c in info)
        doing = (f"idle while {ready} task{'s are' if ready != 1 else ' is'} ready: GPU busy or setup wait? see its console"
                 if ready else 'waiting for work')
        if silent:
            doing += ' · not serving ' + ', '.join(COHORT_LABEL.get(c, c) for c in silent) + ' (gave up? see its console)'
        out.append(dict(base, state='idle', doing=doing))
    rank = {'running': 0, 'busy': 1, 'starting': 2, 'idle': 3, 'DEAD': 4, 'gone': 5, 'finished': 6}
    return sorted(out, key=lambda r: (rank[r['state']], natural(r['host']), natural(r['pid'])))

def render_clean(bases, now=None, gpus=None, host=None, cohorts=COHORTS, width=None, all_nodes=False):
    import shutil
    now = time.time() if now is None else now
    host = host or socket.gethostname()
    gpus = gpu_rows() if gpus is None else gpus
    width = width or max(80, min(shutil.get_terminal_size((120, 40)).columns, 160))
    global _COLLECT_CACHE
    _COLLECT_CACHE = {}
    try:
        return _render_clean(bases, now, gpus, host, cohorts, width, all_nodes)
    finally:
        _COLLECT_CACHE = None


def _render_clean(bases, now, gpus, host, cohorts, width, all_nodes):
    out, summaries = [], []
    active = running_nodes = wanted = 0
    mine = None
    for base in bases:
        for cohort in cohorts:
            lines, summary, extra = clean_block(base, cohort, now, host, width)
            if lines:
                out += [*lines, '']
            summaries.append(summary)
            if extra:
                (a, r), w = extra
                active, running_nodes, wanted = max(active, a), max(running_nodes, r), wanted + w
            if mine is None and lines:
                _, rows = collect(base, cohort, now)
                job = next((x for x in rows if x['status'] == 'running' and x.get('host') == host), None)
                if job:
                    mine = f"▶ s{job['seed']} {short_arm(job['arm'])} ({cohort})"
    out.append(LEGEND)
    out.append('─' * min(width, 100))
    out.append(f"V8 status · {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))} · {', '.join(str(b) for b in bases)}")
    out += summaries
    gpu_text = ' '.join(f'{i}:{u}%' for i, u, _ in gpus) if gpus else 'unknown'
    out.append(f"this node  {host}: {mine or 'no V8 task here'} · GPU {gpu_text}")
    table = node_table(bases, cohorts, now, host)
    shown = table if all_nodes else [r for r in table if r['state'] not in ('gone', 'finished')]
    if table:
        order = ('running', 'busy', 'starting', 'idle', 'DEAD', 'gone', 'finished')
        per = {k: sorted({(r['host'], r['pid']) for r in table if r['state'] == k}) for k in order}
        alive = len(per['running']) + len(per['busy']) + len(per['starting']) + len(per['idle'])
        hosts = len({r['host'] for r in table if r['state'] not in ('gone', 'finished')})
        hidden = [f"{len(per[k])} {k}" for k in ('gone', 'finished') if per[k]]
        more = max(0, wanted - alive)
        out.append(f"nodes      {hosts} host{'s' if hosts != 1 else ''} · "
                   + ' · '.join(f"{len(per[k])} {k}" for k in ('running', 'busy', 'starting', 'idle', 'DEAD') if per[k] or k in ('running', 'DEAD'))
                   + (f"   DEAD: {', '.join(h for h, _ in per['DEAD'])}" if per['DEAD'] else '')
                   + (f"   ({', '.join(hidden)} not shown; --all-nodes)" if hidden and not all_nodes else ''))
        out.append(f"           tasks that can run now: {wanted} · alive workers: {alive}"
                   + (f" → {more} more 4-GPU node{'s' if more != 1 else ''} would help" if more else ''))
        if shown:
            cols = (('host', 'host'), ('gpus', 'gpus'), ('state', 'state'), ('experiment', 'experiment'), ('task', 'task'),
                    ('updates', 'updates'), ('took', 'running for'), ('beat', 'heartbeat'))
            w = {k: max(len(title), *(len(str(r[k])) for r in shown)) for k, title in cols}
            out.append('   ' + '  '.join(f"{title:<{w[k]}}" for k, title in cols) + '  now')
            for r in shown:
                mark = '✗' if r['state'] == 'DEAD' else '▶' if r['state'] == 'running' else ' '
                out.append(f" {mark} " + '  '.join(f"{str(r[k]):<{w[k]}}" for k, _ in cols) + f"  {r['doing']}")
            if per['DEAD']:
                out.append("   DEAD = task assigned but its process is gone (lock free, no heartbeat >1 min) or silent >10 min")
    else:
        out.append("nodes      no worker has started yet" + (f" · tasks that can run now: {wanted}" if wanted else ''))
    return '\n'.join(clip(line, width) for line in out) + '\n'


def render(bases, now=None, gpus=None, host=None, cohorts=COHORTS, style='simple', all_nodes=False):
    now = time.time() if now is None else now
    host = host or socket.gethostname()
    if style == 'clean':
        return render_clean(bases, now=now, gpus=gpus, host=host, cohorts=cohorts, all_nodes=all_nodes)
    live = original_renderer() if style == 'srgc' else None
    if live is not None:
        return '\n'.join(render_srgc(live, base, cohort, now, host) for base in bases for cohort in cohorts)
    gpus = gpu_rows() if gpus is None else gpus
    note = ('(original run_srgc.sh status renderer not found next to this bundle; simplified V8 view)\n'
            if style == 'srgc' else '')
    return note + '\n'.join(render_cohort(base, cohort, now, gpus, host) for base in bases for cohort in cohorts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    def cohort_name(value):
        # run_v8_nodes.sh accepts every configured cohort (configs/cohorts/<name>.json), not only the default two.
        if not re.fullmatch(r'[A-Za-z0-9_-]+', value):
            raise argparse.ArgumentTypeError('cohort must be a simple configured identifier')
        return value
    parser.add_argument('--cohort', action='append', type=cohort_name, help='show only this cohort (repeatable)')
    parser.add_argument('--simple', action='store_true', help='older simplified V8 view')
    parser.add_argument('--classic', action='store_true', help='the original run_srgc.sh status screen layout')
    parser.add_argument('--all-nodes', action='store_true', help='also list gone and finished workers')
    parser.add_argument('--watch', nargs='?', const=3.0, type=float, default=None, metavar='SECONDS',
                        help='refresh every SECONDS (default 3) until Ctrl-C')
    args = parser.parse_args(argv)
    # An explicit V8_ROOT wins; otherwise the roots of this node's running V8 processes, else the default.
    bases = [Path(os.environ['V8_ROOT'])] if os.environ.get('V8_ROOT') else (active_roots() or [root(os.environ)])
    if args.watch is None:
        sys.stdout.write(render(bases, cohorts=tuple(args.cohort or COHORTS), style='simple' if args.simple else 'srgc' if args.classic else 'clean', all_nodes=args.all_nodes))
        return 0
    try:
        while True:
            sys.stdout.write('\033[H\033[2J' + render(bases, cohorts=tuple(args.cohort or COHORTS), style='simple' if args.simple else 'srgc' if args.classic else 'clean', all_nodes=args.all_nodes) + f"\n(refresh every {int(args.watch)}s · Ctrl-C to stop)\n")
            sys.stdout.flush()
            time.sleep(max(1.0, args.watch))
    except KeyboardInterrupt:
        return 0


if __name__ == '__main__':
    sys.exit(main())
