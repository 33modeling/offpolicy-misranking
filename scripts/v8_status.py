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
        row = dict(task=str(task.get('task_id') or task.get('run_id') or path.stem),
                   arm=str(task.get('method_id') or task.get('stage') or task.get('kind') or '-'),
                   seed=task.get('training_seed', '-'), status=job.get('state', 'pending'),
                   total=task.get('horizon') or task.get('endpoint') or task.get('total_updates'),
                   attempt=len(attempts), run_id=task.get('run_id'), kind=task.get('kind'), phase=job.get('phase'), host=str(attempt.get('worker') or '').rsplit('.', 1)[0] or None, worker=attempt.get('worker'), done=None)
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
    return '\n'.join(lines) + '\n'


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


def render_srgc(live, base, cohort, now, host):
    if not (Path(base) / cohort / '.multi-node').exists():
        return f"SRGC V8 {cohort} · {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))}\n{Path(base) / cohort}\nno queue yet (not started)\n"
    report, rows = srgc_report(base, cohort, now, host)
    head, _, rest = live.render(report).partition('\n')
    text = f"{head} · phase {report['phase']}\n{rest}"
    demand = node_lines(worker_records(Path(base) / cohort / '.multi-node', now), rows)[-1:]
    return text + '\n'.join(demand) + '\n'


def render(bases, now=None, gpus=None, host=None, cohorts=COHORTS, style='simple'):
    now = time.time() if now is None else now
    host = host or socket.gethostname()
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
    parser.add_argument('--simple', action='store_true', help='simplified V8 view instead of the run_srgc.sh status screen')
    parser.add_argument('--watch', nargs='?', const=60.0, type=float, default=None, metavar='SECONDS',
                        help='refresh every SECONDS (default 60) until Ctrl-C')
    args = parser.parse_args(argv)
    # An explicit V8_ROOT wins; otherwise the roots of this node's running V8 processes, else the default.
    bases = [Path(os.environ['V8_ROOT'])] if os.environ.get('V8_ROOT') else (active_roots() or [root(os.environ)])
    if args.watch is None:
        sys.stdout.write(render(bases, cohorts=tuple(args.cohort or COHORTS), style='simple' if args.simple else 'srgc'))
        return 0
    try:
        while True:
            sys.stdout.write('\033[H\033[2J' + render(bases, cohorts=tuple(args.cohort or COHORTS), style='simple' if args.simple else 'srgc') + f"\n(refresh every {int(args.watch)}s · Ctrl-C to stop)\n")
            sys.stdout.flush()
            time.sleep(max(5.0, args.watch))
    except KeyboardInterrupt:
        return 0


if __name__ == '__main__':
    sys.exit(main())
