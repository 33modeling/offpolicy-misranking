#!/usr/bin/env python3
"""V8 queue status in the same layout as `sh scripts/run_srgc.sh <dataset> status` (read-only).

    sh scripts/v8_status.sh            # one snapshot of olmo_math, qwen_mbpp and olmo_mbpp
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
from srgc_v8_nodes import retry_budget_failures, fatal_native_failure

COHORTS = ('olmo_math', 'olmo_mbpp', 'qwen_math', 'qwen_mbpp')   # model, then task (user 11-09: 'status 순서 맞춰')
ATTENTION = {'failed', 'interrupted', 'stopped', 'unconfirmed'}
STALE_SECONDS = 180  # workers heartbeat every 10 s while a child runs (srgc_v8_nodes.Runtime.process)
# Clean view liveness (user: 'idle 아닌데 자꾸 idle 이래 이미 삭제한 노드인데', 'status 업데이트 더 빠르게'):
# Heartbeats are written every 3-5 s, but another node's file can be read up to ~60 s stale on a shared
# file system and node clocks differ; a lock held on another node can look free from here. So a lock
# probe is only positive evidence (held = alive), and DEAD needs over 3 min without any heartbeat.
ALIVE_SECONDS = 90       # a worker without a task is alive with a heartbeat in the last 90 s
BUSY_SILENT_SECONDS = 120  # silent this long in one open cohort while alive in another = busy there
HUNG_SECONDS = 600       # task assigned, lock visibly held, no heartbeat at all: hung (or crashed, lock kept)
GAVE_UP_SECONDS = 900    # silent this long in one cohort while alive in the other: no longer serves it
CPU_PHASES = {'prepare', 'freeze', 'primary_finalize', 'finalize'}
_COLLECT_CACHE = None    # set by render_clean: collect() once per (root, cohort) per refresh


def worker_seen(queue, worker, now):
    """Seconds since the worker's heartbeat file (workers/<sha256(canonical id)>.json) was written, or None."""
    key = hashlib.sha256(json.dumps(worker, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    beat = read(Path(queue) / 'workers' / (key + '.json'))
    return now - beat['at'] if beat and isinstance(beat.get('at'), (int, float)) else None


def age_text(seconds):
    if seconds is None:
        return '?'
    if seconds >= 5400:
        return f"{int(seconds // 3600)}h{int(seconds % 3600 // 60):02d}m"
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
    try:
        live_policies = {w.get('dispatch_policy') for w in worker_records(queue, now)
                         if w['age'] <= ALIVE_SECONDS and w.get('dispatch_policy')}
    except Exception:
        live_policies = set()
    for path in sorted((queue / 'jobs').glob('*.json')):
        try:
            job = read(path)
            if not isinstance(job, dict):
                state.setdefault('_unreadable_jobs', []).append(path.name)
                continue
            task = job.get('task') if isinstance(job.get('task'), dict) else {}
            attempts = job.get('attempts') if isinstance(job.get('attempts'), list) else []
            attempt = attempts[-1] if attempts and isinstance(attempts[-1], dict) else {}
            # Worker ids are normally '<host>.<pid>' strings; never let an odd record crash the screen.
            raw_worker = attempt.get('worker')
            shown_worker = str(raw_worker) if raw_worker not in (None, '') else None
            row = dict(key=path.stem, task=str(task.get('task_id') or task.get('run_id') or path.stem),
                       arm=str(task.get('method_id') or task.get('stage') or task.get('kind') or '-'),
                       seed=task.get('training_seed', '-'), status=job.get('state', 'pending'),
                       total=task.get('horizon') or task.get('endpoint') or task.get('total_updates'),
                       attempt=len(attempts), started=attempt.get('started_at'), ended=attempt.get('ended_at'), update=task.get('completed_update'), run_id=task.get('run_id'), kind=task.get('kind'), phase=job.get('phase'), host=shown_worker.rsplit('.', 1)[0] if shown_worker else None, worker=shown_worker, done=None)
            parent = task.get('parent') if isinstance(task.get('parent'), dict) else {}
            start = exit_code(parent.get('completed_update'))   # int parse; branches/extensions start here
            row['start'] = start if isinstance(start, int) and start >= 0 else 0
            if row['status'] == 'running' and not shown_worker:
                row['status'] = 'unconfirmed'
                row['error'] = 'marked running but no worker is recorded for it'
            rid = task.get('run_id')
            if rid:
                manifest = read(directory / 'runs' / rid / 'manifest.json')
                row['_manifest'] = manifest if isinstance(manifest, dict) else None
                run_attempts = manifest.get('attempts') if isinstance(manifest, dict) else None
                if isinstance(run_attempts, list) and run_attempts and isinstance(run_attempts[-1], dict) \
                        and isinstance(run_attempts[-1].get('attempt_id'), str):
                    run_attempt = run_attempts[-1]
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
                if seen is None or seen > STALE_SECONDS:
                    # The worker holds the task's lease lock for the whole execution (admission, model load).
                    # Held is proof of life; 'free' may only mean the lock lives on another node's kernel.
                    row['lease'] = lease_held(queue, path.stem)
                quiet = seen is None or seen > ALIVE_SECONDS
                dead = (seen is None or seen > STALE_SECONDS) and row.get('lease') is not True
                if dead:
                    # Admission, long I/O or a lost worker can all stop heartbeats. This is not proof
                    # that ranks/leases are gone, especially on a remote host with fresh work logs.
                    row['status'] = 'unconfirmed'
                    row['error'] = (f"worker {attempt['worker']} heartbeat age {age_text(seen)}" if seen is not None
                                    else f"worker {attempt['worker']} has no heartbeat") + \
                                   '; process liveness unverified'
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
                row['_log_lines'] = lines[-120:]
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
            if task.get('kind') == 'evaluate':
                row['done'] = None      # the scored run's updates are not evaluation progress (was shown as '▶275')
            if row['status'] == 'pending':
                row['ready'] = task_ready(task)
            if row['status'] == 'failed':
                row.update(failure_story(row, attempt, job, task, directory, state, live_policies, now))
            rows.append(row)
        except Exception as exc:  # one odd record must not take down the whole screen
            state.setdefault('_unreadable_jobs', []).append(f'{path.name} ({type(exc).__name__}: {exc})'[:200])
    return state, rows


TERMINAL_RUN = {'completed_horizon', 'completed_budget'}   # srgc_v8_nodes.TERMINAL
MAX_ATTEMPTS = 2   # V20: lifetime task faults, including failures from previous nodes
RETRY_DELAY = 60.0


def exit_code(value):
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def counted_failures(job):
    """srgc_v8_nodes.counted_failures: interruptions and node-local launch faults are refunded."""
    count = 0
    for attempt in job.get('attempts') or []:
        if not isinstance(attempt, dict):
            continue
        outcome = attempt.get('outcome')
        if outcome is None and exit_code(attempt.get('exit_code')) not in (None, 0, 75):
            outcome = 'failed'
        count += outcome == 'failed'
    return count


def failure_story(row, attempt, job, task, directory, state, live_policies, now):
    """Why a task is recorded as failed, whether that is a real training error, and what happens next.

    The queue only records 'task exit=N; output verified=False'. The cause is in the rank error records,
    the task log, the run manifest (work may in fact be complete) or the exit code itself."""
    rc = exit_code(attempt.get('exit_code'))
    lines = row.get('_log_lines') or []
    ranks = [f for f in ((attempt.get('child_failure') or {}).get('failures') or []) if isinstance(f, dict)]
    if not ranks and attempt.get('id'):
        # torchrun error files written by a crashing rank (also read by v8_runtime_worker.child_result)
        for path in sorted((directory / '.multi-node' / 'progress' / str(attempt['id'])).glob('rank-*-error.json'))[:8]:
            try:
                record = json.loads(path.read_text())
                message = (record.get('error') or {}).get('message')
                if isinstance(message, dict):
                    message = message.get('message')
                if message:
                    ranks.append(dict(rank=record.get('rank', path.stem.split('-')[1]),
                                      message=str(message).strip().splitlines()[-1][:300]))
            except (OSError, ValueError, AttributeError, IndexError):
                continue
    log_error = last_error_line(lines)
    python_error = bool(log_error and re.search(r'(Error|Exception|Traceback|out of memory)', log_error)
                        and not re.search(r'Signal \d+ \(SIGKILL\)', log_error))
    manifest = row.get('_manifest') or {}
    total = row.get('total')
    finished = (task.get('kind') == 'train' and manifest.get('status') in TERMINAL_RUN
                and isinstance(manifest.get('completed_updates'), int) and isinstance(total, int)
                and manifest['completed_updates'] >= total)
    snap, spec = task.get('snapshot_sha256'), task.get('evaluation_spec_hash')
    if task.get('kind') == 'evaluate' and isinstance(snap, str) and isinstance(spec, str):
        finished = (directory / 'evaluations' / f"{snap[:16]}__{spec[:12]}.json").is_file()
    at = f" at update {row['done']}/{total}" if isinstance(row.get('done'), int) and isinstance(total, int) else ''
    real = True
    if finished:
        cause = (f"NOT a training failure: the run is complete ({manifest.get('completed_updates')}/{total} updates)"
                 if task.get('kind') == 'train' else 'NOT an evaluation failure: its result file is complete') + \
                (f"; exit {rc} came after the work was done" if rc is not None else '')
        real = False
    elif ranks:
        cause = '; '.join(f"rank {f.get('rank')}: {f.get('message')}" for f in ranks)
    elif rc in (9, -9, 137) and not python_error:
        cause = ("NOT a training error: killed from outside (SIGKILL, exit %s) - host out of memory (OOM killer), "
                 "a manual kill or the node was removed; the training code reported no error" % rc)
        real = False
    elif rc in (15, -15, 143) and not python_error:
        cause = f"NOT a training error: terminated (SIGTERM, exit {rc}) - stopped from outside"
        real = False
    elif rc == 124:
        cause = 'stall watchdog: no training progress for 30 min, the task was stopped (checkpoints kept)'
    elif rc == 0:
        cause = 'exited normally but its outputs are incomplete or invalid' + \
                (f": {attempt['output_validation_error']}" if attempt.get('output_validation_error') else '')
    elif log_error:
        cause = re.sub(r'^\[rank\d+\]:\s*', '', log_error)
        if re.search(r'out of memory', cause, re.I):
            cause = 'GPU out of memory: ' + cause
    elif attempt.get('error') and not re.fullmatch(r'task exit=-?\d+; output verified=\w+', str(attempt['error'])):
        cause = str(attempt['error'])
    elif rc is None:
        cause = 'no exit code or error was recorded for this attempt (older record); see its log'
    else:
        cause = f"exit {rc} without an error message in its log or rank error files"
    lifetime = counted_failures(job)
    used = retry_budget_failures(job)
    auto = state.get('automatic_retry_policy') if isinstance(state.get('automatic_retry_policy'), dict) else {}
    limit = auto.get('max_failed_attempts') if isinstance(auto.get('max_failed_attempts'), int) else MAX_ATTEMPTS
    delay = auto.get('retry_delay') if isinstance(auto.get('retry_delay'), (int, float)) else RETRY_DELAY
    policy = attempt.get('controller_policy')
    recovered = {h.get('controller_policy') for h in job.get('recovery_history') or [] if isinstance(h, dict)}
    # v11 (9c1e31c) retries failed tasks by default and records it in state.json; older workers follow the marker.
    retrying = bool(state.get('retry_failed_requested_at') or state.get('automatic_retry_policy'))
    ended = attempt.get('ended_at') if isinstance(attempt.get('ended_at'), (int, float)) else None
    native_fault=fatal_native_failure(job)
    if native_fault:
        number=native_fault['signal']
        nxt = (f'BLOCKED: native signal -{number}; automatic relaunch disabled; '
               f'{used} historical failures retained; repair required')
    elif limit == 0:
        nxt = ('UNBOUNDED legacy retry policy is still recorded; update and restart all workers to apply '
               'the finite failure cap; reading status alone does not stop running workers')
    elif used >= limit:
        nxt = (f"BLOCKED: failure cap reached ({used} recorded; limit {limit}); no more automatic retries; "
               'failed checkpoints/logs retained; repair required')
    elif retrying:
        wait = (ended + delay - now) if ended else 0
        attempt_label=f'{used + 1}/{limit}' if limit>0 else f'{used + 1} (no limit)'
        nxt = f"retried automatically: attempt {attempt_label} " + \
              (f"starts in {age_text(wait)}" if wait > 0 else 'is waiting for the next free node'+
               (' (immediate checkpoint restart)' if delay==0 else ''))
    elif live_policies and policy and policy not in live_policies and not (recovered & live_policies):
        nxt = 'failed under older code: a node started with the current bundle retries it automatically'
    else:
        nxt = (f"not retried yet: a node started with the current bundle retries failed tasks automatically "
               + (f"(up to {limit} failures)" if limit>0 else '(no retry limit)'))
    if finished:
        nxt = 'work is already complete (a retry resumes from its checkpoint, nothing finished is redone); ' + nxt
    elif not real and rc in (9, -9, 137):
        nxt += f"; if it was host memory (OOM killer), check dmesg on {row.get('host') or 'that node'} first"
    if lifetime != used:nxt += f'; current verified repair failures {used}, historical failures {lifetime}'
    return dict(failure_cause=cause[:400], failure_next=nxt, failure_real=real, failure_at=at,
                failure_budget_used=used,failure_lifetime_count=lifetime)


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

LEGEND_TABLE = ("train/ext/eval = tasks done/all planned (ext = budget extension; a range or ≥ until that phase is planned) · updates = optimizer updates this plan adds "
                "(a branch or extension counts from where it starts; + = extension updates still to be planned) · fail = real error · marked = failed, no training error · "
                "intr = interrupted, reruns · stuck = marked running but no live node runs it (DEAD node or no worker) · "
                "other = unknown job state")
LEGEND = "✓ done  ▶n running (n updates)  ○ ready  · waiting  ✗ failed  ! no training error  ⏸ interrupted  ? no heartbeat"
COHORT_LABEL = {'olmo_math': 'OLMo MATH', 'qwen_mbpp': 'Qwen MBPP', 'olmo_mbpp': 'OLMo MBPP', 'qwen_math': 'Qwen MATH'}
COHORT_SHORT = {}   # four experiments: 'OLMo'/'Qwen' alone would be ambiguous, so the node table shows the full name
COLOR = False   # set by main() only when stdout is a terminal (never in logs, pipes or tests)
STYLE = {'ok': '32', 'run': '1;32', 'bad': '1;31', 'warn': '33', 'dim': '2', 'head': '1', 'info': '36'}


def paint(text, style):
    return f"\033[{STYLE[style]}m{text}\033[0m" if COLOR and style and text.strip() else text


def mark_style(mark):
    mark = mark.strip()
    if not mark or mark == '-':
        return 'dim'
    return {'✓': 'ok', '✗': 'bad', '?': 'warn', '⏸': 'warn', '○': None, '·': 'dim'}.get(mark[0], 'run' if mark[0] == '▶' else
           'bad' if '✗' in mark else 'warn' if ('?' in mark or '⏸' in mark) else 'run' if '▶' in mark else None)
PHASE_LABEL = {'prepare': 'prepare', 'train': 'training', 'freeze': 'budget freeze', 'extension': 'budget extension',
               'primary_evaluation': 'matched-step evaluation', 'primary_finalize': 'matched-step analysis',
               'evaluation': 'budget evaluation', 'finalize': 'analysis', 'complete': 'complete'}
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
    if status == 'failed' and row.get('failure_real') is False:
        return '!'
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


def runner_config(cohort):
    """base.json deep-merged with configs/cohorts/<cohort>.json from the runner next to this script."""
    def merge(a, b):
        out = dict(a)
        for k, v in b.items():
            out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
        return out
    here = Path(__file__).resolve().parent
    roots = ([Path(os.environ['V8_RUNNER_ROOT'])] if os.environ.get('V8_RUNNER_ROOT') else []) + \
            [here.parent / 'v8' / 'experiments']
    for root in roots:
        try:
            base = json.loads((root / 'configs' / 'base.json').read_text())
            mine = json.loads((root / 'configs' / 'cohorts' / f'{cohort}.json').read_text())
            return merge(base, mine)
        except (OSError, ValueError):
            continue
    return None


def planned_cells(queue, cohort, rows):
    """How many tasks each phase will have in total. Exact once that phase's plan is sealed
    (plans/<phase>.json: train when every On trajectory is done, extension at the budget freeze,
    evaluation after the extension); before that, the range the planner can produce from the config:
    per seed 1 On trajectory + one SR branch per grid point (core ∪ extended) + at most one
    Uniform destination; one extension per finished non-On run; at least one final evaluation per
    run at the matched endpoint, plus budget-endpoint evaluations."""
    phases = {'train': 'train', 'extension': 'ext', 'evaluation': 'eval'}
    have = {k: sum(1 for r in rows if (r.get('phase') or ('evaluation' if r.get('kind') == 'evaluate' else 'train')) == ph)
            for ph, k in phases.items()}
    sealed = {k: (Path(queue) / 'plans' / f'{ph}.json').is_file() for ph, k in phases.items()}
    out = {k: (have[k], have[k], 'exact') for k in phases.values() if sealed[k]}
    cfg = runner_config(cohort)
    seeds = ((cfg or {}).get('seeds') or {}).get('training') or []
    job_seeds = {r['seed'] for r in rows if isinstance(r.get('seed'), int)}
    grid = set(((cfg or {}).get('bank') or {}).get('core_grid') or []) | set(((cfg or {}).get('bank') or {}).get('extended_grid') or [])
    if not seeds or not grid or not job_seeds <= set(seeds):
        for k in phases.values():
            out.setdefault(k, (have[k], None, 'unknown'))
        return out
    n = len(seeds)
    train = out.get('train') or (n * (1 + len(grid)), n * (2 + len(grid)), 'range')
    if train[2] == 'range' and have['train'] > train[1]:
        train = (have['train'], None, 'unknown')      # this queue was planned with a different config
    out['train'] = train
    if 'ext' not in out:
        out['ext'] = ((train[0] - n, train[1] - n, 'range') if train[1] is not None else (have['ext'], None, 'unknown'))
    if 'eval' not in out:
        out['eval'] = (train[0], None, 'at least')
    t = (cfg or {}).get('training') or {}
    H, P = t.get('matched_update_endpoint'), t.get('prefix_updates')
    if isinstance(H, int) and isinstance(P, int) and out['train'][2] != 'exact':   # sealed: the real jobs decide
        out['_train_updates_min'] = n * ((H - P) + sum(H - tau for tau in grid if tau < H))
    return out


def cells_text(done, plan):
    lo, hi, kind = plan
    if kind == 'exact':
        return f"{done}/{lo}"
    if kind == 'range':
        return f"{done}/{lo}" if lo == hi else f"{done}/{lo}-{hi}"
    if kind == 'at least':
        return f"{done}/≥{lo}"
    return f"{done}/?"


def branch_column(arm):
    """Seed-table column of a training arm: On, SR<tau>, or U (the Uniform destination; tau differs per seed)."""
    col = short_arm(arm)
    return 'U' if re.fullmatch(r'U\d+', col) else col


def run_branches(rows):
    """run_id -> (column, is_extension) for every training and extension job of a queue."""
    return {r['run_id']: (branch_column(r['arm']), r.get('phase') == 'extension')
            for r in rows if r.get('kind') != 'evaluate' and r.get('run_id')}


def eval_branch(r, runs):
    """(column, is_extension) of the run an evaluation job scores: from the queue's own training/extension
    jobs, else from the run id (<cohort>.<role>.s<seed>.<method>.<hash>)."""
    if r.get('run_id') in runs:
        return runs[r['run_id']]
    m = re.search(r'\.s\d+\.([A-Za-z0-9_]+)\.[0-9a-f]+$', str(r.get('run_id') or ''))
    if m:return branch_column(m.group(1)),False
    # Older error records can have no run id; retain their recorded evaluation stage.
    return (branch_column(r['arm']) if not r.get('run_id') and r.get('arm') else '?'),False


def task_name(r, runs):
    """'s5 SR25', 's5 SR25 ext', 's5 eval SR25@275', 's5 eval SR25 ext@400'."""
    if r.get('kind') == 'evaluate':
        col, is_ext = eval_branch(r, runs)
        update = f"@{r['update']}" if r.get('update') not in (None, '') else ''
        return f"s{r['seed']} eval {col}{' ext' if is_ext else ''}{update}"
    return f"s{r['seed']} {short_arm(r['arm'])}" + (' ext' if r.get('phase') == 'extension' else '')


def clean_block(base, cohort, now, host, width):
    queue = Path(base) / cohort / '.multi-node'
    label = COHORT_LABEL.get(cohort, cohort)
    if not queue.exists():
        return [], f"{label:<10} not started", None, [], dict(label=label, phase='not started', total=0)
    state, rows = collect(base, cohort, now)
    trained = {r.get('run_id'): r['arm'] for r in rows if r.get('kind') == 'train' and r.get('phase') != 'extension'}
    train_cols, seeds, cells, ext, ev = [], [], {}, {}, {}
    # Draw every planned cell, not only jobs that exist: a branch job is created only once its parent
    # snapshot is saved, extension/evaluation jobs only when their phase is planned.
    cfg = runner_config(cohort)
    cfg_seeds = [x for x in (((cfg or {}).get('seeds') or {}).get('training') or []) if isinstance(x, int)]
    bank = (cfg or {}).get('bank') or {}
    grid = sorted({x for x in (bank.get('core_grid') or []) + (bank.get('extended_grid') or []) if isinstance(x, int)})
    job_seeds = {r['seed'] for r in rows if isinstance(r.get('seed'), int)}
    use_plan = bool(cfg_seeds and grid and job_seeds <= set(cfg_seeds))
    u_tau = {}
    runs = run_branches(rows)
    for r in rows:
        seed = r['seed']
        if seed not in seeds:
            seeds.append(seed)
        if r.get('kind') == 'evaluate':
            ev.setdefault((seed, eval_branch(r, runs)[0]), []).append(r)
        elif r.get('phase') == 'extension':
            ext.setdefault((seed, branch_column(r['arm'])), []).append(r)
        else:
            col = short_arm(r['arm'])
            if use_plan and re.fullmatch(r'U\d+', col):
                u_tau[seed] = int(col[1:]); col = 'U'      # Uniform destination: one column, tau differs per seed
            if col not in train_cols:
                train_cols.append(col)
            cells[(seed, col)] = symbol(r)
    def col_key(c):
        kind = 0 if c == 'On' else 1 if c.startswith('SR') else 2 if c.startswith('U') else 3
        num = int(re.sub(r'\D', '', c) or 0)
        return (kind, num, c)
    train_cols.sort(key=col_key)
    sealed = {ph: (queue / 'plans' / f'{ph}.json').is_file() for ph in ('train', 'extension', 'evaluation')}
    if use_plan and not sealed['train']:     # once sealed, the queue's own jobs are the complete grid
        planned_cols = ['On', *(f'SR{t}' for t in grid), 'U']
        train_cols = planned_cols + [c for c in train_cols if c not in planned_cols]
        seeds = list(dict.fromkeys([*cfg_seeds, *seeds]))
    seeds.sort(key=lambda x: (not isinstance(x, int), x if isinstance(x, int) else str(x)))
    missing = {'train': '-' if sealed['train'] or not use_plan else '·',
               'ext': '-' if sealed['extension'] or not use_plan else '·',
               'eval': '-' if sealed['evaluation'] or not use_plan else '·'}
    def count_cell(items, kind='ext'):
        if not items:
            return missing[kind]
        if len(items) == 1:
            return symbol(items[0])
        if all(i['status'] == 'pending' for i in items):
            return '○' if any(i.get('ready') for i in items) else '·'
        done = sum(i['status'] == 'done' for i in items)
        run = sum(i['status'] == 'running' for i in items)
        if done == len(items):
            return '✓'
        mark = ('✗' if any(i['status'] == 'failed' for i in items) else '▶' if run else
                '?' if any(i['status'] in ('unconfirmed', 'stopped') for i in items) else
                '⏸' if any(i['status'] == 'interrupted' for i in items) else '')
        return f"{done}/{len(items)}" + mark
    cols = list(train_cols)
    for c in sorted({c for _, c in ext} | {c for _, c in ev}, key=col_key):
        if c not in cols:
            cols.append(c)
    table, ext_table, eval_table = {}, {}, {}
    show_later = bool(ext or ev or use_plan)
    for seed in seeds:
        for c in cols:
            table[(seed, c)] = cells.get((seed, c), missing['train'])
            ext_table[(seed, c)] = '-' if c == 'On' else count_cell(ext.get((seed, c), []), 'ext')   # On is never extended
            eval_table[(seed, c)] = count_cell(ev.get((seed, c), []), 'eval')
    tables = [table] + ([ext_table, eval_table] if show_later else [])
    widths = {c: max(len(c), *(len(t[(s, c)]) for t in tables for s in seeds)) for c in cols} if seeds else {}
    phase_text = PHASE_LABEL.get(state.get('phase'), state.get('phase', 'not started'))
    title = f"{label} · {phase_text} "
    lines = [paint(title, 'head') + paint('─' * max(3, min(width, 100) - len(title)), 'dim')]
    def row_text(first, cells, header=False):
        plain, coloured = [first], [paint(first, 'dim') if header else first]
        for c in cols:
            cell = f'{cells[c]:<{widths[c]}}'
            plain.append(cell)
            coloured.append(paint(cell, 'dim' if header else mark_style(cells[c])))
        return clip('  '.join(plain), width) if len('  '.join(plain)) > width else '  '.join(coloured)
    if seeds:
        lines.append(row_text(' train', {c: c for c in cols}, header=True))
        for seed in seeds:
            lines.append(row_text(f'{seed!s:>6}', {c: table[(seed, c)] for c in cols}))
        if show_later:
            for name, part in ((' ext', ext_table), (' eval', eval_table)):
                lines.append(row_text(f'{name:<6}', {c: c for c in cols}, header=True))
                for seed in seeds:
                    lines.append(row_text(f'{seed!s:>6}', {c: part[(seed, c)] for c in cols}))
        if use_plan:
            known = ' · '.join(f"s{k} @{v}" for k, v in sorted(u_tau.items()))
            lines.append(paint(clip("   ext = budget extension of that branch (On is never extended) · eval = evaluations of that branch "
                                    "(one symbol, or done/all when it has several)", width), 'dim'))
            lines.append(paint(clip(f"   · = not created yet (a branch appears when its parent snapshot is saved) · U = Uniform at the "
                                    f"seed's switch point" + (f": {known}" if known else ' (known when that On trajectory is done)'),
                                    width), 'dim'))
    else:
        lines.append('  (no tasks yet)')
    problems = [r for r in rows if r['status'] in ATTENTION]
    unreadable = access_problems(queue)
    if state.get('_unreadable_jobs'):
        unreadable.append(f"{len(state['_unreadable_jobs'])} unreadable job record(s)")
    attention = []
    if state.get('failure'):
        attention.append(f"  ✗ {label}: queue failure: {state['failure']}")
    def when(r):
        t = stamp(r)
        return f" {age_text(now - t)} ago" if t else ''
    def stamp(r):
        for t in (r.get('ended'), r.get('started')):
            if isinstance(t, (int, float)) and not isinstance(t, bool):
                return t
        return 0
    def rank(r):
        if r['status'] == 'failed':
            return 0 if r.get('failure_real', True) else 1
        return {'unconfirmed': 2, 'stopped': 2, 'interrupted': 3}.get(r['status'], 4)
    for r in sorted(problems, key=lambda r: (rank(r), -stamp(r))):
        name = task_name(r, runs)
        if r['status'] == 'failed':
            real = r.get('failure_real', True)
            where = f" on {r['host']}" if r.get('host') else ''
            head = (f"{'✗' if real else '!'} {label} {name} {'FAILED' if real else 'marked failed'}{when(r)}{where}"
                    f"{r.get('failure_at', '')} (attempt {r['attempt']})")
            attention.append('  ' + head)
            attention.append('      why:  ' + (r.get('failure_cause') or r.get('error') or 'no error recorded; see its log'))
            attention.append('      next: ' + (r.get('failure_next') or 'start one node with --retry-failed after fixing'))
            if r.get('log'):
                attention.append('      log:  ' + r['log'])
            continue
        elif r['status'] == 'interrupted':
            text = f"⏸ {label} {name} interrupted{when(r)}: reruns automatically on the next free node"
        else:
            beat = r.get('beat_age')
            if not r.get('worker'):
                text = f"? {label} {name}: {r.get('error') or 'marked running but no worker is recorded'} — not running anywhere"
            else:
                text = (f"? {label} {name} on {r.get('host') or '?'}: " +
                        (f"no heartbeat {age_text(beat)}" if beat is not None else 'no heartbeat') + ' — that node is DEAD (see nodes)')
        attention.append('  ' + text)
    for r in rows:
        if r['status'] not in ('done', 'running', 'pending', 'failed', 'interrupted', 'unconfirmed', 'stopped'):
            attention.append(f"  ? {label} s{r['seed']} {short_arm(r['arm'])}: unknown job state {r['status']!r} "
                             f"(see .multi-node/jobs/{r.get('key')}.json)")
    for u in unreadable[:3]:
        attention.append(f"  ! {label}: cannot read {u}")
    workers = worker_records(queue, now)
    live = [w for w in workers if w['status'] != 'stale']
    counts = {k: 0 for k in ('done', 'running', 'ready', 'waiting', 'attention', 'dead', 'interrupted', 'marked', 'other')}
    for r in rows:
        key = ('done' if r['status'] == 'done' else 'running' if r['status'] == 'running'
               else 'dead' if r['status'] in ('unconfirmed', 'stopped')
               else 'interrupted' if r['status'] == 'interrupted'
               else 'marked' if r['status'] == 'failed' and r.get('failure_real') is False
               else 'attention' if r['status'] == 'failed'
               else ('ready' if r.get('ready') else 'waiting') if r['status'] == 'pending' else 'other')
        counts[key] += 1
    evals = [r for r in rows if r.get('kind') == 'evaluate']
    exts = [r for r in rows if r.get('kind') != 'evaluate' and r.get('phase') == 'extension']
    trains = [r for r in rows if r.get('kind') != 'evaluate' and r.get('phase') != 'extension']
    planned_tasks = planned_cells(queue, cohort, rows)
    phase = PHASE_LABEL.get(state.get('phase'), state.get('phase', 'not started'))
    total = len(rows)
    # Training progress in optimizer updates (what 'how far along' means for a run), plus task counts.
    # Each job counts only the updates it adds itself: a branch or a budget extension resumes from its
    # parent's saved update (task.parent.completed_update), so counting from 0 would count that part again.
    planned = done_updates = 0
    for r in rows:
        if r.get('kind') == 'train' and isinstance(r.get('total'), int):
            start = min(r.get('start') or 0, r['total'])
            planned += r['total'] - start
            reached = r['total'] if r['status'] == 'done' else (r['done'] if isinstance(r.get('done'), int) else start)
            done_updates += max(0, min(reached, r['total']) - start)
    phases_known = {k: v for k, v in planned_tasks.items() if not k.startswith('_')}
    planned_total = max(total, sum(v[0] for v in phases_known.values())) if phases_known else total
    pct = min(100, round(100 * counts['done'] / planned_total)) if planned_total else 0
    filled = min(12, round(12 * counts['done'] / planned_total)) if planned_total else 0
    bar = '█' * filled + '░' * (12 - filled)
    summary = (f"{label:<10} {phase:<13} {bar} {counts['done']}/{total} done ({pct}%)  running {counts['running']}  "
               f"ready {counts['ready']}  waiting {counts['waiting']}  failed {counts['attention']}"
               + (f"  interrupted {counts['interrupted']}" if counts['interrupted'] else '')
               + (f"  on dead node {counts['dead']}" if counts['dead'] else '')
               + (f"  updates {done_updates:,}/{planned:,}" if planned else ''))
    wanted = counts['running'] + counts['ready']
    nodes = (len(live), sum(1 for w in live if any(r.get('worker') == w['worker'] and r['status'] in ('running', 'unconfirmed') for r in rows)))
    stats = dict(label=label, phase=phase, done=counts['done'], total=total, pct=pct, running=counts['running'],
                 ready=counts['ready'], waiting=counts['waiting'], failed=counts['attention'],
                 interrupted=counts['interrupted'], dead=counts['dead'], marked=counts['marked'], other=counts['other'],
                 train=(sum(r['status'] == 'done' for r in trains), len(trains)),
                 ext=(sum(r['status'] == 'done' for r in exts), len(exts)),
                 eval=(sum(r['status'] == 'done' for r in evals), len(evals)),
                 planned=planned_tasks,
                 scheduler_phase=state.get('phase'),side_work=state.get('ready_side_work'),
                 base_planning_error=state.get('base_planning_error'),
                 parallel_work=isinstance(state.get('execution_order'),dict) and state['execution_order'].get('policy')=='ready-checkpoint-evaluation-extension-v1',
                 updates=(done_updates, max(planned, planned_tasks.get('_train_updates_min') or 0)),
                 updates_more=any(v[2] != 'exact' for k, v in planned_tasks.items() if not k.startswith('_')))
    return lines, summary, (nodes, wanted), attention, stats


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
    nodes, queues, info, runs_by_cohort = {}, [], {}, {}
    def entry(worker, host):
        return nodes.setdefault(worker, dict(worker=worker, host=host, beats={}, jobs=[], complete=True))
    for base in bases:
        for cohort in cohorts:
            queue = Path(base) / cohort / '.multi-node'
            if not queue.exists():
                continue
            queues.append(queue)
            state, rows = collect(base, cohort, now)
            runs_by_cohort[cohort] = run_branches(rows)
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
                task = task_name(r, runs_by_cohort.get(cohort, {}))
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
                                   f"no heartbeat {age_text(r['quiet'])}" +
                                   (", task lock held (admission / model load / checkpoint)" if r.get('lease') is True else '')))
                    else:
                        row.update(state='running', doing=short_progress(r, now) or
                                   (f"log written {age_text(r['activity_age'])} ago" if r.get('activity_age') is not None
                                    else 'started, no output yet'))
                else:
                    if local_alive:
                        doing = f"process {pid} still alive here but silent: hung? kill {pid}, rerun same command"
                    else:
                        doing = ('down or hung → restart it (same command)' if r.get('beat_age') is not None
                                 else 'never sent a heartbeat → restart it (same command)')
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
        silent = {c: v[0] for c, v in e['beats'].items()
                  if c not in live and c in info and info[c]['open'] and v[0] > BUSY_SILENT_SECONDS}
        busy = {c: a for c, a in silent.items() if a <= GAVE_UP_SECONDS}
        if busy:
            cohort, age = min(busy.items(), key=lambda x: x[1])
            label = COHORT_LABEL.get(cohort, cohort)
            if info[cohort]['phase'] in CPU_PHASES:
                log = cpu_log_age(info[cohort]['queue'], now)
                doing = f"CPU stage {PHASE_LABEL.get(info[cohort]['phase'], info[cohort]['phase'])}" + \
                        (f", log {age_text(log)} ago" if log is not None else '')
            else:
                doing = f"checking/claiming tasks ({age_text(age)})"
            out.append(dict(base, state='busy', experiment=label, doing=doing))
            continue
        ready = sum(info[c]['ready'] for c in live if c in info)
        doing = (f"idle while {ready} task{'s are' if ready != 1 else ' is'} ready (GPU busy? see its console)"
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
    out, stats, attention = [], [], []
    wanted = 0
    mine = None
    for base in bases:
        for cohort in cohorts:
            lines, _summary, extra, problems, stat = clean_block(base, cohort, now, host, width)
            attention += problems
            stats.append(stat)
            if lines:
                out += [*lines, '']
            if extra:
                wanted += extra[1]
            if mine is None and lines:
                _, rows = collect(base, cohort, now)
                job = next((x for x in rows if x['status'] == 'running' and x.get('host') == host), None)
                if job:
                    mine = f"▶ {task_name(job, run_branches(rows))} ({COHORT_SHORT.get(COHORT_LABEL.get(cohort, cohort), COHORT_LABEL.get(cohort, cohort))})"
    out.append(paint(clip(LEGEND, width), 'dim'))
    import textwrap
    for part in textwrap.wrap(LEGEND_TABLE, max(60, width - 2)):
        out.append(paint(part, 'dim'))
    stamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))
    head = f"━━ V8 status · {stamp} "
    out.append(paint(head + '━' * max(3, min(width, 100) - len(head)), 'head'))
    out.append(paint('   ' + ', '.join(str(b) for b in bases), 'dim'))

    # 1. one aligned row per experiment
    cols = ('phase', 'progress', 'train', 'ext', 'eval', 'running', 'ready', 'waiting', 'failed', 'marked', 'interrupted',
            'on dead node', 'other', 'updates')
    table = []
    for st in stats:
        if not st.get('total'):
            table.append(dict(name=st['label'], phase=st.get('phase', 'not started')))
            continue
        filled = min(12, round(12 * st['pct'] / 100))
        done_u, planned = st['updates']
        plan = {k: v for k, v in (st.get('planned') or {}).items() if not k.startswith('_')}
        cells = {k: cells_text(st.get(k, (0, 0))[0], plan.get(k, (st.get(k, (0, 0))[1], st.get(k, (0, 0))[1], 'exact')))
                 for k in ('train', 'ext', 'eval')}
        table.append(dict(name=st['label'], phase=st['phase'], progress=f"{'█' * filled}{'░' * (12 - filled)} {st['pct']:>3}%",
                          train=cells['train'], ext=cells['ext'], eval=cells['eval'],
                          other=st.get('other', 0), running=st['running'], ready=st['ready'], waiting=st['waiting'],
                          failed=st['failed'], marked=st.get('marked', 0), interrupted=st['interrupted'], **{'on dead node': st['dead']},
                          updates=(f"{done_u:,}/{planned:,}" + ('+' if st.get('updates_more') else '')) if planned else '-'))
    used = [c for c in cols if c in ('phase', 'progress', 'train', 'ext', 'eval', 'running', 'updates')
            or any(isinstance(r.get(c), int) and r.get(c) for r in table)]
    def cell(r, c):
        v = r.get(c, '')
        return '-' if v == 0 else str(v)
    title = {'running': 'run', 'waiting': 'wait', 'failed': 'fail', 'interrupted': 'intr', 'on dead node': 'stuck'}
    name_w = max(10, *(len(r['name']) for r in table)) if table else 10
    w = {c: max(len(title.get(c, c)), *(len(cell(r, c)) for r in table)) for c in used}
    right = {'train', 'ext', 'eval', 'running', 'ready', 'waiting', 'failed', 'marked', 'interrupted', 'on dead node', 'other', 'updates'}
    fmt = lambda c, v: f"{v:>{w[c]}}" if c in right else f"{v:<{w[c]}}"
    out.append(paint(' ' + ' ' * name_w + '  ' + '  '.join(fmt(c, title.get(c, c)) for c in used), 'dim'))
    for r in table:
        parts = [paint(f" {r['name']:<{name_w}}", 'head')]
        for c in used:
            v = fmt(c, cell(r, c))
            style = ('bad' if c in ('failed', 'on dead node') and v.strip() not in ('-', '') else
                     'warn' if c in ('interrupted', 'marked', 'other') and v.strip() not in ('-', '') else
                     'run' if c == 'running' and v.strip() not in ('-', '') else
                     'ok' if c == 'progress' else 'dim' if v.strip() in ('-', '') else None)
            parts.append(paint(v, style))
        out.append('  '.join(parts))
    fill = []
    for st in stats:
        plan = st.get('planned')
        if not plan:
            continue
        plan = {k: v for k, v in plan.items() if not k.startswith('_')}
        lo = sum(v[0] for v in plan.values())
        exact = all(v[2] == 'exact' for v in plan.values())
        hi = None if any(v[1] is None for v in plan.values()) else sum(v[1] for v in plan.values())
        total = f"{lo}" if exact else f"{lo}-{hi}" if hi is not None and hi != lo else f"≥{lo}" if hi is None else f"{lo}"
        parts = ' + '.join(f"{k} {cells_text(0, v).split('/', 1)[1]}" for k, v in plan.items())
        fill.append(f"{st['label']} {total} ({parts})")
    if fill:
        out.append(paint(' cells to fill: ' + ' · '.join(fill), None))
        if any(v[2] != 'exact' for st in stats for k, v in (st.get('planned') or {}).items() if not k.startswith('_')):
            note=('   ready train/ext/eval are dispatched together; exact counts grow with available checkpoints'
                  if any(st.get('parallel_work') for st in stats) else
                  '   exact when each phase is planned: train when every On trajectory is done, matched-step eval before ext, budget eval after ext')
            out.append(paint(note, 'dim'))
    for st in stats:
        side=st.get('side_work')
        if not st.get('parallel_work') or not isinstance(side,dict) or st.get('scheduler_phase')=='complete':continue
        if side.get('budget_ready'):
            note='budget frozen; ext/eval join the rotation when their checkpoints are ready'
        elif side.get('evaluation_independent_of_budget'):
            note='eval: published matched checkpoints run independently; ext: waiting for reference-policy cost budget'
            if side.get('budget_error'):note+='; budget error: '+str(side['budget_error'])
        elif side.get('reason')=='reference_endpoint_unfinished':
            note='waiting for the selected reference-policy endpoint to finish before freezing budget'
        else:
            note='waiting for complete reference-policy endpoints and measured costs before freezing budget'
        out.append(paint(f" {st['label']} ext/eval: {note}", 'dim'))
        for kind,error in (side.get('planning_errors') or {}).items():
            out.append(paint(clip(f" {st['label']} {kind} planner blocked: {error}; other ready work remains eligible",width),'warn'))
        if st.get('base_planning_error'):
            out.append(paint(clip(f" {st['label']} stage planner blocked: {st['base_planning_error']}",width),'warn'))

    # 2. problems, newest failures first
    if attention:
        items = sum(1 for line in attention if not line.startswith('      '))
        out.append('')
        out.append(paint(f" needs attention ({items})", 'head'))
        shown_items, style = 0, None
        for line in attention:
            if not line.startswith('      '):
                shown_items += 1
                if shown_items > 12:
                    out.append(paint(f"   … {items - 12} more (sh scripts/run_v8_nodes.sh --status lists every task)", 'dim'))
                    break
                mark = line.strip()[:1]
                style = 'bad' if mark == '✗' else 'warn' if mark in '?⏸!' else None
                out.append(paint(clip(line, width), style))
            else:
                import textwrap
                m = re.match(r'(\w+:)\s+(.*)', line.lstrip())
                label_part, text = (m.group(1), m.group(2)) if m else ('', line.strip())
                wrapped = textwrap.wrap(text, max(30, width - 12)) or ['']
                if len(wrapped) > 3:
                    wrapped = wrapped[:3]; wrapped[-1] = wrapped[-1][:max(30, width - 13)] + '…'
                for k, part in enumerate(wrapped):
                    out.append(paint(f"      {label_part if k == 0 else '':<5} {part}",
                                     None if label_part in ('why:', 'next:') else 'dim'))

    # 3. what every node is doing
    out.append('')
    table = node_table(bases, cohorts, now, host)
    shown = table if all_nodes else [r for r in table if r['state'] not in ('gone', 'finished')]
    order = ('running', 'busy', 'starting', 'idle', 'DEAD', 'gone', 'finished')
    per = {k: sorted({(r['host'], r['pid']) for r in table if r['state'] == k}) for k in order}
    alive = sum(len(per[k]) for k in ('running', 'busy', 'starting', 'idle'))
    if table:
        hosts = len({r['host'] for r in table if r['state'] not in ('gone', 'finished')})
        hidden = [f"{len(per[k])} {k}" for k in ('gone', 'finished') if per[k]]
        counts = ' · '.join(f"{len(per[k])} {k}" for k in ('running', 'busy', 'starting', 'idle') if per[k]) or '0 running'
        line = paint(f" nodes ({hosts} host{'s' if hosts != 1 else ''})", 'head') + f"  {counts}"
        line += '  ' + (paint(f"✗ {len(per['DEAD'])} DEAD: {', '.join(h for h, _ in per['DEAD'])}", 'bad') if per['DEAD']
                        else paint('0 DEAD', 'dim'))
        out.append(line)
        more = max(0, wanted - alive)
        out.append(paint(f"   tasks that can run now {wanted} · alive workers {alive}", 'dim') +
                   (paint(f"  → {more} more 4-GPU node{'s' if more != 1 else ''} would help", 'info') if more else '') +
                   (paint(f"   ({', '.join(hidden)} not shown; --all-nodes)", 'dim') if hidden and not all_nodes else ''))
        if shown:
            ncols = (('host', 'host'), ('gpus', 'gpus'), ('state', 'state'), ('experiment', 'exp'), ('task', 'task'),
                     ('updates', 'updates'), ('took', 'time'), ('beat', 'beat'))
            rows = [dict(r, experiment=COHORT_SHORT.get(r['experiment'], r['experiment']),
                         beat=r['beat'].replace(' ago', '')) for r in shown]
            nw = {k: max(len(t), *(len(str(r[k])) for r in rows)) for k, t in ncols}
            out.append(paint('   ' + '  '.join(f"{t:<{nw[k]}}" for k, t in ncols) + '  now', 'dim'))
            for r in rows:
                mark = '✗' if r['state'] == 'DEAD' else '▶' if r['state'] == 'running' else '·' if r['state'] == 'busy' else ' '
                text = clip(f" {mark} " + '  '.join(f"{str(r[k]):<{nw[k]}}" for k, _ in ncols) + f"  {r['doing']}", width)
                out.append(paint(text, {'DEAD': 'bad', 'busy': 'warn', 'starting': 'info', 'idle': 'dim'}.get(r['state'])))
            if per['DEAD']:
                out.append(paint("   DEAD = task assigned and no heartbeat for over 3 min (or lock held but silent over 10 min)", 'dim'))
    else:
        out.append(paint(' nodes', 'head') + '  no worker has started yet' + (f" · tasks that can run now {wanted}" if wanted else ''))

    # 4. this node
    gpu_text = ' '.join(f'{u}%' for i, u, _ in gpus) if gpus else 'unknown'
    out.append(paint(' this node', 'head') + f"  {host} · GPU {gpu_text} · " + (paint(mine, 'run') if mine else paint('no V8 task here', 'dim')))
    return '\n'.join(out) + '\n'


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
        # run_v8_nodes.sh accepts every configured cohort, including those outside the default three.
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
    global COLOR
    COLOR = sys.stdout.isatty() and not os.environ.get('NO_COLOR') and os.environ.get('TERM') != 'dumb'
    # An explicit V8_ROOT wins; otherwise the roots of this node's running V8 processes, else the default.
    bases = [Path(os.environ['V8_ROOT'])] if os.environ.get('V8_ROOT') else (active_roots() or [root(os.environ)])
    if args.watch is None:
        sys.stdout.write(render(bases, cohorts=tuple(args.cohort or COHORTS), style='simple' if args.simple else 'srgc' if args.classic else 'clean', all_nodes=args.all_nodes))
        return 0
    try:
        while True:
            sys.stdout.write('\033[H\033[2J\033[3J' + render(bases, cohorts=tuple(args.cohort or COHORTS), style='simple' if args.simple else 'srgc' if args.classic else 'clean', all_nodes=args.all_nodes) + f"\n(refresh every {int(args.watch)}s · Ctrl-C to stop)\n")
            sys.stdout.flush()
            time.sleep(max(1.0, args.watch))
    except KeyboardInterrupt:
        return 0


if __name__ == '__main__':
    sys.exit(main())
