#!/usr/bin/env python3
"""V8 queue status in the same layout as `sh scripts/run_srgc.sh <dataset> status` (read-only).

    sh scripts/v8_status.sh            # one snapshot of all four experiments (same as run_v8_nodes.sh status)
    sh scripts/v8_status.sh --watch    # refresh every 3 s until Ctrl-C (--watch 30 for 30 s)

The screen, top to bottom: header and this node; time left per experiment and for all of them (KST clock, what
makes the range wide); per experiment the stage and "done N of M" per stage with how many are running / can start /
restart by themselves / waiting / STOPPED; every unfinished task grouped by what happens next (running: where, how
far, time left; waiting: the step it waits for); the seed x branch grid; live nodes and why idle ones are idle.

No locks, signals, Torch imports or queue writes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v8_probe import active_roots, last_row, read, root, tail, worker_status  # noqa: E402
from v8_progress_text import scheduler_text
from srgc_v8_nodes import (retry_budget_failures, fatal_native_failure, evaluation_recovery_attempts, native_failure_blocks,
                            native_crash_hosts, failure_limit, granted_view, task_key)

COHORTS = ('olmo_math', 'olmo_mbpp', 'qwen_math', 'qwen_mbpp')   # model, then task (user 11-09: 'status 순서 맞춰')
ATTENTION = {'failed', 'interrupted', 'stopped', 'unconfirmed'}
STALE_SECONDS = 180  # workers heartbeat every 10 s while a child runs (srgc_v8_nodes.Runtime.process)
# Clean view liveness (user: 'idle 아닌데 자꾸 idle 이래 이미 삭제한 노드인데', 'status 업데이트 더 빠르게'):
# Heartbeats are written every 3-5 s, but another node's file can be read up to ~60 s stale on a shared
# file system and node clocks differ; a lock held on another node can look free from here. So a lock
# probe is only positive evidence (held = alive), and a task counts as lost (resumed by the next free node) after over 3 min without any heartbeat.
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
            row = dict(_job=job, key=path.stem, task=str(task.get('task_id') or task.get('run_id') or path.stem),
                       arm=str(task.get('method_id') or task.get('stage') or task.get('kind') or '-'),
                       seed=task.get('training_seed', '-'), status=job.get('state', 'pending'),
                       total=task.get('horizon') or task.get('endpoint') or task.get('total_updates'),
                       attempt=len(attempts), started=attempt.get('started_at'), ended=attempt.get('ended_at'), update=task.get('completed_update'), run_id=task.get('run_id'), kind=task.get('kind'), phase=job.get('phase'), host=shown_worker.rsplit('.', 1)[0] if shown_worker else None, worker=shown_worker, done=None)
            if task.get('kind') == 'evaluate':
                row['eval_spec'] = task.get('evaluation_spec_hash')
            parent = task.get('parent') if isinstance(task.get('parent'), dict) else {}
            start = exit_code(parent.get('completed_update'))   # int parse; branches/extensions start here
            row['start'] = start if isinstance(start, int) and start >= 0 else 0
            if job.get('phase') == 'extension' and isinstance(task.get('hard_cap'), int) and task['hard_cap'] > row['start']:
                # a budget extension has horizon=H and starts at H; it runs on until its budget is spent, at most hard_cap
                row['total'], row['updates_end'] = task['hard_cap'], task.get('horizon')
            if row['status'] == 'running' and not shown_worker:
                row['status'] = 'unconfirmed'
                row['error'] = 'marked running but no worker is recorded for it'
                if lease_held(queue, path.stem) is not True:
                    row['lost'] = True     # srgc_v8_nodes claim takes a running job whose lease is free
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
                    # srgc_v8_nodes claim: a running job whose lease is free is taken by the next free worker
                    # and resumed from its last checkpoint, so the screen shows it as ready, not as a dead node.
                    row['lost'] = True
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


NATIVE_SIGNALS = {4: 'SIGILL', 6: 'SIGABRT', 7: 'SIGBUS', 8: 'SIGFPE', 11: 'SIGSEGV'}


def native_crash_frames(stack, n=3):
    """Innermost Python frames of the crashing thread in a faulthandler dump ('Current thread ...' block).

    The dump lists the thread that received the signal first as 'Current thread'; an empty block or
    '<no Python frame>' means the crash happened in a thread that runs no Python code (a native library)."""
    lines = str(stack or '').splitlines()
    start = next((i for i, line in enumerate(lines)
                  if line.startswith(('Current thread', 'Stack (most recent call first)'))), None)
    if start is None:
        return None
    frames = []
    for line in lines[start + 1:]:
        m = re.match(r'\s+File "(.*)", line (\d+) in (.*)', line)
        if not m:
            break
        path = m.group(1).replace('\\', '/')
        short = path.split('site-packages/', 1)[1] if 'site-packages/' in path else '/'.join(path.split('/')[-2:])
        frames.append(f"{m.group(3).strip()} ({short}:{m.group(2)})")
        if len(frames) >= n:
            break
    return frames


def fault_text(path, size=262144):
    """The faulthandler dump in a rank dump file or, when no rank dump was found, in the task log. A crash
    writes its stack at the END of the log, so the tail is read too (it was only the first 64 KB: user 11-13
    'no fault dump was written for this rank ... 계속 쌓여'). Returns the text from the last
    'Fatal Python error' on, else the head of the file."""
    try:
        with Path(path).open('rb') as handle:
            head = handle.read(65536)
            end = handle.seek(0, 2)
            handle.seek(max(len(head), end - size))
            tail = handle.read(size)
    except OSError:
        return None
    for chunk in (tail, head):
        text = chunk.decode(errors='replace')
        at = text.rfind('Fatal Python error')
        if at >= 0:
            return text[at:]
    return head.decode(errors='replace')


def native_story(ranks):
    """(cause, dump path) for the first rank that died of a native signal, or (None, None)."""
    for f in ranks:
        try:
            number = int(f.get('signal'))
        except (TypeError, ValueError):
            continue
        if number not in NATIVE_SIGNALS:
            continue
        stack, dump = f.get('native_python_stack'), f.get('native_fault_log') or f.get('error_file')
        if not stack and dump:
            stack = fault_text(dump)
        progress = f.get('last_progress') if isinstance(f.get('last_progress'), dict) else {}
        stage = progress.get('stage')
        frames = native_crash_frames(stack)
        where = (' ← '.join(frames) if frames else
                 'a thread with no Python frame (inside a native library: CUDA/NCCL/tokenizer etc.)' if frames == [] else
                 'not in the dump (no fault dump was written for this rank)')
        cause = (f"rank {f.get('rank', '?')} {NATIVE_SIGNALS[number]} (exit -{number}), native crash"
                 + (f" during {stage}" if stage else '') + f"; crashed in {where}")
        # torchrun stops the other ranks after one crashes (SIGTERM): only a real Python error elsewhere is news
        others = [g for g in ranks if g is not f and re.search(r'(Error|Exception)\b', str(g.get('message') or ''))
                  and not re.search(r'SIG(TERM|INT|KILL)|Signal (2|9|15)\b|ChildFailedError', str(g.get('message')))]
        if others:
            cause += f"; also rank {others[0].get('rank', '?')}: {str(others[0]['message'])[:120]}"
        return cause, (dump if dump and str(dump).endswith('.log') and 'native-fault' in str(dump) else None)
    return None, None


def attempt_evidence(directory, attempt, lines):
    """What an attempt's own files say when its job record has no exit code or error (attempts written by older
    code): the exit code / signal torchrun logged, each rank's last recorded stage, the crashing frame of a native
    fault dump, runtime repairs, and where it ran. Returns (text or None, files checked)."""
    aid = str(attempt.get('id') or '')
    progress = Path(directory) / '.multi-node' / 'progress' / aid
    checked = [f".multi-node/logs/{aid}.log", f".multi-node/progress/{aid}/rank-*.json",
               f".multi-node/progress/{aid}/native-fault-rank-*.log", f".multi-node/progress/{aid}/runtime-repair-rank-*.json"]
    parts = []
    text = '\n'.join(lines or [])
    codes = [int(n) for n in re.findall(r'exitcode\s*:\s*(-?\d+)', text)]
    signals = [int(n) for n in re.findall(r'Signal (\d+) \(SIG', text)]
    code = next((c for c in reversed(codes) if c not in (0, -2, -15)), None)
    number = -code if code is not None and code < 0 else next((n for n in reversed(signals) if n not in (2, 15)), None)
    if number:
        try:
            name = signal.Signals(number).name
        except ValueError:
            name = f'signal {number}'
        parts.append(f"{name} (exit -{number}) in its log")
    elif code is not None:
        parts.append(f"exit {code} in its log")
    stages = []
    try:
        files = sorted(progress.glob('rank-*.json'))
    except OSError:
        files = []
    for path in files:
        event = read(path)
        if isinstance(event, dict) and event.get('stage') and re.fullmatch(r'rank-\d+\.json', path.name):
            stages.append(f"rank {path.stem.split('-')[1]} {STAGE_SHORT.get(event['stage'], event['stage'])}")
    if stages:
        parts.append('last stage ' + ', '.join(stages[:4]))
    try:
        dumps = sorted(progress.glob('native-fault-rank-*.log'))
    except OSError:
        dumps = []
    for path in dumps:
        dump = fault_text(path)
        if dump and 'Fatal Python error' in dump:
            frames = native_crash_frames(dump)
            kind = re.search(r'Fatal Python error: ([A-Za-z ]+)', dump)
            parts.append(f"rank {path.name.split('-')[3]} {kind.group(1).strip() if kind else 'native fault'} in " +
                         (' ← '.join(frames) if frames else 'a thread with no Python frame'))
            break
    try:
        repairs = sorted(progress.glob('runtime-repair-rank-*.json'))
    except OSError:
        repairs = []
    names = sorted({k for path in repairs for k in ((read(path) or {}).get('repairs') or {})})
    if names:
        parts.append('runtime repairs ' + ', '.join(names))
    if not parts:
        return None, checked
    host = attempt.get('host') or (str(attempt.get('worker')).rsplit('.', 1)[0] if attempt.get('worker') else None)
    gpus = attempt.get('gpus')
    where = (f"node {host}" if host else '') + (f" GPU {gpu_text(','.join(str(g) for g in gpus))}" if gpus else '')
    return '; '.join(parts) + (f"; {where}" if where else ''), checked


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
    horizon = task.get('horizon') if isinstance(task.get('horizon'), int) else total   # a budget extension may stop before its cap
    finished = (task.get('kind') == 'train' and manifest.get('status') in TERMINAL_RUN
                and isinstance(manifest.get('completed_updates'), int) and isinstance(horizon, int)
                and manifest['completed_updates'] >= horizon)
    snap, spec = task.get('snapshot_sha256'), task.get('evaluation_spec_hash')
    if task.get('kind') == 'evaluate' and isinstance(snap, str) and isinstance(spec, str):
        finished = (directory / 'evaluations' / f"{snap[:16]}__{spec[:12]}.json").is_file()
    at = f" at update {row['done']}/{total}" if isinstance(row.get('done'), int) and isinstance(total, int) else ''
    real = True
    native_cause, native_dump = native_story(ranks)
    if finished:
        cause = (f"NOT a training failure: the run is complete ({manifest.get('completed_updates')}/{total} updates)"
                 if task.get('kind') == 'train' else 'NOT an evaluation failure: its result file is complete') + \
                (f"; exit {rc} came after the work was done" if rc is not None else '')
        real = False
    elif native_cause:
        cause = native_cause     # where the rank crashed, from its own fault dump (user: 'exit -11 ... 고쳐볼래')
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
        found, checked = attempt_evidence(directory, attempt, lines)
        cause = (f"no exit code in the job record (older record); from its files: {found}" if found else
                 'unknown: the job record has no exit code or error, and none of these exist: ' + ', '.join(checked))
    else:
        found, _ = attempt_evidence(directory, attempt, lines)
        try:
            name = f" ({signal.Signals(-rc).name})" if rc < 0 else f" ({signal.Signals(rc - 128).name})" if 128 < rc < 160 else ''
        except ValueError:
            name = ''
        cause = f"exit {rc}{name} without an error message in its log or rank error files" + (f"; {found}" if found else '')
    lifetime = counted_failures(job)
    used = retry_budget_failures(job)
    auto = state.get('automatic_retry_policy') if isinstance(state.get('automatic_retry_policy'), dict) else {}
    limit = failure_limit(auto.get('max_failed_attempts'))   # the dispatcher's own rule (0/None -> its MAX_ATTEMPTS)
    delay = auto.get('retry_delay') if isinstance(auto.get('retry_delay'), (int, float)) else RETRY_DELAY
    policy = attempt.get('controller_policy')
    recovered = {h.get('controller_policy') for h in job.get('recovery_history') or [] if isinstance(h, dict)}
    # v11 (9c1e31c) retries failed tasks by default and records it in state.json; older workers follow the marker.
    retrying = bool(state.get('retry_failed_requested_at') or state.get('automatic_retry_policy'))
    ended = attempt.get('ended_at') if isinstance(attempt.get('ended_at'), (int, float)) else None
    native_fault=fatal_native_failure(job)
    eval_recovery=evaluation_recovery_attempts(job)
    if native_fault and eval_recovery is not None and not native_failure_blocks(job):
        nxt = ('evaluation recovery queued: one identical checkpoint/spec replay; '
               f'{used} of 1 replay failures used; no flag needed')
    elif native_fault and native_failure_blocks(job):
        number=native_fault['signal'];hosts=native_crash_hosts(job)
        nxt = (f"BLOCKED: {signal.Signals(number).name} (exit -{number}) on {len(hosts)} different nodes "
               f"({', '.join(h or '?' for h in hosts)}): likely the task itself, not a node; no more automatic retries; "
               'checkpoints and logs kept; repair required' if eval_recovery is None else
               f"BLOCKED: {signal.Signals(number).name} (exit -{number}) again in its one evaluation replay; "
               'no more automatic retries; repair required')
    elif native_fault and used < limit:
        hosts=native_crash_hosts(job)
        nxt = (f"retried automatically from its last checkpoint on a node other than {', '.join(h or '?' for h in hosts)} "
               f"(it crashed there): attempt {used + 1}/{limit} " +
               ('is waiting for a free node on another host' if row.get('status') == 'failed' else 'started'))
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
    if lifetime != used:
        nxt += (f'; evaluation replay failures {used}, lifetime failures {lifetime}' if eval_recovery is not None else
                f'; current verified repair failures {used}, historical failures {lifetime}')
    # A person is needed only when the dispatcher will not retry it by itself (user 11-10: 'needs attention 은 노드가
    # 날아갔는데도 왜 남아있는거냐. 현재 노드 정보나 뿌려').
    return dict(failure_cause=cause[:400], failure_next=nxt, failure_real=real, failure_at=at, failure_finished=finished,
                failure_budget_used=used,failure_lifetime_count=lifetime, failure_stack=native_dump,
                failure_needs_person=nxt.startswith('BLOCKED') and not finished)


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


# ---- Screen helpers -------------------------------------------------------------------------------

COHORT_LABEL = {'olmo_math': 'OLMo MATH', 'qwen_mbpp': 'Qwen MBPP', 'olmo_mbpp': 'OLMo MBPP', 'qwen_math': 'Qwen MATH'}
COHORT_SHORT = {}   # four experiments: 'OLMo'/'Qwen' alone would be ambiguous, so the node table shows the full name
COLOR = False   # set by main() only when stdout is a terminal (never in logs, pipes or tests)
STYLE = {'done': '34', 'ok': '32', 'run': '1;32', 'bad': '1;31', 'warn': '33', 'dim': '2', 'head': '1', 'info': '36'}


def paint(text, style):
    return f"\033[{STYLE[style]}m{text}\033[0m" if COLOR and style and text.strip() else text


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


def eval_context(directory, rows):
    """Per experiment: median wall time of finished evaluations, and n_questions per evaluation spec
    (from its finished evaluation records), so a running evaluation can say how far it is and how long is left."""
    times = sorted(r['ended'] - r['started'] for r in rows if r.get('kind') == 'evaluate' and r['status'] == 'done'
                   and all(isinstance(r.get(k), (int, float)) and not isinstance(r.get(k), bool) for k in ('started', 'ended'))
                   and r['ended'] > r['started'])
    median = (times[len(times) // 2] if len(times) % 2 else (times[len(times) // 2 - 1] + times[len(times) // 2]) / 2) if times else None
    questions = {}
    for r in rows:
        spec = r.get('eval_spec')
        if r['status'] != 'running' or not isinstance(spec, str) or spec in questions:
            continue
        questions[spec] = None
        for path in sorted((Path(directory) / 'evaluations').glob(f'*__{spec[:12]}.json'))[:5]:
            n = (read(path) or {}).get('n_questions')
            if isinstance(n, int) and n > 0:
                questions[spec] = n
                break
    return dict(median=median, questions=questions)


def short_progress(row, now, context=None, bare=False):
    """'update 62 of 75 · answering question 3 of 8, 812 tok (5s ago)' / 'answering question 75 of 500, 300 tok (5s ago)
    · ~17 min left'.
    The question total is the event's own, else another rank's, else (evaluation) the question count of a finished
    evaluation with the same spec split over the ranks (user 11-15 '고쳐놔 status 알수가 없어')."""
    events = row.get('_events') or []
    if not events:
        return None
    rank, stage, details, age = min(events, key=lambda e: e[3])
    text = STAGE_SHORT.get(stage, stage.replace('_', ' '))
    index, total = details.get('question_index'), details.get('questions_total')
    if not total:
        total = next((e[2].get('questions_total') for e in events if e[2].get('questions_total')), None)
    n = ((context or {}).get('questions') or {}).get(row.get('eval_spec')) if row.get('kind') == 'evaluate' else None
    if not total and n:
        total = -(-n // max(1, len(events)))      # questions per GPU process: ids[rank::world]
    if index:
        text += f" question {index} of {total}" if total else f" question {index} of ?"
    if details.get('response_tokens') is not None:
        text += f", {details['response_tokens']} tok"
    text = f"{text} ({age_text(age)} ago)"
    if bare:
        return text
    if row.get('kind') == 'evaluate':
        median = (context or {}).get('median')
        if median and index and total:
            text += f" · ~{age_text(median * max(0.0, 1 - min(1.0, index / total)))} left"
    elif isinstance(row.get('done'), int) and row.get('total'):
        text = f"update {row['done']} of {row['total']} · " + text
    return text


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


def node_table(bases, cohorts, now, this_host=None, models=None):
    """One row per worker process (worker id = '<host>.<pid>'), over both cohorts.

    A worker writes one heartbeat file per cohort it serves; while it scans, claims or runs a CPU stage
    of one cohort only that cohort's file stops, and during a GPU task only the task's cohort file moves.
    running  a task is assigned and (heartbeat fresh or its lease lock is held)
    busy     no task, heartbeat fresh in one cohort but silent in another that is still open (CPU stage /
             checking / claiming there)
    starting loading the runtime / checking the node        idle   heartbeat fresh, no task
    hung     task assigned, its lock still held but no heartbeat >10 min: no other node can take that task
    gone     no heartbeat >90 s and nothing it still holds (a task it ran with its lock free is resumed by
             the next free node, so it is shown as ready, not here)   finished  same after its queue completed"""
    this_host = this_host or socket.gethostname()
    expected = hashlib.sha256(Path(__file__).with_name('srgc_v8_nodes.py').read_bytes()).hexdigest()
    nodes, queues, info, runs_by_cohort, eval_ctx = {}, [], {}, {}, {}
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
            eval_ctx[cohort] = eval_context(Path(base) / cohort, rows)
            phase = state.get('phase')
            info[cohort] = dict(phase=phase, open=phase != 'complete' and not state.get('failure'), queue=queue,
                                ready=sum(1 for r in rows if (r['status'] == 'pending' and r.get('ready')) or r.get('lost')))
            model = (models or {}).get((str(base), cohort))
            if model:   # the screen's own count: ready + retrying (user 11-16 '일관되지가 않는다고')
                info[cohort]['ready'] = sum(t['state'] in ('ready', 'retrying') for t in model['tasks'])
            for w in worker_records(queue, now):
                e = entry(w['worker'], w['host'])
                e['beats'][cohort] = (w['age'], w['task'])
                if w.get('controller_sha256'):
                    e['controller'] = w['controller_sha256']
                e['complete'] = e['complete'] and phase == 'complete'
                diagnosis=worker_status(base,w['worker'])
                if diagnosis and diagnosis.get('controller_sha256')==w.get('controller_sha256'):
                    e['diagnosis']=diagnosis
            for r in rows:
                if r['status'] in ('running', 'unconfirmed') and r.get('worker') and not r.get('lost'):
                    e = entry(r['worker'], r.get('host') or str(r['worker']).rsplit('.', 1)[0])
                    e['complete'] = e['complete'] and phase == 'complete'
                    e['jobs'].append((base, cohort, r))
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
        version = e.get('controller')
        base = dict(host=e['host'], pid=pid, gpus=gpus, beat=beat, experiment='', task='', updates='', took='',
                    code='current' if version == expected else f'different {version[:12]}' if version else 'unverified')
        if e['jobs']:
            for jbase, cohort, r in e['jobs']:
                task = task_name(r, runs_by_cohort.get(cohort, {}))
                record = ((models or {}).get((str(jbase), cohort)) or {}).get('by_key', {}).get(r['key']) or {}
                updates = f"update {r['done']} of {r['total']}" if r['done'] is not None and r['total'] else \
                          (f"update ? of {r['total']}" if r['total'] and r.get('kind') != 'evaluate' else '')
                took = span(now - r['started']) if isinstance(r.get('started'), (int, float)) else ''
                row = dict(base, experiment=COHORT_LABEL.get(cohort, cohort), task=task, updates=updates, took=took,
                           key=r['key'], cohort=cohort, base=str(jbase), progress=record.get('progress') or updates,
                           left=record.get('left') or '')
                if r['status'] == 'running':
                    hung = r.get('quiet') is not None and r['quiet'] > HUNG_SECONDS and r.get('lease') is True
                    if hung:
                        row.update(state='hung', doing=(f"process {pid} alive here but silent {age_text(r['quiet'])}: hung? kill {pid}, rerun"
                                                        if local_alive else
                                                        f"no heartbeat {age_text(r['quiet'])}, lock still held (hung, or crashed node's lock not released): kill it there, rerun"))
                    elif r.get('quiet') is not None:
                        row.update(state='running', doing=(short_progress(r, now, eval_ctx.get(cohort), bare=models is not None) or
                                   f"no heartbeat {age_text(r['quiet'])}" +
                                   (", task lock held (admission / model load / checkpoint)" if r.get('lease') is True else '')))
                    else:
                        row.update(state='running', doing=short_progress(r, now, eval_ctx.get(cohort), bare=models is not None) or
                                   (f"log written {age_text(r['activity_age'])} ago" if r.get('activity_age') is not None
                                    else 'started, no output yet'))
                else:
                    if local_alive:
                        doing = f"process {pid} still alive here but silent: hung? kill {pid}, rerun same command"
                    else:
                        doing = ('down or hung → restart it (same command)' if r.get('beat_age') is not None
                                 else 'never sent a heartbeat → restart it (same command)')
                    row.update(state='hung', doing=doing)
                out.append(row)
            continue
        live = fresh(e)
        if not live:
            out.append(dict(base, state='finished' if e['complete'] else 'gone',
                            doing='queue complete' if e['complete'] else 'stopped, crashed or removed; nothing waits for it'))
            continue
        diagnosis=e.get('diagnosis')
        if diagnosis and now-diagnosis['at']<=ALIVE_SECONDS:
            code=diagnosis['state']
            state=('gone' if code in ('BLOCKED','STOPPED') else 'finished' if code=='COMPLETE' else
                   'starting' if code=='STARTING' else 'busy' if code in ('CHECKING','CPU_STAGE','RECORDING_RESULT') else 'idle')
            # An idle node says in plain words why, per cohort it serves (user 11-16 '주석이라도 잘 나오게').
            idle_reason=diagnosis.get('idle_reason') if state=='idle' and isinstance(diagnosis.get('idle_reason'),str) else None
            out.append(dict(base,state=state,experiment=COHORT_LABEL.get(diagnosis.get('cohort'),''),
                            doing=idle_reason or scheduler_text(diagnosis,now)))
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
        doing = (f"idle although {ready} task{'s' if ready != 1 else ''} could start (GPU busy? see its console)"
                 if ready else 'waiting for work')
        if silent:
            doing += ' · not serving ' + ', '.join(COHORT_LABEL.get(c, c) for c in silent) + ' (gave up? see its console)'
        out.append(dict(base, state='idle', doing=doing))
    rank = {'hung': 0, 'running': 1, 'busy': 2, 'starting': 3, 'idle': 4, 'gone': 5, 'finished': 6}
    return sorted(out, key=lambda r: (rank[r['state']], natural(r['host']), natural(r['pid'])))

def code_version():
    """The bundle folder name (rlvr-v8-h100-vX.Y.Z) or the git commit this status file runs from."""
    root = Path(__file__).resolve().parents[1]
    m = re.search(r'v\d+\.\d+\.\d+', root.name)
    if m:
        return m.group(0)
    try:
        return subprocess.run(['git', '-c', 'safe.directory=*', '-C', str(root), 'rev-parse', '--short', 'HEAD'],
                              capture_output=True, text=True, timeout=5).stdout.strip() or 'unknown code'
    except (OSError, subprocess.SubprocessError):
        return 'unknown code'


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


# ---- Clean view: one task model -----------------------------------------------------------------
# Why the older screen contradicted itself (user 11-16 '표랑 아래 요약된거랑 하나도 안맞는다니까', '174>=59 이게 뭐야'):
# the top table, the seed grids, the summary line, the node list and 'cells to fill' each counted tasks their own
# way (planned ranges, existing jobs, grid cells, grid columns that never get a task), and a failed task that the
# dispatcher restarts by itself was drawn as a red dead end. Now task_model() reads the queue once into one record
# per task: every existing job plus each task the planner is certain to create (v8/experiments/switching/matrix.py
# rules, plans/*.json once sealed). Every number on the screen is a count over those records, and check_screen()
# reads the drawn cells, table rows and node rows back and prints a red line naming any count that disagrees.

STATES = ('done', 'running', 'ready', 'retrying', 'waiting', 'person')
MARK = {'done': '✓', 'running': '▶', 'ready': '○', 'retrying': '↻', 'waiting': '·', 'person': '✗'}
MARK_STATE = {v: k for k, v in MARK.items()}
PLAN_PHASE = {'train': 'train', 'extension': 'ext', 'primary_evaluation': 'eval', 'evaluation': 'eval'}



def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def median(values):
    values = sorted(v for v in values if number(v) and v > 0)
    if not values:
        return None
    n = len(values)
    return values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2


def extension_spent(directory, run_id, family):
    """GPU-seconds a budget extension has spent so far by the runner's own rule (runner._phase_cost: the ledger
    events of its surviving attempt lineage, components the policy family pays for), and the updates they cover.
    None when the ledger is missing, unreadable or has an unmeasured required event."""
    rdir = Path(directory) / 'runs' / str(run_id)
    try:
        here = Path(__file__).resolve().parent
        roots = ([os.environ['V8_RUNNER_ROOT']] if os.environ.get('V8_RUNNER_ROOT') else []) + [str(here.parent / 'v8' / 'experiments')]
        for r in roots:
            if r not in sys.path:
                sys.path.append(r)
        from switching.ledger import REQUIRED as FAMILY_COST, component_totals, lineage_events
        manifest = read(rdir / 'manifest.json') or {}
        attempts = [(a['attempt_id'], a['start_update']) for a in manifest.get('attempts') or []
                    if isinstance(a, dict) and a.get('start_update') is not None and a.get('attempt_id')]
        events = []
        for path in sorted((rdir / 'attempts').glob('*/events.jsonl')):
            for line in path.read_text().splitlines():
                if line.strip():
                    events.append(json.loads(line))
        if not events or family not in FAMILY_COST:
            return None
        totals, missing = component_totals(lineage_events(events, attempts))
        required = FAMILY_COST[family] - {'prefix', 'cache_build'}
        if any(m.get('component') in required for m in missing):
            return None
        return sum(v for k, v in totals.items() if k in required)
    except Exception:  # noqa: BLE001 - no readable ledger: the screen says 'budget ?'
        return None


def quantile(values, q):
    """Nearest-rank quantile of the positive numbers (q=0.75: the slower quarter starts here)."""
    values = sorted(v for v in values if number(v) and v > 0)
    return values[max(0, math.ceil(q * len(values)) - 1)] if values else None


def serving_groups(base, cohort, now, rows):
    """GPU groups (one worker process = one group) serving this experiment, from heartbeats: a worker heartbeats
    every open experiment it serves, so its share here is 1/(experiments it serves). While it runs a task only that
    experiment's file moves; the others it heartbeat within the last day still count as served. A running task whose
    worker is silent but holds its lock still occupies a group."""
    beats, open_ = {}, {}
    for queue in sorted(Path(base).glob('*/.multi-node')):
        c = queue.parent.name
        st = read(queue / 'state.json') or {}
        open_[c] = st.get('phase') != 'complete' and not st.get('failure')
        try:
            for w in worker_records(queue, now):
                beats.setdefault(w['worker'], {})[c] = (w['age'], w['task'])
        except Exception:  # noqa: BLE001 - an odd heartbeat never crashes the screen
            continue
    running_here = {r.get('worker') for r in rows if r['status'] == 'running' and r.get('worker')}
    share = 0.0
    for worker, by in beats.items():
        fresh = {c for c, (age, _) in by.items() if age <= ALIVE_SECONDS}
        busy = any(task for c, (age, task) in by.items() if c in fresh) or worker in running_here
        if not fresh and worker not in running_here:
            continue
        limit = 86400 if busy else GAVE_UP_SECONDS
        served = {c for c, (age, _) in by.items() if open_.get(c) and age <= limit} | \
                 ({cohort} if worker in running_here else set())
        if cohort in served:
            share += 1 / len(served)
    running = sum(1 for r in rows if r['status'] == 'running')
    return max(running, int(share + 0.5), 1 if share > 0 else 0)


def retry_limit(state):
    """Lifetime failures a task may have before no node restarts it (state.json policy of the newest worker)."""
    auto = state.get('automatic_retry_policy') if isinstance(state.get('automatic_retry_policy'), dict) else {}
    return failure_limit(auto.get('max_failed_attempts'))


def automatic_retry_ends(job, state):
    """srgc_v8_nodes.Pool.retry_limit_reached / native_failure_blocks without a node: True when no node will
    start this task again by itself."""
    if not isinstance(job, dict) or not isinstance(job.get('attempts'), list) or not isinstance(job.get('task'), dict):
        return False
    try:
        if native_failure_blocks(job):
            return True
        limit = 1 if granted_view(job) is None and evaluation_recovery_attempts(job) is not None else retry_limit(state)
        return retry_budget_failures(job) >= limit
    except Exception:  # noqa: BLE001 - an odd record is shown, never crashes the screen
        return False


def task_state(row, state):
    """(state, reason) of one existing job, by the rules the dispatcher uses to pick work."""
    status = row['status']
    if status == 'done' or row.get('failure_finished'):
        return 'done', None
    job = row.get('_job')
    if status == 'running':
        if row.get('quiet') is not None and row['quiet'] > HUNG_SECONDS and row.get('lease') is True:
            return 'person', (f"no heartbeat for {age_text(row['quiet'])} on {row.get('host') or '?'} but its task lock is "
                              "still held, so no other node can take it: stop that process, the next node resumes it")
        return 'running', None
    if status not in ('pending', 'failed', 'interrupted', 'stopped', 'unconfirmed'):
        return 'person', f"unknown job state {status!r} (.multi-node/jobs/{row['key']}.json)"
    if automatic_retry_ends(job, state):
        cause = re.sub(r'\s+', ' ', str(row.get('failure_cause') or row.get('error') or 'no error recorded'))
        m = re.search(r'(SIG[A-Z]+ \(exit -\d+\)).*?; crashed in ([^;]+)', cause)
        if m:
            cause = f"{m.group(1)} in {m.group(2)}"
        try:
            used = retry_budget_failures(job)
        except Exception:  # noqa: BLE001
            used = row.get('attempt')
        hosts = native_crash_hosts(job) if native_failure_blocks(job) else []
        where = f" on {len(hosts)} nodes ({', '.join(hosts)})" if hosts else ''
        return 'person', f"stopped after {used} failures{where}, no more automatic retries · last: {cause.replace(' without an error message in its log or rank error files', '')[:90]}"
    if status == 'pending':
        return ('ready' if row.get('ready') else 'waiting'), None
    if status == 'unconfirmed' and not row.get('lost'):
        return 'person', 'marked running with no worker recorded, but a process holds its task lock'
    if status == 'failed':
        try:
            used = retry_budget_failures(job)
        except Exception:  # noqa: BLE001
            used = row.get('attempt')
        cause = re.sub(r'\s+', ' ', str(row.get('failure_cause') or row.get('error') or 'no error recorded'))
        m = re.search(r'(SIG[A-Z]+ \(exit -\d+\)).*?; crashed in ([^;]+)', cause)
        short = (f"{m.group(1)} in " + ' ← '.join(m.group(2).split(' ← ')[:2]) if m else
                 cause.replace(' without an error message in its log or rank error files', ''))
        return 'retrying', f"failure {used}/{retry_limit(state)}, {short[:110]}"
    if row.get('lost'):
        return 'retrying', "its node stopped sending heartbeats"
    return 'retrying', 'interrupted' if status == 'interrupted' else 'stopped'


def plan_file(queue, name):
    plan = read(Path(queue) / 'plans' / f'{name}.json')
    return plan if isinstance(plan, dict) and isinstance(plan.get('tasks'), list) else None


def extension_end(task, horizon, cap):
    """Update a budget extension is planned to stop at: its remaining budget over the planner's measured cost of one
    update (matrix.build_stage: budget_cap, cost_offset, est_update_gpu_seconds), never past its hard cap."""
    b, o, e = (task.get(k) for k in ('budget_cap_gpu_seconds', 'cost_offset_gpu_seconds', 'est_update_gpu_seconds'))
    if type(horizon) is int and type(cap) is int and number(b) and number(o) and number(e) and e > 0:
        return max(horizon, min(cap, horizon + int(max(0.0, b - o) // e)))
    return cap if type(cap) is int else horizon


def train_span(task, phase, col, H, P, CAP, total=None):
    """(first update the task trains, update it ends at) for a training or extension task."""
    parent = task.get('parent') if isinstance(task.get('parent'), dict) else {}
    start = exit_code(parent.get('completed_update'))
    if start is None or start < 0:
        m = re.fullmatch(r'SR(\d+)', col)
        start = P if col == 'On' and type(P) is int else int(m.group(1)) if m else H if phase == 'ext' and type(H) is int else 0
    horizon = task.get('horizon') if type(task.get('horizon')) is int else total if type(total) is int else H
    if phase == 'ext':
        cap = task.get('hard_cap') if type(task.get('hard_cap')) is int else CAP
        end = extension_end(task, horizon if type(horizon) is int else start, cap)
    else:
        end = horizon
    return start, max(start, end) if type(end) is int else start


def eval_progress(row, context, world=None):
    """(questions reached over all GPU processes, questions in the evaluation) of a running evaluation: each rank
    reports question i of its own share; the total is the ranks' shares, else a finished evaluation's count."""
    events = row.get('_events') or []
    n = ((context or {}).get('questions') or {}).get(row.get('eval_spec'))
    index = sum(e[2]['question_index'] for e in events if type(e[2].get('question_index')) is int)
    shares = [e[2].get('questions_total') for e in events]
    total = None
    if events and all(type(x) is int and x > 0 for x in shares):
        total = sum(shares) if not (n and type(world) is int and len(events) < world) else n
    elif n:
        total = n
    elif type(world) is int and any(type(x) is int and x > 0 for x in shares):
        total = max(x for x in shares if type(x) is int) * world
    return (min(index, total), total) if total else (index, None)


def task_model(base, cohort, now):
    """Every task of one experiment, existing or certain to be created, with its state, place and work."""
    queue = Path(base) / cohort / '.multi-node'
    label = COHORT_LABEL.get(cohort, cohort)
    model = dict(base=str(base), cohort=cohort, label=label, started=queue.exists(), tasks=[], state={},
                 phase='not started', more=[], cols=[], seeds=[], spu=None, spe=None, u_tau={})
    if not queue.exists():
        return model
    state, rows = collect(base, cohort, now)
    cfg = runner_config(cohort) or {}
    tc = cfg.get('training') or {}
    H, P, CAP = (tc.get(k) for k in ('matched_update_endpoint', 'prefix_updates', 'hard_safety_update_cap'))
    world = (cfg.get('runtime') or {}).get('world_size')
    bank = cfg.get('bank') or {}
    grid = sorted({x for x in (bank.get('core_grid') or []) + (bank.get('extended_grid') or []) if type(x) is int})
    cfg_seeds = [s for s in ((cfg.get('seeds') or {}).get('training') or []) if type(s) is int]
    runs = run_branches(rows)
    ctx = eval_context(Path(base) / cohort, rows)
    tasks = model['tasks']

    def add(**fields):
        record = dict(planned=False, row=None, why=None, update=None, of_ext=False, start=None, end=None,
                      done_u=None, frac=0.0, progress=None, left=None)
        record.update(fields)
        tasks.append(record)
        return record

    def phase_of(job_phase, kind):
        return 'eval' if kind == 'evaluate' or job_phase in ('evaluation', 'primary_evaluation') else \
               'ext' if job_phase == 'extension' else 'train'

    for r in rows:
        job = r.get('_job') if isinstance(r.get('_job'), dict) else {}
        task = job.get('task') if isinstance(job.get('task'), dict) else {}
        phase = phase_of(r.get('phase'), r.get('kind'))
        st, why = task_state(r, state)
        if phase == 'eval':
            col, of_ext = eval_branch(r, runs)
            i, n = eval_progress(r, ctx, world) if r['status'] == 'running' else (0, None)
            add(key=r['key'], phase='eval', seed=r['seed'], col=col, state=st, why=why, row=r, update=r.get('update'),
                of_ext=of_ext, frac=1.0 if st == 'done' else min(1.0, i / n) if n else 0.0, questions=(i, n))
            continue
        col = branch_column(r['arm'])
        start, end = train_span(task, phase, col, H, P, CAP, r.get('total'))
        manifest = r.get('_manifest') or {}
        if st == 'done':
            done_u = end
            if phase == 'ext' and type(manifest.get('completed_updates')) is int:   # a budget run stops at its budget
                end = done_u = max(start, manifest['completed_updates'])
        else:
            done_u = min(max(r['done'], start), max(end, r['done'])) if type(r.get('done')) is int else start
            end = max(end, done_u)   # an extension past the planner's estimate is re-predicted from its measured speed below
        rec = add(key=r['key'], phase=phase, seed=r['seed'], col=col, state=st, why=why, row=r, start=start, end=end,
                  done_u=done_u, shown_u=r['done'] if type(r.get('done')) is int else None, task=task)
        if phase == 'train' and col == 'U':
            model['u_tau'][r['seed']] = start

    # Sealed plans: every task in them is (or becomes at the next scan) a job.
    keys = {t['key'] for t in tasks}
    sealed = set()
    for name in ('train', 'extension', 'primary_evaluation', 'evaluation'):
        plan = plan_file(queue, name)
        if plan is None:
            continue
        if name != 'primary_evaluation':
            sealed.add(PLAN_PHASE[name])
        for task in plan['tasks']:
            try:
                key = task_key(task)
            except Exception:  # noqa: BLE001
                continue
            if key in keys or not isinstance(task, dict):
                continue
            keys.add(key)
            phase = PLAN_PHASE[name]
            ready = task_ready(task)
            if phase == 'eval':
                col, of_ext = eval_branch(dict(run_id=task.get('run_id'), arm='evaluate'), runs)
                add(key=key, phase='eval', seed=task.get('training_seed', '-'), col=col, planned=True,
                    state='ready' if ready else 'waiting', update=task.get('completed_update'), of_ext=of_ext)
            else:
                col = branch_column(task.get('method_id') or '-')
                start, end = train_span(task, phase, col, H, P, CAP)
                add(key=key, phase=phase, seed=task.get('training_seed', '-'), col=col, planned=True,
                    state='ready' if ready else 'waiting', start=start, end=end, done_u=start, task=task)

    # Tasks the planner is certain to create (matrix.build_stage): per seed the On trajectory and one SR branch per
    # grid point (core + extended); one budget extension per finished SR/U branch; one evaluation at the matched
    # endpoint per training run. The Uniform destination exists only if the seed's On run triggers the switch rule,
    # and budget-endpoint evaluations only once the budget evaluation plan is made: both are named, not counted.
    job_seeds = {t['seed'] for t in tasks if type(t['seed']) is int}
    use_plan = bool(cfg_seeds and grid and type(H) is int and type(P) is int and job_seeds <= set(cfg_seeds))
    open_u = []
    if use_plan:
        if 'train' not in sealed:
            have = {(t['seed'], t['col']) for t in tasks if t['phase'] == 'train'}
            for s in cfg_seeds:
                for col, start in [('On', P)] + [(f'SR{tau}', tau) for tau in grid]:
                    if (s, col) not in have:
                        add(key=f'planned.train.s{s}.{col}', phase='train', seed=s, col=col, planned=True,
                            state='waiting', start=start, end=H, done_u=start)
            on_done = {t['seed'] for t in tasks if t['phase'] == 'train' and t['col'] == 'On' and t['state'] == 'done'}
            with_u = {t['seed'] for t in tasks if t['phase'] == 'train' and t['col'] == 'U'}
            open_u = [s for s in cfg_seeds if s not in on_done and s not in with_u]
        branches = [t for t in tasks if t['phase'] == 'train' and (t['col'] == 'U' or re.fullmatch(r'SR\d+', t['col']))]
        if 'ext' not in sealed:
            have = {(t['seed'], t['col']) for t in tasks if t['phase'] == 'ext'}
            spans = [t['end'] - t['start'] for t in tasks if t['phase'] == 'ext' and not t['planned'] and t['end'] > t['start']]
            span_u = median(spans) or ((CAP - H) if type(CAP) is int else 0)
            for b in branches:
                if (b['seed'], b['col']) not in have:
                    add(key=f"planned.ext.s{b['seed']}.{b['col']}", phase='ext', seed=b['seed'], col=b['col'], planned=True,
                        state='waiting', start=H, end=H + int(round(span_u)), done_u=H)
        if 'eval' not in sealed:
            have = {(t['seed'], t['col']) for t in tasks if t['phase'] == 'eval' and not t['of_ext'] and t['update'] == H}
            for b in [t for t in tasks if t['phase'] == 'train']:
                if (b['seed'], b['col']) not in have:
                    have.add((b['seed'], b['col']))
                    add(key=f"planned.eval.s{b['seed']}.{b['col']}@{H}", phase='eval', seed=b['seed'], col=b['col'],
                        planned=True, state='waiting', update=H)
        if open_u:
            later = [w for p, w in (('train', 'training'), ('ext', 'extension'), ('eval', 'evaluation')) if p not in sealed]
            later = ', '.join(later[:-1]) + ' and ' + later[-1] if len(later) > 1 else ''.join(later)
            model['more'].append(f"up to {len(open_u)} more U branches ({later} each) for "
                                 f"{', '.join(f's{s}' for s in open_u)}: decided when that seed's On run finishes")
        if 'eval' not in sealed:
            model['more'].append("budget-end evaluations are counted once the extensions finish and their plan is made "
                                 "(the time left already includes them)")
    elif sealed != {'train', 'ext', 'eval'}:
        why = ('the runner config cannot be read' if not cfg else "this queue's seeds differ from the config")
        model['more'].append(f"{why}, so tasks not created yet cannot be counted: totals are the tasks created so far")

    # Measured wall time of one update (On trajectory and SR/U branches separately: the On run also refreshes and
    # checks) and of one evaluation: medians over finished tasks (wall time of the last attempt / updates that
    # attempt trained), else over running tasks with enough progress. The 75th percentile feeds the upper end.
    def attempt_start(t):
        attempts = (t['row'].get('_manifest') or {}).get('attempts')
        first = attempts[-1].get('start_update') if isinstance(attempts, list) and attempts and isinstance(attempts[-1], dict) else None
        return first if type(first) is int and first >= 0 else t['start']
    def took(t, end=None):
        a, b = t['row'].get('started'), now if end is None else t['row'].get('ended')
        return b - a if number(a) and number(b) and b > a else None
    measured = {'on': ([], []), 'branch': ([], []), 'eval': ([], [])}
    for t in tasks:
        if t['row'] is None:
            continue
        if t['phase'] == 'eval':
            if t['state'] == 'done' and took(t, 'end'):
                measured['eval'][0].append(took(t, 'end'))
            elif t['state'] == 'running' and t['frac'] >= 0.2 and took(t):
                measured['eval'][1].append(took(t) / t['frac'])
            continue
        n = (t['done_u'] or 0) - attempt_start(t)
        kind = 'on' if t['col'] == 'On' else 'branch'
        if t['state'] == 'done' and n > 0 and took(t, 'end'):
            t['rate'] = took(t, 'end') / n
            measured[kind][0].append(t['rate'])
        elif t['state'] == 'running' and n >= 3 and took(t):
            t['rate'] = took(t) / n
            measured[kind][1].append(t['rate'])
    speed, notes = {}, []
    for kind, (done_v, run_v) in measured.items():
        values = done_v or run_v
        speed[kind] = (median(values), quantile(values, 0.75)) if values else None
    if speed['on'] is None and speed['branch'] is not None:
        speed['on'] = speed['branch']
        notes.append('On update time not measured yet: SR update time used')
    elif speed['branch'] is None and speed['on'] is not None:
        speed['branch'] = speed['on']
        notes.append('SR update time not measured yet: On update time used')
    elif speed['on'] is None:
        est = [t['task']['est_update_gpu_seconds'] / world for t in tasks
               if isinstance(t.get('task'), dict) and number(t['task'].get('est_update_gpu_seconds'))
               and type(world) is int and world > 0]
        if est:
            speed['on'] = speed['branch'] = (median(est), quantile(est, 0.75))
            notes.append('update time not measured yet: planner estimate (GPU-seconds per update / GPUs)')
    spu = median(measured['branch'][0] + measured['on'][0]) or median(measured['branch'][1] + measured['on'][1]) \
        if speed['branch'] else None
    spe = speed['eval'][0] if speed['eval'] else None
    model.update(spu=spu, spe=spe, speed=speed, notes=notes)

    # A budget extension stops when its next update no longer fits the budget (budget.should_continue). The planner
    # sized one update by its parent branch's own cost; this extension's measured wall time per update against its
    # parent's scales that cost, so a run cheaper than planned is predicted past the planned endpoint (never the
    # hard cap just because it passed the estimate).
    by_run = {t['row'].get('run_id'): t for t in tasks if t['row'] is not None and t['phase'] == 'train'}
    for t in tasks:
        if t['phase'] != 'ext' or t['state'] == 'done' or not isinstance(t.get('task'), dict):
            continue
        task = t['task']
        cap = task.get('hard_cap') if type(task.get('hard_cap')) is int else CAP
        horizon = task.get('horizon') if type(task.get('horizon')) is int else H
        e = task.get('est_update_gpu_seconds')
        b, o = task.get('budget_cap_gpu_seconds'), task.get('cost_offset_gpu_seconds')
        spent = extension_spent(Path(base) / cohort, task.get('run_id'), (task.get('method') or {}).get('family')) \
            if t['row'] is not None else None
        t['budget'] = (o + spent) / b if spent is not None and number(b) and b > 0 and number(o) else None
        parent = (task.get('parent') or {}).get('run_id') or (task.get('method') or {}).get('extension_of')
        ref = (by_run.get(parent) or {}).get('rate') or (speed['branch'] or (None,))[0]
        if t['budget'] is not None and type(horizon) is int and type(cap) is int:
            # budget.should_continue: the next update must still fit; its cost is this run's own so far
            n = t['done_u'] - horizon
            per = spent / n if n >= 1 and spent > 0 else e
            end = min(cap, t['done_u'] + int(max(0.0, b - o - spent) // per)) if number(per) and per > 0 else cap
        else:
            if number(e) and t.get('rate') and ref:
                task = dict(task, est_update_gpu_seconds=e * t['rate'] / ref)
            end = extension_end(task, horizon if type(horizon) is int else t['start'], cap)
        if type(end) is int and t['done_u'] >= end and type(cap) is int:
            end = min(cap, t['done_u'] + 1)   # still running past it: at least one more update
        if type(end) is int:
            t['end'] = max(t['start'], t['done_u'], end)
    spans = [t['end'] - t['start'] for t in tasks if t['phase'] == 'ext' and not str(t['key']).startswith('planned.')
             and t['end'] > t['start']]
    model['ext_span'] = int(round(median(spans))) if spans else ((CAP - H) if type(CAP) is int and type(H) is int else 0)
    for t in tasks:
        if str(t['key']).startswith('planned.ext.') and type(H) is int:
            t['end'] = H + int(round(model['ext_span']))

    for t in tasks:
        if t['phase'] == 'eval':
            left_units = 0.0 if t['state'] == 'done' else 1.0 - t['frac']
            t['left_s'] = left_units * spe if spe else None
            if t['state'] in ('running', 'person') and t['row'] is not None and t['row']['status'] == 'running':
                i, n = t.get('questions') or (0, None)
                t["progress"] = f"question {i} of {n if n else '?'}"   # user 11-16 '87/300 몇 시간': units in words
        else:
            left_units = 0 if t['state'] == 'done' else max(0, t['end'] - t['done_u'])
            sp = speed['on' if t['col'] == 'On' else 'branch']
            t['left_s'] = left_units * sp[0] if sp else None
            if t['state'] == 'running' and sp and took(t) and t['done_u'] < t['end']:
                # the part of the current update already run (never more than one update ahead of the record)
                guess = attempt_start(t) + took(t) / sp[0]
                if guess > t['done_u']:
                    t['done_est'] = min(t['done_u'] + 1, guess, t['end'])
            if t['row'] is not None and t['row']['status'] == 'running':
                shown = t['shown_u'] if t.get('shown_u') is not None else '?'
                if t['phase'] == 'ext':
                    # the planner's stop update is an estimate shared by most branches (user 11-16 '300 이거 의미가
                    # 없잖아'); the run stops when its budget is spent, so show how much of it is spent
                    cap = (t.get('task') or {}).get('hard_cap') if type((t.get('task') or {}).get('hard_cap')) is int else CAP
                    t['progress'] = f"update {shown} · budget " + (f"{int(100 * t['budget'])}%" if t.get('budget') is not None else '?')
                    if type(cap) is int and type(t.get('shown_u')) is int and t['end'] >= cap and cap - t['shown_u'] <= 25:
                        t['progress'] += f" · cap {cap}"
                    if t.get('budget') is None:
                        # without the spent budget the stop point is unknown, so is the time left: the planner's shared
                        # estimate showed '<1m left' on runs that went on for hours (user 11-16 '예상 시간 잘못')
                        t['left_s'] = None
                else:
                    t['progress'] = f"update {shown} of {t['end']}"
        t['left_units'] = left_units
        if t['row'] is not None and t['row']['status'] == 'running':
            t['left'] = ('?' if t['left_s'] is None else '<1m left' if t['left_s'] < 60 else f"~{span(t['left_s'])} left")

    cols = []
    for t in tasks:
        if t['col'] not in cols:
            cols.append(t['col'])
    def col_key(c):
        kind = 0 if c == 'On' else 1 if c.startswith('SR') else 2 if c == 'U' else 3
        return (kind, int(re.sub(r'\D', '', c) or 0), c)
    model['cols'] = sorted(cols, key=col_key)
    model['seeds'] = sorted({t['seed'] for t in tasks}, key=lambda x: (not isinstance(x, int), x if isinstance(x, int) else str(x)))
    model.update(H=H, P=P, CAP=CAP, sealed=sealed, open_u=open_u, use_plan=use_plan,
                 frozen=(Path(base) / cohort / 'frozen' / 'budget.json').exists() or
                 bool((state.get('ready_side_work') or {}).get('budget_ready')),
                 groups=serving_groups(base, cohort, now, rows))
    model['state'] = state
    model['phase'] = PHASE_LABEL.get(state.get('phase'), state.get('phase', 'not started'))
    model['runs'] = runs
    model['by_key'] = {t['key']: t for t in tasks}
    try:
        model['eta'] = model_eta(model)
    except Exception as exc:  # noqa: BLE001 - the time estimate never takes the screen down
        model['eta'], model['eta_error'] = None, f'{type(exc).__name__}: {exc}'
    return model


def model_counts(model):
    """Counts per phase and per state, the one source of every number in the table."""
    out = {ph: {'done': 0, 'total': 0} for ph in ('train', 'ext', 'eval')}
    states = {s: 0 for s in STATES}
    for t in model['tasks']:
        out[t['phase']]['total'] += 1
        out[t['phase']]['done'] += t['state'] == 'done'
        states[t['state']] += 1
    return out, states


def eta_items(model, scenario):
    """Every piece of work still to come in one experiment, as {id: (GPU-group seconds, dependencies, running)},
    by the planner's rules (matrix.build_stage, ready_side_tasks, Runtime.plan('evaluation')):
      train  each training run's remaining updates (On trajectory, one SR branch per grid point, U once known);
             a branch starts when the On trajectory reaches its switch point, U when the On run is complete
      ext    one budget extension per SR/U branch, after that branch and the budget freeze (all reference endpoints)
      eval   the matched-update evaluation of every training run, then budget-endpoint evaluations: the On
             trajectory's once the budget is frozen, each branch's after its extension
    scenario 'low' = median times and the expected work; 'high' = 75th-percentile times, every possible U branch
    (at the longest switch point), an extension not created yet runs to its hard cap, and as many budget endpoints
    per branch as any finished branch needed (at least 2). Returns None when a needed time is unmeasured."""
    H, P, CAP, sp = model.get('H'), model.get('P'), model.get('CAP'), model.get('speed') or {}
    k = 0 if scenario == 'low' else 1
    tasks = model['tasks']
    if not any(t['state'] != 'done' for t in tasks):
        return {}
    def su(col):
        v = sp.get('on' if col == 'On' else 'branch')
        return v[k] if v else None
    se = sp['eval'][k] if sp.get('eval') else None
    items = {}
    def item(iid, units, per, deps=(), running=False):
        if per is None:
            raise LookupError(iid)
        items[iid] = dict(dur=max(0.0, units * per), deps=[d for d in deps if d[0] in items or d[0] in pending], running=running)
    train = {(t['seed'], t['col']): t for t in tasks if t['phase'] == 'train'}
    ext = {(t['seed'], t['col']): t for t in tasks if t['phase'] == 'ext'}
    ids = lambda ph, s, c: (ph, s, c)
    pending = {ids(t['phase'], t['seed'], t['col']) for t in tasks if t['phase'] != 'eval' and t['state'] != 'done'}
    possible_u = [s for s in model.get('open_u') or []] if scenario == 'high' else []
    pending |= {ids(ph, s, 'U?') for s in possible_u for ph in ('train', 'ext')}
    try:
        for (s, c), t in train.items():
            if t['state'] == 'done':
                continue
            on, deps = train.get((s, 'On')), []
            if c != 'On' and t['state'] == 'waiting' and on is not None and on['state'] != 'done':
                if c == 'U':
                    deps.append((ids('train', s, 'On'), None))
                elif su('On') is not None:
                    deps.append((ids('train', s, 'On'), max(0, t['start'] - on['done_u']) * su('On')))
            item(ids('train', s, c), t['end'] - t.get('done_est', t['done_u']), su(c), deps, t['state'] == 'running')
        for s in possible_u:
            if type(H) is int and type(P) is int:
                pending.add(ids('train', s, 'U?'))
                item(ids('train', s, 'U?'), H - P, su('U'), [(ids('train', s, 'On'), None)])
        gate = []
        if not model.get('frozen') and not any(t['phase'] == 'ext' and not t['planned'] for t in tasks):
            u_tau = model.get('u_tau') or {}
            for (s, c), t in train.items():
                ref = c == 'On' or (s in u_tau and c == f'SR{u_tau[s]}') or \
                      (scenario == 'high' and train.get((s, 'On'), {}).get('state') != 'done' and c != 'U')
                if ref and t['state'] != 'done':
                    gate.append((ids('train', s, c), None))
            items[('gate',)] = dict(dur=0.0, deps=gate, running=False)
            gate = [(('gate',), None)]
        branches = [(s, c) for (s, c) in train if c != 'On'] + [(s, 'U?') for s in possible_u]
        for s, c in branches:
            e = ext.get((s, c))
            if e is not None and e['state'] == 'done':
                continue
            before = [(ids('train', s, c), None)] + gate
            if e is None and 'ext' in model.get('sealed', ()):
                continue   # the sealed extension plan has none for this branch
            if e is None or (scenario == 'high' and str(e['key']).startswith('planned.ext.')):
                if type(CAP) is int and type(H) is int:
                    pending.add(ids('ext', s, c))
                    item(ids('ext', s, c), CAP - H if scenario == 'high' else model.get('ext_span') or CAP - H, su(c), before)
                continue
            item(ids('ext', s, c), e['end'] - e.get('done_est', e['done_u']), su(c), before, e['state'] == 'running')
        budget_evals, ext_done = {}, {key for key, e in ext.items() if e['state'] == 'done'}
        for t in (t for t in tasks if t['phase'] == 'eval'):
            budget = t['of_ext'] or (t['update'] is not None and t['update'] != H)
            if budget:
                budget_evals[(t['seed'], t['col'])] = budget_evals.get((t['seed'], t['col']), 0) + 1
            if t['state'] == 'done':
                continue
            dep = (ids('ext', t['seed'], t['col']) if t['col'] != 'On' else ('gate',)) if budget else ids('train', t['seed'], t['col'])
            item(('eval', t['key']), 1.0 - t['frac'], se, [(dep, None)], t['state'] == 'running')
        for s in possible_u:
            item(('eval', 'U?', s), 1, se, [(ids('train', s, 'U?'), None)])
        if 'eval' not in model.get('sealed', ()) and model.get('use_plan', True):
            most = max([v for key, v in budget_evals.items() if key in ext_done] + [2 if scenario == 'high' else 1])
            for s, c in [(s, 'On') for (s, c) in train if c == 'On'] + branches:
                need = (1 if c == 'On' else most) - budget_evals.get((s, c), 0)
                if c != 'On' and (s, c) in ext_done and budget_evals.get((s, c)):
                    need = 0
                for i in range(max(0, need)):
                    item(('budget-eval', s, c, i), 1, se, [(('gate',) if c == 'On' else ids('ext', s, c), None)])
    except LookupError:
        return None
    return items


def eta_paths(items):
    """(earliest start, finish) of every item with unlimited GPU groups: a dependency (id, None) waits for that
    item to finish, (id, lag) for lag seconds after it started (a branch whose switch point the On run reaches)."""
    start, finish = {}, {}
    def visit(i):
        if i in finish:
            return finish[i]
        finish[i] = 0.0   # cycle guard (the planner's graph has none)
        it = items[i]
        t0 = 0.0
        for d, lag in it['deps']:
            if d in items:
                visit(d)
                t0 = max(t0, finish[d] if lag is None else start[d] + lag)
        start[i], finish[i] = t0, t0 + it['dur']
        return finish[i]
    for i in items:
        visit(i)
    return start, finish


def eta_schedule(items, groups):
    """Makespan when `groups` GPU groups always take the waiting item on the longest remaining chain first
    (running items keep their group)."""
    succ = {i: [] for i in items}
    for i, it in items.items():
        for d, lag in it['deps']:
            if d in items:
                succ[d].append((i, lag))
    level = {}
    def bottom(i):
        if i not in level:
            level[i] = max([items[i]['dur'] + bottom(j) for j, lag in succ[i] if lag is None] +
                           [lag + bottom(j) for j, lag in succ[i] if lag is not None] + [items[i]['dur']])
        return level[i]
    for i in items:
        bottom(i)
    t, started, done, busy = 0.0, {}, set(), []
    for i, it in items.items():
        if it['running']:
            started[i] = 0.0
            busy.append((it['dur'], i))
    groups = max(groups, len(busy))
    waiting = sorted((i for i in items if i not in started), key=lambda i: (-level[i], str(i)))
    def ready_at(i):
        at = 0.0
        for d, lag in items[i]['deps']:
            if d not in items:
                continue
            if lag is None:
                if d not in done:
                    return None
                at = max(at, started[d] + items[d]['dur'])
            elif d not in started:
                return None
            else:
                at = max(at, started[d] + lag)
        return at
    while waiting or busy:
        free = groups - len(busy)
        for i in [i for i in waiting if (lambda r: r is not None and r <= t + 1e-9)(ready_at(i))][:max(0, free)]:
            started[i] = t
            busy.append((t + items[i]['dur'], i))
            waiting.remove(i)
        later = [r for r in map(ready_at, waiting) if r is not None and r > t + 1e-9]
        if not busy and not later:
            break   # nothing can ever start (a dependency outside the plan)
        t = min([e for e, _ in busy] + later)
        for e, i in list(busy):
            if e <= t + 1e-9:
                busy.remove((e, i))
                done.add(i)
    return t


def model_eta(model):
    """Time left of one experiment as a range, from all work still to come (eta_items) and the GPU groups serving
    it: low = the longest-chain-first schedule of the expected work at median times (never below total work /
    groups nor the longest chain); high = total work / groups + longest chain of the 'high' work, which bounds any
    dispatcher that never leaves a group idle while a task can start, so it cannot grow while tasks run as measured.
    None when a needed time is unmeasured; 'groups' 0 when no live worker serves it."""
    out = {}
    for scenario in ('low', 'high'):
        items = eta_items(model, scenario)
        if items is None:
            return None
        w = sum(it['dur'] for it in items.values())
        _, finish = eta_paths(items)
        chain = max(finish.values(), default=0.0)
        out[scenario] = dict(items=items, work=w, chain=chain)
    g = model.get('groups') or 0
    out['groups'] = g
    if not g:
        return out
    low = out['low']
    low['left'] = max(eta_schedule(low['items'], g), low['work'] / g, low['chain'])
    out['high']['left'] = max(low['left'], out['high']['work'] / g + out['high']['chain'])
    return out


def kst(t):
    return time.strftime('%m-%d %H:%M', time.gmtime(t + 9 * 3600))


def model_progress(model):
    """(percent of the work done in measured time, time left text): None/'?' when a needed time is unmeasured."""
    tasks = model['tasks']
    if not tasks:
        return None, ''
    if not any(t['state'] != 'done' for t in tasks):
        return 100, 'done'
    eta = model.get('eta')
    if eta is None:
        return None, '?'
    sp = {k: (v or (0.0, 0.0))[0] for k, v in model['speed'].items()}
    done = sum((t['done_u'] - t['start']) * sp['on' if t['col'] == 'On' else 'branch']
               for t in tasks if t['phase'] != 'eval' and t['end'] > t['start'])
    done += sum(t['frac'] for t in tasks if t['phase'] == 'eval') * sp['eval']
    work = done + eta['low']['work']
    pct = min(99, int(100 * done / work)) if work else 0
    if 'left' not in eta['low']:
        return pct, '?'
    lo, hi = eta['low']['left'], eta['high']['left']
    return pct, '~' + span(lo) + ('' if span(lo) == span(hi) else '–' + span(hi))


# ---- The screen, laid out like `run_srgc.sh <dataset> status` (srgc_live_status.render) ------------------------
# User 11-16 '아직도 저 메커니즘 이해를 못했어 다시 만들어 그냥 status 전부다 직관적으로 처음부터. 비슷하게':
# header, this node, then (1) time left, (2) one block per experiment with plain counts per stage, (3) every task that
# is not finished, grouped by what happens next, (4) the seed x branch grid, (5) live nodes. Every number is a count
# over task_model() records, and check_screen() reads sections 2-5 back and prints a red line if any disagree.

STAGE_WORD = {'train': 'training', 'ext': 'extension', 'eval': 'evaluation'}
STAGE_UNIT = {'train': 'training runs', 'ext': 'extension runs', 'eval': 'evaluations'}
STATE_WORDS = (('running', 'running'), ('ready', 'can start'), ('retrying', 'restart by themselves'),
               ('waiting', 'waiting'), ('person', 'STOPPED'))
CELL_LEGEND = ("cells: ✓ done (extension 312✓ = stopped at update 312) · 187▶ running at update 187 (extension 81%▶ = "
               "budget used, eval 36%▶ = questions answered) · ready = can start now · wait = needs an earlier step · "
               "retry = restarts by itself · FAIL = stopped, needs you · - = no such run")
RETRY_COMMAND = 'sh scripts/run_v8_nodes.sh --retry-failed'


def is_budget_eval(t, H):
    return t['phase'] == 'eval' and (t['of_ext'] or (t.get('update') is not None and t['update'] != H))


def tname(t, model):
    """'s5 SR100 training', 's5 SR100 extension', 's5 SR100 eval at 275', 's5 SR100 budget eval at 350'."""
    head = f"s{t['seed']} {t['col']}"
    if t['phase'] != 'eval':
        return f"{head} {STAGE_WORD[t['phase']]}"
    if is_budget_eval(t, model.get('H')):
        return f"{head} budget eval" + (f" at {t['update']}" if t.get('update') is not None else '')
    return f"{head} eval at {t['update'] if t.get('update') is not None else model.get('H') or '?'}"


def progress_words(t):
    """'update 187 of 275', 'update 287, budget 81%', 'question 182 of 500'."""
    p = (t.get('progress') or '').replace(' · ', ', ').replace('budget ?', 'budget not read')
    return p


def left_words(t):
    if t.get('left_s') is None:
        return 'time left unknown'
    return '<1m left' if t['left_s'] < 60 else f"~{span(t['left_s'])} left"


def retry_words(why):
    m = re.match(r'failure (\d+)/(\d+), (.*)', why or '')
    if m:
        return f"failed {m.group(1)} of {m.group(2)} so far ({m.group(3)})"
    return {'interrupted': 'was interrupted', 'stopped': 'was stopped'}.get(why, why or 'was stopped')


def person_words(why):
    m = re.match(r'stopped after (\d+) failures(.*?), no more automatic retries · last: (.*)', why or '')
    if m:
        return f"failed {m.group(1)} times{m.group(2)}, no more automatic retries · last error: {m.group(3)}"
    return why or 'needs a look'


def task_index(model):
    if '_index' not in model:
        model['_index'] = {(t['phase'], t['seed'], t['col']): t for t in model['tasks'] if t['phase'] != 'eval'}
    return model['_index']


def blocker(t, model):
    """(blocking task record or None, words) for a waiting task, by the planner's creation rules (matrix.build_stage,
    ready_side_tasks): a branch starts when its seed's On run reaches its switch point, U once that run is complete,
    an extension after its branch and the budget freeze, a matched evaluation after its training run, a budget-end
    evaluation after its extension (On: after the budget freeze)."""
    idx, s, c = task_index(model), t['seed'], t['col']
    on = idx.get(('train', s, 'On'))
    def freeze():
        if model.get('frozen'):
            return None
        refs = [x for x in model['tasks'] if x['phase'] == 'train' and x['state'] != 'done' and
                (x['col'] == 'On' or x['col'] == f"SR{(model.get('u_tau') or {}).get(x['seed'])}")]
        names = ', '.join(tname(x, model) for x in refs[:3]) + (f", … {len(refs) - 3} more" if len(refs) > 3 else '')
        return (None, 'the budget freeze' + (f", which needs {names} to finish" if refs else ' (made at the next queue scan)'))
    if t['phase'] == 'train':
        if c != 'On' and on is not None and on['state'] != 'done':
            m = re.fullmatch(r'SR(\d+)', c)
            if c == 'U':
                return (on, None)
            if m and (on.get('done_u') or 0) < int(m.group(1)):
                return (on, f"to reach update {m.group(1)}")
        return (None, 'their start checkpoint file (not on disk yet)')
    if t['phase'] == 'ext':
        b = idx.get(('train', s, c))
        if b is not None and b['state'] != 'done':
            return (b, None)
        return freeze() or (None, 'their start checkpoint file (not on disk yet)')
    if is_budget_eval(t, model.get('H')):
        if c == 'On':
            return freeze() or (None, 'their checkpoint file (not on disk yet)')
        e = idx.get(('ext', s, c))
        if e is not None and e['state'] != 'done':
            return (e, None)
        return (None, 'their checkpoint file (not on disk yet)')
    b = idx.get(('train', s, c))
    if b is not None and b['state'] != 'done':
        return (b, None)
    return (None, 'their checkpoint file (not on disk yet)')


def state_phrase(t):
    """What a blocking task is doing, in words."""
    st = t['state']
    if st == 'running':
        return f"running, {left_words(t)}"
    if st == 'ready':
        return 'can start now'
    if st == 'retrying':
        return 'will restart by itself'
    if st == 'person':
        return 'STOPPED, needs you: these never start until it is fixed'
    if st == 'waiting':
        return 'itself waiting'
    return 'done; picked up at the next queue scan'


def time_left(model, now):
    """(short 'about 13h04m–44h21m', sentence with the KST clock, low s, high s); low/high None when unknown."""
    tasks = model['tasks']
    if not model['started']:
        return 'not started', 'not started', None, None
    if not any(t['state'] != 'done' for t in tasks):
        return 'finished', 'finished', 0.0, 0.0
    eta = model.get('eta')
    if eta is None:
        missing = [name for k, name in (('branch', 'training update'), ('eval', 'evaluation'))
                   if not (model.get('speed') or {}).get(k)]
        why = (f"no finished {' or '.join(missing)} measured yet" if missing else
               model.get('eta_error') or 'no finished task measured yet')
        return 'time left unknown', f"unknown — {why}", None, None
    if 'left' not in eta['low']:
        return 'time left unknown', 'unknown — no live node is working on it', None, None
    lo, hi = eta['low']['left'], eta['high']['left']
    return range_text(lo, hi, now)[0], range_text(lo, hi, now)[1], lo, hi


def range_text(lo, hi, now):
    short = 'about ' + span(lo) + ('' if span(lo) == span(hi) else '–' + span(hi))
    a, b = kst(now + lo), kst(now + hi)
    when = a if a == b else f"{a} – {b[6:] if a[:5] == b[:5] else b}"
    return short, f"{short} left, done around {when} KST"


def range_reasons(model):
    """What the longer end of the range adds (eta_items 'high' scenario), in a few words."""
    out = []
    if model.get('open_u'):
        out.append('U branches are added for ' + ', '.join(f's{s}' for s in model['open_u']))
    planned_ext = sum(1 for t in model['tasks'] if str(t['key']).startswith('planned.ext.'))
    if planned_ext and type(model.get('CAP')) is int:
        out.append(f"the {planned_ext} extensions not created yet run to update {model['CAP']}")
    sp = model.get('speed') or {}
    if any(v and v[1] > v[0] * 1.05 for v in sp.values()):
        out.append('runs take as long as the slowest quarter measured so far')
    if 'eval' not in model.get('sealed', ()):
        out.append('each branch needs 2 budget-end evaluations')
    return out


def stage_counts(model):
    out = {ph: {s: 0 for s in (*STATES, 'total')} for ph in ('train', 'ext', 'eval')}
    for t in model['tasks']:
        out[t['phase']][t['state']] += 1
        out[t['phase']]['total'] += 1
    return out


def short_state(t):
    return {'running': f"running, {left_words(t)}", 'ready': 'can start now', 'retrying': 'will restart by itself',
            'waiting': 'waiting', 'person': 'STOPPED, needs you'}[t['state']]


def stage_line(model, ph, c):
    text = f"     {STAGE_WORD[ph]:<11} done {c['done']} of {c['total']} {STAGE_UNIT[ph]}"
    planned = sum(1 for t in model['tasks'] if t['phase'] == ph and t['planned'] and t['state'] != 'done')
    parts = [f"{c[s]} {w}" + (f" ({planned} not created yet)" if s == 'waiting' and planned else '')
             for s, w in STATE_WORDS if c[s]]
    if parts:
        text += ' · ' + ' · '.join(parts)
    left = [t for t in model['tasks'] if t['phase'] == ph and t['state'] != 'done']
    if 0 < len(left) <= 4:   # user 11-16 'ext 50/54 왜 안바뀜': name the last few
        text += ' → not done: ' + '; '.join(f"{tname(t, model)} ({short_state(t)})" for t in left)
    return text


STAGE_RE = re.compile(r'^     (training|extension|evaluation) +done (\d+) of (\d+) ')


def read_stage_line(line):
    """Counts back from a drawn stage line (check_screen)."""
    m = STAGE_RE.match(line)
    if not m:
        return None
    ph = {v: k for k, v in STAGE_WORD.items()}[m.group(1)]
    c = {s: 0 for s in (*STATES, 'total')}
    c['done'], c['total'] = int(m.group(2)), int(m.group(3))
    body = line.split(' → not done: ')[0]
    for s, w in STATE_WORDS:
        k = re.search(r' · (\d+) ' + re.escape(w) + r'(?= ·| \(|$)', body)
        c[s] = int(k.group(1)) if k else 0
    return ph, c


def cell_word(t):
    st = t['state']
    if st == 'done':
        return f"{t['done_u']}✓" if t['phase'] == 'ext' and type(t.get('done_u')) is int else '✓'
    if st == 'running':
        if t['phase'] == 'eval':
            return f"{int(100 * t['frac'])}%▶"
        if t['phase'] == 'ext' and t.get('budget') is not None:
            return f"{int(100 * t['budget'])}%▶"
        return f"{t['shown_u']}▶" if type(t.get('shown_u')) is int else '▶'
    if st == 'person':
        return 'HUNG' if t['row'] is not None and t['row']['status'] == 'running' else 'FAIL'
    return {'ready': 'ready', 'retrying': 'retry', 'waiting': 'wait'}[st]


def cell_text(items, phase=None):
    """One grid cell: one word per task (several budget-end evaluations: joined by ',')."""
    rank = {s: i for i, s in enumerate(STATES)}
    items = sorted(items, key=lambda t: (rank[t['state']], t.get('update') if type(t.get('update')) is int else -1))
    return ','.join(cell_word(t) for t in items) if items else '-'


def cell_counts(text):
    """Read a drawn cell back into counts (check_screen)."""
    c = {s: 0 for s in (*STATES, 'total', 'unreadable')}
    if text == '-':
        return c
    for word in text.split(','):
        c['total'] += 1
        state = ('done' if word.endswith('✓') else 'running' if word.endswith('▶') else
                 {'ready': 'ready', 'retry': 'retrying', 'wait': 'waiting', 'FAIL': 'person', 'HUNG': 'person'}.get(word))
        c[state or 'unreadable'] += 1
    return c


def mark_style(mark):
    mark = mark.strip()
    if mark in ('', '-', 'wait'):
        return 'dim'
    if 'FAIL' in mark or 'HUNG' in mark:
        return 'bad'
    if '▶' in mark:
        return 'run'
    if 'retry' in mark:
        return 'warn'
    if 'ready' in mark:
        return 'info'
    if mark.endswith('✓'):
        return 'done'
    return None


GRID_BLOCKS = (('train', 'training'), ('ext', 'extension'), ('eval275', 'eval@H'), ('evalend', 'eval@budget'))


def grid_lines(model, width):
    """Seed x branch grid of one experiment and the counts read back from the drawn cells."""
    drawn = {ph: {s: 0 for s in (*STATES, 'total', 'unreadable')} for ph in ('train', 'ext', 'eval')}
    seeds, cols, H = model['seeds'], model['cols'], model.get('H')
    lines = [paint(f" {model['label']} by seed and branch", 'head')]
    place = {}
    for t in model['tasks']:
        block = t['phase'] if t['phase'] != 'eval' else 'evalend' if is_budget_eval(t, H) else 'eval275'
        place.setdefault((block, t['seed'], t['col']), []).append(t)
    blocks = [(b, (f"eval@{H}" if b == 'eval275' and H else 'eval' if b == 'eval275' else name))
              for b, name in GRID_BLOCKS if any(k[0] == b for k in place)]
    table = {(b, s, c): cell_text(place.get((b, s, c), [])) for b, _ in blocks for s in seeds for c in cols}
    for (b, _s, _c), text in table.items():
        for k, v in cell_counts(text).items():
            drawn['eval' if b.startswith('eval') else b][k] += v
    widths = {c: max(len(c), *(len(table[(b, s, c)]) for b, _ in blocks for s in seeds)) for c in cols}
    first = 12
    def row_text(head, cells, header=False):
        plain = '  '.join([f'{head:<{first}}', *(f'{cells[c]:<{widths[c]}}' for c in part)]).rstrip()
        if len(plain) > width or not COLOR:
            return clip(plain, width) if not header else paint(clip(plain, width), 'dim')
        return '  '.join([paint(f'{head:<{first}}', 'dim' if header else None),
                          *(paint(f'{cells[c]:<{widths[c]}}', 'dim' if header else mark_style(cells[c])) for c in part)]).rstrip()
    chunks, chunk, used = [], [], first
    for c in cols:
        if chunk and used + 2 + widths[c] > width:
            chunks.append(chunk); chunk, used = [], first
        chunk.append(c); used += 2 + widths[c]
    chunks.append(chunk)
    for part in chunks:
        for b, name in blocks:
            lines.append(row_text(f' {name}', {c: c for c in part}, header=True))
            for s in seeds:
                lines.append(row_text(f'   s{s}', {c: table[(b, s, c)] for c in part}))
    cols_note = "columns: On = on-policy run · SR50 = switches to SR at update 50"
    if model['u_tau']:
        cols_note += " · U = switches to Uniform at the seed's switch point (" + \
            ', '.join(f"s{k} at {v}" for k, v in sorted(model['u_tau'].items(), key=lambda x: str(x[0]))) + ')'
    lines += [paint(part, 'dim') for part in wrap('   ' + cols_note, width, '     ')]
    return lines, drawn


def wrap(text, width, indent='     '):
    import textwrap
    return textwrap.wrap(text, max(60, width - 1), break_on_hyphens=False, break_long_words=False,
                         subsequent_indent=indent) or ['']


def task_order(t):
    return ({'train': 0, 'ext': 1, 'eval': 2}[t['phase']], str(t['seed']), natural(t['col']), str(t.get('update')))


def dependent_name(t, b, model):
    """A waiting task named relative to the step it waits for ('extension', 'eval at 275' of the same branch)."""
    if b is not None and (t['seed'], t['col']) == (b['seed'], b['col']):
        return tname(t, model).split(' ', 2)[2]
    if b is not None and t['seed'] == b['seed']:
        return tname(t, model).split(' ', 1)[1]
    return tname(t, model)


def waiting_lines(model, waiting):
    """Waiting tasks grouped by the step they wait for: one line per running step (with its time left), and one
    line each for steps that are STOPPED, restart by themselves, can start, are waiting, and the budget freeze."""
    groups = {}
    for t in waiting:
        b, words = blocker(t, model)
        if b is None and t['planned'] and 'freeze' not in words:
            words = 'the planner to create them at its next queue scan'
        if b is not None and words and words.startswith('to reach update'):
            words = 'to reach the switch points'   # one line per On run, not per switch point
        groups.setdefault(b['key'] if b is not None else words, (b, words, []))[2].append(t)
    def deps(b, items):
        names, runs = [], {}
        for t in sorted(items, key=task_order):
            name = dependent_name(t, b, model)
            m = re.fullmatch(r'(SR\d+|On|U) (training|extension)', name)
            if m and b is not None:   # 'SR75, SR100, SR125 training' instead of the word on every one
                if m.group(2) not in runs:
                    runs[m.group(2)] = len(names)
                    names.append([m.group(2)])
                names[runs[m.group(2)]].insert(-1, m.group(1))
            else:
                names.append(name)
        names = [n if isinstance(n, str) else ', '.join(n[:-1]) + ' ' + n[-1] for n in names]
        new = sum(t['planned'] for t in items)
        return ', '.join(names) + (' (not created yet)' if new == len(items) and new else
                                   f" ({new} not created yet)" if new else '')
    out = []
    merged = (('person', 'a STOPPED step — they never start until it is fixed', 'bad'),
              ('retrying', 'a step that restarts by itself', 'warn'), ('ready', 'a step that can start now', None),
              ('waiting', 'a step that is itself waiting', 'dim'), ('done', 'a finished step (picked up at the next queue scan)', 'dim'))
    running = sorted((g for g in groups.values() if g[0] is not None and g[0]['state'] == 'running'),
                     key=lambda g: (g[0].get('left_s') is None, g[0].get('left_s') or 0))
    for st, words, style in merged[:1]:
        part = [g for g in groups.values() if g[0] is not None and g[0]['state'] == st]
        if part:
            n = sum(len(g[2]) for g in part)
            out.append((f"after {words} ({n}): " + '; '.join(f"{tname(b, model)} → {deps(b, items)}" for b, _, items in part), style))
    for b, words, items in running[:6]:
        out.append((f"after {tname(b, model)}{' ' + words if words else ''} ({state_phrase(b)}): {deps(b, items)}", None))
    if len(running) > 6:
        rest = running[6:]
        out.append((f"after {len(rest)} more running steps ({sum(len(g[2]) for g in rest)}): " +
                    '; '.join(f"{tname(b, model)} → {deps(b, items)}" for b, _, items in rest), None))
    for st, words, style in merged[1:]:
        part = [g for g in groups.values() if g[0] is not None and g[0]['state'] == st]
        if part:
            n = sum(len(g[2]) for g in part)
            out.append((f"after {words} ({n}): " + '; '.join(f"{tname(b, model)} → {deps(b, items)}" for b, _, items in part), style))
    for b, words, items in groups.values():
        if b is None:
            out.append((f"waiting for {words} ({len(items)}): {deps(None, items)}", 'dim'))
    return out


def unfinished_lines(model, width, node_of, idle_nodes):
    """Every task that is not done, grouped by what happens next. Returns (lines, counts per state as listed)."""
    tasks = [t for t in model['tasks'] if t['state'] != 'done']
    listed = {s: 0 for s in STATES}
    lines = [paint(f" {model['label']}: not finished {len(tasks)} of {len(model['tasks'])}", 'head')]
    if not tasks:
        return [], listed
    by = {s: [t for t in tasks if t['state'] == s] for s in STATES}
    out = lines
    def head(text, style):
        out.append(paint('   ' + text, style))
    if by['person']:
        listed['person'] = len(by['person'])
        head(f"STOPPED — needs you ({len(by['person'])}):", 'bad')
        for t in by['person'][:12]:
            for part in wrap(f"     {tname(t, model)}: {person_words(t['why'])}", width, '       '):
                out.append(paint(part, 'bad'))
        if len(by['person']) > 12:
            out.append(paint(f"     … {len(by['person']) - 12} more: " +
                             ', '.join(tname(t, model) for t in by['person'][12:18]), 'bad'))
        if any(t['row'] is None or t['row']['status'] != 'running' for t in by['person']):
            out.append(paint(f"     → fix the cause, then run `{RETRY_COMMAND}` on one node (the next free node restarts them)", 'bad'))
    if by['running']:
        listed['running'] = len(by['running'])
        head(f"running now ({len(by['running'])}):", 'run')
        rows = [(tname(t, model), f"on {node_of.get(t['key'], (t['row'] or {}).get('host') or '?')}",
                 progress_words(t), left_words(t)) for t in sorted(by['running'], key=task_order)]
        w = [max(len(r[i]) for r in rows) for i in range(3)]
        for r in rows[:16]:
            out.append(clip(f"     {r[0]:<{w[0]}}  {r[1]:<{w[1]}}  {r[2]:<{w[2]}}  {r[3]}", width))
        if len(rows) > 16:
            out.append(f"     … {len(rows) - 16} more")
    if by['retrying']:
        listed['retrying'] = len(by['retrying'])
        head(f"will restart by itself on the next free node, from its last checkpoint ({len(by['retrying'])}):", 'warn')
        for t in by['retrying'][:6]:
            for part in wrap(f"     {tname(t, model)}: {retry_words(t['why'])}", width, '       ')[:3]:
                out.append(paint(part, 'warn'))
        if len(by['retrying']) > 6:
            rest = by['retrying'][6:]
            for part in wrap(f"     … {len(rest)} more: " + ', '.join(tname(t, model) for t in rest[:5]) +
                             (', …' if len(rest) > 5 else ''), width, '       '):
                out.append(paint(part, 'warn'))
    if by['ready']:
        listed['ready'] = len(by['ready'])
        why = (f"{idle_nodes} node{'s are' if idle_nodes != 1 else ' is'} idle — see nodes below for why" if idle_nodes
               else 'every live node is busy')
        head(f"can start now, waiting for a free node ({len(by['ready'])}; {why}):", 'info')
        for part in wrap('     ' + ', '.join(tname(t, model) for t in by['ready'][:8]) +
                         (f", … {len(by['ready']) - 8} more" if len(by['ready']) > 8 else ''), width, '       '):
            out.append(part)
    if by['waiting']:
        listed['waiting'] = len(by['waiting'])
        head(f"waiting for an earlier step ({len(by['waiting'])}):", 'dim')
        for line, style in waiting_lines(model, by['waiting']):
            parts = wrap('     ' + line, width, '       ')
            if len(parts) > 3:
                parts = parts[:3]
                parts[2] = clip(parts[2] + ' …', width)
            for part in parts:
                out.append(paint(part, style) if style else part)
    return out, listed


def live_stage(row, doing):
    """'answering question 75 of 500 on each GPU, 5s ago': the newest rank progress record (short_progress), with
    the per-GPU question count said to be per GPU so it is not read as the task's own progress."""
    if not row.get('_events') or not doing:
        return None
    text = re.sub(r' question (\d+) of (\S+?)(?=,| \(|$)', r' question \1 of \2 on each GPU', doing)
    text = re.sub(r', \d+ tok', '', text)
    return re.sub(r' \((\S+(?: min)?) ago\)$', r', \1 ago', text)


def node_lines(nodes, models_by, all_nodes, wanted, width):
    """Live nodes only: what each one does in words, then one summary line with why the idle ones are idle."""
    shown = nodes if all_nodes else [r for r in nodes if r['state'] not in ('gone', 'finished')]
    per = {}
    for r in nodes:
        per.setdefault(r['state'], set()).add((r['host'], r['pid']))
    n = lambda k: len(per.get(k, ()))
    alive = sum(n(k) for k in ('running', 'busy', 'starting', 'idle'))
    out = []
    if not nodes:
        return [paint(' nodes', 'head') + ': no worker has started yet' +
                (f" · {wanted} tasks could start now" if wanted else '')]
    from collections import Counter
    idle_why = Counter(r['doing'] for r in nodes if r['state'] == 'idle')
    parts = [f"{n('running')} working"] + [f"{n(k)} {k}" for k in ('busy', 'starting') if n(k)]
    if n('idle'):
        why = '; '.join(w for w, k in idle_why.most_common(3))
        parts.append(f"{n('idle')} idle — why: {why}")
    head = paint(f" nodes: {alive} live", 'head') + ' — ' + ', '.join(parts)
    if n('hung'):
        head += ' · ' + paint(f"{n('hung')} HUNG: " + ', '.join(sorted(h for h, _ in per['hung'])), 'bad')
    out.extend(wrap(head, width + (len(head) - len(re.sub(r'\033\[[0-9;]*m', '', head))), '   '))
    word = {'running': 'working', 'hung': 'HUNG', 'busy': 'busy', 'starting': 'starting', 'idle': 'idle',
            'gone': 'gone', 'finished': 'finished'}
    rows = []
    for r in shown:
        model = models_by.get((r.get('base'), r.get('cohort'))) if r.get('key') else None
        t = (model or {}).get('by_key', {}).get(r.get('key')) if model else None
        task = tname(t, model) if t else r.get('task') or ''
        beat = f"seen {r['beat']}"
        if r['state'] == 'running' and t is not None:
            what, left = progress_words(t) or r['doing'], left_words(t)
            stage = live_stage(t['row'], r['doing'])
            # silent node: 'no heartbeat 4 min, task lock held …' says more than the heartbeat age
            beat = f"now {stage}" if stage else r['doing'] if r['doing'].startswith('no heartbeat') else beat
        else:
            what, left = r['doing'], ''
        rows.append(dict(host=r['host'] + (f"  GPU {r['gpus']}" if r.get('gpus') and r['gpus'] != '?' else ''),
                         state=word[r['state']], exp=r.get('experiment') or '', task=task, what=what, left=left,
                         ran=f"ran {r['took']}" if r.get('took') else '', beat=beat,
                         style={'hung': 'bad', 'busy': 'warn', 'starting': 'info', 'idle': 'dim'}.get(r['state'])))
    if rows:
        # one experiment running: its name on every row says nothing (phone width)
        one = len({x['exp'] for x in rows if x['exp']}) <= 1 and len(models_by) <= 1
        keys = ('host', 'state', *(() if one else ('exp',)), 'task', 'what', 'left')
        busy = [x for x in rows if x['state'] == 'working'] or rows
        w = {k: max(len(x[k]) for x in busy) for k in keys}
        w['host'] = max(len(x['host']) for x in rows)
        w['state'] = max(len(x['state']) for x in rows)
        for x in rows:
            if x['state'] == 'working':
                text = '   ' + '  '.join(f"{x[k]:<{w[k]}}" for k in keys) + '  ' + x['beat']
            else:   # hung, idle, busy, starting: the worker's own sentence
                text = '   ' + '  '.join([f"{x['host']:<{w['host']}}", f"{x['state']:<{w['state']}}"] +
                                         ([x['exp']] if x['exp'] and not one else []) +
                                         ([x['task'] + ':'] if x['task'] else []) + [x['what']]) + ' · ' + x['beat']
            out.append(paint(clip(text, width), x['style']))
    more = max(0, wanted - alive)
    out.append(paint(f"   {wanted} task{'s' if wanted != 1 else ''} could run now (running + can start + will restart) "
                     f"on {alive} live node{'s' if alive != 1 else ''}", 'dim') +
               (paint(f" → {more} more node{'s' if more != 1 else ''} would finish sooner", 'info') if more else ''))
    hidden = [f"{n(k)} {k}" for k in ('gone', 'finished') if n(k)]
    if hidden and not all_nodes:
        out.append(paint(f"   ({', '.join(hidden)} not shown; --all-nodes)", 'dim'))
    older = [r for r in shown if r['code'].startswith('different')]
    if older:
        out.append(paint(clip("   older dispatch code on " + ', '.join(f"{r['host']}.{r['pid']} ({r['code']})" for r in older[:4]) +
                              ": restart it with the current bundle", width), 'warn'))
    if n('hung'):
        out.append(paint("   HUNG = its task lock is still held but silent over 10 min, so no other node can take that task: "
                         "stop that process and run the same command again", 'dim'))
    return out


def check_screen(models, sec1, sec2, grid, node_rows):
    """Invariants of what is drawn (user 11-16 '위 아래 일관성'): per experiment, the stage lines (read back from the
    text), the not-finished list and the grid cells (read back) give the same done/total and the same count of
    running / can start / restart / waiting / stopped tasks, and the running tasks are exactly the working node rows."""
    out = []
    for model in models:
        if not model['tasks']:
            continue
        key = model['label'], model['base']
        top, cells, listed = sec1.get(key, {}), grid.get(key, {}), sec2.get(key, {})
        label = model['label']
        for ph in ('train', 'ext', 'eval'):
            a = top.get(ph) or {s: 0 for s in (*STATES, 'total')}
            b = cells.get(ph) or {s: 0 for s in (*STATES, 'total', 'unreadable')}
            if (a['done'], a['total']) != (b['done'], b['total']):
                ids = [t['key'] for t in model['tasks'] if t['phase'] == ph][:6]
                out.append(f"{label} {STAGE_WORD[ph]}: stage line {a['done']} of {a['total']} but grid "
                           f"{b['done']} of {b['total']} (tasks {', '.join(ids)})")
            if b.get('unreadable'):
                out.append(f"{label} {STAGE_WORD[ph]}: {b['unreadable']} grid cell(s) with an unknown word")
        for st, word in STATE_WORDS:
            a = sum((top.get(ph) or {}).get(st, 0) for ph in ('train', 'ext', 'eval'))
            b = sum((cells.get(ph) or {}).get(st, 0) for ph in ('train', 'ext', 'eval'))
            c = listed.get(st, 0)
            if not a == b == c:
                ids = [t['key'] for t in model['tasks'] if t['state'] == st][:6]
                out.append(f"{label} {word}: stage lines {a}, grid {b}, list {c} (tasks {', '.join(ids)})")
    for kind, states in (('running', ('running',)), ('hung', ('hung',))):
        expect = {(m['base'], m['cohort'], t['key']) for m in models for t in m['tasks']
                  if t['row'] is not None and t['row']['status'] == 'running' and
                  ((t['state'] == 'running') if kind == 'running' else (t['state'] == 'person'))}
        listed = {(r.get('base'), r.get('cohort'), r.get('key')) for r in node_rows if r['state'] in states and r.get('key')}
        for name, keys in (('not in the node list', expect - listed), ('in the node list only', listed - expect)):
            if keys:
                out.append(f"{kind} tasks {name}: " + ', '.join(sorted(f"{COHORT_LABEL.get(c, c)} {k}" for _, c, k in keys)[:6]))
    return out


def side_note(model):
    st = model['state']
    side = st.get('ready_side_work')
    if not (model['started'] and isinstance(side, dict) and st.get('phase') != 'complete' and
            isinstance(st.get('execution_order'), dict) and
            st['execution_order'].get('policy') == 'ready-checkpoint-evaluation-extension-v1'):
        return [], []
    if side.get('budget_ready'):
        note = 'budget frozen; ext/eval join the rotation when their checkpoints are ready'
    elif side.get('evaluation_independent_of_budget'):
        note = 'eval: published matched checkpoints run independently; ext: waiting for reference-policy cost budget'
        if side.get('budget_error'):
            note += '; budget error: ' + str(side['budget_error'])
    elif side.get('reason') == 'reference_endpoint_unfinished':
        note = 'waiting for the selected reference-policy endpoint to finish before freezing budget'
    else:
        note = 'waiting for complete reference-policy endpoints and measured costs before freezing budget'
    warn = [f"{kind} planner blocked: {error}; other ready work remains eligible"
            for kind, error in (side.get('planning_errors') or {}).items()]
    return [note], warn


def _render_clean(bases, now, gpus, host, cohorts, width, all_nodes):
    models = [task_model(base, cohort, now) for base in bases for cohort in cohorts]
    started = [m for m in models if m['started']]
    models_by = {(m['base'], m['cohort']): m for m in started}
    nodes = node_table(bases, cohorts, now, host, models=models_by)
    node_of = {r['key']: r['host'] for r in nodes if r.get('key') and r['state'] == 'running'}
    working = {}
    for r in nodes:
        if r['state'] == 'running' and r.get('cohort'):
            working.setdefault((r.get('base'), r['cohort']), set()).add((r['host'], r['pid']))
    idle_nodes = len({(r['host'], r['pid']) for r in nodes if r['state'] == 'idle'})
    mine = None
    for m in started:
        job = next((t for t in m['tasks'] if t['state'] == 'running' and t['row'].get('host') == host), None)
        if job and mine is None:
            mine = f"{tname(job, m)} ({m['label']}) · {progress_words(job)} · {left_words(job)}"

    # header, as in srgc_live_status.render
    stamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))
    gpu_line = ' '.join(f'{u}%' for i, u, _ in gpus) if gpus else 'unknown'
    out = [paint(f"V8 status · {code_version()} · {stamp}", 'head'),
           paint(', '.join(str(b) for b in bases), 'dim'),
           f"this node ({host}): " + (paint(mine, 'run') if mine else 'no V8 task here') + f" · GPU {gpu_line}", '']

    # 1. time left: the first thing on the screen (user 11-16 '남은 시간도 모르겠어 얼마나 남은건지')
    out.append(paint(' time left', 'head'))
    lows, highs, unknown = [], [], []
    for m in models:
        _, sentence, lo, hi = time_left(m, now)
        if not m['started']:
            continue
        k = len(working.get((m['base'], m['cohort']), ()))
        out.append(f"   {m['label']}: {sentence}" + (f" ({k} node{'s' if k != 1 else ''} working on it)"
                                                    if sentence not in ('finished',) else ''))
        if lo is None:
            unknown.append(m['label'])
        elif sentence != 'finished':
            lows.append(lo); highs.append(hi)
            reasons = range_reasons(m)
            if reasons and span(lo) != span(hi):
                for part in wrap('     longer end if ' + ' · '.join(reasons), width, '       '):
                    out.append(paint(part, 'dim'))
    not_started = [m['label'] for m in models if not m['started']]
    if started:
        if unknown:
            total = f"unknown until {', '.join(unknown)} has a measured time"
        elif lows:
            total = range_text(max(lows), max(highs), now)[1]
        else:
            total = 'finished'
        for part in wrap(f"   All experiments: {total}" +
                         (f" (not started, not counted: {', '.join(not_started)})" if not_started else ''), width):
            out.append(part)
    else:
        out.append('   no experiment has started yet')
    out.append('')

    # 2. one block per experiment: stage, plain counts per stage
    sec1 = {}
    out.append(paint(' experiments', 'head'))
    wanted = 0
    for m in models:
        if not m['started']:
            out.append(f"   {m['label']} · not started")
            continue
        pct, _ = model_progress(m)
        short, _, lo, _ = time_left(m, now)
        k = len(working.get((m['base'], m['cohort']), ()))
        out.append(paint(f"   {m['label']}", 'head') + f" · stage: {m['phase']}" +
                   (f" · {pct}% of the work done" if pct is not None else '') +
                   (f" · {short}{' left' if lo is not None and short.startswith('about') else ''}" if short != 'finished' else ' · finished') +
                   f" · {k} node{'s' if k != 1 else ''} working on it")
        counts = stage_counts(m)
        sec1[m['label'], m['base']] = {}
        for ph in ('train', 'ext', 'eval'):
            if not counts[ph]['total'] and ph != 'train':
                sec1[m['label'], m['base']][ph] = counts[ph]
                continue
            line = stage_line(m, ph, counts[ph])
            back = read_stage_line(line)
            sec1[m['label'], m['base']][back[0]] = back[1]
            for part in wrap(line, width, '         '):
                out.append(part)
        wanted += sum(t['state'] in ('running', 'ready', 'retrying') for t in m['tasks'])
        notes, warns = side_note(m)
        for note in m['more'] + notes:
            for part in wrap('     note: ' + note, width, '       '):
                out.append(paint(part, 'dim'))
        for w in warns + ([f"stage planner blocked: {m['state']['base_planning_error']}"] if m['state'].get('base_planning_error') else []):
            out.append(paint(clip('     ' + w, width), 'warn'))
        if m.get('notes'):
            out.append(paint(clip('     note: ' + '; '.join(m['notes']), width), 'dim'))
    out.append('')

    # 3. every task that is not finished, by what happens next (user 11-16 'ext 50/54 왜 안바뀜')
    sec2 = {}
    problems = []
    for m in started:
        if m['state'].get('failure'):
            problems.append(f"{m['label']}: the queue itself stopped: {m['state']['failure']}")
        queue = Path(m['base']) / m['cohort'] / '.multi-node'
        unreadable = access_problems(queue) + ([f"{len(m['state']['_unreadable_jobs'])} unreadable job record(s): " +
                                                ', '.join(m['state']['_unreadable_jobs'][:3])]
                                               if m['state'].get('_unreadable_jobs') else [])
        problems += [f"{m['label']}: cannot read {u}" for u in unreadable[:3]]
        lines, sec2[m['label'], m['base']] = unfinished_lines(m, width, node_of, idle_nodes)
        if lines:
            out += [*lines, '']
    if problems:
        out.append(paint(f" STOPPED — needs you: problems with the queue itself ({len(problems)})", 'bad'))
        for p in problems:
            for part in wrap('   ' + p, width, '     '):
                out.append(paint(part, 'bad'))
        out.append('')

    # 4. seed x branch grid (SRGC's seed table)
    grid = {}
    for m in started:
        if not m['tasks']:
            continue
        lines, grid[m['label'], m['base']] = grid_lines(m, width)
        out += lines
    if grid:
        for part in wrap('   ' + CELL_LEGEND, width, '     '):
            out.append(paint(part, 'dim'))
        out.append('')

    # 5. nodes: live heartbeats (and task locks still held) only, nothing kept from old logs
    out += node_lines(nodes, models_by, all_nodes, wanted, width)

    # invariants of everything above
    broken = check_screen(started, sec1, sec2, grid, nodes)
    if broken:
        out.append(paint(f" ✗ screen counts disagree ({len(broken)}): report this; the queue itself is not changed", 'bad'))
        for line in broken[:8]:
            out.append(paint(clip('   ' + line, width), 'bad'))
    idle = gpus is not None and len(gpus) > 0 and not any(u >= 10 for _, u, _ in gpus)
    for m in started:
        for t in m['tasks']:
            r = t['row']
            if idle and r is not None and r['status'] == 'running' and r.get('host') == host and (r.get('activity_age') or 0) > 1800:
                out.append(paint(clip(f" WARNING {m['label']} {tname(t, m)}: no log or progress record for "
                                      f"{age_text(r['activity_age'])} and this node's GPUs are idle (stack dumps: "
                                      f"{Path(m['base']) / m['cohort'] / '.multi-node' / 'stacks'})", width), 'bad'))
    return '\n'.join(out) + '\n'


def render(bases, now=None, gpus=None, host=None, cohorts=COHORTS, style='clean', all_nodes=False):
    """The one V8 status screen. --simple / --classic used to print two more layouts that counted tasks their own
    way (user 11-16 'status가 정보가 하나도 안맞는다고 여러개가 있는데'); every style is now this screen."""
    now = time.time() if now is None else now
    host = host or socket.gethostname()
    return render_clean(bases, now=now, gpus=gpus, host=host, cohorts=cohorts, all_nodes=all_nodes)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    def cohort_name(value):
        # run_v8_nodes.sh accepts every configured cohort, including those outside the default three.
        if not re.fullmatch(r'[A-Za-z0-9_-]+', value):
            raise argparse.ArgumentTypeError('cohort must be a simple configured identifier')
        return value
    parser.add_argument('--cohort', action='append', type=cohort_name, help='show only this cohort (repeatable)')
    parser.add_argument('--simple', action='store_true', help='same screen (kept so old commands still work)')
    parser.add_argument('--classic', action='store_true', help='same screen (kept so old commands still work)')
    parser.add_argument('--all-nodes', action='store_true', help='also list gone and finished workers')
    parser.add_argument('--watch', nargs='?', const=3.0, type=float, default=None, metavar='SECONDS',
                        help='refresh every SECONDS (default 3) until Ctrl-C')
    args = parser.parse_args(argv)
    global COLOR
    COLOR = sys.stdout.isatty() and not os.environ.get('NO_COLOR') and os.environ.get('TERM') != 'dumb'
    # An explicit V8_ROOT wins; otherwise the roots of this node's running V8 processes, else the default.
    bases = [Path(os.environ['V8_ROOT'])] if os.environ.get('V8_ROOT') else (active_roots() or [root(os.environ)])
    if args.watch is None:
        sys.stdout.write(render(bases, cohorts=tuple(args.cohort or COHORTS), all_nodes=args.all_nodes))
        return 0
    try:
        while True:
            sys.stdout.write('\033[H\033[2J\033[3J' + render(bases, cohorts=tuple(args.cohort or COHORTS), all_nodes=args.all_nodes) + f"\n(refresh every {int(args.watch)}s · Ctrl-C to stop)\n")
            sys.stdout.flush()
            time.sleep(max(1.0, args.watch))
    except KeyboardInterrupt:
        return 0


if __name__ == '__main__':
    sys.exit(main())
