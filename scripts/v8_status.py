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
    """'update 62/75 · answering 3/8, 812 tok (5s ago)' / 'answering 75/500, 300 tok (5s ago) · ~17 min left'.
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
        text += f" {index}/{total}" if total else f" {index}/?"
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
        text = f"update {row['done']}/{row['total']} · " + text
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
                updates = f"{r['done']}/{r['total']}" if r['done'] is not None and r['total'] else \
                          ('?/' + str(r['total']) if r['total'] and r.get('kind') != 'evaluate' else '')
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
            out.append(dict(base,state=state,experiment=COHORT_LABEL.get(diagnosis.get('cohort'),''),
                            doing=scheduler_text(diagnosis,now)))
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
LEGEND_LINES = (
    "읽는 법: 줄=시드, 열=branch (On=On-policy 궤적, SR50=50 업데이트에서 SR로 전환, U=그 시드의 전환점에서 Uniform으로 전환)",
    "  train=학습, ext=예산 연장 학습, eval=평가. train·ext 칸은 작업 1개, eval 칸은 끝난 수/전체 수 (275 평가 + 예산 끝점 평가)",
    "  ✓ 끝남  ▶ 실행 중 (▶120 = 120 업데이트째)  ○ 바로 시작 가능  ↻ 실패·중단, 다음 빈 노드가 체크포인트부터 자동 재시작",
    "  · 앞 단계가 끝나야 시작  ✗ 자동 재시도 끝, 사람이 봐야 함  - 작업 없음.  eval 칸 1/3▶↻ = 3개 중 1개 끝, 1개 실행, 1개 재시작 대기",
    "아래 표: train/ext/eval = 끝난 작업/전체 작업 (전체 = 만든 작업 + 계획상 꼭 만들어질 작업). run ready retry wait ✗ = 위 칸 기호 수",
    "  done = 끝난 일의 비율 (업데이트·평가를 잰 시간으로 환산)  left = 남은 일 ÷ 지금 도는 작업 수.  ? = 잰 시간이 아직 없거나 도는 작업이 없음",
)


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def median(values):
    values = sorted(v for v in values if number(v) and v > 0)
    if not values:
        return None
    n = len(values)
    return values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2


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
            if done_u > end:   # past the planner's budget estimate: the cap is the end
                cap = task.get('hard_cap') if type(task.get('hard_cap')) is int else CAP
                end = max(done_u, cap if type(cap) is int else done_u)
        rec = add(key=r['key'], phase=phase, seed=r['seed'], col=col, state=st, why=why, row=r, start=start, end=end,
                  done_u=done_u, shown_u=r['done'] if type(r.get('done')) is int else None)
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
                    state='ready' if ready else 'waiting', start=start, end=end, done_u=start)

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
            later = '·'.join(w for p, w in (('train', '학습'), ('ext', '연장'), ('eval', '평가')) if p not in sealed)
            model['more'].append(f"U(Uniform) {later} 최대 {len(open_u)}개씩 더: "
                                 f"{', '.join(f's{s}' for s in open_u)}의 On이 끝나면 전환 여부로 정해짐")
        if 'eval' not in sealed:
            model['more'].append("예산 끝점 평가가 더 생길 수 있음: 연장 학습이 끝나 예산 평가 계획이 서면 수가 정해짐")
    elif sealed != {'train', 'ext', 'eval'}:
        why = ('설정 파일을 못 읽어' if not cfg else '이 큐의 시드가 설정과 달라')
        model['more'].append(f"{why} 아직 안 만든 작업을 셀 수 없음: 전체 수는 지금까지 만든 작업 수")

    # Measured time of one training update and of one evaluation: medians over finished tasks (wall time of the
    # last attempt / updates that attempt trained), else over running tasks with enough progress.
    def attempt_start(t):
        attempts = (t['row'].get('_manifest') or {}).get('attempts')
        first = attempts[-1].get('start_update') if isinstance(attempts, list) and attempts and isinstance(attempts[-1], dict) else None
        return first if type(first) is int and first >= 0 else t['start']
    def took(t, end=None):
        a, b = t['row'].get('started'), now if end is None else t['row'].get('ended')
        return b - a if number(a) and number(b) and b > a else None
    per_update, per_eval, run_update, run_eval = [], [], [], []
    for t in tasks:
        if t['row'] is None:
            continue
        if t['phase'] == 'eval':
            if t['state'] == 'done' and took(t, 'end'):
                per_eval.append(took(t, 'end'))
            elif t['state'] == 'running' and t['frac'] >= 0.2 and took(t):
                run_eval.append(took(t) / t['frac'])
            continue
        n = (t['done_u'] or 0) - attempt_start(t)
        if t['state'] == 'done' and n > 0 and took(t, 'end'):
            per_update.append(took(t, 'end') / n)
        elif t['state'] == 'running' and n >= 3 and took(t):
            run_update.append(took(t) / n)
    spu, spe = median(per_update) or median(run_update), median(per_eval) or median(run_eval)
    model.update(spu=spu, spe=spe)

    for t in tasks:
        if t['phase'] == 'eval':
            left_units = 0.0 if t['state'] == 'done' else 1.0 - t['frac']
            t['left_s'] = left_units * spe if spe else None
            if t['state'] in ('running', 'person') and t['row'] is not None and t['row']['status'] == 'running':
                i, n = t.get('questions') or (0, None)
                t["progress"] = f"{i}/{n if n else '?'}"   # questions answered (user 11-16: "q 는 왜 붙여 놨냐")
        else:
            left_units = 0 if t['state'] == 'done' else max(0, t['end'] - t['done_u'])
            t['left_s'] = left_units * spu if spu else None
            if t['row'] is not None and t['row']['status'] == 'running':
                t['progress'] = f"{t['shown_u'] if t.get('shown_u') is not None else '?'}/{t['end']}"
        t['left_units'] = left_units
        if t['row'] is not None and t['row']['status'] == 'running':
            t['left'] = ('~' + span(t['left_s'])) if t['left_s'] is not None else '?'

    cols = []
    for t in tasks:
        if t['col'] not in cols:
            cols.append(t['col'])
    def col_key(c):
        kind = 0 if c == 'On' else 1 if c.startswith('SR') else 2 if c == 'U' else 3
        return (kind, int(re.sub(r'\D', '', c) or 0), c)
    model['cols'] = sorted(cols, key=col_key)
    model['seeds'] = sorted({t['seed'] for t in tasks}, key=lambda x: (not isinstance(x, int), x if isinstance(x, int) else str(x)))
    model['state'] = state
    model['phase'] = PHASE_LABEL.get(state.get('phase'), state.get('phase', 'not started'))
    model['runs'] = runs
    model['by_key'] = {t['key']: t for t in tasks}
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


def model_progress(model):
    """(percent of the work done in measured time, time left text): None/'?' when a needed time is unmeasured."""
    tasks, spu, spe = model['tasks'], model['spu'], model['spe']
    total_u = sum(t['end'] - t['start'] for t in tasks if t['phase'] != 'eval')
    left_u = sum(t['left_units'] for t in tasks if t['phase'] != 'eval')
    total_e = sum(1 for t in tasks if t['phase'] == 'eval')
    left_e = sum(t['left_units'] for t in tasks if t['phase'] == 'eval')
    if not tasks:
        return None, ''
    if not left_u and not left_e:
        return 100, 'done'
    if (total_u and not spu) or (total_e and not spe):
        return None, '?'
    work = total_u * (spu or 0) + total_e * (spe or 0)
    left = left_u * (spu or 0) + left_e * (spe or 0)
    pct = min(99, int(100 * (1 - left / work))) if work else 0
    running = sum(t['state'] == 'running' for t in tasks)
    return pct, ('~' + span(left / running)) if running else '?'


def cell_text(items, phase):
    """One grid cell. train/ext hold one task: its symbol (▶120 = running at update 120). An eval cell, or any cell
    with several tasks: finished/total, then one mark per non-waiting unfinished task (▶×2 = two running)."""
    if not items:
        return '-'
    if phase != 'eval' and len(items) == 1:
        t = items[0]
        return f"▶{t['shown_u']}" if t['state'] == 'running' and type(t.get('shown_u')) is int else MARK[t['state']]
    text = f"{sum(t['state'] == 'done' for t in items)}/{len(items)}"
    for st in ('running', 'ready', 'retrying', 'person'):
        k = sum(t['state'] == st for t in items)
        if k:
            text += MARK[st] + (f"×{k}" if k > 1 else '')
    return text


def cell_counts(text):
    """Read a drawn cell back into counts (check_screen)."""
    c = {s: 0 for s in (*STATES, 'total', 'unreadable')}
    if text == '-':
        return c
    m = re.fullmatch(r'(\d+)/(\d+)((?:[▶○↻✗](?:×\d+)?)*)', text)
    if m:
        c['total'], c['done'] = int(m.group(2)), int(m.group(1))
        for sym, k in re.findall(r'([▶○↻✗])(?:×(\d+))?', m.group(3)):
            c[MARK_STATE[sym]] += int(k or 1)
        c['waiting'] = c['total'] - sum(c[s] for s in STATES if s != 'waiting')
        return c
    c['total'] = 1
    c[MARK_STATE.get(text[:1], 'unreadable')] += 1
    return c


def mark_style(mark):
    mark = mark.strip()
    if not mark or mark == '-' or mark == '·':
        return 'dim'
    if '✗' in mark:
        return 'bad'
    if '▶' in mark:
        return 'run'
    if '↻' in mark:
        return 'warn'
    if mark == '✓':
        return 'done'
    m = re.fullmatch(r'(\d+)/(\d+)', mark)
    if m:
        return 'done' if m.group(1) == m.group(2) else 'dim' if m.group(1) == '0' else None
    return None


def experiment_lines(model, width):
    """Seed grids of one experiment and the counts read back from the drawn cells."""
    title = f"{model['label']} · {model['phase']} "
    lines = [paint(title, 'head') + paint('─' * max(3, min(width, 100) - len(title)), 'dim')]
    drawn = {ph: {s: 0 for s in (*STATES, 'total', 'unreadable')} for ph in ('train', 'ext', 'eval')}
    seeds, cols = model['seeds'], model['cols']
    if not model['tasks']:
        lines.append('  (no tasks yet)')
        return lines, drawn
    place = {}
    for t in model['tasks']:
        place.setdefault((t['phase'], t['seed'], t['col']), []).append(t)
    phases = [ph for ph in ('train', 'ext', 'eval') if any(t['phase'] == ph for t in model['tasks'])] or ['train']
    table = {(ph, s, c): cell_text(place.get((ph, s, c), []), ph) for ph in phases for s in seeds for c in cols}
    for (ph, _s, _c), text in table.items():
        for k, v in cell_counts(text).items():
            drawn[ph][k] += v
    widths = {c: max(len(c), *(len(table[(ph, s, c)]) for ph in phases for s in seeds)) for c in cols}
    def row_text(first, cells, header=False):
        plain = '  '.join([first, *(f'{cells[c]:<{widths[c]}}' for c in part)])
        if len(plain) > width:
            return clip(plain, width)
        return '  '.join([paint(first, 'dim') if header else first,
                          *(paint(f'{cells[c]:<{widths[c]}}', 'dim' if header else mark_style(cells[c])) for c in part)])
    chunks, chunk, used = [], [], 6
    for c in cols:   # wider than the screen: split the columns, never cut cells off
        if chunk and used + 2 + widths[c] > width:
            chunks.append(chunk); chunk, used = [], 6
        chunk.append(c); used += 2 + widths[c]
    chunks.append(chunk)
    for part in chunks:
        for ph in phases:
            lines.append(row_text(f' {ph:<5}', {c: c for c in part}, header=True))
            for s in seeds:
                lines.append(row_text(f'{s!s:>6}', {c: table[(ph, s, c)] for c in part}))
    if model['u_tau']:
        lines.append(paint(clip("   U = Uniform at the seed's switch point: " +
                                ' · '.join(f"s{k} @{v}" for k, v in sorted(model['u_tau'].items(), key=lambda x: str(x[0]))), width), 'dim'))
    for note in model['more']:
        lines.append(paint(clip('   + ' + note, width), 'dim'))
    return lines, drawn


def check_screen(models, table_rows, drawn, node_rows):
    """Invariants of what is drawn (user 11-16 '아니 일관되지가 않는다고'): per experiment and phase, the table's
    finished/total equals the sum of the drawn cells; run/ready/retry/wait/✗ equal the marks in the cells; the
    running tasks are exactly the ▶ node rows and the hung ones the ✗ node rows. Returns red lines (empty if all hold)."""
    out = []
    for model in models:
        if not model['tasks']:
            continue
        label, shown, cells = model['label'], table_rows[model['label'], model['base']], drawn[model['label'], model['base']]
        for ph in ('train', 'ext', 'eval'):
            d, n = (int(x) for x in shown[ph].split('/'))
            if (d, n) != (cells[ph]['done'], cells[ph]['total']):
                ids = [t['key'] for t in model['tasks'] if t['phase'] == ph][:6]
                out.append(f"{label} {ph}: table {d}/{n} but cells {cells[ph]['done']}/{cells[ph]['total']} (tasks {', '.join(ids)})")
            if cells[ph]['unreadable']:
                out.append(f"{label} {ph}: {cells[ph]['unreadable']} cell(s) with an unknown mark")
        for st, name in (('running', 'run'), ('ready', 'ready'), ('retrying', 'retry'), ('waiting', 'wait'), ('person', '✗')):
            in_cells = sum(cells[ph][st] for ph in cells)
            if int(shown[name]) != in_cells:
                ids = [t['key'] for t in model['tasks'] if t['state'] == st][:6]
                out.append(f"{label} {name}: table {shown[name]} but {in_cells} {MARK[st]} in the cells (tasks {', '.join(ids)})")
    for kind, states in (('running', ('running',)), ('hung', ('hung',))):
        expect = {(m['base'], m['cohort'], t['key']) for m in models for t in m['tasks']
                  if t['row'] is not None and t['row']['status'] == 'running' and
                  ((t['state'] == 'running') if kind == 'running' else (t['state'] == 'person'))}
        listed = {(r.get('base'), r.get('cohort'), r.get('key')) for r in node_rows if r['state'] in states and r.get('key')}
        for name, keys in (('not in the node list', expect - listed), ('in the node list only', listed - expect)):
            if keys:
                out.append(f"{kind} tasks {name}: " + ', '.join(sorted(f"{COHORT_LABEL.get(c, c)} {k}" for _, c, k in keys)[:6]))
    return out


def _render_clean(bases, now, gpus, host, cohorts, width, all_nodes):
    models = [task_model(base, cohort, now) for base in bases for cohort in cohorts]
    out, drawn, table_rows, mine = [], {}, {}, None
    for model in models:
        if not model['started']:
            continue
        lines, drawn[model['label'], model['base']] = experiment_lines(model, width)
        out += [*lines, '']
        if mine is None:
            job = next((t for t in model['tasks'] if t['state'] == 'running' and t['row'].get('host') == host), None)
            if job:
                mine = f"▶ {task_name(job['row'], model['runs'])} ({model['label']})"
    for part in LEGEND_LINES:
        out.append(paint(part, 'dim'))
    stamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))
    head = f"━━ V8 status · {code_version()} · {stamp} "
    out.append(paint(head + '━' * max(3, min(width, 100) - len(head)), 'head'))
    out.append(paint('   ' + ', '.join(str(b) for b in bases), 'dim'))

    # 1. one row per experiment; every number is a count over task_model()
    cols = ('phase', 'train', 'ext', 'eval', 'run', 'ready', 'retry', 'wait', '✗', 'done', 'left')
    rows, wanted = [], 0
    for model in models:
        if not model['started']:
            rows.append(dict(name=model['label'], phase='not started'))
            continue
        phases, states = model_counts(model)
        pct, left = model_progress(model)
        row = dict(name=model['label'], phase=model['phase'],
                   **{ph: f"{phases[ph]['done']}/{phases[ph]['total']}" for ph in phases},
                   run=str(states['running']), ready=str(states['ready']), retry=str(states['retrying']),
                   wait=str(states['waiting']), **{'✗': str(states['person'])},
                   done='?' if pct is None else f"{pct}%", left=left)
        table_rows[model['label'], model['base']] = row
        rows.append(row)
        wanted += states['running'] + states['ready'] + states['retrying']
    name_w = max(10, *(len(r['name']) for r in rows)) if rows else 10
    w = {c: max(len(c), *(len(str(r.get(c, ''))) for r in rows)) for c in cols}
    right = set(cols) - {'phase', 'left'}
    fmt = lambda c, v: f"{v:>{w[c]}}" if c in right else f"{v:<{w[c]}}"
    out.append(paint(' ' + ' ' * name_w + '  ' + '  '.join(fmt(c, c) for c in cols), 'dim'))
    for r in rows:
        parts = [paint(f" {r['name']:<{name_w}}", 'head')]
        for c in cols:
            v = fmt(c, str(r.get(c, '')))
            busy = v.strip() not in ('0', '')
            parts.append(paint(v, 'bad' if c == '✗' and busy else 'warn' if c == 'retry' and busy else
                               'run' if c == 'run' and busy else 'dim' if not busy else None))
        out.append('  '.join(parts).rstrip())

    # 2. what will restart by itself, what needs a person, planner notes
    import textwrap
    for model in models:
        retry = [t for t in model['tasks'] if t['state'] == 'retrying']
        if retry:
            text = (f" ↻ {model['label']} restarts by itself on the next free node: " +
                    '; '.join(f"{task_name(t['row'], model['runs'])} ({t['why']})" for t in retry[:4]) +
                    (f"; … {len(retry) - 4} more" if len(retry) > 4 else ''))
            for part in textwrap.wrap(text, max(60, width - 2), break_on_hyphens=False, subsequent_indent='   ')[:3]:
                out.append(paint(part, 'warn'))
        st = model['state']
        side = st.get('ready_side_work')
        if model['started'] and isinstance(side, dict) and st.get('phase') != 'complete' and \
                isinstance(st.get('execution_order'), dict) and st['execution_order'].get('policy') == 'ready-checkpoint-evaluation-extension-v1':
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
            out.append(paint(f" {model['label']} ext/eval: {note}", 'dim'))
            for kind, error in (side.get('planning_errors') or {}).items():
                out.append(paint(clip(f" {model['label']} {kind} planner blocked: {error}; other ready work remains eligible", width), 'warn'))
        if model['started'] and st.get('base_planning_error'):
            out.append(paint(clip(f" {model['label']} stage planner blocked: {st['base_planning_error']}", width), 'warn'))
    person = []
    for model in models:
        if not model['started']:
            continue
        if model['state'].get('failure'):
            person.append(f"✗ {model['label']}: queue failure: {model['state']['failure']}")
        queue = Path(model['base']) / model['cohort'] / '.multi-node'
        unreadable = access_problems(queue) + ([f"{len(model['state']['_unreadable_jobs'])} unreadable job record(s): " +
                                                ', '.join(model['state']['_unreadable_jobs'][:3])]
                                               if model['state'].get('_unreadable_jobs') else [])
        person += [f"✗ {model['label']}: cannot read {u}" for u in unreadable[:3]]
        person += [f"✗ {model['label']} {task_name(t['row'], model['runs'])}: {t['why']}"
                   for t in model['tasks'] if t['state'] == 'person']
    if person:
        out += ['', paint(f" needs a person ({len(person)})", 'head')]
        for line in person[:12]:
            for k, part in enumerate(textwrap.wrap(line, max(60, width - 3), break_on_hyphens=False, subsequent_indent='    ')[:2]):
                out.append(paint('  ' + part, 'bad'))
        if len(person) > 12:
            out.append(paint(f"   … {len(person) - 12} more (sh scripts/run_v8_nodes.sh --status lists every task)", 'dim'))

    # 3. nodes: live heartbeats (and task locks still held) only, nothing kept from old logs
    out.append('')
    nodes = node_table(bases, cohorts, now, host, models={(m['base'], m['cohort']): m for m in models if m['started']})
    shown = nodes if all_nodes else [r for r in nodes if r['state'] not in ('gone', 'finished')]
    order = ('hung', 'running', 'busy', 'starting', 'idle', 'gone', 'finished')
    per = {k: sorted({(r['host'], r['pid']) for r in nodes if r['state'] == k}) for k in order}
    alive = sum(len(per[k]) for k in ('running', 'busy', 'starting', 'idle'))
    if nodes:
        hosts = len({r['host'] for r in nodes if r['state'] not in ('gone', 'finished')})
        hidden = [f"{len(per[k])} {k}" for k in ('gone', 'finished') if per[k]]
        counts = ' · '.join(f"{len(per[k])} {k}" for k in ('running', 'busy', 'starting', 'idle') if per[k]) or '0 running'
        line = paint(f" nodes ({hosts} host{'s' if hosts != 1 else ''})", 'head') + f"  {counts}"
        if per['hung']:
            line += '  ' + paint(f"✗ {len(per['hung'])} hung: {', '.join(h for h, _ in per['hung'])}", 'bad')
        out.append(line)
        more = max(0, wanted - alive)
        out.append(paint(f"   tasks that can run now {wanted} (run + ready + retry) · alive workers {alive}", 'dim') +
                   (paint(f"  → {more} more 4-GPU node{'s' if more != 1 else ''} would help", 'info') if more else '') +
                   (paint(f"   ({', '.join(hidden)} not shown; --all-nodes)", 'dim') if hidden and not all_nodes else ''))
        if shown:
            ncols = (('host', 'host'), ('gpus', 'gpus'), ('state', 'state'), ('experiment', 'exp'), ('task', 'task'),
                     ('progress', 'done/total'), ('left', 'left'), ('took', 'time'), ('beat', 'beat'))
            listed = [dict(r, beat=r['beat'].replace(' ago', ''), progress=r.get('progress') or '', left=r.get('left') or '')
                      for r in shown]
            nw = {k: max(len(t), *(len(str(r[k])) for r in listed)) for k, t in ncols}
            out.append(paint('   ' + '  '.join(f"{t:<{nw[k]}}" for k, t in ncols) + '  now', 'dim'))
            for r in listed:
                mark = '✗' if r['state'] == 'hung' else '▶' if r['state'] == 'running' else '·' if r['state'] == 'busy' else ' '
                text = clip(f" {mark} " + '  '.join(f"{str(r[k]):<{nw[k]}}" for k, _ in ncols) + f"  {r['doing']}", width)
                out.append(paint(text, {'hung': 'bad', 'busy': 'warn', 'starting': 'info', 'idle': 'dim'}.get(r['state'])))
            older = [r for r in listed if r['code'].startswith('different')]
            if older:
                out.append(paint(clip("   older dispatch code on " + ', '.join(f"{r['host']}.{r['pid']} ({r['code']})" for r in older[:4]) +
                                      ": restart it with the current bundle", width), 'warn'))
            if per['hung']:
                out.append(paint("   hung = its task lock is still held but silent over 10 min, so no other node can take that task: "
                                 "stop that process and run the same command again", 'dim'))
    else:
        out.append(paint(' nodes', 'head') + '  no worker has started yet' + (f" · tasks that can run now {wanted}" if wanted else ''))

    # 4. invariants of everything above, then this node
    broken = check_screen([m for m in models if m['started']], table_rows, drawn, nodes)
    if broken:
        out.append(paint(f" ✗ screen counts disagree ({len(broken)}): report this; the queue itself is not changed", 'bad'))
        for line in broken[:8]:
            out.append(paint(clip('   ' + line, width), 'bad'))
    idle = gpus is not None and len(gpus) > 0 and not any(u >= 10 for _, u, _ in gpus)
    for model in models:
        for t in model['tasks']:
            r = t['row']
            if idle and r is not None and r['status'] == 'running' and r.get('host') == host and (r.get('activity_age') or 0) > 1800:
                out.append(paint(clip(f" WARNING {model['label']} {task_name(r, model['runs'])}: no log or progress record for "
                                      f"{age_text(r['activity_age'])} and this node's GPUs are idle (stack dumps: "
                                      f"{Path(model['base']) / model['cohort'] / '.multi-node' / 'stacks'})", width), 'bad'))
    gpu_line = ' '.join(f'{u}%' for i, u, _ in gpus) if gpus else 'unknown'
    out.append(paint(' this node', 'head') + f"  {host} · GPU {gpu_line} · " + (paint(mine, 'run') if mine else paint('no V8 task here', 'dim')))
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
