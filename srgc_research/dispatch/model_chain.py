"""Drain Qwen, Llama and Gemma with isolated runtimes and existing shared task leases."""

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

from srgc_rebuttal.runtime import Busy, atomic_json, lease

REPO = Path(__file__).resolve().parents[2]
MODELS = ('qwen35', 'llama31', 'gemma4')
SCRIPTS = {name: f'run_srgc_{name}.sh' for name in MODELS}
YIELD = 76
BUSY = 75


def enabled(environment=None):
    return (os.environ if environment is None else environment).get('SRGC_MODEL_CHAIN_CHILD') == '1'


def order_for(first, value=None):
    order = (value or ','.join(MODELS)).split(',')
    if not order or len(set(order)) != len(order) or any(name not in MODELS for name in order) or first not in order:
        raise ValueError('model order must contain distinct supported models and the starting model')
    index = order.index(first)
    return order[index:] + order[:index]


def child_command(model, dataset, initial=None):
    # Each public shell selects its own interpreter and Gemma's pinned overlay.
    return ['sh', str(REPO / 'scripts' / SCRIPTS[model]), *(initial or [dataset])]


def run_child(command, environment, heartbeat):
    child = subprocess.Popen(command, cwd=REPO, env=environment, start_new_session=True)
    stopped = []

    def send(sig):
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            pass

    def stop(sig, frame):
        if not stopped:
            stopped.extend((sig, time.monotonic()))
        send(sig)

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        while True:
            heartbeat(child.pid)
            if stopped and time.monotonic() - stopped[1] > 45:
                send(signal.SIGKILL)
            try:
                code = child.wait(timeout=1)
                return 128 + stopped[0] if stopped else 128 - code if code < 0 else code
            except subprocess.TimeoutExpired:
                pass
    finally:
        try:
            if child.poll() is None:
                send(signal.SIGTERM)
                try:
                    child.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    send(signal.SIGKILL)
                    child.wait()
            send(signal.SIGTERM)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def drain(order, dataset, initial, environment, update, *, runner=run_child, poll_seconds=10, sleep=time.sleep):
    states = {name: 'pending' for name in order}
    failures = {}
    child_env = {**environment, 'SRGC_MODEL_CHAIN_CHILD': '1'}
    first_attempt = True
    while True:
        busy = False
        for model in order:
            if states[model] in {'finished', 'failed'}:
                continue
            args = list(initial) if model == order[0] else None
            if args and not first_attempt and len(args)>1 and args[1]=='resume':
                args[1]='run'
            first_attempt = False
            command = child_command(model, dataset, args)
            print(f'MODEL START {model} dataset={dataset}', flush=True)
            states[model] = 'running'
            update('running', model=model, models=dict(states))
            try:
                code = runner(command, dict(child_env), lambda pid, current=model: update(
                    'running', model=current, child_pid=pid, models=dict(states)))
                # The legacy Qwen resume action clears stop markers and exits;
                # continue into its actual queue only after that explicit action.
                if model == 'qwen35' and args and len(args) > 1 and args[1] == 'resume' and code == 0:
                    command = child_command(model, dataset, [dataset, 'run', *args[2:]])
                    code = runner(command, dict(child_env), lambda pid, current=model: update(
                        'running', model=current, child_pid=pid, models=dict(states)))
            except (OSError, subprocess.SubprocessError) as error:
                print(f'MODEL FAILED {model}: {error}', file=sys.stderr, flush=True)
                code = 2
            if code >= 128:
                update('stopped', model=model, exit_code=code, models=dict(states))
                return code
            states[model] = 'finished' if code == 0 else 'waiting' if code in (YIELD, BUSY) else 'failed'
            if states[model] == 'failed':
                failures[model] = code
            print(f'MODEL {model} {states[model]} exit={code}; checking next model', flush=True)
            update('switching', model=model, exit_code=code, models=dict(states))
            if code == BUSY:
                busy = True
                break
        if all(state in {'finished', 'failed'} for state in states.values()):
            update('failed' if failures else 'finished', models=dict(states), failures=failures)
            return 2 if failures else 0
        # Sleep only after every eligible model has been checked, or the entire
        # local GPU allocation is owned. Never spin on peer leases/retry delays.
        reason = 'GPU allocation owned' if busy else 'all unfinished model queues waiting on peers or retry delay'
        update('waiting', models=dict(states), idle_reason=reason, failures=dict(failures))
        print(f'MODEL WAIT: {reason}; retry in {poll_seconds}s', flush=True)
        sleep(poll_seconds)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', choices=MODELS, default='qwen35')
    parser.add_argument('args', nargs=argparse.REMAINDER)
    options = parser.parse_args(argv)
    initial = options.args or ['all']
    if initial[0] not in {'all', 'math', 'mbpp'} or len(initial) > 1 and initial[1] not in {'run', 'resume'}:
        parser.error('model chain accepts all/math/mbpp run or resume only')
    order = order_for(options.start, os.environ.get('SRGC_MODEL_ORDER'))
    poll = float(os.environ.get('SRGC_MODEL_POLL_SECONDS', '10'))
    if not 1 <= poll <= 60:
        parser.error('SRGC_MODEL_POLL_SECONDS must be between 1 and 60')
    from .llama31.storage import group_work, inside
    group, work = group_work(os.environ)
    common = inside(os.environ.get('SRGC_STORAGE_ROOT', str(work / 'srgc-rebuttal')), group)
    directory = common / '.model-chain'
    identity = uuid.uuid4().hex
    allocation = socket.gethostname() + ':' + os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3')
    key = hashlib.sha256(allocation.encode()).hexdigest()
    record = directory / 'workers' / f'{identity}.json'
    metadata = {'worker_id':identity, 'host':socket.gethostname(), 'pid':os.getpid(),
                'started':time.time(), 'dataset':initial[0], 'order':order,
                'devices':os.environ.get('CUDA_VISIBLE_DEVICES','0,1,2,3')}

    def update(status, **details):
        atomic_json(record, {**metadata, 'status': status, 'heartbeat': time.time(), **details})
    stopped = []
    def interrupt(sig, frame):
        stopped.append(sig)
        raise KeyboardInterrupt
    previous = {sig:signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        with lease(directory / 'nodes' / f'{key}.lock'):
            try:
                return drain(order, initial[0], initial, os.environ, update, poll_seconds=poll)
            except KeyboardInterrupt:
                code = 128 + stopped[-1] if stopped else 130
                update('stopped', exit_code=code)
                return code
            except BaseException as error:
                update('failed', error=f'{type(error).__name__}: {error}')
                raise
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)



def snapshot(environment=None, *, now=None):
    """Read coordinator evidence only; never import CUDA or manufacture progress."""
    environment = os.environ if environment is None else environment
    now = time.time() if now is None else now
    from .llama31.storage import group_work, inside
    try:
        group, work = group_work(environment)
        common = inside(environment.get('SRGC_STORAGE_ROOT', str(work / 'srgc-rebuttal')), group)
        paths = list((common / '.model-chain/workers').glob('*.json'))
    except (OSError, ValueError):
        return []
    sessions = {}
    for path in paths:
        try:
            row = json.loads(path.read_text())
            if row.get('status') not in {'running','switching','waiting','finished','stopped','failed'}:
                continue
            if row.get('order') and (not isinstance(row['order'],list) or any(m not in MODELS for m in row['order'])):
                continue
            age = now - float(row['heartbeat'])
            if not 0 <= age <= 86400:
                continue
            row['heartbeat_age_seconds'] = age
            if row['status'] in {'running','switching','waiting'} and age > 30:
                row['status'] = 'stale'
            host = row['host']
            if not isinstance(host,str):
                continue
            key = (host, str(row.get('devices','')))
            if key not in sessions or row['started'] > sessions[key]['started']:
                sessions[key] = row
        except (OSError, ValueError, TypeError, KeyError):
            continue
    return list(sessions.values())


def render_status(environment=None, *, now=None):
    rows = snapshot(environment, now=now)
    if not rows:
        return []
    lines = ['MODEL QUEUE']
    for row in sorted(rows, key=lambda r:r['host']):
        current = row.get('model') or '-'
        order = ' -> '.join(row.get('order') or [])
        lines.append(f"{row['host']} {row['status'].upper()} model={current} | {order}")
    return lines


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Busy as error:
        print(f'MODEL CHAIN busy: {error}', file=sys.stderr)
        raise SystemExit(BUSY) from None
    except (OSError, ValueError, RuntimeError) as error:
        print(f'MODEL CHAIN refused: {error}', file=sys.stderr)
        raise SystemExit(2) from None

