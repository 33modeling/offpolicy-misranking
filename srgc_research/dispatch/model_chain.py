"""Drain Qwen, Llama and Gemma with isolated runtimes and existing shared task leases."""

import argparse
import hashlib
import json
import math
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
PAUSED = 77


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
    """Complete every model's MATH before launching any model's MBPP."""
    phases = ('math',) if dataset == 'math' else ('math', 'mbpp')
    # Preserve CLI scope, tokenizer/source-plan arguments and interpreter choice.
    # An all command prepares both plans before the first phase. An explicit
    # mbpp command can use its already prepared MATH sibling via phase_queues.
    command_dataset = dataset
    arguments = list(initial)
    arguments[0] = command_dataset
    resumed = set()
    for phase in phases:
        def progress(status, *, current=phase, **details):
            update(status, active_dataset=current, **details)
        code = drain_phase(order, command_dataset, arguments,
                           {**environment, 'SRGC_DATASET_PHASE': phase}, progress,
                           runner=runner, poll_seconds=poll_seconds, sleep=sleep,
                           resumed=resumed)
        if code:
            return code
    return 0


def drain_phase(order, dataset, initial, environment, update, *, runner=run_child,
                poll_seconds=10, sleep=time.sleep, resumed=None):
    """Finish each model in order; failures and waiting never advance models."""
    from .resume_inventory import failed_signature, signature

    states = {name: 'pending' for name in order}
    failures = {}
    child_env = {**environment, 'SRGC_MODEL_CHAIN_CHILD': '1'}
    # An explicit resume authorizes clearing existing stop markers once per
    # model for this invocation. A later phase must respect newly placed stops.
    resumed = set() if resumed is None else resumed
    recovered = {}
    retrying = set()
    root_parser = argparse.ArgumentParser(add_help=False)
    root_parser.add_argument('--root', type=Path)
    custom_root, _ = root_parser.parse_known_args(initial)
    def saved_work(model):
        return signature(model, environment['SRGC_DATASET_PHASE'], environment,
                         root=custom_root.root if model == order[0] else None)
    def saved_errors(model):
        return failed_signature(model, environment['SRGC_DATASET_PHASE'], environment,
                                root=custom_root.root if model == order[0] else None)
    while True:
        busy = False
        killed = False
        retry = False
        # Model order is a completion barrier, not a rotation on failure,
        # cooldown or peer ownership. Finish this model's whole dataset before
        # inspecting or launching any later model, including its old ERRORs.
        current_model = next(model for model in order if states[model] != 'finished')
        recovery = {current_model: saved_work(current_model)}
        errors = {current_model} if saved_errors(current_model) is not None else set()
        recovery = {model: value for model, value in recovery.items()
                    if value is not None and (recovered.get(model) != value or model in errors)}
        # Retain recovery mode even if a peer acquires the saved task between
        # scans. A worker failure must not admit that model's fresh work.
        if current_model in retrying:
            recovery[current_model] = saved_work(current_model)
        candidates = [current_model]
        for model in candidates:
            if states[model] in {'finished', 'failed'}:
                continue
            args = (list(initial) if model == order[0] else
                    [dataset, 'resume'] if len(initial)>1 and initial[1]=='resume' else None)
            if args and model in resumed and len(args)>1 and args[1]=='resume':
                args[1]='run'
            if args and len(args)>1 and args[1]=='resume':
                resumed.add(model)
            command = child_command(model, dataset, args)
            resume_only = model in recovery
            attempt_env = {**child_env, 'SRGC_RESUME_ONLY': '1' if resume_only else '0'}
            print(f'MODEL START {model} dataset={environment["SRGC_DATASET_PHASE"]} resume_only={resume_only}', flush=True)
            states[model] = 'running'
            update('running', model=model, resume_only=resume_only, models=dict(states))
            try:
                code = runner(command, dict(attempt_env), lambda pid, current=model, mode=resume_only: update(
                    'running', model=current, resume_only=mode, child_pid=pid, models=dict(states)))
                # The legacy Qwen resume action clears stop markers and exits;
                # continue into its actual queue only after that explicit action.
                if model == 'qwen35' and args and len(args) > 1 and args[1] == 'resume' and code == 0:
                    command = child_command(model, dataset, [dataset, 'run', *args[2:]])
                    code = runner(command, dict(attempt_env), lambda pid, current=model, mode=resume_only: update(
                        'running', model=current, resume_only=mode, child_pid=pid, models=dict(states)))
            except (OSError, subprocess.SubprocessError) as error:
                print(f'MODEL FAILED {model}: {error}', file=sys.stderr, flush=True)
                code = 2
            if code in (129, 137):
                retrying.add(model)
                states[model] = 'waiting'
                update('waiting', model=model, exit_code=code, models=dict(states), resume_only=True,
                       idle_reason='worker interrupted; resuming saved work')
                killed = True
                break
            if code in (130, 143):
                update('stopped', model=model, exit_code=code, models=dict(states))
                return code
            if code == PAUSED:
                update('stopped', model=model, exit_code=code, models=dict(states))
                return code
            if code not in (0, YIELD, BUSY) and (resume_only or saved_work(model) is not None):
                retrying.add(model)
                recovered.pop(model, None)
                states[model] = 'waiting'
                failures[model] = code
                update('waiting', model=model, exit_code=code, models=dict(states),
                       failures=dict(failures), resume_only=True,
                       idle_reason='worker failed with unfinished work; retrying saved task before fresh work')
                print(f'MODEL RETRY {model} exit={code}; saved work remains; retry_limit=none', flush=True)
                retry = True
                break
            if code == 0 and saved_errors(model) is not None:
                retrying.add(model)
                recovered.pop(model, None)
                states[model] = 'waiting'
                retry = True
                update('waiting', model=model, exit_code=code, models=dict(states), resume_only=True,
                       idle_reason='worker exited successfully but unfinished task remains; holding current model')
                break
            states[model] = 'finished' if code == 0 else 'waiting' if code in (YIELD, BUSY) else 'failed'
            if code == 0 and resume_only:
                retrying.discard(model)
                failures.pop(model, None)
                recovered[model] = saved_work(model)
                states[model] = 'pending'
            if states[model] == 'failed':
                failures[model] = code
                update('failed', model=model, exit_code=code, models=dict(states), failures=dict(failures),
                       idle_reason='current model failed before saved work was available; later models blocked')
                return 2
            action = 'current model complete' if states[model] == 'finished' else 'holding current model'
            print(f'MODEL {model} {states[model]} exit={code}; {action}', flush=True)
            update('switching' if states[model] == 'finished' else 'waiting',
                   model=model, exit_code=code, models=dict(states))
            if code == BUSY:
                busy = True
                break
        if all(state in {'finished', 'failed'} for state in states.values()):
            update('failed' if failures else 'finished', models=dict(states), failures=failures)
            return 2 if failures else 0
        if states[current_model] == 'finished':
            continue  # Only a completed full worker releases the next model.
        if recovery and not busy and not killed and not retry and all(states[model] in {'pending', 'failed'} for model in recovery):
            if failures:
                update('failed', models=dict(states), failures=failures)
                return 2
            continue  # Recovery finished; proceed directly to the normal queues.
        # Hold the same model during errors, peer leases and retry delays.
        reason = ('worker failed; retrying saved work before fresh work' if retry
                  else 'worker interrupted; retrying saved work' if killed else 'GPU allocation owned' if busy
                  else 'current model waiting on peers or retry delay; later models blocked')
        update('waiting', model=current_model, models=dict(states), idle_reason=reason, failures=dict(failures),
               resume_only=bool(recovery) or killed or retry)
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
            if not isinstance(row, dict):
                continue
            if row.get('status') not in {'running','switching','waiting','finished','stopped','failed'}:
                continue
            order = row.get('order')
            if order is not None and (not isinstance(order, list) or any(m not in MODELS for m in order)):
                continue
            if row.get('model') is not None and row['model'] not in MODELS:
                continue
            if not isinstance(row.get('devices', ''), str):
                continue
            if any(type(row.get(name)) not in (int, float) or not math.isfinite(row[name])
                   or row[name] < 0 for name in ('heartbeat', 'started')):
                continue
            age = now - row['heartbeat']
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
        dataset = row.get('active_dataset') or row.get('dataset') or '-'
        mode = ' RESUME' if row.get('resume_only') else ''
        order = ' -> '.join(row.get('order') or [])
        lines.append(f"{row['host']} {row['status'].upper()} model={current} dataset={dataset}{mode} | {order}")
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

