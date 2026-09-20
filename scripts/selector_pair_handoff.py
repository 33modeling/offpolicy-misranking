"""Explicitly hand a verified local Pair controller to the pinned queue.

No lock, checkpoint, manifest, receipt or budget is edited by this helper.
Only a verified local controller receives TERM. Its existing cleanup handles
its workers; unfinished updates are not saved on TERM. Restarts use a separately
staged, pinned distributed runtime, never the unchanged legacy serial launcher.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import math
import os
from pathlib import Path
import select
import signal
import socket
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from selector_pair_diagnostic import collect, local_owners, observations, read_small
from selector_pair_deploy import stage_runtime, pinned_files, manifest_for, verify


def shared_available(lock):
    if lock.is_symlink():
        raise RuntimeError('refusing a symlinked root lock')
    with lock.open('rb') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True


def process(proc, pid):
    path = proc / str(pid)
    fields = read_small(path / 'stat').decode().rsplit(') ', 1)[1].split()
    uid = next(line.split()[1:] for line in read_small(path / 'status').decode().splitlines()
               if line.startswith('Uid:'))
    if not uid or any(int(value) != os.getuid() for value in uid):
        raise RuntimeError('process belongs to another user')
    argv = [os.fsdecode(word) for word in read_small(path / 'cmdline').split(b'\0') if word]
    return {'pid': pid, 'start': fields[19], 'ppid': int(fields[1]), 'state': fields[0],
            'argv': argv, 'cwd': (path / 'cwd').resolve(strict=True),
            'exe': (path / 'exe').resolve(strict=True), 'proc': proc}


def identity(value):
    return value['pid'], value['start'], value['argv'], value['cwd'], value['exe']


def option(words, key):
    positions = [index for index, word in enumerate(words) if word == key]
    if len(positions) != 1 or positions[0] + 1 == len(words):
        raise RuntimeError(f'missing or ambiguous {key}; no process stopped')
    return words[positions[0] + 1]


def verify_owner(proc, pid, root, repo):
    value = process(proc, pid)
    words = value['argv']
    if not value['exe'].name.startswith('python'):
        raise RuntimeError('lock holder is not a Python Pair controller')
    # Only the exact launcher argv is accepted, not -c, -m, injected flags, or
    # an unrelated program that merely mentions selector_pair_gpu.py.
    if (len(words) != 5 or words[2] not in {'run', 'develop', 'test', 'freeze'}
            or words[3] != '--root'
            or (value['cwd'] / words[1]).resolve() not in {
                repo / 'src/selector_pair_gpu.py', repo / 'scripts/queue_selector_pair_gpu.py'}
            or (value['cwd'] / words[4]).resolve() != root):
        raise RuntimeError('lock owner does not match this Pair root, checkout and running stage')
    value['mode'] = words[2]
    return value


def owner_checkout(proc, pid, repo):
    value = process(proc, pid)
    candidate = value['cwd']
    if candidate == repo:
        return repo
    cache = repo / '.work/pair-runtimes'
    if (candidate.parent != cache or len(candidate.name) != 40
            or any(c not in '0123456789abcdef' for c in candidate.name)):
        raise RuntimeError('controller checkout is outside this repository and its pinned runtimes')
    files = pinned_files(repo, commit=candidate.name)
    verify(candidate, manifest_for(files, commit=candidate.name))
    return candidate


def verify_local_allocation(proc, owner):
    # Same hostname is insufficient: separate jobs may share that name. These
    # are kernel namespace/cgroup identities, not timestamps or user labels.
    for namespace in ('pid', 'mnt'):
        current = (proc / 'self/ns' / namespace).stat()
        other = (proc / str(owner['pid']) / 'ns' / namespace).stat()
        if (current.st_dev, current.st_ino) != (other.st_dev, other.st_ino):
            raise RuntimeError('controller is in another allocation namespace; no process stopped')
    if read_small(proc / 'self/cgroup') != read_small(proc / str(owner['pid']) / 'cgroup'):
        raise RuntimeError('controller belongs to another allocation cgroup; no process stopped')
    env = process_environment(owner)
    for key in ('CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES'):
        if os.environ.get(key) and env.get(key) != os.environ[key]:
            raise RuntimeError('controller GPU allocation differs from this restart; no process stopped')


def descendants(proc, owner):
    found, pending = {}, [owner['pid']]
    while pending:
        pid = pending.pop()
        for task in (proc / str(pid) / 'task').iterdir():
            try:
                children = read_small(task / 'children').split()
            except FileNotFoundError:
                continue
            for child in children:
                child = int(child)
                if child in found:
                    continue
                try:
                    value = process(proc, child)
                except (FileNotFoundError, ProcessLookupError):
                    continue
                found[child] = value
                pending.append(child)
                if len(found) > 2048:
                    raise RuntimeError('too many descendants; no process stopped')
    return found


def nonces(value):
    try:
        fields = read_small(value['proc'] / str(value['pid']) / 'environ', 1048576).split(b'\0')
    except FileNotFoundError:
        return set()
    return {field.split(b'=', 1)[0] for field in fields
            if field.startswith(b'OM_SELECTION_COST_') and field.endswith(b'=1')}


def nonce_workers(proc, keys):
    if not keys:
        return {}
    found = {}
    for path in proc.iterdir():
        if not path.name.isdecimal():
            continue
        try:
            value = process(proc, int(path.name))
            if value['state'] != 'Z' and nonces(value) & keys:
                found[value['pid']] = value
        except (FileNotFoundError, ProcessLookupError, PermissionError, RuntimeError):
            continue
    return found


def process_environment(value):
    """Read a verified process environment privately; never print its contents."""
    raw = read_small(value['proc'] / str(value['pid']) / 'environ', 1048576)
    return {os.fsdecode(key): os.fsdecode(item)
            for key, item in (field.split(b'=', 1) for field in raw.split(b'\0') if b'=' in field)}


def diagnostic_environment(env, repo):
    # A validator is NOT a metered worker. Retaining the phase nonce lets the
    # live trainer's cleanup accidentally target this diagnostic's process group.
    env = {key: value for key, value in env.items() if not key.startswith('OM_SELECTION_COST_')}
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    env['PYTHONPATH'] = str(repo / 'src') + os.pathsep + env.get('PYTHONPATH', '')
    return env


def process_python(value, env):
    """Retain the actual venv invocation path, not /proc/exe's resolved binary."""
    import shutil

    words = value['argv']
    if not words:
        raise RuntimeError('process has no interpreter argv; no process stopped')
    original = words[0]
    if os.path.isabs(original):
        executable = original
    elif os.sep in original:
        executable = str(value['cwd'] / original)
    else:
        # Resolve relative PATH entries against the observed process cwd.
        search = os.pathsep.join(str(value['cwd'] / part) if not os.path.isabs(part)
                                 else part for part in env.get('PATH', os.defpath).split(os.pathsep))
        executable = shutil.which(original, path=search)
    if not executable:
        raise RuntimeError('original Python invocation cannot be resolved; no process stopped')
    executable = Path(os.path.abspath(executable))
    if (not executable.is_file() or not os.access(executable, os.X_OK)
            or executable.resolve(strict=True) != value['exe']
            or not value['exe'].name.startswith('python')):
        raise RuntimeError('original Python invocation no longer matches its process; no process stopped')
    # Do not resolve this return value: doing so discards the venv's pyvenv.cfg.
    return str(executable)


def validate_runtime(root, repo, owner):
    """Validate compatibility without locks or writes to the experiment root."""
    env = process_environment(owner)
    python = process_python(owner, env)
    env = diagnostic_environment(env, repo)
    code = '''
import sys, tempfile
from pathlib import Path
import selector_pair_gpu as pair
root = Path(sys.argv[1])
if not all(callable(getattr(pair, name, None)) for name in ('queue_lease', 'distributed_stage', 'run_distributed')):
    raise SystemExit('restart target is not a distributed Pair runtime; refusing another exclusive controller')
manifest = pair.manifest(root, bind_runtime=False)
names = (
    'startup-runtime.json', 'startup-defaults-runtime.json', 'startup-resources-runtime.json',
    'shared-checkpoint-recovery-runtime.json', 'budget-stop-evaluation-runtime.json',
    'pair-operations-runtime.json', 'pair-lock-observation-runtime.json',
    'pair-distributed-runtime.json', 'pair-wait-guard-runtime.json',
    'shared-mbpp-quarantine-runtime.json',
    'pair-status-runtime.json', 'pair-curve-progress-runtime.json',
    'pair-branch-queue-runtime.json', 'pair-curve-spawn-runtime.json',
    'pair-recollection-runtime.json',
)
with tempfile.TemporaryDirectory(prefix='selector-pair-runtime-check-') as directory:
    snapshot = Path(directory)
    for name in names:
        source = root / name
        if source.is_symlink():
            raise ValueError('refusing a symlinked runtime receipt')
        if not source.exists():
            continue
        with source.open('rb') as handle:
            raw = handle.read(1048577)
        if len(raw) > 1048576:
            raise ValueError('runtime receipt exceeds the metadata read limit')
        (snapshot / name).write_bytes(raw)
    pair.bind_startup_runtime(snapshot, manifest['code_hashes'])
print('[handoff] frozen manifest and runtime receipt compatibility validated', flush=True)
'''
    print('[handoff] checking restart-target compatibility before stopping any controller', flush=True)
    result = subprocess.run([python, '-B', '-c', code, str(root)], cwd=repo,
                            env=env, capture_output=True, text=True, timeout=120,
                            start_new_session=True)
    if result.returncode:
        raise RuntimeError('frozen runtime compatibility validation failed; live controller was NOT stopped')
    print(result.stdout.strip(), flush=True)


CHECK_CHECKPOINT = '''
import json, sys
from pathlib import Path
import train_selection_gate_grpo as trainer
import train_policy_grpo as checkpoints
captured = []
trainer.train = captured.append
sys.argv = ['train_selection_gate_grpo.py', *json.loads(sys.argv[1])]
trainer.main()
args = captured[0]
config = checkpoints.GrpoConfig(**{name: getattr(args, name) for name in (
    'group_size', 'clip_epsilon', 'learning_rate', 'epochs_per_batch',
    'max_grad_norm', 'advantage_epsilon', 'lora_rank', 'lora_alpha', 'checkpoint_every')})
contract = checkpoints._checkpoint_contract(args, config, args.expected_world_size)
checkpoint, step = checkpoints._latest_checkpoint(Path(args.output), args.target_steps, contract)
if checkpoint is None:
    raise SystemExit('no validated local checkpoint; current training was NOT stopped')
print('[checkpoint] validated local saved update', step, flush=True)
'''


def validate_training_checkpoints(root, repo, processes):
    """Use the trainer's own parser, contract and hash validator, CPU-only."""
    checked = set()
    names = {'selector_pair_train.py', 'train_selection_gate_grpo.py',
             'selection_switch_curve_train.py', 'train_policy_grpo.py'}
    for value in processes.values():
        words = value['argv']
        matches = [i for i, word in enumerate(words) if Path(word).name in names]
        if not matches:
            continue
        if len(matches) != 1:
            raise RuntimeError('ambiguous training command; no process stopped')
        index = matches[0]
        script = (value['cwd'] / words[index]).resolve()
        if script.parent != repo / 'src':
            raise RuntimeError('training child is outside this checkout')
        args = words[index + 1:]
        output = (value['cwd'] / option(args, '--output')).resolve()
        if not output.is_relative_to(root):
            raise RuntimeError('training output is outside this Pair root')
        if output in checked:
            continue
        # Read only. Importing the parser never calls distributed setup or train.
        env = process_environment(value)
        python = process_python(value, env)
        env = diagnostic_environment(env, repo)
        print(f'[checkpoint] verifying saved training state: {output}', flush=True)
        result = subprocess.run([str(python), '-B', '-c', CHECK_CHECKPOINT, json.dumps(args)],
                                cwd=value['cwd'], env=env, capture_output=True, text=True, timeout=120,
                                start_new_session=True)
        if result.returncode:
            # No arbitrary environment/argv/traceback dump into the small report.
            raise RuntimeError(f'local checkpoint validation failed: {output}; live training was NOT stopped')
        print(result.stdout.strip(), flush=True)
        checked.add(output)


def active_receipts(root, owner, keys):
    rows = []
    host = process_environment(owner).get('EXPERIMENTS_NODE_ID', socket.gethostname())
    for _, relative, value in observations(root, limit=None):
        event = value.get('event_id', '')
        if (value.get('pid') == owner['pid'] and value.get('host') == host
                and value.get('state') == 'running' and event
                and Path(event).name == event and event not in {'.', '..'}):
            rows.append((root / relative.parent / 'cost-events' / f'{event}.json', event))
    known = {b'OM_SELECTION_COST_' + event.encode() for _, event in rows}
    if keys - known:
        raise RuntimeError('cannot locate owned phase receipts; no process stopped')
    return rows


def valid_finish_receipt(value, event, progress):
    fields = ('event_id', 'phase', 'ledger', 'gpus', 'gpu_type', 'host')
    return (value.get('event_id') == event and value.get('state') == 'finished'
            and type(value.get('exit_code')) is int
            and all(key in progress and value.get(key) == progress[key] for key in fields)
            and all(type(value.get(key)) in (int, float) and math.isfinite(value[key])
                    and value[key] >= 0 for key in ('seconds', 'allocated_gpu_seconds', 'time')))


def handoff(root, repo, timeout=240, proc=Path('/proc'), *, launch_repo=None, restart_shared=False):
    root, repo, proc = Path(root).resolve(strict=True), Path(repo).resolve(strict=True), Path(proc)
    launch_repo = Path(launch_repo).resolve(strict=True) if launch_repo is not None else repo
    if not (root / 'pair.json').is_file():
        raise RuntimeError('existing pair.json is missing; refusing to initialize a different run')
    lock = root / '.pair.lock'
    shared = shared_available(lock)
    owners = local_owners(lock, proc, kind='READ') if shared and restart_shared else local_owners(lock, proc)
    if shared and (not restart_shared or not owners):
        print('[handoff] root accepts queue workers; no controller stopped', flush=True)
        return 'run'
    if len(owners) != 1:
        raise RuntimeError('controller owner not uniquely visible on this node; no process stopped')
    source_repo = owner_checkout(proc, owners[0], repo)
    owner = verify_owner(proc, owners[0], root, source_repo)
    verify_local_allocation(proc, owner)
    owner_environment = process_environment(owner)
    owner_python = process_python(owner, owner_environment)
    if not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
        raise RuntimeError('safe pidfd signalling unavailable; no process stopped')
    with contextlib.ExitStack() as stack:
        watched = {}

        def watch(value):
            fd = os.pidfd_open(value['pid'], 0)
            stack.callback(os.close, fd)
            if identity(process(proc, value['pid'])) != identity(value):
                raise RuntimeError('process identity changed; no process stopped')
            watched[value['pid']] = fd
            return fd

        owner_fd = watch(owner)
        children = descendants(proc, owner)
        keys = set().union(*(nonces(child) for child in children.values()))
        children.update(nonce_workers(proc, keys))
        for child in children.values():
            watch(child)
        try:
            parent = process(proc, owner['ppid'])
        except (FileNotFoundError, ProcessLookupError):
            parent = None
        if parent and len(parent['argv']) >= 2 and parent['exe'].name in {'bash', 'dash', 'sh'}:
            if (parent['cwd'] / parent['argv'][1]).resolve() == source_repo / 'scripts/run_selector_pair.sh':
                watch(parent)
        receipts = active_receipts(root, owner, keys)
        receipt_progress = {path: json.loads(read_small(path.parent.parent / 'progress.json'))
                            for path, _ in receipts}
        # Ownership/checkpoints refer to the old process checkout. Compatibility
        # must instead be checked against the code that will actually restart.
        validate_runtime(root, launch_repo, owner)
        validate_training_checkpoints(root, source_repo, children)
        # Recheck after potentially slow disk/hash reads. A changed phase must
        # be inspected afresh, never interrupted using an older phase's proof.
        current = descendants(proc, owner)
        current.update(nonce_workers(proc, keys))
        if {pid: identity(v) for pid, v in current.items()} != {pid: identity(v) for pid, v in children.items()}:
            raise RuntimeError('worker set changed during inspection; no process stopped; run this bash again')
        verify_local_allocation(proc, owner)
        if (identity(verify_owner(proc, owner['pid'], root, source_repo)) != identity(owner)
                or process_python(owner, process_environment(owner)) != owner_python
                or local_owners(lock, proc, kind='READ' if shared else 'WRITE') != owners
                or shared_available(lock) != shared
                or active_receipts(root, owner, keys) != receipts):
            raise RuntimeError('owner or phase changed; no process stopped')
        print(f"[handoff] verified local Pair controller pid={owner['pid']} stage={owner['mode']}; sending TERM", flush=True)
        signal.pidfd_send_signal(owner_fd, signal.SIGTERM)
        deadline, next_message = time.monotonic() + timeout, 0
        while True:
            live = [pid for pid, fd in watched.items() if not select.select([fd], [], [], 0)[0]]
            residual = nonce_workers(proc, keys)
            if not live and not residual and shared_available(lock):
                break
            now = time.monotonic()
            if now >= deadline:
                raise RuntimeError('cleanup not confirmed before timeout; no forced kill or restart; files preserved')
            if now >= next_message:
                print(f'[handoff] waiting for owned workers/launcher and root lease; processes={len(set(live) | set(residual))}', flush=True)
                next_message = now + 5
            time.sleep(min(.2, max(0, deadline - now)))
        for path, event in receipts:
            value = json.loads(read_small(path))
            if not valid_finish_receipt(value, event, receipt_progress[path]):
                raise RuntimeError('owned phase has no valid finish receipt; refusing restart')
    # An interactive shell often lacks the launcher's selected GPU variables.
    # Preserve that verified allocation instead of widening it on restart.
    for key in ('CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES'):
        if not os.environ.get(key) and owner_environment.get(key):
            os.environ[key] = owner_environment[key]
    # Compatibility was checked with this interpreter, including its venv.
    # The interactive shell's defaults must not select a different one later.
    os.environ['PAIR_PYTHON'] = owner_python
    print('[handoff] old controller and owned workers exited; saved work and charged costs preserved', flush=True)
    return owner['mode']


def main():
    default_repo = Path(__file__).resolve().parents[1]
    work = Path(os.environ.get('OM_WORK', f"/group-volume/{os.environ.get('OM_USER', 'minsoo3.kim')}/offpolicy-misranking"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(os.environ.get('PAIR_ROOT', work / 'runs/selector-pair-v1')))
    parser.add_argument('--repo', type=Path, default=default_repo)
    parser.add_argument('--timeout', type=float, default=240)
    args = parser.parse_args()
    try:
        if not 0 < args.timeout <= 900:
            raise ValueError('timeout must be between 0 and 900 seconds')
        if not (args.root / 'pair.json').is_file():
            raise RuntimeError('existing pair.json is missing; no run was initialized')
        launch_repo = stage_runtime(args.repo)
        mode = handoff(args.root, args.repo, args.timeout, launch_repo=launch_repo, restart_shared=True)
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
        print(f'[handoff-abort] {exc}', file=sys.stderr, flush=True)
        print(collect(args.root), file=sys.stderr, flush=True)
        return 2
    env = dict(os.environ, PAIR_ROOT=str(args.root.resolve()), E5_FORCE='0')
    print(f'[restart] same Pair root, stage={mode}; pinned distributed runtime={launch_repo}; '
          'normal GPU/node admission remains enabled', flush=True)
    os.execve('/bin/bash', ['bash', str(launch_repo / 'scripts/run_selector_pair.sh'), mode], env)


if __name__ == '__main__':
    raise SystemExit(main())
