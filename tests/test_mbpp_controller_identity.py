"""Live-controller rediscovery never relies on a host name or PID alone."""

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('mbpp_controller_identity', ROOT / 'scripts/mbpp_controller_identity.py')
identity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(identity)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else value.encode())


def envfile(path, env):
    put(path, '\0'.join(f'{key}={value}' for key, value in env.items()) + '\0')


@pytest.fixture
def local(tmp_path):
    proc = tmp_path / 'proc'
    logs = tmp_path / 'logs'
    work = tmp_path / 'work'
    lock = logs / 'mbpp-controller.same-host-gabcd_.lock'
    receipt = lock.with_suffix('.owner.json')
    owner = {'schema': 'mbpp-node-owner-v1', 'state': 'active', 'lock': str(lock),
             'pid': 123, 'start_time': 900, 'token': 'a' * 32}
    put(receipt, json.dumps(owner))
    env = {'OM_WORK': str(work), 'EXPERIMENTS_MBPP_SUITE': 'all',
           'EXPERIMENTS_NODE_ID': 'same-host-gabcd', 'CUDA_VISIBLE_DEVICES': ''}
    envfile(proc / '123/environ', env)
    envfile(proc / '124/environ', {**env, 'MBPP_GUARD_PID': '123', 'OM_MBPP_CONTROLLER_TOKEN': owner['token']})
    put(proc / '123/cmdline', f'python\0scripts/_mbpp_node_guard.py\0--lock\0{lock}\0--\0bash\0scripts/run_experiments.sh\0')
    put(proc / '124/cmdline', 'bash\0scripts/run_experiments.sh\0run\0')
    for pid in (123, 124):
        put(proc / f'{pid}/stat', f'{pid} (test process) S ' + '0 ' * 18 + '900 0 0')
    put(proc / '123/task/123/children', '124')
    for part in ('123', 'self'):
        put(proc / f'{part}/cgroup', '0::/allocation/a\n')
        (proc / f'{part}/ns').mkdir()
        (proc / f'{part}/ns/pid').symlink_to('pid:[1234]')
    return dict(proc=proc, logs=logs, work=work, receipt=receipt, env=env, owner=owner)


def discover(local, **kwargs):
    return identity.live_identity(local['receipt'], local['work'], 'all', local['env'],
                                  proc=local['proc'], **kwargs)


def test_live_guard_preserves_previous_identity_when_probe_name_changes(local):
    local['env']['EXPERIMENTS_NODE_ID'] = 'same-host-gffff'
    assert discover(local) == 'same-host-gabcd'


def test_empty_and_full_mask_can_recover_same_live_guard(local):
    local['env']['CUDA_VISIBLE_DEVICES'] = '1,0'
    assert discover(local, inventory=lambda: {'0': 'GPU-a', '1': 'GPU-b'}) == 'same-host-gabcd'


@pytest.mark.parametrize('mismatch', ['pid-reused', 'released', 'wrong-token', 'remote-namespace',
    'other-cgroup', 'other-work', 'wrong-lock', 'wrong-program', 'dead-child', 'wrong-parent', 'missing-child'])
def test_unproven_local_identity_is_never_adopted(local, mismatch):
    proc = local['proc']
    owner = local['owner']
    if mismatch == 'pid-reused':
        owner['start_time'] += 1
    elif mismatch == 'released':
        owner['state'] = 'released'
    elif mismatch == 'wrong-token':
        owner['token'] = 'b' * 32
    elif mismatch == 'remote-namespace':
        (proc / '123/ns/pid').unlink()
        (proc / '123/ns/pid').symlink_to('pid:[9999]')
    elif mismatch == 'other-cgroup':
        put(proc / '123/cgroup', '0::/allocation/b\n')
    elif mismatch == 'other-work':
        envfile(proc / '123/environ', {**local['env'], 'OM_WORK': '/another/work'})
    elif mismatch == 'wrong-lock':
        owner['lock'] = '/another/lock'
    elif mismatch == 'wrong-program':
        put(proc / '123/cmdline', 'python\0another_guard.py\0')
    elif mismatch == 'dead-child':
        put(proc / '124/stat', '124 (dead) Z ' + '0 ' * 18 + '900 0 0')
    elif mismatch == 'wrong-parent':
        envfile(proc / '124/environ', {**local['env'], 'MBPP_GUARD_PID': '999',
                                     'OM_MBPP_CONTROLLER_TOKEN': owner['token']})
    elif mismatch == 'missing-child':
        put(proc / '123/task/123/children', '')
    put(local['receipt'], json.dumps(owner))
    assert discover(local) is None


def test_unknown_device_equivalence_blocks_duplicate_controller(local):
    local['env']['CUDA_VISIBLE_DEVICES'] = '0,1'
    def failed_inventory():
        raise OSError('nvidia-smi failed')
    with pytest.raises(RuntimeError, match='ambiguous'):
        discover(local, inventory=failed_inventory)


def test_same_scope_but_disjoint_devices_are_not_adopted(local):
    local['env']['CUDA_VISIBLE_DEVICES'] = '2,3'
    envfile(local['proc'] / '123/environ', {**local['env'], 'CUDA_VISIBLE_DEVICES': '0,1'})
    assert discover(local, inventory=lambda: {str(i): f'GPU-{i}' for i in range(4)}) is None


def test_overlapping_gpu_allocations_fail_closed(local):
    local['env']['CUDA_VISIBLE_DEVICES'] = '1,2'
    envfile(local['proc'] / '123/environ', {**local['env'], 'CUDA_VISIBLE_DEVICES': '0,1'})
    with pytest.raises(RuntimeError, match='ambiguous'):
        discover(local, inventory=lambda: {str(i): f'GPU-{i}' for i in range(4)})


@pytest.mark.parametrize('suite', ['fresh', 'difficulty', 'long'])
def test_another_suite_is_not_silently_adopted(local, suite):
    envfile(local['proc'] / '123/environ', {**local['env'], 'EXPERIMENTS_MBPP_SUITE': suite})
    with pytest.raises(RuntimeError, match='another MBPP suite'):
        discover(local)


@pytest.mark.parametrize('previous,current', [('all', 'quality'), ('quality', 'all')])
def test_default_quality_alias_recovers_same_verified_controller(local, previous, current):
    envfile(local['proc'] / '123/environ', {**local['env'], 'EXPERIMENTS_MBPP_SUITE': previous})
    result = identity.live_identity(local['receipt'], local['work'], current, local['env'],
                                    proc=local['proc'], with_pid=True)
    assert result == ('same-host-gabcd', 123)


def test_cli_multiple_live_guards_are_blocked(local, monkeypatch, capsys):
    put(local['logs'] / 'mbpp-controller.other.owner.json', '{}')
    monkeypatch.setattr(identity, 'live_identity', lambda *args, **kwargs: ('same-host-gabcd', 123))
    monkeypatch.setattr(sys, 'argv', ['identity', '--logs', str(local['logs']), '--work', str(local['work']), '--suite', 'all'])
    assert identity.main() == 75
    assert 'multiple proven local' in capsys.readouterr().err


def test_cuda_order_is_not_identity():
    assert identity.same_devices({'CUDA_VISIBLE_DEVICES': '0,1'}, {'CUDA_VISIBLE_DEVICES': '1,0'}) is True


def test_unverifiable_allocation_mask_is_not_guessed():
    assert identity.same_devices({'NVIDIA_VISIBLE_DEVICES': 'GPU-a'}, {'NVIDIA_VISIBLE_DEVICES': 'GPU-b'}) is None


def test_another_user_controller_is_not_adopted(local, monkeypatch):
    uid = os.getuid()
    monkeypatch.setattr(identity.os, 'getuid', lambda: uid + 1)
    assert discover(local) is None


@pytest.mark.parametrize('requested_suite', ['all', 'quality'])
def test_real_guard_and_worker_survive_rediscovery_without_pid_file(tmp_path, requested_suite):
    work = tmp_path / 'work'
    logs = work / 'runs/experiments/logs'
    logs.mkdir(parents=True)
    lock = logs / 'mbpp-controller.same-host-gabcd_.lock'
    worker_script = tmp_path / 'run_experiments.sh'
    worker_pid = tmp_path / 'worker.pid'
    worker_script.write_text(f'sleep 60 &\necho $! > "{worker_pid}"\nwait\n')
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    nvidia = bin_dir / 'nvidia-smi'
    nvidia.write_text('#!/bin/sh\nprintf "0, GPU-a\\n1, GPU-b\\n"\n')
    nvidia.chmod(0o755)
    env = {**os.environ, 'OM_WORK': str(work), 'EXPERIMENTS_MBPP_SUITE': 'all',
           'EXPERIMENTS_NODE_ID': 'same-host-gabcd', 'CUDA_VISIBLE_DEVICES': '',
           'PATH': f'{bin_dir}:{os.environ["PATH"]}', 'SWITCH_PYTHON': sys.executable}
    env.pop('MBPP_GUARD_PID', None)
    guard = subprocess.Popen([sys.executable, str(ROOT / 'scripts/_mbpp_node_guard.py'),
                              '--lock', str(lock), '--', 'bash', str(worker_script)],
                             env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True)
    try:
        deadline = time.monotonic() + 5
        while not worker_pid.exists() and time.monotonic() < deadline:
            time.sleep(.02)
        assert worker_pid.exists() and guard.poll() is None
        child_pid = int(worker_pid.read_text())
        console = logs / 'console.mbpp.same-host-gabcd_.log'
        console.write_text('LIVE ORIGINAL MBPP CONSOLE\n')
        assert not list(logs.glob('*.pid'))
        result = subprocess.run(['bash', str(ROOT / 'scripts/run_experiments.sh'), 'logs'],
                                cwd=ROOT, env={**env, 'CUDA_VISIBLE_DEVICES': '1,0',
                                               'EXPERIMENTS_MBPP_SUITE': requested_suite,
                                               'EXPERIMENTS_NODE_ID': 'same-host-gffff'},
                                capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, result.stdout + result.stderr
        assert 'preserving live local MBPP controller identity: same-host-gabcd' in result.stdout
        assert 'LIVE ORIGINAL MBPP CONSOLE' in result.stdout
        assert '[already running] MBPP' in result.stdout
        assert guard.poll() is None
        os.kill(child_pid, 0)
        assert not list(logs.glob('*.pid'))
        assert '[stop]' not in result.stdout and '[detached]' not in result.stdout
    finally:
        if guard.poll() is None:
            guard.send_signal(signal.SIGTERM)
            try:
                guard.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(guard.pid, signal.SIGKILL)
                guard.wait(timeout=5)
