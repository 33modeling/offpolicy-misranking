"""Real process-group teardown; opt in to CUDA with SWITCH_TEST_CUDA=0."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid

import pytest

import selection_gate_gpu as base


def wait_until(predicate, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.05)
    assert predicate(), "process test timed out"


def alive(pid):
    try:
        return (Path('/proc') / str(pid) / 'stat').read_text().rsplit(') ', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def compute_pids():
    result = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True, timeout=20)
    return {int(line.strip()) for line in result.splitlines() if line.strip().isdigit()}


@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    if request.param == 'cuda':
        if 'SWITCH_TEST_CUDA' not in os.environ:
            pytest.skip('explicit SWITCH_TEST_CUDA device required')
        return os.environ['SWITCH_TEST_CUDA']
    return ''


def make_worker(tmp_path, device):
    worker = tmp_path / 'worker.py'
    worker.write_text('''
import os, signal, sys, time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
if os.environ.get('CUDA_VISIBLE_DEVICES'):
    import torch
    tensor = torch.ones(32 * 1024 * 1024, device='cuda')
    tensor.mul_(2)
    if 'RANK' in os.environ:
        torch.distributed.init_process_group(backend='nccl')
        torch.distributed.all_reduce(tensor[:1])
    torch.cuda.synchronize()
Path(sys.argv[1]).write_text(str(os.getpid()))
while True:
    time.sleep(.1)
''')
    return [sys.executable, str(worker), str(tmp_path / 'worker.pid')]


@pytest.mark.parametrize('leader_exits', [True, False])
@pytest.mark.parametrize('detached', [True, False])
def test_cleanup_kills_resistant_descendant_after_leader_exit(tmp_path, device, leader_exits, detached):
    command = make_worker(tmp_path, device)
    leader = tmp_path / 'leader.py'
    leader.write_text('''
import os, subprocess, sys, time
from pathlib import Path
child = subprocess.Popen(sys.argv[2:], start_new_session=os.environ.get('DETACH_TEST_CHILD') == '1')
deadline = time.monotonic() + 60
while not Path(sys.argv[-1]).exists():
    if child.poll() is not None or time.monotonic() > deadline:
        raise RuntimeError('child did not start')
    time.sleep(.01)
if sys.argv[1] == 'exit':
    raise SystemExit(7)
time.sleep(60)
''')
    processes = []
    popen = subprocess.Popen
    event_id = uuid.uuid4().hex if detached else None
    try:
        process = popen([sys.executable, str(leader), 'exit' if leader_exits else 'wait', *command],
                        env={**os.environ, 'CUDA_VISIBLE_DEVICES': device, 'DETACH_TEST_CHILD': str(int(detached)),
                             f'OM_SELECTION_COST_{event_id}': '1'}, start_new_session=True)
        processes.append(process)
        wait_until(lambda: (tmp_path / 'worker.pid').exists())
        pid = int((tmp_path / 'worker.pid').read_text())
        if device:
            assert pid in compute_pids(), 'fixture must really hold CUDA memory'
        if leader_exits:
            assert process.wait(timeout=5) == 7
        started = time.monotonic()
        base.terminate(processes, event_id=event_id)
        wait_until(lambda: not alive(pid), timeout=2)
        assert time.monotonic() - started < 8
        if device:
            wait_until(lambda: pid not in compute_pids())
    finally:
        for pgid in base.owned_worker_groups(event_id):
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for process in processes:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)


@pytest.mark.parametrize('kind', ['selection_switch_gpu', 'mopps_comparison_gpu'])
@pytest.mark.parametrize('stop', [signal.SIGINT, signal.SIGTERM])
def test_launcher_stop_closes_cost_and_lock_then_restarts(tmp_path, device, kind, stop):
    command = make_worker(tmp_path, device)
    probe = tmp_path / 'probe.py'
    probe.write_text('''
import os, runpy, sys
from pathlib import Path
import selection_gate_gpu as base
root, kind, worker, visible = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
def evaluate(*args):
    command = [sys.executable, worker, str(root / 'worker.pid')]
    if visible:
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=1',
                   worker, str(root / 'worker.pid')]
    with base.lease(root / '.task.lock'):
        base.meter(root, 'train', 'local-process-test', ledger='deployment', devices=1,
            commands=[(command, visible)], timeout=120)
base.evaluate = evaluate
sys.argv = [kind, 'worker', '--root', str(root), '--arm', 'mopps', '--shard', '0']
sys.argv += ['--phase', 'evaluate'] if kind == 'selection_switch_gpu' else ['--seed', '3', '--step', '25']
runpy.run_path(str(base.ROOT / 'src' / (kind + '.py')), run_name='__main__')
''')
    unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(90)'], start_new_session=True)
    launcher = None
    try:
        for attempt in range(2):
            (tmp_path / 'worker.pid').unlink(missing_ok=True)
            launcher = subprocess.Popen([
                'bash', '-c', 'source "$1"; shift; selection_run_worker "$@"', 'test-launcher',
                str(base.ROOT / 'scripts/_selection_worker.sh'), sys.executable, str(probe), str(tmp_path),
                kind, command[1], device], env={**os.environ, 'OM_NODE_LOCK_HELD': '1'},
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
            wait_until(lambda: (tmp_path / 'worker.pid').exists() or launcher.poll() is not None)
            assert launcher.poll() is None, launcher.communicate(timeout=5)
            pid = int((tmp_path / 'worker.pid').read_text())
            if device:
                assert pid in compute_pids()
            with pytest.raises(BlockingIOError):
                with base.lease(tmp_path / '.task.lock'):
                    pass
            os.killpg(launcher.pid, stop)
            time.sleep(.2)
            if launcher.poll() is None:
                os.killpg(launcher.pid, signal.SIGINT)
                os.killpg(launcher.pid, signal.SIGTERM)
            stdout, stderr = launcher.communicate(timeout=15)
            assert launcher.returncode == (130 if stop == signal.SIGINT else 143), stdout + stderr
            wait_until(lambda: not alive(pid), timeout=2)
            if device:
                wait_until(lambda: pid not in compute_pids())
            assert unrelated.poll() is None
            assert base.cost(tmp_path)['complete']
            assert base.cost(tmp_path)['ledgers']['deployment']['failed_events'] == attempt + 1
            assert base.spent(tmp_path) > 0
            with base.lease(tmp_path / '.task.lock'):
                pass
    finally:
        if launcher is not None:
            if launcher.poll() is None:
                os.killpg(launcher.pid, signal.SIGKILL)
            launcher.communicate(timeout=10)
        if (tmp_path / 'cost.jsonl').exists():
            for line in (tmp_path / 'cost.jsonl').read_text().splitlines():
                event_id = json.loads(line)['event_id']
                for pgid in base.owned_worker_groups(event_id):
                    try:
                        os.killpg(pgid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        if (tmp_path / 'worker.pid').exists():
            pid = int((tmp_path / 'worker.pid').read_text())
            if alive(pid):
                os.kill(pid, signal.SIGKILL)
        unrelated.kill()
        unrelated.wait(timeout=5)


def test_signal_between_spawn_and_registration_still_reaps_and_closes_cost(tmp_path):
    script = '''
import os, signal, subprocess, sys
from pathlib import Path
import selection_gate_gpu as b
original, children = b.subprocess.Popen, []
def interrupted_spawn(*args, **kwargs):
    child = original(*args, **kwargs)
    children.append(child)
    os.kill(os.getpid(), signal.SIGINT)
    return child
b.subprocess.Popen = interrupted_spawn
try:
    b.meter(Path(sys.argv[1]), 'spawn', 'CPU', devices=0, timeout=30,
            commands=[([sys.executable, '-c', 'import time; time.sleep(60)'], '')])
except KeyboardInterrupt:
    assert len(children) == 1 and children[0].poll() is not None
    assert b.cost(Path(sys.argv[1]))['complete']
else:
    raise AssertionError('stop signal was lost')
'''
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path)],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr


def test_exited_leader_pid_is_not_used_without_current_nonce_evidence(monkeypatch):
    leader = SimpleNamespace(pid=99999999, poll=lambda: 0, wait=lambda **kwargs: 0)
    monkeypatch.setattr(base, 'owned_worker_groups', lambda _: set())
    monkeypatch.setattr(base.os, 'killpg', lambda *args: pytest.fail('unowned/recycled group signalled'))
    base.terminate([leader], event_id=uuid.uuid4().hex)
