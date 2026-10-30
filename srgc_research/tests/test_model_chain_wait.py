"""Exercise node lease contention with real CPU processes, never GPU workers."""
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager

import pytest

from srgc_research.dispatch import model_chain as chain


ENTRY = '''
import sys
from pathlib import Path
from srgc_research.dispatch import model_chain as chain
def drain(order, dataset, initial, environment, update, **kwargs):
    marker = Path(sys.argv[1])
    marker.write_text('one CPU test dispatch\\n')
    update('finished')
    return 0
chain.drain = drain
try:
    raise SystemExit(chain.main(['--start', 'llama31', 'math']))
except chain.Busy as error:
    print(f'MODEL CHAIN busy: {error}', file=sys.stderr, flush=True)
    raise SystemExit(chain.BUSY)
'''


@contextmanager
def contender(tmp_path, *, wait=None, receipt=False):
    work = tmp_path / 'work'
    directory = work / 'srgc-rebuttal/.model-chain'
    key = hashlib.sha256((socket.gethostname() + ':0,1,2,3').encode()).hexdigest()
    lock = directory / 'nodes' / (key + '.lock')
    marker, errors = tmp_path / 'dispatched', tmp_path / 'stderr.txt'
    environment = {**os.environ, 'GROUP_VOLUME': str(tmp_path), 'OM_WORK': str(work),
                   'CUDA_VISIBLE_DEVICES': '0,1,2,3', 'SRGC_MODEL_POLL_SECONDS': '1',
                   'SRGC_MODEL_ORDER': 'llama31,gemma4,qwen35',
                   'PYTHONDONTWRITEBYTECODE': '1'}
    environment.pop('SRGC_STORAGE_ROOT', None)
    environment.pop('SRGC_MODEL_CHAIN_WAIT', None)
    if wait is not None:
        environment['SRGC_MODEL_CHAIN_WAIT'] = wait
    if receipt:
        workers = directory / 'workers'
        workers.mkdir(parents=True)
        now = time.time()
        # Even a FINISHED receipt cannot prove that the real lease is free.
        row = {'worker_id': 'legacy', 'host': socket.gethostname(), 'pid': os.getpid(),
               'started': now - 10, 'heartbeat': now, 'status': 'finished',
               'devices': '0,1,2,3', 'model': 'llama31', 'dataset': 'math'}
        (workers / 'legacy.json').write_text(json.dumps(row))
    checkpoint = work / 'checkpoint.bin'
    work.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b'existing checkpoint must survive')
    holder = chain.lease(lock)
    holder.__enter__()
    inode = lock.stat().st_ino
    stream = errors.open('w')
    process = subprocess.Popen([sys.executable, '-c', ENTRY, str(marker)],
                               cwd=chain.REPO, env=environment,
                               stdout=subprocess.DEVNULL, stderr=stream)
    try:
        yield process, holder, lock, marker, errors, directory, inode
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        holder.__exit__(None, None, None)
        stream.close()
        assert checkpoint.read_bytes() == b'existing checkpoint must survive'


def await_wait_message(process, errors):
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        text = errors.read_text()
        if 'MODEL CHAIN WAIT:' in text:
            return text
        assert process.poll() is None, text
        time.sleep(.02)
    pytest.fail('Contender did not report its wait')


@pytest.mark.parametrize('receipt', [False, True])
def test_busy_node_waits_then_dispatches_only_after_real_release(tmp_path, receipt):
    with contender(tmp_path, receipt=receipt) as (process, holder, lock, marker, errors, directory, inode):
        text = await_wait_message(process, errors)
        if receipt:
            assert f'recorded coordinator pid={os.getpid()}' in text
        else:
            assert 'record unavailable' in text
        assert not marker.exists()
        assert len(list((directory / 'workers').glob('*.json'))) == int(receipt)
        assert lock.stat().st_ino == inode
        holder.__exit__(None, None, None)
        assert process.wait(timeout=8) == 0, errors.read_text()
        assert marker.read_text() == 'one CPU test dispatch\n'
        assert lock.exists() and lock.stat().st_ino == inode
        rows = [json.loads(p.read_text()) for p in (directory / 'workers').glob('*.json')]
        own = next(row for row in rows if row['pid'] == process.pid)
        assert own['status'] == 'finished'


def test_sigterm_while_waiting_does_not_launch_or_publish_a_worker(tmp_path):
    with contender(tmp_path) as (process, holder, lock, marker, errors, directory, inode):
        await_wait_message(process, errors)
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=5) == 143, errors.read_text()
        assert not marker.exists()
        assert not list((directory / 'workers').glob('*.json'))
        with pytest.raises(chain.Busy):
            with chain.lease(lock):
                pytest.fail('Waiting process released another owner\'s lock')
        assert lock.stat().st_ino == inode


def test_explicit_nonblocking_mode_preserves_busy_exit_code(tmp_path):
    with contender(tmp_path, wait='0') as (process, holder, lock, marker, errors, directory, inode):
        assert process.wait(timeout=5) == chain.BUSY
        assert 'MODEL CHAIN busy:' in errors.read_text()
        assert not marker.exists()
        assert not list((directory / 'workers').glob('*.json'))
        assert lock.stat().st_ino == inode


def test_invalid_wait_setting_does_not_launch(tmp_path, monkeypatch):
    monkeypatch.setenv('SRGC_MODEL_CHAIN_WAIT', 'invalid')
    monkeypatch.setattr(chain, 'drain', lambda *a, **kw: pytest.fail('unexpected dispatch'))
    with pytest.raises(SystemExit) as error:
        chain.main(['--start', 'llama31', 'math'])
    assert error.value.code == 2


def test_start_timestamp_is_reset_after_lease_acquisition(tmp_path, monkeypatch):
    clock = [10]
    monkeypatch.setenv('GROUP_VOLUME', str(tmp_path))
    monkeypatch.setenv('OM_WORK', str(tmp_path / 'work'))
    monkeypatch.delenv('SRGC_STORAGE_ROOT', raising=False)
    monkeypatch.delenv('SRGC_MODEL_CHAIN_WAIT', raising=False)
    monkeypatch.setattr(chain.time, 'time', lambda: clock[0])

    @contextmanager
    def acquired(*args, **kwargs):
        clock[0] = 20
        yield None

    monkeypatch.setattr(chain, 'node_lease', acquired)
    monkeypatch.setattr(chain, 'drain', lambda order, dataset, initial, env, update, **kw:
                        update('finished') or 0)
    assert chain.main(['--start', 'llama31', 'math']) == 0
    rows = list((tmp_path / 'work/srgc-rebuttal/.model-chain/workers').glob('*.json'))
    assert len(rows) == 1
    assert json.loads(rows[0].read_text())['started'] == 20
