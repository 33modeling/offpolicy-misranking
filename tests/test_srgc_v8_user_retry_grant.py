"""User 11-13 '자꾸 에러 난거 먼저 처리해서 다시 시작하게 해야지': tasks BLOCKED by the failure cap / native crashes on
two places get one fresh failure budget, once per task; history kept; never back to a crash place. CPU only."""
import threading

import srgc_v8_nodes as nodes
from test_srgc_v8_retry_debug import prepared


def blocked(tmp_path):
    pool, runtime, task, key, saved = prepared(tmp_path)
    path = pool.root / 'jobs' / (key + '.json'); job = nodes.read(path); job['state'] = 'failed'
    job['attempts'] = [dict(id=f'a{i}', outcome='failed', exit_code=-11, host=h, worker=h + '.1', ended_at=0)
                       for i, h in enumerate(('node-a', 'node-b'))]
    nodes.atomic(path, job)
    state = pool.queue_state()
    state['failure'] = dict(detail='1 task(s) failed; no independent ready/running work remains. Failed attempts/'
                                   'checkpoints are preserved.', task=None, at=1794386986.4768147)
    nodes.atomic(pool.root / 'state.json', state)
    return pool, runtime, task, key, saved, path


def test_blocked_task_restarts_once_elsewhere_and_the_queue_failure_clears(tmp_path):
    pool, runtime, task, key, saved, path = blocked(tmp_path)
    original = saved.read_bytes()
    assert nodes.native_failure_blocks(nodes.read(path))
    pool.initialize(recover=pool.controller_sha256, worker='fresh.1')
    job = nodes.read(path)
    assert job['state'] == 'pending' and job['user_retry_grant']['after_attempt'] == 'a1'
    assert [a['id'] for a in job['attempts']] == ['a0', 'a1'], 'history must be kept'
    assert pool.queue_state()['failure'] is None
    assert pool.queue_state()['failure_history'][-1]['at'] == 1794386986.4768147
    pool.host = 'node-a'; assert not pool.retryable(job), 'never back to a crash place'
    pool.host = 'node-c'; assert pool.retryable(job)
    calls = []; stop = threading.Event(); finished = []
    runtime.done = lambda _: bool(finished)
    def execute(task, matrix, fd, stop_, device_fds, attempt):
        calls.append(pool.host); finished.append(True); stop.set(); return 0
    runtime.execute = execute
    nodes.worker_loop(pool, runtime, 'node-c.9', stop, poll=.001, retry_delay=0)
    job = nodes.read(path)
    assert calls == ['node-c'] and job['state'] == 'done' and saved.read_bytes() == original


def test_grant_is_once_per_task_and_new_failures_block_again(tmp_path):
    pool, runtime, task, key, saved, path = blocked(tmp_path)
    pool.initialize(recover=pool.controller_sha256, worker='fresh.1')
    job = nodes.read(path)
    job['attempts'] += [dict(id=f'b{i}', outcome='failed', exit_code=-11, host=h, worker=h + '.1', ended_at=0)
                        for i, h in enumerate(('node-c', 'node-d'))]
    job['state'] = 'failed'; nodes.atomic(path, job)
    assert nodes.native_failure_blocks(job)
    pool.host = 'node-e'; assert not pool.retryable(job)
    pool.initialize(recover=pool.controller_sha256, worker='fresh.2')
    again = nodes.read(path)
    assert again['state'] == 'failed' and again['user_retry_grant']['after_attempt'] == 'a1', 'not renewed'


def test_one_new_crash_after_the_grant_still_gets_its_one_retry(tmp_path):
    pool, runtime, task, key, saved, path = blocked(tmp_path)
    pool.initialize(recover=pool.controller_sha256, worker='fresh.1')
    job = nodes.read(path)
    job['attempts'].append(dict(id='c0', outcome='failed', exit_code=-11, host='node-c', worker='node-c.1', ended_at=0))
    job['state'] = 'failed'; nodes.atomic(path, job)
    assert nodes.retry_budget_failures(job) == 1 and not nodes.native_failure_blocks(job)
    pool.host = 'node-c'; assert not pool.retryable(job)
    pool.host = 'node-d'; assert pool.retryable(job)


def test_input_or_setup_failures_are_never_cleared(tmp_path):
    pool, runtime, task, key, saved, path = blocked(tmp_path)
    state = pool.queue_state(); state['failure'] = dict(detail='input hash mismatch', task=None, at=1)
    nodes.atomic(pool.root / 'state.json', state)
    assert pool.grant_user_retry('x.1') == []
    assert nodes.read(path)['state'] == 'failed' and pool.queue_state()['failure']['detail'] == 'input hash mismatch'
