"""11-17: tasks blocked by the 30-min faulthandler-watchdog SIGSEGV restart by themselves on a node whose checkout
carries the fix (no --retry-failed), once per task, history kept, never on a crash place. CPU only."""
import threading

import srgc_v8_nodes as nodes
from test_srgc_v8_retry_debug import prepared


def crash(i, host):
    return dict(id=f'c{i}', outcome='failed', exit_code=-11, host=host, worker=host + '.1', ended_at=0)


def blocked(tmp_path):
    pool, runtime, task, key, saved = prepared(tmp_path)
    path = pool.root / 'jobs' / (key + '.json'); job = nodes.read(path); job['state'] = 'failed'
    job['attempts'] = [crash(0, 'old-a')]
    job['user_retry_grant'] = dict(policy=nodes.USER_RETRY_GRANT, after_attempt='c0',
                                   task_sha256=nodes.digest(nodes.semantic_task(job['task'])))
    job['attempts'] += [crash(i, h) for i, h in enumerate(('node-a', 'node-b', 'node-c'), 1)]
    nodes.atomic(path, job)
    state = pool.queue_state()
    state['failure'] = dict(detail='1 task(s) failed; no independent ready/running work remains. Failed attempts/'
                                   'checkpoints are preserved.', task=None, at=5.0)
    nodes.atomic(pool.root / 'state.json', state)
    return pool, runtime, path


def test_blocked_native_crash_task_restarts_once_without_retry_flag(tmp_path):
    pool, runtime, path = blocked(tmp_path)
    assert nodes.native_failure_blocks(nodes.read(path))
    pool.initialize(recover=pool.controller_sha256, worker='fresh.1')
    job = nodes.read(path)
    assert job['state'] == 'pending' and job['user_retry_grant']['policy'] == nodes.WATCHDOG_FIX_GRANT
    assert job['user_retry_grant']['after_attempt'] == 'c3'
    assert job['retry_grant_history'][0]['policy'] == nodes.USER_RETRY_GRANT
    assert [a['id'] for a in job['attempts']] == ['c0', 'c1', 'c2', 'c3']
    assert pool.queue_state()['failure'] is None
    pool.host = 'node-b'; assert not pool.retryable(job), 'never back to a crash place'
    pool.host = 'node-new'; assert pool.retryable(job)
    job['attempts'] += [crash(i, h) for i, h in enumerate(('n1', 'n2', 'n3'), 4)]; job['state'] = 'failed'
    nodes.atomic(path, job)
    pool.initialize(recover=pool.controller_sha256, worker='fresh.2')
    again = nodes.read(path)
    assert again['state'] == 'failed' and again['user_retry_grant']['after_attempt'] == 'c3', 'once per task'


def test_no_grant_from_a_checkout_without_the_fix(tmp_path, monkeypatch):
    pool, runtime, path = blocked(tmp_path)
    real = nodes.Path.read_text
    monkeypatch.setattr(nodes.Path, 'read_text',
                        lambda self, *a, **k: '' if self.name == 'v8_task_entry.py' else real(self, *a, **k))
    pool.initialize(recover=pool.controller_sha256, worker='fresh.1')
    assert nodes.read(path)['state'] == 'failed'


def test_worker_loop_runs_the_granted_task_on_a_new_node(tmp_path):
    pool, runtime, path = blocked(tmp_path)
    pool.initialize(recover=pool.controller_sha256, worker='fresh.1')
    calls = []; stop = threading.Event(); finished = []
    runtime.done = lambda _: bool(finished)
    def execute(task, matrix, fd, stop_, device_fds, attempt):
        calls.append(pool.host); finished.append(True); stop.set(); return 0
    runtime.execute = execute
    pool.host = 'node-new'
    nodes.worker_loop(pool, runtime, 'node-new.9', stop, poll=.001, retry_delay=0)
    assert calls == ['node-new'] and nodes.read(path)['state'] == 'done'
