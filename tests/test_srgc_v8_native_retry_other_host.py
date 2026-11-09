"""v25: a task that died of SIGSEGV is resumed on another host (user 11-10: 'failed로 된거 복구 되는거지?
느린건지 복구가 안된건지 모르겠어 디버깅 계속 해보고 수정해'). CPU metadata only; no actual GPU SIGSEGV."""
import threading

import srgc_v8_nodes as nodes
from test_srgc_v8_retry_debug import prepared


def crash_then(results, pool, calls):
    def execute(task, matrix, fd, stop, device_fds, attempt):
        rc = results[len(calls)]; calls.append((pool.host, rc))
        if rc:
            nodes.atomic(pool.root / 'progress' / attempt['id'] / 'child-failure.json',
                         dict(attempt_id=attempt['id'], exit_code=rc,
                              failures=[dict(rank=1, signal=11, message='CPU synthetic SIGSEGV result')]))
        return rc
    return execute


def test_sigsegv_task_is_resumed_on_another_host_and_the_crash_host_writes_no_queue_failure(tmp_path):
    pool, runtime, task, key, saved = prepared(tmp_path); calls = []; finished = []
    original = saved.read_bytes()
    runtime.done = lambda _: bool(finished)
    runtime.execute = crash_then([-11, 0], pool, calls)
    pool.host = 'node-a'
    rc = nodes.worker_loop(pool, runtime, 'node-a.11', threading.Event(), poll=.001, retry_delay=0)
    job = nodes.read(pool.root / 'jobs' / (key + '.json'))
    assert calls == [('node-a', -11)] and job['state'] == 'failed' and rc == 1   # no second try on node-a
    assert [a['host'] for a in job['attempts']] == ['node-a']
    assert not (pool.queue_state().get('failure') or {}).get('detail'), 'crash host wrote a queue-wide failure'
    assert saved.read_bytes() == original
    stop = threading.Event()
    def execute_b(*args):
        rc = crash_then([-11, 0], pool, calls)(*args); finished.append(rc == 0); stop.set(); return rc
    runtime.execute = execute_b; pool.host = 'node-b'
    nodes.worker_loop(pool, runtime, 'node-b.22', stop, poll=.001, retry_delay=0)
    job = nodes.read(pool.root / 'jobs' / (key + '.json'))
    assert calls == [('node-a', -11), ('node-b', 0)] and job['state'] == 'done'
    assert [a['host'] for a in job['attempts']] == ['node-a', 'node-b']
    assert saved.read_bytes() == original


def test_crash_on_two_different_hosts_stops_automatic_retries_and_status_says_so(tmp_path):
    from test_v8_status import status
    pool, runtime, task, key, saved = prepared(tmp_path)
    path = pool.root / 'jobs' / (key + '.json'); job = nodes.read(path); job['state'] = 'failed'
    job['attempts'] = [dict(id=f'a{i}', outcome='failed', exit_code=-11, host=h, worker=h + '.1', ended_at=0)
                       for i, h in enumerate(('node-a', 'node-b'))]
    nodes.atomic(path, job)
    pool.host = 'node-c'
    assert nodes.native_failure_blocks(job) and not pool.retryable(job) and not pool.retry_later(job)
    row = dict(_log_lines=[], _manifest={}, kind='train', total=275, done=25)
    story = status.failure_story(row, job['attempts'][-1], job, task, runtime.root, pool.queue_state(),
                                 {nodes.DISPATCH_POLICY}, 1000)
    assert story['failure_next'].startswith('BLOCKED: SIGSEGV (exit -11) on 2 different nodes (node-a, node-b)')


def test_old_repeated_relaunches_on_one_node_leave_one_retry_elsewhere_and_status_says_where(tmp_path):
    from test_v8_status import status
    pool, runtime, task, key, saved = prepared(tmp_path)
    path = pool.root / 'jobs' / (key + '.json'); job = nodes.read(path); job['state'] = 'failed'
    job['attempts'] = [dict(id=f'r{i}', outcome='failed', exit_code=-11, worker='h100-a03.77', ended_at=0,
                            controller_policy='node-native-fault-quarantine-v24') for i in range(9)]
    nodes.atomic(path, job)
    assert nodes.retry_budget_failures(job) == 1 and nodes.native_crash_hosts(job) == ['h100-a03']
    pool.host = 'h100-b10'; assert pool.retryable(job)
    pool.host = 'h100-a03'; assert not pool.retryable(job)
    row = dict(_log_lines=[], _manifest={}, kind='train', total=275, done=25, status='failed')
    story = status.failure_story(row, job['attempts'][-1], job, task, runtime.root, pool.queue_state(),
                                 {nodes.DISPATCH_POLICY}, 1000)
    assert story['failure_next'].startswith('retried automatically from its last checkpoint on a node other than '
                                            'h100-a03 (it crashed there): attempt 2/2')
    pool.initialize(recover=pool.controller_sha256, worker='fresh-node')
    assert nodes.read(path) == job, 'the older-controller reset rewrote a native-crash task'


def test_new_crashes_count_one_by_one_even_when_worker_names_repeat(tmp_path):
    """Backstop against V8-136: attempts recorded since v25 never merge, so a fixture or a renamed host
    cannot turn into an unlimited relaunch loop."""
    job = dict(state='failed', attempts=[dict(id=f'n{i}', outcome='failed', exit_code=-11, host='same-name')
                                         for i in range(2)])
    assert nodes.native_crash_count(job) == 2 and nodes.native_failure_blocks(job)


def test_other_gpu_set_of_the_same_host_may_resume_but_the_crashed_set_may_not(tmp_path):
    """The exclusion unit is the GPU set (node_native_identity), so an 8-GPU host's other half still helps."""
    pool = nodes.Pool(tmp_path / 'queue', dict(cohort='fixture')); pool.automatic_retry = True
    job = dict(state='failed', attempts=[dict(id='a', outcome='failed', exit_code=-11, host='h100-a03',
                                             gpus=['0', '1', '2', '3'], worker='h100-a03.9')])
    pool.host, pool.gpus = 'h100-a03', ('0', '1', '2', '3'); assert not pool.retryable(job)
    pool.gpus = ('4', '5', '6', '7'); assert pool.retryable(job)
    assert nodes.native_crash_hosts(job) == ['h100-a03 GPU 0,1,2,3']
