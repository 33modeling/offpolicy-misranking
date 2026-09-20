"""Curve ownership remains visible across server clock skew without fake heartbeats."""

import fcntl
import pytest

from test_selection_switch_status import completed_prefix, core, point, published, rule, running, status
import mbpp_status


def curve(root, offset, location='selection_reduced/curve'):
    core.atomic_json(root / 'switch.json', {'schema': rule.SCHEMA, 'dataset': 'mbpp', 'gate': 'convergence'})
    directory = point(root) / location
    running(directory, 'remote-curve-node', now=10000 + offset, phase='curve')
    lock = directory / '.cost.lock'
    lock.touch()
    return directory, lock


@pytest.mark.parametrize('offset', [-3600, 3600])
@pytest.mark.parametrize('location', ['selection_reduced/curve', 'curve-parent', 'selection_reduced'])
def test_live_curve_with_clock_skew_is_not_lost_or_claimed_as_fresh(tmp_path, offset, location):
    directory, lock = curve(tmp_path, offset, location)
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    with lock.open('rb') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        data = status.snapshot(tmp_path, now=10000, local_gpus=False)
        task = next(t for t in data['tasks'] if t['directory'] == str(directory.relative_to(tmp_path)))
        assert task['status'] == 'RUNNING' and task['owner_active']
        assert task['heartbeat_fresh'] is False
        assert data['active_nodes'] == 1
        assert next(n for n in data['nodes'] if n['host'] == 'remote-curve-node')['state'] == 'RUN'
        assert mbpp_status.active(task)
        assert '잠금 유지' in mbpp_status.remark(task)
        assert '실행 신호 끊김' not in mbpp_status.remark(task)
        nodes = mbpp_status.node_assignments({'suites': [data]})
        assert next(n for n in nodes if n['host'] == 'remote-curve-node')['assignments']
    assert before == {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    assert status.snapshot(tmp_path, now=10000, local_gpus=False)['active_nodes'] == 0


def test_finished_receipt_wins_even_while_meter_lock_remains_held(tmp_path):
    directory, lock = curve(tmp_path, -3600)
    progress = core.read(directory / 'progress.json')
    progress.update(ledger='reporting', gpus=4, gpu_type='test')
    core.atomic_json(directory / 'progress.json', progress)
    core.atomic_json(directory / 'cost-events' / f'{progress["event_id"]}.json', {
        **progress, 'state': 'finished', 'exit_code': 0, 'seconds': 100,
        'allocated_gpu_seconds': 400, 'time': 7000})
    with lock.open('rb') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert status.snapshot(tmp_path, now=10000, local_gpus=False)['active_nodes'] == 0


@pytest.mark.parametrize('changed', [None, 'event_id', 'host', 'pid', 'state'])
def test_skewed_meter_reread_accepts_heartbeat_not_owner_change(tmp_path, monkeypatch, changed):
    directory, lock = curve(tmp_path, -3600)
    path = directory / 'progress.json'
    original = core.read
    reads = []

    def advancing(candidate):
        value = original(candidate)
        if candidate == path:
            reads.append(1)
            value = {**value, 'updated': 6400 + len(reads), 'seconds': len(reads)}
            if changed and len(reads) > 1:
                value[changed] = 'finished' if changed == 'state' else 'new-owner'
        return value

    monkeypatch.setattr(core, 'read', advancing)
    with lock.open('rb') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        data = status.snapshot(tmp_path, now=10000, local_gpus=False)
        tasks = [t for t in data['tasks'] if t['directory'] == str(directory.relative_to(tmp_path))]
        assert len(reads) >= 2
        assert any(t['owner_active'] for t in tasks) is (changed is None)


@pytest.mark.parametrize('state', ['failed', 'finished'])
def test_terminal_phase_not_resurrected_by_a_lock(tmp_path, state):
    directory, lock = curve(tmp_path, -3600)
    progress = core.read(directory / 'progress.json')
    core.atomic_json(directory / 'progress.json', {**progress, 'state': state})
    with lock.open('rb') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert status.snapshot(tmp_path, now=10000, local_gpus=False)['active_nodes'] == 0


def test_missing_lock_is_not_created_and_unrelated_experiment_unchanged(tmp_path):
    directory, lock = curve(tmp_path, -3600)
    lock.unlink()
    assert status.snapshot(tmp_path, now=10000, local_gpus=False)['active_nodes'] == 0
    assert not lock.exists()
    core.atomic_json(tmp_path / 'switch.json', {'schema': rule.SCHEMA, 'dataset': 'math500'})
    lock.touch()
    with lock.open('rb') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert status.snapshot(tmp_path, now=10000, local_gpus=False)['active_nodes'] == 0


def test_published_result_keeps_completion_state_and_live_owner(tmp_path):
    directory, lock = curve(tmp_path, -3600, 'selection_reduced')
    completed_prefix(tmp_path)
    published(directory)
    with lock.open('rb') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        data = status.snapshot(tmp_path, now=10000, local_gpus=False)
        task = next(t for t in data['tasks'] if t['directory'] == str(directory.relative_to(tmp_path)))
        assert task['status'] == 'EVAL' and task['training_published'] and task['owner_active']
        assert not task['heartbeat_fresh']
        assert not task['retryable']
        assert next(n for n in data['nodes'] if n['host'] == 'remote-curve-node')['state'] == 'RUN'


@pytest.mark.parametrize('offset', [-3600, 0, 3600])
def test_nested_curve_does_not_advertise_parent_branch_for_retry(tmp_path, offset):
    directory, lock = curve(tmp_path, offset)
    branch = directory.parent
    completed_prefix(tmp_path)
    published(branch)
    def task():
        return next(t for t in status.snapshot(tmp_path, now=10000, local_gpus=False)['tasks']
                    if t['directory'] == str(branch.relative_to(tmp_path)))
    with lock.open('rb') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = task()
        assert current['status'] == 'EVAL' and not current['retryable']
        assert current['training_published']
    progress = core.read(directory / 'progress.json')
    core.atomic_json(directory / 'progress.json', {**progress, 'state': 'failed'})
    assert task()['retryable'], 'interrupted evaluation must become resumable after its owner exits'


@pytest.mark.parametrize('published_result', [False, True])
def test_held_task_lease_blocks_dispatch_before_progress_is_published(tmp_path, published_result):
    core.atomic_json(tmp_path / 'switch.json', {'schema': rule.SCHEMA, 'dataset': 'mbpp', 'gate': 'convergence'})
    completed_prefix(tmp_path)
    branch = point(tmp_path) / 'selection_reduced'
    branch.mkdir(parents=True)
    if published_result:
        published(branch)
    lock = branch / '.task.lock'
    lock.touch()
    def task():
        return next(t for t in status.snapshot(tmp_path, local_gpus=False)['tasks']
                    if t['directory'] == str(branch.relative_to(tmp_path)))
    with lock.open('rb') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = task()
        assert current['task_lease_held'] and not current['retryable']
        assert current['status'] == ('EVAL' if published_result else 'WAIT')
    assert task()['status'] == ('EVAL' if published_result else 'READY')
    assert task()['retryable'] is published_result


@pytest.mark.parametrize('progress_state', [None, 'failed', 'finished', 'running'])
@pytest.mark.parametrize('published_result', [False, True])
def test_task_lease_is_visible_without_claiming_a_stale_worker(tmp_path, progress_state, published_result):
    core.atomic_json(tmp_path / 'switch.json', {'schema': rule.SCHEMA, 'dataset': 'mbpp', 'gate': 'convergence'})
    completed_prefix(tmp_path)
    branch = point(tmp_path) / 'selection_reduced'
    branch.mkdir(parents=True)
    if published_result:
        published(branch)
    if progress_state:
        core.atomic_json(branch / 'progress.json', {
            'host': 'old-node', 'worker_id': 'old-worker', 'state': progress_state,
            'phase': 'train', 'updated': 1, 'seconds': 90, 'timeout': 100})
    lock = branch / '.task.lock'
    lock.touch()
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    with lock.open('rb') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        data = mbpp_status.snapshot([tmp_path], now=10000)
        suite = data['suites'][0]
        assert mbpp_status.counts(suite)['states']['RUN'] == 1
        work = mbpp_status.current_work(suite)
        assert len(work) == 1 and work[0][0]['task_lease_held']
        assert mbpp_status.task_progress(tmp_path, work[0][0])[0] == '확인 중'
        assigned = [node for node in mbpp_status.node_assignments(data) if node['assignments']]
        assert len(assigned) == 1
        assert assigned[0]['host'] == 'unknown-owner' and assigned[0].get('worker_id') is None
        assert assigned[0]['work_id'].startswith('work-')
        rendered = mbpp_status.render(data, width=160)
        assert 'CURRENT RUN 1' in rendered and 'WORK ITEMS 1 current' in rendered
        assert '작업 노드 0개' in rendered
        assert '90.0%' not in rendered
    assert before == {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    assert not mbpp_status.current_work(mbpp_status.snapshot([tmp_path], now=10000)['suites'][0])


def test_task_lease_does_not_duplicate_its_live_nested_curve(tmp_path):
    directory, meter_lock = curve(tmp_path, -3600)
    task_lock = directory.parent / '.task.lock'
    task_lock.touch()
    with task_lock.open('rb') as task_owner, meter_lock.open('rb') as meter_owner:
        fcntl.flock(task_owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(meter_owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        data = mbpp_status.snapshot([tmp_path], now=10000)
        work = mbpp_status.current_work(data['suites'][0])
        assert len(work) == 1 and work[0][0]['directory'].endswith('/curve')
        nodes = [node for node in mbpp_status.node_assignments(data) if node['assignments']]
        assert len(nodes) == 1 and nodes[0]['host'] == 'remote-curve-node'
        assert 'CURRENT RUN 1' in mbpp_status.render(data, width=160)
