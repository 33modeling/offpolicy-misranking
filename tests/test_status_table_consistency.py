"""Summary, full matrix and current work must account for the same evidence."""

import fcntl
import re

import pytest

from test_selection_switch_status import core, point, rule, status as switch, published, completed_prefix
from test_selector_pair_status import prepared as prepare_pair, status as pair_status, branch
from test_rloo_status import prepared, status as rloo_status, write
import mbpp_status as display


def meter(directory, phase='train', offset=0):
    core.atomic_json(directory / 'progress.json', {
        'event_id': 'event-1', 'host': 'worker-1', 'phase': phase, 'state': 'running',
        'updated': 10000 + offset, 'seconds': 120, 'timeout': 1000})
    return directory / '.cost.lock'


@pytest.mark.parametrize('experiment', ['mbpp', 'pair', 'rloo'])
@pytest.mark.parametrize('nested', [False, True])
def test_saved_result_never_masks_current_execution(tmp_path, prepared, experiment, nested):
    from test_selection_switch_status import published_curve
    from test_selector_pair_status import published as pair_published
    from test_rloo_status import seal
    if experiment == 'mbpp':
        core.atomic_json(tmp_path / 'switch.json', {'schema': rule.SCHEMA, 'dataset': 'mbpp', 'gate': 'convergence'})
        directory = point(tmp_path) / 'selection_reduced'
        published(directory)
        published_curve(directory)
    elif experiment == 'pair':
        prepare_pair(tmp_path)
        pair_published(tmp_path)
        directory = branch(tmp_path)
    else:
        if nested:
            pytest.skip('RLOO publishes one arm meter, not nested curve meters')
        root, out = prepared
        seal(out, 'random')
        directory = out / 'random'
    meter(directory / 'curve' if nested else directory, phase='curve' if nested else 'evaluate')
    if experiment == 'mbpp':
        data = display.snapshot([tmp_path], now=10000)
    elif experiment == 'pair':
        data = pair_status.dashboard_data(pair_status.snapshot(tmp_path, now=10000))
    else:
        data = rloo_status.snapshot(root, now=10000)
    suite = data['suites'][0]
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    assert display.counts(suite)['states']['RUN'] == 1
    assert display.counts(suite)['done'] == 0
    assert display.counts(suite)['saved_done'] == 1
    text = display.render(data, width=160)
    assert 'CURRENT RUN 1' in text and '결과 저장 후 실행 중 1개' in text
    assert before == {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


@pytest.mark.parametrize('second_state', ['RUN', 'WAIT', 'DONE'])
def test_pair_same_hostname_keeps_each_queue_worker_even_when_newer_one_finished(tmp_path, second_state):
    protocol = prepare_pair(tmp_path)
    for index, state in enumerate(('RUN', second_state)):
        write(tmp_path / f'queue-workers/worker-{index}.json', {
            'host': 'identical-host', 'pid': 42, 'worker': f'worker-{index}',
            'protocol_id': protocol['protocol_id'], 'state': state, 'updated': 9998 + index,
            'task': f'development/s0-t{25 if index == 0 else 50}'})
    snapshot = pair_status.snapshot(tmp_path, now=10000)
    assert len(snapshot['nodes']) == 2
    data = pair_status.dashboard_data(snapshot)
    nodes = display.node_assignments(data)
    assert {node['worker_id'] for node in nodes} == {'worker-0', 'worker-1'}
    assert next(node for node in nodes if node['worker_id'] == 'worker-0')['state'] == 'RUN'
    text = display.render(data, width=160)
    assert 'worker-0' in text and 'worker-1' in text
    assert f"CURRENT RUN {2 if second_state == 'RUN' else 1}" in text


def test_pair_same_hostname_matches_meters_to_worker_state_not_hostname(tmp_path):
    protocol = prepare_pair(tmp_path)
    for index, step in enumerate((25, 50)):
        worker = f'worker-{index}'
        write(tmp_path / f'queue-workers/{worker}.json', {
            'host': 'worker-1', 'pid': 42, 'worker': worker, 'state': 'RUN',
            'protocol_id': protocol['protocol_id'], 'updated': 10000, 'task': f'development/s0-t{step}'})
        meter(tmp_path / f'branches/on_policy/states/s0-t{step}/points/view-{step}/selection_reduced')
    snapshot = pair_status.snapshot(tmp_path, now=10000)
    assert len(snapshot['activity']) == 2
    assert {task['worker_id'] for task in snapshot['activity']} == {'worker-0', 'worker-1'}
    data = pair_status.dashboard_data(snapshot)
    assert len(display.node_assignments(data)) == 2
    assert 'CURRENT RUN 2' in display.render(data, width=160)


def test_pair_same_host_pid_and_task_keep_distinct_worker_codes_in_current(tmp_path):
    protocol = prepare_pair(tmp_path)
    codes = ['a' * 31 + '1', 'a' * 31 + '2']
    for code in codes:
        write(tmp_path / f'queue-workers/{code}.json', {
            'host': 'identical-host', 'pid': 42, 'worker': code, 'state': 'RUN',
            'protocol_id': protocol['protocol_id'], 'updated': 10000, 'task': 'development/s0-t25'})
    data = pair_status.dashboard_data(pair_status.snapshot(tmp_path, now=10000))
    assert len(display.node_assignments(data)) == 2
    for width in (60, 160):
        text = display.render(data, width=width)
        assert 'CURRENT RUN 2' in text
        assert all(code in ''.join(text.split()) for code in codes)


def test_published_mbpp_with_held_task_lease_is_not_done_or_assigned_to_old_host(tmp_path):
    from test_selection_switch_status import published_curve
    core.atomic_json(tmp_path / 'switch.json', {'schema': rule.SCHEMA, 'dataset': 'mbpp', 'gate': 'convergence'})
    directory = point(tmp_path) / 'selection_reduced'
    published(directory)
    published_curve(directory)
    core.atomic_json(directory / 'progress.json', {'host': 'old-host', 'state': 'finished', 'updated': 1})
    with (directory / '.task.lock').open('w') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        data = display.snapshot([tmp_path], now=10000)
        assert display.counts(data['suites'][0])['done'] == 0
        assert display.counts(data['suites'][0])['states']['RUN'] == 1
        assigned = [node for node in display.node_assignments(data) if node['assignments']]
        assert len(assigned) == 1 and assigned[0]['host'] == 'unknown-owner'
        text = display.render(data, width=160)
        assert 'CURRENT RUN 1' in text
        assert '현재 작업자·단계 미확인' in text
        assert '작업자 0개' in text


@pytest.mark.parametrize('phase', ['train', 'fresh-r-candidate', 'fresh-r-validation', 'evaluate', 'curve'])
@pytest.mark.parametrize('offset', [-3600, 3600])
@pytest.mark.parametrize('experiment', ['mbpp', 'pair'])
def test_all_metered_phases_remain_in_summary_matrix_and_current_across_clock_skew(tmp_path, phase, offset, experiment):
    if experiment == 'pair':
        prepare_pair(tmp_path)
        directory = branch(tmp_path)
    else:
        core.atomic_json(tmp_path / 'switch.json', {'schema': rule.SCHEMA, 'dataset': 'mbpp'})
        directory = point(tmp_path) / 'selection_reduced'
    lock = meter(directory, phase, offset)
    with lock.open('w') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
        if experiment == 'pair':
            data = pair_status.dashboard_data(pair_status.snapshot(tmp_path, now=10000))
        else:
            data = display.snapshot([tmp_path], now=10000)
        assert display.counts(data['suites'][0])['states']['RUN'] == 1
        assert len(display.node_assignments(data)[0]['assignments']) >= 1
        text = display.render(data, width=160)
        assert 'CURRENT RUN 1' in text and '실행 작업 1개' in text
        assert before == {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


@pytest.mark.parametrize('relative,phase', [
    ('decisions/s3-t25', 'predict'),
    ('node-preflight/node-1/baseline', 'nccl-check'),
    ('branches/cached/states/s0-t25/points/custom/measurement', 'diagnose'),
    ('branches/on_policy/states/s0-t25/points/view-25/selection_reduced/curve/step-40', 'curve'),
])
def test_pair_current_includes_all_known_paths_when_queue_scan_returns_nothing(tmp_path, monkeypatch, relative, phase):
    prepare_pair(tmp_path)
    meter(tmp_path / relative, phase)
    monkeypatch.setattr(pair_status.gpu, 'pair_progress', lambda *a: [])
    data = pair_status.snapshot(tmp_path, now=10000)
    adapted = pair_status.dashboard_data(data)
    assert len(data['activity']) == 1
    text = pair_status.render(data, width=160)
    assert 'CURRENT RUN 1' in text and 'worker-1' in text
    assert display.node_assignments(adapted)[0]['assignments']
    assert sum(display.counts(s)['planned'] for s in adapted['suites']) == 42


def test_pair_stopped_wait_record_is_not_an_idle_allocated_node(tmp_path):
    protocol = prepare_pair(tmp_path)
    core.atomic_json(tmp_path / 'queue-workers/worker.json', {
        'host': 'stopped-worker', 'state': 'WAIT', 'updated': 10000,
        'protocol_id': protocol['protocol_id'], 'task': 'worker stopped; existing results/checkpoints preserved'})
    data = pair_status.snapshot(tmp_path, now=10000)
    assert data['nodes'][0]['state'] == 'EXITED'
    assert not display.idle_nodes(pair_status.dashboard_data(data))


def test_shared_parent_evaluation_visible_without_inventing_running_branches(tmp_path):
    core.atomic_json(tmp_path / 'switch.json', {'schema': rule.SCHEMA, 'dataset': 'mbpp'})
    meter(point(tmp_path) / 'curve-parent', 'curve')
    data = display.snapshot([tmp_path], now=10000)
    count = display.counts(data['suites'][0])
    assert count['planned'] == 48 and count['states']['RUN'] == 0
    text = display.render(data, width=160)
    assert '분기 RUN 0개 | 공통 단계 RUN 1개 | 실행 작업 1개' in text
    assert 'CURRENT RUN 1' in text and 'Shared evaluation' in text


@pytest.mark.parametrize('experiment', ['mbpp', 'rloo'])
def test_admission_is_current_shared_work_not_an_extra_training_arm(tmp_path, experiment):
    if experiment == 'mbpp':
        core.atomic_json(tmp_path / 'switch.json', {'schema': rule.SCHEMA, 'dataset': 'mbpp'})
    meter(tmp_path / 'node-preflight/node-1/baseline', 'nccl-check')
    data = (display.snapshot([tmp_path], now=10000) if experiment == 'mbpp'
            else rloo_status.snapshot(tmp_path, now=10000))
    text = display.render(data, width=160)
    assert 'CURRENT RUN 1' in text and 'GPU admission' in text
    assert '분기 RUN 0개 | 공통 단계 RUN 1개 | 실행 작업 1개' in text
    assert sum(display.counts(s)['planned'] for s in data['suites']) == (48 if experiment == 'mbpp' else 18)


def mini_suite(root='/test'):
    task = {'seed': 0, 'step': 25, 'arm': 'x', 'kind': 'branch', 'directory': 'branch', 'status': 'DONE'}
    return {'root': root, 'display_label': 'Same label', 'prepared': True, 'tasks': [task],
            'registered_tasks': [(0, 25, 'x')], 'state_points': [(0, 25, 'test')]}


@pytest.mark.parametrize('damage', ['conflict-first', 'conflict-last', 'unverified', 'missing'])
def test_full_matrix_never_claims_done_that_summary_rejects(damage):
    suite = mini_suite()
    task = suite['tasks'][0]
    if damage.startswith('conflict'):
        conflict = {**task, 'status': 'FAILED'}
        suite['tasks'] = [conflict, task] if damage == 'conflict-first' else [task, conflict]
    elif damage == 'unverified':
        task['unverified'] = True
    else:
        suite['tasks'] = []
    data = {'updated': 10000, 'suites': [suite], 'arm_names': {'x': 'X'}}
    count = display.counts(suite)
    assert count['done'] == 0 and count['states']['WAIT'] == count['planned'] == 1
    text = display.render(data, width=160)
    row = next(line for line in text.splitlines() if re.match(r'^0\s*/\s*25\s', line))
    assert 'WAIT' in row and 'DONE' not in row


def test_current_count_does_not_merge_different_roots_with_same_label():
    suites = [mini_suite('/a'), mini_suite('/b')]
    for suite in suites:
        suite['tasks'][0].update(status='RUNNING', host='worker-1', heartbeat_fresh=True)
    data = {'updated': 10000, 'suites': suites, 'arm_names': {'x': 'X'}}
    assert 'CURRENT RUN 2' in display.render(data)
    nodes = display.node_assignments(data)
    assert len(nodes) == 2
    assert sum(len(node['assignments']) for node in nodes) == 2
    assert len({node['work_id'] for node in nodes}) == 2


@pytest.mark.parametrize('experiment', ['mbpp', 'pair', 'rloo'])
def test_same_hostname_different_suffixes_stay_separate_in_all_views(tmp_path, prepared, experiment):
    hosts = ['same-node-g1234', 'same-node-g5678']
    if experiment == 'mbpp':
        core.atomic_json(tmp_path / 'switch.json', {'schema': rule.SCHEMA, 'dataset': 'mbpp'})
        directories = [point(tmp_path) / arm for arm in ('selection_reduced', 'random_full')]
    elif experiment == 'pair':
        prepare_pair(tmp_path)
        directories = [tmp_path / 'branches' / name / 'states/s0-t25/points/view-25/selection_reduced'
                       for name in ('cached', 'on_policy')]
    else:
        root, out = prepared
        directories = [out / arm for arm in ('random', 'fresh_r')]
    for directory, host in zip(directories, hosts):
        meter(directory)
        progress = core.read(directory / 'progress.json')
        core.atomic_json(directory / 'progress.json', {**progress, 'host': host})
    if experiment == 'mbpp':
        data = display.snapshot([tmp_path], now=10000)
    elif experiment == 'pair':
        data = pair_status.dashboard_data(pair_status.snapshot(tmp_path, now=10000))
    else:
        data = rloo_status.snapshot(root, now=10000)
    nodes = display.node_assignments(data)
    assert {node['host'] for node in nodes if node['assignments']} == set(hosts)
    for width in (60, 160):
        text = display.render(data, width=width)
        assert 'CURRENT RUN 2' in text and '실행 작업 2개' in text
        assert all(host in text for host in hosts)


def test_identical_legacy_hostname_and_pid_do_not_hide_sibling_admission_work(tmp_path):
    prepare_pair(tmp_path)
    for attempt in ('first', 'second'):
        directory = tmp_path / 'node-preflight' / attempt / 'baseline'
        meter(directory, phase='nccl-' + attempt)
        progress = core.read(directory / 'progress.json')
        core.atomic_json(directory / 'progress.json', {**progress, 'host': 'same-node', 'pid': 42})
    data = pair_status.dashboard_data(pair_status.snapshot(tmp_path, now=10000))
    text = display.render(data, width=160)
    assert 'CURRENT RUN 2' in text and '공통 단계 RUN 2개' in text
    assert '실제 노드 수 미확인' in text
    assert 'nccl-first' in text and 'nccl-second' in text
    nodes = '\n'.join(display.render_nodes(data, width=160))
    assert 'nccl-first' in nodes and 'nccl-second' in nodes


def test_rloo_live_node_not_hidden_by_newer_terminal_record(prepared):
    root, out = prepared
    meter(out / 'random', 'train', offset=-3600)
    write(out / 'fresh_r/progress.json', {'host': 'worker-1', 'state': 'finished',
          'phase': 'evaluation', 'updated': 10001})
    with (out / 'random/.cost.lock').open('w') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        data = rloo_status.snapshot(root, now=10000)
    assert data['suites'][0]['nodes'][0]['state'] == 'RUN'
    assert display.counts(data['suites'][0])['states']['RUN'] == 1


def test_all_switch_views_render_admission_and_branch_on_same_host(tmp_path):
    import experiments_progress
    core.atomic_json(tmp_path / 'switch.json', {'schema': rule.SCHEMA, 'dataset': 'mbpp'})
    meter(tmp_path / 'node-preflight/node-1/baseline', 'nccl-check')
    meter(point(tmp_path) / 'selection_reduced')
    data = switch.snapshot(tmp_path, now=10000, local_gpus=False)
    assert 'nccl-check' in switch.render(data, all_tasks=True, local_gpus=False)
    assert 'nccl-check' in switch.render_compact(data)
    assert any('nccl-check' in row for row in experiments_progress.render_root(tmp_path, data, width=160, kind='switch'))


@pytest.mark.parametrize('certificate', [{}, {'schema': 'wrong'}, {'schema': rule.SCHEMA, 'seed': 99}])
def test_invalid_prefix_is_neither_done_nor_a_ready_dependency(tmp_path, certificate):
    core.atomic_json(tmp_path / 'switch.json', {'schema': rule.SCHEMA, 'dataset': 'mbpp'})
    core.atomic_json(tmp_path / 'prefixes/seed-0/prefix-25.json', certificate)
    data = switch.snapshot(tmp_path, now=10000, local_gpus=False)
    assert data['prefix_done'] == 0
    assert all(task['status'] != 'READY' for task in data['tasks'] if task['seed'] == 0 and task['kind'] == 'branch')


def test_empty_curve_does_not_count_as_done_even_with_matching_result_hash(tmp_path):
    core.atomic_json(tmp_path / 'switch.json', {'schema': rule.SCHEMA, 'dataset': 'mbpp', 'gate': 'convergence'})
    directory = point(tmp_path) / 'selection_reduced'
    published(directory)
    core.atomic_json(directory / 'curve.json', {'schema': rule.SCHEMA, 'points': {},
                     'result_sha256': core.read(directory / 'result.sha256.json')['sha256']})
    data = display.snapshot([tmp_path], now=10000)
    assert display.counts(data['suites'][0])['done'] == 0
    assert data['suites'][0]['training_published'] == 1


def test_one_bad_cost_row_does_not_hide_other_saved_results_or_current_work(tmp_path):
    core.atomic_json(tmp_path / 'switch.json', {'schema': rule.SCHEMA, 'dataset': 'mbpp'})
    directory = point(tmp_path) / 'selection_reduced'
    published(directory)
    (directory / 'cost.jsonl').write_text('null\n')
    meter(point(tmp_path) / 'random_reduced')
    data = display.snapshot([tmp_path], now=10000)
    assert display.counts(data['suites'][0])['done'] == 1
    assert display.counts(data['suites'][0])['states']['RUN'] == 1
    assert 'cost snapshot unreadable' in display.render(data, width=160)


def test_rloo_mismatched_adapter_receipt_cannot_count_as_done(prepared):
    from test_rloo_status import seal
    root, out = prepared
    seal(out, 'random')
    path = out / 'random/evaluation/shard-0.done.json'
    receipt = core.read(path)
    receipt['binding']['adapter_sha256'] = 'b' * 64
    write(path, receipt)
    data = rloo_status.snapshot(root, now=10000)
    assert display.counts(data['suites'][0])['done'] == 0
