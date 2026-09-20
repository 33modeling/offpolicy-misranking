"""Overview counts use the same owner/work identities as the assignment table."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import mbpp_status as status


def task(seed, *, worker=None, host='same-host', **extra):
    return dict(kind='branch', seed=seed, step=25, arm='random_reduced',
                directory=f'states/s{seed}-t25/points/view-25/random_reduced',
                status='RUNNING', heartbeat_fresh=True, host=host, worker_id=worker,
                pid=123, phase='train', **extra)


def suite(root, tasks):
    return dict(root=str(root), tasks=tasks, prepared=True, nodes=[],
                registered_tasks=[(row['seed'], row['step'], row['arm']) for row in tasks],
                state_points=[(row['seed'], row['step'], 'DEV') for row in tasks])


def overview(data):
    return next(line for line in status.render(data, width=120).splitlines()
                if line.startswith('현재 실행:'))


def test_same_legacy_hostname_and_pid_preserve_two_work_items(tmp_path):
    data = dict(updated=1, suites=[suite(tmp_path, [task(0), task(1)])])
    assert '실행 작업 2개 (물리 노드 수 미확인)' in overview(data)
    assert '작업 노드 1개' not in overview(data)
    assert 'WORK ITEMS 2 current' in '\n'.join(status.render_nodes(data, width=120))


def test_same_hostname_two_strong_worker_ids_are_two_workers(tmp_path):
    data = dict(updated=1, suites=[suite(tmp_path, [task(0, worker='uuid-a'), task(1, worker='uuid-b')])])
    assert '작업자 2개' in overview(data)
    assert 'WORKERS 2 current' in '\n'.join(status.render_nodes(data, width=120))


def test_one_strong_worker_with_two_branches_is_not_two_workers(tmp_path):
    data = dict(updated=1, suites=[suite(tmp_path, [task(0, worker='uuid-a'), task(1, worker='uuid-a')])])
    assert '분기 RUN 2개' in overview(data)
    assert '작업자 1개' in overview(data)
    assert 'WORKERS 1 current' in '\n'.join(status.render_nodes(data, width=120))


def test_retained_work_does_not_inflate_selected_overview(tmp_path):
    data = dict(updated=1, suites=[suite(tmp_path / 'current', [task(0, worker='uuid-a')])],
                retained_suites=[suite(tmp_path / 'old', [task(1, worker='uuid-b')])])
    assert '작업자 1개' in overview(data)
    assert 'WORKERS 2 current' in '\n'.join(status.render_nodes(data, width=120))


def test_shared_operation_counts_and_identity_unknown_lease_is_separate(tmp_path):
    known = task('-', worker='admission-owner')
    known.update(kind='phase', directory='node-preflight/worker/1', arm='admission', phase='nccl')
    unknown = task(1, activity_identity_unconfirmed=True, task_lease_held=True)
    unknown.update(heartbeat_fresh=False, owner_active=True, host=None)
    data = dict(updated=1, suites=[suite(tmp_path, [unknown])], operational_root=str(tmp_path),
                operational_tasks=[known])
    text = status.render(data, width=120)
    assert '공통 단계 RUN 1개 | 작업자 1개' in overview(data)
    assert '작업 잠금 확인 1개' in text


def test_single_legacy_owner_does_not_assert_a_physical_node_count(tmp_path):
    data = dict(updated=1, suites=[suite(tmp_path, [task(0)])])
    assert '실행 작업 1개 (물리 노드 수 미확인)' in overview(data)
