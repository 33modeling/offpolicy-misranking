"""Launcher activity must not masquerade as a known preflight stage."""

import pytest

import mbpp_status


def data(tmp_path, *, state='LIVE', detail='[gate] curve curve', tasks=()):
    return {'suites': [{'root': str(tmp_path), 'prepared': True, 'tasks': list(tasks),
                       'nodes': [{'host': 'same-host', 'state': state, 'detail': detail,
                                  'last_age': 0}]}]}


@pytest.mark.parametrize('detail', ['[gate] curve curve', '[claimed] s0-t25',
                                   '[dispatch] checking for next task', '[recover-cost] scan'])
def test_launcher_log_without_assignment_is_unknown_not_preflight(tmp_path, detail):
    snapshot = data(tmp_path, detail=detail)
    rendered = '\n'.join(mbpp_status.render_nodes(snapshot, width=240))
    assert 'UNKN' in rendered
    assert '배정 미확인' in rendered
    assert '작업 시작 전 검증 중' not in rendered
    assert '아직 작업 미배정' not in rendered
    assert '배정 없음' not in rendered
    assert mbpp_status.idle_nodes(snapshot) == []


def task(directory, event, kind='branch'):
    return {'kind': kind, 'seed': 0, 'step': 25, 'arm': 'selection_reduced',
            'directory': directory, 'host': 'same-host', 'event_id': event,
            'status': 'RUNNING', 'heartbeat_fresh': True, 'phase': 'curve'}


def test_live_meter_overrides_generic_launcher_label(tmp_path):
    snapshot = data(tmp_path, tasks=[task('states/s0-t25/points/view-25/selection_reduced', 'e1')])
    rendered = '\n'.join(mbpp_status.render_nodes(snapshot, width=240))
    assert 'RUN' in rendered
    assert 'UNKN' not in rendered
    assert '검증 중' not in rendered


@pytest.mark.parametrize('same_event,expected', [(True, 1), (False, 2)])
def test_explicit_event_identity_survives_nested_work_projection(tmp_path, same_event, expected):
    branch = 'states/s0-t25/points/view-25/selection_reduced'
    snapshot = data(tmp_path, tasks=[task(branch, 'e1'),
                                    task(branch + '/curve/step-25', 'e1' if same_event else 'e2', 'phase')])
    assert len(mbpp_status.current_work(snapshot['suites'][0])) == expected
    nodes = mbpp_status.node_assignments(snapshot)
    assert len([node for node in nodes if node['assignments']]) == expected


@pytest.mark.parametrize('state', ['WAIT', 'HOLD'])
def test_explicit_wait_remains_wait(tmp_path, state):
    snapshot = data(tmp_path, state=state)
    assert len(mbpp_status.idle_nodes(snapshot)) == 1
    rendered = '\n'.join(mbpp_status.render_nodes(snapshot, width=240))
    assert 'WAIT' in rendered
    assert '배정 없음' in rendered
