"""Live curve children remain observable after their controller disappears."""

import fcntl

import pytest

import mbpp_status
from test_selection_switch_status import core, point, published, published_curve, rule, running
from test_selector_pair_status import prepared as prepare_pair, published as pair_published, status as pair_status


@pytest.mark.parametrize('experiment', ['mbpp', 'pair'])
@pytest.mark.parametrize('relative', ['curve-parent', 'selection_reduced/curve/step-50'])
@pytest.mark.parametrize('lockname', ['.point.lock', 'shard-0.lock'])
def test_real_curve_lease_is_live_without_parent_meter(tmp_path, experiment, relative, lockname):
    if experiment == 'mbpp':
        core.atomic_json(tmp_path / 'switch.json', {'schema': rule.SCHEMA, 'dataset': 'mbpp', 'gate': 'convergence'})
        out = point(tmp_path)
        published(out / 'selection_reduced')
        published_curve(out / 'selection_reduced')
        snapshot = lambda: mbpp_status.snapshot([tmp_path], now=10000)
    else:
        prepare_pair(tmp_path)
        out = pair_published(tmp_path).parent
        snapshot = lambda: pair_status.dashboard_data(pair_status.snapshot(tmp_path, now=10000))
    directory = out / relative
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / lockname
    lock.touch()
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    assert len(mbpp_status.current_work(snapshot()['suites'][0])) == 0
    with lock.open('rb') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        data = snapshot()
        suite = data['suites'][0]
        assert len(mbpp_status.current_work(suite)) == 1
        tasks = [task for task in suite['tasks'] if task.get('kind') == 'phase' and task.get('task_lease_held')]
        assert len(tasks) == 1
        assert not tasks[0]['host'] and not tasks[0]['worker_id']
        text = mbpp_status.render(data, width=180)
        assert 'CURRENT RUN 1' in text
        if relative != 'curve-parent':
            counts = mbpp_status.counts(suite)
            assert counts['states']['RUN'] == 1 and counts['done'] == 0
        # A live ledger already represents its children; it must not be counted twice.
        ledger = directory if relative == 'curve-parent' else directory.parent
        running(ledger, 'actual-worker', now=10000, phase='curve')
        data = snapshot()
        assert len(mbpp_status.current_work(data['suites'][0])) == 1
        assert not any(task.get('kind') == 'phase' and task.get('task_lease_held')
                       for task in data['suites'][0]['tasks'])
        (ledger / 'progress.json').unlink()
    assert len(mbpp_status.current_work(snapshot()['suites'][0])) == 0
    assert before == {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
