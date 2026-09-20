"""The uploaded why set: 37 publications, 5 reviews and 6 gate dependencies."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import mbpp_queue_readiness as readiness


def uploaded_state():
    tasks = []
    blocked = {(2, 50, 'selection_reduced'), (3, 25, 'selection_full'),
               (4, 100, 'random_reduced'), (4, 25, 'random_full'), (4, 50, 'selection_reduced')}
    rule = readiness.status.rule
    for seed in (*rule.DEV_SEEDS, *rule.TEST_SEEDS):
        for step in rule.STEPS:
            tasks.append(dict(kind='prefix', seed=seed, step=step, status='DONE'))
            for arm in rule.DEV_ARMS if seed in rule.DEV_SEEDS else rule.TEST_ARMS:
                tasks.append(dict(kind='branch', seed=seed, step=step, arm=arm,
                    directory=f'states/s{seed}-t{step}/points/view-{step}/{arm}',
                    status='REVIEW' if (seed, step, arm) in blocked else 'WAIT' if arm == 'gated' else 'DONE',
                    reason='saved work requires review'))
    return dict(prepared=True, protocol={'dataset': 'mbpp'}, gate_ready=False, tasks=tasks)


def test_uploaded_blockers_are_not_a_reason_for_another_admission():
    data = uploaded_state()
    assert len(readiness.review_blockers(data)) == 5
    assert sum(t['status'] == 'DONE' and t['kind'] == 'branch' for t in data['tasks']) == 37


@pytest.mark.parametrize('state', ['BUDGET', 'EVAL', 'RESUME', 'READY', 'FAILED', 'STALE', 'SAVING', 'INVALID'])
def test_potential_work_is_not_hidden_by_reviewed_development(state):
    data = uploaded_state()
    next(t for t in data['tasks'] if t.get('seed') == 4 and t.get('status') == 'REVIEW')['status'] = state
    assert readiness.review_blockers(data) is None


@pytest.mark.parametrize('evidence', ['owner_active', 'heartbeat_fresh', 'task_lease_held'])
def test_live_work_prevents_terminal_review_decision(evidence):
    data = uploaded_state()
    data['tasks'][-1][evidence] = True
    assert readiness.review_blockers(data) is None


@pytest.mark.parametrize('change', ['prefix', 'missing', 'duplicate', 'math', 'gate', 'phase', 'unprepared'])
def test_incomplete_or_changed_snapshot_does_not_close_the_queue(change):
    data = uploaded_state()
    if change == 'prefix':
        data['tasks'][0]['status'] = 'WAIT'
    elif change == 'missing':
        data['tasks'].pop()
    elif change == 'duplicate':
        data['tasks'].append(data['tasks'][-1])
    elif change == 'math':
        data['protocol']['dataset'] = 'math500'
    elif change == 'gate':
        data['gate_ready'] = True
    elif change == 'phase':
        data['tasks'].append({'kind': 'phase', 'status': 'RUNNING'})
    else:
        data['prepared'] = False
    assert readiness.review_blockers(data) is None


def test_cli_reports_the_actual_blockers_not_completion(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(readiness.status, 'snapshot', lambda *args, **kw: uploaded_state())
    monkeypatch.setattr(sys, 'argv', ['readiness', '--root', str(tmp_path)])
    assert readiness.main() == 80
    output = capsys.readouterr().out
    assert 'saved=37 review=5 gate-wait=6' in output
    assert 'no GPU admission' in output and 'NOT complete' in output
    assert output.count('[review] states/') == 5


def test_real_status_snapshot_recognizes_uploaded_file_pattern(tmp_path):
    import hashlib
    from test_selection_switch_status import core, completed_prefix, published, published_curve

    core.atomic_json(tmp_path / 'switch.json', {'schema': readiness.status.rule.SCHEMA,
                     'dataset': 'mbpp', 'gate': 'convergence'})
    recovered = {(2, 50, 'selection_reduced'), (4, 25, 'random_full'), (4, 50, 'selection_reduced')}
    for task in uploaded_state()['tasks']:
        if task['kind'] == 'prefix':
            completed_prefix(tmp_path, task['seed'], task['step'])
            continue
        directory = tmp_path / task['directory']
        if task['status'] == 'DONE':
            published(directory)
            published_curve(directory)
        elif (task['seed'], task['step'], task['arm']) in recovered:
            result = directory / 'budget-recovery/result.json'
            core.atomic_json(result, {'schema': 'mbpp-budget-recovery/v1',
                             'evaluation_complete': True, 'canonical_complete': False})
            core.atomic_json(result.with_suffix('.sha256.json'),
                             {'sha256': hashlib.sha256(result.read_bytes()).hexdigest()})
        elif task['status'] == 'REVIEW':
            core.atomic_json(directory / 'policy/grpo_stats.jsonl', {'step': task['step'] + 4})
            core.atomic_json(directory / 'failure.json', {'error': 'unclosed cost event; unknown cost'})
    data = readiness.status.snapshot(tmp_path, local_gpus=False)
    assert data['branch_counts'] == {'DONE': 37, 'REVIEW': 5, 'WAIT': 6}
    assert len(readiness.review_blockers(data)) == 5
