"""The uploaded why set: 37 publications, 5 reviews and 6 gate dependencies."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import mbpp_queue_readiness as readiness


def uploaded_state(root):
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
    for task in tasks:
        if task['status'] != 'REVIEW':
            continue
        if (task['seed'], task['step']) in {(2, 50), (4, 25), (4, 50)}:
            task['posthoc_evaluation_saved'] = True
        else:
            policy = root / task['directory'] / 'policy'
            policy.mkdir(parents=True)
            (policy / 'grpo_stats.jsonl').write_text('{"step": 10}\n')
    return dict(root=str(root), prepared=True, protocol={'dataset': 'mbpp'}, gate_ready=False, tasks=tasks)


def test_uploaded_blockers_are_not_a_reason_for_another_admission(tmp_path):
    data = uploaded_state(tmp_path)
    assert len(readiness.review_blockers(data)) == 5
    assert sum(t['status'] == 'DONE' and t['kind'] == 'branch' for t in data['tasks']) == 37


def active_repair_snapshot(root):
    data = uploaded_state(root)
    active = {(2, 50, 'selection_reduced'), (3, 25, 'selection_full'),
              (4, 50, 'selection_reduced')}
    for task in data['tasks']:
        if task['kind'] != 'branch':
            continue
        task.pop('posthoc_evaluation_saved', None)
        key = task['seed'], task['step'], task['arm']
        if key in active:
            task.update(status='EVAL', task_lease_held=True, training_published=True)
        elif task['arm'] != 'gated':
            task['status'] = 'DONE'
    return data


def test_latest_repair_peers_do_not_need_another_admission(tmp_path, monkeypatch, capsys):
    data = active_repair_snapshot(tmp_path)
    assert len(readiness.peer_blockers(data)) == 3
    assert sum(t['kind'] == 'branch' and t['status'] == 'DONE' for t in data['tasks']) == 39
    monkeypatch.setattr(readiness.status, 'snapshot', lambda *args, **kwargs: data)
    monkeypatch.setattr(sys, 'argv', ['readiness', '--root', str(tmp_path)])
    assert readiness.main() == 82
    output = capsys.readouterr().out
    assert '3 peer task leases' in output and 'no GPU admission' in output
    assert '[peer] states/s2-t50/points/view-50/selection_reduced' in output
    assert readiness.review_blockers(data) is None


@pytest.mark.parametrize('change', ['dev_finished', 'lease_released', 'gate_ready', 'missing',
                                   'duplicate', 'prefix', 'notice', 'unprepared', 'math',
                                   'fresh_heartbeat_only', 'unowned_failure', 'missing_field'])
def test_peer_wait_never_hides_new_or_uncertain_work(tmp_path, change):
    data = active_repair_snapshot(tmp_path)
    task = next(t for t in data['tasks'] if t.get('task_lease_held') and t['seed'] == 2)
    if change == 'dev_finished':
        task['status'] = 'DONE'
    elif change == 'lease_released':
        task.pop('task_lease_held')
    elif change == 'fresh_heartbeat_only':
        task.pop('task_lease_held')
        task.update(status='RUNNING', heartbeat_fresh=True)
    elif change == 'gate_ready':
        data['gate_ready'] = True
    elif change == 'missing':
        data['tasks'].pop()
    elif change == 'duplicate':
        data['tasks'].append(data['tasks'][-1])
    elif change == 'missing_field':
        task.pop('seed')
    elif change == 'prefix':
        data['tasks'][0]['status'] = 'WAIT'
    elif change == 'notice':
        data['notices'] = [{'error': 'unreadable manifest'}]
    elif change == 'unprepared':
        data['prepared'] = False
    elif change == 'math':
        data['protocol']['dataset'] = 'math500'
    else:
        next(t for t in data['tasks'] if t['kind'] == 'branch' and t['status'] == 'DONE')['status'] = 'FAILED'
    assert readiness.peer_blockers(data) is None


def test_real_repair_task_locks_defer_admission_until_development_finishes(tmp_path):
    import contextlib
    import fcntl
    from test_selection_switch_status import core, completed_prefix, published, published_curve

    data = active_repair_snapshot(tmp_path)
    core.atomic_json(tmp_path / 'switch.json', {'schema': readiness.status.rule.SCHEMA,
                     'dataset': 'mbpp', 'gate': 'convergence'})
    with contextlib.ExitStack() as locks:
        for task in data['tasks']:
            if task['kind'] == 'prefix':
                completed_prefix(tmp_path, task['seed'], task['step'])
                continue
            directory = tmp_path / task['directory']
            if task['arm'] == 'gated':
                continue
            published(directory)
            if task['status'] == 'DONE':
                published_curve(directory)
            else:
                handle = locks.enter_context((directory / '.task.lock').open('a+'))
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = {path: (path.read_bytes(), path.stat().st_mtime_ns)
                  for path in tmp_path.rglob('*') if path.is_file()}
        live = readiness.status.snapshot(tmp_path, local_gpus=False)
        assert len(readiness.peer_blockers(live)) == 3
        assert before == {path: (path.read_bytes(), path.stat().st_mtime_ns)
                          for path in tmp_path.rglob('*') if path.is_file()}
        published_curve(tmp_path / 'states/s2-t50/points/view-50/selection_reduced')
        assert readiness.peer_blockers(readiness.status.snapshot(tmp_path, local_gpus=False)) is None
    assert readiness.peer_blockers(readiness.status.snapshot(tmp_path, local_gpus=False)) is None


@pytest.mark.parametrize('state', ['BUDGET', 'EVAL', 'RESUME', 'READY', 'FAILED', 'STALE', 'SAVING', 'INVALID'])
def test_potential_work_is_not_hidden_by_reviewed_development(state, tmp_path):
    data = uploaded_state(tmp_path)
    next(t for t in data['tasks'] if t.get('seed') == 4 and t.get('status') == 'REVIEW')['status'] = state
    assert readiness.review_blockers(data) is None


@pytest.mark.parametrize('evidence', ['owner_active', 'heartbeat_fresh', 'task_lease_held'])
def test_live_work_prevents_terminal_review_decision(evidence, tmp_path):
    data = uploaded_state(tmp_path)
    data['tasks'][-1][evidence] = True
    assert readiness.review_blockers(data) is None


@pytest.mark.parametrize('change', ['prefix', 'missing', 'duplicate', 'math', 'gate', 'phase', 'unprepared'])
def test_incomplete_or_changed_snapshot_does_not_close_the_queue(change, tmp_path):
    data = uploaded_state(tmp_path)
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
    data = uploaded_state(tmp_path)
    monkeypatch.setattr(readiness.status, 'snapshot', lambda *args, **kw: data)
    monkeypatch.setattr(sys, 'argv', ['readiness', '--root', str(tmp_path)])
    assert readiness.main() == 80
    output = capsys.readouterr().out
    assert 'saved=37 review=5 gate-wait=6' in output
    assert 'no GPU admission' in output and 'NOT complete' in output
    assert output.count('[review] states/') == 5


@pytest.mark.parametrize('artifact', ['policy_train.json', 'budget_stop.json'])
def test_missing_stop_repair_and_parent_policy_evaluation_are_not_terminal(tmp_path, artifact):
    data = uploaded_state(tmp_path)
    task = next(t for t in data['tasks'] if t['status'] == 'REVIEW' and not t.get('posthoc_evaluation_saved'))
    policy = tmp_path / task['directory'] / 'policy'
    (policy / artifact).write_text('{"use_parent_policy": true}\n')
    assert readiness.review_blockers(data) is None


def test_review_label_without_proven_quarantine_is_not_terminal(tmp_path):
    data = uploaded_state(tmp_path)
    task = next(t for t in data['tasks'] if t['status'] == 'REVIEW' and not t.get('posthoc_evaluation_saved'))
    (tmp_path / task['directory'] / 'policy/grpo_stats.jsonl').unlink()
    task['reason'] = 'saved state path unresolved'
    assert readiness.review_blockers(data) is None


def test_snapshot_without_root_cannot_prove_checkpoint_quarantine(tmp_path):
    data = uploaded_state(tmp_path)
    data.pop('root')
    assert readiness.review_blockers(data) is None


def saved_uploaded_snapshot(tmp_path):
    import hashlib
    from test_selection_switch_status import core, completed_prefix, published, published_curve

    core.atomic_json(tmp_path / 'switch.json', {'schema': readiness.status.rule.SCHEMA,
                     'dataset': 'mbpp', 'gate': 'convergence'})
    recovered = {(2, 50, 'selection_reduced'), (4, 25, 'random_full'), (4, 50, 'selection_reduced')}
    for task in uploaded_state(tmp_path)['tasks']:
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
    return readiness.status.snapshot(tmp_path, local_gpus=False)


def test_real_status_snapshot_recognizes_uploaded_file_pattern(tmp_path):
    data = saved_uploaded_snapshot(tmp_path)
    assert data['branch_counts'] == {'DONE': 37, 'REVIEW': 5, 'WAIT': 6}
    assert len(readiness.review_blockers(data)) == 5


@pytest.mark.parametrize('seed,step,arm', [
    (2, 50, 'selection_reduced'), (4, 25, 'random_full'), (4, 50, 'selection_reduced'),
])
@pytest.mark.parametrize('recovery', ['absent', 'unsealed', 'changed'])
@pytest.mark.parametrize('stop_saved', [False, True])
def test_uploaded_recoverable_policy_is_not_closed_before_publication(
        tmp_path, monkeypatch, seed, step, arm, recovery, stop_saved):
    from test_selection_switch_status import core

    saved_uploaded_snapshot(tmp_path)
    directory = tmp_path / f'states/s{seed}-t{step}/points/view-{step}/{arm}'
    result = directory / 'budget-recovery/result.json'
    if recovery == 'absent':
        result.unlink()
    elif recovery == 'unsealed':
        result.with_suffix('.sha256.json').unlink()
    else:
        core.atomic_json(result, {'schema': 'mbpp-budget-recovery/v1',
                         'evaluation_complete': True, 'canonical_complete': False,
                         'changed_after_seal': True})
    policy = directory / 'policy'
    policy.mkdir()
    for filename in ('adapter_config.json', 'adapter_model.safetensors',
                     'optimizer.pt', 'grpo_stats.jsonl'):
        (policy / filename).write_bytes(b'fixture payload\n')
    core.atomic_json(policy / 'policy_train.json', {
        'schema': 'offpolicy-rlvr-policy/v1', 'start_step': step,
        'completed_steps': step + 1, 'adapter_sha256': 'a' * 64,
        'optimizer_sha256': 'b' * 64, 'grpo_stats_sha256': 'c' * 64})
    if stop_saved:
        core.atomic_json(policy / 'budget_stop.json', {
            'use_parent_policy': False, 'completed_steps': step + 1,
            'requested_target_steps': step + 10, 'stop_reason': 'budget_exhausted'})
    core.atomic_json(directory / 'failure.json', {
        'error': 'branch allocation exhausted before further GPU work; saved work preserved'})

    data = readiness.status.snapshot(tmp_path, local_gpus=False)
    task = next(t for t in data['tasks'] if t.get('directory') == str(directory.relative_to(tmp_path)))
    assert task['status'] == ('EVAL' if stop_saved else 'REVIEW')
    assert not task.get('posthoc_evaluation_saved')
    assert readiness.review_blockers(data) is None
    monkeypatch.setattr(readiness.status, 'snapshot', lambda *args, **kwargs: data)
    monkeypatch.setattr(sys, 'argv', ['readiness', '--root', str(tmp_path)])
    assert readiness.main() == 0
