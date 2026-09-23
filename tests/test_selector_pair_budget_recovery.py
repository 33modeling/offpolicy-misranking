"""Pair recovery evaluates saved work without changing frozen scientific results."""
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip('torch')

import selector_pair_budget_recovery as recovery
import selector_pair_gpu as worker
import selector_pair_results as results
import test_mbpp_budget_recovery as legacy


@pytest.fixture
def evaluator(monkeypatch):
    backend = recovery.backend()
    monkeypatch.setattr(legacy, 'recovery', backend)
    return backend


@pytest.mark.parametrize('checkpoint', [False, True])
def test_saved_pair_evaluation_preserves_training_and_resumes(tmp_path, monkeypatch, evaluator, checkpoint):
    p, directory, _, _ = legacy.fixture(tmp_path, monkeypatch, checkpoint=checkpoint)
    p['dataset'] = 'math'
    before = legacy.bytes_under(directory.parent)
    calls = legacy.fake_evaluation(monkeypatch, directory)
    assert recovery.required(directory)
    result = evaluator.recover(p, directory, list('0123'), {})
    assert result['schema'] == recovery.SCHEMA
    assert result['evaluation_complete'] and not result['canonical_complete']
    assert result['over_budget_gpu_seconds'] == 9
    assert len(calls) == 4
    assert all(path.read_bytes() == value for path, value in before.items())
    assert not (directory / 'result.json').exists()
    assert not (directory / 'policy/budget_stop.json').exists()
    after = legacy.bytes_under(directory.parent)
    assert evaluator.recover(p, directory, list('0123'), {}) == result
    assert len(calls) == 4 and legacy.bytes_under(directory.parent) == after
    import mbpp_budget_recovery
    assert mbpp_budget_recovery.SCHEMA == 'mbpp-budget-recovery/v1'
    assert mbpp_budget_recovery.HERE != evaluator.HERE


@pytest.mark.parametrize('damage', ['adapter_model.safetensors', 'optimizer.pt', 'checkpoint_state.json'])
def test_damaged_saved_pair_never_retrains(tmp_path, monkeypatch, evaluator, damage):
    p, directory, _, policy = legacy.fixture(tmp_path, monkeypatch, checkpoint=True)
    (policy / damage).write_text('corrupt')
    monkeypatch.setattr(evaluator.base, 'meter', lambda *a, **kw: pytest.fail('invalid policy reached GPU'))
    before = legacy.bytes_under(directory.parent)
    with pytest.raises(ValueError, match='no valid saved'):
        evaluator.recover(p, directory, list('0123'), {})
    assert legacy.bytes_under(directory.parent) == before


def test_pair_resumes_only_missing_shards(tmp_path, monkeypatch, evaluator):
    p, directory, _, _ = legacy.fixture(tmp_path, monkeypatch, checkpoint=True)
    evaluator.prepare(p, directory)
    calls = legacy.fake_evaluation(monkeypatch, directory)
    evaluator.evaluate(directory, 0, 0)
    evaluator.evaluate(directory, 0, 2)
    evaluator.recover(p, directory, list('0123'), {})
    assert calls == [(0, 1), (0, 3)]


@pytest.mark.parametrize('remaining,checkpoint,expected', [(1, False, False), (0, False, False),
                                                         (0, True, True), (-1, False, True)])
def test_only_exhausted_unpublished_branches_need_recovery(tmp_path, monkeypatch, evaluator, remaining, checkpoint, expected):
    _, directory, _, _ = legacy.fixture(tmp_path, monkeypatch, remaining=remaining, checkpoint=checkpoint)
    assert recovery.required(directory) is expected
    (directory / 'result.json').write_text('{}')
    assert not recovery.required(directory)


@pytest.mark.parametrize('exhausted', [False, True])
def test_dispatch_and_original_hook_restored(tmp_path, monkeypatch, exhausted):
    branch = tmp_path / 'branches/on_policy'
    out = branch / 'states/s1-t50/points/view-50'
    entry = branch, out, {'config': {'seed': 1, 'drift': 50}}, {}, {}
    calls = []
    original = lambda *a: calls.append('original')
    monkeypatch.setattr(worker, 'execute', original)
    monkeypatch.setattr(worker, 'manifest', lambda _: {})
    monkeypatch.setattr(worker.switch, 'manifest', lambda _: {})
    monkeypatch.setattr(worker, 'environment', lambda _: {})
    monkeypatch.setattr(recovery, 'required', lambda _: exhausted)
    def recover(p, directory, devices, env):
        assert directory == out / 'selection_reduced'
        with pytest.raises(worker.PairLockBusy):
            with worker.pair_lease(directory / '.task.lock'):
                pass
        calls.append('recovery')
        return {'over_budget_gpu_seconds': 5.411}
    monkeypatch.setattr(recovery, 'backend', lambda: SimpleNamespace(recover=recover))
    with recovery.activated(tmp_path):
        failure = worker.attempt_branch(tmp_path, {}, entry, 'selection_reduced', list('0123'))
    assert worker.execute is original
    assert calls == (['recovery'] if exhausted else ['original'])
    if exhausted:
        assert '5.411' in failure['error'] and 'not a budget-compliant' in failure['error']
        assert not (out / 'selection_reduced/result.json').exists()
    else:
        assert failure is None


def test_approved_overrun_resumes_saved_checkpoint_and_records_extra_cost(tmp_path, monkeypatch):
    branch = tmp_path / 'branches/on_policy'
    out = branch / 'states/s1-t50/points/view-50'
    directory = out / 'selection_reduced'
    checkpoint = directory / 'policy/checkpoint-345/checkpoint_state.json'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text('{}')
    (directory / 'decision.json').write_text('{}')
    subset = out / 'subsets/subset-selection_reduced.json'
    subset.parent.mkdir(parents=True)
    subset.write_text('{}')
    worker.core.atomic_json(subset.with_suffix('.sha256.json'), {'sha256': worker.base.digest(subset)})
    config = {'config': {'seed': 1, 'drift': 50}, 'budget_gpu_seconds': 100.,
              'scope': {'gpu_type': 'test-gpu'}}
    entry = branch, out, config, {}, {}
    calls = []
    original = lambda *args: calls.append('execute')
    monkeypatch.setattr(worker, 'execute', original)
    monkeypatch.setattr(worker, 'manifest', lambda _: {})
    monkeypatch.setattr(worker.switch, 'manifest', lambda _: {})
    monkeypatch.setattr(worker, 'environment', lambda _: {})
    monkeypatch.setattr(worker.base, 'spent', lambda _: 101.)
    monkeypatch.setattr(worker.base, 'train_command', lambda *args: ['train'])
    def meter(path, phase, gpu, **kwargs):
        assert path == directory and phase == 'train' and gpu == 'test-gpu'
        assert kwargs['timeout'] == recovery.SUPPLEMENTAL_GPU_SECONDS / worker.base.GPUS
        assert kwargs['ledger'] == 'deployment'
        assert kwargs['commands'] == [(['train'], '0,1,2,3')]
        calls.append('train')
        (directory / 'policy/budget_stop.json').write_text('{}')
    monkeypatch.setattr(worker.base, 'meter', meter)
    with recovery.activated(tmp_path):
        worker.execute(entry, 'selection_reduced', list('0123'))
    assert worker.execute is original
    assert calls == ['train', 'execute']
    receipt = worker.core.read(directory / 'supplemental-allocation.json')
    assert receipt['original_budget_gpu_seconds'] == 100.
    assert receipt['additional_gpu_seconds'] == recovery.SUPPLEMENTAL_GPU_SECONDS


def test_overrun_retries_immediately_after_original_hits_cap(tmp_path, monkeypatch):
    branch = tmp_path / 'branches/on_policy'
    out = branch / 'states/s1-t50/points/view-50'
    directory = out / 'selection_reduced'
    checkpoint = directory / 'policy/checkpoint-345/checkpoint_state.json'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text('{}')
    (directory / 'decision.json').write_text('{}')
    subset = out / 'subsets/subset-selection_reduced.json'
    subset.parent.mkdir(parents=True)
    subset.write_text('{}')
    worker.core.atomic_json(subset.with_suffix('.sha256.json'), {'sha256': worker.base.digest(subset)})
    config = {'config': {'seed': 1, 'drift': 50}, 'budget_gpu_seconds': 100.,
              'scope': {'gpu_type': 'test-gpu'}}
    calls = []
    def original(*args):
        calls.append('execute')
        if len(calls) == 1:
            raise ValueError('branch allocation exhausted before further GPU work')
    monkeypatch.setattr(worker, 'execute', original)
    monkeypatch.setattr(worker, 'manifest', lambda _: {})
    monkeypatch.setattr(worker.switch, 'manifest', lambda _: {})
    monkeypatch.setattr(worker, 'environment', lambda _: {})
    monkeypatch.setattr(worker.base, 'spent', lambda _: 99. if len(calls) == 0 else 101.)
    monkeypatch.setattr(worker.base, 'train_command', lambda *args: ['train'])
    def meter(*args, **kwargs):
        calls.append('train')
        (directory / 'policy/budget_stop.json').write_text('{}')
    monkeypatch.setattr(worker.base, 'meter', meter)
    with recovery.activated(tmp_path):
        worker.execute((branch, out, config, {}, {}), 'selection_reduced', list('0123'))
    assert calls == ['execute', 'train', 'execute']


def recovery_files(root):
    directory = root / 'branches/on_policy/states/s0-t25/points/view-25/selection_reduced/budget-recovery'
    points = [{'step': 25, 'k': 4, 'final': False}, {'step': 60, 'k': 8, 'final': True}]
    costs = {'budget_gpu_seconds': 100., 'used_gpu_seconds': 105., 'over_budget_gpu_seconds': 5.}
    plan = {'schema': recovery.SCHEMA, 'canonical_complete': False, 'arm': 'selection_reduced',
            'start_step': 25, 'completed_steps': 60, 'points': points, **costs}
    worker.core.atomic_json(directory / 'plan.json', plan)
    result = {'schema': recovery.SCHEMA, 'canonical_complete': False, 'evaluation_complete': True,
              'plan_sha256': worker.base.digest(directory / 'plan.json'), **costs,
              'points': [{**point, 'rewards': {'0': .5, '1': .25}} for point in points]}
    worker.core.atomic_json(directory / 'result.json', result)
    worker.core.atomic_json(directory / 'result.sha256.json', {'sha256': worker.base.digest(directory / 'result.json')})
    return directory


@pytest.mark.parametrize('damage', [None, 'seal', 'plan', 'reward', 'step', 'allocation'])
def test_result_export_keeps_recovery_separate_and_checks_bindings(tmp_path, damage):
    directory = recovery_files(tmp_path)
    if damage == 'seal':
        (directory / 'result.sha256.json').write_text('{}')
    elif damage == 'plan':
        (directory / 'plan.json').write_text('{}')
    elif damage:
        value = worker.core.read(directory / 'result.json')
        if damage == 'reward':
            value['points'][-1]['rewards']['0'] = 1.1
        elif damage == 'step':
            value['points'][-1]['step'] = 24
        else:
            value['used_gpu_seconds'] = 0.
        worker.core.atomic_json(directory / 'result.json', value)
        worker.core.atomic_json(directory / 'result.sha256.json', {'sha256': worker.base.digest(directory / 'result.json')})
    before = legacy.bytes_under(tmp_path)
    data = results.budget_recovery_measurements(tmp_path)
    assert results.saved_branch_measurements(tmp_path) == ([], [])
    if damage:
        assert not data['rows'] and len(data['errors']) == 1
    else:
        row, = data['rows']
        assert row['mean_reward'] == .375 and row['updates'] == 35
        assert not row['eligible_for_paired_comparison'] and not row['canonical_complete']
        assert [v['updates'] for v in row['curve_points']] == [0, 35]
        assert '0.375' in results.recovery_table(data)
    assert legacy.bytes_under(tmp_path) == before


def test_status_shows_recovered_evaluation_not_budget_done(tmp_path):
    from test_selector_pair_status import status, prepared
    prepared(tmp_path)
    recovery_files(tmp_path)
    data = status.snapshot(tmp_path)
    task = next(t for t in data['tasks'] if t['seed'] == 0 and t['step'] == 25 and t['name'] == 'on_policy')
    assert task['status'] == 'REVIEW' and task['recovery_evaluation_saved']
    assert not any(t['status'] == 'DONE' for t in data['tasks'])


def test_export_command_includes_recovery_without_paired_completion(tmp_path, monkeypatch):
    recovery_files(tmp_path)
    output = tmp_path / 'pair-results.txt'
    monkeypatch.setattr(sys, 'argv', ['export', '--root', str(tmp_path), '--out', str(output)])
    monkeypatch.setattr(results, 'run_report', lambda *a: pytest.fail('no paired result to validate'))
    results.main()
    text = output.read_text()
    assert 'SAVED-CHECKPOINT BUDGET RECOVERY' in text
    assert 'budget_recovery_measurements' in text


def test_pinned_backend_is_unchanged_from_frozen_science():
    import subprocess
    from selector_pair_deploy import PINNED_COMMIT
    repo = Path(__file__).resolve().parents[1]
    raw = subprocess.check_output(['git', 'show', f'{PINNED_COMMIT}:scripts/mbpp_budget_recovery.py'], cwd=repo)
    assert hashlib.sha256(raw).hexdigest() == recovery.HELPER_SHA256
