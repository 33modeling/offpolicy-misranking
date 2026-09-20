"""Worker startup errors are not peer locks; frozen runtime upgrades stay exact."""

import errno

import pytest

import selection_gate as core
import selection_gate_gpu as base
import selection_switch_gpu as switch
import selector_pair_gpu as pair
import rloo_experiment as rloo
from test_selection_switch_gpu import convergence_manifest, initial_predecessor, simulated_queue
from test_selector_pair_gpu import bootstrap_predecessor, fake_study
from test_selector_pair import allocation
from test_selector_pair_lock_migration import file_bytes, frozen_work
from test_rloo_experiment import fixture


OLD_SWITCH = '7cd13cca9a1299bd3bd571cfb1e4109b65d5f790b82811a85e7604b9ad034606'
OLD_PAIR = 'd8414a62a7ca805e0218487f64eb0fa923f87c2c59def6c88cda808890a4e081'


def previous_switch():
    value = {**switch.code_hashes(), 'src/selection_switch_gpu.py': OLD_SWITCH}
    assert core.fingerprint(value) == switch.PRE_CURVE_SPAWN_CODE
    return value


def previous_pair():
    value = {**pair.code_hashes(), 'src/selection_switch_gpu.py': OLD_SWITCH,
             'src/selector_pair_gpu.py': OLD_PAIR}
    assert core.fingerprint(value) == pair.PRE_PAIR_CURVE_SPAWN_CODE
    return value


def test_curve_worker_spawn_eagain_is_not_reported_as_a_peer(tmp_path, monkeypatch, capsys):
    out = tmp_path / 'states/s0-t25/points/view-25'
    directory = out / 'selection_reduced'
    core.atomic_json(directory / 'policy/budget_stop.json', {'completed_steps': 125})
    config = {'config': {'drift': 25}, 'scope': {'gpu_type': 'H100'}, 'eval_k': 8}
    calls = []

    def failed_spawn(*args, **kwargs):
        calls.append(1)
        raise BlockingIOError(errno.EAGAIN, 'Resource temporarily unavailable')

    monkeypatch.setattr(base.subprocess, 'Popen', failed_spawn)
    with pytest.raises(RuntimeError, match='curve worker startup/allocation failed') as failure:
        switch.curve_once(tmp_path, convergence_manifest(), out, config,
                          'selection_reduced', {'eval_timeout': 10}, list('0123'), {})
    assert len(calls) == 1
    assert isinstance(failure.value.__cause__, BlockingIOError)
    assert 'held by a peer' not in capsys.readouterr().out
    assert not (directory / 'curve.json').exists()
    charged = out / 'curve-parent'
    summary = base.cost(charged)
    assert summary['complete']
    assert summary['ledgers']['reporting']['gpu_seconds'] >= 0
    assert core.read(charged / 'progress.json')['state'] == 'failed'
    with base.lease(charged / '.point.lock'):
        pass


@pytest.mark.parametrize('migrated', [False, True])
def test_switch_upgrade_preserves_frozen_inputs_and_receipts(tmp_path, monkeypatch, migrated):
    old = previous_switch()
    frozen = {'schema': switch.rule.SCHEMA,
              'code_hashes': initial_predecessor() if migrated else old}
    core.atomic_json(tmp_path / 'switch.json', frozen)
    if migrated:
        with monkeypatch.context() as patch:
            patch.setattr(switch, 'code_hashes', lambda: old)
            switch.manifest(tmp_path)
    evaluation = tmp_path / 'budget-stop-evaluation-runtime.json'
    core.atomic_json(tmp_path / 'mbpp-branch-quarantine-runtime.json', {
        'schema': 'selection-switch-mbpp-branch-quarantine-runtime/v1',
        'switch_sha256': base.digest(tmp_path / 'switch.json'),
        'runtime_code_hashes': old,
        'budget_stop_evaluation_runtime_sha256': base.digest(evaluation) if evaluation.exists() else None,
        'storage_audit_sha256': base.digest(base.ROOT / 'scripts/mbpp_storage_audit.py'),
        'change': 'quarantine incomplete MBPP branch evidence before metering; keep independent branches runnable',
        'cost_policy': 'preserve all saved work, costs, selectors, decisions, targets and frozen budgets; no refunds or parent restart',
    })
    before = file_bytes(tmp_path)
    assert switch.manifest(tmp_path) == frozen
    assert switch.manifest(tmp_path) == frozen
    assert all((tmp_path / path).read_bytes() == data for path, data in before.items())
    assert core.read(tmp_path / 'curve-spawn-runtime.json')['runtime_code_hashes'] == switch.code_hashes()
    with pytest.raises(ValueError, match='scientific code changed'):
        switch.validate_code_hashes({**old, 'src/rollout.py': 'unreviewed'})


@pytest.mark.parametrize('migrated', [False, True])
def test_pair_upgrade_preserves_frozen_inputs_and_receipts(tmp_path, monkeypatch, migrated):
    old = previous_pair()
    frozen = frozen_work(tmp_path, bootstrap_predecessor() if migrated else old)
    if migrated:
        with monkeypatch.context() as patch:
            patch.setattr(pair, 'code_hashes', lambda: old)
            patch.setattr(pair, 'PRE_SHARED_RUNTIME_CODES',
                          pair.PRE_SHARED_RUNTIME_CODES - {pair.PRE_PAIR_CURVE_SPAWN_CODE})
            pair.bind_startup_runtime(tmp_path, frozen['code_hashes'])
    before = file_bytes(tmp_path)
    assert pair.manifest(tmp_path) == frozen
    assert pair.manifest(tmp_path) == frozen
    assert all((tmp_path / path).read_bytes() == data for path, data in before.items())
    assert core.read(tmp_path / 'pair-curve-spawn-runtime.json')['runtime_code_hashes'] == pair.code_hashes()
    assert not pair.compatible_code({**old, 'src/selector_pair_train.py': 'unreviewed'})


def test_rloo_preserves_contract_across_exact_unused_controller_upgrade(tmp_path):
    run, out, evaluation = fixture(tmp_path)
    frozen = rloo.ed.read(out / 'experiment.json')
    frozen['code_hashes'].update({'src/selection_switch_gpu.py': OLD_SWITCH,
                                'src/selector_pair_gpu.py': OLD_PAIR,
                                'src/rloo_experiment.py': rloo.PRE_CURVE_SPAWN_COMPAT_CODE})
    core.atomic_json(out / 'experiment.json', frozen)
    before = file_bytes(out)
    assert rloo.prepare(run, out, evaluation) == frozen
    assert rloo.validate(out)[0] == frozen
    assert all((out / path).read_bytes() == data for path, data in before.items())
    assert rloo.ed.digest(rloo.ROOT / 'src/selection_switch_gpu.py') == rloo.SWITCH_CURVE_SPAWN_UPGRADE[1]
    unreviewed = {**frozen['code_hashes'], 'src/selection_switch_gpu.py': 'unreviewed'}
    with pytest.raises(ValueError, match='code changed'):
        rloo.reviewed_code_changes(unreviewed)


def test_unclosed_reporting_event_blocks_only_its_branch(tmp_path, fake_study, monkeypatch):
    protocol, calls, states = fake_study
    _, entries = states(tmp_path, 0, 25)
    blocked = entries['cached'][1] / 'selection_reduced'
    started, _ = allocation('interrupted-evaluation', 'evaluate', 100, 10, ledger='reporting')
    base.journal(blocked / 'cost.jsonl', started)
    original_bytes = (blocked / 'cost.jsonl').read_bytes()
    execute = pair.execute

    def checked_execute(entry, arm, devices):
        base.spent(entry[1] / arm)
        execute(entry, arm, devices)

    monkeypatch.setattr(pair, 'execute', checked_execute)
    monkeypatch.setattr(pair, 'admit_node', lambda *a: pytest.fail('unknown cost is not a GPU retry'))
    with pytest.raises(pair.IncompletePairRun):
        pair.distributed_stage(tmp_path, protocol, [], 'development', wait_seconds=.01)
    assert len(calls) == 17
    assert len(list(tmp_path.glob('development/*/result.json'))) == 8
    assert not (blocked / 'result.json').exists()
    assert (blocked / 'cost.jsonl').read_bytes() == original_bytes
    assert not base.cost(blocked)['complete']


def test_switch_queue_records_curve_spawn_failure_instead_of_peer_wait(tmp_path, monkeypatch):
    calls, _ = simulated_queue(tmp_path, monkeypatch)
    manifest = switch.manifest(tmp_path)
    manifest.update(convergence_manifest())
    actual_curve = switch.curve_once

    def curve(root, p, out, c, arm, suite, devices, env):
        config = {'config': {'drift': c['step']}, 'scope': {'gpu_type': 'H100'}, 'eval_k': 8}
        core.atomic_json(out / arm / 'policy/budget_stop.json', {'completed_steps': c['step'] + 100})
        actual_curve(root, p, out, config, arm, {'eval_timeout': 10}, devices, env)

    def failed_spawn(*args, **kwargs):
        raise BlockingIOError(errno.EAGAIN, 'Resource temporarily unavailable')

    monkeypatch.setattr(switch, 'curve_once', curve)
    monkeypatch.setattr(base.subprocess, 'Popen', failed_spawn)
    assert switch.work(tmp_path, idle_timeout=0) == 1
    # The six gated arms still wait for the incomplete development barrier.
    assert sum(arm != 'prefix' for _, _, arm in calls) == 42
    failures = list(tmp_path.glob('states/*/points/*/*/failure.json'))
    assert len(failures) == 42
    assert all('curve worker startup/allocation failed' in core.read(path)['error'] for path in failures)
