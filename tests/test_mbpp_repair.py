"""Separate MBPP retries never reset or charge the preserved original attempt."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
FAILED = {(2, 50, 'selection_reduced'), (3, 25, 'selection_full'),
          (4, 100, 'random_reduced'), (4, 25, 'random_full'),
          (4, 50, 'selection_reduced')}
STEPS = (25, 50, 100)
DEV_ARMS = ('random_reduced', 'selection_reduced')
TEST_ARMS = ('random_full', 'random_reduced', 'selection_full', 'selection_reduced', 'gated')


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + '\n')


def inventory(root):
    return {str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in root.rglob('*') if path.is_file()}


@pytest.fixture
def repair_module():
    spec = importlib.util.spec_from_file_location('mbpp_repair_tested', ROOT / 'scripts/mbpp_repair.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def saved_source(tmp_path):
    source = tmp_path / 'original-quality'
    protocol = {'dataset': 'mbpp', 'accounting': 'matched', 'gate': 'convergence',
                'selector': 'fresh_r', 'budget_gpu_seconds': 28380,
                'prefix_source': {'root': str(tmp_path / 'original-fresh')},
                'evaluation': {'test': [{'question': 'same holdout', 'answer': 'assert True'}]}}
    save(source / 'switch.json', protocol)
    save(source / 'test.json', protocol['evaluation'])
    reused, failed, dependent = [], [], []
    for seed in range(5):
        for step in STEPS:
            save(source / f'prefixes/seed-{seed}/prefix-{step}.json', {'seed': seed, 'step': step})
            child = source / f'states/s{seed}-t{step}'
            out = child / f'points/view-{step}'
            arms = DEV_ARMS if seed < 3 else TEST_ARMS
            save(child / 'suite.json', {'points': [{'name': f'view-{step}'}]})
            save(child / 'net_protocol.json', {'arms': arms})
            save(out / 'contract.json', {'selected_prefix': {'root': str(source)},
                                       'config': {'seed': seed, 'drift': step}})
            save(out / 'decisions-frozen.json', {'arms': arms, 'frozen': True})
            for arm in arms:
                directory = out / arm
                relative = directory.relative_to(source)
                save(directory / 'decision.json', {'budget_gpu_seconds': 28380, 'arm': arm})
                save(out / 'subsets' / f'subset-{arm}.json', {'original': arm})
                save(out / 'subsets' / f'subset-{arm}.sha256.json', {'original': arm})
                if arm == 'gated':
                    dependent.append(relative)
                    continue
                save(directory / 'cost.jsonl', {'original_cost': 28800 if (seed, step, arm) in FAILED else 28000})
                save(directory / 'execution.json', {'arm': arm})
                save(directory / 'policy/policy_train.json', {'completed_steps': step + 50})
                for name in ('adapter_model.safetensors', 'optimizer.pt'):
                    path = directory / 'policy' / name
                    path.write_bytes(b'original tensor payload')
                if (seed, step, arm) in FAILED:
                    failed.append(relative)
                    save(directory / 'failure.json', {'error': 'preserved prior attempt'})
                    save(directory / 'selector-work/worker.json', {'original': 'do not reuse'})
                    save(directory / 'budget-recovery/result.json', {'canonical_complete': False})
                else:
                    reused.append(relative)
                    save(directory / 'result.json', {'complete': True})
                    save(directory / 'result.sha256.json', {'sha256': 'fixture seal'})
                    save(directory / 'curve.json', {'points': {'before': {}, 'after': {}}})
    assert (len(reused), len(failed), len(dependent)) == (37, 5, 6)
    return source, protocol, reused, failed, dependent


def fixture_validators(monkeypatch, module, protocol):
    monkeypatch.setattr(module, '_validate_source', lambda source: protocol)
    monkeypatch.setattr(module, '_validate_clone', lambda root: None)


def test_repair_preserves_source_and_reuses_only_complete_branches(
        tmp_path, monkeypatch, repair_module, saved_source):
    source, protocol, reused, failed, dependent = saved_source
    fixture_validators(monkeypatch, repair_module, protocol)
    before = inventory(source)
    target = tmp_path / 'repair-quality'
    receipt = repair_module.prepare(source, target)
    assert receipt == json.loads((target / 'repair.json').read_text())
    assert set(repair_module.RERUN_BRANCHES) == {str(path) for path in failed}
    assert set(repair_module.DEPENDENT_BRANCHES) == {str(path) for path in dependent}
    assert set(repair_module.REUSED_BRANCHES) == {str(path) for path in reused}
    assert receipt['snapshot_files']
    assert inventory(source) == before
    assert (target / 'switch.json').read_bytes() == (source / 'switch.json').read_bytes()
    assert not (target / 'switch.json').is_symlink()
    for relative in reused:
        for name in ('result.json', 'result.sha256.json', 'curve.json', 'cost.jsonl', 'policy/policy_train.json'):
            path = target / relative / name
            assert path.read_bytes() == (source / relative / name).read_bytes()
            assert not path.is_symlink()
        for name in ('policy/adapter_model.safetensors', 'policy/optimizer.pt'):
            path = target / relative / name
            assert path.is_symlink()
            assert path.resolve() == (source / relative / name).resolve()
    for relative in failed:
        assert {path.name for path in (target / relative).iterdir()} == {'decision.json'}
        assert (target / relative / 'decision.json').read_bytes() == (source / relative / 'decision.json').read_bytes()
        archived = target / 'original-attempts' / relative
        assert (archived / 'cost.jsonl').read_bytes() == (source / relative / 'cost.jsonl').read_bytes()
        assert (archived / 'failure.json').read_bytes() == (source / relative / 'failure.json').read_bytes()
        assert not (target / relative.parent / 'subsets' / f'subset-{relative.name}.json').exists()
        assert not (target / relative.parent / 'subsets' / f'subset-{relative.name}.sha256.json').exists()
    assert len(list((target / 'states').glob('*/points/*/decisions-frozen.json'))) == 15
    for relative in dependent:
        assert not (target / relative / 'result.json').exists()
    assert not (target / 'model.json').exists()
    assert not list((target / 'states').rglob('gate-frozen.json'))


def test_repair_does_not_import_old_gate_or_live_worker_metadata(tmp_path, monkeypatch, repair_module, saved_source):
    source, protocol, *_ = saved_source
    fixture_validators(monkeypatch, repair_module, protocol)
    excluded = ('model.json', 'gate-fit/failure.json', 'logs/launcher.log',
                'states/s3-t25/gate.json',
                'states/s3-t25/points/view-25/gate-frozen.json')
    for name in excluded:
        save(source / name, {'stale': 'not a repair decision'})
    before = inventory(source)
    target = tmp_path / 'repair-quality'
    repair_module.prepare(source, target)
    assert all(not (target / name).exists() for name in excluded)
    assert inventory(source) == before


def test_repeated_prepare_preserves_existing_retry_work(tmp_path, monkeypatch, repair_module, saved_source):
    source, protocol, _reused, failed, _dependent = saved_source
    fixture_validators(monkeypatch, repair_module, protocol)
    target = tmp_path / 'repair-quality'
    receipt = repair_module.prepare(source, target)
    save(target / failed[0] / 'cost.jsonl', {'retry_cost': 'must not reset'})
    save(target / failed[0] / 'policy/checkpoint-000060/checkpoint_state.json', {'completed_steps': 60})
    before = inventory(target)
    assert repair_module.prepare(source, target) == receipt
    assert inventory(target) == before


def test_tampered_snapshot_is_rejected_without_repairing_in_place(tmp_path, monkeypatch, repair_module, saved_source):
    source, protocol, reused, _failed, _dependent = saved_source
    fixture_validators(monkeypatch, repair_module, protocol)
    target = tmp_path / 'repair-quality'
    repair_module.prepare(source, target)
    save(target / reused[0] / 'result.json', {'complete': True, 'changed': True})
    before = inventory(target)
    with pytest.raises(ValueError):
        repair_module.prepare(source, target)
    assert inventory(target) == before


def test_failed_source_validation_never_creates_a_retry_root(tmp_path, monkeypatch, repair_module, saved_source):
    source = saved_source[0]
    before = inventory(source)
    def invalid(_source):
        raise ValueError('saved publication cannot be certified')
    monkeypatch.setattr(repair_module, '_validate_source', invalid)
    target = tmp_path / 'repair-quality'
    with pytest.raises(ValueError, match='cannot be certified'):
        repair_module.prepare(source, target)
    assert not target.exists()
    assert inventory(source) == before


@pytest.mark.parametrize('target_kind', ['same', 'inside', 'parent'])
def test_retry_root_cannot_overwrite_or_contain_original(tmp_path, monkeypatch, repair_module, saved_source, target_kind):
    source, protocol, *_ = saved_source
    fixture_validators(monkeypatch, repair_module, protocol)
    before = inventory(source)
    target = {'same': source, 'inside': source / 'repair', 'parent': source.parent}[target_kind]
    with pytest.raises((ValueError, FileExistsError)):
        repair_module.prepare(source, target)
    assert inventory(source) == before


@pytest.fixture
def strict_source(saved_source, repair_module, monkeypatch):
    import selection_switch_gpu as switch

    source, protocol, reused, failed, dependent = saved_source
    protocol.update(schema=switch.rule.SCHEMA, steps=list(STEPS), code_hashes={'fixture': 'reviewed'})
    save(source / 'switch.json', protocol)
    monkeypatch.setattr(switch, 'validate_code_hashes', lambda recorded: recorded)
    for seed in range(5):
        for step in STEPS:
            out = source / f'states/s{seed}-t{step}/points/view-{step}'
            contract = json.loads((out / 'contract.json').read_text())
            contract['budget_gpu_seconds'] = protocol['budget_gpu_seconds']
            save(out / 'contract.json', contract)
            arms = DEV_ARMS if seed < 3 else tuple(arm for arm in TEST_ARMS if arm != 'gated')
            save(out / 'decisions-frozen.json', {'decisions': {
                arm: repair_module.digest(out / arm / 'decision.json') for arm in arms}})
    for relative in reused:
        directory = source / relative
        result_hash = repair_module.digest(directory / 'result.json')
        save(directory / 'result.sha256.json', {'sha256': result_hash})
        save(directory / 'curve.json', {'result_sha256': result_hash, 'points': {'before': {}, 'after': {}}})
    return saved_source


def test_actual_source_validation_accepts_exact_37_5_6_without_modifying_evidence(repair_module, strict_source):
    source, protocol, *_ = strict_source
    before = inventory(source)
    assert repair_module._validate_source(source) == protocol
    assert inventory(source) == before


@pytest.mark.parametrize('field,value', [
    ('dataset', 'math500'), ('accounting', 'budget'), ('gate', 'final'),
    ('selector', 'difficulty'), ('steps', [25, 50]), ('schema', 'unknown'),
])
def test_actual_source_validation_rejects_other_scientific_conditions(repair_module, strict_source, field, value):
    source, protocol, *_ = strict_source
    save(source / 'switch.json', {**protocol, field: value})
    before = inventory(source)
    with pytest.raises(ValueError, match='frozen MBPP quality'):
        repair_module._validate_source(source)
    assert inventory(source) == before


@pytest.mark.parametrize('problem', ['completed-rerun', 'dependent-executed', 'changed-decision',
                                   'missing-prefix', 'changed-parent', 'changed-budget',
                                   'missing-publication', 'changed-curve', 'fitted-gate'])
def test_actual_source_validation_rejects_changed_authorized_work(repair_module, strict_source, problem):
    source, _protocol, reused, failed, dependent = strict_source
    if problem == 'completed-rerun':
        directory = source / failed[0]
        save(directory / 'result.json', {'complete': True})
        save(directory / 'result.sha256.json', {'sha256': repair_module.digest(directory / 'result.json')})
        save(directory / 'curve.json', {'result_sha256': repair_module.digest(directory / 'result.json')})
    elif problem == 'dependent-executed':
        save(source / dependent[0] / 'cost.jsonl', {'started': 'must preserve'})
    elif problem == 'changed-decision':
        save(source / failed[0] / 'decision.json', {'budget_gpu_seconds': 999999})
    elif problem == 'missing-prefix':
        (source / 'prefixes/seed-4/prefix-100.json').unlink()
    elif problem in {'changed-parent', 'changed-budget'}:
        contract_path = source / failed[0].parent / 'contract.json'
        contract = json.loads(contract_path.read_text())
        if problem == 'changed-parent':
            contract['selected_prefix']['root'] = str(source.parent / 'different-root')
        else:
            contract['budget_gpu_seconds'] += 1
        save(contract_path, contract)
    elif problem == 'missing-publication':
        (source / reused[0] / 'result.sha256.json').unlink()
    elif problem == 'changed-curve':
        save(source / reused[0] / 'curve.json', {'result_sha256': 'changed'})
    else:
        save(source / 'model.json', {'already': 'fitted'})
    before = inventory(source)
    with pytest.raises(ValueError):
        repair_module._validate_source(source)
    assert inventory(source) == before


def test_active_source_peer_blocks_copy_without_creating_retry_root(tmp_path, repair_module, saved_source):
    source = saved_source[0]
    lock = source / 'states/s2-t50/points/view-50/selection_reduced/.task.lock'
    lock.write_bytes(b'')
    ready = tmp_path / 'peer-ready'
    process = subprocess.Popen([sys.executable, '-c',
        'import fcntl, pathlib, sys, time\n'
        'with open(sys.argv[1], "r+b") as lock:\n'
        '    fcntl.flock(lock, fcntl.LOCK_EX)\n'
        '    pathlib.Path(sys.argv[2]).write_text("locked")\n'
        '    time.sleep(30)\n', str(lock), str(ready)])
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(.01)
        assert ready.exists()
        before = inventory(source)
        target = tmp_path / 'repair-quality'
        with pytest.raises(ValueError, match='still active'):
            repair_module.prepare(source, target)
        assert not target.exists()
        assert inventory(source) == before
        assert process.poll() is None
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_source_read_only_lock_uses_shared_probe_for_nfs(tmp_path, monkeypatch, repair_module):
    source = tmp_path / 'source'
    source.mkdir()
    lock = source / '.task.lock'
    lock.write_bytes(b'preserve lock inode')
    original = repair_module.fcntl.flock
    probes = []

    def probe(handle, operation):
        assert handle.mode == 'rb'
        assert operation == repair_module.fcntl.LOCK_SH | repair_module.fcntl.LOCK_NB
        probes.append(operation)
        return original(handle, operation)

    monkeypatch.setattr(repair_module.fcntl, 'flock', probe)
    before = inventory(source)
    with repair_module.source_locks(source):
        assert probes
    assert inventory(source) == before
