"""Reviewed observation changes never rewrite RLOO experiment evidence."""

import json

import pytest

import rloo_experiment as rloo
from test_rloo_experiment import fixture


def legacy_contract(out):
    c = rloo.ed.read(out / 'experiment.json')
    c['code_hashes']['src/rloo_experiment.py'] = rloo.PRE_QUEUE_OBSERVATION_CODE
    c['code_hashes']['src/selector_pair_gpu.py'] = rloo.PAIR_OBSERVATION_UPGRADE[0]
    rloo.ed.atomic_json(out / 'experiment.json', c)
    return c


def test_prepare_preserves_old_contract_inputs_and_adds_only_reviewed_receipt(tmp_path):
    run, out, evaluation = fixture(tmp_path)
    frozen = legacy_contract(out)
    before = {p: p.read_bytes() for p in out.rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match='runtime receipt'):
        rloo.validate(out)
    assert rloo.prepare(run, out, evaluation, dry=True) == frozen
    assert not (out / 'queue-observation-runtime.json').exists()
    assert rloo.prepare(run, out, evaluation) == frozen
    assert rloo.validate(out)[0] == frozen
    assert all(p.read_bytes() == value for p, value in before.items())
    after = {p: p.read_bytes() for p in out.rglob('*') if p.is_file()}
    assert rloo.prepare(run, out, evaluation) == frozen
    assert after == {p: p.read_bytes() for p in after}


@pytest.mark.parametrize('name', ['src/selector_pair_gpu.py', 'src/train_policy_rloo.py', 'src/rollout.py'])
def test_unreviewed_runtime_or_learner_change_still_rejected(tmp_path, name):
    run, out, evaluation = fixture(tmp_path)
    c = legacy_contract(out)
    c['code_hashes'][name] = 'unreviewed'
    rloo.ed.atomic_json(out / 'experiment.json', c)
    before = (out / 'experiment.json').read_bytes()
    with pytest.raises(ValueError, match='code changed'):
        rloo.prepare(run, out, evaluation)
    assert (out / 'experiment.json').read_bytes() == before
    assert not (out / 'queue-observation-runtime.json').exists()


def test_tampered_runtime_receipt_not_overwritten(tmp_path):
    run, out, evaluation = fixture(tmp_path)
    legacy_contract(out)
    rloo.prepare(run, out, evaluation)
    path = out / 'queue-observation-runtime.json'
    rloo.ed.atomic_json(path, {'tampered': True})
    before = path.read_bytes()
    with pytest.raises(ValueError, match='contract changed'):
        rloo.prepare(run, out, evaluation)
    assert path.read_bytes() == before


@pytest.mark.parametrize('name', rloo.DISPLAY_MODULES)
def test_isolated_display_module_drift_is_recorded_not_rejected(tmp_path, name):
    run, out, evaluation = fixture(tmp_path)
    c = legacy_contract(out)
    c['code_hashes'][name] = 'display-before-status-change'
    rloo.ed.atomic_json(out / 'experiment.json', c)
    frozen = (out / 'experiment.json').read_bytes()
    with pytest.raises(ValueError, match='runtime receipt'):
        rloo.validate(out)
    assert rloo.prepare(run, out, evaluation) == c
    receipt = rloo.ed.read(out / 'queue-observation-runtime.json')
    assert receipt['changes'][name] == {'frozen_sha256': 'display-before-status-change',
                                        'runtime_sha256': rloo.ed.digest(rloo.ROOT / name)}
    assert (out / 'experiment.json').read_bytes() == frozen
    assert rloo.validate(out)[0] == c


def test_display_exception_fails_closed_when_the_module_is_referenced(tmp_path, monkeypatch):
    run, out, evaluation = fixture(tmp_path)
    c = legacy_contract(out)
    c['code_hashes']['src/matrix_status.py'] = 'display-before-status-change'
    rloo.ed.atomic_json(out / 'experiment.json', c)
    monkeypatch.setattr(rloo, 'display_is_isolated', lambda name: False)
    with pytest.raises(ValueError, match='code changed since preparation: src/matrix_status.py'):
        rloo.prepare(run, out, evaluation)
    assert not (out / 'queue-observation-runtime.json').exists()


@pytest.mark.parametrize('reference', ['import matrix_status', 'from matrix_status import main',
                                       "importlib.import_module('matrix_status')"])
def test_display_isolation_scans_source_references(tmp_path, monkeypatch, reference):
    source = tmp_path / 'src'
    source.mkdir()
    (source / 'matrix_status.py').write_text('print(1)\n')
    (source / 'other.py').write_text('x = 1\n')
    monkeypatch.setattr(rloo, 'ROOT', tmp_path)
    assert rloo.display_is_isolated('src/matrix_status.py')
    (source / 'other.py').write_text(reference + '\n')
    assert not rloo.display_is_isolated('src/matrix_status.py')


def test_prepare_refreshes_receipt_after_a_later_reviewed_runtime(tmp_path):
    run, out, evaluation = fixture(tmp_path)
    legacy_contract(out)
    rloo.prepare(run, out, evaluation)
    path = out / 'queue-observation-runtime.json'
    receipt = rloo.ed.read(path)
    stale = json.loads(json.dumps(receipt))
    stale['changes']['src/rloo_experiment.py']['runtime_sha256'] = 'earlier-reviewed-runtime'
    rloo.ed.atomic_json(path, stale)
    frozen = (out / 'experiment.json').read_bytes()
    with pytest.raises(ValueError, match='runtime receipt'):
        rloo.validate(out)
    assert rloo.prepare(run, out, evaluation) == rloo.ed.read(out / 'experiment.json')
    assert rloo.ed.read(path) == receipt
    assert (out / 'experiment.json').read_bytes() == frozen
    rloo.validate(out)


def test_receipt_of_another_contract_is_refused_not_refreshed(tmp_path):
    run, out, evaluation = fixture(tmp_path)
    legacy_contract(out)
    rloo.prepare(run, out, evaluation)
    path = out / 'queue-observation-runtime.json'
    foreign = rloo.ed.read(path)
    foreign['changes']['src/rloo_experiment.py']['frozen_sha256'] = 'another-contract'
    rloo.ed.atomic_json(path, foreign)
    before = path.read_bytes()
    with pytest.raises(ValueError, match='contract changed'):
        rloo.prepare(run, out, evaluation)
    assert path.read_bytes() == before


def test_current_pair_runtime_is_a_reviewed_revision():
    current = rloo.ed.digest(rloo.ROOT / 'src/selector_pair_gpu.py')
    assert current in rloo.PAIR_OBSERVATION_REVIEWED, (
        'src/selector_pair_gpu.py changed; confirm RLOO still never imports it, then pin '
        f'{current} in PAIR_OBSERVATION_REVIEWED so prepared RLOO matrices keep validating')


def test_checkpoint_retention_trainer_revisions_are_reviewed_drift_only_for_exact_pairs():
    import rloo_experiment as rloo
    recorded = {str(path.relative_to(rloo.ROOT)): rloo.ed.digest(path)
                for path in sorted((rloo.ROOT / 'src').glob('*.py'))}
    for name, (frozen, current) in rloo.CHECKPOINT_RETENTION_UPGRADES.items():
        assert recorded[name] == current, name
        recorded[name] = frozen
    changes = rloo.reviewed_code_changes(recorded)
    assert set(changes) == set(rloo.CHECKPOINT_RETENTION_UPGRADES)
    name = 'src/train_policy_grpo.py'
    recorded[name] = '0' * 64
    with pytest.raises(ValueError, match="code changed since preparation: src/train_policy_grpo.py"):
        rloo.reviewed_code_changes(recorded)


def test_rloo_frozen_gain_file_is_byte_identical():
    import rloo_experiment as rloo
    assert rloo.ed.digest(rloo.ROOT / 'src/gain_vs_reliability.py') == \
        'ccd77161107f0acec82f6dc2a21c841957b6573885a8b4de4cc5b4cf17f6d36b'
