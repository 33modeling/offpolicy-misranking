"""Reviewed observation changes never rewrite RLOO experiment evidence."""

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
