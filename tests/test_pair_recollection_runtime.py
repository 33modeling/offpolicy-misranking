"""Peer-publication recovery upgrades preserve frozen science and receipt bytes."""

import pytest

import selection_gate as core
import selector_pair_gpu as pair
import rloo_experiment as rloo
from test_selector_pair_gpu import bootstrap_predecessor
from test_selector_pair_lock_migration import file_bytes, frozen_work
from test_rloo_experiment import fixture


OLD_PAIR = '0451e210de533ef3c8ec48a322d0f25dff90a91eeb9a73f2be532b86cc5158f4'


@pytest.mark.parametrize('migrated', [False, True])
def test_recollection_upgrade_preserves_the_previous_runtime_chain(tmp_path, monkeypatch, migrated):
    previous = {**pair.code_hashes(), 'src/selector_pair_gpu.py': OLD_PAIR}
    assert core.fingerprint(previous) == pair.PRE_PAIR_RECOLLECTION_CODE
    frozen = frozen_work(tmp_path, bootstrap_predecessor() if migrated else previous)
    if migrated:
        with monkeypatch.context() as patch:
            patch.setattr(pair, 'code_hashes', lambda: previous)
            patch.setattr(pair, 'PRE_SHARED_RUNTIME_CODES',
                          pair.PRE_SHARED_RUNTIME_CODES - {pair.PRE_PAIR_RECOLLECTION_CODE})
            pair.bind_startup_runtime(tmp_path, frozen['code_hashes'])
        assert (tmp_path / 'pair-curve-spawn-runtime.json').is_file()
        assert not (tmp_path / 'pair-recollection-runtime.json').exists()
    before = file_bytes(tmp_path)
    assert pair.manifest(tmp_path) == frozen
    assert pair.manifest(tmp_path) == frozen
    assert all((tmp_path / path).read_bytes() == data for path, data in before.items())
    receipt_path = tmp_path / 'pair-recollection-runtime.json'
    receipt = core.read(receipt_path)
    assert receipt['runtime_code_hashes'] == pair.code_hashes()
    core.atomic_json(receipt_path, {**receipt, 'cost_policy': 'changed'})
    with pytest.raises(ValueError, match='frozen contract changed'):
        pair.manifest(tmp_path)
    assert not pair.compatible_code({**previous, 'src/selector_pair_train.py': 'unreviewed'})


def test_rloo_accepts_only_reviewed_pair_recollection_revision(tmp_path):
    run, out, evaluation = fixture(tmp_path)
    frozen = rloo.ed.read(out / 'experiment.json')
    frozen['code_hashes'].update({'src/selector_pair_gpu.py': OLD_PAIR,
                                'src/rloo_experiment.py': rloo.PRE_PAIR_RECOLLECTION_COMPAT_CODE})
    core.atomic_json(out / 'experiment.json', frozen)
    before = file_bytes(out)
    assert rloo.prepare(run, out, evaluation) == frozen
    assert rloo.validate(out)[0] == frozen
    assert all((out / path).read_bytes() == data for path, data in before.items())
    with pytest.raises(ValueError, match='code changed'):
        rloo.reviewed_code_changes({**frozen['code_hashes'], 'src/selector_pair_gpu.py': 'unreviewed'})
