"""Curve observation upgrades preserve every frozen scientific artifact."""

import pytest

import selection_gate as core
import selector_pair_gpu as gpu
from test_selector_pair_gpu import bootstrap_predecessor
from test_selector_pair_lock_migration import file_bytes, frozen_work


def previous_code():
    code = gpu.code_hashes()
    code['src/selector_pair_gpu.py'] = 'f02238e97e9d691e2e13491f33653916ab5a51db82f4c98a72fa299e5b9739bf'
    code['scripts/run_selector_pair.sh'] = '562eea6fbb53a7e024572839860bad02c3af304016ec63e7269a6b054054f12d'
    assert core.fingerprint(code) == gpu.PRE_PAIR_CURVE_PROGRESS_CODE
    return code


@pytest.mark.parametrize('upgraded', [False, True])
def test_curve_upgrade_preserves_frozen_results_and_existing_receipts(tmp_path, monkeypatch, upgraded):
    previous = previous_code()
    recorded = bootstrap_predecessor() if upgraded else previous
    value = frozen_work(tmp_path, recorded)
    if upgraded:
        with monkeypatch.context() as patch:
            patch.setattr(gpu, 'code_hashes', lambda: previous)
            patch.setattr(gpu, 'PRE_SHARED_RUNTIME_CODES', gpu.PRE_SHARED_RUNTIME_CODES - {gpu.PRE_PAIR_CURVE_PROGRESS_CODE})
            gpu.bind_startup_runtime(tmp_path, recorded)
        (tmp_path / 'pair-curve-progress-runtime.json').unlink()
        (tmp_path / 'pair-branch-queue-runtime.json').unlink()
    before = file_bytes(tmp_path)
    assert gpu.compatible_code(recorded)
    assert gpu.manifest(tmp_path) == value
    assert all((tmp_path / path).read_bytes() == data for path, data in before.items())
    receipt = core.read(tmp_path / 'pair-curve-progress-runtime.json')
    assert receipt['runtime_code_hashes'] == gpu.code_hashes()
    after = file_bytes(tmp_path)
    assert gpu.manifest(tmp_path) == value
    assert file_bytes(tmp_path) == after
    assert not gpu.compatible_code({**previous, 'src/selector_pair_gpu.py': 'unreviewed'})
