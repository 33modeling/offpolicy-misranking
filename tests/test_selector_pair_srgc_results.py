import json
import sys

import pytest

import selector_pair_results as results
import selector_pair_srgc as srgc
from test_selector_pair_gpu import fake_study
from test_selector_pair_parallel import study
from test_selector_pair_srgc import current_policy, srgc_study


def test_partial_srgc_decision_is_exported_before_any_continuation(tmp_path, srgc_study, monkeypatch):
    p, _, _, states = srgc_study
    srgc.activate(tmp_path, p)
    identity, entries = states(tmp_path, 3, 25)
    srgc.measure(tmp_path / 'sr-gc/s3-t25', identity, entries['on_policy'], ['0', '1', '2', '3'], p)
    monkeypatch.setattr(results, 'run_report', lambda *args: pytest.fail('no completed Pair to validate'))
    before = {path: path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}
    target = tmp_path / 'results.txt'
    monkeypatch.setattr(sys, 'argv', ['results', '--root', str(tmp_path), '--out', str(target)])
    results.main()
    text = target.read_text()
    data = json.loads(text.split('DATA_JSON\n', 1)[1])
    assert 'SR-GC DECISIONS: partial' in text
    assert 'd_a,d_b,d,selector' in text
    assert data['adaptive_method'] == 'SR-GC'
    assert data['srgc']['decisions'][0]['selector'] == 'cached'
    assert data['srgc']['decisions'][0]['d'] == -1.
    assert len(data['srgc']['pending_states']) == 5
    assert not data['complete'] and data['rows'] == []
    assert data['exporter']['version'] == 'selector-pair-results/v8'
    assert data['srgc_repeated']['sr_is_absorbing'] is True
    assert data['srgc_repeated']['trajectories'][0]['first_sr_step'] == 25
    assert all(path.read_bytes() == saved for path, saved in before.items())


def test_corrupt_srgc_decision_is_not_exported_as_valid(tmp_path, srgc_study):
    p, _, _, states = srgc_study
    srgc.activate(tmp_path, p)
    identity, entries = states(tmp_path, 3, 25)
    directory = tmp_path / 'sr-gc/s3-t25'
    srgc.measure(directory, identity, entries['on_policy'], ['0', '1', '2', '3'], p)
    value = json.loads((directory / 'decision.json').read_text())
    value['selector'] = 'on_policy'
    (directory / 'decision.json').write_text(json.dumps(value))
    data = results.srgc_results(tmp_path)
    assert data['status'] == 'invalid' and data['decisions'] == [] and data['errors']


@pytest.mark.parametrize('damage', ['fifo', 'symlink', 'directory', 'malformed'])
def test_invalid_srgc_metadata_does_not_hang(tmp_path, damage):
    import os
    path = tmp_path / 'pair-sr-gc-runtime.json'
    (tmp_path / 'pair.json').write_text('{}')
    if damage == 'fifo':
        os.mkfifo(path)
    elif damage == 'symlink':
        path.symlink_to('/etc/passwd')
    elif damage == 'directory':
        path.mkdir()
    else:
        path.write_text('{')
    assert results.srgc_results(tmp_path)['status'] == 'invalid'


@pytest.mark.parametrize('isolated', [False, True])
def test_srgc_report_uses_pinned_science_and_never_launches_gpu(tmp_path, monkeypatch, isolated):
    import selector_pair_deploy as deploy
    root, repo, runtime = tmp_path / 'run', tmp_path / 'repo', tmp_path / 'runtime'
    root.mkdir()
    repo.mkdir()
    (root / 'pair-sr-gc-runtime.json').write_text('{}')
    if isolated:
        (repo / '.pair-runtime.json').write_text('{}')
    calls = []
    monkeypatch.setattr(deploy, 'stage_runtime', lambda path: calls.append(path) or runtime)

    class Process:
        returncode = 0
        def communicate(self, timeout):
            return 'ok', ''

    launches = []
    monkeypatch.setattr(results.subprocess, 'Popen', lambda command, **kwargs:
                        launches.append((command, kwargs)) or Process())
    result = results.run_report(root, repo, 60)
    command, options = launches[0]
    assert result.returncode == 0 and 'report_selector_pair_srgc.py' in command[1]
    assert options['cwd'] == (repo if isolated else runtime)
    assert options['env']['CUDA_VISIBLE_DEVICES'] == ''
    assert calls == ([] if isolated else [repo])
