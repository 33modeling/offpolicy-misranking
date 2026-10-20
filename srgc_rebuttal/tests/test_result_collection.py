"""Exercise the real scoped reporter/exporter with already-collected fixtures."""

import copy
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts import srgc_result_collection as collection
from scripts import srgc_switch_validation_report as validation


def collected(tmp_path):
    root = tmp_path / 'math'
    folder = root / 'seed-5'
    folder.mkdir(parents=True)
    path = folder / 'switch_fixed50-endpoint.json'
    value = {'seed': 5, 'synthetic': True, 'per_question_reward': {'fixture': 0.5}}
    path.write_text(json.dumps(value))
    report = dict(rows=[dict(dataset='math', seed=5, output=str(folder), arms={})],
                  errors=[], warnings=[], summaries=[], arms=[], source_results={str(path): value})
    return root, path, report


@pytest.mark.parametrize('scope', ['timing', 'rules'])
@pytest.mark.parametrize('json_output', [False, True])
def test_scoped_report_cli_saves_its_exact_collected_snapshot(tmp_path, capsys, scope, json_output):
    root, source, report = collected(tmp_path)
    original = source.read_bytes()
    report['scope'] = scope
    # Only acquisition is injected; argparse, reporter, publisher and files are real.
    with patch.object(validation, 'collect', return_value=report) as acquire:
        code = validation.main(['--dataset', 'math', '--scope', scope,
                                *(['--json'] if json_output else [])])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    acquire.assert_called_once_with('math', scope=scope)
    latest = root / 'results' / scope / 'results.json'
    saved = json.loads(latest.read_text())
    assert saved['scope'] == scope
    assert saved['source_results'] == report['source_results']
    bundle = Path(saved['collections'][0]['directory'])
    assert json.loads((bundle / 'raw/seed-5/switch_fixed50-endpoint.json').read_text()) == report['source_results'][str(source)]
    assert source.read_bytes() == original
    assert not (root / 'results/results.json').exists()
    assert not (root / 'results/switch_validation').exists()
    if json_output:
        assert json.loads(captured.out) == saved
    stream = captured.err if json_output else captured.out
    assert f'COLLECTED JSON: {latest}' in stream
    assert f'COLLECTED FILES: {bundle}' in stream


def test_scoped_exports_do_not_replace_each_other_or_p0(tmp_path):
    root, _, report = collected(tmp_path)
    previous = {}
    for scope in ('switch_validation', 'timing', 'rules'):
        collection.publish(dict(copy.deepcopy(report), scope=scope), scope)
        for path, content in previous.items():
            assert path.read_bytes() == content
        path = root / 'results' / scope / 'results.json'
        previous[path] = path.read_bytes()
    assert not (root / 'results/results.json').exists()


@pytest.mark.parametrize('json_output', [False, True])
def test_both_artifact_locations_are_visible_on_the_correct_stream(tmp_path, capsys, json_output):
    _, _, report = collected(tmp_path)
    collection.publish(report, 'support')
    collection.print_paths(report, json_output=json_output)
    captured = capsys.readouterr()
    selected, other = ((captured.err, captured.out) if json_output else (captured.out, captured.err))
    location = report['collections'][0]
    assert f"COLLECTED JSON: {location['report_path']}" in selected
    assert f"COLLECTED FILES: {location['directory']} (1 originals)" in selected
    assert other == ''
