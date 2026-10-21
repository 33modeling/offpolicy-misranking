"""Read-only initial-cache diagnosis, including real shell argument routing."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from srgc_research.dispatch import gemma_initial_audit as audit
from srgc_research.dispatch import gemma_run


@pytest.fixture
def experiment(tmp_path):
    root = tmp_path / 'experiment with spaces'
    (root / 'experiments').mkdir(parents=True)
    (root / 'inputs').mkdir()
    for dataset in ['math', 'mbpp']:
        plan = {'model': audit.MODEL, 'model_revision': audit.REVISION, 'seeds': [5],
                'responses': 8, 'verifier': 'test:verifier', 'max_new_tokens': 2048,
                'attention': 'eager', 'input_pattern': '../inputs/' + dataset + '-seed-{seed}.json',
                'sampling': {'temperature': 1.0, 'top_p': 1.0, 'top_k': 0}}
        (root / 'experiments' / ('gemma4-12b-pt-' + dataset + '.json')).write_text(json.dumps(plan))
        candidate = dataset + '-p0'
        data = {'candidate_ids': [candidate], 'records': {candidate: {'prompt': 'Solve the task. Answer:', 'answer': '6'}},
                'cached_rewards': {candidate: [1, 0, 0, 0, 0, 0, 0, 0]},
                'provenance': {'model': audit.MODEL, 'model_revision': audit.REVISION, 'prompt_format': 'gemma4_pt_plain_text_single_bos'}}
        path = root / 'inputs' / (dataset + '-seed-5.json')
        path.write_text(json.dumps(data))
        directory = path.with_suffix('.cache')
        directory.mkdir()
        (directory / 'protocol.json').write_text(json.dumps({**plan, 'cache_seed': 5}))
        texts = ['Answer: 6', '', 'Answer: 6\nQuestion: other\nAnswer: 9', 'def f():\n    return 6', 'Here is the code:\ndef f():\n    return 6', '```python\ndef f():\n    return 6\n```', 'unknown', 'other']
        receipt = {'id': candidate, 'rewards': data['cached_rewards'][candidate], 'responses': texts}
        (directory / (hashlib.sha256(candidate.encode()).hexdigest() + '.json')).write_text(json.dumps(receipt))
    return root


def file_bytes(root):
    return {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_audit_reports_recorded_success_and_raw_generation_evidence_without_writing(experiment):
    before = file_bytes(experiment)
    result = audit.audit(experiment)
    assert not result['errors']
    assert len(result['datasets']) == 2
    for row in result['datasets']:
        assert row['initial_success_percent'] == row['raw_success_percent'] == 12.5
        assert row['raw_coverage_complete']
        assert row['features']['empty_text'] == 1
        assert row['features']['multiple_answer_markers'] == 1
        assert row['features']['prompt_continuation_marker'] == 1
        assert not row['errors']
    code = result['datasets'][1]
    assert code['features']['python_syntax_error'] > 0
    assert code['features']['python_fence'] == 1
    assert code['features']['has_function'] == 2
    assert file_bytes(experiment) == before
    assert 'not held-out evaluation' in audit.render(result)


def test_missing_raw_receipts_are_unknown_not_zero(experiment):
    for p in (experiment / 'inputs/math-seed-5.cache').glob('*.json'):
        if p.name != 'protocol.json':
            p.unlink()
    row = audit.audit(experiment, 'math')['datasets'][0]
    assert row['initial_success_percent'] == 12.5
    assert row['raw_success_percent'] is None
    assert not row['raw_coverage_complete']
    assert 'unavailable' in audit.render({'datasets': [row], 'measurement': 'initial', 'errors': []})


def test_rescoring_is_opt_in_and_reports_differences(experiment):
    with patch.object(audit, '_rescore', return_value=0.0) as score:
        audit.audit(experiment, 'math')
        score.assert_not_called()
        row = audit.audit(experiment, 'math', rescore=2)['datasets'][0]
    assert len(row['rescore']) == 2
    assert row['errors'] == ['rescore differs: math-p0 response 0']


def test_foreign_model_or_cache_seed_is_rejected(experiment):
    path = experiment / 'inputs/math-seed-5.cache/protocol.json'
    data = json.loads(path.read_text()); data['cache_seed'] = 6; path.write_text(json.dumps(data))
    result = audit.audit(experiment, 'math')
    assert result['errors'] and 'sampling seed' in result['errors'][0]


def test_mismatched_bundle_receipt_is_not_silently_counted(experiment):
    path = experiment / 'inputs/math-seed-5.json'
    data = json.loads(path.read_text()); data['cached_rewards']['math-p0'] = [0] * 8; path.write_text(json.dumps(data))
    row = audit.audit(experiment, 'math')['datasets'][0]
    assert row['errors'] == ['bundle/receipt reward mismatch: math-p0']


def test_entry_and_shell_are_gpu_free_and_preserve_paths(experiment, tmp_path, monkeypatch):
    with patch.object(audit, 'main', return_value=0) as entry:
        assert gemma_run.main(['math', 'audit', '--root', str(experiment)]) == 0
        entry.assert_called_once_with(['math', '--root', str(experiment)])
    monkeypatch.setenv('GEMMA_PYTHON', sys.executable)
    monkeypatch.setenv('PYTHONDONTWRITEBYTECODE', '1')
    trap = tmp_path / 'gpu-import-trap'
    for name in ['torch', 'transformers', 'peft']:
        (trap / name).mkdir(parents=True)
        (trap / name / '__init__.py').write_text("raise RuntimeError('audit imported GPU/model packages')\n")
    monkeypatch.setenv('PYTHONPATH', str(trap))
    before = file_bytes(experiment)
    output = tmp_path / 'audit with spaces.json'
    repo = Path(__file__).resolve().parents[2]
    completed = subprocess.run(['sh', str(repo / 'scripts/run_srgc_gemma4.sh'), 'audit', '--root', str(experiment), '--output', str(output)], cwd=repo, capture_output=True, text=True, timeout=20)
    assert completed.returncode == 0, completed.stderr
    assert 'GEMMA INITIAL-POLICY AUDIT' in completed.stdout
    assert json.loads(output.read_text())['datasets'][0]['initial_success_percent'] == 12.5
    assert file_bytes(experiment) == before


def test_output_cannot_replace_experiment_or_existing_report(experiment, tmp_path):
    inside = experiment / 'audit.json'
    with pytest.raises(SystemExit):
        audit.main(['math', '--root', str(experiment), '--output', str(inside)])
    assert not inside.exists()
    outside = tmp_path / 'already-there.json'
    outside.write_text('keep this')
    with pytest.raises(FileExistsError):
        audit.main(['math', '--root', str(experiment), '--output', str(outside)])
    assert outside.read_text() == 'keep this'


def test_incomplete_cache_is_not_reported_as_a_complete_low_score(experiment):
    path = experiment / 'inputs/math-seed-5.json'
    data = json.loads(path.read_text())
    data['candidate_ids'].append('math-not-generated')
    data['records']['math-not-generated'] = {'prompt': 'unseen', 'answer': '2'}
    path.write_text(json.dumps(data))
    row = audit.audit(experiment, 'math')['datasets'][0]
    assert row['initial_success_percent'] is None
    assert row['raw_success_percent'] == 12.5
    assert not row['raw_coverage_complete']


def test_rescore_errors_are_reported_without_replacing_saved_rewards(experiment):
    before = file_bytes(experiment)
    with patch.object(audit, '_rescore', side_effect=RuntimeError('verifier unavailable')):
        row = audit.audit(experiment, 'math', rescore=1)['datasets'][0]
    assert row['rescore'][0]['recorded_reward'] == 1
    assert row['rescore'][0]['error'] == 'RuntimeError: verifier unavailable'
    assert 'current_reward' not in row['rescore'][0]
    assert row['errors']
    assert file_bytes(experiment) == before


def test_literal_language_line_rescore_is_only_a_diagnostic(experiment):
    path = experiment / 'inputs/mbpp-seed-5.json'
    data = json.loads(path.read_text())
    candidate = data['candidate_ids'][0]
    data['records'][candidate]['answer'] = 'assert add(2,3) == 5'
    data['cached_rewards'][candidate] = [0] * 8
    path.write_text(json.dumps(data))
    receipt_path = path.with_suffix('.cache') / (hashlib.sha256(candidate.encode()).hexdigest() + '.json')
    receipt = json.loads(receipt_path.read_text())
    receipt.update(rewards=[0] * 8, responses=['```\npython\ndef add(a,b):\n    return a+b\n```'] * 8)
    receipt_path.write_text(json.dumps(receipt))
    before = file_bytes(experiment)
    row = audit.audit(experiment, 'mbpp', rescore=1)['datasets'][0]
    assert row['features']['literal_python_language_line'] == 8
    assert row['rescore'][0]['recorded_reward'] == row['rescore'][0]['current_reward'] == 0
    assert row['rescore'][0]['diagnostic_score_without_literal_python_line'] == 1
    assert not row['errors']
    assert file_bytes(experiment) == before
