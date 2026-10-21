"""Read-only evidence distinguishes frozen rewards from format diagnostics."""

import hashlib
import json
import subprocess
import sys
from unittest.mock import patch

import pytest

from srgc_rebuttal.plan import input_path, load_plan
from srgc_rebuttal.runtime import atomic_json
from srgc_research.dispatch import instruct_initial_audit as audit
from srgc_research.tests.test_model_results import make_cohort, saved


@pytest.fixture(params=['qwen35', 'llama31'])
def cohort(request, tmp_path, monkeypatch):
    result = make_cohort(request.param, tmp_path)
    monkeypatch.setenv('GROUP_VOLUME', result.env['GROUP_VOLUME'])
    monkeypatch.setenv('OM_WORK', result.env['OM_WORK'])
    return result


def record_responses(cohort):
    from srgc_rebuttal.build_cache import CacheStore

    plan_path = cohort.plans['mbpp']
    plan = load_plan(plan_path)
    path = input_path(plan_path, plan, 5)
    bundle = json.loads(path.read_text())
    candidate = bundle['candidate_ids'][0]
    # This is a synthetic correct program/test pair, not a model observation.
    bundle['records'][candidate]['answer'] = 'assert identity(4) == 4\nassert identity(-1) == -1'
    atomic_json(path, bundle)
    text = '```\npython\ndef identity(value):\n    return value\n```'
    protocol = {key: plan[key] for key in ('model', 'model_revision', 'verifier', 'responses', 'max_new_tokens', 'attention')}
    store = CacheStore(path, bundle, {**protocol, 'cache_seed': 5})
    store.bind()
    store.write(candidate, [0.] * 8, [text] * 8, 0)
    return candidate


def test_audit_exposes_literal_fence_false_negative_without_rewriting_the_run(cohort):
    record_responses(cohort)
    before = saved(cohort)
    result = audit.audit(cohort.model, 'mbpp', root=cohort.root, rescore=1)
    assert not result['errors']
    row = result['datasets'][0]
    assert row['initial_success_percent'] is None and not row['raw_coverage_complete']
    assert row['features']['literal_python_language_line'] == 8
    check = row['rescore'][0]
    assert check['recorded_reward'] == check['current_reward'] == 0
    assert check['diagnostic_score_without_literal_python_line'] == 1
    assert saved(cohort) == before


def test_missing_raw_data_is_unknown_and_default_audit_never_executes_a_verifier(cohort):
    before = saved(cohort)
    with patch('srgc_research.dispatch.gemma_initial_audit._rescore', side_effect=AssertionError('rescore not requested')):
        result = audit.audit(cohort.model, root=cohort.root)
    assert not result['errors'] and len(result['datasets']) == 10
    assert all(row['initial_success_percent'] is None and not row['features'] for row in result['datasets'])
    assert saved(cohort) == before


def test_public_module_is_gpu_free_and_prints_a_table(cohort):
    result = subprocess.run([sys.executable, '-m', 'srgc_research.dispatch.instruct_initial_audit',
                             cohort.model, '--root', str(cohort.root)],
                            env={**cohort.env, 'CUDA_VISIBLE_DEVICES': ''}, capture_output=True,
                            text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'Cache prompts' in result.stdout and 'Raw prompts' in result.stdout


def test_rejects_wrong_model_inputs_without_rewriting(cohort):
    path = input_path(cohort.plans['math'], load_plan(cohort.plans['math']), 5)
    data = json.loads(path.read_text())
    data['provenance']['model'] = 'different/model'
    atomic_json(path, data)
    before = saved(cohort)
    assert audit.audit(cohort.model, 'math', root=cohort.root)['errors']
    assert saved(cohort) == before


@pytest.mark.parametrize('tamper', ['sampling_seed', 'records_sha256'])
def test_rejects_foreign_receipts_and_protocols(cohort, tamper):
    candidate = record_responses(cohort)
    path = input_path(cohort.plans['mbpp'], load_plan(cohort.plans['mbpp']), 5).with_suffix('.cache')
    target = path / ('protocol.json' if tamper == 'records_sha256' else hashlib.sha256(candidate.encode()).hexdigest() + '.json')
    value = json.loads(target.read_text())
    value[tamper] = 'invalid-record-identity'
    atomic_json(target, value)
    before = saved(cohort)
    assert audit.audit(cohort.model, 'mbpp', root=cohort.root)['errors']
    assert saved(cohort) == before


def test_audit_output_cannot_replace_inputs_or_an_existing_report(cohort, tmp_path):
    before = saved(cohort)
    with pytest.raises(SystemExit) as error:
        audit.main([cohort.model, '--root', str(cohort.root), '--output', str(cohort.plans['math'])])
    assert error.value.code == 2 and saved(cohort) == before
    destination = tmp_path / 'existing-report.json'
    destination.write_text('existing report')
    with pytest.raises(FileExistsError):
        audit.main([cohort.model, '--root', str(cohort.root), '--output', str(destination)])
    assert destination.read_text() == 'existing report'
