"""Check published model comparison against pinned source and actual prepare logic."""
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BOARD = ROOT / 'research-bug-board'
REPORT = json.loads((BOARD / 'model-analysis.json').read_text())['reports'][0]


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def render(value, query=''):
    script = """import {readFileSync} from 'node:fs';
    const {renderAnalysis} = await import(process.argv[1]);
    const input = JSON.parse(readFileSync(0, 'utf8'));
    process.stdout.write(renderAnalysis(input.data, input.query));"""
    result = subprocess.run(['node', '--input-type=module', '-e', script,
                             (BOARD / 'analysis.mjs').as_uri()],
                            input=json.dumps({'data': value, 'query': query}),
                            text=True, capture_output=True, check=True)
    return result.stdout


def test_every_source_reference_resolves_at_the_audited_commit():
    pin = REPORT['source_commit']
    git('merge-base', '--is-ancestor', pin, 'HEAD')
    for row in REPORT['rows']:
        assert row['evidence'], row['id']
        for ref in row['evidence']:
            assert ref['commit'] == pin
            lines = git('show', f"{pin}:{ref['path']}").decode().splitlines()
            assert lines[ref['line'] - 1].strip() == ref['excerpt']
    for change in REPORT['changes'] + REPORT['timeline']:
        git('merge-base', '--is-ancestor', change['commit'], pin)


@pytest.mark.parametrize('model', ['qwen35', 'llama31', 'gemma4_it'])
def test_adapter_digest_matches_both_revisions(model):
    m = REPORT['validation']['models'][model]
    names = m['adapter_files']
    assert len(names) == len(set(names))
    for revision, expected in m['digests'].items():
        h = hashlib.sha256()
        for name in names if model == 'qwen35' else sorted(names):
            h.update(name.encode())
            h.update(git('show', f'{revision}:{name}'))
        assert h.hexdigest() == expected
    assert len(set(m['digests'].values())) == 1
    assert m['recent_changed'] == []


def test_frozen_engine_manifest_is_complete_and_unchanged():
    pin, base = REPORT['source_commit'], REPORT['recent_baseline']
    names = git('ls-tree', '--name-only', pin, 'srgc_rebuttal/').decode().splitlines()
    names = [n for n in names if n.endswith('.py')]
    assert len(names) == REPORT['validation']['engine']['files']
    assert all(git('show', f'{pin}:{n}') == git('show', f'{base}:{n}') for n in names)


@pytest.mark.parametrize('model', ['qwen35', 'llama31', 'gemma4_it'])
def test_real_prepare_defaults_and_split_fingerprints(model, tmp_path):
    from srgc_research.tests.test_model_results import make_cohort
    from srgc_rebuttal.plan import input_path, load_plan

    cohort = make_cohort(model, tmp_path)
    v = REPORT['validation']
    for dataset, path in cohort.plans.items():
        plan = load_plan(path)
        assert {k: plan[k] for k in v['common_defaults']} == v['common_defaults']
        for seed in plan['seeds']:
            bundle = json.loads(input_path(path, plan, seed).read_text())
            for key, digest in v['splits'][dataset]['ids_sha256'].items():
                ids = bundle[key]
                # Split IDs are seed-invariant; cache rollout sampling is not.
                assert len(ids) == v['splits'][dataset]['counts'][key]
                assert hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest() == digest
            assert bundle['cached_rewards'] == {}


def test_full_renderer_exposes_every_comparison_and_pinned_source():
    body = render({'reports': [REPORT]})
    for row in REPORT['rows']:
        assert f'id="compare-{row["id"]}"' in body
        for ref in row['evidence']:
            assert f'/blob/{ref["commit"]}/{ref["path"]}#L{ref["line"]}' in body
    assert len(REPORT['rows']) == len({r['id'] for r in REPORT['rows']})
    assert '미해결 · 사용자 보고 SyntaxError' in body


def test_search_filters_comparisons_instead_of_matching_json_field_names():
    data = {'reports': [REPORT]}
    body = render(data, 'fingerprint')
    assert 'id="compare-fingerprint"' in body
    assert 'id="compare-optimizer"' not in body
    empty = render(data, 'no_such_model_term_123')
    assert '검색에 맞는 비교 항목이 없습니다.' in empty
    assert 'id="compare-' not in empty


def test_renderer_escapes_content_and_rejects_invalid_source_identity():
    data = json.loads(json.dumps({'reports': [REPORT]}))
    row = data['reports'][0]['rows'][0]
    row['title'] = '<script>alert("x")</script>'
    row['evidence'] = [{'commit': 'x', 'path': '../secret', 'line': 1, 'label': '<img src=x onerror=x>'}]
    body = render(data)
    assert '<script>' not in body
    assert '<img ' not in body
    assert '&lt;script&gt;' in body
    assert '/blob/x/' not in body


@pytest.mark.parametrize('data', [None, {}, {'reports': []}])
def test_missing_analysis_leaves_a_readable_fallback(data):
    assert '모델 분석을 읽지 못했습니다.' in render(data)


def test_append_only_report_history_renders_latest_without_duplicate_anchors():
    data = {'reports': [REPORT, {**REPORT, 'title': 'Latest report'}]}
    body = render(data)
    assert body.count('id="compare-model"') == 1
    assert '<h2>Latest report</h2>' in body
