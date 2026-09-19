"""Display labels must not guess or rewrite frozen experimental conditions."""

import importlib.util
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    'status_summary', Path(__file__).resolve().parents[1] / 'scripts/_status_summary.py')
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


@pytest.mark.parametrize('name,key', [
    ('selection-switch-mbpp-v1', 'fresh'),
    ('selection-switch-mbpp-quality-v1', 'quality'),
    ('selection-switch-mbpp-difficulty-v1', 'difficulty'),
    ('selection-switch-mbpp-long-v1', 'long'),
])
def test_known_mbpp_root_names_have_full_distinct_fallback_labels(name, key):
    label = summary.MBPP_SUITE_LABELS[key]
    assert summary.mbpp_suite_label(Path('/work/runs') / name) == label
    assert summary.mbpp_suite_label(name, {'schema': 'legacy'}) == label
    assert summary.suite_label(name) == 'MBPP ' + label


@pytest.mark.parametrize('selector,accounting,key', [
    ('fresh_r', 'budget', 'fresh'), ('fresh_r', 'matched', 'quality'), ('difficulty', 'budget', 'difficulty'),
])
def test_explicit_frozen_settings_override_directory_name_without_mutation(selector, accounting, key):
    protocol = {'dataset': 'mbpp', 'selector': selector, 'accounting': accounting, 'gate': 'convergence'}
    before = dict(protocol)
    for root in ('custom-study', 'selection-switch-mbpp-v1', 'selection-switch-mbpp-quality-v1'):
        assert summary.mbpp_suite_label(root, protocol) == summary.MBPP_SUITE_LABELS[key]
        assert summary.suite_label(root, protocol) == 'MBPP ' + summary.MBPP_SUITE_LABELS[key]
    assert protocol == before


@pytest.mark.parametrize('protocol', [None, {}, {'selector': 'fresh_r'}, {'selector': 'future', 'accounting': 'budget'},
                                    {'selector': ['invalid'], 'accounting': 'budget'}])
def test_unknown_custom_roots_retain_their_name_without_guessing(protocol):
    assert summary.mbpp_suite_label('/work/selection-switch-mbpp-custom-v2', protocol) == 'selection-switch-mbpp-custom-v2'


def test_wrong_dataset_does_not_claim_a_mbpp_protocol():
    protocol = {'dataset': 'math500', 'selector': 'fresh_r', 'accounting': 'budget'}
    assert summary.mbpp_suite_label('selection-switch-mbpp-v1', protocol) == 'selection-switch-mbpp-v1'
    assert summary.suite_label('selection-switch-mbpp-v1', protocol) == 'selection-switch-mbpp-v1'


def test_unregistered_explicit_protocol_does_not_inherit_known_root_label():
    root = 'selection-switch-mbpp-v1'
    assert summary.mbpp_suite_label(root, {'selector': 'difficulty', 'accounting': 'matched'}) == root


def test_long_budget_is_distinct_from_base_on_policy_even_at_a_custom_root():
    protocol = {'dataset': 'mbpp', 'selector': 'fresh_r', 'accounting': 'budget',
                'gate': 'final', 'budget_gpu_seconds': 87120}
    assert summary.mbpp_suite_label('/saved/custom-long', protocol) == 'On-policy · 장시간 예산'
    assert summary.suite_label('/saved/custom-long', protocol) == 'MBPP On-policy · 장시간 예산'
    protocol['budget_gpu_seconds'] = 28380
    assert summary.mbpp_suite_label('/saved/custom-base', protocol) == 'On-policy · 선택비용 포함'


def test_other_experiment_suite_labels_remain_unchanged():
    assert summary.suite_label('selection-switch-v1') == 'on-policy'
    assert summary.suite_label('mopps-comparison-v1') == 'MoPPS'
    assert summary.suite_label('selection-switch-difficulty-v1') == 'difficulty'
    assert summary.suite_label('selection-switch-long-v1') == 'long'


@pytest.mark.parametrize('function,raw,expected', [
    ('selector_label', 'fresh_r', 'On-policy'), ('selector_label', 'difficulty', 'Difficulty'),
    ('accounting_label', 'budget', '선택비용 포함'), ('accounting_label', 'matched', '선택비용 별도'),
    ('gate_label', 'final', '최종 보상 기준'), ('gate_label', 'convergence', '비용 보정 학습 효율 기준'),
])
def test_protocol_field_display_labels(function, raw, expected):
    assert getattr(summary, function)(raw) == expected


@pytest.mark.parametrize('raw', ['future-value', None, 7, ['unexpected']])
def test_unknown_protocol_field_values_are_preserved(raw):
    for function in (summary.selector_label, summary.accounting_label, summary.gate_label):
        assert function(raw) == raw
