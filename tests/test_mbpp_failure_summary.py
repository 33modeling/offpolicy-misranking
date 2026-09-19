import importlib.util
import sys
from pathlib import Path

import selection_gate as core
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
SPEC = importlib.util.spec_from_file_location('mbpp_summary', ROOT / 'scripts/mbpp_failure_summary.py')
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def failure(root, arm='selection_full'):
    directory = root / 'states/s3-t25/points/view-25' / arm
    core.atomic_json(directory / 'failure.json', {'host': 'node-a', 'time': 100,
        'error': 'train worker failed: [1]\n' + 'context\n' * 5000 + 'ncclUnhandledCudaError: Call to CUDA function failed'})
    core.atomic_json(directory / 'progress.json', {'phase': 'train', 'state': 'failed', 'updated': 101})
    core.atomic_json(directory / 'fresh-r/selected.sha256.json', {})
    core.atomic_json(directory / 'execution.sha256.json', {})
    (directory / 'train-0.log').write_text('NCCL WARN Cuda failure 1\nCall to CUDA function failed\n' +
        'torchrun shutdown noise\n' * 80 + 'ChildFailedError: worker exited\n')
    return directory


@pytest.mark.parametrize('root_count', [3, 4])
def test_saved_work_audit_is_small_read_only_and_shows_archived_progress(tmp_path, root_count):
    from test_waive_stalled_attempts import event
    import selection_gate_gpu as base
    roots = [tmp_path / f'root-{i}' for i in range(root_count)]
    for root in roots:
        directory = failure(root)
        core.atomic_json(directory / 'decision.json', {'budget_gpu_seconds': 28380.})
        for state in ('started', 'finished'):
            base.journal(directory / 'cost.jsonl', event('score1', 'fresh-r-candidate', state,
                         seconds=7095., exit_code=1))
        core.atomic_json(directory / 'policy/checkpoint-45/adapter_model.safetensors', {'test': 'not read'})
        archive = directory / 'discarded/old-attempt/fresh-r'
        core.atomic_json(archive / 'candidate/prompt-1.json', {'test': 'not read'})
        (archive / 'candidate-0.partial').write_text('not read')
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    result = summary.storage_report(roots)
    assert len(result.encode()) <= 4096
    assert result.count('ROOT root-') == root_count
    assert 'remaining=0.0' in result and 'latest=checkpoint-45' in result
    assert 'discarded/old-attempt' in result and 'gradients=1' in result and 'partial=1' in result
    assert before == {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


def test_saved_work_audit_does_not_mistake_missing_storage_or_open_cost_for_zero(tmp_path):
    assert 'ROOT MISSING' in summary.storage_report([tmp_path / 'absent'])
    directory = failure(tmp_path)
    (directory / 'cost.jsonl').write_text('{"state":"started", "event_id":"open"}\n')
    assert 'unknown (open/missing cost event)' == summary.charged(directory)


def test_nccl_warning_survives_shutdown_noise_and_run_is_not_modified(tmp_path):
    failure(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    text = summary.root_summary(tmp_path)
    assert 'NCCL WARN Cuda failure 1' in text
    assert 'fresh-r/selected.sha256.json' in text and 'execution.sha256.json' in text
    assert 'historical, not proof of a live failure' in text
    assert 'not hash-validated' in text
    assert len(text.encode()) <= summary.ROOT_BYTES
    assert before == {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


@pytest.mark.parametrize('root_count', [3, 4])
def test_roots_and_huge_unicode_logs_fit_one_sixteen_kib_report(tmp_path, root_count):
    roots = [tmp_path / f'root-{i}' for i in range(root_count)]
    for root in roots:
        for arm in ('selection_full', 'random_full'):
            failure(root, arm)
    logs = tmp_path / 'runs/experiments/logs'
    logs.mkdir(parents=True)
    for index in range(4):
        (logs / f'console.mbpp.node-{index}.log').write_text('한글 CUDA 내용' * 10000)
    text = summary.report(tmp_path, roots)
    assert len(text.encode('utf-8')) <= summary.MAX_BYTES
    assert all('ROOT ' + root.name in text for root in roots)
    assert text.count('\nNODE ') == 2


def test_nccl_admission_without_task_failure_keeps_rank_error_and_versions(tmp_path):
    core.atomic_json(tmp_path / 'node-preflight/node-a/admission.json', {
        'state': 'failed', 'host': 'node-a', 'runtime_commit': 'worker-revision',
        'attempts': [{'name': 'baseline', 'error': 'ChildFailedError', 'ranks': [{
            'rank': 0, 'torch': '2.7.1', 'cuda_runtime': '12.6', 'nccl': [2, 26, 2],
            'error': 'ncclUnhandledCudaError: Cuda failure 1 Call to CUDA function failed'}]}]})
    text = summary.root_summary(tmp_path)
    assert 'runtime=worker-revision' in text and 'nccl=[2, 26, 2]' in text
    assert 'Cuda failure 1' in text
    assert 'No saved branch failure' in text


def test_missing_or_corrupt_records_are_explicit_and_do_not_hide_other_suites(tmp_path):
    good, bad, missing = (tmp_path / name for name in ('good', 'bad', 'missing'))
    failure(good)
    bad.mkdir()
    (bad / 'switch.json').write_text('{broken')
    text = summary.report(tmp_path, [good, bad, missing])
    assert 'manifest unreadable' in text and 'ROOT missing' in text
    assert 'Cuda failure 1' in text


def test_symlinked_logs_and_records_never_include_outside_data(tmp_path):
    root = tmp_path / 'root'
    directory = failure(root)
    outside = tmp_path / 'outside.txt'
    outside.write_text('DO_NOT_INCLUDE_OUTSIDE_DATA')
    (directory / 'train-0.log').unlink()
    (directory / 'train-0.log').symlink_to(outside)
    (root / 'switch.json').symlink_to(outside)
    assert 'DO_NOT_INCLUDE_OUTSIDE_DATA' not in summary.report(root, [root])


def test_clipping_counts_utf8_bytes_and_marks_omission():
    text = summary.clipped('가나다' * 1000, 127)
    assert len(text.encode()) <= 127 and '[... omitted ...]' in text


@pytest.mark.parametrize('accounting,label', [('budget', '선택비용 포함'), ('matched', '선택비용 별도')])
def test_failure_summary_labels_actual_condition_and_retains_raw_protocol_audit(tmp_path, accounting, label):
    root = tmp_path / 'selection-switch-mbpp-v1'
    protocol = {'dataset': 'mbpp', 'selector': 'fresh_r', 'accounting': accounting, 'gate': 'convergence'}
    core.atomic_json(root / 'switch.json', protocol)
    before = (root / 'switch.json').read_bytes()
    text = summary.root_summary(root)
    assert f'EXPERIMENT On-policy · {label}' in text and f'ROOT {root.name}' in text
    assert f'SELECTOR On-policy  ACCOUNTING {label}  GATE 비용 보정 학습 효율 기준' in text
    assert 'protocol (raw audit): ' + str(protocol) in text
    assert 'No saved branch failure found; missing logs do not prove success.' in text
    assert f'EXPERIMENT On-policy · {label}' in summary.storage_report([root])
    assert (root / 'switch.json').read_bytes() == before


def test_failure_summary_missing_and_wrong_manifest_remain_explicit(tmp_path):
    root = tmp_path / 'selection-switch-mbpp-quality-v1'
    text = summary.root_summary(root)
    assert 'manifest missing: switch.json' in text and 'SELECTOR ?  ACCOUNTING ?  GATE ?' in text
    core.atomic_json(root / 'switch.json', {'dataset': 'math500', 'selector': 'fresh_r', 'accounting': 'matched'})
    text = summary.root_summary(root)
    assert f'EXPERIMENT {root.name}' in text and 'manifest dataset is not MBPP' in text
    assert "'dataset': 'math500'" in text
