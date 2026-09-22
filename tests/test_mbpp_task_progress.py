"""Per-node progress reads bounded local evidence, never GPU/model payloads."""
import os
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import mbpp_status as dashboard


def task(**extra):
    return {'phase': 'train', 'directory': 'branch', 'status': 'RUNNING', 'kind': 'branch',
            'host': 'node-a', 'seed': 0, 'step': 25, 'arm': 'random_reduced',
            'seconds': 25., 'timeout': 100., 'training_step': 35, **extra}


def test_node_specific_training_allocation_not_suite_completion(tmp_path):
    suite = {'root': str(tmp_path), 'tasks': [task(), task(host='node-b', seconds=70.)]}
    text = '\n'.join(dashboard.render_nodes({'suites': [suite]}, width=200))
    rows = [' '.join(line.split()) for line in text.splitlines()]
    assert any('node-a ' in line and ' RUN 25.0% ' in line for line in rows)
    assert any('node-b ' in line and ' RUN 70.0% ' in line for line in rows)
    assert '업데이트 10회 완료' in text and '결과 완료율 아님' in text


@pytest.mark.parametrize('gradient', [False, True])
def test_four_shard_processed_counts_take_precedence_over_time(tmp_path, gradient):
    directory = tmp_path / 'branch'
    directory.mkdir()
    (directory / 'progress.json').write_text('{}')
    for rank in range(4):
        line = (f'[fresh_r] candidate shard={rank} prompt=9 (2/10)' if gradient else
                '[now] rollout 3/10 (30%, 2s/개)')
        (directory / f'curve-{rank}.log').write_text(line)
    percent, basis = dashboard.task_progress(tmp_path, task(phase='curve', seconds=99.))
    assert percent == ('20.0%' if gradient else '30.0%')
    assert ('8/40' if gradient else '12/40') in basis


def test_stale_shard_logs_do_not_make_a_new_attempt_look_complete(tmp_path):
    directory = tmp_path / 'branch'
    directory.mkdir()
    (directory / 'progress.json').write_text('{}')
    for rank in range(4):
        path = directory / f'curve-{rank}.log'
        path.write_text('rollout 10/10')
        os.utime(path, (0, 0))
    percent, basis = dashboard.task_progress(tmp_path, task(phase='curve'))
    assert percent == '25.0%' and '시간 한도' in basis


def test_unknown_progress_is_not_zero_and_limit_reached_is_not_done(tmp_path):
    assert dashboard.task_progress(tmp_path, task(seconds=None, timeout=None))[0] == '확인 중'
    assert dashboard.task_progress(tmp_path, task(seconds=200.))[0] == '100.0%'
    assert task(seconds=200.)['status'] == 'RUNNING'


def test_outside_log_symlinks_are_not_read(tmp_path):
    root = tmp_path / 'root'
    directory = root / 'branch'
    directory.mkdir(parents=True)
    (directory / 'progress.json').write_text('{}')
    outside = tmp_path / 'private.log'
    outside.write_text('rollout 100/100')
    (directory / 'curve-0.log').symlink_to(outside)
    assert dashboard.task_progress(root, task(phase='curve'))[0] == '25.0%'


def phase_logs(root, lines):
    directory = root / 'branch'
    directory.mkdir()
    (directory / 'progress.json').write_text('{}')
    for rank in range(4):
        (directory / f'fresh-r-candidate-{rank}.log').write_text(lines)
    return directory


def test_completed_generation_does_not_hide_incomplete_gradient(tmp_path):
    phase_logs(tmp_path, 'rollout 100/100\n[fresh_r] candidate shard=0 prompt=77 (78/100)\n')
    percent, basis = dashboard.task_progress(tmp_path, task(phase='fresh-r-candidate', seconds=100.))
    assert percent == '100.0% / 78.0%'
    assert '응답 생성: 100.0% (400/400개)' in basis
    assert 'Gradient candidate: 78.0% (312/400개)' in basis


def test_three_independent_stage_percentages(tmp_path):
    directory = phase_logs(tmp_path, 'rollout 100/100\n[fresh_r] candidate shard=0 prompt=77 (78/100)\n')
    (directory / 'cost.jsonl').write_text(json.dumps({
        'event_id': 'previous', 'phase': 'fresh-r-validation', 'state': 'finished', 'exit_code': 0}) + '\n')
    percent, basis = dashboard.task_progress(tmp_path, task(phase='fresh-r-candidate'))
    assert percent == '100.0% / 100.0% / 78.0%'
    assert basis.index('fresh-r-validation') < basis.index('응답 생성') < basis.index('Gradient candidate')


def test_missing_shard_does_not_turn_one_completed_worker_into_100_percent(tmp_path):
    directory = phase_logs(tmp_path, 'rollout 100/100\n[fresh_r] candidate shard=0 prompt=99 (100/100)\n')
    (directory / 'fresh-r-candidate-3.log').write_text('model loading\n')
    percent, basis = dashboard.task_progress(tmp_path, task(phase='fresh-r-candidate', seconds=200.))
    assert percent == '? / ?'
    assert '3/4 shards' in basis and '100.0%' not in percent


def test_shards_at_different_stages_preserve_each_stage(tmp_path):
    directory = phase_logs(tmp_path, 'rollout 100/100\n[fresh_r] candidate shard=0 prompt=99 (100/100)\n')
    (directory / 'fresh-r-candidate-3.log').write_text('rollout 12/100\n')
    percent, basis = dashboard.task_progress(tmp_path, task(phase='fresh-r-candidate'))
    assert percent == '78.0% / ?'
    assert '312/400' in basis


def test_new_attempt_overrides_previous_completion_of_same_phase(tmp_path):
    directory = phase_logs(tmp_path, 'rollout 50/100\n')
    records = [dict(phase='fresh-r-validation', event_id='old', state='finished', exit_code=0),
               dict(phase='fresh-r-validation', event_id='new', state='started'),
               dict(phase='fresh-r-candidate', event_id='old-current', state='finished', exit_code=0)]
    (directory / 'cost.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in records))
    percent, basis = dashboard.task_progress(tmp_path, task(phase='fresh-r-candidate'))
    assert percent == '? / 50.0%'
    assert '완료 미확인' in basis


def test_time_allocation_is_separate_from_completed_stage_percentages(tmp_path):
    directory = phase_logs(tmp_path, '')
    (directory / 'cost.jsonl').write_text(json.dumps(dict(
        phase='verify-inputs', state='finished', event_id='one', exit_code=0)) + '\n')
    percent, basis = dashboard.task_progress(tmp_path, task(seconds=78.))
    assert percent == '100.0% / 시간 78.0%'
    assert '결과 완료율 아님' in basis


def test_late_old_receipt_cannot_complete_a_new_attempt(tmp_path):
    directory = phase_logs(tmp_path, 'rollout 50/100\n')
    records = [dict(phase='fresh-r-validation', event_id='new', state='started'),
               dict(phase='fresh-r-validation', event_id='old', state='finished', exit_code=0)]
    (directory / 'cost.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in records))
    assert dashboard.task_progress(tmp_path, task(phase='fresh-r-candidate'))[0] == '? / 50.0%'


@pytest.mark.parametrize('width', [80, 100, 120, 200])
def test_multistage_node_table_preserves_all_percentages_and_labels(tmp_path, width):
    directory = phase_logs(tmp_path, 'rollout 100/100\n[fresh_r] candidate shard=0 prompt=77 (78/100)\n')
    (directory / 'cost.jsonl').write_text(json.dumps(dict(
        phase='verify-inputs', state='finished', event_id='one', exit_code=0)) + '\n')
    suite = {'root': str(tmp_path), 'tasks': [task(phase='fresh-r-candidate')]}
    output = '\n'.join(dashboard.render_nodes({'suites': [suite]}, width=width))
    assert '78.0%' in output and output.count('100.0%') >= 2
    assert '단계별' in output and 'node-a' in output
    assert all(dashboard.columns(line) <= width for line in output.splitlines())


def test_phase_history_does_not_read_outside_root(tmp_path):
    directory = phase_logs(tmp_path, 'rollout 10/100\n')
    outside = tmp_path.parent / f'{tmp_path.name}-private.jsonl'
    outside.write_text(json.dumps(dict(phase='private-phase', state='finished', exit_code=0)) + '\n')
    (directory / 'cost.jsonl').symlink_to(outside)
    assert 'private-phase' not in dashboard.task_progress(tmp_path, task(phase='fresh-r-candidate'))[1]


def test_incomplete_counter_never_rounds_up_to_100(tmp_path):
    phase_logs(tmp_path, 'rollout 9999/10000\n')
    assert dashboard.task_progress(tmp_path, task(phase='fresh-r-candidate'))[0] == '99.9%'
