"""Per-node progress reads bounded local evidence, never GPU/model payloads."""
import os
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
