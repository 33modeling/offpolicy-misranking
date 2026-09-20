"""Upload-sized diagnostics retain recovery blockers without reading payloads."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import mbpp_diagnostic_parts as parts
import mbpp_failure_summary as summary
import selection_gate as core


def joined(paths):
    return ''.join(path.read_text().split('\n\n', 1)[1] for path in paths)


@pytest.mark.parametrize('text', ['x' * (parts.PART_BYTES - 512),
                                   '한글복구사유' * 200000, 'a\n' * 1000000])
def test_parts_preserve_all_content_and_fit_byte_limit(tmp_path, text):
    paths = parts.write_parts(iter([text[:17], text[17:]]), tmp_path)
    assert joined(paths) == text
    assert all(path.stat().st_size <= 1_900_000 for path in paths)
    assert len(paths) <= 3
    assert paths == sorted(paths)
    again = parts.write_parts([text], tmp_path)
    assert again[0].parent != paths[0].parent
    assert joined(paths) == text


def test_every_recovery_blocker_and_checkpoint_metadata_is_exported_read_only(tmp_path, monkeypatch):
    root = tmp_path / 'runs/quality'
    for seed in range(5):
        branch = root / f'states/s{seed}-t50/points/view-50/selection_reduced'
        core.atomic_json(branch / 'budget-recovery/review.json', {'error': f'unique-review-{seed}\n' + 'context\n' * 2000})
        core.atomic_json(branch / 'budget-recovery/failure.json', {'error': f'unique-failure-{seed}'})
        core.atomic_json(branch / 'policy/checkpoint-90/checkpoint_state.json', {'completed_steps': 90})
        (branch / 'policy/checkpoint-90/adapter_model.safetensors').write_text('PRIVATE_MODEL')
    core.atomic_json(branch / 'budget-recovery/result.json', {
        'evaluation_complete': True, 'canonical_complete': False, 'points': ['PRIVATE_ROLLOUT']})
    core.atomic_json(root / 'gate-fit/failure.json', {'error': 'missing development labels'})
    log = tmp_path / 'runs/experiments/logs/console.mbpp.node-a.log'
    log.parent.mkdir(parents=True)
    log.write_text('noise\n' * 20000 + '[WAIT] dependent gate\n[node-launcher-exit] rc=80\n')
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    original = Path.open

    def guarded(path, *args, **kwargs):
        assert path.suffix not in ('.safetensors', '.pt'), 'payload must not be opened'
        return original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, 'open', guarded)
        paths = parts.write_parts(parts.sections(tmp_path, [root]), tmp_path / 'reports')
    text = joined(paths)
    for seed in range(5):
        assert f'unique-review-{seed}' in text and f'unique-failure-{seed}' in text
    assert '"completed_steps": 90' in text
    assert '"canonical_complete": false' in text
    assert 'rc=80' in text and 'missing development labels' in text
    assert 'PRIVATE_' not in text and '[... omitted ...]' not in text
    assert all(path.stat().st_size <= 1_900_000 for path in paths)
    assert before == {p: p.read_bytes() for p in before}
    assert not list(root.rglob('*.lock'))


def test_missing_corrupt_and_external_records_remain_explicit(tmp_path):
    root = tmp_path / 'runs/quality'
    branch = root / 'states/s2-t50/points/view-50/selection_reduced'
    (branch / 'budget-recovery').mkdir(parents=True)
    (branch / 'budget-recovery/review.json').write_text('{bad')
    outside = tmp_path / 'private.json'
    outside.write_text('{"error": "PRIVATE_OUTSIDE"}')
    (branch / 'budget-recovery/failure.json').symlink_to(outside)
    (branch / 'policy').symlink_to(tmp_path)
    text = ''.join(parts.sections(tmp_path, [root, tmp_path / 'missing']))
    assert '_read_error' in text and 'outside the requested root' in text
    assert 'ROOT MISSING' in text and 'MISSING states/s2-t50' in text
    assert 'PRIVATE_OUTSIDE' not in text


def test_existing_why_entry_point_writes_parts(tmp_path, monkeypatch, capsys):
    root = tmp_path / 'runs/quality'
    core.atomic_json(root / 'switch.json', {'dataset': 'mbpp'})
    monkeypatch.setattr(sys, 'argv', ['why', '--work', str(tmp_path), '--root', str(root)])
    assert summary.main() == 0
    paths = sorted((tmp_path / 'reports/selection-switch').glob('mbpp-why-*/*.txt'))
    assert paths and all(path.stat().st_size <= 1_900_000 for path in paths)
    assert '[parts]' in capsys.readouterr().out


def test_eight_hundred_old_parts_produce_at_most_three_overleaf_files(tmp_path):
    chunks = ('x' * 8192 for _ in range(800))
    paths = parts.write_parts(chunks, tmp_path)
    assert len(paths) == 3
    assert all(path.stat().st_size < 2_000_000 for path in paths)
    assert sum(path.stat().st_size for path in paths) <= 5_700_000
    assert 'EXPORT LIMIT' in paths[-1].read_text()
    assert 'NOT a complete export' in paths[-1].read_text()


def test_single_file_keeps_all_content_beyond_old_total_limit(tmp_path):
    text = '한글\n' * 900000 + 'LAST_DIAGNOSTIC_SECTION\n'
    paths = parts.write_single((text[i:i+8192] for i in range(0, len(text), 8192)), tmp_path)
    assert len(paths) == 1 and paths[0].stat().st_size > 5_700_000
    assert joined(paths) == text and 'EXPORT LIMIT' not in paths[0].read_text()
    assert list(paths[0].parent.iterdir()) == paths
    again = parts.write_single(['another export'], tmp_path)
    assert again[0] != paths[0] and joined(paths) == text


@pytest.mark.parametrize('failure', [OSError('disk unavailable'), KeyboardInterrupt()])
def test_single_file_failure_does_not_publish_partial_txt(tmp_path, failure):
    preserved = tmp_path / 'previous.txt'
    preserved.write_text('prior diagnostic')

    def chunks():
        yield 'partial content'
        raise failure

    with pytest.raises(type(failure)):
        parts.write_single(chunks(), tmp_path)
    assert list(tmp_path.iterdir()) == [preserved]
    assert preserved.read_text() == 'prior diagnostic'


@pytest.mark.parametrize('mode', ['flag', 'environment', 'default'])
def test_single_file_opt_in_does_not_change_default(tmp_path, monkeypatch, capsys, mode):
    root = tmp_path / 'runs/quality'
    core.atomic_json(root / 'switch.json', {'dataset': 'mbpp'})
    args = ['why', '--work', str(tmp_path), '--root', str(root)]
    monkeypatch.delenv('MBPP_WHY_SINGLE', raising=False)
    if mode == 'flag':
        args.append('--single-file')
    elif mode == 'environment':
        monkeypatch.setenv('MBPP_WHY_SINGLE', '1')
    monkeypatch.setattr(sys, 'argv', args)
    before = (root / 'switch.json').read_bytes(), (root / 'switch.json').stat().st_mtime_ns
    assert summary.main() == 0
    paths = list((tmp_path / 'reports/selection-switch').glob('mbpp-why-*/*.txt'))
    assert len(paths) == 1
    message = capsys.readouterr().out
    text = paths[0].read_text()
    if mode == 'default':
        assert '[parts]' in message and 'At most 3 files' in text
    else:
        assert '[single]' in message and 'no total output-size cap' in text
        assert 'Overleaf' not in text and '[upload]' not in message
    assert before == ((root / 'switch.json').read_bytes(), (root / 'switch.json').stat().st_mtime_ns)


def test_real_repair_bash_why_exports_one_read_only_txt(tmp_path):
    import os
    import subprocess

    repo = Path(__file__).resolve().parents[1]
    work = tmp_path / 'shared work'
    root = work / 'runs/selection-switch-mbpp-quality-repair-v1'
    core.atomic_json(root / 'switch.json', {'dataset': 'mbpp', 'gate': 'convergence'})
    core.atomic_json(root / 'gate-fit/failure.json', {'error': 'TEST_GATE_BLOCKER'})
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()}
    env = {k: v for k, v in os.environ.items() if not k.startswith(('MBPP_', 'SWITCH_MBPP_'))}
    env.update(OM_WORK=str(work), SWITCH_PYTHON=sys.executable, MBPP_WHY_SINGLE='1')
    result = subprocess.run(['bash', 'scripts/run_mbpp_repair.sh', 'why'], cwd=repo,
                            env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    paths = list((work / 'reports/selection-switch').glob('mbpp-why-*/*.txt'))
    assert len(paths) == 1 and '[single]' in result.stdout
    assert 'TEST_GATE_BLOCKER' in paths[0].read_text()
    assert not list(root.rglob('*.lock'))
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()}


def test_all_roots_blockers_precede_large_inventories_and_logs(tmp_path, monkeypatch):
    roots = [tmp_path / f'root-{i}' for i in range(4)]
    for index, root in enumerate(roots):
        core.atomic_json(root / 'states/s2-t50/points/view-50/selection_reduced/budget-recovery/review.json',
                         {'error': f'critical-blocker-{index}'})
    monkeypatch.setattr(parts, 'policy_inventory', lambda *args: iter(['inventory\n' * 900000]))
    paths = parts.write_parts(parts.sections(tmp_path, roots), tmp_path / 'reports')
    text = joined(paths)
    assert len(paths) == 3
    assert all(f'critical-blocker-{index}' in text for index in range(4))
    assert 'EXPORT LIMIT' in text


def test_large_embedded_decision_model_is_not_copied(tmp_path):
    branch = tmp_path / 'states/s2-t50/points/view-50/selection_reduced'
    core.atomic_json(branch / 'decision.json', {'budget_gpu_seconds': 28380,
                     'action': 'select', 'model': {'history': 'UNNEEDED' * 100000}})
    text = ''.join(parts.sections(tmp_path, [tmp_path]))
    assert 'UNNEEDED' not in text
    assert '28380' in text and len(text.encode()) < 10000


def test_repeated_holding_is_condensed_without_losing_exit(tmp_path):
    path = tmp_path / 'console.log'
    path.write_text('[holding] next pass\n' * 100 + '[WAIT] missing checkpoint\n[node-launcher-exit] rc=80\n')
    text = ''.join(parts.log_excerpt(tmp_path, path))
    assert text.count('[holding]') == 1
    assert 'rc=80' in text and 'missing checkpoint' in text and 'Collapsed 100' in text


def test_admission_history_cannot_bury_current_exit_or_another_root(tmp_path):
    import os

    roots = [tmp_path / 'runs/quality', tmp_path / 'runs/difficulty']
    for root in roots:
        for index in range(12):
            path = root / f'node-preflight/node-{index}/admission.json'
            core.atomic_json(path, {'state': 'passed', 'index': index, 'detail': 'historical' * 10000})
            os.utime(path, (index + 1, index + 1))
    log = tmp_path / 'runs/experiments/logs/console.mbpp.node.log'
    log.parent.mkdir(parents=True)
    log.write_text('[node-launcher-exit] rc=80\n')
    paths = parts.write_parts(parts.sections(tmp_path, roots), tmp_path / 'reports')
    text = joined(paths)
    assert 'EXPORT LIMIT' not in text
    assert text.index('rc=80') < text.index('RECENT ADMISSIONS')
    assert text.count('ADMISSION HISTORY total=12') == 2
    assert text.count('FILE node-preflight/') == 2 * parts.ADMISSION_LIMIT
    assert text.count('FILE node-preflight/node-11/') == 2
    assert 'FILE node-preflight/node-0/' not in text


def test_nested_curve_lease_evidence_is_exported_without_creating_locks(tmp_path):
    import fcntl

    branch = tmp_path / 'states/s2-t50/points/view-50/selection_reduced'
    core.atomic_json(branch / 'curve/progress.json', {'event_id': 'phase', 'state': 'running'})
    with (branch / 'curve/.cost.lock').open('w') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        text = ''.join(parts.sections(tmp_path, [tmp_path]))
        assert 'FILE states/s2-t50/points/view-50/selection_reduced/curve/progress.json' in text
        assert 'curve/.cost.lock held' in text
    assert not (branch / '.task.lock').exists()
    assert '.task.lock missing' in text


@pytest.mark.parametrize('phase_path', ['curve/step-90', 'budget-recovery/curve/step-90'])
@pytest.mark.parametrize('has_progress', [False, True])
def test_nested_phase_failures_and_bounded_logs_are_exported(tmp_path, phase_path, has_progress):
    import os

    root = tmp_path / 'root'
    branch = root / 'states/s2-t50/points/view-50/selection_reduced'
    phase = branch / phase_path
    core.atomic_json(phase / 'failure.json', {'phase': 'evaluate', 'error': 'NESTED_FAILURE'})
    if has_progress:
        core.atomic_json(phase / 'progress.json', {'phase': 'evaluate', 'state': 'failed'})
    for index in range(6):
        path = phase / f'evaluate-{index}.log'
        path.write_text(f'NESTED_LOG_{index}\n' * 1000)
        os.utime(path, (index + 1, index + 1))
    for name in ('discarded', 'discarded-old'):
        discarded = branch / name / 'curve'
        core.atomic_json(discarded / 'failure.json', {'error': 'DISCARDED_FAILURE'})
        (discarded / 'evaluate-0.log').write_text('DISCARDED_LOG')
    outside = tmp_path / 'outside'
    core.atomic_json(outside / 'failure.json', {'error': 'PRIVATE_FAILURE'})
    (outside / 'evaluate-0.log').write_text('PRIVATE_LOG')
    (phase / 'evaluate-9.log').symlink_to(outside / 'evaluate-0.log')
    (branch / 'external-phase').symlink_to(outside, target_is_directory=True)
    paths = parts.write_parts(parts.sections(tmp_path, [root]), tmp_path / 'reports')
    text = joined(paths)
    assert 'NESTED_FAILURE' in text
    assert text.count(f'ERROR EXCERPT states/s2-t50/points/view-50/selection_reduced/{phase_path}/') == 4
    assert 'NESTED_LOG_5' in text and 'NESTED_LOG_2' in text
    assert 'NESTED_LOG_0' not in text and 'NESTED_LOG_1' not in text
    assert 'DISCARDED_FAILURE' not in text and 'DISCARDED_LOG' not in text
    assert 'PRIVATE_FAILURE' not in text and 'PRIVATE_LOG' not in text
    assert len(text.encode()) < 10000
