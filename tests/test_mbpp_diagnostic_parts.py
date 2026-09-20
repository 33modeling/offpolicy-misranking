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
