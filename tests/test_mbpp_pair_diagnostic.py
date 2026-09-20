"""One Bash command exports both experiments, without changing either run."""

import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import mbpp_pair_diagnostic as combined
import selector_pair_diagnostic as pair
from test_selector_pair_diagnostic import snapshot

REPO = Path(__file__).resolve().parents[1]


def record(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def pair_evidence(root):
    directory = root / 'branches/on_policy/states/s0-t25/points/view-25/selection_reduced'
    record(directory / 'progress.json', {'state': 'running', 'updated': 1, 'host': 'PAIR_TEST_NODE'})
    record(directory / 'cost.jsonl', {'event_id': 'open-test', 'state': 'started',
                                      'phase': 'train', 'detail': 'x' * 1_200_000 + 'LAST_PAIR_DETAIL'})
    record(directory / 'failure.json', {'error': 'cost evidence needed'})
    return directory


@pytest.mark.parametrize('custom', [False, True])
def test_real_bash_writes_only_one_uncapped_txt_for_both_runs(tmp_path, custom):
    work, home = tmp_path / 'shared work', tmp_path / 'home'
    home.mkdir()
    source = work / 'runs/selection-switch-mbpp-quality-v1'
    repair = work / 'runs/selection-switch-mbpp-quality-repair-v1'
    root = work / 'runs/selector-pair-v1'
    env = {k: v for k, v in os.environ.items() if not k.startswith(('MBPP_', 'SWITCH_MBPP_')) and k != 'PAIR_ROOT'}
    if custom:
        source, repair, root = [work / name for name in ('custom original', 'custom repair', 'custom pair')]
        env.update(MBPP_REPAIR_SOURCE=str(source), MBPP_REPAIR_ROOT=str(repair), PAIR_ROOT=str(root))
    for path, marker in ((source, 'ORIGINAL_GATE_BLOCKER'), (repair, 'REPAIR_GATE_BLOCKER')):
        record(path / 'switch.json', {'dataset': 'mbpp', 'gate': 'convergence'})
        record(path / 'gate-fit/failure.json', {'error': marker})
    pair_evidence(root)
    lock = root / '.pair.lock'
    lock.write_bytes(b'original lease')
    before = snapshot(work)
    with lock.open('rb') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(['bash', 'scripts/check_mbpp_pair.sh'], cwd=REPO,
                                env={**env, 'HOME': str(home), 'OM_WORK': str(work),
                                     'SWITCH_PYTHON': sys.executable},
                                capture_output=True, text=True, timeout=20)
        assert result.returncode == 0, result.stdout + result.stderr
        with lock.open('rb') as probe, pytest.raises(BlockingIOError):
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
    files = [p for p in home.rglob('*') if p.is_file()]
    assert len(files) == 1 and files[0].name == 'mbpp-pair-why-single.txt'
    assert files[0].stat().st_size > 1024 * 1024
    text = files[0].read_text()
    for marker in ('ORIGINAL_GATE_BLOCKER', 'REPAIR_GATE_BLOCKER', 'PAIR_TEST_NODE',
                   'LAST_PAIR_DETAIL', 'ROOT_LOCK blocked', 'SELECTOR PAIR TASK WAIT EVIDENCE'):
        assert marker in text
    assert 'COST EVIDENCE OMITTED' not in text and 'EXPORT LIMIT' not in text
    assert result.stdout.count('[saved]') == 1 and str(files[0]) in result.stdout
    assert len(result.stdout) < 1024
    assert snapshot(work) == before
    assert not (work / 'reports').exists()


def test_uncapped_pair_does_not_change_default_limits(tmp_path):
    pair_evidence(tmp_path)
    limited = pair.cost_report(tmp_path)
    assert len(limited.encode()) <= pair.COST_REPORT_BYTES and 'COST EVIDENCE OMITTED' in limited
    full = pair.cost_report(tmp_path, uncapped=True)
    assert len(full.encode()) > 1024 * 1024 and 'LAST_PAIR_DETAIL' in full
    assert 'COST EVIDENCE OMITTED' not in full
    assert pair.cost_report(tmp_path) == limited


def test_missing_roots_are_reported_without_creation(tmp_path, monkeypatch, capsys):
    work = tmp_path / 'missing-work'
    home = tmp_path / 'reports'
    monkeypatch.setenv('OM_WORK', str(work))
    for name in ('MBPP_REPAIR_SOURCE', 'SWITCH_MBPP_QUALITY_ROOT', 'MBPP_REPAIR_ROOT', 'PAIR_ROOT'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(sys, 'argv', ['combined', '--report-dir', str(home)])
    assert combined.main() == 0
    files = list(home.glob('*/*.txt'))
    assert len(files) == 1 and not work.exists()
    text = files[0].read_text()
    assert text.count('ROOT MISSING') == 2 and 'ROOT_LOCK missing' in text
    assert '[saved]' in capsys.readouterr().out


def test_bad_mbpp_section_does_not_hide_pair_diagnostic(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        yield 'MBPP partial evidence\n'
        raise OSError('unreadable MBPP section')

    monkeypatch.setattr(combined.mbpp, 'sections', fail)
    text = ''.join(combined.sections(tmp_path, [tmp_path / 'missing'], tmp_path / 'pair'))
    assert 'INCOMPLETE MBPP DIAGNOSTIC' in text
    assert 'SELECTOR PAIR LOCK DIAGNOSTIC' in text and 'SELECTOR PAIR COST EVIDENCE' in text


def test_single_writer_rejects_path_traversal_before_creating_output(tmp_path):
    with pytest.raises(ValueError, match='filename prefix'):
        combined.mbpp.write_single(['data'], tmp_path, prefix='../outside')
    assert list(tmp_path.iterdir()) == []
