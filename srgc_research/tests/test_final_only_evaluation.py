"""Skip reporting evaluations while retaining real CPU optimizer recovery and endpoints."""
import json
from unittest.mock import patch

import pytest

torch = pytest.importorskip('torch')

from scripts import srgc_step_checkpoints as checkpoints
from srgc_research.dispatch import reward_curves as curves
from srgc_research.tests.test_boundary_evaluation_resume import make_engine
from srgc_research.tests.test_node_kill_resume import run_to


@pytest.mark.parametrize('task', ['prefix', 'on_policy', 'random', 'sr', 'switch'])
def test_default_records_checkpoints_without_calling_intermediate_eval(tmp_path, monkeypatch, task):
    monkeypatch.delenv('SRGC_INTERMEDIATE_EVAL', raising=False)
    engine = make_engine(tmp_path, task)
    stop = 25 if task == 'prefix' else 50
    with patch.object(engine.backend, 'evaluate', side_effect=AssertionError('periodic evaluation blocks training')):
        run_to(engine, tmp_path, task, stop)
        engine.capture_boundary()
    name = 'prefix' if task == 'prefix' else task
    directory = tmp_path / 'reward-curves' / name
    saved = directory / f'step-{stop}.pt'
    assert torch.load(saved, weights_only=False)['step'] == stop
    latest = torch.load(tmp_path / f'{task}-latest.pt', weights_only=False)
    assert latest['step'] == stop and latest['backend']['optimizer']['state']
    assert not (directory / f'step-{stop}.json').exists()
    rewards = json.loads((directory / 'training.json').read_text())
    assert rewards['points'][-1]['step'] == stop
    # Repeated boundary checks must reuse an already valid archived checkpoint.
    before = saved.read_bytes(), saved.stat().st_mtime_ns
    engine.capture_boundary()
    assert (saved.read_bytes(), saved.stat().st_mtime_ns) == before


def test_resume_from25_advances_to26_without_repeating_evaluation(tmp_path, monkeypatch):
    monkeypatch.delenv('SRGC_INTERMEDIATE_EVAL', raising=False)
    engine = make_engine(tmp_path, 'prefix')
    run_to(engine, tmp_path, 'prefix', 25)
    state = torch.load(tmp_path / 'prefix-latest.pt', weights_only=False)
    resumed = make_engine(tmp_path, 'on_policy')
    resumed.load_state_dict(state, fork_arm='on_policy')
    with patch.object(resumed.backend, 'evaluate', side_effect=AssertionError('resume evaluation')):
        resumed.update()
    assert resumed.step == 26 and resumed.backend.optimizer_steps == 26
    assert torch.load(tmp_path / 'on_policy-latest.pt', weights_only=False)['step'] == 26


def test_invalid_intermediate_eval_switch_fails_before_training(tmp_path, monkeypatch):
    monkeypatch.setenv('SRGC_INTERMEDIATE_EVAL', 'yes')
    with pytest.raises(ValueError, match='SRGC_INTERMEDIATE_EVAL'):
        make_engine(tmp_path, 'prefix')


def test_explicit_periodic_eval_remains_available(tmp_path, monkeypatch):
    monkeypatch.setenv('SRGC_INTERMEDIATE_EVAL', '1')
    engine = make_engine(tmp_path, 'prefix')
    run_to(engine, tmp_path, 'prefix', 25)
    assert len(engine.backend.evaluations) == 1
    assert (tmp_path / 'reward-curves/prefix/step-25.json').exists()


def test_checkpoint_only_export_is_explicitly_unmeasured(tmp_path, monkeypatch):
    from srgc_research.dispatch.reward_curve_results import collect
    monkeypatch.setenv('SRGC_INTERMEDIATE_EVAL', '0')
    engine = make_engine(tmp_path, 'on_policy')
    run_to(engine, tmp_path, 'on_policy', 50)
    identity = json.loads((tmp_path / 'run.json').read_text())
    read = lambda p: json.loads(p.read_text()) if p.exists() else None
    row = collect(tmp_path, 'on_policy', {'shared_prefix_updates':25, 'total_updates':275},
                  identity, ['held-a','held-b'], read)
    assert row['held_out_rewards'] == []
    assert len(row['training_rewards']) == 25
    assert [p['step'] for p in row['deferred_held_out_evaluations']] == [25,50]
    assert all(p['status'] == 'not_measured' and p['checkpoint_hash_verified']
               for p in row['deferred_held_out_evaluations'])


def test_saved_evaluations_survive_switch_to_final_only(tmp_path, monkeypatch):
    monkeypatch.setenv('SRGC_INTERMEDIATE_EVAL', '1')
    engine = make_engine(tmp_path, 'on_policy')
    run_to(engine, tmp_path, 'on_policy', 50)
    old = {p:p.read_bytes() for p in (tmp_path/'reward-curves').glob('*/step-*.json')}
    state = torch.load(tmp_path/'on_policy-latest.pt', weights_only=False)
    monkeypatch.setenv('SRGC_INTERMEDIATE_EVAL', '0')
    resumed = make_engine(tmp_path, 'on_policy')
    resumed.load_state_dict(state)
    resumed.update()
    assert resumed.step == 51 and resumed.backend.evaluations == []
    assert all(p.read_bytes() == raw for p,raw in old.items())


def test_disabling_evaluation_preserves_cpu_optimizer_trajectory(tmp_path, monkeypatch):
    states = []
    for periodic in ['0','1']:
        monkeypatch.setenv('SRGC_INTERMEDIATE_EVAL', periodic)
        folder = tmp_path/periodic
        engine = make_engine(folder, 'on_policy')
        run_to(engine, folder, 'on_policy', 51)
        states.append(engine.backend.state_dict())
    torch.testing.assert_close(states[0], states[1], rtol=0, atol=0)
