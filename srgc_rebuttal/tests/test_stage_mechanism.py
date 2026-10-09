import copy
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import srgc_stage_mechanism as mechanism
from scripts import srgc_stage_report as report
from scripts import srgc_replicate_worker as worker
from scripts.srgc_sr_refresh import extra_arm
from srgc_rebuttal.srgc import Config
from srgc_rebuttal.timing import CostMeter
from srgc_rebuttal.toy_backend import ToyBackend, make_problem


class ProbeToy:
    """CPU integration fixture with TorchBackend's rollout and gather interfaces."""
    gpu_count = 0

    def __init__(self, features, answers):
        self.inner = ToyBackend(features, answers, projection_dim=16, seed=5, learning_rate=.01)
        self.cost_meter = CostMeter()
        self.starts = []
        self.score_seeds = []

    def _rollout(self, prompt, responses, seed):
        rewards, _ = self.inner._rollout(prompt, responses, seed)
        return [np.array([1])] * responses, rewards, 1

    def _gather(self, value):
        return value

    def score_gradients(self, ids, **kwargs):
        self.score_seeds.append(kwargs['seed'])
        for i in ids:
            self._rollout(i, kwargs['responses'], kwargs['seed'])
        return self.inner.score_gradients(ids, **kwargs)

    def train(self, ids, **kwargs):
        self.starts.append(self.inner.state_dict())
        for i in ids:
            self._rollout(i, kwargs['responses'], kwargs['seed'])
        return self.inner.train(ids, **kwargs)

    def evaluate(self, ids, *, responses, seed):
        return {i: float(self._rollout(i, responses, seed)[1].mean()) for i in ids}

    def state_dict(self):
        return self.inner.state_dict()

    def load_state_dict(self, state):
        self.inner.load_state_dict(state)

    def synchronize(self):
        pass


def study(*, stages=(0, 2, 4), horizon=2):
    features, answers, candidates, validation, evaluation, cache = make_problem(
        5, candidates=40, validation=2, evaluation=4)
    data = dict(candidate_ids=candidates, ranking_validation_ids=validation,
                evaluation_ids=evaluation, cached_rewards=cache)
    backend = ProbeToy(features, answers)
    return mechanism.StageStudy(backend, data, Config(seed=5, projection_dim=16),
                                stages=stages, horizon=horizon)


def finish(value):
    for _ in range(2000):
        if value.done:
            return value
        value.advance()
    raise AssertionError('study did not finish')


def equal_model(a, b):
    for key in ('weights', 'm', 'v'):
        np.testing.assert_array_equal(a[key], b[key])
    assert a['updates'] == b['updates']


def test_true_initial_state_and_exact_workload():
    value = finish(study())
    assert value.measurements[0]['stage'] == 0
    assert value.backend.starts[0]['updates'] == 0
    assert value.step == 4 + 3 * 6 * 2
    assert len(value.rows) == 18
    assert len(value.training_records) == 36
    assert mechanism.TOTAL_WORK == 850
    assert {r['mode'] for r in value.rows} == set(mechanism.MODES)


def test_every_branch_restores_model_and_adam_moments():
    value = finish(study(stages=(0,)))
    starts = value.backend.starts
    for index in range(0, 12, 2):
        equal_model(starts[0], starts[index])
    assert starts[1]['updates'] == 1


def test_branches_never_contaminate_carrier():
    value = finish(study())
    independent = study()
    carrier = independent.carrier
    carrier.run_until(4)
    equal_model(value.backend.state_dict(), independent.backend.state_dict())
    assert len(value.carrier.history) == len(carrier.history)
    assert [r['train_ids'] for r in value.carrier.history] == [r['train_ids'] for r in carrier.history]


@pytest.mark.parametrize('phase', ['carrier', 'acquire_A', 'acquire_B', 'baseline',
                                   'restore_branch', 'train', 'evaluate_branch', 'restore_carrier', 'done'])
def test_resume_in_every_phase_preserves_learning_and_selections(phase):
    value = study()
    while value.phase != phase:
        value.advance()
    state = value.state_dict()
    resumed = study()
    resumed.load_state_dict(state)
    finish(value)
    finish(resumed)
    equal_model(value.backend.state_dict(), resumed.backend.state_dict())
    assert value.rows == resumed.rows
    assert value.measurements == resumed.measurements
    assert value.training_records == resumed.training_records


def test_resume_rejects_code_and_protocol_changes():
    value = study()
    state = value.state_dict()
    for field in ('study_code_sha256', 'protocol', 'sampling_protocol'):
        changed = {**state, field: 'different'}
        with pytest.raises(ValueError, match='changed'):
            value.load_state_dict(changed)


def test_independent_block_never_changes_selected_sets():
    value = study()
    while value.phase != 'baseline':
        value.advance()
    before = copy.deepcopy(value.measurement['selected'])
    value.measurement['B']['cosines'].reverse()
    value.measurement['B']['success_rates'] = {i: 1-p for i, p in value.measurement['B']['success_rates'].items()}
    value.finalize_measurement()
    assert value.measurement['selected'] == before
    assert len(set(value.backend.score_seeds)) == 4


def test_shared_pool_four_selected_and_fixed_retention():
    value = finish(study())
    for m in value.measurements:
        assert len(m['candidate_ids']) == len(set(m['candidate_ids'])) == 40
        for mode, ids in m['selected'].items():
            assert len(ids) == len(set(ids)) == 4
            assert set(ids) <= set(m['candidate_ids'])
            records = [r for r in value.training_records if r['stage'] == m['stage'] and r['mode'] == mode]
            assert len(records) == 2
            assert all(r['train_ids'] == ids for r in records)
            assert all(len(rewards) == 8 for r in records for rewards in r['rewards'].values())


def test_sr_cache_and_fresh_agree_when_rewards_agree():
    ids = [str(i) for i in range(40)]
    p = {i: (j % 9) / 8 for j, i in enumerate(ids)}
    chosen = mechanism.select_sets(ids, p, p, list(range(40)), seed=5, step=0, k=4,
                                  tie_order={i: j for j, i in enumerate(ids)})
    assert chosen['sr'] == chosen['sr_fresh']
    assert chosen['on_policy'] == ids[-4:][::-1]


def test_correlation_constant_is_missing_not_zero():
    assert mechanism.correlation([0, 0], [1, 2]) is None


def test_no_rloo_and_no_evaluation_leakage():
    value = study()
    with pytest.raises(ValueError, match='not RLOO'):
        mechanism.StageStudy(value.backend, value.data, Config(objective='rloo'))
    data = {**value.data, 'evaluation_ids': [value.data['candidate_ids'][0]]}
    with pytest.raises(ValueError, match='disjoint'):
        mechanism.StageStudy(value.backend, data, value.config)


def test_costs_separate_carrier_diagnostics_and_branch_training():
    value = study()
    events = []
    value.backend.cost_meter.record = events.append
    finish(value)
    finished = [e for e in events if e['state'] == 'finished']
    assert all(e['gpu_seconds'] == 0 for e in finished)  # This fixture has no GPU.
    assert len([c for c in value.charges if c['component'] == 'diagnostic.A']) == 3
    assert len([c for c in value.charges if c['component'] == 'diagnostic.B']) == 3
    assert len([c for c in value.charges if c['component'].endswith('.train')]) == 36
    assert all(e['wall_seconds'] >= 0 for e in finished)
    assert not value.backend.cost_meter.active


def test_physical_count_and_group_means_not_response_pseudoreplication():
    def row(seed, delta, attention='eager'):
        return dict(dataset='math', seed=seed, result=dict(implementation_sha256='engine',
            study_code_sha256='probe', checkpoint_policy={'attention': attention},
            rows=[dict(stage=s, mode=m, reward=.5 + delta * (s/400) * (m == 'sr'))
                  for s in mechanism.STAGES for m in mechanism.MODES]))
    summaries = report.summarize([row(5, .02), row(6, .04), row(7, .5, 'sdpa')])
    target = next(s for s in summaries if s['identity'][-1] == 'eager' and
                  s['stage'] == 'late-minus-early' and (s['left'], s['right']) == ('sr', 'on_policy'))
    assert target['n'] == 2
    assert target['mean_pp'] == pytest.approx(3.)
    assert target['sample_sd_pp'] == pytest.approx(2 ** .5)
    with pytest.raises(ValueError, match='duplicate'):
        report.summarize([row(5, .02), row(5, .03)])


def test_new_queue_preserves_existing_work_and_dispatches_only_ten_studies():
    assert extra_arm('stage_mechanism') == 'stage_mechanism'
    assert worker.conditions('mechanism') == [(0, 'stage_mechanism')]
    assert len(worker.conditions('all')) == 12
    assert len(worker.conditions('support')) == 3
    assert len(worker.conditions('switch_validation')) == 7
    with patch.object(worker, 'default_plan', return_value=Path('/tmp/test-plan')), \
            patch.object(worker, 'route_plan', side_effect=lambda p, **kw: p), \
            patch.object(worker, 'select_plan', side_effect=lambda p, s: p):
        tasks = worker.tasks_for('all', 'mechanism')
    assert len(tasks) == len({t.key for t in tasks}) == 10


@pytest.fixture(scope='module')
def complete_endpoint():
    value = finish(study(stages=mechanism.STAGES, horizon=mechanism.HORIZON))
    identity = dict(seed=5, plan_sha256='plan', input_sha256='input', implementation_sha256='engine')
    endpoint = dict(**identity, arm=mechanism.ARM, protocol=mechanism.PROTOCOL,
        study_code_sha256=mechanism.code_hash(), prefix_checkpoint_sha256='prefix',
        stages=list(mechanism.STAGES), horizon=mechanism.HORIZON, modes=list(mechanism.MODES),
        carrier_updates=mechanism.STAGES[-1], total_work_updates=mechanism.TOTAL_WORK,
        initial_state='fresh-seeded-base-model-not-prefix', checkpoint_policy={'attention': 'eager'},
        rows=value.rows, measurements=value.measurements, charges=value.charges,
        training_records=value.training_records)
    return value.data, identity, endpoint


def test_full_protocol_endpoint_roundtrip(complete_endpoint):
    data, identity, endpoint = complete_endpoint
    encoded = json.dumps(endpoint, allow_nan=False)
    report.validate_endpoint(json.loads(encoded), identity, 'prefix', data)


@pytest.mark.parametrize('bad', ['prefix', 'updates', 'gain', 'baseline', 'candidate', 'selection', 'training'])
def test_report_rejects_mislabeled_or_incomplete_data(complete_endpoint, bad):
    data, identity, endpoint = complete_endpoint
    value = copy.deepcopy(endpoint)
    if bad == 'prefix':
        value['prefix_checkpoint_sha256'] = 'wrong'
    elif bad == 'updates':
        value['total_work_updates'] = 275
    elif bad == 'gain':
        value['rows'][0]['gain_pp'] = 123
    elif bad == 'baseline':
        value['rows'][0]['baseline_reward'] = .12345
    elif bad == 'candidate':
        value['measurements'][0]['candidate_ids'][1] = value['measurements'][0]['candidate_ids'][0]
    elif bad == 'selection':
        value['measurements'][0]['selected']['sr'] = value['measurements'][0]['candidate_ids'][:3]
    else:
        value['training_records'].pop()
    with pytest.raises(ValueError):
        report.validate_endpoint(value, identity, 'prefix', data)


def test_capture_restores_backend_after_failure():
    value = study()
    original = value.backend._rollout
    with pytest.raises(RuntimeError), mechanism.capture_rewards(value.backend):
        raise RuntimeError('fixture')
    assert value.backend._rollout == original


def test_real_torch_lora_backend_checkpoint_and_capture(tmp_path):
    import torch
    from scripts.srgc_step_checkpoints import save_checkpoint
    from srgc_rebuttal.tests.test_torch_backend import ModelBackendTests
    fixture = ModelBackendTests()
    fixture.setUp()
    backend = fixture.backend
    data = dict(candidate_ids=['p0', 'p1', 'p2', 'p3'], ranking_validation_ids=['p4'],
                evaluation_ids=['p5'], cached_rewards={f'p{i}': [0., 1.] * 4 for i in range(4)})
    config = Config(seed=5, scoring_prompts=4, training_prompts=2, projection_dim=16)
    initial = backend.state_dict()
    value = mechanism.StageStudy(backend, data, config, stages=(0,), horizon=1)
    while value.phase != 'evaluate_branch':
        value.advance()
    save_checkpoint(value, tmp_path, mechanism.ARM)
    saved = torch.load(tmp_path / 'stage_mechanism-latest.pt', weights_only=False, map_location='cpu')
    restored = mechanism.StageStudy(backend, data, config, stages=(0,), horizon=1)
    restored.load_state_dict(saved)
    finish(restored)
    assert len(restored.rows) == 6
    assert len(restored.training_records) == 6
    for name, weight in initial['trainable'].items():
        torch.testing.assert_close(backend.state_dict()['trainable'][name], weight, rtol=0, atol=0)
    assert backend.state_dict()['optimizer']['state'] == initial['optimizer']['state']


def test_mechanism_status_at_zero_and_legacy_results_remain_separate():
    from scripts import srgc_extra_status as status
    from scripts.srgc_sr_refresh import results
    from srgc_rebuttal.tests.test_extra_arm_launch import ExtraArmLaunchTests
    from srgc_rebuttal.runtime import atomic_json
    fixture = ExtraArmLaunchTests()
    fixture.setUp()
    try:
        plan = fixture.plan()
        folder, _, _ = fixture.prefix(plan)
        task = worker.Task('math', 5, 0, mechanism.ARM, plan)
        atomic_json(folder / 'stage_mechanism-progress.json', dict(seed=5, arm=mechanism.ARM,
                    step=0, stage=0, phase='acquire_A', branch_updates=0, mode='on_policy'))
        row = status.inspect(task)
        assert row['state'] == 'checkpoint_saved'
        assert (row['step'], row['total'], row['stage']) == (0, 850, 0)
        # The old P1 command must not reinterpret a multi-stage study as one reward.
        atomic_json(folder / 'stage_mechanism-endpoint.json', {'sentinel': 'not a P1 endpoint'})
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            results(SimpleNamespace(plan=plan, json=False))
        assert 'stage_mechanism' not in output.getvalue()
    finally:
        fixture.doCleanups()


def test_both_archived_engines_restore_mechanism_branches():
    from scripts.srgc_saved_runtime import SAVED_HASH, PREVIOUS_HASH
    from srgc_rebuttal.tests.test_extra_arm_launch import ExtraArmLaunchTests, ROOT
    from srgc_rebuttal.runtime import atomic_json
    fixture = ExtraArmLaunchTests()
    fixture.setUp()
    try:
        plan = fixture.plan()
        folder, _, _ = fixture.prefix(plan)
        for digest in (SAVED_HASH, PREVIOUS_HASH):
            for name in ('prefix-ready.json', 'run.json'):
                path = folder / name
                atomic_json(path, {**json.loads(path.read_text()), 'implementation_sha256': digest})
            result = subprocess.run([sys.executable, '-c', '''
import sys
from scripts.srgc_saved_runtime import bootstrap
bootstrap(['--plan', sys.argv[1], '--seed', '5'])
from srgc_rebuttal.tests.test_stage_mechanism import study, finish, equal_model
a = study()
while a.phase != 'train':
    a.advance()
b = study()
b.load_state_dict(a.state_dict())
finish(a)
finish(b)
assert a.rows == b.rows
equal_model(a.backend.state_dict(), b.backend.state_dict())
''', str(plan)], cwd=ROOT, capture_output=True, text=True, timeout=60)
            assert result.returncode == 0, result.stdout + result.stderr
    finally:
        fixture.doCleanups()


@pytest.mark.parametrize('dataset', ['math', 'mbpp', 'all'])
@pytest.mark.parametrize('action', ['run', 'status', 'results', 'json'])
def test_shell_dispatch_without_extra_python_options(tmp_path, dataset, action):
    fake = tmp_path / 'python'
    fake.write_text(f'#!{sys.executable}\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
    fake.chmod(0o755)
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(['sh', str(root/'scripts/run_srgc_mechanism.sh'), dataset, action],
        cwd='/tmp', env={**os.environ, 'PAIR_PYTHON': str(fake), 'SWITCH_PYTHON': str(fake)},
        capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    args = json.loads(result.stdout)
    name = dict(run='srgc_replicate_worker.py', status='srgc_extra_status.py',
                results='srgc_stage_report.py', json='srgc_stage_report.py')[action]
    expected = ['scripts/'+name, '--dataset', dataset]
    if action in ('run', 'status'):
        expected += ['--scope', 'mechanism']
    if action == 'run':
        expected += ['--max-attempts', os.environ.get('SRGC_MECHANISM_MAX_ATTEMPTS', os.environ.get('SRGC_MAX_ATTEMPTS', '50'))]
    if action == 'json':
        expected += ['--json']
    assert args == expected
