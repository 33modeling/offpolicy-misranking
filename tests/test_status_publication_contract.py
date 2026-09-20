"""Status must read the real publisher's format, not a look-alike fixture."""

from pathlib import Path

import pytest

from test_net_gain_gate_gpu import base, completed, core, gpu, protocol, source
from test_selector_pair_status import prepared as pair_prepared, status as pair_status
from test_selection_switch_status import status as switch_status, rule
from test_rloo_status import status as rloo_status, inputs, write, seal
import mbpp_status as display
import selection_switch_gpu as switch
import net_gain_gate as generic_net


def installed_switch_publisher(monkeypatch):
    # Preserve shared globals, then use the production install path. This
    # fixture isolates publication from selection and the toy input contract.
    decision, select_once = gpu.decision, gpu.select_once
    for module, names in ((gpu, ('net', 'HERE', 'TEST_ARMS', 'SELECTORS', 'CODE_FILES',
                                'study', 'protocol', 'select_once', 'measurement_worker', 'decision')),
                          (base, ('verify', 'train_command'))):
        for name in names:
            monkeypatch.setattr(module, name, getattr(module, name))
    switch.install_runtime()
    monkeypatch.setattr(gpu, 'decision', decision)
    monkeypatch.setattr(gpu, 'select_once', select_once)


@pytest.mark.parametrize('experiment', ['pair', 'mbpp'])
@pytest.mark.parametrize('damage', [None, 'result-schema', 'curve-schema', 'receipt'])
def test_real_runtime_publication_is_recognized_without_changing_artifacts(tmp_path, monkeypatch, experiment, damage):
    installed_switch_publisher(monkeypatch)
    out, c = source(tmp_path / 'source')
    monkeypatch.setattr(base, 'verify', lambda _: c)
    monkeypatch.setattr(base, 'policy', lambda *args: Path(c['source_run']))
    monkeypatch.setattr(base, 'rewards', lambda *args: {'q0': .5})
    completed(out, 'random_full')
    p = {**protocol(), 'schema': rule.SCHEMA, 'schedule': rule.SCHEDULE}
    gpu.run_arm(out, {'eval_timeout': 5}, p, 'random_full', list('0123'), {})
    result = out / 'random_full/result.json'
    assert core.read(result)['schema'] == gpu.net.SCHEMA == rule.SCHEMA
    curve = out / 'random_full/curve.json'
    core.atomic_json(curve, {'schema': rule.SCHEMA, 'result_sha256': base.digest(result),
                            'points': {'100': {'updates': 0, 'reward': .5}}})
    if damage == 'result-schema':
        core.atomic_json(result, {**core.read(result), 'schema': generic_net.SCHEMA})
        core.atomic_json(result.with_suffix('.sha256.json'), {'sha256': base.digest(result)})
        core.atomic_json(curve, {**core.read(curve), 'result_sha256': base.digest(result)})
    elif damage == 'curve-schema':
        core.atomic_json(curve, {**core.read(curve), 'schema': generic_net.SCHEMA})
    elif damage == 'receipt':
        core.atomic_json(result.with_suffix('.sha256.json'), {'sha256': 'wrong'})
    root = tmp_path / 'status-root'
    state_root = root / 'branches/on_policy' if experiment == 'pair' else root
    point = state_root / 'states/s3-t100/points/view-100'
    point.parent.mkdir(parents=True)
    point.symlink_to(out, target_is_directory=True)
    before = {p: p.read_bytes() for p in out.rglob('*') if p.is_file()}
    if experiment == 'pair':
        task = pair_status.observe_branch(root, 3, 100, 'random', 'on_policy', ready=True, observations=[])
    else:
        core.atomic_json(root / 'switch.json', {'schema': rule.SCHEMA, 'dataset': 'mbpp', 'gate': 'convergence'})
        data = switch_status.snapshot(root, local_gpus=False)
        task = next(t for t in data['tasks'] if (t['seed'], t['step'], t['arm']) == (3, 100, 'random_full'))
    assert (task['status'] == 'DONE') is (damage is None)
    assert before == {p: p.read_bytes() for p in out.rglob('*') if p.is_file()}


def test_all_42_pair_slots_accept_the_publishers_result_schema(tmp_path, monkeypatch):
    installed_switch_publisher(monkeypatch)
    pair_prepared(tmp_path)
    choices = {f's{s}-t{t}': {'selector': 'cached'} for s in pair_status.pair.TEST_SEEDS for t in pair_status.pair.STEPS}
    core.atomic_json(tmp_path / 'test-decisions.json', choices)
    monkeypatch.setattr(pair_status.gpu, 'decisions', lambda *args: choices)
    initial = pair_status.snapshot(tmp_path)
    for task in initial['tasks']:
        directory = tmp_path / task['directory']
        base.bind(directory / 'result.json', {'schema': gpu.net.SCHEMA, 'complete': True})
        receipt = {'sha256': base.digest(directory / 'result.json')}
        base.bind(directory / 'result.sha256.json', receipt)
        base.bind(directory / 'curve.json', {'schema': rule.SCHEMA, 'result_sha256': receipt['sha256'],
                                           'points': {'25': {'reward': .5}}})
    data = pair_status.dashboard_data(pair_status.snapshot(tmp_path))
    assert display.counts(data['suites'][0])['done'] == 42
    assert '완료 확인 42개' in display.render(data, width=160)


@pytest.mark.parametrize('active_kind', ['branch', 'baseline', 'admission', None])
def test_rloo_nine_saved_results_do_not_mean_finished_while_work_remains(tmp_path, active_kind):
    root = tmp_path / 'rloo'
    for seed in range(3):
        run, evaluation = inputs(tmp_path / f'inputs-{seed}')
        cfg = rloo_status.read(run / 'run_config.json')
        write(run / 'run_config.json', {**cfg, 'seed': seed})
        out = root / 'math500-d0' / f's{seed}'
        rloo_status.experiment.prepare(run, out, evaluation)
        for arm in ('before', *rloo_status.experiment.ARMS):
            seal(out, arm)
    directory = (root / 'math500-d0/s0/random' if active_kind == 'branch' else
                 root / 'math500-d0/s0/before' if active_kind == 'baseline' else
                 root / 'node-preflight/node-1/baseline')
    if active_kind:
        write(directory / 'progress.json', {'host': 'live-node', 'state': 'running', 'updated': 10000,
                                           'phase': 'evaluate', 'event_id': 'active'})
    data = rloo_status.snapshot(root, now=10000)
    count = display.displayed_counts(data['suites'][0], data)
    assert count['saved_done'] == 9
    assert count['done'] == (8 if active_kind == 'branch' else 9)
    assert (count['progress'] == '100.0%') is (active_kind is None)
    text = display.render(data, width=160)
    assert ('전체 종료 아님' in text) is bool(active_kind)
    assert ('CURRENT RUN 1' in text) is bool(active_kind)
    if active_kind:
        assert '100.0%' not in text
