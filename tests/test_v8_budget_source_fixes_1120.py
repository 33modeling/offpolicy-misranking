"""V8-160 and V8-161 fixed in the runner source (not only through the queue overlays)."""
import inspect
import os
from pathlib import Path
import threading

import pytest

from scripts import srgc_v8_nodes as nodes


def toy(tmp_path, monkeypatch):
    runner_path = os.environ.get('V8_NODES_TEST_RUNTIME')
    if not runner_path:
        pytest.skip('Set V8_NODES_TEST_RUNTIME to v8/experiments')
    monkeypatch.syspath_prepend(str(Path(nodes.__file__).parent))
    monkeypatch.setenv('V8_CODE_ROOT', str(Path(nodes.__file__).resolve().parents[1]))
    from test_srgc_v8_nodes import ToyRuntime
    runtime = ToyRuntime(runner_path, tmp_path)
    from switching import runner
    runtime.barrier('prepare', None, threading.Event(), [])
    monkeypatch.setattr(runner, '_execute_train', inspect.unwrap(runner._execute_train))
    plan = runtime.plan('train')[0]
    on = next(t for t in plan if t['training_seed'] == 5 and t['method_id'] == 'on_monitored')
    branch = next(t for t in plan if t['training_seed'] == 5 and t['method_id'] == 'bank_sr_t025')
    runner.execute_train(on, runtime.context(5))
    runner.execute_train(branch, runtime.context(5))
    return runtime, runner, branch


def test_zero_update_extension_finishes_on_source_runner(tmp_path, monkeypatch):
    runtime, runner, branch = toy(tmp_path, monkeypatch)
    from switching import snapshots
    from switching.common import read_json, read_jsonl, sha256_file
    tasks, _ = runtime.mx.build_stage('budget-extension', runtime.cfg, runtime.locator(), runtime.root,
                                      seeds=[5], budget=0.0)
    task = next(t for t in tasks if t['method']['extension_of'] == branch['run_id'])
    parent = Path(task['parent']['path'])
    parent_sha = sha256_file(parent)
    runner.execute_train(task, runtime.context(5))  # no overlay installed
    folder = runtime.root / 'runs' / task['run_id']
    manifest = read_json(folder / 'manifest.json')
    assert manifest['status'] in nodes.TERMINAL and manifest['completed_updates'] == 75
    assert not manifest.get('errors')
    [index] = read_jsonl(folder / 'snapshots.jsonl')
    assert index['completed_update'] == 75
    before = snapshots.branch_identity(snapshots.load(parent))
    after = snapshots.branch_identity(snapshots.load(folder / index['path']))
    assert (before['model_hash'], before['optimizer_hash'], before['rng_hash']) == \
           (after['model_hash'], after['optimizer_hash'], after['rng_hash'])
    assert sha256_file(parent) == parent_sha


def test_budget_eval_includes_extension_endpoints(tmp_path, monkeypatch):
    runtime, runner, branch = toy(tmp_path, monkeypatch)
    from switching.bank import SeedBank
    from switching.common import read_json
    # B lets the cheap SR branch continue past the matched horizon (75) into its extension
    bank = SeedBank(runtime.root, runtime.cfg, runtime.cfg['cohort_id'], 5, 'rebuttal_bank', evals={})
    on_cost = next(r for r in bank.policy_rows() if r['policy_id'] == 'on_continued')['continuation_gpu_seconds']
    budget = on_cost * 4
    tasks, _ = runtime.mx.build_stage('budget-extension', runtime.cfg, runtime.locator(), runtime.root,
                                      seeds=[5], budget=budget)
    ext = next(t for t in tasks if t['method']['extension_of'] == branch['run_id'])
    runner.execute_train(ext, runtime.context(5))
    assert read_json(runtime.root / 'runs' / ext['run_id'] / 'manifest.json')['completed_updates'] > 75
    cfg = dict(runtime.cfg, seeds=dict(runtime.cfg['seeds'], training=[5]))
    data = runtime.cluster.load_bundle(runtime.root / 'inputs' / 'seed-5.json')
    bank = SeedBank(runtime.root, cfg, cfg['cohort_id'], 5, 'rebuttal_bank', evals={})
    row = next(r for r in bank.policy_rows(budget=budget) if r.get('run_id') == branch['run_id']
               and r.get('budget_status') == 'selected' and r['budget_endpoint_update'] > 75)
    assert row['budget_endpoint_run_id'] == ext['run_id']
    evals = runtime.cluster.budget_eval_tasks(runtime.root, cfg, data, budget)
    assert any(t['run_id'] == ext['run_id'] and t['completed_update'] == row['budget_endpoint_update'] for t in evals)
