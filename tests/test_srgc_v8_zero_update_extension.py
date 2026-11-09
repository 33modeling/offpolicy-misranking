"""A frozen-budget extension with no optimizer step must retain its actual fork state."""
import importlib
import inspect
import os
from pathlib import Path
import threading

import pytest

from scripts import srgc_v8_nodes as nodes


def test_zero_update_budget_extension_records_parent_endpoint_without_training(tmp_path,monkeypatch):
    runner_path=os.environ.get('V8_NODES_TEST_RUNTIME')
    if not runner_path:pytest.skip('original V8 runtime required')
    monkeypatch.syspath_prepend(str(Path(nodes.__file__).parent))
    monkeypatch.setenv('V8_CODE_ROOT',str(Path(nodes.__file__).resolve().parents[1]))
    from test_srgc_v8_nodes import ToyRuntime
    runtime=ToyRuntime(runner_path,tmp_path)
    from switching import runner,snapshots
    from switching.common import read_json,read_jsonl,sha256_file
    runtime.barrier('prepare',None,threading.Event(),[])
    # Remove any earlier test's idempotent overlay to reproduce the frozen guard.
    monkeypatch.setattr(runner,'_execute_train',inspect.unwrap(runner._execute_train))
    on=next(t for t in runtime.plan('train')[0] if t['training_seed']==5 and t['method_id']=='on_monitored')
    runner.execute_train(on,runtime.context(5))
    branch=next(t for t in runtime.plan('train')[0] if t['training_seed']==5 and t['method_id']=='bank_sr_t025')
    runner.execute_train(branch,runtime.context(5))
    tasks,_=runtime.mx.build_stage('budget-extension',runtime.cfg,runtime.locator(),runtime.root,seeds=[5],budget=0.0)
    task=next(t for t in tasks if t['method']['extension_of']==branch['run_id'])
    assert task['cost_offset_gpu_seconds']>task['budget_cap_gpu_seconds']
    parent=Path(task['parent']['path']);parent_sha=sha256_file(parent)
    with pytest.raises(ValueError,match='retained snapshots missing'):
        runner.execute_train(task,runtime.context(5))
    repair=importlib.import_module('v8_task_entry').repair_budget_fork_checkpoint(runner)
    assert repair and repair['extra_optimizer_updates']==0
    runner.execute_train(task,runtime.context(5))
    folder=runtime.root/'runs'/task['run_id'];manifest=read_json(folder/'manifest.json')
    assert manifest['status'] in nodes.TERMINAL and manifest['completed_updates']==75
    assert manifest['errors'],'original failed attempt was erased'
    assert all(row['start_update']==75 for row in manifest['attempts'])
    assert not any(read_jsonl(p) for p in folder.glob('attempts/*/updates.jsonl'))
    [index]=read_jsonl(folder/'snapshots.jsonl');assert index['completed_update']==75
    saved=folder/index['path'];assert sha256_file(saved)==index['sha256']
    before=snapshots.branch_identity(snapshots.load(parent));after=snapshots.branch_identity(snapshots.load(saved))
    assert before['model_hash']==after['model_hash'] and before['optimizer_hash']==after['optimizer_hash']
    assert before['rng_hash']==after['rng_hash']
    assert sha256_file(parent)==parent_sha
