"""Source runtime identity/recovery regressions; CPU only."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'v8/experiments'))
from switching import coderepo
from scripts import srgc_v8_nodes as nodes


def adapter_hash():
    spec=importlib.util.spec_from_file_location('qwen_identity_fixture',ROOT/'scripts/srgc_qwen35.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.adapter_digest()


def test_qwen_recorded_identity_contains_engine_and_adapter():
    base=coderepo.implementation_digest(ROOT);adapter=adapter_hash()
    recorded=hashlib.sha256((base+adapter).encode()).hexdigest()
    selected=coderepo.source_runtime(ROOT,'qwen',recorded)
    assert selected==dict(mode='current',implementation_sha256=recorded,base_implementation_sha256=base,adapter_sha256=adapter)
    with pytest.raises(ValueError,match='unavailable'):coderepo.source_runtime(ROOT,'qwen',base)


@pytest.mark.parametrize('saved',['9435e80003f41e1f65cb9dcc4074f1b06f063880d4823f433d5cd8fb5e2d74a7',
                                 'f581eb043e89409e0e68d8ed77201fa030e34bd93d4d5babbab65304c24a5d6b'])
def test_original_saved_package_and_qwen_composite_are_verified(saved):
    assert coderepo.source_runtime(ROOT,'olmo',saved)['mode']=='saved'
    combined=hashlib.sha256((saved+adapter_hash()).encode()).hexdigest()
    result=coderepo.source_runtime(ROOT,'qwen',combined)
    assert result['mode']=='saved' and result['base_implementation_sha256']==saved
    with pytest.raises(ValueError):coderepo.source_runtime(ROOT,'olmo','f'*64)


def test_failed_setup_can_retry_corrected_source_without_erasing_failure(tmp_path):
    old=dict(schema=1,cohort='fixture',root=str(tmp_path),trainer_sha256='engine',config_sha256='cfg',
             world_size=4,coordinator_sha256='old',runner_sha256={'cluster.py':'old'})
    pool=nodes.Pool(tmp_path,old)
    with pool.control():
        pool.initialize();pool.fail('locate failed because old comparator ignored Qwen adapter')
    updated=dict(old,coordinator_sha256='new',runner_sha256={'cluster.py':'fixed'})
    pool=nodes.Pool(tmp_path,updated)
    with pool.control():pool.initialize()
    state=nodes.read(tmp_path/'state.json')
    assert state['phase']=='prepare' and state['failure'] is None
    assert len(state['failure_history'])==1
    assert list((tmp_path/'setup-history').glob('*.json'))
    # Once any task exists, do not alter the experiment binding.
    nodes.atomic(tmp_path/'jobs/test.json',{'state':'pending'})
    newer=nodes.Pool(tmp_path,dict(updated,coordinator_sha256='another'))
    with newer.control():
        with pytest.raises(ValueError,match='identity differs'):newer.initialize()


@pytest.mark.parametrize('profile',['olmo','qwen'])
def test_saved_runtime_is_actually_loaded_before_cpu_v8_training(tmp_path,profile):
    plan=json.loads((ROOT/'srgc_rebuttal/experiments/pair_seeds.json').read_text())
    plan['output_root']='runs';path=tmp_path/'plan.json';path.write_text(json.dumps(plan))
    saved='f581eb043e89409e0e68d8ed77201fa030e34bd93d4d5babbab65304c24a5d6b'
    folder=tmp_path/'runs/seed-5';folder.mkdir(parents=True)
    recorded=hashlib.sha256((saved+adapter_hash()).encode()).hexdigest() if profile=='qwen' else saved
    (folder/'prefix-ready.json').write_text(json.dumps({'implementation_sha256':recorded}))
    code='''import sys,json
from pathlib import Path
root,plan,out,saved,profile,recorded=map(str,sys.argv[1:])
sys.path.insert(0,root);sys.path.insert(0,str(Path(root)/'v8/experiments'))
from switching import coderepo
coderepo.activate(root)
selection=coderepo.source_runtime(root,profile,recorded)
coderepo.activate_source_runtime(root,plan,5,selection)
from srgc_rebuttal import srgc,runtime
assert runtime.code_digest()==saved
assert 'frozen_srgc' in srgc.__file__
from switching import backends,smoke,spec,matrix,runner
from switching.engine import make_engine_class
from switching.common import sha256_file
feat,ans,data=backends.toy_problem(0)
ctx=smoke.toy_ctx(Path(out),5,srgc,make_engine_class(srgc),feat,ans,data,50)
smoke.make_prefix(ctx,5,25,srgc)
p=Path(out)/'prefixes/seed-5/prefix.pt'
cfg=spec.resolve('olmo_math',role='smoke',extra={'training':{'matched_update_endpoint':50,'hard_safety_update_cap':75}})
task=matrix.train_task(cfg,role='smoke',seed=5,method_id='bank_sr_t025',parent={'kind':'prefix','path':str(p),'sha256':sha256_file(p),'completed_update':25,'run_id':None},horizon=50,stage='bank-extended')
result=runner.execute_train(task,ctx)
assert result['completed_updates']==50,result
print('SAVED_RUNTIME_CPU_TRAINING_OK')
'''
    result=subprocess.run([sys.executable,'-c',code,str(ROOT),str(path),str(tmp_path/'cpu'),saved,profile,recorded],
                          env=dict(os.environ,CUDA_VISIBLE_DEVICES='',V8_CODE_ROOT=str(ROOT),OMP_NUM_THREADS='1',
                                   V8_FSYNC='0',PYTHONDONTWRITEBYTECODE='1'),
                          text=True,capture_output=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'SAVED_RUNTIME_CPU_TRAINING_OK' in result.stdout


@pytest.mark.parametrize('cohort',['olmo_math','qwen_mbpp'])
def test_production_locator_uses_the_verified_training_identity_and_explains_missing_prefix(tmp_path,monkeypatch,cohort):
    from switching import cluster,spec
    cfg=spec.resolve(cohort,role='rebuttal_bank')
    plan=json.loads((ROOT/'srgc_rebuttal/experiments/pair_seeds.json').read_text())
    plan.update(output_root='runs',model=cfg['model']['repository'],model_revision=cfg['model']['revision'])
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan))
    saved='f581eb043e89409e0e68d8ed77201fa030e34bd93d4d5babbab65304c24a5d6b'
    identity=hashlib.sha256((saved+adapter_hash()).encode()).hexdigest() if cohort=='qwen_mbpp' else saved
    for seed in [5,6,7,8,9]:
        folder=tmp_path/f'runs/seed-{seed}';folder.mkdir(parents=True)
        nodes.atomic(folder/'run.json',{'implementation_sha256':identity})
        nodes.atomic(folder/'prefix-ready.json',dict(implementation_sha256=identity,checkpoint_sha256='a'*64,completed_updates=25))
    monkeypatch.setenv('V8_CODE_ROOT',str(ROOT));monkeypatch.setenv('V8_ALLOW_ANY_ROOT','1')
    monkeypatch.setattr(cluster,'resolve_plan',lambda *args:path)
    args=SimpleNamespace(code_root=str(ROOT),cohort=cohort,v8_root=str(tmp_path/'v8'),allow_mismatch=False)
    cluster.cmd_locate(args)
    located=nodes.read(tmp_path/'v8'/cohort/'locator.json')
    assert all(r['implementation_matches_code_root'] and r['runtime_selection']['mode']=='saved' for r in located['seeds'].values())
    (tmp_path/'runs/seed-5/prefix-ready.json').unlink()
    with pytest.raises(SystemExit,match='missing_prefix'):cluster.cmd_locate(args)
