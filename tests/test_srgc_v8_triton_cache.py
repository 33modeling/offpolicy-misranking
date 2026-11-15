"""Task children get a node-local Triton cache unless one is configured; CPU only (run_child is faked)."""
import importlib
import os
from pathlib import Path
import threading
import types

ROOT=Path(__file__).resolve().parents[1]


def launch(tmp_path,monkeypatch,environment):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'));worker=importlib.import_module('v8_runtime_worker')
    import sys
    seen={}
    def run_child(command,log,env,**kwargs):seen.update(env);return 0
    package=types.ModuleType('srgc_rebuttal');package.__path__=[]
    monkeypatch.setitem(sys.modules,'srgc_rebuttal',package)
    monkeypatch.setitem(sys.modules,'srgc_rebuttal.cluster',types.SimpleNamespace(run_child=run_child))
    monkeypatch.setitem(sys.modules,'srgc_rebuttal.progress',types.SimpleNamespace(signature=lambda d:()))
    runtime=types.SimpleNamespace(environment=dict(environment),gpus=['0','1'],cohort='qwen_math',heartbeat=lambda *a:None)
    extra=dict(SRGC_PROGRESS_DIR=str(tmp_path/'progress'))
    assert worker.run_srgc_child(runtime,['true'],tmp_path/'task.log',(),threading.Event(),extra=extra,label='t')==0
    return seen


def test_default_triton_cache_is_node_local(tmp_path,monkeypatch):
    env=launch(tmp_path,monkeypatch,dict(TMPDIR=str(tmp_path)))
    assert env['TRITON_CACHE_DIR']==str(tmp_path/f'v8-triton-{os.getuid()}')
    assert env['CUDA_VISIBLE_DEVICES']=='0,1' and env['PYTHONFAULTHANDLER']=='1'


def test_configured_triton_cache_is_kept(tmp_path,monkeypatch):
    assert launch(tmp_path,monkeypatch,dict(TRITON_CACHE_DIR='/local/cache'))['TRITON_CACHE_DIR']=='/local/cache'
