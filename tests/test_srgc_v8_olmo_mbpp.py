"""OLMo MBPP uses the same automatic dispatch and final analysis; CPU-only controls."""
import importlib.util
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest

from scripts import srgc_v8_nodes as nodes

ROOT=Path(__file__).resolve().parents[1]


def test_partial_startup_selects_only_available_cohorts(tmp_path,capsys):
    selected=('qwen_mbpp','olmo_mbpp')
    assert nodes.main(['--root',str(tmp_path/'absent'),'--status'],selected_cohorts=selected)==0
    assert json.loads(capsys.readouterr().out)=={c:dict(phase='not_started') for c in selected}
    assert not (tmp_path/'absent').exists()


def test_dispatch_passes_available_entries_instead_of_defaulting_to_a_missing_cohort(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    spec=importlib.util.spec_from_file_location('mbpp_dispatch',ROOT/'scripts/v8_dispatch.py')
    dispatch=importlib.util.module_from_spec(spec);spec.loader.exec_module(dispatch)
    path=tmp_path/'plan.json'
    path.write_text(json.dumps(dict(root=str(tmp_path/'queue'),argv=[],errors={'olmo_math':'unavailable source'},
        entries=[dict(cohort=c,script=str(ROOT/'scripts/srgc_v8_nodes.py'),runner=str(ROOT/'v8/experiments'),python=sys.executable)
                 for c in ('qwen_mbpp','olmo_mbpp')])))
    seen=[]
    def main(args,*,selected_cohorts):
        seen.append(selected_cohorts);return 0
    monkeypatch.setattr(dispatch.nodes,'main',main)
    monkeypatch.setattr(sys,'argv',['v8_dispatch.py',str(path)])
    assert dispatch.main()==1  # preserve the unavailable cohort's reported error
    assert seen==[('qwen_mbpp','olmo_mbpp')]


def test_olmo_mbpp_keeps_the_original_mbpp_interpreter():
    env=dict(PAIR_PYTHON='/math/python',SWITCH_PYTHON='/mbpp/python',QWEN_PYTHON='/qwen/python')
    assert nodes.cohort_python('olmo_mbpp','fallback',env)=='/mbpp/python'
    env['V8_PYTHON_OLMO_MBPP']='/specific/mbpp/python'
    assert nodes.cohort_python('olmo_mbpp','fallback',env)=='/specific/mbpp/python'


def test_original_olmo_mbpp_runner_finishes_and_production_collect_analyze_emit_results(tmp_path,monkeypatch):
    import os
    runner=os.environ.get('V8_NODES_TEST_RUNTIME')
    if not runner:pytest.skip('original V8 runner required')
    monkeypatch.setenv('V8_CODE_ROOT',str(ROOT))
    monkeypatch.setenv('GROUP_VOLUME',str(tmp_path))
    from test_srgc_v8_nodes import ToyRuntime
    runtime=ToyRuntime(runner,tmp_path,'olmo_mbpp')
    pool=nodes.Pool(runtime.root/'.multi-node',runtime.binding)
    assert nodes.worker_loop(pool,runtime,'cpu-node',threading.Event(),poll=.001,retry_delay=0)==0
    from switching import cluster
    monkeypatch.setattr(cluster.spec,'resolve',lambda *a,**k:runtime.cfg)
    args=SimpleNamespace(cohort='olmo_mbpp',v8_root=str(tmp_path))
    cluster.cmd_collect(args);cluster.cmd_analyze(args)
    rows=nodes.read(runtime.root/'analysis/policy_outcomes_bank.json')
    assert rows and all(r.get('eval_status')=='measured' for r in rows if r.get('status') in ('triggered','not_triggered'))
    analysis=nodes.read(runtime.root/'analysis/analysis.json')
    assert analysis['primary_rebuttal']['primary_contrast']['n_training_seeds']==3
    assert analysis['e3'] and analysis['e4']
    assert (runtime.root/'analysis/report.md').stat().st_size>0
    assert all(nodes.read(p)['state']=='done' for p in (pool.root/'jobs').glob('*.json'))
