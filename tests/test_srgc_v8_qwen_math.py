"""Qwen MATH routing and failed-environment isolation; no GPU/model/network calls."""
import sys
import tempfile
from pathlib import Path

from scripts import srgc_v8_nodes as nodes
from scripts import v8_queue_runtime as source

ROOT=Path(__file__).resolve().parents[1]


def test_qwen_math_uses_existing_math_plan_and_qwen_interpreter(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'v8/experiments'))
    from switching import cluster,spec
    monkeypatch.setenv('SRGC_QWEN_ROOT',str(tmp_path/'qwen-existing'))
    cfg=spec.resolve('qwen_math',role='rebuttal_bank')
    path=cluster.resolve_plan(cfg,5,ROOT)
    assert path==tmp_path/'qwen-existing/experiments/qwen35-9b-math.json'
    assert cfg['runtime']['profile']=='qwen' and cfg['dataset']=='math'
    assert not (tmp_path/'qwen-existing').exists()
    env=dict(PAIR_PYTHON='/pair/python',SWITCH_PYTHON='/mbpp/python',QWEN_PYTHON='/qwen/python')
    assert nodes.cohort_python('qwen_math','fallback',env)=='/qwen/python'
    env['V8_PYTHON_QWEN_MATH']='/specific/qwen-math/python'
    assert nodes.cohort_python('qwen_math','fallback',env)=='/specific/qwen-math/python'


def test_bad_qwen_math_environment_does_not_block_three_ready_cohorts(tmp_path,monkeypatch):
    monkeypatch.setattr(tempfile,'tempdir',str(tmp_path))
    base=tmp_path/'absent'
    env=dict(V8_ROOT=str(base),V8_RUNNER_ROOT=str(ROOT/'v8/experiments'),V8_PYTHON=sys.executable,
             V8_PYTHON_QWEN_MATH=str(tmp_path/'missing-python'))
    plan=source.selections(ROOT,env,[])
    assert set(plan.errors)=={'qwen_math'}
    assert {c for group in plan for c in group['cohorts']}=={'olmo_math','qwen_mbpp','olmo_mbpp'}
    assert not base.exists()
