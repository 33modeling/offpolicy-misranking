"""Original planner/runner on synthetic CPU data; not H100 or paper measurements."""
import json
import os
from pathlib import Path
import threading
import pytest

from scripts import srgc_v8_nodes as nodes


def test_original_runner_finishes_all_four_cohorts_sequentially(tmp_path,monkeypatch):
    pytest.importorskip('torch')
    pytest.importorskip('transformers')
    from test_srgc_v8_nodes import ToyRuntime,verify_toy_complete
    root=Path(__file__).resolve().parents[1]
    monkeypatch.setenv('V8_CODE_ROOT',str(root));monkeypatch.setenv('V8_FSYNC','0')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    runner=str(root/'v8/experiments')
    pairs=[]
    for name in nodes.DEFAULT_COHORTS:
        runtime=ToyRuntime(runner,tmp_path,name)
        pairs.append((nodes.Pool(runtime.root/'.multi-node',runtime.binding),runtime))
    stop=threading.Event();timer=threading.Timer(30,stop.set);timer.start()
    try:rc=nodes.worker_loop_many(pairs,'synthetic-sequential-node',stop,poll=.001,cohort_order=nodes.DEFAULT_COHORTS)
    finally:timer.cancel();timer.join()
    assert rc==0 and not stop.is_set(),'original synthetic runner failed to leave a sequential cohort'
    summaries={}
    for name in nodes.DEFAULT_COHORTS:
        summaries[name]=verify_toy_complete(runner,tmp_path,name,monkeypatch,min_workers=1)
        summaries[name]['workers']=1
    attempts={}
    for name in nodes.DEFAULT_COHORTS:
        attempts[name]=[a for p in (tmp_path/name/'.multi-node/jobs').glob('*.json') for a in nodes.read(p)['attempts']]
    for first,second in zip(nodes.DEFAULT_COHORTS,nodes.DEFAULT_COHORTS[1:]):
        assert max(a['ended_at'] for a in attempts[first])<=min(a['started_at'] for a in attempts[second])
    (tmp_path/'sequential-original-runner-summary.json').write_text(json.dumps(dict(
        scope='synthetic CPU integration, no scientific measurements',workers=1,cohorts=summaries,
        order=list(nodes.DEFAULT_COHORTS),all_complete=True,new_GPU_jobs=0),indent=2)+'\n')
