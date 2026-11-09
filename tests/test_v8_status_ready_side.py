"""Read-only display of actual ext/eval prerequisites; do not infer a remote cause."""
import time

import pytest

from test_v8_status import fixture,status,write


@pytest.mark.parametrize('ready,reason,expected',[
    (False,'reference_policy_costs_incomplete','waiting for complete reference-policy endpoints and measured costs'),
    (False,'reference_endpoint_unfinished','waiting for the selected reference-policy endpoint to finish'),
    (True,None,'budget frozen; ext/eval join the rotation when their checkpoints are ready'),
])
def test_status_explains_ready_side_work_without_an_old_global_barrier(tmp_path,ready,reason,expected):
    now=time.time();base=fixture(tmp_path,now);queue=base/'olmo_math/.multi-node'
    write(queue/'state.json',dict(phase='train',failure=None,
        execution_order=dict(policy='ready-checkpoint-evaluation-extension-v1'),
        ready_side_work=dict(budget_ready=ready,reason=reason)))
    before={p:p.read_bytes() for p in queue.rglob('*') if p.is_file()}
    text=status.render_clean([base],now=now,gpus=[],host='viewer',cohorts=('olmo_math',),width=200)
    assert expected in text and 'matched-step eval before ext' not in text
    assert {p:p.read_bytes() for p in queue.rglob('*') if p.is_file()}==before
