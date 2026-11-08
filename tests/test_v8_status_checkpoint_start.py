"""Never turn a loaded prefix into zero, or an unverified parent into progress."""
import json
import time

import pytest

from test_v8_status import fixture, status, write


@pytest.mark.parametrize('start',[0,25,100])
def test_loaded_start_is_visible_before_the_first_new_update(tmp_path,start):
    now=time.time();base=fixture(tmp_path,now)
    run=base/'olmo_math/runs/traj-5'
    (run/'attempts/x/updates.jsonl').unlink()
    write(run/'manifest.json',{'attempts':[{'attempt_id':'x','start_update':start,'start':'fork'}]})
    _,rows=status.collect(base,'olmo_math',now)
    row=next(r for r in rows if r['run_id']=='traj-5')
    assert row['done']==start
    text=status.render_clean([base],now=now,gpus=[],host='node01',width=180)
    assert f' {start}/275 ' in text


def test_parent_description_alone_does_not_prove_checkpoint_was_loaded(tmp_path):
    now=time.time();base=fixture(tmp_path,now)
    run=base/'olmo_math/runs/traj-5';(run/'attempts/x/updates.jsonl').unlink()
    path=base/'olmo_math/.multi-node/jobs/b.json';job=json.loads(path.read_text())
    job['task']['parent']={'kind':'prefix','completed_update':25,'path':'not-loaded.pt'};write(path,job)
    _,rows=status.collect(base,'olmo_math',now)
    assert next(r for r in rows if r['run_id']=='traj-5')['done'] is None
    text=status.render_clean([base],now=now,gpus=[],host='node01',width=180)
    node_row=next(line for line in text.splitlines() if line.startswith(' ▶ node01'))
    assert '0/275' not in node_row and '?/275' in node_row


def test_completed_new_update_takes_precedence_over_the_loaded_start(tmp_path):
    now=time.time();base=fixture(tmp_path,now)
    write(base/'olmo_math/runs/traj-5/manifest.json',
          {'attempts':[{'attempt_id':'x','start_update':25,'start':'fork'}]})
    _,rows=status.collect(base,'olmo_math',now)
    assert next(r for r in rows if r['run_id']=='traj-5')['done']==37
