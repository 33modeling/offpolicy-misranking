import json
from pathlib import Path

from scripts import v8_probe as probe


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path


def test_missing_root_is_not_created(tmp_path):
    root=tmp_path/'not-created'
    assert all(r['phase']=='not_started' for r in probe.snapshot(root,'host'))
    assert not root.exists()


def test_probe_uses_existing_storage_conventions():
    assert probe.root({})==Path('/group-volume/minsoo3.kim/offpolicy-misranking/v8')
    assert probe.root({'OM_WORK':'/existing'})==Path('/existing/v8')
    assert probe.root({'V8_ROOT':'/explicit'})==Path('/explicit')


def test_actual_worker_path_is_found_without_exposing_unrelated_environment(tmp_path):
    worker=tmp_path/'123';worker.mkdir()
    (worker/'cmdline').write_bytes(b'python3\0/existing/scripts/srgc_v8_nodes.py\0')
    (worker/'environ').write_bytes(b'V8_ROOT=/existing/live-data\0PRIVATE_VALUE=must-not-appear\0')
    other=tmp_path/'456';other.mkdir()
    (other/'cmdline').write_bytes(b'python3\0unrelated.py\0')
    (other/'environ').write_bytes(b'V8_ROOT=/not-this-experiment\0')
    assert probe.active_roots(tmp_path)==[Path('/existing/live-data')]


def test_latest_attempt_only_and_partial_line_never_becomes_checkpoint_proof(tmp_path):
    q=tmp_path/'olmo_math/.multi-node';run=tmp_path/'olmo_math/runs/r'
    save(q/'state.json',{'phase':'train'})
    save(q/'jobs/j.json',{'state':'running','attempts':[{'id':'a','worker':'host.42'}],
                        'task':{'run_id':'r','kind':'train'}})
    save(run/'manifest.json',{'status':'active','attempts':[{'attempt_id':'old'},{'attempt_id':'new'}]})
    old=run/'attempts/old/updates.jsonl';save(old,{'completed_updates':275})
    log=run/'attempts/new/updates.jsonl';log.parent.mkdir(parents=True);log.write_text('{"completed_updates":26}\n{"completed_updates":')
    save(q/'jobs/remote.json',{'state':'running','attempts':[{'id':'b','worker':'other.42'}],
                             'task':{'run_id':'foreign','kind':'train'}})
    before={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    job=probe.snapshot(tmp_path,'host')[0]['local_jobs'][0]
    assert job['reported_update_not_checkpoint_verified']==26
    assert len(probe.snapshot(tmp_path,'host')[0]['local_jobs'])==1
    assert before=={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


def test_loading_without_run_manifest_is_shown_without_claiming_training(tmp_path):
    q=tmp_path/'qwen_mbpp/.multi-node'
    save(q/'jobs/j.json',{'state':'running','attempts':[{'id':'a','worker':'host.42'}],
                        'task':{'run_id':'r','kind':'train'}})
    path=q/'logs/a.log';path.parent.mkdir(parents=True);path.write_text('Loading weights:100%\n')
    job=probe.snapshot(tmp_path,'host')[1]['local_jobs'][0]
    assert job['run_manifest']=='not_created'
    assert job['last_log_lines']==['Loading weights:100%']
    assert 'reported_update_not_checkpoint_verified' not in job


def test_short_hostname_cannot_adopt_a_different_fqdn_worker(tmp_path):
    queue=tmp_path/'olmo_math/.multi-node'
    save(queue/'jobs/foreign.json',dict(state='running',task=dict(run_id='foreign'),
         attempts=[dict(id='f',worker='node.cluster.example.123')]))
    assert not probe.snapshot(tmp_path,'node')[0]['local_jobs']
    assert len(probe.snapshot(tmp_path,'node.cluster.example')[0]['local_jobs'])==1
