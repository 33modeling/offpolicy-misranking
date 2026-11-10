"""Observed scheduling reasons and current-process progress; no GPU/model work."""
import importlib
import json
from pathlib import Path
import threading
import time

import pytest

from scripts import srgc_v8_nodes as nodes


@pytest.fixture
def supervisor(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    return importlib.import_module('v8_supervisor')


def test_completed_task_progress_is_not_reported_as_work_of_an_idle_launch(tmp_path,supervisor):
    (tmp_path/'rank-0.json').write_text(json.dumps(dict(stage='generation',updated=time.time())))
    control=dict(kind=None,state='S',env={'V8_LAUNCH_ID':'L','SRGC_PROGRESS_DIR':str(tmp_path)})
    assert supervisor.work_events({42:control},'L',tmp_path)==[]


def test_resumed_child_does_not_inherit_previous_attempt_progress(tmp_path,supervisor):
    started=time.time()
    (tmp_path/'rank-0.json').write_text(json.dumps(dict(stage='policy_update',updated=started-600)))
    child=dict(kind='task',state='S',started_at=started,env={'V8_LAUNCH_ID':'L','SRGC_PROGRESS_DIR':str(tmp_path)})
    assert supervisor.work_events({42:child},'L',tmp_path)==[]


def test_old_work_record_is_no_progress_even_while_process_is_alive(tmp_path,supervisor):
    now=time.time()
    event=dict(stage='generation',updated=now-300,progress_kind='returned_decode_steps',response_tokens=12)
    (tmp_path/'rank-0.json').write_text(json.dumps(event))
    child=dict(kind='task',state='S',started_at=now-600,env={'V8_LAUNCH_ID':'L','SRGC_PROGRESS_DIR':str(tmp_path)})
    text=' '.join(supervisor.work_events({42:child},'L',tmp_path))
    assert 'state=NO_PROGRESS' in text and 'last new token' in text and '300s ago' in text
    assert '12 tokens' in text
    event['updated']=time.time();(tmp_path/'rank-0.json').write_text(json.dumps(event))
    assert 'state=RUNNING' in ' '.join(supervisor.work_events({42:child},'L',tmp_path))


def test_diagnostic_heartbeat_does_not_reset_same_state_or_invent_model_progress(tmp_path,monkeypatch):
    clock={'wall':1000.,'mono':100.}
    monkeypatch.setattr(nodes.time,'time',lambda:clock['wall'])
    monkeypatch.setattr(nodes.time,'monotonic',lambda:clock['mono'])
    status=nodes.WorkerStatus(tmp_path,'host.42','controller')
    status.publish('WAIT_CHECKPOINT','olmo_math','parent not ready',dependencies=2)
    first=status.path.read_bytes()
    clock.update(wall=1005.,mono=105.)
    status.publish('WAIT_CHECKPOINT','olmo_math','parent not ready',dependencies=2)
    assert status.path.read_bytes()==first
    clock.update(wall=1020.,mono=120.)
    status.publish('WAIT_CHECKPOINT','olmo_math','parent not ready',dependencies=2)
    record=nodes.read(status.path)
    assert record['at']==1020. and record['since']==1000.
    assert not any(key in record for key in ('last_work','optimizer_steps','completed_updates'))


@pytest.mark.parametrize('mode,expected',[('parent','WAIT_CHECKPOINT'),('leased','WAIT_OTHER_WORKERS'),('gpu','WAIT_GPU')])
def test_worker_logs_observed_wait_reason_without_starting_later_cohort(tmp_path,capsys,mode,expected):
    from test_srgc_v8_sequential_cohorts import pairs,events
    rows=pairs(tmp_path);pool,runtime=rows[0];pool.initialize()
    state=pool.queue_state();state['phase']='train';nodes.atomic(pool.root/'state.json',state)
    leases=[]
    if mode=='parent':runtime.ready=lambda task:False
    if mode=='leased':
        for task in runtime.plan('train')[0]:
            key=pool.register(task,'train');lease=nodes.Lock(pool.root/'leases'/(key+'.lock'))
            assert lease.acquire();leases.append(lease)
    stop=threading.Event();timer=threading.Timer(.15,stop.set);timer.start()
    try:
        assert nodes.worker_loop_many(rows,'host.42',stop,poll=.005,cohort_order=nodes.DEFAULT_COHORTS,
                                      gpu_gate=(lambda:(False,'another process occupies the GPU set')) if mode=='gpu' else None)==75
    finally:
        timer.cancel();timer.join()
        for lease in leases:lease.close()
    text=capsys.readouterr().out
    assert 'state='+expected in text
    assert 'state=STOPPED' in text
    assert not any(row['stage']=='execute' for row in events(tmp_path))
    assert not (rows[1][0].root/'state.json').exists()


def test_supervisor_uses_own_scheduler_reason_and_rejects_previous_launch(tmp_path,supervisor,monkeypatch):
    monkeypatch.setenv('V8_LAUNCH_ID','L')
    status=nodes.WorkerStatus(tmp_path,'host.42','controller')
    status.publish('WAIT_OTHER_WORKERS','olmo_math','tasks hold execution leases',leased=10,ready=0)
    control=dict(kind=None,state='S',env={'V8_LAUNCH_ID':'L','V8_ROOT':str(tmp_path)})
    text=' '.join(supervisor.before_model_work({42:control},'L',tmp_path,tmp_path/'console.log',host='host'))
    assert 'state=WAIT_OTHER_WORKERS' in text and 'leased=10' in text and 'same state' in text
    assert 'NOT ASSIGNED' not in text
    record=nodes.read(status.path);record['launch_id']='previous';nodes.atomic(status.path,record)
    assert 'WAIT_OTHER_WORKERS' not in ' '.join(supervisor.before_model_work({42:control},'L',tmp_path,tmp_path/'console.log',host='host'))


def test_status_table_shows_wait_reason_from_same_controller(tmp_path,supervisor,monkeypatch):
    status=importlib.import_module('v8_status');now=time.time();worker='host.42'
    queue=tmp_path/'olmo_math/.multi-node';pool=nodes.Pool(queue,dict(cohort='olmo_math'));pool.initialize()
    nodes.atomic(queue/'workers'/(nodes.digest(worker)+'.json'),dict(worker=worker,pid=42,at=now,task=None,controller_sha256='C'))
    diagnostic=nodes.WorkerStatus(tmp_path,worker,'C');diagnostic.publish('WAIT_CHECKPOINT','olmo_math','parent checkpoint missing',dependencies=2)
    rows=status.node_table([tmp_path],['olmo_math'],now,this_host='other-host')
    assert len(rows)==1 and 'WAIT_CHECKPOINT' in rows[0]['doing'] and 'dependencies=2' in rows[0]['doing']
    record=nodes.read(diagnostic.path);record['controller_sha256']='old';nodes.atomic(diagnostic.path,record)
    assert 'WAIT_CHECKPOINT' not in status.node_table([tmp_path],['olmo_math'],now,this_host='other-host')[0]['doing']


def test_diagnostic_write_failure_does_not_fail_training_or_reset_checkpoints(tmp_path,monkeypatch):
    from test_srgc_v8_sequential_cohorts import pairs
    original=nodes.atomic
    def fail_diagnostics(path,value):
        if '.v8-worker-status' in Path(path).parts:raise PermissionError('diagnostic fixture')
        return original(path,value)
    monkeypatch.setattr(nodes,'atomic',fail_diagnostics)
    assert nodes.worker_loop_many(pairs(tmp_path),'host.42',threading.Event(),poll=.001,
                                  cohort_order=nodes.DEFAULT_COHORTS)==0


def test_wedged_verification_has_finite_exit_and_cannot_initialize_after_retirement(tmp_path):
    pool=nodes.Pool(tmp_path/'olmo_math/.multi-node',dict(cohort='olmo_math'))
    release=threading.Event()
    class Runtime:
        independent_startup=True;timeout=.05;verify_timeout=.05;pending_barrier=None   # V8-159: own verification limit
        verification_stage='saving shared dispatch policy receipt'
        def verify_node(self):release.wait(1)
    runtime=Runtime();start=time.monotonic()
    try:
        assert nodes.worker_loop_many([(pool,runtime)],'node',threading.Event(),poll=.005,
                                      cohort_order=('olmo_math',))==1
        assert time.monotonic()-start<.5
    finally:release.set()
    time.sleep(.03)
    assert not (pool.root/'binding.json').exists() and not (pool.root/'state.json').exists()


def test_queue_control_contention_is_reported_and_ready_work_resumes_after_release(tmp_path,capsys,monkeypatch):
    from contextlib import contextmanager
    from test_srgc_v8_sequential_cohorts import pairs,events
    rows=pairs(tmp_path);pool,runtime=rows[0];original=pool.control
    verified=threading.Event();verify=runtime.verify_node
    def verify_node():verify();verified.set()
    runtime.verify_node=verify_node
    @contextmanager
    def contended():
        if verified.is_set():yield None
        else:
            with original() as control:yield control
    monkeypatch.setattr(pool,'control',contended)
    stop=threading.Event();timer=threading.Timer(.08,stop.set);timer.start()
    try:
        assert nodes.worker_loop_many(rows,'waiting-node',stop,poll=.002,cohort_order=nodes.DEFAULT_COHORTS)==75
    finally:timer.cancel();timer.join()
    text=capsys.readouterr().out
    assert 'state=WAIT_QUEUE_LOCK' in text and str(pool.root/'control.lock') in text
    assert 'state=WAIT_UNEXPLAINED' not in text
    assert not any(row['stage']=='execute' for row in events(tmp_path))
    monkeypatch.setattr(pool,'control',original)
    assert nodes.worker_loop_many(rows,'waiting-node',threading.Event(),poll=.002,
                                  cohort_order=nodes.DEFAULT_COHORTS)==0


def test_wedged_task_inspection_exits_without_claiming_or_failing_shared_queue(tmp_path):
    from test_srgc_v8_sequential_cohorts import pairs,events
    rows=pairs(tmp_path);pool,runtime=rows[0]
    pool.initialize();state=pool.queue_state();state['phase']='train';nodes.atomic(pool.root/'state.json',state)
    runtime.independent_startup=True;runtime.scan_timeout=.04
    release=threading.Event();plan=runtime.plan
    def wedged(phase):release.wait(1);return plan(phase)
    runtime.plan=wedged
    stop=threading.Event();timer=threading.Timer(.5,stop.set);timer.start()
    try:
        assert nodes.worker_loop_many(rows,'wedged-inspector',stop,poll=.002,
                                      cohort_order=nodes.DEFAULT_COHORTS)==1
        assert not stop.is_set()
    finally:
        release.set();timer.cancel();timer.join()
    time.sleep(.03)
    assert pool.queue_state()['phase']=='train' and pool.queue_state()['failure'] is None
    assert not any(row['stage']=='execute' for row in events(tmp_path))
    assert all(not nodes.read(p)['attempts'] for p in (pool.root/'jobs').glob('*.json'))
    assert not (rows[1][0].root/'state.json').exists()
