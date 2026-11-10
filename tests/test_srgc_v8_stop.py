import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import pytest

from scripts import v8_processes as proc


def row(ppid,start,v8=False,root='/job',tag=None):
    env={'V8_ROOT':root}
    if tag:env[proc.TAG]=tag
    return dict(ppid=ppid,start=start,state='S',v8=v8,env=env)


def test_only_owned_launch_and_descendants_are_selected():
    rows={10:row(1,'a',v8=True),11:row(10,'b'),20:row(1,'c',v8=True,root='/other')}
    assert proc.select(rows,root='/job')=={10:'a',11:'b'}
    rows={10:row(1,'a',tag='ours'),11:row(1,'b',tag='ours'),20:row(1,'c',tag='other')}
    assert proc.select(rows,tag='ours')=={10:'a',11:'b'}


def test_orphaned_tagged_cpu_child_is_selected_by_root_without_selecting_unrelated_python():
    rows={10:row(1,'orphan',tag='old-launch'),20:row(1,'unrelated'),
          30:row(1,'other-root',root='/other',tag='old-launch')}
    assert proc.select(rows,root='/job')=={10:'orphan'}


def test_stop_command_releases_control_lock_inherited_by_orphaned_cpu_child(tmp_path):
    import fcntl
    root=tmp_path/'experiments';queue=root/'olmo_math/.multi-node';queue.mkdir(parents=True)
    lock=queue/'control.lock';lock.touch();inode=lock.stat().st_ino
    saved=queue/'checkpoint-sentinel';saved.write_bytes(b'optimizer RNG and reward records retained')
    child_script=tmp_path/'cpu-stage.py';pidfile=tmp_path/'cpu.pid'
    child_script.write_text('import os,signal,time,sys\nfrom pathlib import Path\n'
        'signal.signal(signal.SIGTERM,signal.SIG_IGN)\n'
        'Path(sys.argv[2]).write_text(str(os.getpid()))\n'
        'while True:time.sleep(.02)\n')
    launcher=tmp_path/'v8_dispatch.py'
    launcher.write_text('import os,fcntl,subprocess,sys\n'
        'fd=os.open(sys.argv[1],os.O_RDWR)\nfcntl.flock(fd,fcntl.LOCK_EX)\n'
        'subprocess.Popen([sys.executable,sys.argv[2],str(fd),sys.argv[3]],pass_fds=(fd,),'
        'stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)\nos.close(fd)\n')
    scripts=Path(__file__).resolve().parents[1]/'scripts';childpid=None;foreign=None
    env=dict(os.environ,V8_ROOT=str(root),V8_LAUNCH_ID='orphan-control-fixture')
    try:
        subprocess.run([sys.executable,str(launcher),str(lock),str(child_script),str(pidfile)],env=env,check=True,timeout=3)
        deadline=time.monotonic()+3
        while not pidfile.exists() and time.monotonic()<deadline:time.sleep(.005)
        childpid=int(pidfile.read_text())
        foreign_env=dict(os.environ,V8_ROOT=str(root));foreign_env.pop(proc.TAG,None)
        foreign=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],env=foreign_env)
        targets=proc.select(proc.table(),root=root)
        assert childpid in targets and foreign.pid not in targets
        # The launching V8 parent has exited. This file's inherited flock is
        # the sole remaining blocker, not its mere existence on disk.
        fd=os.open(lock,os.O_RDONLY)
        try:
            with pytest.raises(BlockingIOError):fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        finally:os.close(fd)
        result=subprocess.run(['sh',str(scripts/'v8_stop.sh')],capture_output=True,text=True,timeout=5,
            env=dict(os.environ,V8_ROOT=str(root),V8_STOP_WAIT_SECONDS='.05',PYTHONDONTWRITEBYTECODE='1'))
        assert result.returncode==0,result.stdout+result.stderr
        assert 'V8 processes stopped' in result.stdout
        fd=os.open(lock,os.O_RDONLY)
        try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        finally:os.close(fd)
        assert lock.stat().st_ino==inode and saved.read_bytes()==b'optimizer RNG and reward records retained'
        assert foreign.poll() is None
    finally:
        if childpid is not None:
            try:os.kill(childpid,signal.SIGKILL)
            except ProcessLookupError:pass
        if foreign is not None:
            foreign.kill();foreign.wait()


def test_reused_pid_and_unrelated_process_are_never_signalled():
    seen=[];rows={10:row(1,'new'),20:row(1,'safe')}
    proc.send({10:'old'},signal.SIGKILL,rows=rows,killer=lambda *args:seen.append(args))
    assert not seen


def test_supervisor_reads_current_attempt_progress_and_excludes_other_launch(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    from scripts import v8_supervisor as supervisor
    current=tmp_path/'attempt';current.mkdir()
    foreign=tmp_path/'other';foreign.mkdir()
    (current/'rank-0.json').write_text(json.dumps(dict(stage='policy_update',updated=time.time())))
    (foreign/'rank-0.json').write_text(json.dumps(dict(stage='foreign',updated=time.time())))
    ours=row(1,'a',tag='ours');ours['env']['SRGC_PROGRESS_DIR']=str(current)
    other=row(1,'b',tag='other');other['env']['SRGC_PROGRESS_DIR']=str(foreign)
    ours['kind']=other['kind']='task'
    events=supervisor.work_events({10:ours,20:other},'ours',tmp_path/'fallback')
    assert len(events)==1 and 'Optimizer update completed' in events[0] and 'foreign' not in events[0]
    assert supervisor.work_events({},'ours',current)==[]
    # Older coordinator inherits the supervisor directory, but still needs a live task.
    legacy=row(1,'c',tag='ours');legacy['kind']='task'
    assert supervisor.work_events({11:legacy},'ours',current)==events


def test_missing_model_work_shows_assigned_task_or_actual_blocker_not_preparing(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    from scripts import v8_supervisor as supervisor
    queue=tmp_path/'olmo_math/.multi-node';(queue/'jobs').mkdir(parents=True)
    (queue/'state.json').write_text(json.dumps(dict(phase='train',failure=None)))
    (queue/'jobs/a.json').write_text(json.dumps(dict(state='running',task=dict(run_id='r',kind='train',
         method_id='on_monitored',training_seed=5),attempts=[dict(id='a',worker='host.42')])))
    console=tmp_path/'console.log';console.write_text('NCCL admission running; inspect /actual/preflight.log\n')
    ours=row(1,'x',root=str(tmp_path),tag='ours')
    lines=supervisor.before_model_work({42:ours},'ours',tmp_path,console,host='host')
    text='\n'.join(lines)
    assert 'ASSIGNED olmo_math seed=5 method=on_monitored' in text
    assert '/actual/preflight.log' in text and 'Preparing this node' not in text
    assert 'not_created' in text and 'no model token/update record' in text
    # Same-node jobs from a different launch must not become evidence of our work.
    foreign=supervisor.before_model_work({99:row(1,'y',root=str(tmp_path),tag='ours')},'ours',tmp_path,console,host='host')
    assert not any(line.startswith('ASSIGNED olmo_math') for line in foreign)
    (queue/'state.json').write_text(json.dumps(dict(phase='prepare',failure=dict(detail='original source missing'))))
    blocked=supervisor.before_model_work({42:ours},'ours',tmp_path,console,host='host')
    assert any('BLOCKED olmo_math: original source missing' in line for line in blocked)


def test_detached_rank_ignoring_term_is_stopped_but_foreign_launch_survives(tmp_path):
    script=tmp_path/'cluster_child.py'
    script.write_text('import signal,time\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\nprint("READY",flush=True)\nwhile True:time.sleep(.02)\n')
    ours=subprocess.Popen([sys.executable,str(script)],env=dict(os.environ,V8_ROOT='/fixture-ours'),stdout=subprocess.PIPE,text=True,start_new_session=True)
    other=subprocess.Popen([sys.executable,str(script)],env=dict(os.environ,V8_ROOT='/fixture-other'),stdout=subprocess.PIPE,text=True,start_new_session=True)
    try:
        assert ours.stdout.readline().strip()=='READY';assert other.stdout.readline().strip()=='READY'
        targets=proc.select(proc.table(),root='/fixture-ours');assert ours.pid in targets and other.pid not in targets
        start=time.monotonic();assert proc.stop(targets,grace=.05)==[]
        ours.wait(timeout=3)
        assert time.monotonic()-start<3 and other.poll() is None
    finally:
        for p in [ours,other]:
            if p.poll() is None:p.kill()
            p.wait()


@pytest.mark.parametrize('normal_exit',[False,True])
def test_supervisor_second_interrupt_stops_detached_rank_without_stdout_join(tmp_path,normal_exit):
    scripts=Path(__file__).resolve().parents[1]/'scripts'
    for name in ['v8_supervisor.py','v8_processes.py','v8_probe.py','v8_control_output.py','v8_progress_text.py']:shutil.copyfile(scripts/name,tmp_path/name)
    rank=tmp_path/'cluster_child.py'
    rank.write_text('import signal,time\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\nprint("RANK_READY",flush=True)\nwhile True:time.sleep(.02)\n')
    bootstrap=tmp_path/'v8_bootstrap.py';pidfile=tmp_path/'rank.pid'
    bootstrap.write_text('import subprocess,sys,signal,time\nfrom pathlib import Path\n'
                         'p=subprocess.Popen([sys.executable,str(Path(__file__).with_name("cluster_child.py"))],start_new_session=True)\n'
                         'Path(__file__).with_name("rank.pid").write_text(str(p.pid))\n'
                         'signal.signal(signal.SIGTERM,signal.SIG_IGN)\nsignal.signal(signal.SIGINT,signal.SIG_IGN)\n'
                         +('' if normal_exit else 'while True:time.sleep(.02)\n'))
    parent=subprocess.Popen([sys.executable,str(tmp_path/'v8_supervisor.py')],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    childpid=None
    try:
        assert parent.stdout.readline().strip()=='RANK_READY'
        childpid=int(pidfile.read_text());start=time.monotonic()
        if not normal_exit:
            parent.send_signal(signal.SIGINT);time.sleep(.1);parent.send_signal(signal.SIGINT)
        parent.communicate(timeout=5)
        assert parent.returncode==(0 if normal_exit else 130)
        assert time.monotonic()-start<5
        assert childpid not in proc.table() or proc.table()[childpid]['state']=='Z'
    finally:
        if parent.poll() is None:parent.kill();parent.wait()
        if childpid is not None:
            try:os.kill(childpid,signal.SIGKILL)
            except ProcessLookupError:pass
