"""Bounded V8 Ctrl-C handling without changing trainer/runtime queue identities."""
import os
import json
import math
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid

from v8_processes import TAG,select,progress_every,stop,stop_grace,table
from v8_probe import read,root,snapshot,tail,worker_status
from v8_control_output import _pending,emit,run_control
from v8_progress_text import describe,scheduler_text


def work_events(rows,launch,fallback):
    """Read the active launch's actual per-attempt directory, never another launch's progress."""
    tasks=[row for row in rows.values() if row['env'].get(TAG)==launch and row.get('kind')=='task' and row.get('state')!='Z']
    if not tasks:return []  # launch/control processes do not prove a training child is still alive
    directories={}
    for row in tasks:
        directory=Path(row['env'].get('SRGC_PROGRESS_DIR') or fallback)
        started=row.get('started_at')
        if not isinstance(started,(int,float)):
            try:started=time.time()-time.clock_gettime(time.CLOCK_BOOTTIME)+float(row['start'])/os.sysconf('SC_CLK_TCK')
            except (KeyError,ValueError,TypeError,AttributeError,OSError):started=None
        directories.setdefault(directory,[]).append(started)
    parts=[]
    for directory,starts in sorted(directories.items()):
        valid_starts=[value for value in starts if value is not None]
        for rank in range(4):
            event=read(directory/f'rank-{rank}.json')
            if isinstance(event,dict) and isinstance(event.get('updated'),(int,float)):
                if valid_starts and event['updated']<min(valid_starts):continue
                age=time.time()-event['updated']
                kind='stage entered' if event.get('progress_kind')=='entered_only' else 'last new token' if event.get('progress_kind')=='returned_decode_steps' else 'last work record'
                state='NO_PROGRESS' if age>=120 else 'STARTING_TASK' if event.get('progress_kind')=='entered_only' else 'RUNNING'
                stamp=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(event['updated']))
                parts.append(f'state={state} | GPU process {rank} | {describe(event.get("stage","?"),event)} | {kind} {stamp} ({age:.0f}s ago)')
    return parts


def before_model_work(rows,launch,base,console,host=None):
    """Use the existing queue receipts/logs like SRGC NODE/BLOCKED status, never infer preparation."""
    host=host or socket.gethostname()
    owned={str(pid):row for pid,row in rows.items() if row['env'].get(TAG)==launch}
    bases={Path(row['env']['V8_ROOT']) for row in owned.values() if row['env'].get('V8_ROOT')} or {Path(base)}
    lines=[]
    for folder in sorted(bases):
        diagnoses=[]
        for pid in owned:
            record=worker_status(folder,host+'.'+pid)
            if record and record.get('launch_id')==launch:
                diagnoses.append(record)
        if diagnoses:
            record=max(diagnoses,key=lambda r:r['at'])
            lines.append('NODE '+host+': '+scheduler_text(record,time.time()))
            if record.get('log'):lines.append('TASK LOG: '+record['log'])
            continue
        for item in snapshot(folder,host):
            label=item['cohort'];failure=item.get('failure')
            for error in item.get('record_errors',[]):lines.append('STATUS ERROR '+label+': '+error)
            if failure:
                detail=failure.get('detail',str(failure)) if isinstance(failure,dict) else str(failure)
                lines.append(f'BLOCKED {label}: {detail}; queue={folder/label/".multi-node/state.json"}')
            mine=[job for job in item['local_jobs'] if job.get('worker','').rsplit('.',1)[-1] in owned]
            for job in mine:
                age=max(0,time.time()-job['attempt_started']) if isinstance(job.get('attempt_started'),(int,float)) else None
                status='NO_PROGRESS' if age is not None and age>=120 else 'ASSIGNED'
                lines.append(f'ASSIGNED {label} seed={job.get("seed")} method={job.get("method")}; '
                             f'state={status}; no model token/update record for this launch yet'
                             +(f' ({age:.0f}s since assignment)' if age is not None else '')+
                             f'; run state={job["run_manifest"]}; log={job["log"]}')
                if job['last_log_lines']:lines.append('TASK OUTPUT: '+job['last_log_lines'][-1])
            if not mine and not failure:
                counts=', '.join(f'{k}={v}' for k,v in sorted(item['recorded_job_counts'].items())) or 'no recorded jobs'
                lines.append(f'NOT ASSIGNED {label}: queue phase={item["phase"]}; recorded queue jobs: {counts}')
    recent=[line.strip() for line in tail(console,8192) if line.strip() and 'known_task_counts' not in line]
    if recent:lines.append('LATEST LAUNCHER OUTPUT: '+recent[-1][:1200])
    return lines or ['No task/progress evidence is readable for this launch; current execution phase is unknown.']


def describe_processes(remaining):
    kinds={}
    for row in remaining.values():kinds[row.get('kind') or 'control']=kinds.get(row.get('kind') or 'control',0)+1
    names={'task':'training/evaluation','cpu':'CPU stage','control':'dispatcher/actor'}
    return ', '.join(f'{count} {names.get(kind,kind)}' for kind,count in sorted(kinds.items()))


DEFAULT_UPDATE_SECONDS=120.0
DEFAULT_SETTLE_SECONDS=8.0     # > 2 dispatcher polls (3 s): a claim already in flight shows up before the restart
DEFAULT_RELAUNCH_SECONDS=60.0
RELAUNCH_CAP_SECONDS=1800.0
# Worker scheduler states (srgc_v8_nodes WorkerStatus) in which this launch holds a task without a task process yet
# (claimed, preflight, recording its result) or runs a CPU stage: never restart there.
TASK_STATES=frozenset({'ASSIGNED','PREFLIGHT','RECORDING_RESULT','CPU_STAGE'})
# Last line a node child prints before rc 1 (srgc_v8_nodes worker loop / main); 'WORKER EXIT' summaries are skipped.
EXIT_LINE=re.compile(r'^\[[^\]]*\] (?:EXIT|STOPPED|BLOCKED|\S+: FAILED):|^V8 (?:queue error|startup)')
# srgc_v8_nodes.aggregate_task_failure: every task failed up to its cap (never an input/setup/identity failure).
AGGREGATE_FAILURE=re.compile(r'"detail":"\d+ task\(s\) failed; no independent ready/running work remains\.[^"]*","task":null')


def _seconds(environment,name,default):
    value=(os.environ if environment is None else environment).get(name,'')
    try:seconds=float(value) if value.strip() else default
    except ValueError:return default
    return max(0.0,seconds) if math.isfinite(seconds) else default


def update_interval(environment=None):
    """Seconds between code checks (V8_AUTO_UPDATE_SECONDS, default 120; 0 turns updates off).
    A check also runs right after each task of this node ends."""
    return _seconds(environment,'V8_AUTO_UPDATE_SECONDS',DEFAULT_UPDATE_SECONDS)


def settle_seconds(environment=None):
    return _seconds(environment,'V8_UPDATE_SETTLE_SECONDS',DEFAULT_SETTLE_SECONDS)


def relaunch_base(environment=None):
    """First delay before relaunching a node child that exited on a stuck inspection or a quiescent block
    (V8_RELAUNCH_SECONDS, default 60; 0 turns relaunching off)."""
    return _seconds(environment,'V8_RELAUNCH_SECONDS',DEFAULT_RELAUNCH_SECONDS)


def relaunch_delay(count,base):
    """Doubling backoff, bounded by RELAUNCH_CAP_SECONDS (count = relaunches in a row without a task run)."""
    return min(max(base,RELAUNCH_CAP_SECONDS),base*2**min(max(int(count),1)-1,32))


def _git(repo,run,*args,timeout=60):
    result=run(['git','-C',str(repo),*args],capture_output=True,text=True,timeout=timeout)
    return result.returncode,result.stdout.strip()


def _clean(repo,run):
    code,changes=_git(repo,run,'status','--porcelain','--untracked-files=no')
    return code==0 and not changes


def pending_update(repo,run=subprocess.run,skip=None):
    """Fetch and return the upstream commit a fast-forward would reach, else None. Never touches the working tree.

    None for a checkout with local edits, a diverged checkout, no upstream, or the commit in `skip`."""
    try:
        if not (Path(repo)/'.git').exists():return None
        if _git(repo,run,'fetch','-q',timeout=180)[0]:return None
        if not _clean(repo,run):return None
        code,old=_git(repo,run,'rev-parse','HEAD');code2,upstream=_git(repo,run,'rev-parse','@{u}')
        if code or code2 or not old or not upstream or old==upstream or upstream==skip:return None
        if _git(repo,run,'merge-base','--is-ancestor','HEAD',upstream)[0]:return None
        return upstream
    except (OSError,subprocess.SubprocessError):return None


def fast_forward(repo,target,run=subprocess.run):
    """Fast-forward HEAD to exactly `target`; return (old, new) when HEAD moved, else None. Never merges,
    rebases or touches local edits."""
    try:
        if not _clean(repo,run):return None
        code,old=_git(repo,run,'rev-parse','HEAD')
        if code or not old or old==target:return None
        if _git(repo,run,'merge-base','--is-ancestor','HEAD',target)[0]:return None
        if _git(repo,run,'merge','--ff-only','-q',target)[0]:return None
        code,new=_git(repo,run,'rev-parse','HEAD')
        return (old,new) if not code and new==target else None
    except (OSError,subprocess.SubprocessError):return None


def checkout_update(repo,run=subprocess.run):
    """Fast-forward the code checkout to its upstream; return (old, new) when HEAD moved, else None."""
    target=pending_update(repo,run)
    return fast_forward(repo,target,run) if target else None


def launch_busy(rows,launch):
    """A training/evaluation process of this launch is running (no code update under a task)."""
    return any(rows[pid].get('kind') for pid in select(rows,tag=launch) if pid in rows)


def node_busy(rows,launch,base,host=None):
    """A task process runs, or the launch's dispatcher holds a task (claimed/preflight/recording) or a CPU stage."""
    if launch_busy(rows,launch):return True
    host=host or socket.gethostname()
    owned={pid:row for pid,row in rows.items() if row['env'].get(TAG)==launch}
    folders={Path(row['env']['V8_ROOT']) for row in owned.values() if row['env'].get('V8_ROOT')} or {Path(base)}
    for folder in folders:
        for pid in owned:
            record=worker_status(folder,host+'.'+str(pid))
            if record and record.get('launch_id')==launch and record.get('state') in TASK_STATES:return True
    return False


def node_acknowledged(rows,launch,base,since,host=None):
    """The launch's dispatcher published WAIT_UPDATE at or after `since` (the hold's wall time): a whole loop pass
    then claimed nothing because of the hold, with no CPU stage pending (srgc_v8_nodes). Without this record the
    restart waits, so a claim, a CPU stage or a GPU re-admission that began just before the hold is never cut off."""
    host=host or socket.gethostname()
    owned={pid:row for pid,row in rows.items() if row['env'].get(TAG)==launch}
    folders={Path(row['env']['V8_ROOT']) for row in owned.values() if row['env'].get('V8_ROOT')} or {Path(base)}
    for folder in folders:
        for pid in owned:
            record=worker_status(folder,host+'.'+str(pid))
            if (record and record.get('launch_id')==launch and record.get('state')=='WAIT_UPDATE'
                    and record.get('at',0)>=since):return True
    return False


def relaunch_reason(console):
    """Why the node child exited with rc 1, when relaunching it is safe; None otherwise (configuration errors,
    BLOCKED input, a node that cannot serve the active cohort, shared queue failures, unknown exits)."""
    for line in reversed(tail(console,65536)):
        line=line.strip()
        if not EXIT_LINE.search(line):continue
        if '] EXIT: ' in line and ' did not finish in ' in line and 'queue lock is released' in line:
            return 'task inspection timed out'
        if ('] STOPPED: no eligible work remains beside failed tasks' in line
                and 'planning_errors' not in line and 'queue_failure' not in line):
            return 'every task it can serve is blocked by recorded failures'
        if ' is not complete; later cohorts remain queued; ' in line and '] STOPPED: ' in line and AGGREGATE_FAILURE.search(line):
            return 'every task it can serve is blocked by recorded failures'
        return None
    return None


class Updates:
    """Background `git fetch`: the newest fast-forward target, checked every interval and when kicked.

    A checkout moved under this supervisor (a manual `git pull`, or a checkout shared by several nodes that another
    node already fast-forwarded) also counts: this launch still runs the code it started with."""
    def __init__(self,repo,interval,skip=None,run=subprocess.run):
        self.repo,self.interval,self.skip,self.git=repo,interval,skip,run
        self.start=self.head()
        self.target=None;self.kicked=threading.Event();self.closed=threading.Event()
        threading.Thread(target=self.run,daemon=True).start()

    def kick(self):self.kicked.set()

    def head(self):
        try:
            if not (Path(self.repo)/'.git').exists():return None
            code,value=_git(self.repo,self.git,'rev-parse','HEAD')
            return value if not code and value else None
        except (OSError,subprocess.SubprocessError):return None

    def check(self):
        target=pending_update(self.repo,self.git,skip=self.skip)
        if target is None:
            head=self.head()
            try:clean=bool(head) and _clean(self.repo,self.git)
            except (OSError,subprocess.SubprocessError):clean=False
            if clean and self.start and head!=self.start and head!=self.skip:target=head
        return target

    def run(self):
        while not self.closed.is_set():
            self.kicked.wait(self.interval);self.kicked.clear()
            if self.closed.is_set():return
            self.target=self.check()


def main():
    args=sys.argv[1:];bootstrap=Path(__file__).with_name('v8_bootstrap.py')
    if any(a in ('--status','--help','-h') for a in args):
        os.execv(sys.executable,[sys.executable,str(bootstrap),*args])
    launch=uuid.uuid4().hex;env=dict(os.environ,**{TAG:launch})
    progress=root(env)/'.node-progress'/socket.gethostname()/launch
    env['SRGC_PROGRESS_DIR']=str(progress)
    hold=progress/'code-update-hold.json';env['V8_UPDATE_HOLD']=str(hold)
    stopped=threading.Event();force=threading.Event();finished=threading.Event();disconnected=threading.Event();received=[]
    child=None;relay_thread=None
    def interrupt(sig,*_):
        if stopped.is_set():force.set()
        else:received.append(sig);stopped.set()
    previous={sig:signal.signal(sig,interrupt) for sig in (signal.SIGINT,signal.SIGTERM)}
    previous[signal.SIGHUP]=signal.signal(signal.SIGHUP,lambda *_:disconnected.set())
    def report():
        while not finished.wait(15):
            rows=table();parts=work_events(rows,launch,progress)
            if not parts:
                try:parts=before_model_work(rows,launch,root(env),log)
                except (OSError,ValueError,TypeError,KeyError) as exc:
                    parts=[f'STATUS ERROR: cannot read this launch state: {type(exc).__name__}: {exc}; console log={log}']
            try:print('[V8 STATUS] '+' | '.join(parts),flush=True)
            except (OSError,ValueError):return  # disconnected console must not stop its independent child
    def relay(log):
        with log.open('rb') as reader:
            while True:
                block=reader.read(65536)
                if not block and finished.is_set():
                    block=reader.read(65536)
                    if not block:return
                if block:
                    try:
                        stream=getattr(sys.stdout,'buffer',None)
                        if stream is None:sys.stdout.write(block.decode(errors='replace'));sys.stdout.flush()
                        else:stream.write(block);stream.flush()
                    except (OSError,ValueError):return
                else:finished.wait(.05)
    try:
        if stopped.is_set():return 128+received[0]
        directory=Path(tempfile.gettempdir())/f'v8-console-{os.getuid()}'
        directory.mkdir(mode=0o700,exist_ok=True)
        if directory.is_symlink() or directory.stat().st_uid!=os.getuid() or directory.stat().st_mode&0o077:
            raise ValueError('V8 console directory must be private and owned by this user')
        log=directory/(launch+'.log');env['V8_CONSOLE_LOG']=str(log)
        with log.open('xb') as handle:
            child=subprocess.Popen([sys.executable,str(bootstrap),*args],env=env,start_new_session=True,
                                   stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT)
        relay_thread=threading.Thread(target=relay,args=(log,),daemon=True);relay_thread.start()
        threading.Thread(target=report,daemon=True).start()
        interval=update_interval();repo=Path(__file__).resolve().parents[1];host=socket.gethostname()
        updates=Updates(repo,interval,skip=os.environ.get('V8_SUPERVISOR_SKIP_UPDATE')) if interval else None
        next_sample=0.0;was_busy=False;ran_task=False;held=None;held_at=0.0;held_wall=0.0;settle=settle_seconds()
        def clear_hold():
            try:hold.unlink()
            except OSError:pass
        def restart(extra):
            """Replace this supervisor (and so the launch) with a fresh one from the checkout as it is now."""
            finished.set();clear_hold()
            if relay_thread:relay_thread.join(timeout=2)
            for sig,handler in previous.items():signal.signal(sig,handler)
            os.environ.update(extra)
            deadline=time.monotonic()+.5
            for done in list(_pending):done.wait(max(0,deadline-time.monotonic()))  # bounded flush of messages
            os.execv(sys.executable,[sys.executable,str(Path(__file__).resolve()),*args])
        while child.poll() is None:
            now=time.monotonic()
            if now>=next_sample:
                next_sample=now+1.0
                busy=node_busy(table(),launch,root(env),host);ran_task=ran_task or busy
                # User 11-17/18: pushed fixes must reach busy nodes too. A task boundary (no task of this launch any
                # more) triggers a check at once; a periodic check runs every V8_AUTO_UPDATE_SECONDS.
                if updates and was_busy and not busy:updates.kick()
                was_busy=busy
                target=updates.target if updates else None
                if target is None and held:clear_hold();held=None
                elif target and target!=held:
                    # The dispatcher takes no new task while this hold exists; the running task is never touched.
                    try:
                        hold.parent.mkdir(parents=True,exist_ok=True)
                        held_wall=time.time()
                        tmp=hold.with_name(hold.name+'.tmp');tmp.write_text(json.dumps(dict(target=target,supervisor_pid=os.getpid(),at=held_wall)))
                        os.replace(tmp,hold);held=target;held_at=now
                        emit(f'[v8] pushed code {target[:7]} is ready; this node takes no new task and restarts with it '
                             +('when its running task ends (the task is not stopped)' if busy else 'now (no task running)'))
                    except OSError as exc:
                        # Without the hold a claim could race the restart: no automatic update for this launch.
                        emit(f'[v8] cannot write the code-update hold {hold}: {exc}; automatic code update is off '
                             'for this launch',error=True);updates.closed.set();updates.kick();updates=None
                elif (target and not busy and now-held_at>=settle and node_acknowledged(table(),launch,root(env),held_wall,host)
                      and not node_busy(table(),launch,root(env),host)):
                    emit(f'[v8] task boundary: restarting this node with pushed code {target[:7]} '
                         '(no task running; queue and checkpoints untouched)')
                    remaining=stop(select(table(),tag=launch),tag=launch,grace=stop_grace())
                    if remaining:emit('[v8] code-update restart: processes still in kernel wait: '+str(remaining),error=True)
                    else:
                        moved=fast_forward(repo,target)
                        if not moved and updates.head()==target:moved=(updates.start or '?',target)  # checkout already there
                        if moved:emit(f'[v8] code updated {moved[0][:7]} -> {moved[1][:7]}; restarting this node with it')
                        else:emit(f'[v8] code update to {target[:7]} was refused by git (fast-forward only); '
                                  'restarting on the current code; this commit is not tried again by this node',error=True)
                        restart({'V8_SUPERVISOR_RELAUNCHES':'0',**({} if moved else {'V8_SUPERVISOR_SKIP_UPDATE':target})})
            if disconnected.is_set():
                disconnected.clear()
                try:
                    progress.mkdir(parents=True,exist_ok=True)
                    with (progress/'session-disconnect.jsonl').open('a') as output:
                        output.write(json.dumps(dict(at=time.time(),signal='SIGHUP',action='continue_current_launch',
                                                     supervisor_pid=os.getpid(),console_log=str(log)))+'\n')
                except OSError:pass  # losing a terminal or metadata write cannot cancel the learner
                emit(f'[v8] terminal disconnected (SIGHUP); this launch continues. Console log: {log}')
            if stopped.wait(.1):
                grace=stop_grace()
                emit(f'[v8 stop] stopping this launch: SIGTERM sent; waiting up to {grace:g}s (V8_STOP_WAIT_SECONDS) '
                     'for a running task to stop at its next safe point, record exit 75 and exit (resumable from its '
                     'last saved update; a rank blocked in a collective is ended by the child watchdog). '
                     'Press Ctrl-C again to kill everything immediately.')
                def waiting(remaining,elapsed):
                    events=work_events(table(),launch,progress)
                    emit(f'[v8 stop] still waiting {elapsed:.0f}/{grace:g}s for {describe_processes(remaining)} process(es) to '
                         'reach a safe update boundary'+(f'; {" | ".join(events)}' if events else '')+
                         '; Ctrl-C again kills now')
                remaining=stop(select(table(),tag=launch),tag=launch,grace=grace,force=force,progress=waiting,every=progress_every())
                if remaining:emit('[v8 stop] processes still in kernel wait: '+str(remaining),error=True)
                return 128+received[0]
        # No stdout pipe join: descendants cannot trap the controller in read().
        remaining=stop(select(table(),tag=launch),tag=launch,grace=1)
        if remaining:emit('[v8 stop] remaining owned processes: '+str(remaining),error=True)
        if remaining:return 3
        reason=relaunch_reason(log) if child.returncode==1 and not stopped.is_set() else None
        base=relaunch_base()
        if reason and base:
            # MISSED-2 (11-18): a stuck task inspection or a quiescent block ends the node child with rc 1; the node
            # comes back by itself (doubling delay up to RELAUNCH_CAP_SECONDS). Configuration/BLOCKED input exits
            # are not relaunched. A relaunch never resets a task, a failure count or a checkpoint.
            try:count=0 if ran_task else int(os.environ.get('V8_SUPERVISOR_RELAUNCHES','0') or 0)
            except ValueError:count=0
            count+=1;delay=relaunch_delay(count,base)
            emit(f'[v8] node child exited rc=1 ({reason}); relaunching this node in {delay:.0f}s '
                 f'(relaunch {count} without a task run in between; delay doubles up to {max(base,RELAUNCH_CAP_SECONDS):.0f}s; '
                 'queues and checkpoints untouched; Ctrl-C cancels). Console log: '+str(log))
            if stopped.wait(delay):return 128+received[0]
            extra={'V8_SUPERVISOR_RELAUNCHES':str(count)}
            if interval:
                moved=checkout_update(repo)  # no child runs now: a pending fast-forward is safe
                if moved:emit(f'[v8] code updated {moved[0][:7]} -> {moved[1][:7]} before the relaunch')
            restart(extra)
        return child.returncode if child.returncode>=0 else 128-child.returncode
    finally:
        finished.set()
        try:hold.unlink()
        except OSError:pass
        leftovers=select(table(),tag=launch)
        if leftovers:stop(leftovers,tag=launch,grace=0 if stopped.is_set() else 1)
        if relay_thread:relay_thread.join(timeout=.2)
        for sig,handler in previous.items():signal.signal(sig,handler)


if __name__=='__main__':run_control(main)
