"""CPU end-to-end through the real node loop (srgc_v8_nodes.worker_loop_many) under the real supervisor, with fake
GPUs: a busy node picks up pushed code at its next task boundary without stopping its task, and a node child that
exits on a stuck inspection or a quiescent block is relaunched with a bounded backoff, while configuration exits are
not. CPU fixture only: no scientific measurement."""
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FILES = ('v8_supervisor.py', 'v8_processes.py', 'v8_probe.py', 'v8_control_output.py', 'v8_progress_text.py')
CHILD = '''import os,sys,time
while not any(os.path.exists(p) for p in sys.argv[1:]):time.sleep(.05)
'''
BOOT = '''import json,os,signal,socket,subprocess,sys,threading,time
from pathlib import Path
here=Path(__file__).resolve().parent;repo=Path(os.environ['REAL_REPO'])
sys.path[:0]=[str(repo),str(repo/'tests')]
from scripts import srgc_v8_nodes as nodes
from test_srgc_v8_sequential_cohorts import Runtime
version=(here/'version.txt').read_text().strip();base=Path(os.environ['V8_ROOT'])
with open(os.environ['BOOT_LOG'],'a') as out:out.write(version+'\\n')
boots=len(Path(os.environ['BOOT_LOG']).read_text().split())
scenario=os.environ['SCENARIO']
class Node(Runtime):
    def execute(self,task,matrix,fd,stop,device_fds,attempt):
        self.event('execute',task['run_id'])
        nodes.atomic(base/'executions'/f'{time.time_ns()}.json',dict(version=version,task=task['run_id'],attempt=attempt['id']))
        # the task process of this launch (v8_processes marks cluster_child.py as a task)
        child=subprocess.Popen([sys.executable,str(base/'cluster_child.py'),str(base/('release-'+task['run_id'])),str(base/'release-all')])
        while child.poll() is None:
            if stop.is_set():child.terminate();child.wait();return 75
            time.sleep(.05)
        if self.fail:return 1
        (self.root/('done-'+task['run_id'])).write_text('CPU fixture only')
        return 0
stop=threading.Event();signal.signal(signal.SIGTERM,lambda *_:stop.set())
order=('qwen_mbpp',)
if scenario=='blocked':names=('olmo_mbpp',);order=('olmo_mbpp','qwen_mbpp')   # this node lacks qwen_mbpp's runtime
else:names=('qwen_mbpp',)
pairs=[]
for name in names:
    runtime=Node(base,name,fail=scenario=='fail')
    pairs.append((nodes.Pool(runtime.root/'.multi-node',runtime.binding),runtime))
if scenario=='wedge' and boots==1:
    pool,runtime=pairs[0];pool.initialize();state=pool.queue_state();state['phase']='train';nodes.atomic(pool.root/'state.json',state)
    runtime.independent_startup=True;runtime.scan_timeout=.3;plan=runtime.plan
    def wedged(phase):time.sleep(2);return plan(phase)
    runtime.plan=wedged
rc=nodes.worker_loop_many(pairs,socket.gethostname()+'.'+str(os.getpid()),stop,poll=.05,cohort_order=order)
with open(base/'exits.txt','a') as out:out.write(str(rc)+'\\n')
sys.exit(rc)
'''


def git(cwd, *args):
    subprocess.run(['git', '-c', 'user.name=t', '-c', 'user.email=t@t', *args], cwd=cwd, check=True,
                   capture_output=True)


@pytest.fixture
def checkout(tmp_path):
    if shutil.which('git') is None:
        pytest.skip('git required')
    upstream = tmp_path / 'upstream.git'
    git(tmp_path, 'init', '-q', '--bare', str(upstream))
    work = tmp_path / 'node'
    git(tmp_path, 'clone', '-q', str(upstream), str(work))
    (work / 'scripts').mkdir()
    for name in FILES:
        shutil.copyfile(ROOT / 'scripts' / name, work / 'scripts' / name)
    (work / 'scripts' / 'v8_bootstrap.py').write_text(BOOT)
    (work / 'scripts' / 'version.txt').write_text('v1\n')
    (work / '.gitignore').write_text('__pycache__/\n')
    git(work, 'add', '-A'); git(work, 'commit', '-qm', 'v1'); git(work, 'push', '-q', 'origin', 'HEAD')
    branch = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=work, capture_output=True,
                            text=True).stdout.strip()
    git(work, 'branch', '-q', '--set-upstream-to', 'origin/' + branch)
    dev = tmp_path / 'dev'
    git(tmp_path, 'clone', '-q', str(upstream), str(dev))
    base = tmp_path / 'runs'; base.mkdir()
    (base / 'cluster_child.py').write_text(CHILD)
    return tmp_path, work, dev, base


def start(tmp_path, work, scenario, **extra):
    settings = dict(V8_AUTO_UPDATE_SECONDS='0.5', V8_UPDATE_SETTLE_SECONDS='1', V8_RELAUNCH_SECONDS='0.5',
                    V8_STOP_WAIT_SECONDS='5')
    settings.update(extra)
    env = dict(os.environ, V8_ROOT=str(tmp_path / 'runs'), BOOT_LOG=str(tmp_path / 'boots.txt'), SCENARIO=scenario,
               REAL_REPO=str(ROOT), PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='', V8_USER_RETRY_GRANT='0',
               V8_FSYNC='0', TMPDIR=str(tmp_path), **settings)
    env.pop('V8_SUPERVISOR_RELAUNCHES', None); env.pop('V8_SUPERVISOR_SKIP_UPDATE', None)
    out = (tmp_path / 'supervisor.out').open('w')
    return subprocess.Popen([sys.executable, str(work / 'scripts' / 'v8_supervisor.py')], env=env, stdout=out,
                            stderr=subprocess.STDOUT, start_new_session=True)


def push_v2(dev):
    (dev / 'scripts' / 'version.txt').write_text('v2\n')
    git(dev, 'commit', '-qam', 'v2'); git(dev, 'push', '-q', 'origin', 'HEAD')


def boots(tmp_path):
    path = tmp_path / 'boots.txt'
    return path.read_text().split() if path.exists() else []


def executions(base):
    return [json.loads(p.read_text()) for p in sorted((base / 'executions').glob('*.json'))]


def jobs(base, cohort='qwen_mbpp'):
    return {p.stem: json.loads(p.read_text()) for p in (base / cohort / '.multi-node' / 'jobs').glob('*.json')}


def output(tmp_path):
    return (tmp_path / 'supervisor.out').read_text()


def wait_for(predicate, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(.1)
    return False


def finish(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def test_busy_node_takes_pushed_code_at_its_next_task_boundary_without_stopping_the_task(checkout):
    tmp_path, work, dev, base = checkout
    process = start(tmp_path, work, 'tasks')
    try:
        assert wait_for(lambda: len(executions(base)) == 1, 30), output(tmp_path)
        first = executions(base)[0]
        assert first['version'] == 'v1'
        push_v2(dev)
        # The task keeps running; no restart, no checkout change and no second task while it runs.
        assert wait_for(lambda: 'takes no new task' in output(tmp_path), 15), output(tmp_path)
        time.sleep(3)
        assert boots(tmp_path) == ['v1'] and len(executions(base)) == 1, output(tmp_path)
        assert (work / 'scripts' / 'version.txt').read_text().strip() == 'v1'
        (base / ('release-' + first['task'])).write_text('done')
        # Right after that task ends the node restarts on v2, and the next task runs on v2.
        assert wait_for(lambda: boots(tmp_path) == ['v1', 'v2'], 30), output(tmp_path)
        assert (work / 'scripts' / 'version.txt').read_text().strip() == 'v2'
        assert wait_for(lambda: len(executions(base)) == 2, 30), output(tmp_path)
        assert executions(base)[1]['version'] == 'v2', 'a new task started on the old code'
        (base / 'release-all').write_text('done')
        assert process.wait(60) == 0, output(tmp_path)
        runs = [row['task'] for row in executions(base)]
        assert len(runs) == len(set(runs)) == 4, 'every task ran exactly once (forward progress, no repeat)'
        attempts = [a for job in jobs(base).values() for a in job['attempts']]
        assert len(attempts) == 4 and all(a.get('exit_code') == 0 for a in attempts), attempts
        assert json.loads((base / 'qwen_mbpp/.multi-node/state.json').read_text())['phase'] == 'complete'
    finally:
        finish(process)


def test_stuck_inspection_exit_is_relaunched_and_the_queue_then_completes(checkout):
    tmp_path, work, dev, base = checkout
    process = start(tmp_path, work, 'wedge')
    try:
        (base / 'release-all').write_text('done')
        assert process.wait(90) == 0, output(tmp_path)
        text = output(tmp_path)
        assert boots(tmp_path) == ['v1', 'v1'], text
        assert (base / 'exits.txt').read_text().split() == ['1', '0']
        assert 'task inspection did not finish' in text and 'relaunching this node in' in text
        assert 'task inspection timed out' in text
        runs = [row['task'] for row in executions(base)]
        assert len(runs) == len(set(runs)) == 4
        state = json.loads((base / 'qwen_mbpp/.multi-node/state.json').read_text())
        assert state['phase'] == 'complete' and state['failure'] is None
    finally:
        finish(process)


def test_failure_capped_block_is_relaunched_with_doubling_delay_and_never_reruns_capped_tasks(checkout):
    tmp_path, work, dev, base = checkout
    process = start(tmp_path, work, 'fail')
    try:
        (base / 'release-all').write_text('done')
        assert wait_for(lambda: len(boots(tmp_path)) >= 3, 60), output(tmp_path)
        before = {key: len(job['attempts']) for key, job in jobs(base).items()}
        assert before and all(n == 4 for n in before.values()), before   # MAX_ATTEMPTS, not more
        assert wait_for(lambda: len(boots(tmp_path)) >= 4, 30), output(tmp_path)
        text = output(tmp_path)
        assert 'every task it can serve is blocked by recorded failures' in text
        assert 'relaunching this node in 0s (relaunch 1 ' in text or 'relaunching this node in 1s (relaunch 1 ' in text
        assert 'relaunching this node in 1s (relaunch 2 ' in text and 'relaunching this node in 2s (relaunch 3 ' in text
        # Relaunches only re-inspect: no new attempt, no reset of failure records.
        assert {key: len(job['attempts']) for key, job in jobs(base).items()} == before
        assert len(executions(base)) == 8
        process.send_signal(signal.SIGTERM)
        assert process.wait(30) == 143
    finally:
        finish(process)


def test_configuration_blocked_exit_is_not_relaunched(checkout):
    tmp_path, work, dev, base = checkout
    process = start(tmp_path, work, 'blocked')
    try:
        (base / 'release-all').write_text('done')
        assert process.wait(60) == 1, output(tmp_path)
        time.sleep(1.5)
        text = output(tmp_path)
        assert 'BLOCKED: qwen_mbpp is unavailable on this node' in text, text
        assert boots(tmp_path) == ['v1'] and 'relaunching' not in text
    finally:
        finish(process)


def test_relaunch_reason_and_backoff():
    sys.path.insert(0, str(ROOT / 'scripts'))
    import v8_supervisor as sup

    def reason(tmp, *lines):
        path = tmp / 'log'; path.write_text('\n'.join(lines) + '\n')
        return sup.relaunch_reason(path)
    import tempfile
    with tempfile.TemporaryDirectory() as folder:
        tmp = Path(folder)
        start_lines = ('V8 startup: olmo_math not served by this node (x)', '[h.1] V8 worker: cohorts=qwen_mbpp')
        timeout = ('[h.1] EXIT: qwen_mbpp task inspection did not finish in 300s at plan; this worker exits so the '
                   'queue lock is released for the other nodes. Start the same command again on this node.')
        quiet = ('[h.1] STOPPED: no eligible work remains beside failed tasks; worker exits instead of waiting; '
                 '[{"cohort":"qwen_mbpp","failure_limit":4,"failures":4,"task":"t"}]; failed checkpoints/logs retained')
        assert reason(tmp, *start_lines, timeout, '[V8 STATUS] x state=BLOCKED y') == 'task inspection timed out'
        assert reason(tmp, *start_lines, '[h.1] WORKER EXIT: blocked_work; x', quiet).startswith('every task')
        # BLOCKED input (planner errors), shared queue failures and configuration exits are never relaunched.
        assert reason(tmp, quiet.replace('"failure_limit"', '"planning_errors":{"base":"x"},"failure_limit"')) is None
        assert reason(tmp, quiet.replace('"failure_limit"', '"queue_failure":{"detail":"x"},"failure_limit"')) is None
        assert reason(tmp, '[h.1] BLOCKED: qwen_mbpp is unavailable on this node; x') is None
        assert reason(tmp, '[h.1] EXIT: this node cannot serve the active experiment qwen_mbpp (x); later') is None
        assert reason(tmp, '[h.1] STOPPED: qwen_mbpp is not complete; later', '[h.1] WORKER EXIT: blocked_work') is None
        capped = ('[h.1] STOPPED: qwen_mbpp is not complete; later cohorts remain queued; {"at":1.0,"detail":"2 task(s) '
                  'failed; no independent ready/running work remains. Failed attempts/checkpoints are preserved.","task":null}')
        assert reason(tmp, capped, '[h.1] WORKER EXIT: blocked_work; x').startswith('every task')
        assert reason(tmp, capped.replace('"task":null', '"task":"t"')) is None
        assert reason(tmp, '[h.1] STOPPED: qwen_mbpp is not complete; later cohorts remain queued; '
                           '{"detail":"invalid frozen inputs after recheck","task":null}') is None
        assert reason(tmp, timeout, '[h.1] qwen_mbpp: FAILED: x', '[h.1] WORKER EXIT: y') is None
        assert reason(tmp, 'V8 queue error: ValueError: x') is None and reason(tmp, *start_lines) is None
    assert [sup.relaunch_delay(n, 60) for n in (1, 2, 3, 6, 7, 50)] == [60, 120, 240, 1800, 1800, 1800]
    assert sup.relaunch_base({}) == 60 and sup.relaunch_base({'V8_RELAUNCH_SECONDS': '0'}) == 0
    assert sup.update_interval({}) == 120 and sup.update_interval({'V8_AUTO_UPDATE_SECONDS': 'x'}) == 120


def test_update_hold_is_ignored_without_its_supervisor(tmp_path):
    from scripts import srgc_v8_nodes as nodes
    hold = tmp_path / 'hold.json'
    assert nodes.update_hold({}) is None and nodes.update_hold({'V8_UPDATE_HOLD': str(hold)}) is None
    hold.write_text(json.dumps(dict(target='abcdef0123', supervisor_pid=os.getppid())))
    assert nodes.update_hold({'V8_UPDATE_HOLD': str(hold)}).startswith('code update pending (abcdef0)')
    hold.write_text(json.dumps(dict(target='abcdef0123', supervisor_pid=os.getppid() + 999999)))
    assert nodes.update_hold({'V8_UPDATE_HOLD': str(hold)}) is None, 'an orphaned child must not idle forever'


def test_hold_is_acknowledged_only_after_a_pending_cpu_stage_ends(tmp_path, monkeypatch):
    """Real node loop, CPU fixture: while a CPU stage runs in the runtime actor the node never publishes WAIT_UPDATE
    (the supervisor's restart signal), and it claims no task while the hold exists."""
    import threading
    sys.path.insert(0, str(ROOT / 'tests'))
    from scripts import srgc_v8_nodes as nodes
    from test_srgc_v8_sequential_cohorts import pairs, events
    hold = tmp_path / 'hold.json'
    hold.write_text(json.dumps(dict(target='abcdef0123', supervisor_pid=os.getppid(), at=time.time())))
    monkeypatch.setenv('V8_UPDATE_HOLD', str(hold))
    rows = pairs(tmp_path, cohorts=('qwen_mbpp',))
    release = threading.Event()
    for _, runtime in rows:
        runtime.independent_startup = True; runtime.pending_barrier = None
        def begin(action, fd, stop, device_fds, runtime=runtime):
            runtime.barrier(action, fd, stop, device_fds); runtime.pending_barrier = (action, os.dup(fd))
        def poll(stop, runtime=runtime):
            if runtime.pending_barrier is None or not release.is_set():
                return None
            action, held = runtime.pending_barrier; runtime.pending_barrier = None
            return action, held, None
        runtime.begin_barrier = begin; runtime.poll_barrier = poll
    stop = threading.Event(); result = []
    thread = threading.Thread(target=lambda: result.append(nodes.worker_loop_many(
        rows, 'cpu-node', stop, poll=.02, cohort_order=('qwen_mbpp',))))
    thread.start()
    status = tmp_path / '.v8-worker-status' / (nodes.digest('cpu-node') + '.json')
    seen = []
    try:
        assert wait_for(lambda: rows[0][1].pending_barrier is not None, 30)
        end = time.monotonic() + 2
        while time.monotonic() < end:
            if status.exists():
                seen.append(json.loads(status.read_text())['state'])
            time.sleep(.02)
        assert seen and 'WAIT_UPDATE' not in seen, seen
        release.set()
        assert wait_for(lambda: status.exists() and json.loads(status.read_text())['state'] == 'WAIT_UPDATE', 30)
        time.sleep(.5)
        assert not [e for e in events(tmp_path) if e['stage'] == 'execute'], 'a task was claimed under the hold'
        hold.unlink()
        thread.join(60)
        assert result == [0]
        runs = [e['task'] for e in events(tmp_path) if e['stage'] == 'execute']
        assert len(runs) == len(set(runs)) == 4
    finally:
        stop.set(); release.set(); thread.join(10)


def test_restart_needs_the_nodes_wait_update_record_after_the_hold(tmp_path):
    sys.path.insert(0, str(ROOT / 'scripts'))
    import v8_supervisor as sup
    from scripts import srgc_v8_nodes as nodes
    rows = {4242: dict(env={sup.TAG: 'L', 'V8_ROOT': str(tmp_path)}, kind=None, ppid=1)}
    folder = tmp_path / '.v8-worker-status'; folder.mkdir()
    path = folder / (nodes.digest('h.4242') + '.json')
    def write(state, at):
        path.write_text(json.dumps(dict(worker='h.4242', launch_id='L', state=state, at=at)))
    write('CHECKING', 200.0)   # e.g. a GPU re-admission that started after the hold
    assert not sup.node_acknowledged(rows, 'L', tmp_path, 100.0, host='h')
    write('WAIT_UPDATE', 50.0)  # an older hold's record
    assert not sup.node_acknowledged(rows, 'L', tmp_path, 100.0, host='h')
    write('WAIT_UPDATE', 150.0)
    assert sup.node_acknowledged(rows, 'L', tmp_path, 100.0, host='h')
    assert not sup.node_acknowledged(rows, 'other', tmp_path, 100.0, host='h')


def test_checkout_moved_under_the_supervisor_counts_as_an_update(checkout):
    tmp_path, work, dev, base = checkout
    sys.path.insert(0, str(ROOT / 'scripts'))
    import v8_supervisor as sup
    updates = sup.Updates(work, 3600)
    try:
        assert updates.check() is None
        push_v2(dev)
        git(work, 'pull', '-q', '--ff-only')   # manual pull (or another node on a shared checkout)
        head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=work, capture_output=True, text=True).stdout.strip()
        assert updates.check() == head and head != updates.start
        (work / 'scripts' / 'version.txt').write_text('local edit\n')
        assert updates.check() is None, 'a checkout with local edits is never taken'
    finally:
        updates.closed.set(); updates.kick()
