"""CPU end-to-end: an idle node's supervisor fast-forwards its git checkout and restarts with the pushed code;
a node with a running task process is never restarted (user 11-17 'idle인것도 고쳐줘')."""
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
BOOT = '''import hashlib,json,os,socket,subprocess,sys,time
from pathlib import Path
here=Path(__file__).resolve().parent
version=(here/'version.txt').read_text().strip()
with open(os.environ['BOOT_LOG'],'a') as out:out.write(version+'\\n')
if os.environ.get('FAKE_TASK'):
    # a training child of this launch (v8_processes marks cluster_child.py as a task)
    (here/'cluster_child.py').write_text('import time\\ntime.sleep(60)\\n')
    subprocess.Popen([sys.executable,str(here/'cluster_child.py')])
# Like the real node loop: once the supervisor's hold exists, a loop pass that claims nothing records WAIT_UPDATE.
worker=socket.gethostname()+'.'+str(os.getpid())
status=Path(os.environ['V8_ROOT'])/'.v8-worker-status'/(hashlib.sha256(json.dumps(worker).encode()).hexdigest()+'.json')
end=time.time()+60
while time.time()<end:
    if os.path.exists(os.environ.get('V8_UPDATE_HOLD','')) and not status.exists():
        status.parent.mkdir(parents=True,exist_ok=True)
        status.write_text(json.dumps(dict(worker=worker,launch_id=os.environ['V8_LAUNCH_ID'],state='WAIT_UPDATE',at=time.time())))
    time.sleep(.05)
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
    (work / '.gitignore').write_text('scripts/cluster_child.py\n__pycache__/\n')
    git(work, 'add', '-A'); git(work, 'commit', '-qm', 'v1'); git(work, 'push', '-q', 'origin', 'HEAD')
    git(work, 'branch', '-q', '--set-upstream-to', 'origin/' + subprocess.run(
        ['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=work, capture_output=True, text=True).stdout.strip())
    dev = tmp_path / 'dev'
    git(tmp_path, 'clone', '-q', str(upstream), str(dev))
    return tmp_path, work, dev


def start(tmp_path, work, **extra):
    env = dict(os.environ, V8_ROOT=str(tmp_path / 'runs'), V8_STOP_WAIT_SECONDS='3', V8_AUTO_UPDATE_SECONDS='0.3',
               BOOT_LOG=str(tmp_path / 'boots.txt'), PYTHONDONTWRITEBYTECODE='1', **extra)
    return subprocess.Popen([sys.executable, str(work / 'scripts' / 'v8_supervisor.py')], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, start_new_session=True)


def push_v2(dev):
    (dev / 'scripts' / 'version.txt').write_text('v2\n')
    git(dev, 'commit', '-qam', 'v2'); git(dev, 'push', '-q', 'origin', 'HEAD')


def boots(tmp_path):
    path = tmp_path / 'boots.txt'
    return path.read_text().split() if path.exists() else []


def wait_for(predicate, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(.1)
    return False


def test_idle_node_restarts_with_pushed_code(checkout):
    tmp_path, work, dev = checkout
    process = start(tmp_path, work)
    try:
        assert wait_for(lambda: boots(tmp_path) == ['v1'], 10)
        push_v2(dev)
        assert wait_for(lambda: boots(tmp_path) == ['v1', 'v2'], 30), boots(tmp_path)
        assert (work / 'scripts' / 'version.txt').read_text().strip() == 'v2'
        assert process.poll() is None, 'the node keeps serving after the restart'
    finally:
        os.killpg(process.pid, signal.SIGKILL); process.wait()


def test_node_with_a_running_task_is_not_restarted(checkout):
    tmp_path, work, dev = checkout
    process = start(tmp_path, work, FAKE_TASK='1')
    try:
        assert wait_for(lambda: boots(tmp_path) == ['v1'], 10)
        push_v2(dev)
        time.sleep(4)
        assert boots(tmp_path) == ['v1']
        assert (work / 'scripts' / 'version.txt').read_text().strip() == 'v1', 'no checkout change under a task'
    finally:
        os.killpg(process.pid, signal.SIGKILL); process.wait()


def test_local_edits_are_never_overwritten(checkout):
    tmp_path, work, dev = checkout
    (work / 'scripts' / 'version.txt').write_text('local\n')
    process = start(tmp_path, work)
    try:
        assert wait_for(lambda: boots(tmp_path) == ['local'], 10)
        push_v2(dev)
        time.sleep(3)
        assert boots(tmp_path) == ['local']
        assert (work / 'scripts' / 'version.txt').read_text().strip() == 'local'
    finally:
        os.killpg(process.pid, signal.SIGKILL); process.wait()
