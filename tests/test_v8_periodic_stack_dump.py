"""CPU check: the task entry drops the historical 30-min faulthandler watchdog (SIGSEGV source)."""
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CHILD = '''import faulthandler,os,sys,time
from pathlib import Path
def install_diagnostics(root,task,rank):
    handle=open(Path(root)/'stack.txt','a',buffering=1)
    faulthandler.enable(file=handle,all_threads=True)
    faulthandler.dump_traceback_later(0.2,repeat=True,file=handle)
    return handle
def main():
    handle=install_diagnostics(os.environ['FIXTURE_ROOT'],'t',0)
    time.sleep(1.0)
'''


@pytest.mark.parametrize('periodic', [None, '1'])
def test_task_entry_cancels_periodic_dump(tmp_path, periodic):
    child = tmp_path / 'cluster_child.py'
    child.write_text(CHILD)
    env = dict(os.environ, FIXTURE_ROOT=str(tmp_path), SRGC_PROGRESS_DIR=str(tmp_path / 'progress'),
               V8_ORIGINAL_CHILD_SHA256=hashlib.sha256(child.read_bytes()).hexdigest(), RANK='0',
               CUDA_VISIBLE_DEVICES='',
               PYTHONPATH=os.pathsep.join([str(REPO / 'v8/experiments'), str(REPO / 'scripts')]))
    env.pop('V8_PERIODIC_STACK_DUMP', None)
    if periodic:
        env['V8_PERIODIC_STACK_DUMP'] = periodic
    result = subprocess.run([sys.executable, str(REPO / 'scripts/v8_task_entry.py'), str(child)],
                            env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-2000:]
    dumps = (tmp_path / 'stack.txt').read_text().count('Timeout (')
    assert (dumps > 0) if periodic else (dumps == 0)
