"""Controller completion must not ignore live work beside sealed results."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('evidence', [None, 'RUNNING', 'heartbeat_fresh', 'owner_active', 'task_lease_held'])
def test_controller_does_not_release_root_with_active_work(tmp_path, evidence):
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    task = {'status': 'DONE'}
    if evidence == 'RUNNING':
        task['status'] = evidence
    elif evidence:
        task[evidence] = True
    snapshot = {'development_done': 18, 'test_done': 30, 'tasks': [task]}
    (scripts / 'selection_switch_status.py').write_text(f'print({json.dumps(snapshot)!r})\n')
    launcher = (ROOT / 'scripts/run_experiments.sh').read_text()
    function = 'root_complete() {' + launcher.split('root_complete() {', 1)[1].split('\n}', 1)[0] + '\n}'
    result = subprocess.run(['bash', '-c', function + '\nroot_complete "$1"', 'test', str(tmp_path)],
                            cwd=tmp_path, env={**os.environ, 'PY': sys.executable},
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == (0 if evidence is None else 1), result.stderr
