"""Operational log tags do not change worker exit status or ownership."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / 'scripts/_selection_worker.sh'


@pytest.mark.parametrize('name,environment,tag', [
    ('selector_pair_gpu.py', {'SWITCH_DATASET': 'mbpp'}, 'pair'),
    ('queue_rloo.py', {'EXPERIMENTS_MBPP_SUITE': 'all'}, 'rloo'),
    ('selection_switch_gpu.py', {'SWITCH_DATASET': 'mbpp'}, 'mbpp'),
    ('queue_selection_switch_gpu.py', {'EXPERIMENTS_MBPP_SUITE': 'all'}, 'mbpp'),
    ('unrelated.py', {}, None),
])
@pytest.mark.parametrize('code', [0, 1, 75, 80])
def test_phase_and_error_lines_keep_prefix_and_gain_correct_suffix(tmp_path, name, environment, tag, code):
    worker = tmp_path / name
    worker.write_text('import sys\nprint("[gate] curve curve: 15s / 120s", flush=True)\n'
                      'print("[failed] worker detail", file=sys.stderr, flush=True)\n'
                      f'sys.exit({code})\n')
    env = {key: value for key, value in os.environ.items()
           if key not in {'EXPERIMENTS_MBPP_SUITE', 'SWITCH_DATASET'}}
    result = subprocess.run(['bash', '-c', 'set -euo pipefail; source "$1"; shift; selection_run_worker "$@"',
                             'log-test', str(HELPER), sys.executable, str(worker)],
                            env={**env, **environment}, capture_output=True, text=True, timeout=5)
    assert result.returncode == code
    suffix = f' [{tag}]' if tag else ''
    assert '[gate] curve curve: 15s / 120s' + suffix in result.stdout.splitlines()
    assert '[failed] worker detail' + suffix in (result.stdout + result.stderr).splitlines()


def test_existing_tag_is_not_duplicated_and_final_unterminated_line_is_kept(tmp_path):
    worker = tmp_path / 'selector_pair_gpu.py'
    worker.write_text('import sys\nprint("[gate] curve curve [pair]")\nsys.stdout.write("last line")\n')
    result = subprocess.run(['bash', '-c', 'source "$1"; shift; selection_run_worker "$@"',
                             'log-test', str(HELPER), sys.executable, str(worker)],
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 0
    assert result.stdout.splitlines() == ['[gate] curve curve [pair]', 'last line [pair]']


@pytest.mark.parametrize('name', ['selector_pair_gpu.py', 'queue_rloo.py'])
def test_direct_workers_distinguish_same_hostname_and_preserve_explicit_identity(tmp_path, name):
    worker = tmp_path / name
    worker.write_text('import os\nprint(os.environ["EXPERIMENTS_NODE_ID"])\n')
    script = '''set -euo pipefail
hostname() { printf 'same-node\\n'; }
timeout() { printf '%s\\n' "$TEST_GPU_UUID"; }
source "$1"
shift
selection_run_worker "$@"
'''
    env = {key: value for key, value in os.environ.items() if key != 'EXPERIMENTS_NODE_ID'}
    def run(uuid, explicit=None):
        setting = {**env, 'TEST_GPU_UUID': uuid}
        if explicit is not None:
            setting['EXPERIMENTS_NODE_ID'] = explicit
        result = subprocess.run(['bash', '-c', script, 'identity-test', str(HELPER), sys.executable, str(worker)],
                                env=setting, capture_output=True, text=True, timeout=5)
        assert result.returncode == 0, result.stderr
        return result.stdout.split()[0]
    first, second = run('GPU-one'), run('GPU-two')
    assert first.startswith('same-node-g') and second.startswith('same-node-g')
    assert first != second
    assert run('GPU-one') == first
    assert run('GPU-two', 'operator-node-g1234') == 'operator-node-g1234'
