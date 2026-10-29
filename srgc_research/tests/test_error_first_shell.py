"""Exercise the public Llama shell and real child processes without GPUs."""

import json
import os
import subprocess
import sys

from srgc_research.dispatch import model_chain


def test_llama_shell_finishes_llama_math_and_mbpp_before_starting_failed_gemma(tmp_path):
    driver = tmp_path / 'python'
    driver.write_text('#!' + sys.executable + '\n' + '''
import json, os, sys
from pathlib import Path
sys.path.insert(0, os.getcwd())
root = Path(os.environ['RECOVERY_TEST_ROOT'])
if sys.argv[1:3] == ['-m', 'srgc_research.dispatch.model_chain']:
    from srgc_research.dispatch import model_chain as chain, resume_inventory
    def signature(model, dataset, environment, **kwargs):
        path = root / f'{model}.{dataset}.json'
        if path.is_file() and json.loads(path.read_text())['status'] == 'failed':
            return (path.stat().st_mtime_ns, path.stat().st_size)
    resume_inventory.signature = signature
    resume_inventory.failed_signature = signature
    original = chain.drain
    # main() passes its heartbeat-keeping sleep; replace it, never pass it twice.
    chain.drain = lambda *args, **kwargs: original(*args, **{**kwargs, 'sleep': lambda _: None})
    raise SystemExit(chain.main(sys.argv[3:]))
modules = {'srgc_research.dispatch.llama_run': 'llama31',
           'srgc_research.dispatch.gemma_run': 'gemma4',
           'srgc_research.dispatch.gemma4_text.entry': 'gemma4',
           'srgc_research.dispatch.gemma_text_run': 'gemma4',
           'srgc_research.dispatch.gemma_it_run': 'gemma4',
           'srgc_research.dispatch.qwen_run': 'qwen35'}
assert sys.argv[1] == '-m', sys.argv
model = modules[sys.argv[2]]
assert os.environ['SRGC_MODEL_CHAIN_CHILD'] == '1'
phase, recovery = os.environ['SRGC_DATASET_PHASE'], os.environ['SRGC_RESUME_ONLY']
path = root / f'{model}.{phase}.json'
row = json.loads(path.read_text()) if path.is_file() else {'attempt': 0}
if recovery == '0':
    assert not path.is_file() or row['status'] != 'failed', 'fresh work overtook this model ERROR recovery'
if model == 'gemma4':
    for dataset in ('math', 'mbpp'):
        assert json.loads((root / f'llama31.{dataset}.json').read_text())['status'] == 'complete'
        assert (root / f'llama31.full.{dataset}').exists(), 'Gemma ran before full Llama MATH and MBPP'
attempt = row['attempt'] + 1
checkpoint = root / f'{model}.{phase}.pt'
if not checkpoint.exists():
    checkpoint.write_bytes(b'saved optimizer, selector and RNG state')
limit = {'llama31': 7, 'gemma4': 102}.get(model, 0) if phase == 'math' else 0
code = 2 if attempt < limit else 0
path.write_text(json.dumps({'attempt': attempt, 'status': 'failed' if code else 'complete'}))
if code == 0 and recovery == '0':
    (root / f'{model}.full.{phase}').write_text('complete')
with (root / 'launches.jsonl').open('a') as handle:
    handle.write(json.dumps([model, phase, recovery, code]) + '\\n')
raise SystemExit(code)
''')
    driver.chmod(0o755)
    for model, attempt in (('llama31', 4), ('gemma4', 99)):
        (tmp_path / f'{model}.math.json').write_text(json.dumps({'status': 'failed', 'attempt': attempt}))
        (tmp_path / f'{model}.math.pt').write_bytes(b'saved optimizer, selector and RNG state')
    environment = {**os.environ, 'RECOVERY_TEST_ROOT': str(tmp_path),
                   'GROUP_VOLUME': str(tmp_path), 'OM_WORK': str(tmp_path / 'work'),
                   'SRGC_STORAGE_ROOT': str(tmp_path / 'work/srgc-rebuttal'),
                   'SRGC_MODEL_CHAIN': '1', 'SRGC_MODEL_ORDER': ','.join(model_chain.MODELS),
                   **{name + '_PYTHON': str(driver) for name in ('LLAMA', 'GEMMA', 'QWEN')}}
    environment.pop('SRGC_MODEL_CHAIN_CHILD', None)
    result = subprocess.run(model_chain.child_command('llama31', 'all'), cwd=model_chain.REPO,
                            env=environment, capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    launches = [json.loads(line) for line in (tmp_path / 'launches.jsonl').read_text().splitlines()]
    assert launches[:4] == [['llama31', 'math', '1', code] for code in (2, 2, 0)] + [['llama31', 'math', '0', 0]]
    assert launches[4] == ['llama31', 'mbpp', '0', 0]
    assert launches[5:9] == [['gemma4', 'math', '1', code] for code in (2, 2, 0)] + [['gemma4', 'math', '0', 0]]
    assert launches[9] == ['gemma4', 'mbpp', '0', 0]
    assert launches[10:] == [['qwen35', phase, '0', 0] for phase in ('math', 'mbpp')]
    for model in ('llama31', 'gemma4'):
        assert (tmp_path / f'{model}.math.pt').read_bytes() == b'saved optimizer, selector and RNG state'
    records = list((tmp_path / 'work/srgc-rebuttal/.model-chain/workers').glob('*.json'))
    assert len(records) == 1
    record = json.loads(records[0].read_text())
    assert record['status'] == 'finished' and record['failures'] == {}
