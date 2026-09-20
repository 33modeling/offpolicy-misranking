"""Fifteen CPU workers use real branch leases; no cluster or GPU access."""

import os
import subprocess
import sys
import time
from pathlib import Path

import selection_gate as core
import selection_gate_gpu as base
import selection_switch as rule
import selection_switch_gpu as switch


ROOT = Path(__file__).resolve().parents[1]
WORKER = r'''
import importlib.util, os, sys, time
from pathlib import Path
from types import SimpleNamespace
import selection_switch_gpu as s
sys.modules['additive_experiment'] = SimpleNamespace(model_environment=lambda _: {})
spec = importlib.util.spec_from_file_location('node_queue', sys.argv[2])
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)
root = Path(sys.argv[1])
s.manifest = lambda _: s.core.read(root / 'switch.json')
s.admitted_devices = lambda _: list('0123')
s.status = lambda _: None
s.mbpp_resume_blocked = lambda *args: False
s.protocol = lambda child: s.core.read(child / 'net_protocol.json')
s.base.entries = lambda child: iter((child / 'points').iterdir())

def fit(root):
    if (root / 'model.json').exists():
        return True
    for seed in s.rule.DEV_SEEDS:
        for step in s.rule.STEPS:
            out = next(s.base.entries(s.child_root(root, seed, step)))
            for arm in s.rule.DEV_ARMS:
                if not (out / arm / 'curve.json').exists():
                    raise ValueError('premature expensive fit before development curves')
    with s.base.lease(root / '.fit.lock'):
        s.base.bind(root / 'model.json', {'fitted': True})
    return True
s.fit_once = fit
s.bind_gate = lambda root, child, protocol: {} if (root / 'model.json').exists() else None

def freeze_gate(out, *args):
    assert (root / 'model.json').exists()
    s.base.bind(out / 'gate-frozen.json', {'frozen': True})
s.freeze_gate = freeze_gate

def run(out, suite, protocol, arm, devices, env):
    directory = out / arm
    if (directory / 'result.json').exists():
        return
    if arm == 'gated':
        assert (root / 'model.json').exists() and (out / 'gate-frozen.json').exists()
    s.base.bind(directory / 'result.json', {'complete': True})
    s.base.bind(directory / 'result.sha256.json', {'sha256': s.base.digest(directory / 'result.json')})
s.runtime.run_arm = run

def curve(root, p, out, contract, arm, suite, devices, env):
    directory = out / arm
    claim = directory / 'claim.json'
    with claim.open('x') as handle:
        import json
        json.dump({'pid': os.getpid(), 'started': time.monotonic()}, handle)
    (root / f'running-{os.getpid()}').touch()
    deadline = time.monotonic() + 40
    while not (root / 'release').exists():
        if time.monotonic() > deadline:
            raise RuntimeError('fifteen workers did not overlap')
        time.sleep(.01)
    s.base.bind(directory / 'curve.json', {'result_sha256': s.base.digest(directory / 'result.json')})
    s.base.bind(directory / 'finished.json', {'time': time.monotonic()})
s.curve_once = curve
s.main = lambda: s.work(root, idle_timeout=0)
raise SystemExit(queue.run())
'''


def wait_for_count(root, count, workers, timeout=30):
    deadline = time.monotonic() + timeout
    while len(list(root.glob('running-*'))) < count and time.monotonic() < deadline:
        assert all(worker.poll() is None for worker in workers), 'worker exited before claiming work'
        time.sleep(.02)
    assert len(list(root.glob('running-*'))) == count


def test_four_active_nodes_accept_eleven_more_without_duplicate_or_restarted_work(tmp_path):
    seeds = (*rule.DEV_SEEDS, *rule.TEST_SEEDS)
    core.atomic_json(tmp_path / 'switch.json', {
        'dataset': 'mbpp', 'gate': 'convergence', 'budget_gpu_seconds': 28380.,
        'sources': {str(seed): {'config': {}} for seed in seeds},
    })
    saved = {}
    for seed in seeds:
        for step in rule.STEPS:
            core.atomic_json(switch.prefix_dir(tmp_path, seed) / f'prefix-{step}.json', {})
            child = switch.child_root(tmp_path, seed, step)
            out = child / 'points' / f'view-{step}'
            arms = list(rule.DEV_ARMS if seed in rule.DEV_SEEDS else rule.TEST_ARMS)
            protocol = {'mode': 'study' if seed in rule.DEV_SEEDS else 'test', 'arms': arms}
            core.atomic_json(child / 'net_protocol.json', protocol)
            core.atomic_json(child / 'suite.json', {})
            core.atomic_json(out / 'contract.json', {'seed': seed, 'step': step})
            for arm in arms:
                directory = out / arm
                core.atomic_json(directory / 'decision.json', {'budget_gpu_seconds': 28380.})
                if seed in rule.DEV_SEEDS:
                    core.atomic_json(directory / 'result.json', {'existing': True})
                    core.atomic_json(directory / 'result.sha256.json', {'sha256': base.digest(directory / 'result.json')})
                    for path in directory.glob('result*.json'):
                        saved[path] = (path.read_bytes(), path.stat().st_mtime_ns)
            core.atomic_json(out / 'decisions-frozen.json', {
                'protocol_sha256': core.fingerprint(protocol),
                'decisions': {arm: base.digest(out / arm / 'decision.json') for arm in arms if arm != 'gated'},
            })
    manifest = (tmp_path / 'switch.json').read_bytes()
    workers = []
    try:
        for count in (4, 15):
            while len(workers) < count:
                workers.append(subprocess.Popen(
                    [sys.executable, '-c', WORKER, str(tmp_path), str(ROOT / 'scripts/queue_selection_switch_gpu.py')],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                    env={**os.environ, 'PYTHONPATH': str(ROOT / 'src'), 'CUDA_VISIBLE_DEVICES': ''},
                ))
            wait_for_count(tmp_path, count, workers)
        (tmp_path / 'release').touch()
        for worker in workers:
            stdout, stderr = worker.communicate(timeout=40)
            assert worker.returncode == 0, stdout + stderr
        assert len(list(tmp_path.glob('states/*/points/*/*/curve.json'))) == 48
        claims = {path.parent: core.read(path) for path in tmp_path.glob('states/*/points/*/*/claim.json')}
        assert len(claims) == 48
        finishes = {directory: core.read(directory / 'finished.json')['time'] for directory in claims}
        assert max(sum(claim['started'] <= instant < finishes[directory] for directory, claim in claims.items())
                   for instant in (claim['started'] for claim in claims.values())) == 15
        assert not list(tmp_path.rglob('failure.json'))
        assert saved == {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in saved}
        assert (tmp_path / 'switch.json').read_bytes() == manifest
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
                worker.communicate()
