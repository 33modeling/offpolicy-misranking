"""Real peer-held startup locks must return to the chain before GPU admission."""
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import pytest

from srgc_rebuttal.runtime import lease

ROOT = Path(__file__).resolve().parents[2]

CHILD = r'''
import json, os, sys
from pathlib import Path
from srgc_research.dispatch import startup
from srgc_research.dispatch.gemma4 import runtime
from srgc_research.dispatch.gemma4_it import model
root, mode = Path(sys.argv[1]), sys.argv[2]
runtime.PACKAGES = {"srgc-test-package": "1.0"}
model.OFFICIAL_FILES = json.loads((root.parent / "official-fixture.json").read_text())
model.validate_tokenizer_metadata = lambda *args: None
if mode == "session":
    from unittest.mock import patch
    from srgc_research.dispatch import gemma_it_run
    def queued(args):
        print("REACHED_QUEUE", flush=True)
        return 0
    with patch.object(gemma_it_run, "run", queued):
        raise SystemExit(gemma_it_run.main(["all", "run", "--root", str(root)]))
def entry(args):
    if mode == "packages":
        runtime.package_overlay(os.environ)
    else:
        model.resolve_snapshot(os.environ)
    startup.worker_registered()
    print("REACHED_QUEUE", flush=True)
    return 0
raise SystemExit(startup.wrap_entry(entry, root)(["all"]))
'''


@pytest.fixture
def setup(tmp_path):
    work = tmp_path / 'work'
    root = work / 'srgc-rebuttal/gemma-it'
    root.parent.mkdir(parents=True)
    snapshot = tmp_path / 'model'
    snapshot.mkdir()
    files = {'config.json': json.dumps({'model_type': 'gemma4_unified'}).encode(),
             'model.safetensors': b'synthetic CPU-only model identity'}
    official = {}
    for name, raw in files.items():
        (snapshot / name).write_bytes(raw)
        digest = hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest() if name.endswith('.json') else hashlib.sha256(raw).hexdigest()
        official[name] = [len(raw), digest, 'blob' if name.endswith('.json') else 'sha256']
    (root.parent / 'official-fixture.json').write_text(json.dumps(official))
    protocol = {'packages': {'srgc-test-package': '1.0'}, 'python': list(sys.version_info[:2]), 'machine': platform.machine()}
    key = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:20]
    target = work / 'gemma-runtime-cache/packages' / key
    metadata = target / 'srgc_test_package-1.0.dist-info/METADATA'
    metadata.parent.mkdir(parents=True)
    metadata.write_text('Metadata-Version: 2.1\nName: srgc-test-package\nVersion: 1.0\n')
    (target / '.runtime.json').write_text(json.dumps(protocol))
    marker_key = hashlib.sha256(str(snapshot).encode()).hexdigest()
    locks = {'session': root.parent / ('.'+root.name+'.session.lock'),
             'packages': target.with_suffix('.lock'),
             'model': work / 'gemma-runtime-cache/verified-models' / (marker_key+'.lock')}
    env = {**os.environ, 'GROUP_VOLUME': str(tmp_path), 'OM_WORK': str(work),
           'SRGC_GEMMA_IT_MODEL_PATH': str(snapshot), 'SRGC_MODEL_CHAIN_CHILD': '1',
           'PYTHONPATH': str(ROOT)+os.pathsep+str(ROOT/'scripts'), 'CUDA_VISIBLE_DEVICES': ''}
    return root, env, locks


def invoke(setup, mode, timeout=5):
    root, env, _ = setup
    child = subprocess.Popen([sys.executable, '-c', CHILD, str(root), mode],
                             cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        out, err = child.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        child.kill()
        out, err = child.communicate()
        pytest.fail(f'{mode}: startup stayed blocked on the peer lock instead of yielding; {out} {err}')
    return child.returncode, out, err


@pytest.mark.parametrize('mode', ['packages', 'model', 'session'])
def test_peer_owned_startup_yields_then_retries_to_the_queue(setup, mode):
    root, _, locks = setup
    with lease(locks[mode]):
        inode = locks[mode].stat().st_ino
        code, out, err = invoke(setup, mode)
        assert code == 75, (out, err)
        assert 'REACHED_QUEUE' not in out
        assert not root.exists(), 'busy startup created experiment/queue state'
        assert locks[mode].stat().st_ino == inode
        # No lock deletion, replacement or stolen ownership.
        from srgc_rebuttal.runtime import Busy
        with pytest.raises(Busy), lease(locks[mode]):
            pass
    code, out, err = invoke(setup, mode)
    assert code == 0 and 'REACHED_QUEUE' in out, (out, err)
    assert locks[mode].stat().st_ino == inode


@pytest.mark.parametrize('mode', ['packages', 'model', 'session'])
def test_free_startup_still_runs_original_validation(setup, mode):
    code, out, err = invoke(setup, mode)
    assert code == 0 and 'REACHED_QUEUE' in out, (out, err)


def test_invalid_package_versions_are_not_accepted_to_avoid_waits(setup):
    root, _, locks = setup
    target = locks['packages'].with_suffix('')
    (target/'srgc_test_package-1.0.dist-info/METADATA').write_text('Name: srgc-test-package\nVersion: 0.0\n')
    code, out, err = invoke(setup, 'packages')
    assert code != 0 and 'packages changed' in err
    assert 'REACHED_QUEUE' not in out


def test_changed_model_bytes_are_still_rejected(setup):
    _, env, _ = setup
    path = Path(env['SRGC_GEMMA_IT_MODEL_PATH'])/'model.safetensors'
    path.write_bytes(b'x'*path.stat().st_size)
    code, out, err = invoke(setup, 'model')
    assert code != 0 and 'differs from the pinned' in err
    assert 'REACHED_QUEUE' not in out


def test_other_shared_workers_can_still_join_the_same_session(setup):
    import fcntl

    _, _, locks = setup
    with locks['session'].open('a+') as holder:
        fcntl.flock(holder, fcntl.LOCK_SH)
        code, out, err = invoke(setup, 'session')
        assert code == 0 and 'REACHED_QUEUE' in out, (out, err)


@pytest.mark.parametrize('mode', ['packages', 'model', 'session'])
def test_direct_nonchain_start_keeps_waiting_and_proceeds_after_release(setup, mode):
    root, env, locks = setup
    env = {**env, 'SRGC_MODEL_CHAIN_CHILD': '0'}
    child = None
    try:
        with lease(locks[mode]):
            child = subprocess.Popen([sys.executable, '-c', CHILD, str(root), mode],
                                     cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            time.sleep(.4)
            assert child.poll() is None
        out, err = child.communicate(timeout=5)
        assert child.returncode == 0 and 'REACHED_QUEUE' in out, (out, err)
    finally:
        if child is not None and child.poll() is None:
            child.kill()
            child.communicate()


def test_busy_inside_verification_is_not_misclassified_as_a_peer_lock(tmp_path, monkeypatch):
    from srgc_rebuttal.runtime import Busy
    from srgc_research.dispatch import startup
    from srgc_research.dispatch.startup_locks import retry_lease

    monkeypatch.setenv('SRGC_MODEL_CHAIN_CHILD', '1')
    path = tmp_path/'gemma-runtime-cache/verified-models/fixture.lock'
    with startup.registration(tmp_path/'experiment'), pytest.raises(Busy, match='inside verification'):
        with retry_lease(lease)(path, wait=True):
            raise Busy('inside verification')


@pytest.mark.parametrize('registered', [False, True])
def test_unrelated_locks_and_registered_workers_keep_original_lease_semantics(tmp_path, monkeypatch, registered):
    from contextlib import contextmanager
    from srgc_research.dispatch import startup
    from srgc_research.dispatch.startup_locks import retry_lease

    seen = []
    @contextmanager
    def original(path, *, wait=False):
        seen.append(wait)
        yield None

    monkeypatch.setenv('SRGC_MODEL_CHAIN_CHILD', '1')
    with startup.registration(tmp_path/'experiment'):
        if registered:
            startup.worker_registered()
        path = (tmp_path/'gemma-runtime-cache/verified-models/x.lock' if registered
                else tmp_path/'.dispatch/resume-first.lock')
        with retry_lease(original)(path, wait=True):
            pass
    assert seen == [True]
