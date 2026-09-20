import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("switch_runtime", ROOT / "scripts/selection_switch_runtime.py")
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def repository(tmp_path):
    repo = tmp_path / "live"
    (repo / "src").mkdir(parents=True)
    (repo / "scripts").mkdir()
    (repo / "src/science.py").write_text("VALUE = 'original'\n")
    (repo / "src/probe.py").write_text('''
import hashlib, json, os, subprocess, sys, time
from pathlib import Path
import science
root = Path(os.environ['OUT_ROOT'])
source = Path(science.__file__)
frozen = hashlib.sha256(source.read_bytes()).hexdigest()
root.mkdir(parents=True, exist_ok=True)
(root / 'started.json').write_text(json.dumps({'file': str(source), 'hash': frozen,
    'value': science.VALUE, 'repo': os.environ['OM_REPO'], 'args': sys.argv[1:]}))
deadline = time.monotonic() + 15
while not (root / 'continue').exists():
    if time.monotonic() >= deadline: raise RuntimeError('fixture timeout')
    time.sleep(.01)
if hashlib.sha256(source.read_bytes()).hexdigest() != frozen:
    raise ValueError('switch protocol or scientific code changed; preserve the frozen run')
child = subprocess.check_output([sys.executable, '-c', 'import science; print(science.VALUE)'], text=True).strip()
(root / 'finished.json').write_text(json.dumps({'value': science.VALUE, 'child': child}))
''')
    launcher = '#!/usr/bin/env bash\nset -euo pipefail\ncd "$(dirname "$0")/.."\nexec "$SWITCH_PYTHON" src/probe.py "$@"\n'
    for name in runtime.LAUNCHERS.values():
        (repo / "scripts" / name).write_text(launcher)
    git(repo, "init", "--quiet")
    git(repo, "config", "user.email", "test@example.invalid")
    git(repo, "config", "user.name", "Runtime Test")
    git(repo, "add", ".")
    git(repo, "commit", "--quiet", "-m", "initial runtime")
    return repo


def commit_change(repo):
    (repo / "src/science.py").write_text("VALUE = 'changed'\n")
    git(repo, "add", "src/science.py")
    git(repo, "commit", "--quiet", "-m", "next runtime")


def wait_for(path, worker):
    deadline = time.monotonic() + 10
    while not path.exists() and worker.poll() is None and time.monotonic() < deadline:
        time.sleep(.01)
    assert path.exists(), worker.communicate(timeout=5) if worker.poll() is not None else "fixture did not start"


def test_shared_queue_entrypoint_is_pinned_and_yields_live_peer_waits(tmp_path):
    repo = repository(tmp_path)
    for name in ("run_selection_switch.sh", "selection_switch_runtime.py", "_selection_worker.sh",
                 "queue_selection_switch_gpu.py", "node_fault_state.py"):
        shutil.copy2(ROOT / "scripts" / name, repo / "scripts" / name)
    (repo / "scripts/mbpp_budget_recovery.py").write_text(
        'def required(*args):\n    raise AssertionError("peer-wait fixture must not recover budgets")\n')
    (repo / "scripts/setup_env.sh").write_text('export DATASETS_DIR="$OM_WORK/data"\n')
    (repo / "scripts/_e5_node.sh").write_text('e5_acquire_node() { return 0; }\n')
    (repo / "src/bootstrap_math_verify.py").write_text("print('/unused-test-dependencies')\n")
    (repo / "scripts/selection_nccl_preflight.py").write_text('''
import os, sys
command = sys.argv[sys.argv.index('--')+1:]
assert command[1] == 'scripts/queue_selection_switch_gpu.py'
os.execvpe(command[0], command, os.environ)
''')
    (repo / "src/selection_switch_gpu.py").write_text('''
import json, os, sys
from pathlib import Path
def wait_for_peers(*args, **kwargs):
    raise AssertionError('the node queue must own the wait')
def fit_once(root):
    return False
def mbpp_resume_blocked(*args):
    return False
def work(root, **kwargs):
    raise AssertionError('peer-wait fixture must not start GPU work')
def main():
    root = Path(os.environ['OUT_ROOT'])
    if sys.argv[1] == 'prepare':
        root.mkdir(parents=True, exist_ok=True)
        (root / 'switch.json').write_text('{}')
    elif sys.argv[1] == 'run':
        assert not wait_for_peers([{'active': True}], last_progress=0, idle_timeout=600)
        (root / 'queue.json').write_text(json.dumps({'repo': os.environ['OM_REPO'], 'module': __file__}))
    return 0
if __name__ == '__main__':
    sys.exit(main())
''')
    (repo / "bin").mkdir()
    gpu = repo / "bin/nvidia-smi"
    gpu.write_text('#!/usr/bin/env bash\nprintf "0\\n0\\n0\\n0\\n"\n')
    gpu.chmod(0o755)
    git(repo, "add", ".")
    git(repo, "commit", "--quiet", "-m", "shared queue fixture")
    root, cache = tmp_path / "run", tmp_path / "cache"
    env = {**os.environ, "OM_WORK": str(tmp_path / "work"), "SWITCH_ROOT": str(root),
           "SWITCH_RUNTIME_CACHE": str(cache), "SWITCH_PYTHON": sys.executable,
           "PATH": str(repo / "bin") + os.pathsep + os.environ["PATH"], "PYTHONPATH": str(repo / "src"),
           "CUDA_VISIBLE_DEVICES": "0,1,2,3", "SWITCH_FOREGROUND": "1", "SWITCH_HOLD_SECONDS": "0",
           "SWITCH_QUEUE_PASS": "1", "SWITCH_AUTO_RECOVER": "0", "SWITCH_KEEPALIVE": "0"}
    env.pop("SWITCH_RUNTIME_REPO", None)
    result = subprocess.run(["bash", str(repo / "scripts/run_selection_switch.sh")], cwd=repo,
                            env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads((root / "queue.json").read_text())
    assert Path(receipt["repo"]).parent == cache
    assert Path(receipt["module"]).parent == Path(receipt["repo"]) / "src"
    assert "[queue-yield]" in result.stdout


@pytest.mark.parametrize("pinned", [False, True])
@pytest.mark.parametrize("kind", ["switch", "mopps"])
def test_midrun_checkout_change_and_next_worker_import(tmp_path, pinned, kind):
    repo = repository(tmp_path)
    root = tmp_path / "run"
    cache = tmp_path / "cache"
    env = {**os.environ, "OUT_ROOT": str(root), "OM_REPO": str(repo), "PYTHONPATH": str(repo / "src"),
           "SWITCH_PYTHON": sys.executable, "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""}
    if pinned:
        command = [sys.executable, str(ROOT / "scripts/selection_switch_runtime.py"),
                   "--repo", str(repo), "--cache", str(cache), "--kind", kind, "--", "run"]
    else:
        command = ["bash", str(repo / "scripts" / runtime.LAUNCHERS[kind]), "run"]
    worker = subprocess.Popen(command, cwd=repo, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        wait_for(root / "started.json", worker)
        start = json.loads((root / "started.json").read_text())
        commit_change(repo)
        (root / "continue").touch()
        stdout, stderr = worker.communicate(timeout=15)
        if pinned:
            assert worker.returncode == 0, stdout + stderr
            assert json.loads((root / "finished.json").read_text()) == {"value": "original", "child": "original"}
            assert Path(start["file"]).is_relative_to(cache)
            assert start["repo"] != str(repo)
            assert hashlib.sha256(Path(start["file"]).read_bytes()).hexdigest() == start["hash"]
        else:
            assert worker.returncode != 0
            assert "scientific code changed; preserve the frozen run" in stderr
            assert not (root / "finished.json").exists()
    finally:
        if worker.poll() is None:
            worker.kill()
        worker.communicate(timeout=5)


def test_cached_clone_is_reused_without_writes_or_remote(tmp_path):
    repo = repository(tmp_path)
    target, revision = runtime.snapshot(repo, tmp_path / "cache")
    before = (target / "src/science.py").read_bytes()
    assert runtime.snapshot(repo, tmp_path / "cache") == (target, revision)
    assert (target / "src/science.py").read_bytes() == before
    assert git(target, "remote") == ""
    assert (target / ".git").is_dir()


@pytest.mark.parametrize("untracked", [False, True])
def test_dirty_source_is_not_discarded_or_launched(tmp_path, untracked):
    repo = repository(tmp_path)
    path = repo / "src" / ("unreviewed.py" if untracked else "science.py")
    path.write_text("unreviewed change\n")
    with pytest.raises(ValueError, match="nothing was discarded"):
        runtime.snapshot(repo, tmp_path / "cache")
    assert path.read_text() == "unreviewed change\n"
    assert not (tmp_path / "cache").exists()


def test_modified_cached_clone_is_not_replaced_under_existing_workers(tmp_path):
    repo = repository(tmp_path)
    target, _ = runtime.snapshot(repo, tmp_path / "cache")
    path = target / "src/science.py"
    path.write_text("unreviewed cache change\n")
    with pytest.raises(ValueError, match="dirty"):
        runtime.snapshot(repo, tmp_path / "cache")
    assert path.read_text() == "unreviewed cache change\n"


def test_relative_inputs_and_parent_root_survive_reentry(tmp_path, monkeypatch):
    repo = repository(tmp_path)
    monkeypatch.setenv("OM_WORK", "work")
    monkeypatch.setenv("SWITCH_ROOT", "work/parent")
    monkeypatch.setenv("OUT_ROOT", str(tmp_path / "run"))
    monkeypatch.setenv("OM_REPO", str(tmp_path / "old-checkout"))
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(repo / "src"), str(tmp_path / "old-checkout/src"), "/external/deps"]))
    calls = []
    monkeypatch.setattr(runtime.os, "execvpe", lambda *args: calls.append(args))
    runtime.launch(repo, tmp_path / "cache", "mopps", ["prepare", "--pool", "data/pool.jsonl", "--pool-manifest=data/manifest.json"])
    _, command, env = calls[0]
    assert command[-3:] == ["--pool", str(repo / "data/pool.jsonl"), f"--pool-manifest={repo}/data/manifest.json"]
    assert env["SWITCH_ROOT"] == str(repo / "work/parent")
    assert env["MOPPS_ROOT"] == str(tmp_path / "run")
    assert env["OM_WORK"] == str(repo / "work")
    assert env["SWITCH_PYTHON"] == sys.executable
    assert env["PYTHONPATH"].split(os.pathsep) == [str(Path(env["OM_REPO"]) / "src"), "/external/deps"]


def test_snapshot_cache_cannot_be_inside_live_repository(tmp_path):
    repo = repository(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        runtime.snapshot(repo, repo / "cache")


@pytest.mark.parametrize("kind,mode", [("switch", "run"), ("mopps", "run"), ("mopps", "retry")])
@pytest.mark.parametrize("preflight_code", [0, 78, 130])
@pytest.mark.parametrize("hold_seconds", [0, 1])
def test_real_shell_entrypoint_pins_before_setup_and_keeps_runtime_and_storage(tmp_path, kind, mode, preflight_code, hold_seconds):
    repo = repository(tmp_path)
    for name in (*runtime.LAUNCHERS.values(), "selection_switch_runtime.py", "selection_switch_errors.py", "_selection_worker.sh", "node_fault_state.py"):
        shutil.copy2(ROOT / "scripts" / name, repo / "scripts" / name)
    (repo / "scripts/setup_env.sh").write_text('export DATASETS_DIR="$OM_WORK/data"\nexport PYTHONPATH="$OM_REPO/src:$PYTHONPATH"\n')
    (repo / "scripts/_e5_node.sh").write_text('e5_acquire_node() { return 0; }\n')
    (repo / "src/bootstrap_math_verify.py").write_text("print('/unused-test-dependencies')\n")
    (repo / "scripts/selection_switch_status.py").write_text('''
import json, os
from pathlib import Path
count = len((Path(os.environ['OUT_ROOT']) / 'probe-calls').read_text().splitlines())
print(json.dumps({'development_done': 18 if count >= 2 else 0, 'test_done': 30 if count >= 2 else 0}))
''')
    (repo / "scripts/selection_nccl_preflight.py").write_text('''
import os, sys
from pathlib import Path
root = Path(os.environ['OUT_ROOT'])
with (root / 'probe-calls').open('a') as handle:
    handle.write('probe\\n')
code = int(os.environ['TEST_PREFLIGHT_CODE'])
if code:
    print('[blocked] NCCL admission failed before claiming work', flush=True)
    raise SystemExit(code)
command = sys.argv[sys.argv.index('--')+1:]
os.execvpe(command[0], command, os.environ)
''')
    controller = '''
import json, os, sys
from pathlib import Path
if sys.argv[1] == 'prepare':
    root = Path(os.environ['OUT_ROOT'])
    root.mkdir(parents=True, exist_ok=True)
    (root / 'switch.json').write_text('{}')
    (root / 'mopps.json').write_text('{}')
elif sys.argv[1] == 'check-code':
    print('exact')
elif sys.argv[1] == 'status':
    count = len((Path(os.environ['OUT_ROOT']) / 'probe-calls').read_text().splitlines())
    if count >= 2:
        print('s3 t25 mopps DONE published\\n' * 12)
else:
    import probe
'''
    (repo / "src/selection_switch_gpu.py").write_text(controller)
    (repo / "src/mopps_comparison_gpu.py").write_text(controller)
    (repo / "bin").mkdir()
    gpu = repo / "bin/nvidia-smi"
    gpu.write_text('#!/usr/bin/env bash\nprintf "0\\n0\\n0\\n0\\n"\n')
    gpu.chmod(0o755)
    git(repo, "add", ".")
    git(repo, "commit", "--quiet", "-m", "real launcher fixture")
    root = tmp_path / "run"
    if mode == "retry":
        for seed in (3, 4):
            for step in (25, 50):
                for arm in ("mopps", "random_online"):
                    failure = root / f"states/s{seed}-t{step}/{arm}/failure.json"
                    failure.parent.mkdir(parents=True)
                    failure.write_text('{"error": "historical worker failure"}')
    env = {**os.environ, "OM_WORK": str(tmp_path / "work"), "OM_REPO": str(tmp_path / "stale-repo"),
           "SWITCH_RUNTIME_CACHE": str(tmp_path / "cache"), "SWITCH_PYTHON": sys.executable,
           "MOPPS_PYTHON": sys.executable, "PATH": str(repo / "bin") + os.pathsep + os.environ["PATH"],
           "PYTHONPATH": str(repo / "src"), "CUDA_VISIBLE_DEVICES": "0,1,2,3",
           "TEST_PREFLIGHT_CODE": str(preflight_code), "SWITCH_HOLD_SECONDS": str(hold_seconds)}
    env["SWITCH_ROOT" if kind == "switch" else "MOPPS_ROOT"] = str(root)
    env.pop("SWITCH_RUNTIME_REPO", None)
    worker = subprocess.Popen(["bash", str(repo / "scripts" / runtime.LAUNCHERS[kind]), mode],
                              cwd=repo, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        if preflight_code:
            stdout, stderr = worker.communicate(timeout=15)
            assert worker.returncode == preflight_code, stdout + stderr
            assert "NCCL admission failed before claiming work" in stdout
            assert "[waiting]" not in stdout
            assert not (root / "started.json").exists()
            assert (root / "probe-calls").read_text().splitlines() == ["probe"]
            if preflight_code == 78:
                assert "historical branch errors are not the cause" in stdout
                assert "[failure]" not in stdout
            if mode == "retry":
                assert "[retry stopped]" in stdout
                assert len(list(root.glob("states/*/*/failure.json"))) == 8
            return
        wait_for(root / "started.json", worker)
        start = json.loads((root / "started.json").read_text())
        commit_change(repo)
        (root / "continue").touch()
        stdout, stderr = worker.communicate(timeout=15)
        assert worker.returncode == 0, stdout + stderr
        assert "[launcher-start]" in stdout and "[launcher-exit]" in stdout and "rc=0" in stdout
        assert Path(start["file"]).is_relative_to(tmp_path / "cache")
        assert start["repo"] == str(Path(start["file"]).parent.parent)
        assert json.loads((root / "finished.json").read_text()) == {"value": "original", "child": "original"}
        assert len(list(root.glob("logs/launcher.*.log"))) == 1
        assert len((root / "probe-calls").read_text().splitlines()) == (8 if mode == "retry" else 2 if hold_seconds else 1)
        if mode == "run" and hold_seconds:
            assert "[holding]" in stdout and "releasing the node" in stdout
        if mode == "retry":
            args = start["args"]
            assert args[args.index("--idle-timeout")+1] == "0"
    finally:
        if worker.poll() is None:
            worker.kill()
        worker.communicate(timeout=5)


def test_four_processes_reuse_snapshot_while_source_checkout_changes(tmp_path):
    repo = repository(tmp_path)
    workers = []
    roots = [tmp_path / f"run-{i}" for i in range(4)]
    try:
        for root in roots:
            env = {**os.environ, "OUT_ROOT": str(root), "OM_REPO": str(repo),
                   "PYTHONPATH": str(repo / "src"), "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""}
            workers.append(subprocess.Popen([sys.executable, str(ROOT / "scripts/selection_switch_runtime.py"),
                           "--repo", str(repo), "--cache", str(tmp_path / "cache"), "--", "run"],
                           cwd=repo, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
        for root, worker in zip(roots, workers):
            wait_for(root / "started.json", worker)
        starts = [json.loads((root / "started.json").read_text()) for root in roots]
        assert len({row["repo"] for row in starts}) == 1
        commit_change(repo)
        for root in roots:
            (root / "continue").touch()
        for root, worker in zip(roots, workers):
            stdout, stderr = worker.communicate(timeout=15)
            assert worker.returncode == 0, stdout + stderr
            assert json.loads((root / "finished.json").read_text())["child"] == "original"
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
            worker.communicate(timeout=5)
