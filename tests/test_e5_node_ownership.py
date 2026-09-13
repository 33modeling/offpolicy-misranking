"""CPU process/lock checks for E5 controller ownership and multi-node arms."""

import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]


def wait_file(path):
    deadline = time.monotonic() + 5
    while not path.exists():
        assert time.monotonic() < deadline, path
        time.sleep(0.02)


def stop(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def test_cleanup_stops_the_old_seed_loop_before_reacquiring_its_lock(tmp_path):
    scripts = tmp_path / "legacy/scripts"
    scripts.mkdir(parents=True)
    legacy = scripts / "run_e5.sh"
    legacy.write_text('''export OUT_ROOT="$TEST_OUT"
exec 8>"$OM_LOCAL_LOCK_DIR/primary.lock"
flock 8
sleep 120 &
touch "$TEST_READY"
wait
echo respawned > "$TEST_READY.respawn"
''')
    locks = tmp_path / "locks"
    locks.mkdir()
    env = {**os.environ, "TEST_OUT": str(tmp_path / "e5/math500-d400"),
           "TEST_READY": str(tmp_path / "ready"), "OM_LOCAL_LOCK_DIR": str(locks),
           "PY": sys.executable}
    env.pop("OUT_ROOT", None)
    old = subprocess.Popen(["bash", str(legacy)], env=env, start_new_session=True)
    unrelated = subprocess.Popen(["sleep", "120"], env={**env, "OUT_ROOT": str(tmp_path / "qwen")},
                                 start_new_session=True)
    try:
        wait_file(tmp_path / "ready")
        result = subprocess.run(["bash", "-c", 'source scripts/_e5_node.sh; e5_cleanup_previous "$TEST_OUT" && e5_acquire_node'],
                                cwd=ROOT, env=env, capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, result.stdout + result.stderr
        assert f"pid={old.pid}" in result.stdout
        assert "node ownership acquired" in result.stdout
        assert old.wait(timeout=5) != 0
        assert not (tmp_path / "ready.respawn").exists()
        assert unrelated.poll() is None
    finally:
        stop(old)
        stop(unrelated)


@pytest.mark.parametrize("unsupported", [False, True])
def test_node_lock_never_succeeds_without_an_acquired_lock(tmp_path, unsupported):
    script = '''source scripts/_e5_node.sh
flock() { if [ "$UNSUPPORTED" = 1 ]; then echo 'Operation not supported' >&2; fi; return 1; }
stat() { echo overlayfs; }
PY=true
e5_acquire_node
'''
    result = subprocess.run(["bash", "-c", script], cwd=ROOT,
                            env={**os.environ, "OM_LOCAL_LOCK_DIR": str(tmp_path), "UNSUPPORTED": str(int(unsupported))},
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 75, result.stdout + result.stderr
    assert "node ownership acquired" not in result.stdout
    assert not list(tmp_path.glob(".e5-flock-*"))


@pytest.mark.parametrize("shared_lock_directory", [False, True])
def test_three_node_controllers_share_arms_but_not_node_locks(tmp_path, shared_lock_directory):
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    for name in ("run_e5.sh", "run_downstream_independent.sh", "_e5_node.sh"):
        shutil.copy2(ROOT / "scripts" / name, repo / "scripts" / name)
    (repo / "scripts/setup_env.sh").write_text('export OM_WORK="$TEST_WORK"\nexport DATASETS_DIR="$TEST_WORK/data"\nexport VENV_DIR="$TEST_VENV"\n')
    bins = tmp_path / "venv/bin"
    bins.mkdir(parents=True)
    adapter = tmp_path / "adapter.py"
    adapter.write_text('''import json, os, pathlib, subprocess, sys, time
args = sys.argv[1:]
if args[0] == '-c':
    raise SystemExit(subprocess.call([sys.executable, *args]))
if args[0] == 'src/cleanup_run_processes.py':
    # Different compute nodes have distinct /proc trees. Scope this one-host
    # simulation by SIM_NODE while exercising the real process matcher.
    raise SystemExit(subprocess.call([sys.executable, os.environ['CLEANUP'], *args[1:],
                                     '--require-environment', 'SIM_NODE=' + os.environ['SIM_NODE']]))
if args[0] == 'src/bootstrap_math_verify.py':
    print('/fixture'); raise SystemExit(0)
if args[0] == '-m':
    arm, out = args[2], pathlib.Path(args[3])
    with (out / ('claim-' + arm)).open('a') as f: f.write(os.environ['SIM_NODE'] + '\\n')
    time.sleep(0.25)
    (out / (arm + '.trained')).touch()
    raise SystemExit(0)
assert args[0] == 'src/evidence_downstream.py', args
stage = args[1]
if stage == 'prepare-test': raise SystemExit(0)
out = pathlib.Path(args[args.index('--out') + 1])
if stage == 'prepare':
    assert os.environ['OM_E5_CONTROLLER_PID'] == str(os.getppid()) or os.environ['OM_NODE_LOCK_HELD'] == '1'
    # Descendant processes must not carry the parent's node-lock descriptor.
    assert not pathlib.Path('/proc/self/fd/7').exists()
    assert not pathlib.Path('/proc/self/fd/8').exists()
    import fcntl
    lock = os.readlink('/proc/' + os.environ['OM_E5_CONTROLLER_PID'] + '/fd/8')
    with open(lock, 'a') as f:
        try: fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: pass
        else: raise AssertionError('controller did not hold its node lock')
    (out / 'subsets').mkdir(parents=True, exist_ok=True)
    for arm in ('random', 'passrate_beta', 'fresh_r', 'g11'):
        (out / 'subsets' / ('train-' + arm + '.args')).write_bytes(('\\0'.join(['-m', 'fixture', arm, str(out)]) + '\\0').encode())
elif stage == 'policy-ready':
    arm = args[args.index('--arm') + 1]
    raise SystemExit(0 if (out / (arm + '.trained')).exists() else 1)
elif stage == 'evaluate': time.sleep(0.05)
elif stage == 'summarize': print('{"complete": true}')
else: raise AssertionError(args)
''')
    python = bins / "python"
    python.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{adapter}" "$@"\n')
    python.chmod(0o755)
    smi = bins / "nvidia-smi"
    smi.write_text('#!/bin/sh\necho 0\n')
    smi.chmod(0o755)
    hostname = bins / "hostname"
    hostname.write_text('#!/bin/sh\nprintf "node-%s\\n" "$SIM_NODE"\n')
    hostname.chmod(0o755)
    stat = bins / "stat"
    stat.write_text('#!/bin/sh\nif [ "$SIM_SHARED" = 1 ]; then echo nfs; else exec /usr/bin/stat "$@"; fi\n')
    stat.chmod(0o755)
    work = tmp_path / "shared"
    pool = work / "data/math_train"
    pool.mkdir(parents=True)
    (pool / "math_train.jsonl").write_text("{}\n")
    (pool / "dataset_manifest.json").write_text('{"source_revision": "fixture"}')
    tag = "olmo3-1025-7b-base-rlzero-grpo-h100-v2"
    for seed in range(3):
        point = work / "runs" / tag / f"family-math500-s{seed}" / f"{tag}-s{seed}-math500-d400"
        point.mkdir(parents=True)
        (point / "DONE").write_text("done")
        (point / "run_config.json").write_text("{}")
    workers = []
    try:
        for node in range(3):
            env = {**os.environ, "TEST_WORK": str(work), "TEST_VENV": str(bins.parent),
                   "SIM_NODE": str(node), "CLEANUP": str(ROOT / "src/cleanup_run_processes.py"),
                   "OM_LOCAL_LOCK_DIR": str(tmp_path / ("node-locks-shared" if shared_lock_directory else f"node-{node}")),
                   "SIM_SHARED": str(int(shared_lock_directory)), "OM_NODE_LOCK_HELD": "0",
                   "CUDA_VISIBLE_DEVICES": "0,1,2,3", "PATH": str(bins) + os.pathsep + os.environ['PATH']}
            workers.append(subprocess.Popen(["bash", "scripts/run_e5.sh"], cwd=repo, env=env,
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                            start_new_session=True))
        outputs = []
        for worker in workers:
            out, err = worker.communicate(timeout=30)
            assert worker.returncode == 0, out + err
            assert out.count("node ownership acquired") == 1, out
            assert out.count("== seed") == 3, out
            outputs.append(out)
        base = work / "runs/e5-reduced/math500-d400"
        for seed in range(3):
            for arm in ("random", "passrate_beta", "fresh_r", "g11"):
                assert len((base / f"s{seed}/claim-{arm}").read_text().splitlines()) == 1
        assert any("claimed on another node" in out for out in outputs)
    finally:
        for worker in workers:
            stop(worker)


def test_force_preserves_the_real_qwen_launcher_holding_the_node(tmp_path):
    script = tmp_path / "scripts/run_additional_experiments.sh"
    script.parent.mkdir()
    script.write_text('exec 8>"$OM_LOCAL_LOCK_DIR/primary.lock"\nflock 8\ntouch "$TEST_READY"\nsleep 120 &\nwait\n')
    env = {**os.environ, "OM_LOCAL_LOCK_DIR": str(tmp_path), "TEST_READY": str(tmp_path / "ready"),
           "PY": sys.executable, "E5_FORCE": "1"}
    process = subprocess.Popen(["bash", str(script), "--run", "qwen35"], env=env, start_new_session=True)
    try:
        wait_file(tmp_path / "ready")
        result = subprocess.run(["bash", "-c", 'source scripts/_e5_node.sh; e5_acquire_node'],
                                cwd=ROOT, env=env, capture_output=True, text=True, timeout=5)
        assert result.returncode == 75, result.stdout + result.stderr
        assert str(process.pid) in result.stdout
        assert process.poll() is None
        assert "[force] stopping" not in result.stdout
    finally:
        stop(process)


def test_shared_lock_release_does_not_bypass_an_existing_host_lock(tmp_path):
    env = {**os.environ, "OM_LOCAL_LOCK_DIR": str(tmp_path), "PY": "true"}
    remote = subprocess.Popen(
        ["bash", "-c", 'exec 8>"$OM_LOCAL_LOCK_DIR/primary.lock"; flock 8; touch "$OM_LOCAL_LOCK_DIR/remote"; sleep 120 & wait'],
        env=env, start_new_session=True,
    )
    controller = None
    helper = 'source scripts/_e5_node.sh; stat() { echo nfs; }; hostname() { echo node-test; }; e5_acquire_node'
    try:
        wait_file(tmp_path / "remote")
        controller = subprocess.Popen(
            ["bash", "-c", helper + ' || exit $?; touch "$OM_LOCAL_LOCK_DIR/local"; sleep 120 & wait'],
            cwd=ROOT, env=env, start_new_session=True, stdout=subprocess.DEVNULL,
        )
        wait_file(tmp_path / "local")
        stop(remote)
        result = subprocess.run(["bash", "-c", helper], cwd=ROOT, env=env,
                                capture_output=True, text=True, timeout=5)
        assert result.returncode == 75, result.stdout + result.stderr
        assert "per-host node lock is held" in result.stdout
        assert "node ownership acquired" not in result.stdout
        assert controller.poll() is None
    finally:
        stop(remote)
        if controller is not None:
            stop(controller)


def test_occupied_gpu_is_not_admitted_after_cleanup():
    source = (ROOT / "scripts/run_downstream_independent.sh").read_text()
    start = source.index("# Wait for the GPUs to drain")
    end = source.index('mkdir -p "$OUT/logs"', start)
    harness = '''CUDA_VISIBLE_DEVICES=0,1,2,3
nvidia-smi() { echo 5001; }
sleep() { :; }
''' + source[start:end] + '\necho unexpected-model-start\n'
    result = subprocess.run(["bash", "-c", harness], capture_output=True, text=True, timeout=3)
    assert result.returncode == 75, result.stdout + result.stderr
    assert "no new GPU work started" in result.stdout
    assert "unexpected-model-start" not in result.stdout
