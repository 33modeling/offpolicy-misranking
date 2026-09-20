"""Small, read-only lock evidence must not reset or interfere with experiments."""

import contextlib
import fcntl
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/selector_pair_diagnostic.py"


@pytest.fixture
def diagnostic():
    spec = importlib.util.spec_from_file_location("selector_pair_diagnostic_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def proc(tmp_path):
    directory = tmp_path / "proc"
    directory.mkdir()
    (directory / "locks").write_text("")
    return directory


@contextlib.contextmanager
def lease(root, mode=fcntl.LOCK_EX):
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".pair.lock"
    if not lock.exists():
        lock.write_bytes(b"preserve lock inode and bytes\n")
    with lock.open("rb") as handle:
        fcntl.flock(handle, mode | fcntl.LOCK_NB)
        yield lock


def snapshot(root):
    return {str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mtime_ns,
                                          path.stat().st_ino)
            for path in root.rglob("*") if path.is_file() and not path.is_symlink()}


def progress(root, host="observed-peer-node", **fields):
    path = root / "branches/on_policy/states/s0-t25/points/view-25/selection_reduced/progress.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"state": "running", "updated": time.time(), "host": host,
            "phase": "training", "pid": 45678}
    data.update(fields)
    path.write_text(json.dumps(data))
    return path


def owner_record(lock, pid, *, waiting=False, kind="WRITE", inode=None):
    stat = lock.stat()
    device = f"{os.major(stat.st_dev):02x}:{os.minor(stat.st_dev):02x}"
    return (f"1: {'-> ' if waiting else ''}FLOCK  ADVISORY  {kind} {pid} "
            f"{device}:{stat.st_ino if inode is None else inode} 0 EOF\n")


def assert_status(report, status):
    assert any("ROOT_LOCK" in line and status in line for line in report.splitlines()), report
    assert len(report.encode("utf-8")) <= 4096


def test_missing_root_and_lock_are_not_created(diagnostic, proc, tmp_path):
    root = tmp_path / "missing-volume/runs/selector-pair-v1"
    report = diagnostic.collect(root, proc=proc)
    assert_status(report, "missing")
    assert not (tmp_path / "missing-volume").exists()


def cost_fixture(root, *, receipt=False):
    directory = root / 'branches/on_policy/states/s0-t25/points/view-25/selection_reduced'
    directory.mkdir(parents=True)
    start = dict(event_id='open-event', phase='train', ledger='deployment', gpus=4,
                 gpu_type='H100', host='old-node', pid=12345, state='started', time=100.)
    (directory / 'cost.jsonl').write_text(json.dumps(start) + '\n')
    (directory / 'progress.json').write_text(json.dumps({**start, 'state': 'running',
                                                        'updated': 130., 'seconds': 30.}))
    if receipt:
        target = directory / 'cost-events/open-event.json'
        target.parent.mkdir()
        target.write_text(json.dumps({**start, 'state': 'finished', 'seconds': 42.,
                                     'allocated_gpu_seconds': 168., 'exit_code': 1, 'time': 142.}))
    return directory


@pytest.mark.parametrize('receipt', [False, True])
def test_cost_evidence_exports_exact_open_start_and_receipt_without_repair(diagnostic, tmp_path, receipt):
    root = tmp_path / 'pair'
    directory = cost_fixture(root, receipt=receipt)
    before = snapshot(root)
    report = diagnostic.cost_report(root)
    assert '"open_event_ids": ["open-event"]' in report
    assert 'OPEN_EVENT ' in report and '"time": 100.0' in report
    prefix = 'FILE' if receipt else 'MISSING'
    assert f'{prefix} branches/on_policy/states/s0-t25/points/view-25/selection_reduced/cost-events/open-event.json' in report
    assert ('"allocated_gpu_seconds": 168.0' in report) is receipt
    assert 'No cost repair, inferred durations' in report
    assert snapshot(root) == before
    assert not (directory / '.cost.lock').exists()


def test_cost_evidence_preserves_torn_ledger_and_exports_damage(diagnostic, tmp_path):
    root = tmp_path / 'pair'
    directory = cost_fixture(root)
    with (directory / 'cost.jsonl').open('ab') as handle:
        handle.write(b'{"event_id":"open-event","state":"fin')
    before = snapshot(root)
    report = diagnostic.cost_report(root)
    assert 'bytes_hex=' in report and '"open_event_ids": ["open-event"]' in report
    assert snapshot(root) == before


def test_cost_evidence_cli_writes_one_bounded_txt(diagnostic, tmp_path, monkeypatch, capsys):
    root = tmp_path / 'pair'
    cost_fixture(root)
    target = tmp_path / 'reports'
    target.mkdir()
    monkeypatch.setattr(sys, 'argv', [str(SCRIPT), '--root', str(root), '--report-dir', str(target), '--costs'])
    before = snapshot(root)
    assert diagnostic.main() == 0
    paths = list(target.iterdir())
    assert len(paths) == 1 and paths[0].name.startswith('selector-pair-cost-')
    assert paths[0].stat().st_size <= 1024 * 1024
    assert 'SELECTOR PAIR COST EVIDENCE' in paths[0].read_text()
    assert snapshot(root) == before
    assert '[saved]' in capsys.readouterr().out


def test_no_argument_bash_exports_cost_evidence_to_one_txt(tmp_path):
    work = tmp_path / 'work'
    root = work / 'runs/selector-pair-v1'
    cost_fixture(root)
    home = tmp_path / 'home'
    home.mkdir()
    before = snapshot(root)
    environment = {key: value for key, value in os.environ.items() if key != 'PAIR_ROOT'}
    result = subprocess.run(['bash', str(SCRIPT.with_name('check_selector_pair.sh'))],
                            env={**environment, 'HOME': str(home), 'OM_WORK': str(work)},
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    reports = list(home.iterdir())
    assert len(reports) == 1 and reports[0].name.startswith('selector-pair-cost-')
    assert reports[0].stat().st_size <= 1024 * 1024
    assert '"open_event_ids": ["open-event"]' in reports[0].read_text()
    assert snapshot(root) == before


def test_cost_evidence_missing_root_and_external_symlink_are_not_followed(diagnostic, tmp_path):
    root = tmp_path / 'absent'
    assert 'examined=0' in diagnostic.cost_report(root)
    assert not root.exists()
    external = tmp_path / 'outside'
    external.mkdir()
    (external / 'cost.jsonl').write_text('DO-NOT-COPY-EXTERNAL\n')
    branch = root / 'branches/on_policy/states/s0-t25/points/view-25'
    branch.mkdir(parents=True)
    (branch / 'selection_reduced').symlink_to(external, target_is_directory=True)
    report = diagnostic.cost_report(root)
    assert 'metadata escapes Pair root' in report
    assert 'DO-NOT-COPY-EXTERNAL' not in report


def test_existing_root_without_lock_is_not_initialized(diagnostic, proc, tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "pair.json").write_text('{"existing": true}')
    before = snapshot(root)
    assert_status(diagnostic.collect(root, proc=proc), "missing")
    assert snapshot(root) == before
    assert not (root / ".pair.lock").exists()


@pytest.mark.parametrize("mode,status", [(fcntl.LOCK_EX, "blocked"),
                                        (fcntl.LOCK_SH, "shared-compatible")])
def test_real_exclusive_and_shared_flocks_are_distinguished(diagnostic, proc, tmp_path, mode, status):
    root = tmp_path / "root"
    with lease(root, mode) as lock:
        before = snapshot(root)
        report = diagnostic.collect(root, proc=proc)
        assert_status(report, status)
        assert snapshot(root) == before
        with lock.open("rb") as probe:
            with pytest.raises(BlockingIOError):
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_released_file_is_shared_compatible_not_busy(diagnostic, proc, tmp_path):
    root = tmp_path / "root"
    with lease(root):
        pass
    before = snapshot(root)
    assert_status(diagnostic.collect(root, proc=proc), "shared-compatible")
    assert snapshot(root) == before


def test_unreadable_lock_is_not_reported_as_missing_or_unlocked(diagnostic, proc, tmp_path):
    root = tmp_path / "root"
    (root / ".pair.lock").mkdir(parents=True)
    assert_status(diagnostic.collect(root, proc=proc), "unreadable")
    assert (root / ".pair.lock").is_dir()


def test_confirmed_local_owner_requires_matching_kernel_write_lock(diagnostic, proc, tmp_path):
    root = tmp_path / "root"
    with lease(root) as lock:
        pid = 32123
        (proc / str(pid)).mkdir()
        (proc / str(pid) / "cmdline").write_bytes(
            b"/opt/venv/bin/python\0src/selector_pair_gpu.py\0run\0--root\0" +
            os.fsencode(root) + b"\0--secret-token\0DO-NOT-EXPOSE-ARGV-SECRET\0")
        (proc / str(pid) / "environ").write_bytes(b"DO_NOT_EXPOSE_ENV=environment-secret\0")
        (proc / "locks").write_text(owner_record(lock, pid))
        report = diagnostic.collect(root, proc=proc)
        assert_status(report, "blocked")
        assert any("confirmed-local" in line and str(pid) in line
                   for line in report.splitlines()), report
        assert "selector_pair_gpu.py" in report
        assert "DO-NOT-EXPOSE-ARGV-SECRET" not in report
        assert "environment-secret" not in report


@pytest.mark.parametrize("record", ["waiting", "read", "wrong-inode"])
def test_waiters_readers_and_other_inodes_are_not_confirmed_owners(diagnostic, proc, tmp_path, record):
    root = tmp_path / "root"
    with lease(root) as lock:
        kwargs = {"waiting": True} if record == "waiting" else (
            {"kind": "READ"} if record == "read" else {"inode": lock.stat().st_ino + 1})
        (proc / "locks").write_text(owner_record(lock, 987654, **kwargs))
        report = diagnostic.collect(root, proc=proc)
        assert_status(report, "blocked")
        assert not any("confirmed-local" in line and "987654" in line
                       for line in report.splitlines()), report


def test_metadata_host_is_observation_not_confirmed_lock_owner(diagnostic, proc, tmp_path):
    root = tmp_path / "root"
    progress(root)
    with lease(root):
        report = diagnostic.collect(root, proc=proc)
    lines = [line for line in report.splitlines() if "observed-peer-node" in line]
    assert lines, report
    assert all("confirmed-local" not in line for line in lines), report
    assert any("age" in line.lower() for line in lines), report


def test_collect_preserves_checkpoint_results_costs_and_progress(diagnostic, proc, tmp_path):
    root = tmp_path / "root"
    progress(root)
    for relative, payload in {
        "pair.json": b'{"frozen":true}',
        "pair-wait-guard-runtime.json": b'{"preserve":true}',
        "development/s0-t25/result.json": b'{"done":true}',
        "branches/on_policy/cost.jsonl": b'{"open":true}\n',
        "branches/on_policy/policy/checkpoint-000025/optimizer.pt": b"saved optimizer",
        "branches/on_policy/policy/checkpoint-000025/adapter_model.safetensors": b"saved model",
    }.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    with lease(root):
        before = snapshot(root)
        report = diagnostic.collect(root, proc=proc)
        assert snapshot(root) == before
        assert_status(report, "blocked")


def test_report_is_bounded_even_with_large_unicode_metadata(diagnostic, proc, tmp_path):
    root = tmp_path / "root"
    progress(root, host="노드" * 2000, phase="학습" * 2000)
    workers = root / "queue-workers"
    workers.mkdir()
    for index in range(12):
        (workers / f"worker-{index}.json").write_text(json.dumps({
            "state": "WAIT", "host": "노드" * 2000, "updated": time.time(),
        }))
    with lease(root):
        report = diagnostic.collect(root, proc=proc)
    assert_status(report, "blocked")
    assert len(report.encode("utf-8")) <= 4096


def test_metadata_scan_skips_payload_directories_and_symlinks(diagnostic, proc, tmp_path):
    root = tmp_path / "root"
    progress(root)
    branches = root / "branches/on_policy"
    for name in ("policy", "curve-checkpoints", "selector-work"):
        directory = branches / name
        directory.mkdir()
        (directory / "progress.json").write_text(json.dumps({
            "host": "excluded-payload-host", "state": "running", "updated": time.time(),
        }))
    outside = tmp_path / "outside-run"
    outside.mkdir()
    (outside / "progress.json").write_text(json.dumps({
        "host": "outside-symlink-host", "state": "running", "updated": time.time(),
    }))
    (branches / "linked-outside").symlink_to(outside, target_is_directory=True)
    with lease(root):
        report = diagnostic.collect(root, proc=proc)
    assert "observed-peer-node" in report
    assert "excluded-payload-host" not in report
    assert "outside-symlink-host" not in report


@pytest.mark.parametrize("data", ["{not valid json", "null", "[]", '"string"', "x" * 70000])
def test_bad_progress_does_not_hide_lock_state(diagnostic, proc, tmp_path, data):
    root = tmp_path / "root"
    path = progress(root)
    path.write_text(data)
    with lease(root):
        assert_status(diagnostic.collect(root, proc=proc), "blocked")


def test_cli_writes_unique_bounded_reports_without_creating_missing_root(tmp_path):
    root = tmp_path / "missing-group-volume/runs/selector-pair-v1"
    reports = tmp_path / "reports"
    reports.mkdir()
    for _ in range(2):
        result = subprocess.run([sys.executable, str(SCRIPT), "--root", str(root),
                                 "--report-dir", str(reports)],
                                capture_output=True, text=True, timeout=10, check=False)
        assert result.returncode == 0, result.stdout + result.stderr
        assert re.search(r"selector-pair-lock-\S+\.txt", result.stdout), result.stdout
    artifacts = list(reports.glob("selector-pair-lock-*.txt"))
    assert len(artifacts) == 2
    assert all(0 < path.stat().st_size <= 4096 for path in artifacts)
    assert not (tmp_path / "missing-group-volume").exists()


@pytest.mark.parametrize("configuration", ["PAIR_ROOT", "OM_WORK"])
def test_cli_uses_configured_pair_root_without_initializing_it(tmp_path, configuration):
    work = tmp_path / "absent-work"
    root = work / "runs/selector-pair-v1"
    reports = tmp_path / "reports"
    reports.mkdir()
    env = {key: value for key, value in os.environ.items()
           if key not in {"PAIR_ROOT", "OM_WORK", "OM_USER"}}
    env[configuration] = str(root if configuration == "PAIR_ROOT" else work)
    result = subprocess.run([sys.executable, str(SCRIPT), "--report-dir", str(reports)],
                            env=env, capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    artifact, = reports.glob("selector-pair-lock-*.txt")
    assert str(root) in artifact.read_text()
    assert not work.exists()
