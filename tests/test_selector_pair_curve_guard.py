"""CPU coverage for the operational guard around the frozen Pair curve worker."""

import contextlib
import errno
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import queue_selector_pair_gpu as guard
import selector_pair_diagnostic as diagnostic
import selector_pair_handoff as handoff
from test_selector_pair_wait_regressions import curve_case

REPO = Path(__file__).resolve().parents[1]


@contextlib.contextmanager
def held(path, mode=fcntl.LOCK_EX):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle, mode | fcntl.LOCK_NB)
        yield handle


def test_shard_probe_distinguishes_missing_stale_shared_and_exclusive_locks(tmp_path):
    guard.check_shards(tmp_path)
    assert list(tmp_path.iterdir()) == []
    lock = tmp_path / "shard-1.lock"
    lock.write_bytes(b"original lease data")
    before = lock.read_bytes(), lock.stat().st_ino, lock.stat().st_mtime_ns
    guard.check_shards(tmp_path)
    with held(lock, fcntl.LOCK_SH):
        guard.check_shards(tmp_path)
    with held(lock):
        with pytest.raises(BlockingIOError):
            guard.check_shards(tmp_path)
        (tmp_path / "shard-1.done.json").write_text("{}")
        guard.check_shards(tmp_path)
    assert before == (lock.read_bytes(), lock.stat().st_ino, lock.stat().st_mtime_ns)


def test_point_lease_stays_exclusive_through_probe_and_meter(curve_case, monkeypatch):
    original = guard.check_shards
    checked = []

    def check(target):
        with (target / ".point.lock").open("rb") as handle, pytest.raises(BlockingIOError):
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        checked.append(target)
        original(target)

    monkeypatch.setattr(guard, "check_shards", check)
    original_lease = guard.worker.base.lease
    original_curve = guard.worker.switch.curve_once
    with guard.activated(curve_case.root):
        curve_case.run()
        assert guard.worker.base.lease is original_lease
    assert len(checked) == 4 and len(curve_case.launches) == 16
    assert guard.worker.switch.curve_once is original_curve
    assert (curve_case.directory / "curve.json").exists()


@pytest.mark.parametrize("failure", [PermissionError(errno.EACCES, "unreadable"),
                                      OSError(errno.EIO, "read failed"),
                                      BlockingIOError(errno.EAGAIN, "open failed")])
def test_shard_open_failure_is_error_not_peer_pending(curve_case, monkeypatch, failure):
    target = curve_case.directory / "curve/step-50/shard-0.lock"
    original = Path.open

    def open_path(path, *args, **kwargs):
        if path == target and args == ("rb",):
            raise failure
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_path)
    original_lease = guard.worker.base.lease
    original_curve = guard.worker.switch.curve_once
    expected = RuntimeError if isinstance(failure, BlockingIOError) else type(failure)
    with pytest.raises(expected), guard.activated(curve_case.root):
        curve_case.run()
    assert guard.worker.base.lease is original_lease
    assert guard.worker.switch.curve_once is original_curve
    assert curve_case.launches == []
    with held(target.parent / ".point.lock"):
        pass


def test_worker_startup_failure_remains_error_and_restores_wrapper(curve_case, monkeypatch):
    def fail(*args, **kwargs):
        raise BlockingIOError(errno.EAGAIN, "worker startup")

    monkeypatch.setattr(guard.worker.base, "meter", fail)
    original = guard.worker.base.lease
    with guard.activated(curve_case.root), pytest.raises(RuntimeError, match="startup/allocation"):
        curve_case.run()
    assert guard.worker.base.lease is original


def test_guard_only_observes_known_curve_points(curve_case, monkeypatch):
    inspected = []
    monkeypatch.setattr(guard, "check_shards", inspected.append)
    paths = [curve_case.out / "curve-parent/.point.lock",
             curve_case.directory / "curve/step-50/.point.lock",
             curve_case.out / "selection_reduced/curve/step-50/.point.lock",
             curve_case.directory / "curve/unknown/.point.lock",
             curve_case.out / "unrelated/.point.lock", curve_case.directory / ".task.lock"]

    def curve(*args):
        for path in paths:
            with guard.worker.base.lease(path):
                pass

    monkeypatch.setattr(guard.worker.switch, "curve_once", curve)
    with guard.activated(curve_case.root):
        curve_case.run()
    assert inspected == [path.parent for path in paths[:2]]
    inspected.clear()
    with guard.activated(curve_case.root / "other-root"):
        curve_case.run()
    assert inspected == []


def test_symlinked_shard_lock_is_error_without_following_it(tmp_path):
    external = tmp_path / "external"
    external.write_bytes(b"preserved")
    point = tmp_path / "point"
    point.mkdir()
    (point / "shard-0.lock").symlink_to(external)
    with pytest.raises(RuntimeError, match="symlinked"):
        guard.check_shards(point)
    assert external.read_bytes() == b"preserved"


def frozen_protocol(root):
    p = {"schema": guard.worker.pair.SCHEMA, "code_hashes": guard.worker.code_hashes(),
         "branch_manifests": {}}
    p["protocol_id"] = guard.worker.core.fingerprint(p)
    guard.worker.core.atomic_json(root / "pair.json", p)
    return p


def test_guard_receipt_is_immutable_and_scientific_manifest_unchanged(tmp_path):
    protocol = frozen_protocol(tmp_path)
    manifest = tmp_path / "pair.json"
    before = manifest.read_bytes(), manifest.stat().st_mtime_ns
    guard.bind_receipt(tmp_path, protocol)
    receipt = tmp_path / guard.RECEIPT
    value = json.loads(receipt.read_text())
    assert value["root"] == str(tmp_path) and value["protocol_id"] == protocol["protocol_id"]
    assert value["guard_sha256"] == guard.worker.base.digest(Path(guard.__file__))
    published = receipt.read_bytes(), receipt.stat().st_mtime_ns
    guard.bind_receipt(tmp_path, protocol)
    assert published == (receipt.read_bytes(), receipt.stat().st_mtime_ns)
    assert before == (manifest.read_bytes(), manifest.stat().st_mtime_ns)
    value["guard_sha256"] = "different runtime"
    receipt.write_text(json.dumps(value))
    changed = receipt.read_bytes()
    with pytest.raises(ValueError, match="frozen contract changed"):
        guard.bind_receipt(tmp_path, protocol)
    assert receipt.read_bytes() == changed and manifest.read_bytes() == before[0]


def test_receipt_requires_valid_protocol_and_runtime_exclusive_lease(tmp_path, monkeypatch):
    protocol = frozen_protocol(tmp_path)
    original = guard.worker.base.bind
    calls = []

    def bind(path, value):
        with (tmp_path / ".pair-runtime.lock").open("rb") as handle, pytest.raises(BlockingIOError):
            fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
        calls.append(path)
        original(path, value)

    monkeypatch.setattr(guard.worker.base, "bind", bind)
    with pytest.raises(ValueError, match="protocol changed"):
        guard.bind_receipt(tmp_path, {**protocol, "protocol_id": "wrong"})
    assert calls == []
    guard.bind_receipt(tmp_path, protocol)
    assert calls == [tmp_path / guard.RECEIPT]


def test_run_restores_entrypoints_on_error(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", [guard.__file__, "run", "--root", str(tmp_path)])
    original_stage = guard.worker.run_distributed
    original_curve = guard.worker.switch.curve_once
    original_admission = guard.worker.admit_node

    def fail():
        raise RuntimeError("startup failed")

    monkeypatch.setattr(guard.worker, "main", fail)
    with pytest.raises(RuntimeError, match="startup failed"):
        guard.run()
    assert guard.worker.run_distributed is original_stage
    assert guard.worker.switch.curve_once is original_curve
    assert guard.worker.admit_node is original_admission


@pytest.mark.parametrize("admission", [False, True])
def test_conflicting_guard_receipt_prevents_admission_and_stage(tmp_path, monkeypatch, admission):
    protocol = frozen_protocol(tmp_path)
    (tmp_path / guard.RECEIPT).write_text('{"guard_sha256":"different-runtime"}')
    monkeypatch.setattr(sys, "argv", [guard.__file__, "run", "--root", str(tmp_path)])
    calls = []
    monkeypatch.setattr(guard.worker, "admit_node", lambda *a: calls.append("admission"))
    monkeypatch.setattr(guard.worker, "run_distributed", lambda *a: calls.append("stage"))

    def main():
        with guard.worker.pair_lease(tmp_path / ".pair.lock", shared=True):
            if admission:
                guard.worker.admit_node(tmp_path, protocol)
            guard.worker.run_distributed(tmp_path, protocol, [], "run")

    monkeypatch.setattr(guard.worker, "main", main)
    with pytest.raises(ValueError, match="frozen contract changed"):
        guard.run()
    assert calls == []


@pytest.mark.parametrize("failure,code", [
    (guard.worker.PairWaitTimeout("peer wait expired"), 76),
    (guard.worker.PairLockBusy(Path("/pair/.pair.lock")), 75),
    (guard.worker.NodeAdmissionError("node rejected"), 78),
    (ValueError("invalid contract"), "[pair] invalid contract"),
    (FileNotFoundError("missing manifest"), "[pair] missing manifest"),
])
def test_adapter_preserves_frozen_cli_exit_codes(monkeypatch, failure, code):
    import light_selection_gate_gpu

    events = []
    monkeypatch.setattr(light_selection_gate_gpu, "install_signal_handlers",
                        lambda: events.append("signals"))
    monkeypatch.setattr(guard.worker, "show_pair_activity", lambda root: events.append(str(root)))

    def fail():
        raise failure

    monkeypatch.setattr(guard, "run", fail)
    with pytest.raises(SystemExit) as result:
        guard.main()
    assert result.value.code == code and events[0] == "signals"
    if code == 75:
        assert events == ["signals", "/pair"]


@pytest.mark.parametrize("failure", [OSError(errno.EIO, "read failure"), RuntimeError("runtime failed")])
def test_adapter_preserves_runtime_error_diagnostics_and_traceback(monkeypatch, failure):
    import light_selection_gate_gpu

    events = []
    monkeypatch.setattr(light_selection_gate_gpu, "install_signal_handlers", lambda: None)
    monkeypatch.setattr(guard.worker, "resource_diagnostics", lambda: events.append("diagnostics"))

    def fail():
        raise failure

    monkeypatch.setattr(guard, "run", fail)
    with pytest.raises(type(failure)) as result:
        guard.main()
    assert result.value is failure and events == ["diagnostics"]


@pytest.mark.parametrize("mode", ["run", "develop", "freeze", "test"])
@pytest.mark.parametrize("absolute", [False, True])
def test_worker_helper_routes_exact_pair_argv_and_preserves_tag(tmp_path, mode, absolute):
    python = tmp_path / "python-probe"
    python.write_text('#!/usr/bin/env bash\n[[ "$2" != probe ]] || exit 3\nprintf "%s\\n" "$@"\n')
    python.chmod(0o755)
    source = "src/selector_pair_gpu.py"
    if absolute:
        source = str(REPO / source)
    result = subprocess.run(["bash", "-c", 'set -euo pipefail; source "$1"; shift; selection_run_worker "$@"',
                             "test", str(REPO / "scripts/_selection_worker.sh"),
                             str(python), source, mode, "--root", str(tmp_path)],
                            cwd=REPO, env={**os.environ, "EXPERIMENTS_NODE_ID": "cpu-test-node"},
                            text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [value + " [pair]" for value in
        [str(REPO / "scripts/queue_selector_pair_gpu.py"), mode, "--root", str(tmp_path)]]


@pytest.mark.parametrize("args", [
    ["src/selector_pair_gpu.py", "prepare", "--root", "/tmp/root"],
    ["src/selector_pair_gpu.py", "run", "--root", "/tmp/root", "--extra"],
    ["-c", "src/selector_pair_gpu.py", "run", "--root", "/tmp/root"],
    ["/other/src/selector_pair_gpu.py", "run", "--root", "/tmp/root"],
])
def test_worker_helper_does_not_rewrite_other_commands(tmp_path, args):
    python = tmp_path / "python-probe"
    python.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n')
    python.chmod(0o755)
    result = subprocess.run(["bash", "-c", 'source "$1"; shift; selection_run_worker "$@"',
                             "test", str(REPO / "scripts/_selection_worker.sh"), str(python), *args],
                            cwd=REPO, env={**os.environ, "EXPERIMENTS_NODE_ID": "cpu-test-node"},
                            text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [arg + " [pair]" for arg in args]


@pytest.mark.parametrize("mode", ["run", "develop", "freeze", "test"])
def test_handoff_and_diagnostic_recognize_exact_adapter(tmp_path, monkeypatch, mode):
    args = [sys.executable, "scripts/queue_selector_pair_gpu.py", mode, "--root", str(tmp_path)]
    value = {"argv": args, "cwd": REPO, "exe": Path(sys.executable).resolve()}
    monkeypatch.setattr(handoff, "process", lambda *a: value.copy())
    assert handoff.verify_owner(tmp_path, 123, tmp_path, REPO)["mode"] == mode
    proc = tmp_path / "123"
    proc.mkdir()
    (proc / "cmdline").write_bytes(b"\0".join(os.fsencode(arg) for arg in args) + b"\0")
    assert diagnostic.process_label(tmp_path, 123) == f"queue_selector_pair_gpu.py mode={mode}"
    for words in ([*args, "--extra"], [args[0], "-c", *args[1:]],
                  [args[0], "/other/queue_selector_pair_gpu.py", *args[2:]],
                  [*args[:2], "prepare", *args[3:]], [*args[:4], "/other/root"]):
        value["argv"] = words
        with pytest.raises(RuntimeError, match="does not match"):
            handoff.verify_owner(tmp_path, 123, tmp_path, REPO)
