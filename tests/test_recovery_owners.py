"""A missing meter cannot hide its surviving detached event workers."""

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


SPEC = importlib.util.spec_from_file_location(
    "recovery_owners", Path(__file__).resolve().parents[1] / "scripts/_recovery_owners.py")
owners = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(owners)


@pytest.fixture
def event(tmp_path, monkeypatch):
    monkeypatch.setattr(owners.base, "node_id", lambda: "this-node")
    def dead(pid, sig):
        assert sig == 0
        raise ProcessLookupError(pid)
    monkeypatch.setattr(owners.os, "kill", dead)
    return {"host": "this-node", "pid": 123, "event_id": "attempt"}, tmp_path


def process(proc_root, pid=456, *, state="S", environment=b"", malformed=False):
    directory = proc_root / str(pid)
    directory.mkdir()
    stat = f"{pid} (worker name) " + " ".join([state] + ["0"] * 18 + ["98765"])
    (directory / "stat").write_text("invalid" if malformed else stat)
    (directory / "environ").write_bytes(environment)
    return directory


def test_remote_owner_does_not_probe_local_pid_or_proc(event, monkeypatch):
    start, proc = event
    start["host"] = "peer-node"
    monkeypatch.setattr(owners.os, "kill", lambda *_: pytest.fail("remote PID must not be probed"))
    assert owners.local_event_owner(start, {}, proc_root=proc / "absent") == "remote"


@pytest.mark.parametrize("pid", [None, 0, -1, "123", True])
def test_missing_or_invalid_owner_pid_is_unknown(event, pid):
    start, proc = event
    start["pid"] = pid
    assert owners.local_event_owner(start, {}, proc_root=proc) == "unknown"


def test_missing_host_is_unknown(event):
    start, proc = event
    start.pop("host")
    assert owners.local_event_owner(start, {}, proc_root=proc) == "unknown"


@pytest.mark.parametrize("progress", [{}, {"pid": 123}])
def test_live_or_reused_owner_pid_prevents_recovery(event, monkeypatch, progress):
    start, proc = event
    seen = []
    monkeypatch.setattr(owners.os, "kill", lambda pid, sig: seen.append((pid, sig)))
    assert owners.local_event_owner(start, progress, proc_root=proc / "absent") == "live"
    assert seen == [(progress.get("pid", start["pid"]), 0)]


def test_conflicting_recorded_and_progress_owners_are_unknown(event, monkeypatch):
    start, proc = event
    monkeypatch.setattr(owners.os, "kill", lambda *_: pytest.fail("conflicting ownership is not evidence of a stopped job"))
    assert owners.local_event_owner(start, {"pid": 789}, proc_root=proc) == "unknown"


@pytest.mark.parametrize("error", [PermissionError(), OSError("unavailable"), OverflowError()])
def test_unverifiable_owner_pid_is_unknown(event, monkeypatch, error):
    start, proc = event
    def denied(*_):
        raise error
    monkeypatch.setattr(owners.os, "kill", denied)
    assert owners.local_event_owner(start, {}, proc_root=proc) == "unknown"


def test_dead_owner_without_tagged_children_is_stopped(event):
    start, proc = event
    process(proc, environment=b"OM_SELECTION_COST_other=1\0")
    assert owners.local_event_owner(start, {}, proc_root=proc) == "stopped"


def test_dead_owner_with_live_detached_event_child_is_live(event):
    start, proc = event
    process(proc, environment=b"OTHER=1\0OM_SELECTION_COST_attempt=1\0")
    assert owners.local_event_owner(start, {}, proc_root=proc) == "live"


def test_zombie_event_child_does_not_block_recovery(event):
    start, proc = event
    directory = process(proc, state="Z", environment=b"OM_SELECTION_COST_attempt=1\0")
    (directory / "environ").unlink()
    assert owners.local_event_owner(start, {}, proc_root=proc) == "stopped"


@pytest.mark.parametrize("filename", ["stat", "environ"])
def test_unreadable_same_uid_process_prevents_proof_of_stopped(event, monkeypatch, filename):
    start, proc = event
    directory = process(proc)
    method = "read_text" if filename == "stat" else "read_bytes"
    original = getattr(Path, method)
    def denied(path, *args, **kwargs):
        if path == directory / filename:
            raise PermissionError(path)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, method, denied)
    assert owners.local_event_owner(start, {}, proc_root=proc) == "unknown"


def test_unreadable_pid_metadata_is_unknown(event, monkeypatch):
    start, proc = event
    directory = process(proc)
    original = Path.stat
    def denied(path, *args, **kwargs):
        if path == directory:
            raise PermissionError(path)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "stat", denied)
    assert owners.local_event_owner(start, {}, proc_root=proc) == "unknown"


def test_other_uid_process_need_not_be_read(event, monkeypatch):
    start, proc = event
    directory = process(proc, environment=b"OM_SELECTION_COST_attempt=1\0")
    original = Path.stat
    def different_uid(path, *args, **kwargs):
        if path == directory:
            return SimpleNamespace(st_uid=os.getuid() + 1)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "stat", different_uid)
    assert owners.local_event_owner(start, {}, proc_root=proc) == "stopped"


def test_malformed_same_uid_process_is_unknown(event):
    start, proc = event
    process(proc, malformed=True)
    assert owners.local_event_owner(start, {}, proc_root=proc) == "unknown"


def test_missing_scan_root_is_unknown(event):
    start, proc = event
    assert owners.local_event_owner(start, {}, proc_root=proc / "absent") == "unknown"


def test_pid_vanished_during_scan_is_ignored(event, monkeypatch):
    start, proc = event
    original = Path.iterdir
    monkeypatch.setattr(Path, "iterdir", lambda path: iter([proc / "456"]) if path == proc else original(path))
    assert owners.local_event_owner(start, {}, proc_root=proc) == "stopped"


def test_missing_file_of_surviving_pid_is_unknown(event):
    start, proc = event
    directory = process(proc)
    (directory / "environ").unlink()
    assert owners.local_event_owner(start, {}, proc_root=proc) == "unknown"


def test_live_child_remains_live_even_if_another_process_is_unreadable(event):
    start, proc = event
    process(proc, pid=456, malformed=True)
    process(proc, pid=457, environment=b"OM_SELECTION_COST_attempt=1\0")
    assert owners.local_event_owner(start, {}, proc_root=proc) == "live"
