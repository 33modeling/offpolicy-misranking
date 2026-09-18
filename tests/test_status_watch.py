"""Watching status must not retain outdated READY/DONE display code after pull."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("status_watch", ROOT / "scripts/_status_watch.py")
watch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(watch)


def test_unchanged_viewer_does_not_restart(tmp_path, monkeypatch):
    path = tmp_path / "viewer.py"
    path.write_text("original")
    viewer = watch.StatusWatch([path])
    calls = []
    monkeypatch.setattr(watch.os, "execv", lambda *args: calls.append(args))
    viewer.refresh()
    assert calls == []


def test_code_change_reexecutes_only_viewer_with_identical_arguments(tmp_path, monkeypatch, capsys):
    path = tmp_path / "viewer.py"
    path.write_text("original")
    viewer = watch.StatusWatch([path])
    path.write_text("updated!")
    calls = []
    argv = ["scripts/experiments_status.py", "--switch-root", "/saved/run", "--watch", "--json"]
    monkeypatch.setattr(watch.sys, "argv", argv)
    monkeypatch.setattr(watch.os, "execv", lambda *args: calls.append(args))
    viewer.refresh()
    assert calls == [(watch.sys.executable, [watch.sys.executable, *argv])]
    output = capsys.readouterr()
    assert output.out == "" and "workers untouched" in output.err
    assert path.read_text() == "updated!"
