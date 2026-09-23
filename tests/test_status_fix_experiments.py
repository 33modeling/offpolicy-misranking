"""Combined status screens: one nvidia-smi query per frame, and a progress screen that fits the terminal.

nvidia-smi waits out a 20 s timeout per call on a GPU-faulted node. `status` queried it once per
root (six roots: two minutes per frame) and `progress`, which never shows GPUs, snapshotted every
root twice. `progress` also rendered 80 columns whatever the terminal and clipped the node and
elapsed time away at phone width.
"""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time

import selection_gate as core

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_mopps_comparison_status import fixture as mopps_fixture  # noqa: E402
from test_selection_switch_status import completed_prefix, four_nodes, point, prefix, prepared, running  # noqa: E402


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def four_roots(work, now):
    """Primary switch, MoPPS and two sibling switch suites, all with running work."""
    runs = work / "runs"
    four_nodes(runs / "selection-switch-v1", now)
    mopps_fixture(runs / "mopps-comparison-v1", runs / "selection-switch-v1", now)
    for name, host in (("difficulty", "h100-pod-7f9c2b-node-12"), ("long", "gpu-a-very-long-container-hostname-17")):
        root = runs / f"selection-switch-{name}-v1"
        prepared(root)
        completed_prefix(root)
        running(point(root) / "selection_reduced", host, now=now, phase="fresh-r-validation")
        running(prefix(root, 1), f"{name}-pfx-node", now=now, pid=300)
    (runs / "experiments/logs").mkdir(parents=True, exist_ok=True)
    return runs


def launcher(tmp_path, work, *args, **env):
    """run_experiments.sh through a PATH nvidia-smi that logs every call."""
    binaries, log = tmp_path / "bin", tmp_path / "nvidia-smi.calls"
    binaries.mkdir(exist_ok=True)
    fake = binaries / "nvidia-smi"
    fake.write_text(f'#!/usr/bin/env bash\necho "$*" >> {log}\n'
                    'case "$*" in *query-gpu=index*) echo "0, 746, 81559, 0" ;; esac\n')
    fake.chmod(0o755)
    log.write_text("")
    environ = {**os.environ, "PATH": f"{binaries}:{os.environ['PATH']}", "OM_WORK": str(work),
               "SWITCH_PYTHON": sys.executable, "EXPERIMENTS_NODE_ID": "node-9", **env}
    for key in ("SWITCH_ROOT", "MOPPS_ROOT", "EXPERIMENTS_MBPP_SUITE", "OUT_ROOT", "COLUMNS"):
        if key not in env:
            environ.pop(key, None)
    result = subprocess.run(["bash", "scripts/run_experiments.sh", *args], cwd=ROOT, env=environ,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    queries = [line for line in log.read_text().splitlines() if "query-gpu=index" in line]
    return result.stdout, queries


def test_status_queries_this_nodes_gpus_once_for_every_root(tmp_path):
    work = tmp_path / "work"
    four_roots(work, time.time())
    before = {path: path.read_bytes() for path in work.rglob("*") if path.is_file()}
    out, queries = launcher(tmp_path, work, "status")
    assert len(queries) == 1, queries                     # was one per root: 4 here
    assert out.count("THIS NODE GPUS") == 1 and "gpu0 746MiB 0%" in out
    assert "selection-switch-difficulty-v1: DONE" in out and "selection-switch-long-v1: DONE" in out
    assert before == {path: path.read_bytes() for path in work.rglob("*") if path.is_file()}


def test_status_json_keeps_the_gpu_view_in_both_prepared_snapshots(monkeypatch, tmp_path):
    combined = load("experiments_status")
    now = time.time()
    runs = four_roots(tmp_path / "work", now)
    calls = []
    view = {"host": "node-9", "available": True, "gpus": [], "processes": []}
    assert combined.mopps_status.node_view is combined.switch_status.node_view
    monkeypatch.setattr(combined.switch_status.node_view, "local_gpus", lambda: calls.append(1) or view)
    data = combined.snapshot(runs / "selection-switch-v1", runs / "mopps-comparison-v1", now=now)
    assert data["selection_switch"]["local_gpus"] == view and data["mopps_comparison"]["local_gpus"] == view
    assert len(data["other_experiments"]) == 2
    combined.render(data, width=120)
    assert len(calls) == 1


def test_mopps_snapshot_can_skip_the_gpu_query_and_its_cli_still_shows_gpus(monkeypatch, tmp_path):
    mopps = load("mopps_comparison_status")
    now = time.time()
    root, parent = tmp_path / "mopps", tmp_path / "switch"
    mopps_fixture(root, parent, now)
    calls = []
    monkeypatch.setattr(mopps.node_view, "local_gpus",
                        lambda: calls.append(1) or {"host": "h", "available": False, "gpus": [], "processes": []})
    assert mopps.snapshot(root, now=now, local_gpus=False)["local_gpus"] == [] and calls == []
    assert "THIS NODE GPUS" in mopps.render(mopps.snapshot(root, now=now)) and calls == [1]


def test_progress_reads_each_root_once_and_never_runs_nvidia_smi(monkeypatch, tmp_path):
    progress = load("experiments_progress")
    now = time.time()
    work = tmp_path / "work"
    four_roots(work, now)
    seen = []
    for module in (progress.switch_status, progress.mopps_status):
        original = module.snapshot
        def counted(root, *, _original=original, **kwargs):
            seen.append((Path(root).name, kwargs.get("local_gpus", True)))
            return _original(root, **kwargs)
        monkeypatch.setattr(module, "snapshot", counted)
    monkeypatch.setattr(progress.switch_status.node_view, "local_gpus",
                        lambda: (_ for _ in ()).throw(AssertionError("progress must not query GPUs")))
    text = progress.render(work, width=120, now=now)
    assert sorted(seen) == sorted({(name, False) for name, _ in seen}), seen   # once each, no GPU query
    assert len(seen) == 4 and "NODES  " in text and "long-pfx-node" in text.split("\nNODES  ")[1]
    _, queries = launcher(tmp_path, work, "progress")
    assert queries == []


def test_progress_follows_the_terminal_width(tmp_path):
    work = tmp_path / "work"
    four_roots(work, time.time())
    narrow, _ = launcher(tmp_path, work, "progress", COLUMNS="60")
    assert max(map(len, narrow.splitlines())) <= 60, narrow
    piped, _ = launcher(tmp_path, work, "progress")     # no terminal, no COLUMNS: the old 80
    assert max(map(len, piped.splitlines())) == 80, piped
    explicit, _ = launcher(tmp_path, work, "progress", "--width", "100", COLUMNS="60")
    assert max(map(len, explicit.splitlines())) > 80    # --width still wins


def test_phone_width_keeps_updates_elapsed_node_and_task_visible(tmp_path):
    progress = load("experiments_progress")
    now = time.time()
    work = tmp_path / "work"
    runs = four_roots(work, now)
    stats = point(runs / "selection-switch-v1") / "selection_reduced/policy/grpo_stats.jsonl"
    stats.parent.mkdir(parents=True, exist_ok=True)
    stats.write_text("".join(f'{{"step": {step}}}\n' for step in range(26, 38)))
    for width in (60, 70, 80):
        text = progress.render(work, width=width, now=now)
        lines = text.splitlines()
        assert all(len(line) <= width for line in lines), (width, [l for l in lines if len(l) > width])
        flat = " ".join(" ".join(lines).split())
        assert "REVIEW: saved work blocked." in flat and "duplicate names may share a label)" in flat
        # Every running branch: its updates, elapsed time and full node label survive.
        for host in ("node-1", "node-2", "node-3", "node-4", "h100-pod-7f9c2b-node-12",
                     "gpu-a-very-long-container-hostname-17", "difficulty-pfx-node", "long-pfx-node"):
            assert any(line.split()[-1:] == [host] and "20m34s" in line for line in lines), (width, host)
        assert any("12u 20m34s node-1" in line for line in lines)
        nodes = text[text.index("\nNODES  "):]
        for task in ("s0/t25 difficulty: selection_reduced", "s0/t25 long: selection_reduced",
                     "s3/t25 MoPPS: mopps", "s4/t25 on-policy: prefix"):
            assert task in nodes, (width, task)
        # The header still reads legend, node count, legend.
        assert [line.split()[0] for line in lines[1:] if line[:1].strip()][:3] == ["RUN:", "NODES", "EVAL:"]


def test_wide_progress_layout_is_unchanged(tmp_path):
    """At 100 columns and wider the screen is byte-for-byte the previous layout."""
    progress = load("experiments_progress")
    now = time.time()
    work = tmp_path / "work"
    four_roots(work, now)
    lines = progress.render(work, width=120, now=now).splitlines()
    assert lines[1:4] == ["RUN: updates (u), elapsed, node. BUDGET: allocation exhausted; needs review.",
                          "NODES TRAINING NOW  8  (recorded host labels; duplicate names may share a label)",
                          "EVAL: evaluation only. RESUME: checkpoint validation. REVIEW: saved work blocked."]
    assert "  RUN  s0/t25 selection_reduced fresh-r-validation       20m34s gpu-a-very-long-container-hostname-17" in lines
    assert "  RUN    gpu-a-very-long-contain~ fresh-r-val~      s0/t25 long: selection_reduced" in lines
    assert "  RUN    node-1                   train             s3/t25 MoPPS: mopps" in lines
