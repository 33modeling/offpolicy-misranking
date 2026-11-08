"""Launch visibility for the 'Loading weights: 100% then silence' report (11-08).

The child writes straight to its log file, stdin is closed, the parent prints a periodic
status line, the seed runs' child environment is reproduced, NCCL admission runs once per
worker, and the torchrun child prints progress/phase lines and keeps per-rank stack dumps.
"""
from __future__ import annotations

import importlib.util
import io
import os
from pathlib import Path
import sys
import threading
import types

import pytest

REPO = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nodes = load("srgc_v8_nodes_under_test", REPO / "scripts/srgc_v8_nodes.py")


def bare_runtime(tmp_path, gpus=("0", "1", "2", "3")):
    runtime = object.__new__(nodes.Runtime)
    runtime.gpus = list(gpus)
    runtime.environment = dict(os.environ, PYTHONUNBUFFERED="1")
    runtime.root = tmp_path / "olmo_math"
    runtime.cfg = {"runtime": {"profile": "olmo"}, "seeds": {"training": [5]}}
    runtime.admitted = None
    return runtime


def test_child_output_goes_to_file_and_stdin_is_closed(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("V8_STATUS_SECONDS", "0.3")
    monkeypatch.setattr(nodes, "gpu_utilisation", lambda gpus: "97%/1000MiB")
    runtime = bare_runtime(tmp_path)
    log = tmp_path / "logs" / "a.log"
    code = ("import sys,time;print('Loading weights: 100%',flush=True);"
            "print('stdin', repr(sys.stdin.read()), flush=True);time.sleep(1.2);print('PROGRESS done',flush=True)")
    rc = runtime.process([sys.executable, "-c", code], log, [], threading.Event(),
                         extra={"SRGC_PROGRESS_DIR": str(tmp_path / "progress")}, label="trajectory-5")
    assert rc == 0
    text = log.read_text()
    assert "Loading weights: 100%" in text and "stdin ''" in text and "PROGRESS done" in text
    out = capsys.readouterr().out
    assert "trajectory-5 running" in out and "GPU 97%/1000MiB" in out and "last: " in out
    assert (tmp_path / "progress").is_dir()


def test_status_line_reports_silence(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("V8_STATUS_SECONDS", "0.3")
    monkeypatch.setattr(nodes, "gpu_utilisation", lambda gpus: "0%/0MiB")
    runtime = bare_runtime(tmp_path)
    rc = runtime.process([sys.executable, "-c", "import time;print('x',flush=True);time.sleep(1.5)"],
                         tmp_path / "b.log", [], threading.Event(), label="t")
    assert rc == 0
    out = capsys.readouterr().out
    assert "last output 1s ago" in out or "last output 0s ago" in out


def test_stop_event_terminates_child(tmp_path, monkeypatch):
    monkeypatch.setenv("V8_STATUS_SECONDS", "0")
    runtime = bare_runtime(tmp_path)
    stop = threading.Event()
    threading.Timer(0.5, stop.set).start()
    rc = runtime.process([sys.executable, "-c", "import time;time.sleep(30)"], tmp_path / "c.log", [], stop)
    assert rc != 0


def test_admission_runs_once_and_merges_overrides(tmp_path, monkeypatch):
    runtime = bare_runtime(tmp_path)
    fake = types.ModuleType("srgc_rebuttal")
    admission = types.ModuleType("srgc_rebuttal.admission")
    cluster = types.ModuleType("srgc_rebuttal.cluster")
    plan_mod = types.ModuleType("srgc_rebuttal.plan")
    calls = []

    def admit(root, environment, run_child, *, pass_fds=(), plan=None, **_):
        calls.append((root, environment["CUDA_VISIBLE_DEVICES"], plan))
        environment["NCCL_NVLS_ENABLE"] = "0"
        environment["SRGC_LOCAL_MODEL"] = "/models/olmo"
        return {"receipt": "r"}

    admission.admit = admit
    cluster.run_child = lambda *a, **k: 0
    plan_mod.load_plan = lambda path: {"model": "m", "model_revision": "r", "path": str(path)}
    monkeypatch.setitem(sys.modules, "srgc_rebuttal", fake)
    monkeypatch.setitem(sys.modules, "srgc_rebuttal.admission", admission)
    monkeypatch.setitem(sys.modules, "srgc_rebuttal.cluster", cluster)
    monkeypatch.setitem(sys.modules, "srgc_rebuttal.plan", plan_mod)
    monkeypatch.setenv("V8_CODE_ROOT", str(tmp_path))
    runtime.root.mkdir(parents=True)
    (runtime.root / "locator.json").write_text('{"seeds": {"5": {"plan_path": "/p/plan.json"}}}')
    runtime.admit([])
    runtime.admit([])
    assert len(calls) == 1 and calls[0][1] == "0,1,2,3" and calls[0][2]["path"] == "/p/plan.json"
    assert runtime.environment["NCCL_NVLS_ENABLE"] == "0"
    assert runtime.environment["SRGC_LOCAL_MODEL"] == "/models/olmo"
    assert "CUDA_VISIBLE_DEVICES" not in {k for k in runtime.environment if runtime.environment[k] == "0,1,2,3"}


def test_admission_failure_names_the_log(tmp_path, monkeypatch):
    runtime = bare_runtime(tmp_path)
    admission = types.ModuleType("srgc_rebuttal.admission")
    cluster = types.ModuleType("srgc_rebuttal.cluster")
    plan_mod = types.ModuleType("srgc_rebuttal.plan")

    def admit(*a, **k):
        raise RuntimeError("four-GPU admission failed (exit=1)")

    admission.admit = admit
    cluster.run_child = lambda *a, **k: 0
    plan_mod.load_plan = lambda path: {}
    monkeypatch.setitem(sys.modules, "srgc_rebuttal", types.ModuleType("srgc_rebuttal"))
    monkeypatch.setitem(sys.modules, "srgc_rebuttal.admission", admission)
    monkeypatch.setitem(sys.modules, "srgc_rebuttal.cluster", cluster)
    monkeypatch.setitem(sys.modules, "srgc_rebuttal.plan", plan_mod)
    monkeypatch.setenv("V8_CODE_ROOT", str(tmp_path))
    runtime.root.mkdir(parents=True)
    (runtime.root / "locator.json").write_text('{"seeds": {"5": {"plan_path": "/p"}}}')
    with pytest.raises(RuntimeError, match="preflight.log"):
        runtime.admit([])


def test_gpu_utilisation_parses_and_tolerates_missing_tool(monkeypatch):
    class Done:
        stdout = "0, 97, 1000\n1, 95, 900\n2, 0, 3\n"
    monkeypatch.setattr(nodes.subprocess, "run", lambda *a, **k: Done())
    assert nodes.gpu_utilisation(["0", "1"]) == "97%/1000MiB,95%/900MiB"

    def missing(*a, **k):
        raise FileNotFoundError("nvidia-smi")
    monkeypatch.setattr(nodes.subprocess, "run", missing)
    assert nodes.gpu_utilisation(["0"]) == "unknown"


child = None


def load_child(monkeypatch):
    global child
    monkeypatch.syspath_prepend(str(REPO / "v8/experiments"))
    child = load("v8_cluster_child_under_test", REPO / "v8/experiments/cluster_child.py")
    return child


def test_child_progress_wrapper_prints_throttled_on_rank0(monkeypatch, capsys):
    module = load_child(monkeypatch)
    progress = types.ModuleType("srgc_rebuttal.progress")
    seen = []
    progress.record = lambda stage, **d: seen.append(stage)
    backend = types.ModuleType("srgc_rebuttal.torch_backend")
    backend.progress = progress.record
    pkg = types.ModuleType("srgc_rebuttal")
    pkg.progress = progress
    monkeypatch.setitem(sys.modules, "srgc_rebuttal", pkg)
    monkeypatch.setitem(sys.modules, "srgc_rebuttal.progress", progress)
    monkeypatch.setitem(sys.modules, "srgc_rebuttal.torch_backend", backend)
    module.install_progress(0, interval=0)
    backend.progress("generation", prompt=3, responses=8)
    assert seen == ["generation"]
    assert "PROGRESS" in capsys.readouterr().out and "stage=generation" in "".join(seen + ["stage=generation"])
    assert module.install_progress(0, interval=0) is not None   # second install does not double-wrap
    backend.progress("generation", prompt=5)
    assert capsys.readouterr().out.count("PROGRESS") == 1
    quiet = types.ModuleType("srgc_rebuttal.torch_backend")
    quiet.progress = lambda stage, **d: None
    monkeypatch.setitem(sys.modules, "srgc_rebuttal.torch_backend", quiet)
    progress.record = lambda stage, **d: None
    module.install_progress(1, interval=0)
    quiet.progress("generation", prompt=4)
    assert "PROGRESS" not in capsys.readouterr().out


def test_child_diagnostics_file(monkeypatch, tmp_path):
    module = load_child(monkeypatch)
    monkeypatch.setenv("V8_STACK_DUMP_SECONDS", "0")
    handle = module.install_diagnostics(tmp_path, "trajectory-5", 2)
    handle.flush()
    assert (tmp_path / ".multi-node/stacks/trajectory-5-rank2.txt").read_text().startswith("== ")
    import faulthandler
    faulthandler.disable()
    handle.close()
