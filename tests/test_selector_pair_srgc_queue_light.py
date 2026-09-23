"""CPU-only scheduling checks for the pinned SR-GC Pair runner."""
import contextlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def runner(monkeypatch):
    class IncompletePairRun(Exception):
        pass

    class PairWaitTimeout(Exception):
        pass

    worker = SimpleNamespace(IncompletePairRun=IncompletePairRun,
                             PairWaitTimeout=PairWaitTimeout,
                             queue_lease=lambda _: contextlib.nullcontext(),
                             completed_state_leases=lambda *_, **__: contextlib.nullcontext())
    monkeypatch.setitem(sys.modules, "selector_pair_gpu", worker)
    monkeypatch.setitem(sys.modules, "selector_pair_srgc_score", SimpleNamespace())
    path = Path(__file__).resolve().parents[1] / "scripts/selector_pair_srgc.py"
    spec = importlib.util.spec_from_file_location("selector_pair_srgc_queue_light", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, worker


@pytest.mark.parametrize("development_pending", [False, True])
def test_frozen_decisions_then_development_before_test(runner, monkeypatch, development_pending):
    module, worker = runner
    calls = []
    monkeypatch.setattr(module, "attempt_freeze", lambda *args: calls.append("freeze") or None)
    monkeypatch.setattr(module, "report", lambda *args: calls.append("report"))

    def stage(root, protocol, devices, name):
        calls.append(name)
        if name == "development" and development_pending:
            raise worker.IncompletePairRun("development pending")

    worker.distributed_stage = stage
    if development_pending:
        with pytest.raises(worker.IncompletePairRun, match="development pending"):
            module.run_stages(Path("/tmp/pair"), {}, [], "run")
        assert calls == ["freeze", "development", "test"]
    else:
        module.run_stages(Path("/tmp/pair"), {}, [], "run")
        assert calls == ["freeze", "development", "test", "report"]
