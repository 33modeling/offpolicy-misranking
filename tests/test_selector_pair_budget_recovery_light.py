"""CPU-only checks for the approved over-budget Pair handoff."""
import contextlib
import json
import sys
from types import SimpleNamespace

import selector_pair_budget_recovery as recovery


def test_saved_s1_t50_checkpoint_gets_meters_training_before_resume(tmp_path, monkeypatch):
    branch = tmp_path / "branches/on_policy"
    out = branch / "states/s1-t50/points/view-50"
    directory = out / "selection_reduced"
    checkpoint = directory / "policy/checkpoint-345/checkpoint_state.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("{}")
    (directory / "decision.json").write_text("{}")
    c = {"config": {"seed": 1, "drift": 50}, "budget_gpu_seconds": 100.,
         "scope": {"gpu_type": "test-gpu"}}
    calls = []
    held = set()

    @contextlib.contextmanager
    def lease(path):
        assert path not in held
        held.add(path)
        try:
            yield
        finally:
            held.remove(path)

    def meter(path, phase, gpu_type, **kwargs):
        assert directory / ".task.lock" in held
        assert path == directory and phase == "train" and gpu_type == "test-gpu"
        assert kwargs["commands"] == [(["train"], "0,1,2,3")]
        assert kwargs["timeout"] == recovery.SUPPLEMENTAL_GPU_SECONDS / 4
        assert kwargs["ledger"] == "deployment"
        assert kwargs["env"]["PAIR_PROTOCOL_ROOT"] == str(tmp_path)
        calls.append("train")
        (directory / "policy/budget_stop.json").write_text("{}")

    def bind(path, value):
        if path.exists():
            assert json.loads(path.read_text()) == value
        else:
            path.write_text(json.dumps(value))

    def execute(*_):
        assert not held
        calls.append("execute")

    base = SimpleNamespace(spent=lambda _: 101., bind=bind, meter=meter,
                           train_command=lambda *_: ["train"], GPUS=4)
    core = SimpleNamespace(number=lambda value, *_: float(value))
    worker = SimpleNamespace(execute=execute, BRANCHES=("on_policy",), pair_lease=lease,
                             base=base, core=core, manifest=lambda _: {},
                             switch=SimpleNamespace(manifest=lambda _: {}),
                             environment=lambda _: {})
    monkeypatch.setitem(sys.modules, "selector_pair_gpu", worker)
    entry = (branch, out, c, {}, {})
    with recovery.activated(tmp_path):
        worker.execute(entry, "selection_reduced", list("0123"))
    assert worker.execute is execute
    assert calls == ["train", "execute"]
    receipt = json.loads((directory / "supplemental-allocation.json").read_text())
    assert receipt["original_budget_gpu_seconds"] == 100.
    assert receipt["additional_gpu_seconds"] == recovery.SUPPLEMENTAL_GPU_SECONDS
