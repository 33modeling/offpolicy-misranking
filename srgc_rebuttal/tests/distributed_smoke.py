"""Run with torchrun --standalone --nproc_per_node=2 -m ...distributed_smoke."""

from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch

import torch
import torch.distributed as dist

from .test_torch_backend import ModelBackendTests
from .test_step_checkpoints import make_engine
from srgc_rebuttal.cost_ledger import PhaseLedger
from srgc_rebuttal.distributed import primary
from srgc_rebuttal.timing import torch_meter


def main():
    torch.set_num_threads(1)
    # Obtain the one-device expected update before initializing distributed mode.
    fixture = ModelBackendTests()
    fixture.setUp()
    one = fixture.backend
    one._rollout = fixture.deterministic_rollout
    initial = one.state_dict()
    one.train(["p0", "p1", "p2", "p3"], responses=8, objective="grpo", seed=1)
    expected = one.state_dict()
    dist.init_process_group("gloo")
    fixture.setUp()
    parallel = fixture.backend
    parallel.load_state_dict(initial)
    parallel._rollout = fixture.deterministic_rollout
    events = []
    parallel.cost_meter.record = events.append
    # Unequal local scoring workloads must not introduce per-prompt barriers.
    with parallel.cost_meter.phase("selection"):
        gradients = parallel.score_gradients(["p0", "p1", "p2"], responses=8, group_size=4, seed=1)
    assert set(gradients) == {"p0", "p1", "p2"}
    with parallel.cost_meter.phase("training"):
        parallel.train(["p0", "p1", "p2", "p3"], responses=8, objective="grpo", seed=1)
    actual = parallel.state_dict()
    for name, value in actual["trainable"].items():
        torch.testing.assert_close(value, expected["trainable"][name], rtol=1e-5, atol=1e-8)
    if dist.get_rank() == 0:
        assert events[-1]["state"] == "finished"
        assert len(events[-1]["rank_timings"]) == 2
        print("PASS: unequal-rank timing and global four-prompt update match one rank")
    assert primary(lambda: "rank-zero decision") == "rank-zero decision"
    folder = Path(primary(lambda: tempfile.mkdtemp(prefix="srgc-step-save-")))
    try:
        engine = make_engine(folder)
        checkpoint_events = []
        ledger = PhaseLedger(folder / "distributed-costs")
        def record(event):
            ledger.record(event)
            checkpoint_events.append(event)
        engine.backend.cost_meter = torch_meter(record, cuda=False)
        engine.update()
        path = folder / "prefix-latest.pt"
        saved = torch.load(path, weights_only=False)
        assert saved["step"] == 1
        assert saved["backend"]["optimizer_steps"] == 1
        if dist.get_rank() == 0:
            assert checkpoint_events[-1]["phase"] == "checkpoint_save"
            assert len(checkpoint_events[-1]["rank_timings"]) == 2
        with patch.object(torch, "save", side_effect=OSError("injected checkpoint failure")):
            try:
                engine.update()
            except RuntimeError as exc:
                assert "injected checkpoint failure" in str(exc)
            else:
                raise AssertionError("checkpoint failure did not reach every rank")
        assert torch.load(path, weights_only=False)["step"] == 1
        assert not path.with_suffix(".tmp").exists()
        dist.barrier()
        if dist.get_rank() == 0:
            assert not ledger.totals()["complete"]
            print("PASS: per-update checkpoint, rank timing and failed-write recovery on two ranks")
    finally:
        primary(lambda: shutil.rmtree(folder))
    def fail(*args):
        raise OSError("injected rank-zero write failure")
    for action in (lambda: primary(fail), lambda: torch_meter(fail, cuda=False).begin_phase("fault")):
        try:
            action()
        except RuntimeError as exc:
            assert "injected rank-zero write failure" in str(exc)
        else:
            raise AssertionError("rank-zero failure was not propagated to every rank")
    dist.barrier()
    if dist.get_rank() == 0:
        print("PASS: rank-zero startup and timing-write failures reach every rank without hanging")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
