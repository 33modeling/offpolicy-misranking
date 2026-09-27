"""Run with torchrun --standalone --nproc_per_node=2 -m ...distributed_smoke."""

import torch
import torch.distributed as dist

from .test_torch_backend import ModelBackendTests
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
