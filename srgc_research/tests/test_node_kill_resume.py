"""Kill a real learner during torch.save and recover its complete training state."""

import multiprocessing
import signal
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from scripts import srgc_step_checkpoints as checkpoints
from srgc_rebuttal.srgc import Config, Engine
from srgc_rebuttal.tests.test_reference import ScriptedBackend
from srgc_rebuttal.timing import torch_meter


class OptimizingBackend(ScriptedBackend):
    def __init__(self):
        super().__init__()
        torch.manual_seed(82)
        self.model = torch.nn.Linear(1, 1)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=.01)
        self.cost_meter = torch_meter(cuda=False)

    def train(self, ids, **kwargs):
        super().train(ids, **kwargs)
        generator = torch.Generator().manual_seed(kwargs["seed"])
        x = torch.tensor([[sum(map(ord, rid)) % 17 / 17] for rid in ids])
        target = torch.rand((len(ids), 1), generator=generator)
        self.optimizer.zero_grad()
        loss = (self.model(x) - target).square().mean()
        loss.backward()
        self.optimizer.step()
        return {"sample_reward": .5, "loss": loss.item()}

    def state_dict(self):
        return {**super().state_dict(), "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])


def make_engine(folder, task):
    folder.mkdir(parents=True, exist_ok=True)
    ids = [f"s{i}" for i in range(40)] + [f"o{i}" for i in range(360)]
    cache = {rid: ([0, 1] * 4 if rid.startswith("s") else [1] * 8) for rid in ids}
    base = checkpoints.checkpoint_engine(Engine, folder, task, {"attention": "eager"}, total_updates=275)
    engine = base(OptimizingBackend(), ids, ["v"], cache, arm="random" if task == "prefix" else task,
                  config=Config(seed=5, projection_dim=2), step=0 if task == "prefix" else 25)
    engine.backend.strengths = {rid: float(len(ids) - n) for n, rid in enumerate(engine.sr_ranked_ids)}
    return engine


def run_to(engine, folder, task, stop):
    interval = 5 if task == "prefix" else 25
    while engine.step < stop:
        engine.update()
        # The unmodified run_experiment saves these legacy boundaries itself.
        if engine.step % interval == 0:
            checkpoints.save_checkpoint(engine, folder, task)


@pytest.mark.parametrize("task", ("prefix", "random", "sr", "on_policy", "switch"))
def test_sigkill_during_checkpoint_write_recovers_weights_optimizer_and_selection(tmp_path, task):
    context = multiprocessing.get_context("fork")
    parent, child = context.Pipe(duplex=False)
    folder = tmp_path / "interrupted"
    saved_step, final_step = (13, 25) if task == "prefix" else (53, 80)
    checkpoint = folder / f"{task}-latest.pt"

    def learner():
        parent.close()
        engine = make_engine(folder, task)
        run_to(engine, folder, task, saved_step)
        previous = checkpoint.read_bytes()

        def partial_save(state, handle):
            assert state["step"] == saved_step + 1
            handle.write(b"unfinished checkpoint")
            handle.flush()
            child.send_bytes(previous)
            # Parent sends a real SIGKILL while the replacement is incomplete.
            signal.pause()

        with patch.object(torch, "save", side_effect=partial_save):
            engine.update()

    process = context.Process(target=learner)
    try:
        process.start()
        child.close()
        assert parent.poll(30), "learner did not reach interrupted checkpoint write"
        previous = parent.recv_bytes()
        process.kill()
        process.join(10)
        assert process.exitcode == -signal.SIGKILL
        assert checkpoint.read_bytes() == previous
        assert checkpoint.with_suffix(".tmp").read_bytes() == b"unfinished checkpoint"

        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert saved["step"] == saved_step
        assert saved["backend"]["optimizer"]["state"]
        resumed = make_engine(folder, task)
        resumed.load_state_dict(saved)
        run_to(resumed, folder, task, final_step)
        assert not checkpoint.with_suffix(".tmp").exists()

        baseline_folder = tmp_path / "baseline"
        baseline = make_engine(baseline_folder, task)
        run_to(baseline, baseline_folder, task, final_step)
        actual, expected = resumed.state_dict(), baseline.state_dict()
        torch.testing.assert_close(actual["backend"], expected["backend"], rtol=0, atol=0)
        for name in ("step", "arm", "active_selection", "used_training_ids", "sampling_cycle",
                     "rule", "switched_at", "random_ids", "sr_ranked_ids"):
            assert actual[name] == expected[name], name
        ignored = {"selection_gpu_seconds", "training_gpu_seconds"}
        assert [{k: v for k, v in r.items() if k not in ignored} for r in actual["history"]] == [
            {k: v for k, v in r.items() if k not in ignored} for r in expected["history"]]
        updates = [r["completed_updates"] for r in actual["history"]]
        assert updates == list(range(1 if task == "prefix" else 26, final_step + 1))
        if task == "switch":
            assert actual["switched_at"] is not None
    finally:
        if process.is_alive():
            process.kill()
            process.join(10)
        parent.close()
