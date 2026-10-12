"""Durable scalar rewards and separately metered held-out checkpoint curves.

This operational extension does not modify the frozen learner or its identity.
Its own implementation hash is bound to each measurement protocol.
"""

import hashlib
import json
import math
import random
from contextlib import contextmanager
from pathlib import Path

from srgc_rebuttal.runtime import Busy, atomic_json, lease

SCHEMA = "srgc-reward-curve-v1"
INTERVAL = 25
RESPONSES = 8


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def extension_digest():
    paths = (Path(__file__), Path(__file__).with_name("reward_curve_rank.py"))
    return hashlib.sha256(b"".join(p.read_bytes() for p in paths)).hexdigest()


def finite_reward(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("reward must be a finite fraction in [0, 1]")
    return float(value)


def training_points(history, *, start, stop):
    if not isinstance(history, list):
        raise ValueError("training history must be a list")
    points = []
    previous = start
    for item in history:
        if not isinstance(item, dict):
            raise ValueError("training history records must be objects")
        step = item.get("completed_updates")
        if type(step) is not int or step != previous + 1 or item.get("checkpoint") != step - 1 or step > stop:
            raise ValueError("training reward steps are not a contiguous completed-update history")
        metrics = item.get("metrics", {})
        reward = finite_reward(metrics["sample_reward"])
        points.append({"step": step, "reward": reward, "reward_percent": 100 * reward,
                       "selector": item.get("selector"), "kind": "selected_training_batch"})
        previous = step
    if previous != stop:
        raise ValueError("training history does not reach its saved step")
    return points


def windows(points):
    buckets = {}
    for row in points:
        end = ((row["step"] - 1) // INTERVAL + 1) * INTERVAL
        buckets.setdefault(end, []).append(row)
    return [{"start_step": end - INTERVAL + 1, "end_step": end,
             "recorded_updates": len(rows), "complete": len(rows) == INTERVAL,
             "reward": sum(r["reward"] for r in rows) / len(rows),
             "reward_percent": 100 * sum(r["reward"] for r in rows) / len(rows),
             "kind": "selected_training_batch_window_mean"}
            for end, rows in sorted(buckets.items())]


def recording_command(command):
    """Wrap only Gemma/Llama training ranks, retaining their pinned arguments."""
    command = list(command)
    for model in ("gemma4", "llama31"):
        module = f"srgc_research.dispatch.{model}.rank"
        if module not in command or "--stage" not in command or command[command.index("--stage") + 1] != "train":
            continue
        index = command.index(module)
        if index == 0 or command[index - 1] != "--module":
            raise ValueError("unexpected model training rank command")
        return [*command[:index], "srgc_research.dispatch.reward_curve_rank", "--model", model, *command[index + 1:]]
    return command


@contextmanager
def preserve_random_state(backend):
    import numpy as np
    import torch

    python_state, numpy_state = random.getstate(), np.random.get_state()
    model = getattr(backend, "model", None)
    training = getattr(model, "training", None)
    device = getattr(backend, "device", torch.device("cpu"))
    devices = [device.index] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            if model is not None:
                model.eval()
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        if model is not None:
            model.train(training)


@contextmanager
def primary_lease(path):
    from srgc_rebuttal.distributed import primary

    held = []

    def acquire():
        context = lease(path)
        try:
            context.__enter__()
        except Busy:
            return False
        held.append(context)
        return True

    owned = primary(acquire)
    try:
        yield owned
    finally:
        if owned:
            primary(lambda: held.pop().__exit__(None, None, None))



def equal_state(left, right):
    """Compare archived learner state with the weights actually being resumed."""
    import numpy as np
    import torch
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(equal_state(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(equal_state(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def recording_engine(base, folder, task, plan_path, plan, data):
    from srgc_rebuttal.distributed import primary
    from srgc_rebuttal.plan import digest
    from srgc_rebuttal.srgc import stream_seed
    from srgc_rebuttal.cost_ledger import PhaseLedger
    from srgc_rebuttal.timing import torch_meter

    folder = Path(folder)

    class RecordingEngine(base):
        def protocol(self, name):
            marker = json.loads((folder / "run.json").read_text())
            identity = {k: marker[k] for k in ("seed", "plan_sha256", "input_sha256", "implementation_sha256")}
            protocol = {"schema": SCHEMA, **identity, "arm": name,
                        "extension_sha256": extension_digest(), "interval_updates": INTERVAL,
                        "responses_per_prompt": RESPONSES,
                        "evaluation_ids_sha256": fingerprint(data["evaluation_ids"]),
                        "question_count": len(data["evaluation_ids"]),
                        "stream": "reporting-evaluation", "training_batch_reward_is_held_out": False}
            path = folder / "reward-curves" / name / "protocol.json"

            def bind():
                if path.exists() and json.loads(path.read_text()) != protocol:
                    raise ValueError(f"reward curve protocol changed: {path}")
                if not path.exists():
                    atomic_json(path, protocol)
                return protocol

            return primary(bind)

        def save_training_rewards(self):
            protocol = self.protocol(task)
            start = 0 if task == "prefix" else plan["shared_prefix_updates"]
            points = training_points(self.history, start=start, stop=self.step)
            primary(lambda: atomic_json(folder / "reward-curves" / task / "training.json", {
                **protocol, "step": self.step, "points": points, "windows_25": windows(points)}))

        def capture_boundary(self):
            if self.step == 0 or self.step % INTERVAL:
                return
            name = "prefix" if self.step == plan["shared_prefix_updates"] else task
            directory = folder / "reward-curves" / name
            # Prefix weights are shared. Another arm can publish the same
            # step-25 measurement, without making four allocations wait for it.
            with primary_lease(directory / ".measurement.lock") as owned:
                if not owned:
                    return
                protocol = self.protocol(name)
                point_path = directory / f"step-{self.step}.json"
                checkpoint = directory / f"step-{self.step}.pt"
                binding = {"protocol_sha256": fingerprint(protocol), "step": self.step}

                def existing():
                    if point_path.exists():
                        row = json.loads(point_path.read_text())
                        if any(row.get(k) != v for k, v in binding.items()) or not checkpoint.exists() or row.get("checkpoint_sha256") != digest(checkpoint):
                            raise ValueError("saved reward curve point/checkpoint binding changed")
                        validate_evaluation(row, data["evaluation_ids"])
                        import torch
                        archived = torch.load(checkpoint, map_location="cpu", weights_only=False)
                        current = self.backend.state_dict()
                        current.pop("optimizer", None)
                        if archived.get("reward_curve_binding") != binding or not equal_state(archived["backend"], current):
                            raise ValueError("saved reward curve weights differ from the resumed learner")
                        return True
                    return False

                if primary(existing):
                    return
                receipts = directory / "cost-receipts" / f"step-{self.step}"
                meter = torch_meter(lambda event: PhaseLedger(receipts).record(event))

                # A checkpoint without a completed point may be a partial write
                # from a failed attempt. Replace it from these actual weights.
                with meter.phase("checkpoint_save", self.step, self.backend.gpu_count):
                    state = self.state_dict()
                    # Archive learner weights and scalar history for future
                    # evaluation. Optimizer recovery stays in *-latest.pt.
                    state["backend"].pop("optimizer", None)
                    state["evaluation_only"] = True
                    state["reward_curve_binding"] = binding

                    def write():
                        import os
                        import torch
                        temporary = checkpoint.with_suffix(".tmp")
                        try:
                            with temporary.open("wb") as handle:
                                torch.save(state, handle)
                                handle.flush()
                                os.fsync(handle.fileno())
                            temporary.replace(checkpoint)
                        finally:
                            temporary.unlink(missing_ok=True)
                    primary(write)
                # Reuse the unchanged runner's final evaluation at 275.
                if self.step == plan["total_updates"]:
                    return
                original_meter = self.backend.cost_meter
                self.backend.cost_meter = meter
                try:
                    with preserve_random_state(self.backend), meter.phase("evaluation", self.step, self.backend.gpu_count):
                        per_question = self.backend.evaluate(data["evaluation_ids"],
                            seed=stream_seed(self.config.seed, self.step, "reporting-evaluation"), responses=RESPONSES)
                finally:
                    self.backend.cost_meter = original_meter
                point = {**binding, "schema": SCHEMA, "arm": name, "seed": self.config.seed,
                         "kind": "fixed_held_out_evaluation", "responses_per_prompt": RESPONSES,
                         "question_count": len(data["evaluation_ids"]), "per_question_reward": per_question,
                         "reward": sum(per_question.values()) / len(per_question),
                         "checkpoint_sha256": primary(lambda: digest(checkpoint)),
                         "cost_scope": "all_attempts_at_this_step",
                         "cost_receipts": primary(lambda: PhaseLedger(receipts).totals())}
                validate_evaluation(point, data["evaluation_ids"])
                primary(lambda: atomic_json(point_path, point))
                primary(lambda: print(f"REWARD CURVE seed={self.config.seed} arm={name} step={self.step} "
                                      f"held_out={100 * point['reward']:.4f}% saved={point_path}", flush=True))

        def update(self):
            # A resumed boundary may predate this extension or have interrupted
            # its measurement. Evaluate its actual weights before changing them.
            self.capture_boundary()
            result = super().update()
            self.save_training_rewards()
            self.capture_boundary()
            return result

    return RecordingEngine


def validate_evaluation(row, ids):
    values = row.get("per_question_reward")
    if not isinstance(values, dict) or set(values) != set(ids):
        raise ValueError("held-out reward IDs differ from the fixed evaluation set")
    for value in values.values():
        reward = finite_reward(value)
        if not math.isclose(reward * RESPONSES, round(reward * RESPONSES), abs_tol=1e-9):
            raise ValueError("held-out rewards do not match eight binary responses")
    reward = finite_reward(row["reward"])
    if not math.isclose(reward, sum(values.values()) / len(values), abs_tol=1e-12):
        raise ValueError("held-out mean and per-question rewards differ")
