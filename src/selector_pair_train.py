"""Existing GRPO/curve trainer plus durable, allocation-bound cost timestamps.

Only the new pair runner uses this entry point. No learner, checkpoint frequency,
optimizer, rollout, or existing experiment contract is changed.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import selection_gate as core
import selection_gate_gpu as base


def active_event():
    events = [key.removeprefix("OM_SELECTION_COST_") for key, value in os.environ.items()
              if key.startswith("OM_SELECTION_COST_") and value == "1"]
    if len(events) != 1:
        raise ValueError("trainer requires exactly one active metered allocation")
    return events[0]


def checkpoint_writer(original):
    def write(path, value):
        if path.name == "checkpoint_state.json":
            # This file travels inside the atomic checkpoint directory rename.
            # A kill cannot publish weights without also publishing their cost
            # receipt. Timestamp is immediately before metadata commit/rename.
            core.atomic_json(path.parent / "cost-receipt.json", {
                "step": value["completed_steps"], "adapter_sha256": value["adapter_sha256"],
                "checkpoint_state_id": core.fingerprint(value),
                "event_id": active_event(), "time": time.time()})
        original(path, value)
    return write


def stamp(policy, step, adapter):
    path = policy / "curve-cost/final.json"
    digest = base.digest(adapter)
    if path.exists():
        old = core.read(path)
        if old["step"] != step or old["adapter_sha256"] != digest:
            raise ValueError("cost receipt belongs to a different checkpoint")
        return  # A resumed checkpoint retains its original first publication cost.
    base.bind(path, {"step": step, "adapter_sha256": digest,
                     "event_id": active_event(), "time": time.time()})


def main():
    from selector_pair_gpu import manifest
    manifest(Path(os.environ["PAIR_PROTOCOL_ROOT"]))
    import selection_switch_curve_train as archive
    import train_selection_gate_grpo as trainer
    import train_policy_grpo as checkpoints
    archive.KEEP = (*archive.KEEP, "cost-receipt.json")
    archive.install()
    checkpoints._atomic_json = checkpoint_writer(checkpoints._atomic_json)
    trainer.main()
    if int(os.environ.get("RANK", "0")) == 0:
        policy = Path(sys.argv[sys.argv.index("--output")+1])
        stop = core.read(policy / "budget_stop.json")
        if not stop["use_parent_policy"]:
            stamp(policy, stop["completed_steps"], policy / "adapter_model.safetensors")


if __name__ == "__main__":
    main()
