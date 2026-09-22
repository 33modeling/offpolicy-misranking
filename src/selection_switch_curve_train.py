"""Trainer entry for convergence-gate switch roots.

Publish evaluation adapters under <output>/curve-checkpoints/step-<n>/ after
each checkpoint save. Full checkpoints, including optimizer state, remain in
their original directories. The delete hook also protects published checkpoints
when used with an older trainer. Learner updates and budgets are unchanged.
"""
from __future__ import annotations

import shutil
import sys
from functools import wraps
from pathlib import Path

KEEP = ("adapter_model.safetensors", "adapter_config.json", "checkpoint_state.json")
_rmtree = shutil.rmtree


def archive_checkpoint(path):
    path = Path(path)
    if path.is_dir() and path.name.startswith("checkpoint-") and (path / "adapter_model.safetensors").is_file():
        step = int(path.name.split("-", 1)[1])
        target = path.parent / "curve-checkpoints" / f"step-{step}"
        if not (target / "adapter_model.safetensors").is_file():
            temporary = target.with_name(target.name + ".tmp")
            _rmtree(temporary, ignore_errors=True)
            temporary.mkdir(parents=True)
            for name in KEEP:
                if (path / name).is_file():
                    shutil.copy2(path / name, temporary / name)
            temporary.rename(target)


def archive_then_remove(path, *args, **kwargs):
    """Legacy delete hook: retain checkpoint directories instead of deleting."""
    path = Path(path)
    if path.name.startswith("checkpoint-"):
        archive_checkpoint(path)
        return None
    return _rmtree(path, *args, **kwargs)


def archive_after_save(save):
    if getattr(save, "archives_after_save", False):
        return save

    @wraps(save)
    def wrapped(model, optimizer, out_dir, completed_steps, rank, contract):
        result = save(model, optimizer, out_dir, completed_steps, rank, contract)
        if rank == 0:
            archive_checkpoint(Path(out_dir) / f"checkpoint-{completed_steps:06d}")
        return result

    wrapped.archives_after_save = True
    return wrapped


def install():
    import train_policy_grpo as checkpoints
    import train_selection_gate_grpo as trainer
    checkpoints._save_checkpoint = archive_after_save(checkpoints._save_checkpoint)
    trainer._save_checkpoint = checkpoints._save_checkpoint
    shutil.rmtree = archive_then_remove


if __name__ == "__main__":
    install()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import train_selection_gate_grpo as trainer
    trainer.main()
