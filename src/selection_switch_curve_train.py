"""Trainer entry for convergence-gate switch roots.

The frozen GRPO trainer keeps only its two newest checkpoints and removes all of
them when it publishes the final policy. A convergence gate needs the held-out
reward along the run, so this entry archives each checkpoint's adapter under
<output>/curve-checkpoints/step-<n>/ before the trainer removes it, then runs
the trainer unchanged. Nothing about the updates, the policy or the budget
changes; the optimizer state is not kept.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

KEEP = ("adapter_model.safetensors", "adapter_config.json", "checkpoint_state.json")
_rmtree = shutil.rmtree


def archive_then_remove(path, *args, **kwargs):
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
    return _rmtree(path, *args, **kwargs)


def install():
    shutil.rmtree = archive_then_remove


if __name__ == "__main__":
    install()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import train_selection_gate_grpo as trainer
    trainer.main()
