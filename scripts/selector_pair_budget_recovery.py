"""Evaluate exhausted Pair checkpoints separately, without changing frozen budgets."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
from pathlib import Path

SCHEMA = "selector-pair-budget-recovery/v1"
HERE = Path(__file__).resolve()
HELPER_SHA256 = "60313da41bbbd312c23c1c0a5c9125f2d4ffe246104c946063444711e61e5236"
_backend = None


def backend():
    global _backend
    if _backend is None:
        # A private module namespace reuses the reviewed evaluator without
        # changing the MBPP worker's schema, entry point, or saved artifacts.
        path = HERE.with_name("mbpp_budget_recovery.py")
        if hashlib.sha256(path.read_bytes()).hexdigest() != HELPER_SHA256:
            raise ValueError("Pair saved-policy evaluator changed")
        spec = importlib.util.spec_from_file_location("pair_saved_policy_evaluator", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.SCHEMA, module.HERE = SCHEMA, HERE
        _backend = module
    return _backend


def required(directory):
    import selector_pair_gpu as worker
    if (directory / "result.json").exists() or not (directory / "decision.json").is_file():
        return False
    choice = worker.core.read(directory / "decision.json")
    cap = worker.core.number(choice["budget_gpu_seconds"], "branch budget", 0.)
    used = worker.base.spent(directory)
    finalized = any((directory / "policy" / name).is_file()
                    for name in ("budget_stop.json", "policy_train.json"))
    return used > cap or (used == cap and not finalized)


@contextlib.contextmanager
def activated(root):
    import selector_pair_gpu as worker
    original = worker.execute

    def execute(entry, arm, devices):
        branch, out, c, _, _ = entry
        expected = branch / "states" / f"s{c['config']['seed']}-t{c['config']['drift']}" / "points" / f"view-{c['config']['drift']}"
        if branch.parent.parent == root and branch.name in worker.BRANCHES and out == expected:
            with worker.pair_lease(out / arm / ".task.lock"):
                if required(out / arm):
                    worker.manifest(root)
                    evaluator = backend()
                    result = evaluator.recover(worker.switch.manifest(branch), out / arm,
                                               devices, worker.environment(c))
                    raise ValueError(
                        "saved-checkpoint evaluation complete; original allocation exceeded by "
                        f"{result['over_budget_gpu_seconds']:.3f} GPU-s; "
                        "see budget-recovery/result.json; not a budget-compliant Pair completion")
        return original(entry, arm, devices)

    worker.execute = execute
    try:
        yield
    finally:
        worker.execute = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--point", type=int, required=True)
    parser.add_argument("--shard", type=int, choices=range(4), required=True)
    args = parser.parse_args()
    if args.point < 0:
        parser.error("point must be nonnegative")
    evaluator = backend()
    from light_selection_gate_gpu import install_signal_handlers
    install_signal_handlers()
    evaluator.switch.install_runtime()
    evaluator.evaluate(args.directory, args.point, args.shard)


if __name__ == "__main__":
    main()
