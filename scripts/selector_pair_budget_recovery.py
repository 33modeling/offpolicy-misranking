"""Resume approved Pair overruns; evaluate other exhausted checkpoints separately."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
from pathlib import Path
import sys

SCHEMA = "selector-pair-budget-recovery/v1"
HERE = Path(__file__).resolve()
HELPER_SHA256 = "60313da41bbbd312c23c1c0a5c9125f2d4ffe246104c946063444711e61e5236"
_backend = None
SUPPLEMENTAL_GPU_SECONDS = 28800.
SUPPLEMENTAL_BRANCHES = {
    ("on_policy", 1, 50, "selection_reduced"),
    ("on_policy", 4, 100, "random_full"),
}


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


def supplemental_required(branch, out, c, arm):
    """Only the explicitly approved branch may train past its frozen cap."""
    import selector_pair_gpu as worker
    directory = out / arm
    if ((branch.name, c["config"]["seed"], c["config"]["drift"], arm) not in SUPPLEMENTAL_BRANCHES
            or not (directory / "decision.json").is_file()
            or (directory / "result.json").exists()
            or any((directory / "policy" / name).exists()
                   for name in ("budget_stop.json", "policy_train.json"))
            or not any((directory / "policy").glob("checkpoint-*/checkpoint_state.json"))):
        return False
    cap = worker.core.number(c["budget_gpu_seconds"], "branch budget", 0.)
    return worker.base.spent(directory) >= cap


def supplemental_train(root, entry, arm, devices):
    import selector_pair_gpu as worker
    branch, out, c, _, _ = entry
    directory = out / arm
    worker.manifest(root)
    worker.switch.manifest(branch)
    cap = worker.core.number(c["budget_gpu_seconds"], "branch budget", 0.)
    used = worker.base.spent(directory)
    worker.base.bind(directory / "supplemental-allocation.json", {
        "schema": "selector-pair-supplemental-allocation/v1",
        "branch": branch.name, "seed": c["config"]["seed"], "start_step": c["config"]["drift"],
        "original_budget_gpu_seconds": cap,
        "additional_gpu_seconds": SUPPLEMENTAL_GPU_SECONDS,
        "purpose": "complete saved Pair training; report actual over-budget cost",
    })
    print(f"[pair-over-budget] {branch.name} s{c['config']['seed']}-t{c['config']['drift']} "
          f"{arm}: resuming saved checkpoint; "
          f"used={used:.3f}, original cap={cap:.3f}, additional allocation="
          f"{SUPPLEMENTAL_GPU_SECONDS:.0f} GPU-s", file=sys.stderr, flush=True)
    env = {**worker.environment(c), "PAIR_PROTOCOL_ROOT": str(root)}
    worker.base.meter(directory, "train", c["scope"]["gpu_type"],
                      commands=[(worker.base.train_command(out, c, arm, SUPPLEMENTAL_GPU_SECONDS),
                                 ",".join(devices))],
                      env=env, timeout=SUPPLEMENTAL_GPU_SECONDS / worker.base.GPUS,
                      ledger="deployment")
    if not (directory / "policy" / "budget_stop.json").is_file():
        raise ValueError("supplemental training did not publish a final policy; saved checkpoints remain")


@contextlib.contextmanager
def activated(root):
    import selector_pair_gpu as worker
    original = worker.execute

    def execute(entry, arm, devices):
        branch, out, c, _, _ = entry
        expected = branch / "states" / f"s{c['config']['seed']}-t{c['config']['drift']}" / "points" / f"view-{c['config']['drift']}"
        if branch.parent.parent == root and branch.name in worker.BRANCHES and out == expected:
            with worker.pair_lease(out / arm / ".task.lock"):
                if supplemental_required(branch, out, c, arm):
                    supplemental_train(root, entry, arm, devices)
                elif required(out / arm):
                    worker.manifest(root)
                    evaluator = backend()
                    result = evaluator.recover(worker.switch.manifest(branch), out / arm,
                                               devices, worker.environment(c))
                    raise ValueError(
                        "saved-checkpoint evaluation complete; original allocation exceeded by "
                        f"{result['over_budget_gpu_seconds']:.3f} GPU-s; "
                        "see budget-recovery/result.json; not a budget-compliant Pair completion")
        try:
            return original(entry, arm, devices)
        except ValueError as exc:
            if ("branch allocation exhausted" not in str(exc)
                    or branch.parent.parent != root or out != expected
                    or not supplemental_required(branch, out, c, arm)):
                raise
            with worker.pair_lease(out / arm / ".task.lock"):
                supplemental_train(root, entry, arm, devices)
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
