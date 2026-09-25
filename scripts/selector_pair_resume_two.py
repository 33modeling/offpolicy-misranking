"""Resume the two approved saved Pair finals through the ordinary launcher."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import selection_gate as core

TARGETS = {1: (50, "selection_reduced"), 4: (100, "random_full")}
RESULT_SCHEMA = "offpolicy-selected-prefix-switch/v1"
FINAL_SCHEMA = "selector-pair-saved-final-evaluation/v1"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def directory(root, seed, step, branch, arm):
    return root / f"branches/{branch}/states/s{seed}-t{step}/points/view-{step}/{arm}"


def output_root(root):
    default = root.with_name("selector-pair-final-eval-v1" if root.name == "selector-pair-v1"
                             else root.name + "-final-eval")
    return Path(os.environ.get("PAIR_FINAL_EVAL_ROOT", default)).absolute()


def canonical_complete(path):
    result = core.read(path / "result.json")
    seal = core.read(path / "result.sha256.json")
    curve = core.read(path / "curve.json")
    return (result.get("schema") == RESULT_SCHEMA and result.get("complete") is True
            and seal == {"sha256": digest(path / "result.json")}
            and curve.get("schema") == RESULT_SCHEMA and curve.get("result_sha256") == seal["sha256"]
            and isinstance(curve.get("points"), dict) and bool(curve["points"]))


def eligible(root):
    """Read-only dispatch check, not a new scientific completion certificate."""
    if not (root / "pair.json").is_file():
        return False
    p = core.read(root / "pair.json")
    if p.get("schema") == "offpolicy-selector-pair/setup-v1":
        return False
    if (p.get("schema") != "offpolicy-selector-pair/v1"
            or set(p.get("branch_manifests", {})) != {"on_policy", "cached", "adaptive-on_policy", "adaptive-cached"}):
        raise ValueError("unknown Pair experiment")
    if p.get("protocol_id") != core.fingerprint({k: v for k, v in p.items() if k != "protocol_id"}):
        raise ValueError("Pair manifest changed")
    for branch, value in p["branch_manifests"].items():
        if digest(root / "branches" / branch / "switch.json") != value:
            raise ValueError("Pair branch manifest changed")
    for seed, (step, arm) in TARGETS.items():
        path = directory(root, seed, step, "on_policy", arm)
        if ((path / "result.json").exists() or not (path / "policy/policy_train.json").is_file()
                or not (path / "budget-recovery/plan.json").is_file()):
            return False
    if not (root / "test-decisions.json").is_file():
        return False
    barrier = core.read(root / "test-decisions.json")
    if barrier.get("protocol_id") != p["protocol_id"]:
        raise ValueError("Pair decision barrier changed")
    folder = "sr-gc" if barrier.get("schema") == "offpolicy-selector-pair/sr-gc-v1" else "decisions"
    for seed in range(5):
        for step in (25, 50, 100):
            arms = [(b, "selection_reduced") for b in ("on_policy", "cached")]
            if seed >= 3:
                key = f"s{seed}-t{step}"
                choice_path = root / folder / key / "decision.json"
                if digest(choice_path) != barrier["decisions"][key]:
                    raise ValueError("Pair saved decision changed")
                selector = core.read(choice_path)["selector"]
                if selector not in {"on_policy", "cached"}:
                    raise ValueError("unknown Pair selector")
                arms = [(b, "selection_full") for b in ("on_policy", "cached", f"adaptive-{selector}")]
                arms.append(("on_policy", "random_full"))
            for branch, arm in arms:
                if branch == "on_policy" and TARGETS.get(seed) == (step, arm):
                    continue
                try:
                    if not canonical_complete(directory(root, seed, step, branch, arm)):
                        return False
                except FileNotFoundError:
                    return False
    return True


def completed(root, seed):
    """Validate the saved-final publication binding for the status view."""
    target = output_root(root) / f"seed-{seed}"
    path = target / "result.json"
    if not path.is_file():
        return False
    result, plan = core.read(path), core.read(target / "plan.json")
    step, arm = TARGETS[seed]
    source = directory(root, seed, step, "on_policy", arm)
    if (result.get("schema") != FINAL_SCHEMA or plan.get("schema") != FINAL_SCHEMA
            or result.get("evaluation_complete") is not True or result.get("canonical_complete") is not False
            or result.get("seed") != seed or result.get("start_step") != step
            or plan.get("source_root") != str(root) or plan.get("directory") != str(source)
            or core.read(path.with_suffix(".sha256.json")) != {"sha256": digest(path)}
            or result.get("plan_sha256") != digest(target / "plan.json")):
        raise ValueError("saved final evaluation binding changed")
    final = plan["points"][-1]
    if (not final.get("final") or final.get("adapter") != str(source / "policy")
            or final["hashes"].get("policy_train.json") != digest(source / "policy/policy_train.json")):
        raise ValueError("saved final evaluation does not match current policy")
    return True


def run(root, devices):
    import selector_pair_finish_saved as finish
    from light_selection_gate_gpu import install_signal_handlers
    install_signal_handlers()
    if not eligible(root):
        raise ValueError("only the approved two saved finals may use this resume path")
    if len(devices) != 4 or len(set(devices)) != 4 or any(not d for d in devices):
        raise ValueError("four distinct allocated GPUs required")
    failed = False
    for seed in TARGETS:
        try:
            result = finish.finish(root, output_root(root), seed, devices)
            print(f"[pair-final-complete] s{seed} step={result['completed_steps']} "
                  "evaluation saved; original budget eligibility unchanged", flush=True)
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            print(f"[pair-final-wait] s{seed}: {exc}", file=sys.stderr, flush=True)
            failed = True
    return int(failed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("probe", "run", "frozen-run"))
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.mode == "probe":
        # 3 means ordinary queue work remains; other failures must not be hidden.
        return 0 if eligible(root) else 3
    if args.mode == "frozen-run":
        import selector_pair_deploy as deploy
        runtime = deploy.stage_runtime(Path(__file__).resolve().parents[1])
        os.execv("/bin/bash", ["bash", str(runtime / "scripts/run_selector_pair.sh"), "run"])
    return run(root, os.environ.get("CUDA_VISIBLE_DEVICES", "").split(","))


if __name__ == "__main__":
    raise SystemExit(main())
