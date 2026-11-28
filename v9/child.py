#!/usr/bin/env python3
"""torchrun child for one V9 task (one process per GPU).

    torchrun --standalone --nproc_per_node=4 v9/child.py --cohort olmo_math --task-file <task.json> --root $V9_ROOT

Task kinds:
  train     V8 execute_train (fork from a retained V8/V9 state, resumable from latest.pt)
  evaluate  V8 execute_eval  (immutable final-split evaluation of one retained snapshot)
  measure   v9lib.measure.execute_measure (X5; scoring and log-probs only, no optimizer step)
  bench     v9lib.bench.execute_bench (X12; public-benchmark evaluation of one retained snapshot)

OLMo/Qwen cohorts reuse v8/experiments/cluster_child.py unchanged (model loading, LoRA, verifier
runtime, attention, protected rollouts); only the task dispatch is extended to `measure`.
Llama cohorts enter the original Llama adapter contexts (see v9lib/llama.py). A fake locator
(local CPU checks only, V9_ALLOW_FAKE=1) builds the code repository's ToyBackend or a tiny OLMo-3.
Exit codes follow the V8 child: 75 = busy elsewhere or interrupted (resumable), written to $V8_EXIT_FILE.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from v9lib import env  # noqa: E402

env.activate()


def dispatch_factory(original_train, beta=None):
    from v9lib.measure import execute_measure

    def dispatch(task, ctx):
        if task.get("kind") == "measure":
            return execute_measure(task, ctx, beta=beta)
        if task.get("kind") == "bench":
            from v9lib.bench import execute_bench
            return execute_bench(task, ctx)
        return original_train(task, ctx)
    return dispatch


def v8_entry_fixes(root, task_id, *, repairs=True):
    """The cluster fixes V8 runs every task with (scripts/v8_task_entry.py), for V9 tasks too:

    - the 30-min periodic faulthandler dump that cluster_child.install_diagnostics arms is cancelled: its watchdog
      thread walks every thread's frames without the GIL and SIGSEGVs a rank on Python 3.12 (every V8 native crash
      11-08..11-16 ended 30/60/90 min after start). SIGUSR1 and fatal-signal dumps stay;
    - the NCCL wait limit (srgc_rebuttal pins 1800 s) is raised to V8_COLLECTIVE_TIMEOUT_SECONDS (default 6 h), so a
      rank that waits for a slower rank's generation does not abort;
    - a rank that dies of a native signal writes its stack to $V9_ROOT/.state/progress/<task>/native-fault-rank-N-*.log
      (the worker puts the crash place into the failure reason);
    - V8's runtime repairs (checkpoint, memory and MBPP checker launch fixes; no scoring change), OLMo/Qwen only.
    V8_PERIODIC_STACK_DUMP=1 keeps the old periodic dump. Returns the names of the fixes applied."""
    import faulthandler
    sys.path.insert(0, str(env.REPO / "scripts"))   # this checkout's entry (the V8 launcher uses its own sibling)
    import v8_task_entry as entry
    os.environ.setdefault("SRGC_PROGRESS_DIR", str(Path(root) / ".state" / "progress" / task_id))
    entry.install_native_fault_handler()
    done = ["native-fault-log"]
    if entry.disable_periodic_stack_dump():
        done.append("periodic-stack-dump-off")
    if entry.extend_collective_timeout() is not None:
        done.append("collective-timeout")
    import cluster_child
    original = cluster_child.install_diagnostics
    if not getattr(original, "_v9_fixed", False):
        def install_diagnostics(*a, **k):
            handle = original(*a, **k)
            if os.environ.get("V8_PERIODIC_STACK_DUMP") != "1":
                faulthandler.cancel_dump_traceback_later()
            entry.install_native_fault_handler()   # cluster_child rebinds the fatal-signal file; restore ours
            return handle
        install_diagnostics._v9_fixed = True
        cluster_child.install_diagnostics = install_diagnostics
    if repairs:
        entry.install_runtime_repairs()
        done.append("runtime-repairs")
    return done


PLAN_OVERRIDE_KEYS = ("training_prompts", "scoring_prompts_per_set")


def install_plan_overrides(overrides):
    """X15: the task changes only the prompts per update and the candidates per draw of the code-repo plan.
    cluster_child imports srgc_rebuttal.plan.load_plan after activating the code-repo runtime, so the wrapper is
    installed on that activation; every other plan field is unchanged."""
    bad = set(overrides) - set(PLAN_OVERRIDE_KEYS)
    if bad:
        raise SystemExit(f"plan_overrides: unsupported keys {sorted(bad)}")
    from switching import coderepo

    def patch():
        import srgc_rebuttal.plan as plan_mod
        if getattr(plan_mod.load_plan, "_v9_overrides", False):
            return
        load = plan_mod.load_plan

        def load_plan(*pa, **pk):
            plan = dict(load(*pa, **pk))
            plan.update(overrides)
            return plan
        load_plan._v9_overrides = True
        plan_mod.load_plan = load_plan

    def wrap(fn):
        def run(*a, **k):
            out = fn(*a, **k)
            patch()  # after activation the (possibly saved) srgc_rebuttal is the one cluster_child imports
            return out
        return run
    coderepo.activate = wrap(coderepo.activate)
    coderepo.activate_source_runtime = wrap(coderepo.activate_source_runtime)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--task-file", required=True, type=Path)
    ap.add_argument("--root", required=True, help="V9 root ($WORK/v9)")
    args = ap.parse_args()
    task = json.loads(args.task_file.read_text())["tasks"][0]
    task_id = task.get("task_id") or task["run_id"]
    loc = json.loads((Path(args.root) / args.cohort / "locator.json").read_text())
    seed_loc = loc["seeds"][str(task["training_seed"])]
    if seed_loc.get("fake"):
        if os.environ.get("V9_ALLOW_FAKE") != "1":
            raise SystemExit("fake locator on a real run root; refusing")
        from fake.fake_child import main as fake_main
        return fake_main(args, task, seed_loc)
    from switching import runner
    fixes = v8_entry_fixes(args.root, task_id, repairs=not args.cohort.startswith("llama_"))
    if os.environ.get("RANK", "0") == "0":
        print(f"V9 child: V8 entry fixes {', '.join(fixes)}", flush=True)
    if args.cohort.startswith("llama_"):
        from v9lib.llama_child import main as llama_main
        runner.execute_train = dispatch_factory(runner.execute_train)
        return llama_main(args, task_id)
    import cluster_child  # v8/experiments/cluster_child.py (unchanged)
    runner.execute_train = dispatch_factory(runner.execute_train)
    if task.get("plan_overrides"):
        install_plan_overrides(task["plan_overrides"])
    sys.argv = [sys.argv[0], "--cohort", args.cohort, "--matrix", str(args.task_file), "--task", task_id,
                "--v8-root", str(args.root)]
    return cluster_child.main()


if __name__ == "__main__":
    main()
