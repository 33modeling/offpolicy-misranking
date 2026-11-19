#!/usr/bin/env python3
"""torchrun child for one V9 task (one process per GPU).

    torchrun --standalone --nproc_per_node=4 v9/child.py --cohort olmo_math --task-file <task.json> --root $V9_ROOT

Task kinds:
  train     V8 execute_train (fork from a retained V8/V9 state, resumable from latest.pt)
  evaluate  V8 execute_eval  (immutable final-split evaluation of one retained snapshot)
  measure   v9lib.measure.execute_measure (X5; scoring and log-probs only, no optimizer step)

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
        return original_train(task, ctx)
    return dispatch


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
    if args.cohort.startswith("llama_"):
        from v9lib.llama_child import main as llama_main
        runner.execute_train = dispatch_factory(runner.execute_train)
        return llama_main(args, task_id)
    import cluster_child  # v8/experiments/cluster_child.py (unchanged)
    runner.execute_train = dispatch_factory(runner.execute_train)
    sys.argv = [sys.argv[0], "--cohort", args.cohort, "--matrix", str(args.task_file), "--task", task_id,
                "--v8-root", str(args.root)]
    return cluster_child.main()


if __name__ == "__main__":
    main()
