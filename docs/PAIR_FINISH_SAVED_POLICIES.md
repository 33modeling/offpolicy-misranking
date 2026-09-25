# Finish the two saved Pair policies without changing the original experiment

The copied `hash.txt` points at missing checkpoint directories in old recovery
plans: On-policy s1/t50 step 355 and Random s4/t100 step 155. The separate
optimizer search records newer final-policy locations at steps 457 and 256.
Those records motivate checking the current policies; they do not themselves
certify the policy bytes or prove that evaluation completed.

## Normal launcher

The normal `bash scripts/run_selector_pair.sh` command now resumes these two
saved final evaluations automatically once the other 40 sealed branch results
are present. No separate finisher command is required. This path ignores the
obsolete recovery runner pin and missing checkpoint selection, but still
validates the current policy, optimizer, source data and lineage. The deployed
runtime also dispatches this path through its reviewed worker helper, keeping
the original science launcher and trainer hashes unchanged.

`bash scripts/run_selector_pair.sh status` includes these evaluations. A
completed additional evaluation is explicitly distinguished from an eligible
matched-budget paired result; original canonical results are never fabricated.

## Standalone alternative

`scripts/finish_selector_pair_two.sh` is a standalone launcher that embeds
`selector_pair_finish_saved.py`. Download only this file, then run it from the
existing experiment checkout on a node with four allocated GPUs:

```bash
bash /path/to/finish_selector_pair_two.sh
```

One node handles both branches sequentially. The same command on two nodes
uses separate output leases to avoid duplicate work. To assign explicitly:

```bash
bash /path/to/finish_selector_pair_two.sh run --seed 1
bash /path/to/finish_selector_pair_two.sh run --seed 4
```

`plan` performs CPU validation only. `results` prints the new sealed results.
The launcher uses the existing `PAIR_PYTHON`/`VENV_DIR`/`OM_WORK` environment.
`PAIR_ROOT` selects the original Pair root. `PAIR_FINAL_EVAL_ROOT` defaults to
`$OM_WORK/runs/selector-pair-final-eval-v1`, outside the source experiment.
The Python run's output and errors are saved in a new `/tmp/pair-final-eval-*.txt`.

## Exact behavior

- Only On-policy s1/t50 `selection_reduced` and s4/t100 `random_full` are eligible.
- Check the saved final policy using the existing full policy-lineage and
  artifact-hash validator, the original contract, subset, decision and evaluation
  prompts. Missing/corrupt final policies stop; there is no parent restart.
- Evaluate the current final policy and recorded checkpoints chosen by the
  frozen curve schedule. No interpolation, training, rescoring, or replacement
  optimizer is used. The old missing checkpoint and old recovery runner hash
  are not used as the new evaluation plan.
- Reuse sealed evaluation shards only when their policy, protocol, sampling
  and prompt coverage match. Copy them into the new output; source files remain
  untouched. Incomplete new evaluations resume only their missing shards.
- New plans, results, seals, progress, locks and reporting-cost ledgers live
  only in the separate output root. Source locks are opened read-only and held
  shared while reading; active writers are never killed or bypassed. Workers
  inherit leases so a lost controller cannot cause duplicate GPU work.
- Preserve original plans/results and their hash mismatches as evidence.
  Preserve known original cost summaries separately. New GPU time is charged
  to reporting. This does not certify complete historical cost accounting.

Each finished branch has `seed-1/result.json` or `seed-4/result.json` plus a seal
under the new output root. `evaluation_complete=true` means this separate
evaluation finished. `canonical_complete=false` remains explicit: an over-budget
continuation is not silently converted into budget-compliant Pair completion or
an H measurement. Original `result.json` files are never created or overwritten.

## Validation and limits

CPU tests cover a missing old checkpoint with a valid later policy, preserved
source bytes and modification times, corrupted weights/optimizer/stats,
existing-shard reuse, interruption/resume, changed inputs, busy source/output
leases, output-path isolation, and standalone launch/status/log behavior.
These tests use small synthetic policy files and mocked generation. Real GPU
execution, remote checkpoint bytes and numerical rewards require running the
script on the experiment node. No node was accessed or training launched locally.
