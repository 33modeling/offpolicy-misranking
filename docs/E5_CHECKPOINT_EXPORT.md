# E5 Checkpoint Export

Run on the machine containing the original experiment directories:

```sh
python3 scripts/export_e5_checkpoint_curves.py
```

The default root is `$OM_WORK/runs/e5-reduced` (fallback:
`/group-volume/minsoo3.kim/offpolicy-misranking/runs/e5-reduced`). Folder names
such as `math400-d0/s0`, `math500-d100/s1`, and `math500-d400/s2` are discovered
automatically. A different parent can be supplied with `--root`:

```sh
python3 scripts/export_e5_checkpoint_curves.py --root /path/to/runs --out /path/to/exports/e5-checkpoints.json
```

The script prints the paths of a portable JSON report and a plotting CSV.
Transfer both files back to the manuscript workspace; model tensors are not
included. The JSON includes checkpoint inventory, per-seed observations,
evaluation sample counts when available, source hashes, and missing-evaluation
paths. Absolute checkpoint steps and additional updates are separate columns.

Supported evidence:

- `downstream_results.csv` plus `experiment.json` or `policy_train.json` for
  the starting and final steps; the final step is never assumed to be 100.
- `curve.json` with a `points` mapping and explicit `updates`/`reward` values.
- `eval-*.json`, `eval.json`, `evaluation.json`, or `summary.json` containing
  `mean_reward`; checkpoint step comes from the adapter or parent directory.
- Four-shard checkpoint evaluations (`shard-N.jsonl`, `.contract.json`, and
  `.done.json`), checked for completion, hashes, prompt coverage, and sample count.
- `checkpoint_state.json` and adapter-only `checkpoint-N`, `step-N`, or
  `policy_step_N` directories for inventory, even without evaluation results.

Summary records retain their source provenance; they are not independently
re-evaluated. Different evaluation `k` values are kept separate. Inspect issues
and evaluation identities before combining seeds. Duplicate conflicting values
invalidate an arm rather than selecting one result. Unrecognized log schemas
are not guessed, and training reward/reliability logs are not held-out reward.

No GPU is used and no experiment files, locks, checkpoints, or running jobs are
modified. Existing exports cannot be overwritten. Outputs must be outside the
source run tree. A checkpoint without reward is listed under `needs_evaluation`;
it is not assigned zero, interpolated, or treated as a measured point. If only
model checkpoints remain, evaluation is still required before plotting them;
this export command does not launch that evaluation or repeat training.
