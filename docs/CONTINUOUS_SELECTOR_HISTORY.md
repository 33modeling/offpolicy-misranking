# Continuous Selector Histories

The comparison follows on-policy and cached SR from the same step-zero start
through their full saved training trajectories. Step 100 is not an end limit.
Steps 25, 50, 100 (or other configured cutoffs) are observation times on those
trajectories, not separate starting checkpoints. Nonzero-start experiments
are never concatenated into a fictitious continuous trajectory.

```bash
git pull --ff-only origin master
bash scripts/export_continuous_selector_history.sh
```

Defaults: `$OM_WORK/runs/e5-reduced`, where `OM_WORK` falls back to
`/group-volume/minsoo3.kim/offpolicy-misranking`. The script automatically finds
`math400-d0/sN`, `math500-d0/sN`, and the other original E5-style step-zero runs.
Use `--root PATH` for a different storage root. `--decision-steps 25 50 100 200`
changes analysis cutoffs, not the training duration. All saved steps beyond the
largest requested cutoff are still exported.

Results are written under a new `exports/continuous-history-TIMESTAMP` directory:

- `continuous-history.json`: existing evaluation points, full per-step logs,
  model/optimizer checkpoint locations, and records available through each cutoff.
- `evaluation-curves.csv`: measured held-out rewards, kept separate from training.
- `training-log.csv`: existing rewards, gradient norms, loss and update timers.

Return `continuous-history.json` for analysis. Saved model tensors stay on the
server. This standard-library command does not load tensors, start GPU work,
run evaluations, retrain models, fit regression, or modify running Pair jobs.
All input files and existing exports are preserved.

## Scope

This command corrects data extraction and comparison scope, not the running
Pair experiment's frozen protocol. It reuses existing step-zero runs; it neither
extends them nor claims their original end step was different. A missing saved
checkpoint evaluation remains missing, rather than being replaced by training
reward or interpolation. A missing timer is not zero.

The `as_of` records include only steps at or before the specified cutoff.
This permits a time-truncated replay, but does not certify when historical
evaluation files became available. Metadata in the full-history section is
for provenance and must not be supplied as future-informed prediction features.

The two methods have different policies after step zero. Comparing their later
slopes is a forecast of these two trajectories, not proof of the counterfactual
effect of changing selectors on one identical current policy. Total cost-to-target
also needs selection/setup costs; per-update timers alone do not provide it.
The exporter deliberately does not manufacture an H value.
