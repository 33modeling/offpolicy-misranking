# Current Experiment Allocations

Recorded: 2026-09-25 04:21 KST (2026-09-24 19:21 UTC).
Source: the user's explicit report in the current conversation, not a remote
process-health check. This is a handoff snapshot, not live status.

| Experiment | User-reported active count |
| --- | ---: |
| run pair | 2 |
| SR-GC switch | 8 |
| RLOO | 1 |
| Total | 11 |

Use this allocation when discussing additional nodes, timing, or restarts.
Do not conflate the two Pair workers with the eight Switch workers. Do not
restart healthy jobs merely to update status/results scripts. Preserve all
checkpoints, including optimizer state. Confirm newer logs or operator reports
before treating a job as failed, completed, or available for reassignment.

Switch inspection commands (no GPU work; existing jobs need not restart):

```bash
bash scripts/run_selector_pair_switch_rewards.sh status
bash scripts/run_selector_pair_switch_rewards.sh results
```

Exports: `~/selector-pair-switch-status.txt` and
`~/selector-pair-switch-results.txt`. Phase wall time and GPU-hours are separate;
their sums across workers are not the concurrent job's elapsed completion time.

## Pair Two-Final Resume (2026-09-25)

The remaining branches identified by the user are On-policy `s1/t50`
(`selection_reduced`) and Random `s4/t100` (`random_full`). The copied
`~/hash.txt` diagnosis pins missing checkpoints 355 and 155 and reports a
recovery-runner hash mismatch; newer saved final policies were inventoried at
457 and 256. These are copied observations, not live GPU status.

`bash scripts/run_selector_pair.sh` now automatically resumes their final and
curve evaluations when the other 40 sealed results are present. It validates
the current final policy, optimizer, source inputs and lineage, without
rebinding the obsolete recovery plan or training again. A busy source/output
lease is respected. Failed evaluations resume only missing shards.

The original results, checkpoint files and cost ledgers are unchanged. New
outputs use `runs/selector-pair-final-eval-v1/seed-{1,4}` by default
(`PAIR_FINAL_EVAL_ROOT` overrides it). The existing `status` command shows
running and completed evaluations, explicitly excluding these over-budget
results from matched-budget paired comparisons. Completion of GPU evaluation
must be confirmed on the allocated node; local fixture tests are not evidence
that a remote job restarted. No web publication is part of this change.
