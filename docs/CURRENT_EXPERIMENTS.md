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
