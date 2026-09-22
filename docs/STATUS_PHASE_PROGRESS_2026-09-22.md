# Separate Phase Progress in MBPP, Pair and RLOO

The shared node table previously selected either rollout generation or gradient
progress from the current phase logs. Completed earlier phases disappeared, and
a single 100% value could look like completion of the entire task.

The node `Progress` cell now retains separate values, for example:

```text
Progress                     Remarks
100.0% / 100.0% / 78.0%       fresh-r-validation: 100.0%; response generation: 100.0%; gradient candidate: 78.0%
```

- Each percentage has its own phase label and counts in Remarks, in the same order.
- Earlier phases use their latest start/finish journal records, not the existence
  of a log or a previous successful attempt superseded by a new start.
- The current phase retains both response-generation and gradient counters.
  Four shard counters are aggregated separately for each stage. Missing, stale,
  or invalid counters stay `?`; a finished shard cannot stand for missing peers.
- A current phase's old finish record cannot overwrite its live counters.
- Time-allocation use is identified separately and does not imply result completion.
- An incomplete processed counter cannot round up to 100%.
- This is a read-only display change in the shared MBPP/Pair/RLOO renderer.
  It does not restart training, change budgets, or alter completion validation.

Refresh with the existing commands; no GPU restart is needed:

```bash
git pull --ff-only
bash scripts/run_mbpp_experiments.sh status
bash scripts/run_selector_pair.sh status
bash scripts/run_rloo.sh status
```

Regression cases cover simultaneous stages, uneven shards, retries, old logs,
incomplete counters, path confinement, and narrow/wide terminal layouts.
Verification: 277 tests passed; one existing RLOO nested-curve case was skipped
because that workflow publishes a single arm meter, not nested curve meters.
