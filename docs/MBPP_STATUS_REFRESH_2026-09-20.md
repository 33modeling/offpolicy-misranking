# MBPP Status Refresh

The user reported that FULL STATUS, RUN and DONE appeared not to update.
No current server screenshot was available during this change. Local
reproductions identified three independently testable display issues:

1. A metered phase can publish its atomic finish receipt before the final
   `progress.json` update. If interrupted between those writes, status continued
   to show RUN until the last heartbeat aged out, including after a valid result
   and curve had appeared. The read-only viewer now honors matching finished
   receipts. Event ID, phase, host, ledger and GPU allocation must all match;
   unrelated receipts cannot hide live work. Failure receipts do not imply DONE.
2. The matrix and totals included active nested evaluations, but the `--all`
   task listing did not use those descendants when rendering its branch state.
   All three now use the same activity context.
3. Posthoc budget-recovery evaluations were mentioned only in individual task
   remarks. They now also have an explicit count in the top summary, per-suite
   summary and FULL STATUS header. They remain outside canonical equal-budget
   DONE and inside the original remaining/48 denominator.

No training, experiment budget, result payload, ledger, or worker is modified by
status refresh. The normal `status` command remains a single snapshot;
`status --watch` refreshes and reloads the read-only viewer each frame. Tests
cover sequential snapshots and the actual shell watch with a RUN-to-DONE
publication between two frames, without restarting a worker.

Initial reproduction: eight failures and six passing negative-control cases.
The first regression run after the fix passed 158 status/publication/watch tests.
The follow-up watch/refresh/compact/Pair-display/idle-node run passed 93 tests,
including the real two-frame publication test. The two suites overlap.
Remote observation is still necessary to identify which issue affected the
user's specific screen; this record does not assert a live-cluster diagnosis.
