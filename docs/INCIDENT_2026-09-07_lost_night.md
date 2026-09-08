# 2026-09-07 night: six nodes, twenty-four H100s, one point of progress

A record of what went wrong, what caused it, and what was changed, so the same
class of failure is visible within minutes next time instead of a night later.

## What the operator saw

- 2026-09-07 16:00Z status: 11/40 points, six workers alive, PROGRESS TRAINING.
- 2026-09-07 23:08Z status: 12/40 points, five workers alive, PROGRESS
  `NOT TRAINING for 5h08m`. Same families, same stage names, same worker table.
- Over the night the five live workers cycled `oracle-gradients -> scoring` on
  their d0 point roughly once an hour and wrote no DONE, no GRPO step and no
  rollout byte.

## Root causes, in the order they bite

1. **A finished point could not be re-entered (mine, 2026-09-07).**
   The h100 profile raised the generation batch (math500 8 -> 32, mbpp 8 -> 16).
   `repair_run_config.py` deliberately skips points that already have `DONE`, so
   a finished point kept `gen_batch=8`. When the completion check rejected such a
   point, `run_point` re-entered it and the pinned `run_point.sh` refused with
   `[config-abort] existing artifacts use a different run config: ['gen_batch']`.
   math500/s1 and math500/s2 died this way; rlvr-4 burned four hours on it and
   then stopped sending heartbeats.

2. **Nothing said why a point was rejected or why an attempt failed.**
   `run_complete` sent every check's output to `/dev/null` and returned 1, and
   the attempt loop printed only `try N/3`. So a family that finished a point,
   failed the completion check and re-ran it looked exactly like a family that
   was making progress. This is why the night is unexplained: the evidence was
   never written down.

3. **A stale CUDA line made every later failure look like a CUDA fault.**
   CUDA runtime faults are exempt from the failure-loop guard (they are transient
   and are retried). `family_last_error` took the newest error line from *any*
   log in the family, so one old `unspecified launch failure` in a stage log
   classified every subsequent, unrelated failure as `runtime`. The guard never
   counted, and the worker retried the same family for ever.

4. **A failed family held the worker.**
   The 2026-09-07 rule "a family that fails is retried by the same worker" was
   meant to stop families being abandoned. Combined with (3) it meant a broken
   family owned a node until morning.

5. **A LOOPING family kept every worker in an infinite wait.**
   When the guard did trip, the family stayed in `remaining`, the queue loop
   only exits when `remaining` is zero, and the worker printed
   `[queue] waiting for N families ...` every 60s for ever while holding four
   GPUs. Nothing in the status table distinguished that from useful waiting.

## Changes

| Commit | Change |
| --- | --- |
| b5a6536 | `run_complete` records `COMPLETE_REASON`; `run_point` prints `[done-but-incomplete] <family>/<point>: <reason>` and repairs that one point's batch fields before re-entry (`repair_run_config --include-done`); the startup sweep clears a loop marker that recorded a re-entry `config-abort`. |
| 3ca590f | Every failed attempt prints `[point-failed] <family>/<point> try N/M rc=R: <last error line>` to the worker log and `<run>/logs/supervisor.log`; every claim prints `[family-plan] <family>: d0=complete d25=DONE-but-rejected(<reason>) ...`; `scripts/why.sh` collects the evidence into one small text file. |
| this commit | `family_last_error` reads the failing attempt's own `[point-failed]` line instead of any old error line; a failure moves the worker to the next family at once and the failed one returns after a growing cooldown; a worker whose only remaining families are LOOPING says so and exits non-zero instead of waiting for ever; `why.sh` leads with a one-line-per-family diagnosis. |

## Rules taken from this

- A check that rejects work must record the reason where the operator can read
  it. `>/dev/null 2>&1` on a decision path is a bug.
- Evidence about "why did this attempt fail" must come from that attempt, never
  from a directory-wide grep that can pick up a different failure.
- No policy may let one broken unit hold a node: on failure, move on and come
  back later.
- Any state the operator must clear by hand must stop the worker with an
  instruction, not become a silent wait.
