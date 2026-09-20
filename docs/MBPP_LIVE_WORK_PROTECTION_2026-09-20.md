# Live work protection: MBPP and shared launcher

## Confirmed defects

- Plain MBPP `run` deliberately converted a repeated invocation into `restart`,
  including when the code was unchanged. This interrupted an owned training or
  curve phase and could repeat work since the last checkpoint. That behavior
  was incorrect for a normal launch and has been removed.
- The generic node controller swept prepared roots and signalled process groups
  based on matching command, root and node markers. Those markers did not prove
  that the new controller owned the existing job. Both automatic process sweeps
  have been removed, including on a busy pass or after an explicit controller stop.
- Direct controller lifecycle commands accepted extra arguments, so a malformed
  restart/stop could still signal a controller. These now fail before mutation.
- A published training result with a live nested curve still advertised itself
  as retryable to the queue, despite appearing RUN in the dashboard.

## Current behavior

- Repeated MBPP run leaves the controller alive, even with installed code changes
  or missing legacy runtime metadata. It does not pull shared code in that path.
- Only explicit restart/stop requests terminate the identified controller.
  Proven token-owned children of a dead MBPP controller can still be reclaimed
  by the existing ownership guard. Unrelated jobs are not swept.
- Pair and RLOO reject invalid run options before admission; their launchers
  force non-destructive node locking. They were checked with the same malformed
  invocation tests. Neither needs a learner or frozen-contract change here.
- Live branch/nested curve activity suppresses retryability. An existing held
  MBPP task lease also prevents advertising READY before progress publication,
  including while the branch evaluates the shared parent curve. Released leases
  and interrupted curve metadata allow normal reporting resumption.
- Lease-checked cost recovery remains; wall-clock age alone is not ownership.
  Existing checkpoints, cost ledgers, completion seals and budgets are preserved.
- Automatic stall watchdogs now require the current controller's unique token
  as well as the cost-event marker. A duplicate/busy controller cannot stop a
  peer or publish a fault strike for that peer's stalled-looking metadata.
  Scoped teardown signals identity-checked PIDs and their children, not an entire
  process group that may contain unrelated work. The existing stall threshold
  and train-only phase defaults are unchanged.

## Log identification

The shared foreground worker launcher appends `[mbpp]`, `[pair]`, or `[rloo]`
to each worker stdout/stderr line, including `[gate] curve curve` progress.
The actual Pair/RLOO worker takes precedence over inherited MBPP dataset flags.
Existing leading markers stay intact for status parsers. Untagged experiments
are unchanged. The formatter is drained before returning, does not hold node
locks, and never replaces the real worker PID used for signal handling.

## Scope and remaining evidence

These are launcher and observation fixes, not new GPU measurements. Local tests
use real CPU subprocesses and flock leases; they cannot establish the current
state of the remote GPU cluster.

The supplied MBPP WHY still documents exhausted allocations, including a DEV
branch needed for the gate, and the storage report documents missing checkpoint
artifacts. This change does not fabricate canonical results, relax budgets,
refund costs, or turn rc=80 into success. Separately saved posthoc evaluations
remain distinct from canonical gate inputs. Those experiment-level blockers
require their actual saved artifacts or an explicitly revised experiment design.
