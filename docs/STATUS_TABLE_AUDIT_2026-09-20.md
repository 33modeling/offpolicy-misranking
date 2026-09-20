# Status table and node identity audit

Scope: MBPP, Selector Pair, RLOO. Operational scripts only; no scientific
source hashes, experiment budgets, saved results, or controller ownership are
rewritten by this change. Status remains read-only.

## Confirmed defects and fixes

- Summary and full matrix interpreted duplicate/unverified registered slots
  differently. Both now use the same registered branch index.
- Shared admission, decision and evaluation work could disappear from CURRENT.
  Show it separately from registered training branches; do not enlarge the
  experiment denominator (MBPP 48 per condition, Pair 42, RLOO 18).
- Pair status depended on a time/entry-limited queue scan. Read the registered
  state-point meters and operational paths directly as well.
- Clock-skew protection covered only curve meters. A held cost lease with a
  matching running meter now preserves other metered phases too. This proves
  ownership, not forward progress or a fresh wall-clock heartbeat.
- CURRENT collapsed different roots with equal labels, and sibling operations
  with the same hostname. Only duplicate parent/child observations are folded;
  sibling paths survive even when both hostname and PID are identical.
- Direct Pair/RLOO launches omitted node identity initialization. Their shared
  worker wrapper now initializes the existing hostname/GPU/container suffix
  before queue receipts or meters are written. Explicit EXPERIMENTS_NODE_ID is
  preserved. MBPP already initializes this identity in its controller.
- Stopped Pair workers no longer appear as allocated WAIT nodes. A terminal
  RLOO receipt cannot override a live meter on the same recorded node.
- Empty MBPP curves and invalid prefix metadata cannot count as DONE. RLOO
  evaluation adapter bindings must match the policy manifest. Malformed cost
  rows produce visible warnings without hiding unrelated work.

## Legacy identity limits

Status preserves the complete recorded node ID, including its suffix. Two
different suffixes produce two node rows. For old records with exactly the
same hostname and no suffix, separate work paths remain visible, but physical
node counts cannot be recovered reliably from hostname/PID alone. The display
warns when several active operations share such an ambiguous ID. Records
already overwritten on disk cannot be reconstructed by status.

Identity initialization applies to newly launched workers; pulling code does
not change the environment of running workers. No restart or signal is issued
by this fix. Full model/input validation remains the check/report command's
responsibility; status does not hash large model tensors.

## Verification

CPU regression coverage includes same-host/different-suffix workers, explicit
identity overrides, identical-host/PID sibling operations, clock offsets in
both directions, nested curves, custom state points, shared admission,
conflicting table rows, corrupted cost records, and invalid evaluation seals.
GPU/server behavior is not verified locally.

A broader status sweep reported 847 passed and 11 failed. All 11 failures are
in unrelated Qwen display assertions in test_status_reward_audit.py; the same
11 failures were reproduced in a clean worktree at pre-change d61f690 (that
file: 28 passed, 11 failed). They are not fixed or hidden by this change.

## Follow-up: issues missed by 657fdd5

The initial fix was insufficient. Pair's queue already writes a UUID in each
queue-workers receipt, but status discarded it and retained only the newest
receipt per hostname. Two different live workers could collapse into one;
a newer completed worker could hide an older running worker with the same
hostname. Preserve (host, worker ID) through the snapshot, task association,
CURRENT deduplication, node table, and idle table. Match meters to a queue
worker using the state point, not hostname alone; PID is only an additional
disambiguator, never a global identity. Show the complete worker code.

The renderer also prioritized a saved DONE record over a live meter. Runtime
display now prioritizes RUN, while saved_done preserves publication evidence.
Pair must carry live flags even for a published branch. Parent publication
details remain visible when a more specific nested meter represents the row.

The MBPP controller used only development_done=18 and test_done=30 to skip a
root or release its node. Completion now also requires no RUNNING task, fresh
running heartbeat, held meter ownership, or held branch task lease. This is
not permission to reassign a leased task; existing queue locks remain intact.

These fixes do not resolve every MBPP non-start condition. The supplied
mbpp_shy_new.txt contains a development branch (s2/t50/selection_reduced)
with 28387.865 GPU-s consumed against 28376.947 GPU-s allocated. A genuine
rc=80 review/dependency stop is not DONE and is not bypassed. No cost refund,
budget extension, checkpoint reset, or posthoc-to-canonical relabel is made.

New regression cases exercise equal host/PID/task but different worker UUIDs,
RUN plus newer WAIT/DONE receipts, per-state meter/worker matching, published
results with active meters for all three experiments, and the actual shell
completion predicate with each kind of active evidence.

Follow-up verification: the broad MBPP/status/controller run had 794 passes,
one expected-case mismatch in the changed idle worker label, and one skipped
RLOO nested-curve case (RLOO has no such meter layout). After correcting the
label regression and updating the worker-count heading expectation, the
focused suite passed 135 tests with that one skip; the switch/RLOO/node/watch
suite passed another 149 tests. Bash syntax and git diff checks passed.
