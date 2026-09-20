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
