# Switch / MoPPS runtime audit, 2026-09-15

## Scope and evidence

The operator reports two experiments running across three nodes. This statement
is the current deployment context; local tools cannot observe the secure cluster.
No cluster stop, restart, recovery, SSH connection or credential request was made.

The supplied exports end at 10:05-10:06 UTC (19:05-19:06 KST):
`switch-why-20260915T100511Z-9ZQFr8.txt` and
`mopps-why-20260915T100615Z-Qn04Kw.txt`. Their hashes and per-event findings are
in [the original report analysis](SELECTION_SWITCH_REPORT_ANALYSIS_2026-09-15.md).
They establish past failures, not the state of the three current workers.

Changes already present during this audit were preserved:

- `6e46394`: bounded NCCL fabric-setting probe ladder, with only passing
  overrides inherited by training. This supersedes the earlier immediate
  CUDA-802 refusal. No local test proves this repairs the H100 fabric problem.
- `7e454cd`: operator-selected stale-cost estimate recovery. This supersedes
  the earlier receipt-only automatic recovery. This audit does not undo that
  choice or change its arithmetic.
- `c881a85`: keep run-mode allocations between queue passes. Holding a node
  does not mean a training task is running or that a failed dependency cleared.

## Findings and repairs

### Between-pass sleep inherited locks and lacked worker stop handling

Both launchers used a direct `sleep` for 600-3600 seconds after their worker
returned. The worker traps had already been removed. Signaling only the parent
could leave the sleep process holding inherited node-lock descriptors until it
ended. Whole-session stop and later orphan-helper cleanup are not substitutes
for cleaning up the parent-only signal path.

The wait now uses the existing shared worker lifecycle, closes descriptors 7/8
in sleep, reaps sleep on INT/TERM and emits a holding heartbeat every 15 seconds.
The node stays allocated as requested; failure costs, positive wait policy,
retry eligibility and task locks are unchanged. Switch status labels this
heartbeat `HOLD` / `between passes`, rather than counting it as active training
or losing the node from the display after 60 seconds of silence.

Regression checks signal only the shell PID, not the whole process group,
verify that sleep has no node-lock descriptor, verify its exit and reacquire
the lock. Both real shell entrypoints also run two passes across a live-checkout
revision change and release the node on fixture completion. These shell tests
stub GPU admission and the expensive learner; they do not claim cluster training.

### Recorded errors were not attributed clearly enough

The MoPPS export contains successful node-6 admission records and a node-6
launcher tail reprinting node-4's older trainer failure. Switch also retains
older failure records while a new owner is progressing. A log tail alone
therefore does not prove that the same exception just recurred.

`errors` now shows saved failure UTC/host separately from latest progress
UTC/host/PID/phase, plus the latest available admission record per host and
its pinned runtime commit. It explicitly distinguishes a passing probe from
successful training and a saved failure from a new exception. Existing errors
remain visible; no failure, result or cost file is deleted or rewritten.

### CUDA 802 remains an unverified cluster issue

The exports establish failed four-rank admission on nodes 4/5 and passing
four-rank admission on node 6. The later fabric-setting ladder is not included
in those exports. Its finite retry order, environment preservation and cost
closure are covered locally; its success on the affected nodes is unknown.
Do not classify every new error as 802 without the new rank traceback.

### Cost recovery estimates remain estimates

The four incomplete deployment events in the earlier exports blocked
development branches and therefore Gate fitting. The newer operator-selected
stale-recovery command can close these with an explicitly tagged estimate.
Adding 60 seconds to the last observed heartbeat/log write does **not** prove
an upper bound on the true termination time. Its provenance must remain in
the ledger; do not describe estimated recovery as an exact measured cost or
claim that it is mathematically guaranteed never to undercount. This audit
does not run recovery on any live or exported experiment.

### No defensible finish time from these exports

At 19:06 KST the source experiment had prefixes 12/15, development 6/18 and
held-out 0/30. MoPPS had eight failures and four missing-prefix blockers.
Those are historical counts, not a contradiction of the operator's current
three-node run. Each continuation has a 29,040 GPU-second allocation, but
evaluation has a separate 14,400-second timeout and prefix work/dependencies
remain. Dividing a branch count by three is not an ETA. Open costs, failed
dependencies and a between-pass hold make a finite finish-time promise unsafe.

## Verification

- Broad CPU/process suite before the final diagnostic/hold repair: 403 passed,
  16 skipped (15 opt-in CUDA cases and one optional plotting case), two PEFT
  fixture warnings. `/tmp/three-node-audit-20260915.xml`.
- CUDA-selected suite: 30 passed, comprising 15 actual CUDA tests and 15 CPU
  parameter cases. Actual GPU checks cover NCCL/DDP admission, allocation
  rejection, all four 537-token cache-guard variants and eight worker cleanup /
  restart cases. `/tmp/three-node-audit-cuda-20260915.xml`.
- Hardware: one RTX 3050 6GB. No four-H100 execution or production throughput
  measurement is claimed. GPU compute-process list was empty after tests.
- Final focused diagnostic/hold results are recorded in the bug-fix log.
- Scientific source-map fingerprints are unchanged: Switch
  `b7803071821dd7aa68035370e37c77fdbaecec13d9e96ea6574d91b10758f5a5`;
  MoPPS `b49a7417bf1f00c8c164c6bd9d0aa480dd4d7438e2de4d6f8d505d53a3e2dab2`.

## Current operations

Do not stop or restart the three workers merely for this audit. Running
controllers keep their pinned code; lifecycle changes take effect on their
next launch, not during a current task. Updated read-only diagnostics can be
used without replacing an active worker.

For a current snapshot, from the repository with the existing root environment:

```bash
bash scripts/run_selection_switch.sh why
bash scripts/run_mopps_comparison.sh why
```

These are read-only experiment inspections that write report files, not
training/retry commands. Their printed paths identify the two files needed
to determine whether an error is new and which current worker produced it.
