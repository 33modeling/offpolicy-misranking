# Execution And ETA Corrections

The subsequent selected-prefix switch failures and repairs have their own
[bug-fix log](SELECTION_SWITCH_BUGFIX_LOG.md), including unresolved mid-run
interruptions reported on 2026-09-15. Do not conflate those reports with the
earlier net-gain incidents recorded below.

## User-Reported State

The user reports that the nine gate experiments were started concurrently,
with six completed and the three previously failing selection controls now
running. This is not nine sequential two-hour jobs. Do not restart all nine
or present this report as an independently observed live cluster status.

## What Was Wrong

1. A rough source-training duration was presented as an approximately two-hour
   completion expectation without verifying the complete workload or recovery
   throughput. That expectation was not established and is withdrawn.
2. The default gate delegated scoring to finite differences at step 0.1 without
   exposing its existing exact derivative option. Actual calibration errors
   were much larger than acceptable. Repeating those runs could not repair them.
3. The first exact-scoring recovery omitted transformer activation memory
   management. All three uploaded seed logs then showed H100 OOM at prompt 356.
   Small CPU correctness tests did not establish H100 memory feasibility.
4. The original metering helper immediately terminated healthy siblings after
   one failed. Their mid-log termination was a consequence of that policy, not
   independent evidence of node eviction or extra CUDA failures.
5. Old and new errors appeared together in exports without enough runtime
   context. Log interpretation must use the failure record, phase and matching
   worker file rather than the first old abort in an append-only log.
6. Status labelled unused compute allocation as a completion-time floor and
   printed a finish timestamp even without observed evaluation duration. An
   allocation cap is neither a forecast nor a lower bound on actual runtime.
7. The separate existing queue was suggested as additional GPU work without
   checking the DONE rows in the already available queue dump. Those rows show
   its five remaining default GPU steps complete. Script existence is not
   evidence of unfinished work, and restarting that queue cannot be promised
   to occupy extra idle nodes.

## Recorded Repairs

- `6a90e1c`: separate exact autograd scoring recovery; preserve prior costs,
  cached results and original scientific source hashes.
- `5076cfb`: shared group-volume storage for diagnostic exports, with printed
  file paths instead of home-only files.
- `c42d5a7`: record the confirmed prompt-356 H100 OOM diagnosis.
- `28e200a`: per-decoder non-reentrant activation checkpointing without enabling
  dropout; allow healthy shards to finish within the original deadline;
  refuse unchanged automatic retries after a new checkpointed OOM; record
  runtime hashes, complete tracebacks and failing operation token lengths.
- The current status correction separates `[budget]`, `[history]` and `[ETA]`.
  It no longer presents allocation-derived hours as a predicted finish time.
  Reporting/evaluation time is not subtracted from the training allocation,
  and a stopped training branch has no remaining training allocation to run.

## Time Definitions

Unless explicitly configured, the suite budget is the median recorded source
GRPO step duration multiplied by 100 steps and four GPUs, rounded upward to
60 GPU-seconds. It is fixed when the suite is prepared. The 100-step setting
is an implementation default, not a mathematically optimal horizon.

For a metered log `elapsed_s / limit_s`, the denominator is the phase's time
limit, not its estimated completion time. Scoring and training receive the
remaining branch allocation divided by four GPUs. Diagnostic and failed-work
charges reduce that remainder. Final benchmark evaluation is reported outside
the training cap. Retrying does not restore the original budget.

Nine concurrent experiments are not nine serial budgets. Three unfinished
arms can occupy at most three nodes in this one-node-per-arm scheduler.
Additional node allocations do not divide one arm across multiple nodes.

## Debugging Delays And Previous Unfinished Work

The user explicitly reports time lost to debugging and experiments that did
not run the previous day. That time must not disappear from turnaround or
operational cost reporting. Keep three distinct quantities:

- Metered experiment GPU-seconds: `cost.jsonl` records allocation within phase
  intervals, including failed subprocess attempts and idle ranks within those
  intervals. These costs remain charged inside the frozen branch budget.
- Allocated-node GPU-seconds: allocation start/end records are needed to count
  node time spent waiting for a fix, idle between launchers, or blocked before
  a metered phase. If the node was released, debugging wall time is not billed
  GPU allocation time. These intervals are currently unmeasured, NOT zero.
- End-to-end elapsed time: experiment start to actual completion, including
  debugging, waits, interruptions and missed execution windows. Parallel
  jobs' elapsed durations must not simply be added together.

The available tail exports do not establish every node's allocation start/end
or the complete debugging timeline. No defensible total lost GPU-hours or
complete wall-clock ETA can be calculated from them. Do not refund failed
events or fabricate missing intervals to make the advertised duration fit.

`queue-dump-20260914T001318Z.txt` (2026-09-14 09:13:18 KST snapshot) records:

| Work | State in that snapshot | Treatment |
| --- | --- | --- |
| Reuse split-half d0 and d400 | DONE | Not newly available GPU work |
| E5 public benchmarks d0 and d400 | DONE | Not newly available GPU work |
| E5 d100 continuation | DONE | Not newly available GPU work |
| Fixed-checkpoint gate d0/s0,s1,s2 and d400/s0,s1,s2 | FAILED: recomputed g11 differs from original E5 scoring | Six unresolved historical failures, distinct from the net-gain gate |
| Mixed-pool point | RUNNING, downstream arms/gate waiting | Subsequently excluded from the manuscript/default queue; not counted as completed, do not restart implicitly |
| CPU analyses/export | PARTIAL | Does not justify allocating extra GPU nodes |

This snapshot does not prove that every fixed-checkpoint failure began on the
previous calendar day, nor that they have remained unresolved on the remote
cluster since export. Keep their last observed FAILED state rather than
silently marking them completed or merging them with the new nine-arm suite.

The current net-gain OOM repair does not fix the older g11 consistency failure.
The user's report of six completed and three running net-gain arms is a later,
separate source of state; no new export has independently verified it here.

## Verification And Remaining Limits

The memory repair passed 182 CPU orchestration/regression tests and 31 Torch
CPU tests, including small FP32/BF16 OLMo gradient equivalence, reduced saved
activation storage, sibling completion and deadline cleanup. No local H100
test was possible. A successful repaired H100 completion and measured
remaining-time estimate have not been verified from a new uploaded export.

The gate launcher still handles its own nine-arm suite. Other existing
experiments use `run_queue.sh`; automatic transfer from the gate launcher to
that queue has been discussed but has NOT been implemented. Do not claim that
the running nodes will pick up those experiments automatically after a pull.

For future fixes: preserve completed artifacts, test the actual numerical
backend and representative activation shapes, retain failed-work costs,
record implementation changes, and distinguish local verification from
observed remote completion. Never promise a fixed completion time from a
budget cap or from the small-model tests alone.
