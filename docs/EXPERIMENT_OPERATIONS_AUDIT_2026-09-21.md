# Experiment Operations Audit: 2026-09-21

## Reproduced Defects

- Default MBPP status observed the original quality root while the repair
  launcher worked in a separate root. A verified repair is now the primary
  48-branch view; the original remains visible as history outside that total.
  Repair schema, root identities, source manifest digest and identical copied
  manifest are required. Explicit suite selection and scheduling are unchanged.
- Pair curve evaluation could start a duplicate shard worker after its point
  supervisor exited but a shard still held its lease. The operational adapter
  probes unfinished shard leases after acquiring the point lease and defers a
  busy point through the existing pending path. Independent points and branches
  remain runnable. No lock file is removed and no peer process is stopped.
- One unreadable MBPP cost ledger could abort an entire results export. Ledger
  reads are isolated; readable measured endpoints remain available, while the
  affected cost total stays unknown. A missing or cyclic curve ledger cannot
  certify a closed total.
- An unreadable path in Pair's legacy progress scan could abort status. The
  operational viewer retains its independent known-meter scan, preserving
  readable peers and their RUN state.
- A damaged RLOO arm could remove other independently validated arms from a
  partial export. Operational reporting now isolates arm-local validation
  failures, rechecks the shared contract, and retains certified siblings.
  Invalid arms have no inferred reward; comparisons require both complete
  validated arms. Direct strict point validation is unchanged.

## Verification Scope

CPU tests use real file locks, real subprocesses and the Bash entrypoints.
Curve worker execution is replaced by a CPU fixture, not a paid GPU workload.
Checks include supervisor loss with a surviving shard, partial result reuse,
unchanged lock inodes, byte-preserved policies and costs, duplicate node names,
saved DONE with continuing work, invalid UTF-8, directories in place of files,
cyclic and dangling symlinks, permission failures, and partial TXT exports.

Pair root/runtime/barrier waits were tested with changing peer heartbeats:
their total deadline remains 180 seconds. Completed branches with a held
state-publication lease also stop waiting at the deadline. A separate simulated
four-hour healthy curve keeps its progress and is allowed to finish.

Integrated CPU/process regression: 2,133 passed, 10 skipped (including eight
opt-in CUDA cases). The final guard CLI and wait regressions separately passed
46 tests after the last exit-code cases were added. Bash syntax and generated
handoff-source synchronization checks passed. No remote H100 execution is
claimed. Local integrated report: `/tmp/experiment-operations-20260921.xml`.

These checks do not establish the cause of every remote WAIT. The older
exclusive-root-lock diagnosis is superseded by the new snapshot below. A local
test is not proof that a server allocation resumed. Never unlink a lock or infer
owner death from a hostname or timestamp alone.

The frozen scientific sources, Pair launcher, selectors, trainers, targets,
budgets, checkpoints and measured results are not changed by these repairs.
The isolated restart runtime is pinned to
`c321461fcb22ede9735777a6626d9a68186762ea`, including interrupted Pair cost
recovery, operational receipt validation, the curve guard and combined diagnostics.
The self-contained restart bundle is regenerated from its maintained sources.
Final deployment/guard/handoff regression: 180 passed. This includes staging
the actual pinned Git commit in a temporary checkout and checking that its
guard, launch hook, ownership checks and frozen science match the reviewed
files, with no live checkout update or rewrite on reuse.

## Bash Entry Points

Read-only status after updating the checkout:

```bash
git pull --ff-only
bash scripts/run_mbpp_experiments.sh status
bash scripts/run_selector_pair.sh status
bash scripts/run_rloo.sh status
```

Each results command writes one bounded TXT, including usable partial results:

```bash
bash scripts/run_mbpp_repair.sh results
bash scripts/run_paper_results.sh results pair
bash scripts/run_rloo.sh results
```

Outputs are `~/mbpp-repair-results.txt`, `~/selector-pair-results.txt`, and
`~/rloo-results.txt`. Original MBPP results remain separately available through
`bash scripts/run_mbpp_experiments.sh results`; they are not pooled with repair
attempts as if their cost histories were identical.

For a current blocked Pair allocation, collect one read-only diagnostic TXT:

```bash
bash scripts/check_selector_pair.sh
```

The command prints its output path. It does not restart or stop a worker.

For a one-off MBPP diagnostic upload without the total file-size cap:

```bash
MBPP_WHY_SINGLE=1 bash scripts/run_mbpp_repair.sh why
```

This invocation writes one TXT and prints its path. The environment override
applies only to that command; ordinary `why` still uses at most three
upload-sized parts. All generated sections are streamed without the old
5.7 MB aggregate truncation. Existing per-record/log safety bounds and
exclusion of model, optimizer and rollout payloads remain explicit.
Publication is atomic; a failed export does not leave a partial TXT.
Focused diagnostic and launcher regression: 144 passed, including actual Bash
execution, content beyond the old cap, UTF-8 preservation and failure cleanup.

To send MBPP and Pair together as exactly one diagnostic TXT:

```bash
bash scripts/check_mbpp_pair.sh
```

The command includes both original and repair MBPP roots plus the configured
Pair root, writes one `mbpp-pair-why-single.txt` in a new folder under the user's
home, and prints its full path. It creates no intermediate per-experiment TXT
files. Combined output has no aggregate cap, including Pair's lock, queue and
cost evidence; normal individual diagnostic commands keep their existing
limits. Per-record/read/log safety limits and excluded binary payloads remain
explicit. Missing roots are reported without creating them. A read failure in
one diagnostic section does not suppress the other experiment's evidence.
Combined diagnostic, launcher and handoff regression: 264 passed, including
actual Bash export, custom roots, held locks, unchanged file bytes/mtime/inodes,
and Pair evidence beyond its normal 1 MiB limit.

## Copied Diagnostic: 2026-09-20 23:40 UTC

Source: `~/mbpp-pair-why-single.txt`, 1,675,720 bytes, 11,263 lines,
checkout `9e46420a6fe7`. This is a captured remote snapshot, not live access.

- MBPP repair has 42 nongated endpoints and 39 completed curves: the queue
  reports 39 DONE, 3 RUN, 6 WAIT. The three curves hold real task/cost leases.
  Development has 17 of 18 curves; `s2-t50/selection_reduced` is still running
  and blocks gate fitting. The other two active curves are held-out controls,
  `s3-t25/selection_full` and `s4-t50/selection_reduced`. The six gated branches
  are dependent work, not six independently runnable jobs at this point.
- The original MBPP root has 37 completed branches. Its earlier rc=80 logs
  are not a new failure of the repair root; the two histories must not be mixed.
- Pair's root accepts shared workers. Cached `s0-t25` has an active parent
  curve, and cached `s2-t100` has active training, supported by both leases and
  fresh meter records. Neither should be interrupted to repair other branches.
- Nine development attempts are blocked by open main cost events with no
  atomic finish receipt. Pair did not invoke the existing interrupted-cost
  recovery used by Switch/MBPP. Recovery estimates must retain their evidence
  and remain distinguishable from measured finish-receipt costs; no costs may
  be refunded and no already-published cost history may be rewritten.
- Six other saved attempt errors report the branch quarantine runtime receipt
  mismatch. Those errors are several hours old, and this upload omits the
  actual runtime receipts, so it does not prove a mismatch in the current
  runtime. The current reviewed migration must be validated before GPU
  admission, rather than adding an unverified hash exception.
- Pair's 18 saved attempt records classify as 15 WAIT, 2 RUN, 1 DONE. These
  are attempt records, not a revalidated experiment completion total.

The operational follow-up preserves the scientific sources and adds Pair
cost recovery with active/published-work exclusions, read-only adapter receipt
validation before a handoff can stop a controller, and MBPP peer-only readiness
deferral before GPU admission. Diagnostic exports include previously omitted
runtime/publication/queue evidence so old errors can be separated from current
artifacts. Healthy workers do not need a restart; update idle allocations.

The v4 Pair results TXT carries `cost_provenance`, including source ledger
hashes and explicit reconstructed-cost warnings. Reward measurements remain
unchanged. A recovered interruption without an atomic finish receipt is not
called a directly measured finish or a guaranteed cost upper bound.

To apply this release on an idle allocation without editing the checkout used
by a live worker, create a separate Git worktree from the published master:

```bash
git fetch origin master
FIX=$(mktemp -d /tmp/offpolicy-ops-fix.XXXXXX)
git worktree add --detach "$FIX" origin/master
cd "$FIX"
```

Then run the command for that allocation's experiment, not both on one node:

```bash
# Pair allocation that is waiting, not currently training or evaluating:
bash scripts/restart_selector_pair.sh
```

```bash
# MBPP allocation that is waiting, not currently training or evaluating:
EXPERIMENTS_AUTO_PULL=0 bash scripts/run_mbpp_repair.sh restart
```

The Pair launcher stages the pinned runtime and validates local ownership
before handoff. The MBPP command explicitly restarts only its allocation's
controller; automatic checkout updates are disabled in this detached worktree.
Do not restart the three active MBPP curves or two active Pair tasks identified
in this snapshot just to update idle workers. Remote progress after deployment
still requires confirmation; the copied diagnostic is not a live server view.

Final follow-up integrated regression: **2,257 passed, 10 skipped** in 70.35s.
This includes the newly pinned Git deployment, generated Bash synchronization,
real-process handoff acceptance/refusal, MBPP queue readiness, cost recovery,
status and partial results exports. The skips include eight opt-in CUDA cases;
no remote GPU execution is claimed. There were 47 existing Python fork/thread
deprecation warnings, no test failures. Report:
`/tmp/experiment-operations-new-why-20260921.xml`.

## MBPP Follow-up After Partial Pair Recovery

The user confirmed partial Pair recovery but reported that MBPP was unchanged.
No newer remote diagnostic had been copied at this point. The previous upload
actually shows increasing rollout counts for all three active MBPP curves,
not merely a periodic supervisor heartbeat. That older snapshot cannot establish
the cause of the user's current server state.

Three separate defects were reproduced and corrected without changing frozen
scientific code, budgets, saved measurements, or the Pair runtime:

- The MBPP queue adapter omitted the frozen driver's signal-handler setup.
  A real CPU meter receiving SIGTERM left only a cost start record and a live
  detached child holding its shard lease. The adapter now installs the same
  signal handlers as the original driver. SIGTERM and SIGINT regressions verify
  a matching atomic finish receipt, owned child termination, and release of task,
  cost and shard leases without deleting or replacing their lock files.
- Default MBPP status selected a verified prepared repair, while default
  `run`/`restart` still selected the original quality root. Thus an idle node
  could run the old rc=80 path while the viewer showed repair work. Default
  startup now chooses the same already-prepared repair. It does not create a
  repair, authorize additional retries, or relax runtime validation. Explicit
  `quality` and custom quality roots retain their existing meaning; an ordinary
  repeated `run` still does not interrupt a live controller.
  Strict repair validation now also runs before an explicit controller restart
  or any cost recovery, not merely before GPU admission. Invalid repair
  metadata leaves the existing controller and every cost ledger untouched.
- Allocation identity treated `all` and `quality` as different suites although
  the default `all` plan contains only `quality`. This blocked switching between
  the normal and repair launchers on the same verified allocation. Only these
  two aliases are now equivalent; process identity, random owner token, GPU
  allocation, namespace, user and work-directory checks remain mandatory.

After fetching the new master into a separate worktree as above, an **idle MBPP
allocation** can apply the fix using the default launcher:

```bash
EXPERIMENTS_PULL=0 EXPERIMENTS_AUTO_PULL=0 bash scripts/run_mbpp_experiments.sh restart
```

The launch log prints `[mbpp-route]` when it selects the verified repair.
Running evaluations on other allocations must be left alone. A new read-only
snapshot, when needed, remains one TXT from `bash scripts/check_mbpp_pair.sh`.

Final MBPP follow-up regression: **2,283 passed, 10 skipped** in 97.84s,
including actual Bash routing, actual signal/child/flock cleanup, preservation
of a live controller on an invalid repair, and Pair/RLOO/status/results checks.
Eight CUDA checks remain opt-in; remote GPU resumption was not verified locally.
The 47 fork/thread deprecation warnings are unchanged. Report:
`/tmp/mbpp-resume-followup-20260921.xml`.

## New-node Allocation Follow-up

The user reported no assignment on a new MBPP node and Pair assignments on ten
nodes but WAIT on subsequent nodes. The available diagnostic still predates
these reports; current runnable counts and the exact remote blocker remain
unverified. No active remote worker was stopped during this investigation.

Pair has no ten-worker limit. Its development stage contains eighteen branches;
after development validation and decision freezing, its test stage contains
twenty-four branches. These are separate stages, not forty-two simultaneously
eligible tasks. Real-process CPU queue tests admit all eighteen or twenty-four
independent branches concurrently. A partially completed stage can have fewer
eligible branches, but this does not establish the current remote count.
An additional restart regression preserves eight completed branches with no
queue receipts and an obsolete RUN worker record, then starts eleven processes.
Ten distinct processes claim the ten remaining branches; the eleventh waits.
All finish after release without rewriting saved results or old worker records.

One MBPP scheduling omission was identified: the hold poll recognized READY or
retryable branch work but not a newly ready gate fit after all development
results arrived. The regular pass could still fit the gate, so this omission
delays dispatch until the hold deadline rather than proving a permanent block.
The MBPP wrapper defaults to a fifteen-second hold; inherited overrides can
make it longer. The follow-up adds a read-only gate-fit readiness wake-up,
without bypassing worker validation or claiming peer-owned work.

The remote diagnosis still requires a fresh combined TXT from the affected
allocation using `bash scripts/check_mbpp_pair.sh`. Do not infer ownership from
hostname equality or stale timestamps, remove held lock files, or restart the
ten working Pair allocations to make an idle allocation pass admission.

Verification: the actual Bash gate-fit wake regression failed in its two
positive cases before the patch and passes all ten cases afterward. The wider
MBPP/controller/Pair queue suite passes **975 tests** in 47.38s, with 56 Python
fork/thread deprecation warnings and no failures. Bash syntax and whitespace
checks pass. Report: `/tmp/new-node-allocation-20260921.xml`. These are local
CPU/process tests, not proof that either remote allocation issue is resolved.

## Authorized Pair Scheduling Amendment

The user explicitly requested that independent held-out controls no longer
wait for development completion. This changes the previous global execution
order, not the DEV/TEST split or the scientific inputs to the fitted rule.

- Development keeps priority. When its remaining branches are peer-owned or
  otherwise unavailable on this controller, the controller can claim one of
  eighteen fixed held-out controls, then checks development again.
- Fixed controls are exactly TEST seeds 3/4 at steps 25/50/100, with
  on-policy selection, cached selection, and the on-policy random control.
  Six adaptive continuations still require all development labels and frozen
  held-out decisions. Their outcomes never enter gate fitting or prediction.
- `pair-parallel-controls-runtime.json` records the amendment and binds the
  protocol, unchanged manifest, operational code, and exact control registry.
  Initial authorization refuses pre-existing unfrozen held-out work instead of
  retroactively approving it. Existing valid frozen decisions remain valid.
- Old guard and cost-recovery receipts are preserved byte-for-byte through a
  narrowly bound predecessor migration. Frozen source files, budgets,
  checkpoints, costs and published measurements are not rewritten.
- Existing DEV workers retain their leases and can finish. An old worker that
  later reaches the previous all-TEST barrier can reject early control work;
  a new worker must perform the amended freeze. A frozen-science test still
  verifies the original refusal outside the authorized operational context.
- Status exposes pre-gate fixed-control readiness only for a valid amendment;
  result TXT exports retain explicit schedule provenance separately from
  measured results, cost provenance, and paired completion.

This is an explicit schedule amendment, not a claim that wall-clock execution
conditions were unchanged or that the remote queue was independently observed
to resume. Healthy remote workers must not be stopped to update idle nodes.

Focused verification passed **30 tests**: eighteen actual CPU processes claim
distinct fixed controls while all DEV leases remain owned, the real scheduler
completes all 42 branches and resumes without repeating work, adaptive work
requires frozen choices, and altered fixed outcomes do not change the DEV model.
Receipt migration, hook restoration, pending-work handling and artifact path
validation are also covered. GPU training is substituted in these local tests;
this does not constitute a remote GPU execution check.

Final MBPP/Pair/RLOO/status/results/controller regression suite, including the
exact pinned runtime and generated Bash checks: **2,399 passed, 10 skipped** in
80.83s, with 76 Python fork/thread deprecation warnings. The first broad run
detected an out-of-sync generated launcher; it was regenerated before this full
passing rerun. No test was deselected in the final run. Report:
`/tmp/pair-fixed-controls-final-20260921.xml`. The deployment pins runtime commit
`d628e7467f209248f91bb7c272dabafd26dd73a3`; Bash syntax and whitespace checks pass.
