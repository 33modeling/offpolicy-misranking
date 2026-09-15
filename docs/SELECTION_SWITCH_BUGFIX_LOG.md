# Selection Switch Bug-Fix Log

Updated: 2026-09-15 (KST). This is the dedicated incident and repair record for
`scripts/run_selection_switch.sh` and its isolated MoPPS comparison. Keep new
reports and verification evidence here; do not replace unresolved incidents
with a list of successful unit tests.

Related records: [switch runbook](SELECTION_SWITCH_RUN.md),
[switch design](SELECTION_SWITCH_EXPERIMENT.md),
[MoPPS comparison](MOPPS_COMPARISON.md), and
[earlier net-gain execution corrections](NET_GATE_EXECUTION_CORRECTIONS_2026-09-14.md).
The earlier net-gain incidents are not proof of the cause of these switch failures.

## Current Status

- OPEN: the user reports that execution starts but stops partway through.
  No complete traceback, matching current worker log, or exit status for this
  latest interruption has been provided. Its cause and resolution are NOT verified.
  The user reported another stop after receiving resume instructions. The exact
  deployed revision and the final error for that recurrence remain unconfirmed.
- FIXED LOCALLY, PUSHED: an exact original `cb01401` frozen run was incorrectly
  rejected by the runtime compatibility list. Reproduced locally and repaired
  in `ac40a60`. This does not establish that the latest cluster interruption
  has the same cause.
- OPEN: the underlying cause of the reported `prefix-train worker failed`
  exception has not been established from a matching full worker traceback.
  Better logging is implemented; that is not a training-failure repair.
- FIXED LOCALLY, PUSHED (`2daa140`): live-checkout updates could change code under
  a running controller and its later child processes. The same frozen-run error
  was reproduced by changing the source mid-run; pinned runtimes passed that
  reproduction. Whether this is the cause of the user's latest stop is unverified.
- No successful post-fix H100 completion or uninterrupted long cluster run
  has been independently verified here.
- Last user-reported allocation: four nodes running. This is not a live
  observation or evidence that those nodes are still running now.

## Reported Symptoms

The user reported repeated errors, inability to make progress, and time lost
to restarts and debugging. Preserve these reports separately from diagnoses:

| Report | Evidence and disposition |
| --- | --- |
| KV cache left enabled during gradient recomputation | Teacher-forced scoring forward paths were patched in `96ad9ed`; cluster completion remains unverified. |
| `unclosed cost event at {directory}` / `unknown cost cannot be treated as zero` | Cost completion/recovery and research-prefix resume changes were implemented. Unknown deployment cost still correctly blocks unsafe continuation. No evidence establishes that every reported cluster event has been closed. |
| `prefix-train worker failed` / worker exit codes pointing to logs | Supervisor-level symptom; `4798f93` exposes child log tails. Underlying reported worker failure remains unconfirmed. |
| `switch protocol or scientific code changed; preserve the frozen run` | One exact compatibility omission reproduced and fixed in `ac40a60`. Unknown code changes continue to be rejected. |
| Starts, then stops partway through | Latest report, OPEN. Do not infer OOM, scheduler eviction, idle timeout, or a hash mismatch without the matching output. |
| `waiting`, unclear status, questions about parallel nodes | Nonblocking task scheduling, peer waiting, and separate read-only status were improved. Waiting is not by itself evidence that parallelism is disabled. |
| Broken glyph animation | User-reported, OPEN and not diagnosed. The affected surface and reproduction were not identified; no animation repair is claimed. |

Reported line-number groups included `582/247/75`, `637/295/77`, `714/274/77`,
`738/371/119/154`, and most recently `737/621/57/79`. These are user-provided
identifiers, not complete stack traces. They do not all match the current
checkout. Do not infer a specific function or root cause from them alone.

## Repair History

All commits below were pushed to `origin/master`. Dates are commit dates in KST,
not independently established incident or cluster deployment timestamps.
"Implemented" below does not mean verified on the user's cluster.

| Date | Commit | Change and scope |
| --- | --- | --- |
| 2026-09-14 | `cb01401` | Original selected-prefix switch experiment, matched paid controls, and shared queue. Historical starting version, not a repair. |
| 2026-09-14 | `a63e69d` | Recover legacy initial fresh-r scalar metadata on CPU from validated saved gradients; preserve source artifacts and record preparation failures. |
| 2026-09-15 | `96ad9ed` | Explicit `use_cache=False` in teacher-forced gradient/scoring forwards, including checkpointed recomputation paths. Generation caching was not globally disabled. |
| 2026-09-15 | `bb32da3` | Protect metered phase startup and cleanup; recover matching atomic finish receipts; skip busy publication tasks so other nodes can claim ready work. |
| 2026-09-15 | `69bec8d` | Resume interrupted research prefixes while explicitly retaining unknown historical costs and archived progress. Keep waiting while locked peer work has fresh running heartbeats. No deployment-budget waiver. |
| 2026-09-15 | `4798f93` | Include failed child stderr/log tails in supervisor errors and add operational error inspection. This exposes failures; it does not establish a fix for their underlying training cause. |
| 2026-09-15 | `7dc108a` | Compact read-only status with node, prefix, continuation, failure, and dependency views. No original switch scientific-code hash changes. |
| 2026-09-15 | `857c3fa` | Add isolated MoPPS versus executed Gate comparison and online-random control. This is an experiment extension, not a repair of the original switch worker. No original switch code-map file changed. |
| 2026-09-15 | `ac40a60` | Accept the missing exact original frozen runtime, preserve previous migration receipts, provide file/hash mismatch details and read-only `check-code`, preflight before GPU admission, and retain compatibility for already frozen MoPPS sidecars. |
| 2026-09-15 | `2daa140` | Execute switch/MoPPS launchers and workers from a verified local detached clone, not the live checkout; pin `OM_REPO`/imports; show launcher log tails and record controller start/exit codes. Scientific code maps unchanged. |

### Frozen-Run Compatibility Repair

The validator before `ac40a60` accepted four reviewed predecessor maps but
omitted the original `cb01401` map. A run frozen with that original code could
therefore fail even though the subsequent reviewed runtime fixes were compatible
with its already frozen inputs. The preparation-only legacy recovery addition
does not require recomputing or replacing an existing selected subset.

Local reproduction used actual Git versions, not a guessed replacement hash:

- Running the prior `857c3fa` validator with the original `cb01401` map raised
  the reported generic frozen-run error.
- The repaired validator accepted that map and preserved all pre-existing
  fixture bytes. Repeated resume was idempotent.
- Actual `a63e69d`, `96ad9ed`, `bb32da3`, `69bec8d`, and `4798f93` maps also
  resumed while preserving their prior manifest and runtime receipts.
- `4798f93`, `7dc108a`, and `857c3fa` have identical original switch code maps.
  The MoPPS extension itself did not change the original switch map.
- Compatibility is restricted to reviewed predecessor maps with other
  scientific files unchanged. Unknown predecessors and scientific changes
  remain errors; the guard was not disabled.
- The repair appends `code-compat-runtime.json`. It does not rewrite
  `switch.json`, existing receipt chains, selected subsets, checkpoints,
  gate decisions, completed results, cost journals, or branch budgets.
- Already frozen `857c3fa` MoPPS runs can retain `mopps.json` and existing
  work with a separate compatibility receipt. The source switch remains
  read-only from the comparison runner.

### Mid-Run Code Isolation Repair

The earlier fix validated compatible code maps but left the controller running
from the mutable shared checkout. Later imports, subprocess script paths, and
code validation could therefore see different bytes after a checkout update.
An inherited `OM_REPO` could also make `setup_env.sh` prepend a different
checkout's `src` directory. Startup success alone does not protect either path.

`2daa140` re-enters a verified, detached local clone before preparation or GPU
admission. The controller, later child processes, `OM_REPO`, and Python imports
use that clone. Relative input paths and the original run/model/data/venv roots
are retained. Snapshots are keyed by commit, created under a filesystem lock,
and reused without replacing an existing running snapshot. No remote Git origin
is retained in the clone. The source is never reset or cleaned; dirty source or
modified caches fail before admission instead of being silently discarded.

Both original switch and MoPPS scientific code maps are byte-identical to those
before this repair. No compatibility list, training algorithm, budget, source
artifact, result, or cost journal was changed by the isolation patch.

The `errors` command previously searched task `failure.json` files but omitted
launcher logs. A controller-level validation exception could therefore be absent
from that diagnostic. It now also shows bounded launcher tails. Launchers record
start revision/PID and exit code, including ordinary shell failure and handled
termination. A hard kill cannot guarantee an exit marker. A log tail is evidence,
not an automatic diagnosis, and does not prove a cluster repair succeeded.

Local reproduction and tests for this repair:

- Unpinned fixture: start, change the committed live source, continue. The
  controller raises the same scientific-code/frozen-run mismatch.
- Pinned fixture: the same update leaves the running controller and its next
  subprocess on the original source bytes; both complete.
- Actual switch and MoPPS shell entrypoints, with CPU/fake GPU admission,
  pass this scenario while keeping output storage and interpreter paths.
- Four concurrent local processes reuse one snapshot and complete after the
  live checkout changes. Dirty sources, changed caches, relative paths, stale
  inherited imports, missing task error records, and log path escapes are tested.
- Full local regression: **290 passed, 1 skipped**, with the same two tiny PEFT
  fixture warnings. This is not a real H100/shared-cluster-filesystem run.

The preceding compatibility omission and this reproducible isolation defect are
distinct. Do not retroactively claim that either one explains every reported
interruption. The latest cluster interruption stays OPEN pending matching evidence.

## Verification Record

For `ac40a60`, local regression results were **274 passed, 1 skipped**.
The skip concerns optional plotting. Two warnings came from tiny PEFT test
fixtures without a base-model configuration file.

Coverage includes original/latest frozen-run migration, partially written and
pre-existing receipt chains, tampered receipt rejection, scientific-code change
rejection, read-only CPU preflight, and four local processes concurrently
migrating the same fixture root. Unknown deployment costs remain blocking.
MoPPS coverage includes preserving old protocols and the source run, rejecting
changed scientific code/design, and actual tiny CPU policy/optimizer/posterior
checkpoint-resume tests.

The four-process test uses one local filesystem. It is not a four-node H100,
NCCL, shared cluster filesystem, verifier-throughput, or long-duration test.
No cluster authentication or execution was attempted. Do not report these
CPU checks as remote operational success.

Command for the current regression suite (290 passes after `2daa140`):

```bash
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  PYTHONPATH=src:/home/kms/.local/lib/python3.12/site-packages \
  .work/.venv-cu126/bin/python -m pytest -q \
  tests/test_mopps.py tests/test_mopps_comparison_gpu.py \
  tests/test_selection_gate_gpu.py tests/test_net_gain_gate_gpu.py \
  tests/test_selection_switch.py tests/test_selection_switch_gpu.py \
  tests/test_selection_switch_cost.py tests/test_selection_switch_errors.py \
  tests/test_selection_switch_status.py tests/test_net_gate_memory_math.py \
  tests/test_selection_switch_runtime.py \
  tests/test_logit_chunking.py tests/test_low_order_backend.py tests/test_protocol.py
```

## Verification And Reporting Failures

The assistant omitted the first released frozen-runtime version from prior
compatibility regression coverage. That should have been caught before asking
the user to update and resume. Subsequent tests passing did not erase that gap.

The assistant also used "fixed" too broadly when only a locally reproduced
compatibility defect had been repaired. CPU regression success, remote
deployment, and successful long-running cluster execution are different claims.
The user's latest mid-run interruption is still unresolved and must stay OPEN
until its own evidence supports a diagnosis and successful recovery.

For future entries, record the symptom, evidence, actual root cause if known,
commit, preserved artifacts, local test scope, and remote verification status.
Do not close a worker failure merely because its wrapper now prints better logs.
Do not close a mid-run interruption because startup preflight passes. Runtime
validation also happens after startup, including during final status reporting.

## Operator Boundaries And Next Evidence

The cluster is a restricted security environment. The user handles authentication,
deployment, and execution and supplies logs. The assistant works on the local
repository and pushes reviewed changes; it must not attempt cluster access or
ask for credentials. A push is not evidence of deployment on any node.

For the first isolation deployment, gracefully stop the affected old live-checkout
launchers and confirm their worker trees have stopped. Pull once for a shared
checkout, then use that same revision on its nodes. A newly pinned worker logs
`[runtime] commit=... pinned=...`; later changes to the original checkout cannot
change that worker's source. This does not authorize mixing scientifically
different revisions in one frozen run. Never modify an active runtime cache,
kill unrelated experiments, or delete run roots, locks, receipts, checkpoints,
or cost records to bypass checks.

Read-only diagnostics, using the same `SWITCH_ROOT` as the failing run:

```bash
git rev-parse --short HEAD
bash scripts/run_selection_switch.sh check-code
bash scripts/run_selection_switch.sh errors --limit 1
bash scripts/run_selection_switch.sh status
```

For the latest mid-run interruption, the missing evidence is the final launcher
output/full traceback with filenames, the matching worker log and phase, exit
status, node/host, and deployed revision. Scheduler termination evidence is useful
if it exists. `Killed`, SIGTERM, a traceback, or an ordinary idle exit must not
be treated as interchangeable. No new root cause has yet been established.

Current queue behavior: it skips busy tasks, waits while peers have fresh running
heartbeats, and can exit when it finds no claimable work or its inactive-peer
idle limit expires. A task is not retried repeatedly in one invocation after
failure. This describes inspected code, not the cause of the latest report.

## Costs, Nodes, And Comparison Scope

- The user reports substantial time lost to failed runs and debugging. Total
  lost wall time and allocated GPU-hours have not been measured; they are
  unknown, not zero. Scheduler allocation and termination records are needed
  for a defensible total. Preserve failed-work charges and unclosed events.
- One task uses one four-GPU node; a single task is not distributed over sixteen
  nodes. Same-seed prefix segments are sequential; different seeds and ready
  continuations can run concurrently. Five independent prefix chains limit
  initial useful parallelism. More nodes do not fix failures or guarantee a
  finish time. The user's four-node report is the latest supplied allocation.
- The user explicitly requested reward-based selection versus the actual Gate,
  with MoPPS named as prior work. `857c3fa` adds this as a separate-root comparison
  with twelve new continuations, reusing six original executed Gate results.
  The primary comparison is Gate versus MoPPS, not merely MoPPS versus random.
  This is an implementation/integration, not a completed cluster result or a
  full reproduction of the original paper's training stack. See its runbook
  for the frozen budget, initialization, sampling differences, and limitations.

## Record Maintenance

Clarification recorded on 2026-09-15: the assistant recommended the existing
switch comparison to test Gate utility and added MoPPS as the requested reward-
based comparator. Missing verified cluster outcomes do not mean the comparison
code is absent, nor do they establish a negative Gate result. The current
`run_selection_switch.sh run` executes the original 48 continuation arms;
`run_mopps_comparison.sh run` executes the separate 12-job extension. Gate is
compared with the full-budget selection/random controls in the original study
and with MoPPS in the extension. SEL/RND supply the diagnostic-paid action labels.
A draft to chain both queues automatically was not shipped during this
clarification; the existing entrypoints and frozen experimental designs remain
unchanged. The user should not be told to change comparators merely because
the assistant has not received the experiment outputs.

Append new evidence with its date and source. Distinguish user-reported state,
local reproduction, source inspection, and verified cluster outcomes. Link each
repair commit and state whether it was pushed, deployed, and remotely verified
separately. Keep failed attempts and uncertainty visible instead of rewriting
the history as a sequence of completed fixes.

## 2026-09-15: Ctrl+C Cleanup And MoPPS Startup Dependency

User reports: GPU workers remain after Ctrl+C; both queues wait; every MoPPS
node immediately waits, including newly allocated nodes. Cluster logs and
source certificates have not been supplied for this report. The separate
Switch two-active/two-waiting report remains unconfirmed: no Switch dependency
barrier is removed in this repair and full four-node utilization is not claimed.

Confirmed defects:

- `selection_gate_gpu.terminate` skipped an exited process-group leader and
  escalated only when the leader itself timed out. A CUDA child could survive
  either an already-dead leader or a leader that exited on TERM. The old code
  failed the real RTX 3050 CUDA child-exit test before the repair; the fixture
  confirmed the child in `nvidia-smi` and then cleaned up only its own process.
- The MoPPS Python entrypoint never installed the shared SIGTERM unwind handler.
  Its shell forwarded TERM on Ctrl+C, so the supervisor could exit without
  executing metered cleanup or closing the cost event.
- Repeated shell stop traps and SIGINT during child registration/cost writing
  could interrupt shutdown itself.
- `mopps_comparison_gpu.ready` required the original Gate barrier and state
  contract, even for already-certified prefixes. New nodes cannot satisfy that
  shared input dependency. Previous four-process queue tests mocked `ready=True`,
  so they did not exercise the faulty readiness rule. The revised queue test
  uses real readiness with prefix certificates and no Gate model/barriers.

Repair scope:

- Signal owned process groups even when the leader has exited. After a five-
  second TERM grace period, KILL surviving groups and verify live members are
  gone before finalizing costs. Unconfirmed termination leaves cost unknown.
- The expanded real torchrun/NCCL check caught a further gap after the first
  group-only CUDA tests passed: elastic ranks use `start_new_session=True`.
  Tag descendants with a unique inherited cost-event environment key and track
  those keys across sessions and reparenting. Re-scan during TERM/KILL cleanup.
  Never signal a long-exited leader PID without current ownership evidence.
- Defer stop signals while registering a child and finalizing cleanup/receipts.
  Both entrypoints install SIGINT/SIGTERM unwind handlers. Both launchers use
  the same stop helper, install traps before spawning and wait through repeated
  stop signals before releasing the node lock.
- New MoPPS imports need only their immutable registered prefix. Create a
  private view in the comparison root, validate the full prefix/optimizer
  lineage and bind the same pool, evaluation, RNG, allocation and selector.
  Do not write the parent root. Existing imported contracts remain unchanged.
  Final comparisons still validate the original Gate barrier and actual result;
  no MoPPS outcome becomes a Gate feature or development label.
- Missing-prefix waits now name the missing certificate rather than implying
  a Gate dependency. A missing prefix is still a real prerequisite, not zero
  work or a reason to fabricate a checkpoint.
- Exact predecessor code maps remain accepted with additive runtime receipts.
  Frozen manifests, prior receipts, checkpoints, costs and results are preserved;
  unreviewed scientific changes and unknown deployment costs remain rejected.

Local evidence: the initial direct-process CUDA lifecycle suite passed 12 cases (six
CUDA, six CPU), covering both entrypoints, repeated INT/TERM, group-leader exit,
GPU PID removal, cost closure, lock release, restart and unrelated-process
survival. These tests replace the evaluation payload with a small allocation;
they are not original-model training outcomes. That first result was not
sufficient for torchrun: the expanded suite failed four cases (including two
cold-start fixture timeouts), with separately sessioned ranks surviving TERM.
Those test-owned processes were explicitly cleaned up; no other jobs were
stopped. After event-nonce tracking, all eight CUDA cases passed, including real
single-rank NCCL, both entrypoints, both stop signals and restarts. A separate
nine-case CPU lifecycle run also passed. Final code verification is recorded
below after completion; these results do not certify four-H100/NCCL execution.

Deployment: the assistant cannot access the secure cluster. A pull does not
patch an already-running pinned worker. Stop the old waiting MoPPS launchers,
pull the repair and rerun the same command/root on free nodes. Old GPU workers
must be confirmed stopped before reusing their GPUs; do not globally kill
Python, reset GPUs, delete locks or rewrite cost records.

Current reviewed source-map fingerprints (frozen run manifests are not rewritten):

- Switch: `43f53caa042b27810fe3ba45da025198953378e6f858868cdb93e004bea3a60e`.
- MoPPS: `0bb0f42a82215281267a5f8741f1e9b87d16ccaaf945ace32c81f50c6400f33a`.

Regression fixtures run on local Linux, including a tmpfs temporary directory
for the final suite to avoid unrelated disk contention. Short mocked-task
overlap tests use an explicit overlap rendezvous; the Switch test starts with
already-published decisions and its publication-lock test remains separate.
These are process/queue contract checks, not measurements of four-node
training throughput or shared-cluster filesystem behavior.

Final verification on the reviewed code:

- CPU/process regression command: 309 passed, 1 optional plotting skip;
  9 deselected by `not cuda`. Report: `/tmp/switch-mopps-release-regression-20260915.xml`.
  The one non-GPU shell test whose name contains `cuda` was rerun separately
  and passed, giving 310 passing CPU/process checks in the final validation.
- Actual RTX 3050 CUDA suite: 8 passed, covering four detached/same-group rank
  cases and four Switch/MoPPS INT/TERM cases with two launches each. The latter
  use real `torch.distributed.run` and a single-rank NCCL collective. Report:
  `/tmp/switch-mopps-cuda-release-20260915.xml`.
- `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader`
  was empty after the final suite. Shell syntax and `git diff --check` passed.
- No cluster login, four-H100 execution, lost-time total or original-model
  comparison outcome was verified. The runtime patch and this record are one
  commit; operator deployment remains separate from Git publication.
