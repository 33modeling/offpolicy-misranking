# Cache-to-training activity reporting on 2026-09-28

The user reported cache completion followed by repeated RUN/DONE and backup
checks without a checkpoint. Local checks reproduced the normal cache-to-prefix
handoff but did not reproduce or establish the remote stall's cause. No remote
log, queue directory or GPU process was accessible for this check.

The ordinary worker now reports its task, attempt, child PID, actual phase and
rank progress receipts, or idle queue dependency states alongside backup scans.
Reporting failures are surfaced without interrupting training; existing
progress receipts and the stall watchdog are not rewritten. This is a
diagnostic improvement, not a claim that the reported remote stall is fixed.
The experiment implementation digest, cache, checkpoints, scientific protocol,
task order, retry limits and device/task locks remain unchanged.

New CPU regressions exercise all 30 cache/prefix/continuation handoffs, a held
cache dependency, periodic phase/rank reporting without receipt writes, malformed
progress, and preservation of the frozen experiment digest. A local simulation
does not verify execution on the user's H100 nodes.
All 172 unit/integration tests passed without skips; shell syntax and whitespace
checks passed. The experiment digest remains
`1869fe1cf898d4ff3a6d5e9054790836442b5e0b81b485fb04bc27de4ebab20a`.

# Live status and result verification on 2026-09-28

Added a CPU-only status adapter outside the frozen experiment package. The
ordinary status command separates current optimizer step, completed updates,
task denominator, node and cache availability. It reads the latest existing
phase receipt written after the queue attempt started, not the largest step
from an earlier interrupted attempt. A stopped task's last observed step is
not shown as active work. Evaluation/checkpoint phases do not imply a published
reward, and completed updates do not imply a saved checkpoint. Missing or
malformed progress is not fabricated as zero. Existing workers need not be
restarted to use the updated read-only status command. No training protocol,
prefix dependency, queue scheduling, cache or checkpoint cadence was changed.

All 167 unit/integration tests passed without skips. New status cases cover
selection before the first update, completed/saving/restored steps, continuation
and evaluation, retry exclusion, stopped workers, completed receipts, malformed
records, exported caches without separate receipts, and watch-mode interruption.
Both MATH and MBPP status/results entry points passed with Torch, Transformers
and PEFT imports blocked, without modifying inputs or initializing a queue.

Results tests use synthetic receipts in temporary directories, matching the
existing endpoint and cost schemas. They cover all twenty arms and five seeds,
complete cost reconciliation including shared-prefix/cache/invocation overhead,
legitimate zero-cost absent stages, null interrupted costs, partial means,
malformed endpoint/cost isolation, twenty-row CSV output, JSON/text sidecars,
and the worker's automatic final summary/cost-comparison publication. The
existing results aggregation and export code was retained. These tests do not
assert that uninspected remote H100 results are complete or error-free.

Shell syntax and whitespace checks passed. The experiment digest remains
`1869fe1cf898d4ff3a6d5e9054790836442b5e0b81b485fb04bc27de4ebab20a`.

## Current training-step logs on 2026-09-28

The ordinary worker's storage adapter now prints the current optimizer step
before selection/rollout work begins and the completed count after each update.
Shared-prefix logs use the plan's 25-update denominator; continuations use the
plan's total of 275, retaining the restored global step. Only rank zero prints.
Cache prompt counts are separate and are not presented as training progress.
No queue scheduling, training, checkpoint frequency or cache rules changed.
Running training processes do not hot-reload these logs.

All 156 unit/integration tests passed without skips. New regressions verify
that the first running line precedes gradient scoring, completion counts only
advance after successful updates, restored steps continue correctly, the final
prefix step is 25/25, continuations start at 26/275, and nonzero ranks are silent.
The two-process CPU distributed smoke test also passed with one set of training
lines, per-update saving, measured checkpoint costs and failed-write recovery.
Whitespace checks passed. The frozen experiment implementation digest remains
`1869fe1cf898d4ff3a6d5e9054790836442b5e0b81b485fb04bc27de4ebab20a`.
No remote H100 process was accessed or restarted.

## Per-update checkpoint saving on 2026-09-28

New ordinary shell workers save after each completed optimizer update for
the shared prefix and all four continuations. A storage adapter under
`scripts/` fills the gaps between the frozen runner's five-/25-update saves;
it does not change the experiment computations, cache, plan or queue identity.
Checkpoint metadata records the interval and adapter SHA-256. Extra snapshot
and write time is measured in the existing checkpoint cost ledger. Writes
publish atomically after flush/fsync; failed writes preserve the prior file
and leave their cost receipt unfinished, not zero. Partially executed updates
are not checkpointed. Existing processes keep their old saving cadence until
the worker is restarted; updating files or the backup watcher alone cannot
change an already-running training process.

Backup watchers now print one status line per completed scan, including
waiting, unchanged and busy states. Polls are followed by a 30-second wait;
copying duration is additional. A scan is not a claim that a new checkpoint
was saved. The normal shell command still resumes existing work automatically.

All 153 unit/integration tests passed without skips. Added coverage verifies
prefix and every continuation, intermediate-state restoration matching the
unchanged selector/model/optimizer trajectory, checkpoint cost stages,
atomic failed-write recovery, preservation of the prior save on failed
updates, unchanged cache task commands, and worker dispatch restoration.
The two-process CPU distributed smoke test passed actual checkpoint saving,
two-rank timing aggregation and propagation of an injected rank-zero write
failure without hanging or replacing the previous checkpoint. Shell syntax
and whitespace checks passed. The experiment implementation digest remains
`1869fe1cf898d4ff3a6d5e9054790836442b5e0b81b485fb04bc27de4ebab20a`.
No H100 worker, running experiment or remote checkpoint was accessed or changed.

## One-command automatic start/continuation on 2026-09-28

Removed the newly introduced shell `resume` mode at the author's request.
The ordinary `sh scripts/run_srgc.sh math|mbpp` command now initializes a
cohort only when none is active, otherwise rejoining the active cohort with
its existing cache/checkpoints. It enables failed-task retries within the
unchanged three-attempt limit. Live task/device leases, intentional stops and
protocol mismatches are still respected; existing work is never silently reset.
Explicit run names remain opt-in. Healthy running workers need not restart:
their per-prompt cache saving and five-/25-update checkpoint saving were
already enabled. The separate rolling-backup feature remains intact.

All 145 unit/integration tests passed without skips. New regressions check
first start followed by a failed-task restart with byte-identical saved cache,
checkpoint and queue protocol, reuse of a nondefault active cohort, exclusion
of a live task, and rejection rather than replacement of an incompatible
active run. Shell and worker-entry tests cover automatic routing/retry and
removal of the shell resume mode. Shell syntax and whitespace checks passed.
The experiment implementation digest remains
`1869fe1cf898d4ff3a6d5e9054790836442b5e0b81b485fb04bc27de4ebab20a`.
No running H100 experiment, saved checkpoint or node process was modified.

## External checkpoint backups on 2026-09-28

Added a group-volume checkpoint watcher outside the frozen experiment package.
New worker launches start it automatically; existing workers can keep running
while a separate `backup-watch` command monitors their published checkpoints.
It retains two observed versions per checkpoint, uses a shared backup lease,
copies from an open descriptor to survive concurrent atomic replacement, and
verifies SHA-256 plus ZIP CRC before publishing. Invalid new checkpoints do
not displace earlier valid copies. Receipts record the source run identity,
file signatures, digest and copy duration. The original checkpoint is never
rewritten by backup. Copies remain on the same group volume, not off-volume.

`sh scripts/run_srgc.sh DATASET resume` rejoins the active cohort and permits
failed-task retries within the existing three-attempt limit. It does not
reset inputs, clear an intentional stop, bypass identity/device leases or
automatically restore a damaged checkpoint from backup. The underlying saved
model/optimizer/selector resume behavior and save intervals are unchanged.

All 142 tests passed without skips, including atomic replacement while copying,
two-generation retention, corrupt-file refusal, identity/path guards, lease
exclusion, existing-queue invariance, worker-entry automatic backup, and real
PyTorch tensor/optimizer checkpoint readback. The backup CLI also passed with
Torch, Transformers and PEFT imports blocked; the existing package's NumPy
dependency remains. Shell syntax and whitespace checks passed. Before and after
this addition, the experiment implementation digest is unchanged:
`1869fe1cf898d4ff3a6d5e9054790836442b5e0b81b485fb04bc27de4ebab20a`.
No H100 worker or its actual checkpoint files were accessed, stopped or modified.

## Random candidate-40 training and cache recovery on 2026-09-28

The latest author instruction supersedes global non-repeating training passes.
Both MATH and MBPP now draw 40 distinct random candidates from the full 400:
Random takes random four within that draw; SR takes its cached-score top four;
On-policy keeps its existing gradient top-four selection and 25-update refresh.
Random/SR redraw each update; prompts may recur across draws. Switch uses the
same new SR training rule from the transition update onward. SR-GC remains
the original 40-vs-40 diagnostic, separate from training selection. Candidate
and selected IDs are logged, and CPU sampling/ranking is inside measured
training time. Old sampling checkpoints cannot silently resume as new runs.

The shell launcher uses a new shared `candidate40-v2` cohort, avoiding the
old queue's implementation mismatch without weakening identity guards or
deleting old work. Both workers of a dataset join the same new cohort. Group
storage, response/cost receipts and existing interpreter selection remain in
place. Cache decode progress now reflects actual completed token steps, and
verification progress reflects completed responses, not synthetic heartbeats.
A restart with every response receipt saved exports without reloading 7B or
regenerating responses. Missing interrupted timings are not reported as zero.

All 131 unit/integration tests passed without skips. Tests exercise the real
MATH and MBPP input IDs for all five shipped seeds, using synthetic rewards
only in memory; no input bundle or scientific outcome was modified. They also
verify distinct candidate draws, SR within-draw ranking, Random within-draw
sampling, deterministic resume, unchanged diagnostic size, post-switch SR,
old-protocol rejection, progress throttling, export-only cache recovery and
unchanged tiny-OLMo sampled responses with/without the progress callback.
The two-process CPU smoke test also passed distributed update/timing agreement
and startup/write-failure propagation. Shell syntax and diff whitespace checks passed. H100 execution and the cause
of the user's stalled remote node remain unverified without its logs/access.

## Restore 40-vs-40 SR-GC and expose cache progress on 2026-09-28

The author's latest instruction retains the original 40-vs-40 SR-GC contrast,
separate from the four-prompt training batch. It compares all 40 randomly
sampled On-policy candidates with the highest-scoring 40 unused SR prompts.
The intermediate selected-four contrast below is superseded. Random/SR
non-repeating passes and their persisted used-prompt state remain in place.

Worker child logs are relayed live instead of appearing only in task files.
Cache status reports saved prompt receipts, exported count and last-write age.
All large runtime/compilation caches and temporary files are group-local;
symlink escapes for response caches are rejected, and exact per-seed response
and cost paths are printed before GPU work. Per-prompt rank-local partial
cost snapshots preserve completed generation/verification/write stages before
whole-phase completion. They are not added to finalized costs a second time.

The actual stopped node and its last error output have not been supplied.
These verified local defects do not establish the cause of that node's stall;
no remote recovery or H100 run is claimed. Queue identity/lease protection
remains enabled rather than mixing existing training under a changed protocol.

All 125 unit/integration tests passed with no skips. The two-process CPU
distributed smoke test passed both timing/update agreement and failure
propagation checks. New regressions cover live child output, partial cache
receipt status, group-local compilation/temporary caches, cache symlink
escape refusal and durable partial cost snapshots without double counting.

## Full-pool sampling and selected-four SR-GC on 2026-09-28 (Superseded Contrast)

The author requested removal of the initial fixed 40-prompt Random pool,
SR selection in cached score order while excluding already trained prompts,
and the same unused top-four SR proposal in SR-GC. Random now consumes a
seeded permutation of all candidates; SR consumes score order. Each pass
excludes previously trained prompts, including the shared prefix. A new pass
starts only after the entire pool is exhausted. This retains the requested
training horizon without pretending that 400 prompts can supply 1,000 unique
training slots. On-policy's 25-update retained-batch rule is unchanged.

SR-GC compares the four selected On-policy prompts against the next unused
SR four. Scoring requests the 40-candidate/four-SR union, at most 44 prompts,
plus the existing single validation reference. A comparison alone does not
consume SR prompts. On transition, training uses the same proposed SR batch.
Actual used IDs and pass number are saved only after successful training and
persist through checkpoint resume and prefix forks. Old sampling checkpoints
are rejected; implementation-hash protection remains enabled.

All 120 unit/integration tests passed with no skips using the existing isolated
CPU test environment, including tiny OLMo tests. New controller tests cover
the four-vs-four inner product, the 44-prompt bound, full-pool exhaustion,
SR ordering, exclusion of prefix prompts, comparison without consumption,
transition-batch identity, resume, partial pass boundaries, failed training,
and rejection of legacy/invalid sampling state. No H100 job, historical result,
input bundle or pretrained-model environment was changed by these checks.
The synthetic CPU demo completed all four arms through update 275 for seeds
5 and 6. The two-process CPU smoke test also passed: distributed global-batch
updates and unequal-rank timing match the single-process reference, and
rank-zero startup/timing-write failures propagate without hanging. These are
software checks, not H100 throughput measurements or empirical paper results.

## Simple shell launcher on 2026-09-28

Added the POSIX-compatible `scripts/run_srgc.sh` entry point. Users select only
`math` or `mbpp`, optionally followed by `status`, `results` or `costs`.
Existing Python selection, bounded CPU thread pools, scheduler GPU visibility
and the shared fresh-run name are handled inside the launcher. No experiment
runtime module, cache format, plan or input changed in this update.

`sh -n` and all four launcher tests passed. Tests execute the real shell script
from another working directory with an instrumented Python executable, covering
both datasets, paths with spaces, GPU visibility, reports and invalid arguments.
No H100 experiment was launched by these checks.

## Fresh group-volume start on 2026-09-28

The author requested a clean start instead of preserving/migrating old caches.
`worker --fresh NAME` now stages only prompt inputs into a new group-volume
cohort, clears cached rewards/provenance and ignores old queue/migration state.
Publication is atomic under the storage lock. Two simultaneous CLI processes
were tested to join the same cohort, and a repeated invocation preserves new
work instead of resetting the second node's cache. Default reports follow the
active fresh cohort. Existing files are not deleted.

Cache-only attention changed from eager to PyTorch SDPA; the backend is bound
in cache provenance. Training/scoring retain eager attention. Per-rank cache
progress now reports completed prompts, last-prompt seconds and estimated
remaining time. This runtime change requires the requested fresh cohort;
old implementation hashes are not bypassed. H100 speedup is not measured here.
All 106 unit/integration tests passed with no skips, including actual tiny
OLMo SDPA cache generation (eight responses and repeatable sampling), clean
start despite old partial migration state, and two-process shared startup.

## Group-volume storage repair on 2026-09-28

The launcher now routes runtime inputs/cache and checkpoints/results to group
storage when the code checkout is on a user volume. A missing group mount,
user-volume destination or symlink escape is rejected. Existing group-local
plans retain their paths. Model/dataset library cache writes are redirected
to group storage. The launcher and its storage helper are outside the frozen
`srgc_rebuttal/*.py` implementation hash; no scientific runtime module, plan
or existing input is changed by this repair.

Added tests cover fresh staging, unchanged plan/input/implementation hashes,
resuming an existing queue with copied real-format prompt receipts, live-worker
and held-lock refusal, path guards and the CPU-only storage CLI. Migration
preserves originals and refuses silent replacement of an existing destination.
Actual node filesystems and running experiments are not available locally;
the caller must stop all dataset workers before migration.
All 103 unit/integration tests passed with no skips. The six storage tests
also passed separately, and the runtime package modules match `fcf64cd`
byte-for-byte, preserving their implementation digest for resumed work.

## MBPP input packaging repair on 2026-09-28

Earlier dataset tests generated temporary fixtures but did not check that the
actual deployment contained MBPP input files. The default plan consequently
failed with FileNotFoundError if the separate preparation step had not succeeded.

The five real `mbpp-seed-5.json` through `mbpp-seed-9.json` bundles are now
included in Git, with explicit ignore exceptions and a source/hash manifest.
They were generated from all 974 rows of the pinned full MBPP release using
the existing preparation command, verified both offline and against the pinned
Hub dataset. Splits remain 400/100/300 with 50 online validation prompts.
Cached rewards remain empty for GPU generation; no results were fabricated.

The new packaging regression checks the actual five files, plan and input
hashes, source provenance, split validation and queue initialization without
dataset downloads or preparation. Runtime Python, plan settings and existing
MATH inputs are unchanged. No H100 workload was launched.
All 97 unit/integration tests passed with no skips. The real MBPP
`plan --check-inputs --allow-pending-cache` command also passed using Python
without the `datasets` package installed.

## Existing-runtime repair on 2026-09-28

The additional-study entry point now reuses Pair/MBPP interpreter selection,
the operational OLMo runtime gate and local model loader, and the existing
offline verifier bundle. SSH workers use the same entry point. Exact new
Transformers/PEFT pins no longer reject an otherwise compatible working
environment. No existing experiment environment was modified or reinstalled.

All 96 unit/integration tests passed with no skips. The six added regression
tests cover interpreter selection, existing runtime/model-loader delegation,
local snapshot provenance, and guarded recovery of an admission-only queue.
Queues with tasks, cache receipts, checkpoints or live workers still reject
implementation changes. The two-rank CPU distributed smoke also passed both
global-update/timing parity and failure propagation without hanging.

Verification used temporary CPU tooling with Transformers 4.57.6 and PEFT
0.20.0; reuse of other compatible versions is covered by mocked compatibility
tests, not a claim of GPU validation on every version. The user's exact
Transformers traceback was not supplied. No H100 training or scientific
result was produced by these tests. Existing inputs and protocol are unchanged.

## Master publication on 2026-09-28

The three experiment commits through `937b949` were fast-forwarded onto
`master` at the author's request. Current execution guides now point to
`master`; the branch names below describe historical verification runs.
No experiment protocol, input data or historical result was changed by
this publication. User-owned untracked files were left untouched.

The full 90-test suite passed again on `master` with no skips, using CPU
PyTorch, Transformers 4.57.6 and the deployment-pinned PEFT 0.20.0. The
runtime code, tests and input bundles match `937b949` exactly; only branch
guidance and repository instructions changed. No H100 workload was launched.

## Node-parallel and MBPP verification on 2026-09-27

Branch: `experiments/srgc-cost-replication`. Changes are confined to the code
repository. The manuscript repository and scientific outcomes are unchanged.

- 76 unit/integration tests passed with no skips, including the tiny OLMo-3
  generation, gradient, optimizer and timing tests from the migration suite.
- Two spawned CPU worker processes executed all 30 synthetic cache/prefix/arm
  tasks exactly once and respected per-seed dependencies. Four workers also
  passed the existing 25-task already-cached case.
- Tested immutable cache handoff, input-change rejection, completed-task skip,
  bounded failed retries, preserved attempts and abandoned-task recovery.
- Tested partially overlapping GPU-UUID leases and real subprocess inheritance
  of a task lock after the parent descriptor closed. Immediate stop terminated
  the owned test process and released its lease.
- Tested worker-loop orchestration, mocked multi-host SSH preflight/startup
  acknowledgements, failed remote-directory handling and final paired reports.
- Tested MBPP input preparation from 800 synthetic fixture records, consistent
  split across seeds, no overwrite of different data, independent plan/output
  selection and dataset/plan mismatch rejection. Fixtures are temporary test
  data, not experimental inputs or reported results.
- CPU `status --dataset math` reports five ready cache tasks, five prefixes
  waiting for cache and twenty continuations waiting for prefix; no worker was
  launched by this inspection.

Environment: Python 3.12, CPU PyTorch 2.14.0, Transformers 4.57.6, PEFT 0.21.0.
No real remote SSH node, shared cross-node filesystem, CUDA allocation, MBPP
download or 7B experiment was exercised. Actual node access is still required
to validate cluster hardware and measure the new scientific outcomes.

## Code-repository migration verification on 2026-09-27

Branch: `experiments/srgc-cost-replication` in `33modeling/offpolicy-misranking`.
The active implementation, tests and five input bundles are maintained here,
not in the manuscript repository. New CLI: `scripts/run_srgc_rebuttal.py`.

Migration and instrumentation checks:

- 60 unit/integration tests passed, with no skips, including tiny OLMo-3
  generation/scoring/training/evaluation and exclusive timing reconciliation.
- Unequal two-rank CPU scoring shards complete without deadlock; distributed
  four-prompt training matches the single-rank update. Raw rank receipts are
  recorded and collected successfully.
- Tests cover nested timers, straggler accounting, CPU versus allocated GPU
  time, completed retries, interrupted phases/invocations, missing totals and
  Python launcher help from outside the repository.
- Five existing real input bundles were checked against the pinned source
  rows; the preparation manifest was regenerated for the new paths/hashes.
  Cached rewards remain empty. No model training result was invented.

Environment: Python 3.12, CPU PyTorch 2.14.0, Transformers 4.57.6, PEFT 0.21.0,
math-verify 0.9.0. The deployment requirement pins PEFT 0.20.0; this local run
does not establish GPU performance or exact deployment-package equivalence.
No pretrained weights, real GPU jobs or multi-node experiment were launched.

## Pre-migration verification on 2026-09-27

The author clarified selection refresh to every 25 updates. The new runner
keeps the top four prompts until the next refresh and generates fresh training
responses at each update. V6 stays frozen. This verifies implementation and
input preparation, not the submitted experiments or new 7B outcomes.

Current checks:

- Full unit/integration suite, including periodic selection, mid-block resume,
  shared model/optimizer prefix, actual candidate IDs, reused 40-vs-40 D,
  both temporal conditions, split leakage, queue leases and complete paired
  summaries. Exact final counts are recorded in the review report.
- Tiny OLMo-3 generation, dense scoring gradients, GRPO/RLOO LoRA updates,
  micro-batching, chunked-head gradients and optimizer restore.
- Two CPU distributed processes: gathered projected gradients and the global
  four-prompt optimizer update match a single process.
- Durable phase receipts preserve completed retry costs. Open timers mark
  measurement incomplete and the summary omits a complete cost total.
- Cache resume uses stable prompt-specific seeds and atomic response receipts;
  source/model/settings changes and duplicate imported rollout indices fail.
- Five real MATH-train bundles: 400/100/300 disjoint splits, 50 explicit online
  validation prompts, all 800 selected gold answers self-verify. Input hashes
  match prepared_inputs.json. Pending caches are rejected by training checks.
- Three dataset plans retain selection_interval=25. All 271 submitted V6 files
  match archive 3d144ee666e480766e253d06221f6e29feaef25e.

Test environment: Python 3.12, CPU PyTorch 2.13.0, Transformers 4.57.6,
PEFT 0.20.0, huggingface-hub 0.36.2, math-verify 0.9.0. The same numerical
suite also passed with the local Transformers 5.14.1 installation before
validating the pinned 4.57.6 deployment requirement. No pretrained 7B weights
were downloaded. Prepared real inputs contain no fabricated cached rewards.

Not run: real multi-node GPU work, initial-policy cache generation, additional
7B training seeds, GPU peak memory or throughput measurements. The current
computer cannot use its NVIDIA driver; GPU node access remains necessary.

## Historical verification (2026-09-26, superseded protocol)

The record below describes the original every-update implementation before
the author's interval clarification; it is not the current execution status.

# Verification on 2026-09-26

The source protocol is the V6 manuscript at `e649d2e`. This is a newly written
implementation, not a verification of archived training code or a reproduction
of the paper's reward/cost values.

Passed:

- 30 unit/integration tests: full 40-vs-40 D, overlap cancellation, cosine
  versus inner-product behavior, LOO subgroups, GRPO token normalization,
  temporal boundaries and missing checks, per-step scoring, no extra D model
  call, stopping scoring on switching, shared-prefix and optimizer restoration,
  resume determinism, split leakage, complete paired-seed summaries, and tiny
  OLMo-3 fresh generation/dense gradients/LoRA GRPO and RLOO updates.
- Four spawned CPU workers: all 25 synthetic queue tasks executed exactly
  once, independent work overlapped, and each continuation waited for its own
  prefix. Restart skipped completed work. Failure/retry, duplicate leases,
  changed-input rejection and recovery after final-endpoint publication passed.
- Tiny OLMo-3: micro-batches of one and four yield matching scoring gradients
  and optimizer updates; eight active responses use four batched forwards at
  micro-batch two, with only one projection per parameter per prompt. Chunked
  LM-head log-probabilities and derivatives agree with full model forwards.
- Summaries reject mixed implementation and prefix hashes across paired arms.
- Two-process CPU distributed test: gathered per-prompt gradients and the
  global four-prompt update agree with the one-process implementation.
- math-verify adapter: a correct boxed answer receives 1 and an incorrect
  answer receives 0.
- CPU categorical-policy demo: two seeds, four arms each, common prefix 25,
  total update 30; output explicitly labeled synthetic and kept outside Git.
- Python compilation of all implementation modules.
- Additional-seed plan: seeds 5–9, four arms, shared prefix 25, endpoint 275;
  single-node and SSH worker commands generated without launching experiments.
- V7 manuscript: 94 existing tests, all artifact generators, source/PDF/text
  bindings and LaTeX build passed (9 main pages, 29 total). All 271 submitted
  V6 files match the final archive baseline `3d144ee`; V7 TeX/figure sources
  retain that manuscript without new scientific changes.

Test environment: Python 3.12, NumPy 2.5.2, PyTorch 2.14.0+cpu,
Transformers 4.57.6, PEFT 0.21.0, math-verify 0.9.0. Dependencies were installed
in a temporary isolated environment. No pretrained model weights were downloaded.

Not run: remote-node execution or the four-GPU OLMo-3 7B experiments. The five real input bundles are
not present. Their exact prompt formatting, cached rewards, verifier extraction
settings and online ranking-validation IDs must be supplied before execution.
No new experimental result or measured 7B runtime is claimed. Cross-node
filesystem locking and hardware peak memory require cluster-level validation;
local multi-process tests are not a substitute for that validation.
