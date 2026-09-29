# Multi-node execution for rebuttal seeds

## Simple shell commands

Run `sh scripts/run_srgc.sh math` on the two
MATH nodes and `sh scripts/run_srgc.sh mbpp` on the two MBPP nodes. No Python
flags, migration command or preparation command is needed. Each node needs
four allocated H100s. Existing scheduler GPU visibility is preserved.
Leave healthy running workers alone. After a node failure, repeat the same
command on the replacement allocation; there is no separate continuation mode.
The launcher selects the existing Pair/MBPP Python and group-volume storage.
With no active cohort it initializes the shared `candidate40-v2` cohort once.
With a matching implementation identity it rejoins the active cohort and reuses
saved progress; an implementation change starts/joins a separate cohort while
preserving the old run. The same command handles both cases automatically.
The shell enables failed-task retries with a 120-second delay and a cumulative
50-attempt limit, configurable with `SRGC_MAX_ATTEMPTS`.
Live task/device leases prevent duplicate execution. Old files remain untouched.

Use `sh scripts/run_srgc.sh math status`, `sh scripts/run_srgc.sh math results`
or `sh scripts/run_srgc.sh math costs`; substitute `mbpp` for the other dataset.
These reporting commands do not start workers or create a fresh cohort.

`status` separates `current_step` from `completed_steps`, with the task's
planned denominator and assigned node. Cache progress is labeled `CACHE
available_prompts=N/400`; decoding/prompt counts are never optimizer steps.
Available cache counts include exported bundles even when separate per-prompt
receipts were not imported. `saved_prompts` and `exported` remain separate.

The status adapter reads the current attempt's existing phase receipts, so
updating and running `status` does not require restarting a live worker. It
reports selection/training in progress, completed updates, checkpoint I/O and
evaluation separately. An update count is not a claim that a checkpoint is
already saved. Older attempts are excluded after a restart; stopped attempts
show `last_attempt_step`, not an actively running step. Missing evidence is
shown as `-`, not an inferred zero. Completed updates do not imply completed
evaluation or a published reward. No model/checkpoint tensors are loaded.
JSON status retains the old numeric `step` (last observed completed updates)
and adds the explicit progress fields. `results` retains the existing reward,
cost, JSON/CSV export and complete-seed averaging rules.

Training logs distinguish the shared 25-update prefix from each continuation:

```text
TRAIN seed=5 phase=shared-prefix arm=on_policy step=1/25 status=running completed=0/25
TRAIN seed=5 phase=shared-prefix arm=on_policy step=1/25 status=completed completed=1/25
TRAIN seed=5 phase=shared-prefix arm=on_policy step=25/25 status=completed completed=25/25
TRAIN seed=5 phase=continuation arm=sr step=26/275 status=running completed=25/275
```

`running` is printed before selection/rollout work starts for that optimizer
update; `completed` advances only after the update finishes. Resumed workers
show their restored step rather than restarting the counter. Only rank zero
prints these lines. Cache `prompt N/100` counts are not training steps. These
logs apply to training processes newly launched through the updated worker;
already-running training processes do not reload this code automatically.

## Checkpoint Backup and Resume

Checkpoints live on the group volume. Newly launched ordinary shell workers
save after **every completed optimizer update**, for both the shared prefix
and all four continuations. A node failure resumes the last completed save,
including model/optimizer and selector state, and redoes the unsaved update.
This does not preserve a partially completed rollout, gradient computation or
update. Completed cache prompt receipts are reused. Completed endpoints are
skipped. Loss of an allocated node does not mean loss of the group volume.

Already-running processes retain their old cadence (prefix: every five
updates; continuations: every 25). Pulling this change or starting a backup
watcher cannot change a live training process. To apply the shorter interval,
update the checkout and restart the worker with the same ordinary shell command.
Stopping it before its next old-cadence save discards the currently unsaved
work, so prefer restarting just after a save when the allocation permits.
No cache reset or new cohort is needed.

`scripts/srgc_step_checkpoints.py` adds storage-only saves between the frozen
runner's original boundaries. Selection, training and identity guards remain
unchanged; checkpoint metadata records the interval and storage adapter SHA-256.
Extra snapshot/write time is recorded in the existing `checkpoint_save` cost
ledger. Intermediate saves are atomic, flushed and synced before publication.
The low-level direct `srgc_rebuttal.run_experiment` command retains its original
cadence; the per-update policy applies through the ordinary shell worker.

New worker launches also run a CPU-only backup watcher every 30 seconds. It
copies published `.pt` files, not temporary writes, keeps the last two observed
versions per checkpoint, and verifies SHA-256 and PyTorch ZIP integrity before
publishing a copy. An invalid new file never replaces a prior valid backup.
Two nodes coordinate through a separate backup lease. Backups are under the
active run's `checkpoint-backups/seed-N/CHECKPOINT/SHA256/`, with `checkpoint.pt`
and an identity/checksum receipt. They are independent copies on the same group
volume, not protection against loss of that entire volume. A 30-second poll
may miss an intermediate save, and a node failure before copying leaves the
ordinary latest checkpoint as the newest recovery point.

For workers already running before this launcher update, leave training alone
and start one watcher per dataset in separate persistent terminals:

```sh
sh scripts/run_srgc.sh math backup-watch
```

```sh
sh scripts/run_srgc.sh mbpp backup-watch
```

`backup` instead of `backup-watch` performs one immediate pass and exits.
Watching is CPU-only and neither changes queues/inputs nor starts training.
The initial scan, state changes, new copies, errors and final scan print
`BACKUP CHECK` with UTC time, files found, new copies (`saved`), unchanged
files and errors. Repeated unchanged scans are silent. `waiting_for_checkpoint`
means no published checkpoint exists yet; `no_new_checkpoint` means the
existing files have already been backed up. These are watcher status lines,
not claims of training progress or new copies every 30 seconds. Each scan
is followed by a 30-second wait; copying time is additional.
Ordinary workers also print `WORKER` activity independently of the backup
watcher, on task/state changes and every 30 seconds while waiting or running.
Running lines identify the seed/task, attempt, child PID, latest phase receipt,
each rank's last observed work and the task log. Idle lines list queue states
and the tasks awaiting dependencies or another worker's lease. These lines
read existing receipts; their appearance alone does not count as progress or
reset the stall watchdog. `phase=child_startup` means no phase receipt for this
attempt has been written yet, not that model loading has been verified.
If only backup messages are visible on an older worker, use
`sh scripts/run_srgc.sh math status` (or `mbpp status`) in a second terminal
without stopping it. Updating the checkout does not hot-reload worker logs.
The experiment package/implementation hash is unchanged by this addition.
Watchers bind to the active cohort at startup; restart the watcher when
explicitly selecting a different cohort. Backups copy the existing checkpoint
contents; they do not reconstruct missing timing measurements after a crash.

After a node failure, use the existing group volume and unchanged code/plan:

```sh
sh scripts/run_srgc.sh math
# On an MBPP node instead:
sh scripts/run_srgc.sh mbpp
```

There is no separate shell resume mode. The ordinary command detects the
active cohort, reuses caches/checkpoints and retries failed tasks within the
existing three-attempt limit. It does not
override an intentional queue stop, exhausted retries, live GPU leases or
identity mismatches. Normal resume loads `*-latest.pt`; it does not silently
replace a damaged latest checkpoint with a backup. Restoring a backup requires
stopping the affected task, verifying its identity/receipt, and explicitly
replacing that task's checkpoint before resuming.

Run one four-H100 worker on each allocated node (full H100 GPUs with at least
75,000 MiB each). Two nodes, eight GPUs total, are sufficient. Workers share a queue and take
independent tasks as soon as dependencies finish. No cross-node gradient
all-reduce is needed: the four replicas of one task stay on one node.

The fixed plan has five new seeds (5–9). Each seed generates its missing cache,
runs a shared 25-update On-policy prefix once, then four arms continue to total
update 275. This is five cache tasks, five prefix tasks and twenty continuations.
Existing verified caches and completed tasks are skipped. A seed can proceed
as soon as its own dependencies finish; other seeds need not finish first.

The normal worker releases all four continuations as soon as `prefix.pt` and
`prefix-ready.json` validate the common 25-update model/optimizer checkpoint.
It does not wait for the prefix producer's process teardown or queue lease to
close. A bare `prefix-latest.pt`, an incomplete save or an invalid identity/hash
does not release any arm. Cache handoff still waits for its producer to finish.
Random, SR, On-policy and Switch have no dependencies on one another; seed
ordering is a claim preference, not a barrier. Task and GPU leases still prevent
duplicate work or overlapping GPU allocations.

With `all run`, MATH and MBPP initially provide five independent cache/prefix
chains each: at most ten useful concurrent tasks until prefixes become ready.
After publication, up to forty continuations can run across both datasets on
distinct four-GPU allocations. This is a dependency limit, not a ten-node cap.
Workers check for newly ready work every two seconds by default; an explicit
`--poll-seconds` overrides this. Multi-queue idle logs show both datasets and
task logs read progress from the dataset actually running.

To load scheduler changes, update and restart only idle workers; leave workers
currently training alone. These scheduler changes preserve experiment code
identity, caches and checkpoints. Do not start another worker on GPUs still
leased by an old idle worker.

## MATH or MBPP

For the first start or for continuing after a node failure, use these same
commands on the respective node pairs. Do not stop a healthy running worker
just to enable continuation; checkpoint saving is already active:

```bash
# MATH nodes 1 and 2.
sh scripts/run_srgc.sh math
# MBPP nodes 3 and 4.
sh scripts/run_srgc.sh mbpp
```

`--fresh` initializes one shared queue per dataset/run name under
`$OM_WORK/srgc-rebuttal/fresh/`, using the prompt/split inputs. Locally generated
reward caches start empty; complete historical Pair cache imports retain their
recorded provenance. It does not migrate old checkpoints or queue records.
The second node joins that queue instead of resetting it. Repeating the same
name resumes that fresh cohort; choose another name for another clean start.
Default status/results commands follow the selected fresh cohort. Old files
are ignored, not deleted. No claim of faster measured H100 throughput is made.
Initial cache generation now uses PyTorch SDPA, records that backend in cache
provenance, and logs per-rank completed prompts and estimated remaining time.
While generating, real completed decode steps are reported every 30 seconds,
starting with the first generated token; each verified response also updates
progress. A stuck model generates no artificial heartbeat. If all response
receipts were saved before interruption, restart exports them without loading
the model again. Training/scoring attention and eight-response cache settings
are unchanged. Random/SR training sampling follows the revised protocol above.

The Python entry point accepts `--dataset math` (default) or `--dataset mbpp`
before or after the command. Both datasets' real seed-5--9 input bundles are
included in the checkout. Run the worker on each allocated node:

```bash
# Run this on node A and node B, each with four allocated GPUs.
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/run_srgc_rebuttal.py worker --dataset mbpp

# The prepared MATH inputs can use the same automatic cache/training queue.
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/run_srgc_rebuttal.py worker --dataset math
```

Choose one dataset per allocation. Do not run both workers on the same GPUs.
For concurrent MATH and MBPP with two workers each, allocate four four-H100
nodes (sixteen GPUs). Run the MATH worker on nodes 1 and 2,
and the MBPP worker on nodes 3 and 4. With only two nodes available, one worker
per dataset is also supported, with less concurrency within each dataset.
The entry point routes the source plans to group storage; use the active run
paths printed by `status` rather than the checkout's relative run paths.
Inputs, cache receipts, checkpoints, task leases and result reports are
separate; device leases are shared between the built-in datasets.
`--plan` selects a custom cohort; an explicitly
conflicting `--dataset` is rejected. Two four-GPU nodes are supported; the
queue also works with one or more nodes. No run-duration estimate is implied.

| Available nodes (four GPUs each) | Scheduling |
| --- | --- |
| 1 | Executes all tasks in sequence, reusing each prefix |
| 2 | Runs two ready tasks concurrently, automatically taking the next task |
| 5 | All five prefixes can run together, then five continuations at a time |
| 10 | Up to ten ready continuations at a time |
| 20 | Up to all twenty continuations after their respective prefixes finish |

Five nodes are an optional faster allocation; more nodes shorten the
continuation queue. Prefix work initially has only five independent tasks.
Actual speedup depends on rollout lengths, hardware and the longest arm;
no measured 7B speedup or completion time is claimed.

## Prepare the shared run directory

### Group-volume storage

Large artifacts must stay on group storage, not a user-volume checkout.
The Python entry point stages byte-identical plans and inputs under
`${SRGC_STORAGE_ROOT:-$OM_WORK/srgc-rebuttal}`. The default `OM_WORK` is
`/group-volume/${OM_USER:-minsoo3.kim}/offpolicy-misranking`.
Both the root and resolved input/output paths must be inside `GROUP_VOLUME`
(default `/group-volume`). A missing group volume is an error, not permission
to fall back to the user volume. If the original plan and all artifacts are
already on group storage, their existing locations are retained.

On a user-volume checkout, cache receipts now live in
`$OM_WORK/srgc-rebuttal/inputs/{seed-file}.cache/` and checkpoints/results in
`$OM_WORK/srgc-rebuttal/runs/{additional-seeds,mbpp-seeds}/`.
The launcher prints both resolved locations before starting work.

For an experiment already generating caches in the old checkout, first stop
all four workers with Ctrl+C and wait for their child processes to exit.
After updating the checkout, run once per dataset:

```bash
python scripts/run_srgc_rebuttal.py storage --dataset math --migrate
python scripts/run_srgc_rebuttal.py storage --dataset mbpp --migrate
```

Then restart the same worker commands on the two nodes assigned to each dataset.
Migration refuses live worker records or held task/execution leases. It copies
completed prompt receipts, timing records and checkpoints without deleting
originals or changing plan/input/implementation hashes. Finished prompts are
reused on restart; an interrupted prompt may need regeneration. Original
interrupted timing receipts remain incomplete, not invented measured totals.
Do not restart an old launcher against the preserved original paths. If a
dataset was stopped with the persistent `stop` command, use `resume --dataset`
after migration before restarting its workers.

Use a fixed checkout of `offpolicy-misranking` branch
`master` on storage accessible at the **same path**
on every node. Inputs, outputs and locks must be shared too. The filesystem
must implement coherent POSIX `flock` across nodes and atomic rename; local
scratch copies with separate lock directories cannot coordinate these workers.
The automated multi-process tests exercise local filesystem locking, not
your cluster's filesystem. Use the site's validated shared-lock filesystem.

Reuse the working seed-3/4 environment on each node. The entry point selects
`PAIR_PYTHON` for MATH or `SWITCH_PYTHON` for MBPP, otherwise the existing
`${VENV_DIR:-$OM_WORK/.venv-cu126}/bin/python`. Do not reinstall its packages.
The runtime reuses the operational OLMo compatibility check, model loader and
offline Math-Verify bundle. Weights resolve from `OM_OLMO3_MODEL_PATH`, the
existing `MODELS_DIR` snapshot or the pinned local Hugging Face cache;
workers do not download model weights. Use the same packages and GPU type
for comparable measurements.
Do not edit Python code, plan or inputs while the queue is active.

Prepare the five real input bundles described in [README.md](README.md), then:

```bash
python scripts/run_srgc_rebuttal.py plan --dataset math --check-inputs --allow-pending-cache
```

The real MATH-train input bundles for seeds 5–9 are prepared in `inputs/`.
The corresponding `inputs/mbpp-seed-{seed}.json` files are also included.
Their pinned source revision and file hashes are recorded in
`experiments/prepared_mbpp_inputs.json`; no dataset download or `datasets`
installation is needed to start these prepared MBPP workers. `prepare` remains
available for explicitly rebuilding inputs or preparing a custom plan.
See [REBUTTAL_READY.md](REBUTTAL_READY.md) for the fixed split, explicit 50-prompt
online validation set, cache-generation commands and preparation manifest.
The queue now generates missing initial-policy rewards itself, on different
nodes in parallel. Cache completion atomically freezes the final input hash
before that seed's prefix can start. Only cached rewards and their provenance
may change during this handoff; prompts, splits, plan and code stay immutable.
After handoff, rewards are immutable too. Cache time receipts are retained,
including interrupted/incomplete measurements; missing time is never zero.

## Start workers

From the shared repository root on each allocated node:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/run_srgc_rebuttal.py worker --dataset math
```

Replace the device list with that node's four allocated GPUs. The worker
checks their H100 type, memory capacity and occupancy, holds an exclusive
device lease, uses the existing runtime compatibility check, and runs the existing
`scripts/selection_nccl_preflight.py` four-rank CUDA/NCCL/DDP probe **before
claiming any task**. Failure blocks training. Only probe-verified runtime
workarounds are inherited. Evidence and separate admission GPU-time receipts
are in `.queue/admission/`. Manual `run` and `cache` use the same gate.
It then starts four local processes with `torchrun`. Locks are per physical GPU UUID, so partially
overlapping GPU groups also conflict. It does not stop unrelated GPU jobs. Use the same
`--node-lock-root` for different queues sharing the same devices; its default
is `srgc_rebuttal/runs/gpu-node-locks` next to the output cohort directory.

For SSH-accessible allocated nodes, first print the commands without launching:

```bash
python scripts/run_srgc_rebuttal.py commands --dataset mbpp \
  --hosts node-a node-b \
  --repo /shared/offpolicy-misranking \
  --python /shared/env/bin/python
```

After replacing those placeholders with real nodes and paths, change
`commands` to `launch` to start all workers. Launch first checks every node's
code/plan/input identity, common queue visibility, cross-node file-lock
exclusion, available CUDA devices, GPU model and package versions. All nodes
must pass before workers are started. It then waits for each worker's startup
receipt, not just a successful SSH connection. Inspect `.rebuttal-worker-logs/`
if a node fails startup; acknowledged workers keep running if another fails.
This SSH mode uses each remote
environment's `CUDA_VISIBLE_DEVICES`, or devices 0–3 when unset. For scheduler
allocations, start `worker` within each allocation so its GPU visibility is
preserved. `launch` starts workers only; it does not wait for training to finish.

## Progress and restart

```bash
python scripts/run_srgc_rebuttal.py status --dataset mbpp
python scripts/run_srgc_rebuttal.py status --dataset mbpp --watch
python scripts/run_srgc_rebuttal.py results --dataset mbpp
python scripts/run_srgc_rebuttal.py costs --dataset mbpp
python scripts/run_srgc_rebuttal.py summary --dataset mbpp
```

Per-seed outputs are under the selected dataset's output root.
Queue receipts, every attempt, worker heartbeats and per-task append-only logs
are under the cohort's `.queue/`. Status lists task counts, node ownership,
active tasks, heartbeat ages and recoverable work. Heartbeat age is diagnostic:
it never authorizes takeover while a task lease is still held. The last worker
also writes `results-summary.json` and `cost-comparison.json` at the output root.
Each seed's prefix has a checksum receipt; all arms record that same checksum.
Task and execution locks protect against duplicate workers and overlapping
manual starts. Code, plan and input hashes reject incompatible resumes.

Child output is now relayed to the worker terminal while remaining in the
append-only task log. Cache messages identify model loading, generation start
and completed prompts per rank. The shell `status` command also shows raw saved
prompt receipts out of 400, exported reward count and time since the last write.
Saved receipts can increase while exported rewards remain zero until final
export. These diagnostics do not claim that a live stalled node has recovered,
and do not disable code/plan/cache compatibility or active process locks.
Every write launch prints each seed's exact response-cache and cost directories.
Hugging Face, Torch, TorchInductor, Triton, CUDA compilation caches and temporary
files are explicitly routed to group storage. A response-cache or runtime-cache
symlink escaping the group volume is rejected before GPU work starts.
Per-prompt live cost snapshots are written under each seed cache's `live-costs/`;
completed synchronized phase costs remain under `cost-receipts/`.

The next node can resume a task from its saved checkpoint after its old
process and lock have gone away. New ordinary shell workers save every update;
already-running legacy workers retain their five-/25-update cadence until
restarted. Completed tasks are skipped before loading
a model. Explicit failures are retained for inspection; after resolving an
environmental failure, restart a worker with `--retry-failed`. Retries are
bounded by `--max-attempts` (default 3) and spaced by `--retry-delay` (default
60 seconds). Abandoned and interrupted attempts count toward the same limit.
Other ready work can proceed while a failed task waits. A code or input
change requires a new output directory and plan, not an in-place resume.
The sole code-change exception is an admission-only failure before any task,
attempt, cache receipt or checkpoint exists and after all workers have stopped
or failed. That empty queue may adopt the repaired implementation, preserving
its old protocol in `.queue/startup-history/`. Plan/input changes and queues
with actual work remain protected against incompatible resumes.

SIGTERM/SIGINT is forwarded to the worker's own child process group; it waits
for that group to exit before releasing leases. Preserve all attempt logs:
the child launcher inherits task/device lease descriptors, so killing only
the parent worker cannot release an active child's allocation to another
worker. After the child exits and its leases are released, abandoned tasks
are recoverable. Interrupted tasks resume from their own saved checkpoints.
Durable phase receipts count completed repeated work across attempts. An
interrupted timer prevents reporting a complete GPU-time total for that arm.
The summary requires every planned seed and all
four arms, including unfavorable outcomes.

`status` and `results` are CPU-only. `status --watch 10` refreshes every ten
seconds; `--json` gives machine-readable output. Home exports are
`~/srgc-rebuttal-math-status.txt` and `~/srgc-rebuttal-math-results.txt`
(replace `math` with `mbpp` for MBPP). `--output PATH` overrides the text path.
Results also save a JSON sidecar and cohort-local `results.txt`, `results.json`
and `results.csv`. All twenty seed/arm rows are retained. A corrupt receipt
does not hide valid results from other seeds; errors are reported with a
nonzero exit code. Missing costs are null/blank, not zero; means require all
five planned seeds. JSON includes cache, shared-prefix and exclusive stage
costs. CSV includes selection, training, preparation, evaluation, checkpoint
and startup costs. Node admission remains separate research overhead.
Results remain readable after a code change even though incompatible resumes
are blocked. `summary` still requires every seed and arm to finish.
These are new seeds 5--9, not a replacement for the seed-3/4 historical export.
This runner evaluates final rewards at the common terminal update; it does not
yet reproduce the historical per-checkpoint reward curves or plot PDF/PNG
exports. It retains selection/check history and per-stage timing receipts.

Dependencies wait for the predecessor's task lease to close, including its
final timing receipts. The worker stops its owned process group after 1,800
seconds without actual task progress (`--stall-seconds` overrides this).
Worker heartbeats and other tasks cannot reset that clock. Per-rank progress
is in `.queue/progress/`. Stage model weights beforehand: downloading is not
training progress. Nonfinite gradients stop the task. OOM never silently
reduces response counts or changes the protocol. Rank-zero startup,
checkpoint and timing-write failures propagate to other ranks.

CPU tests and a two-process distributed CPU test do not certify H100 memory
headroom, actual NCCL health or cross-node shared storage. The allocated
nodes must pass runtime admission; no real H100 runtime is claimed here.

To stop just the chosen cohort, drain current tasks or interrupt them explicitly:

```bash
python scripts/run_srgc_rebuttal.py stop --dataset mbpp
python scripts/run_srgc_rebuttal.py stop --dataset mbpp --now
python scripts/run_srgc_rebuttal.py resume --dataset mbpp
```

`resume` clears the shared stop flag; start `worker` on the allocated nodes
again (or use `launch`). It does not create a scheduler allocation or SSH host.
No unrelated process is stopped. New implementation versions cannot resume a
queue bound to different code: retain the old checkout for existing runs or
use a new plan/output root. Queue v2 does not rewrite older queue-v1 records.

## Verified scope

Two CPU workers completed the full 30-task cache/prefix/arm DAG exactly once;
four CPU worker processes completed all 25 already-cached training tasks exactly once,
overlapped independent work, and waited for each seed's own prefix. Tests also
cover bounded retries, immutable cache handoff, partial GPU overlap, inherited
child leases, controlled stop, SSH preflight/acknowledgement, final summary
publication, hash rejection and numerical equivalence
of the computation changes. See [VALIDATION.md](VALIDATION.md). No real remote
node or 7B GPU training job was launched while preparing this code.

## Orphaned ranks and stale execution locks

`torchrun` starts each rank in its own session, so a worker that is interrupted
(Ctrl-C, `stop --now`, scheduler SIGTERM) can leave ranks alive after the
launcher is gone. They keep GPU memory and `.<task>.execution.lock` /
`.manifest.lock`, and every later attempt of that task fails at admission or
blocks on the lock while only `BACKUP CHECK` lines keep printing.

`scripts/run_srgc.sh ... run` now applies `scripts/srgc_process_guard.py`
automatically: on worker startup, **before GPU admission and device-lock
acquisition**, and again before each child starts it reaps Python processes of
the same plan that have no live launcher ancestor. After a child exits it terminates
the whole tree it saw while running. To clean a node by hand before relaunching:

```bash
python scripts/srgc_process_guard.py --plan <group-storage plan path> --dry-run   # list
python scripts/srgc_process_guard.py --plan <group-storage plan path>             # SIGTERM, then SIGKILL after 30 s
```

The guard lives outside the hashed `srgc_rebuttal` package, so an existing
queue keeps its `implementation_sha256` and continues without a new root.
Recovery examines only this node's `/proc` and preserves processes with a live
worker ancestor. It does not delete lock files. Multiple nodes continue sharing
one task queue, with device locks keyed by physical GPU UUID rather than local
indices `0,1,2,3`. Restart the ordinary command on an affected stopped node;
healthy workers on other nodes can keep running. A lock still held by a live
worker must continue to exclude a second worker on those same GPUs.

## Worker log format

Every line a worker prints, including relayed child output, is rewritten to
`HH:MM:SS TAG     <task> key=value ...` by `scripts/srgc_log_format.py`:

```
14:02:11 TASK    seed-8.prefix start
14:03:40 CACHE   rank=0 prompts=12/100 last=41.2s eta=60.1min
14:09:02 TRAIN   seed-8.prefix update=1/25
14:09:03 CKPT    seed-8.prefix update=1 saved
14:09:30 BACKUP  state=copied found=1 saved=1 errors=0
14:09:31 WORKER  seed-8.prefix running attempt=1 update=1/25 phase=training:started ranks=4/4 last_activity=3-9s pid=3878952 log=...
14:41:12 TASK    seed-8.prefix done exit=0
```

`WORKER` lines appear only when their content changes (rank ages excluded),
with a heartbeat at most every 10 minutes; `BACKUP` lines only when a copy,
error or state change happens; a step's "running" echo and the per-prompt
"generating" echo are dropped because the following line already reports them.
Anything unrecognised is kept verbatim under `LOG`.

## Gold answers math-verify cannot parse

`ValueError: gold answer could not be parsed; fix input before training` used
to abort a prefix/arm child whenever `math_verify.parse` returned nothing for
a gold (its 5 s alarm expires on long golds when the node is loaded; all 800
golds per seed parse on an idle machine). The training child now installs
`scripts/srgc_verifier_fallback.py`: golds are parsed once with a retry at
60 s, and a gold that still fails is scored by the original experiment's
normalized exact match on the response's last `Answer:` line, logged once as
`VERIFY fallback exact-match gold=...`. Parsable golds keep math-verify.

## One worker for both datasets

```bash
sh scripts/run_srgc.sh all run      # MATH queue first; MBPP tasks whenever MATH has nothing claimable
sh scripts/run_srgc.sh all status   # both status reports, MATH then MBPP
```

`all run` starts one worker per node with `--dataset math --with-dataset mbpp`
(`scripts/srgc_multi_queue.py`). Checkpoint backups and orphan reaping cover
both plans; failed tasks retry automatically. `status` prints a per-seed table
(`done` / `running 3/25` / `waiting` / `failed x3`), the running tasks with
their step and node, the tasks that need attention with their log path, and
the live workers.

The worker writes heartbeats to both queues. Only the active dataset's receipt
names the task; the other report shows `serving <dataset>:<task>` instead of
attributing a same-named task to the wrong dataset. A graceful `stop --dataset`
pauses only that queue, and the worker can keep serving the other one. This also
works when the preferred MATH queue was already stopped before `all run`.
If both queues are complete or stopped, no new GPU admission is performed.

Failure summaries include blocked tasks from both queues. Reading a failure or
status log is bounded to its final 256 KiB (40 lines for failure output, 200
for the status error search), so a large accumulated task log is not loaded
into memory. One process guard covers both datasets and reports each child
failure once. These scheduler/reporting fixes preserve the experiment code
identity and existing checkpoints; load them when restarting an idle worker.

After an incompatible code update, ordinary `all run` automatically preserves
the old run and starts/joins a separate run for the current implementation.
No environment-variable prefix is required. Compatible runs continue unchanged;
plan changes and corrupt queue identities still fail instead of being ignored.
The `[new-run]` message reports both paths, and `automatic-restart.json` records
the transition. Names are deterministic and selection is locked, so concurrent
nodes join the same replacement without resetting progress. Input records come
from the prior active cohort, while generated caches are rebuilt; historical
Pair cache imports keep their existing provenance.

An explicit name remains optional:

```sh
SRGC_RUN_NAME=codefix-20260929 sh scripts/run_srgc.sh all run
```

Use the same name on every allocated node and on retries: it creates or joins
the same new MATH/MBPP queues, never resets them. Old results and checkpoints
remain in their original roots; code/plan identity checks stay enabled. This
is a new experiment, not continuation of the old training trajectory. Stop
workers intended for the old code before updating their checkout.
The shell's existing GPU cleanup can terminate same-user GPU processes; run
on a dedicated free node or set `SRGC_SKIP_GPU_CLEANUP=1` to disable cleanup
and retain normal GPU admission checks.

## SR with refreshed success rates (Limitations: cache refresh)

`scripts/srgc_sr_refresh.py` adds the arm `sr_refresh`: SR's rule (train the
four prompts whose success rate is closest to 0.5), but the success rates are
re-measured under the current policy on On-policy's schedule. Every 25 updates
it draws 40 candidates and generates eight fresh responses each (320 rollouts,
without scoring gradients or validation generation), ranks them by |rate - 0.5| and
trains the top four until the next refresh. `pool` scope re-measures all 400
candidates per refresh instead (3200 rollouts; about one cache build each).

The arm forks from the seed's verified shared prefix in the existing run root
and writes `sr_refresh-{latest.pt,progress.json,endpoint.json}` (or
`sr_refresh-pool-*`) beside the recorded arms; the queue, the four arms and the
hashed package are untouched. Run one seed per node once its prefix is done:

```bash
sh scripts/run_srgc_sr_refresh.sh math 5              # seeds 5..9, candidates scope
sh scripts/run_srgc_sr_refresh.sh math 5 pool         # optional full-pool variant
sh scripts/run_srgc_sr_refresh.sh math results        # rewards and selection GPU-seconds per seed
```

CPU tests: `python -m unittest srgc_rebuttal.tests.test_sr_refresh`.

## Switch with repeated transitions (Limitations: repeated transitions)

`scripts/srgc_switch_repeat.py` adds the arm `switch_repeat`. It starts as the
recorded Switch arm (same SR-GC rule for On-policy -> SR). After transitioning
it keeps scoring the 40-vs-40 contrast at every 25-update check and returns to
On-policy under the mirror rule (two consecutive positive D, or positive /
non-positive / positive with a positive sum); each transition resets the
window, and the batch scored at the returning check is the one trained, so no
scoring is repeated. `endpoint.json` records every transition
(`transitions: [{step, to}]`; `switched_at` keeps the first one).

```bash
sh scripts/run_srgc_sr_refresh.sh math 5 switch_repeat   # one seed per node, prefix must be complete
sh scripts/run_srgc_sr_refresh.sh math results           # rewards, transitions and selection GPU-seconds
```

CPU tests: `python -m unittest srgc_rebuttal.tests.test_switch_repeat`.

The [limitation experiment inventory](../docs/LIMITATION_EXPERIMENTS_KO.md)
records the fixed-schedule control and three other extra arms, execution
prerequisites, cost boundaries and unimplemented controls.
Extra arms follow the same active Pair/prepared cohort
as the main dataset launcher. Their GPU processes set runtime caches on group
storage, and results reject mismatched experiment/prefix identities rather than
combining incompatible endpoints. Missing costs are reported as unknown.
These extra arms are not automatically dispatched by the four-arm queue.

## Fixed-schedule switching control (rule versus a preset transition)

`scripts/srgc_switch_fixed.py` adds `switch_fixed<N>`: On-policy through
update N exactly as the recorded arms (same refreshes and 40-vs-40 scoring, so
its pre-transition cost equals Switch's), no SR-GC decision, then SR from
update N+1 as the recorded Switch does after its transition. N must be a
multiple of the 25-update selection interval; 100 and 125 mirror the recorded
seed-4 and seed-3 transitions.

The boundary refresh at checkpoint N is charged before selecting the SR
training batch for update N+1. The corrected implementation records
`fixed-boundary-before-training-v2` in checkpoints and refuses unversioned
fixed-control resumes. Keep old runs separate; do not relabel their endpoints.
The launcher requires shared-prefix <= N < total updates. Result exports
discover all saved `switch_fixed<N>` arms (including 200) for planned seeds
and apply the same input/code/prefix identity checks to each.

```bash
sh scripts/run_srgc_sr_refresh.sh math 5 switch_fixed100
sh scripts/run_srgc_sr_refresh.sh math 5 switch_fixed125
sh scripts/run_srgc_sr_refresh.sh math 5 switch_fixed200
sh scripts/run_srgc_sr_refresh.sh math results     # shows fixed arms next to switch / switch_repeat
```

CPU tests: `python -m unittest srgc_rebuttal.tests.test_switch_fixed`.

## Direction-versus-magnitude analysis of the contrast (no GPU)

The SR-GC contrast is D = ||v|| (||g_on|| cos_on - ||g_sr|| cos_sr). From the
next child start, every refresh record also carries the norms and cosines of
the mean gradients, the trained-four versus random-four validation inner
products and the candidate cosine spread and top-4 gap
(`scripts/srgc_direction_records.py`, applied by the training child entry and
the extra arms; this instrumentation does not alter the package, and records
written earlier simply lack these fields). The separate 2026-09-29 verifier
repair intentionally changes the package identity. Tabulate and plot with:

```bash
python scripts/srgc_direction_analysis.py --plan <group-storage plan>           # -> <run root>/analysis/direction/
python scripts/srgc_direction_analysis.py --root <run root> \
    --legacy-d <paper repo>/v7/evidence/2026-09-24/selector-pair-srgc-all-d-1119.txt   # overlay recorded seed 3/4 D
```

Outputs `direction.csv`, `summary.txt` (per arm and step: D mean/sd and sign
count across seeds, cos_on - cos_sr, ||g_sr||/||g_on||, ranking gap) and
`direction.png` when matplotlib is installed (the `.venv-cu126` has it).
When On-policy or fixed controls have no decision D but all five finite
norm/cosine terms, analysis reconstructs the diagnostic contrast and labels
`d_source=reconstructed_diagnostic`. Recorded decisions are retained as
`d_source=decision`; older records without the decomposition remain missing.
Analysis never writes a switching decision back into the progress history.
The mechanistic reading being tested: cos_on - cos_sr shrinks toward zero over
training while ||g_sr|| stays above ||g_on||, so D turns negative when the
direction advantage is exhausted; the candidate cosine gaps shrink at the same time.

## GPU memory is checked before every child (CUDA OOM / NCCL DistBackendError)

A child launched while a dying or foreign process still holds the node's GPUs
fails minutes later with CUDA out of memory, or with `DistBackendError: NCCL
error` on the peers of the rank that died. The guard now, before each child:

1. reaps orphaned SRGC processes and removes this user's leftover
   `/dev/shm/nccl-*` / `torch_*` segments (killed ranks leave them; a full
   `/dev/shm` breaks the next NCCL init);
2. waits until every visible GPU is below `SRGC_GPU_FREE_MIB` (2000) used,
   reporting `GUARD waiting for GPU memory to free: ... (pid N ... MiB)` once a
   minute, for up to `SRGC_GPU_WAIT_SECONDS` (600);
3. if the GPUs stay busy, records `GUARD GPUs still busy ...` in the task log and
   fails the attempt with exit 75 without launching torchrun, so the retry two
   minutes later starts on free GPUs instead of OOM-ing.

Repeated OOM on a node therefore means a process outside this worker owns the
GPUs (another launcher, a Qwen worker, a manual `run`/`cache`); the `GUARD`
line names its pid.

## One GPU lease namespace for every launcher

OLMo workers, manual `run`/`cache` and Qwen workers used different lock
directories for their "exclusive" GPU leases, so two of them could start on the
same four GPUs and collide (CUDA OOM on one, `NCCL error` on the other's
ranks). The guard now takes every device lease in the legacy directory **and**
in the canonical lease namespace `<group volume>/.srgc-gpu-node-locks` (the one
the Qwen worker already uses), so any SRGC launcher on the node excludes the
others. Manual `run`/`cache` now go through the same guard (orphan reaping,
shm cleanup, free-GPU wait) as worker children.

## Why the first block sits at `update 0/25`, and the attention kernel

A refresh scores about 130 prompts (40 candidates, 40 SR prompts, 50
validation prompts) with eight 2048-token responses each, one prompt at a time
per rank, followed by a backward pass per response. The very first block of a
prefix therefore spends one to three hours at `update 0/25 · selection`
before the first `TRAIN ... update=1/25` line; the NODE line now shows how far
the ranks are: `gpus 4/4 busy (scored 12,13,12,11 prompts; last 3-9s)`.

The frozen runner loads the model with **eager** attention (README: "BF16/eager
OLMo-3 7B"), while the cache builder uses sdpa. Eager attention is several
times slower than sdpa for 2048-token generation and backward passes.
`SRGC_ATTENTION=sdpa` (or `flash_attention_2` where installed) makes the
training child load that kernel instead; the kernel is recorded in every
checkpoint's `checkpoint_policy.attention` and printed as `ATTENTION ...` at
child start. Numerics differ at floating-point rounding level only; the
protocol (sampling, seeds, steps) is unchanged. Set it in the worker's
environment before `run_srgc.sh ... run` and keep it constant within a run.

## Rollouts are saved as they are produced (interrupted blocks resume)

Policy checkpoints exist after every update, but a selection refresh (about
130 prompts x 8 responses) and the final 300-prompt evaluation had no save
point inside them, so a CUDA OOM, NCCL failure or GPU pre-emption during those
hours restarted the block from zero and a repeatedly failing run never left
`update 0/25`. `scripts/srgc_resumable_rollouts.py` (installed by both training
entry points) now writes each finished rollout to
`seed-N/rollout-cache/<task>/<phase>-<seed>/` on the shared run root; a
restarted attempt loads them and generates only the missing prompts, with the
same per-prompt sampling seeds. The directory is removed when the block
completes. Between rollouts the CUDA cache is released, and one CUDA
out-of-memory on a prompt is retried once after freeing memory. The launchers
now default `SRGC_ATTENTION=sdpa` (set `SRGC_ATTENTION=eager` to keep the
frozen runner's kernel); the kernel used is recorded in every checkpoint.
