# Multi-node execution for rebuttal seeds

Run one four-H100 worker on each allocated node (full H100 GPUs with at least
75,000 MiB each). Two nodes, eight GPUs total, are sufficient. Workers share a queue and take
independent tasks as soon as dependencies finish. No cross-node gradient
all-reduce is needed: the four replicas of one task stay on one node.

The fixed plan has five new seeds (5–9). Each seed generates its missing cache,
runs a shared 25-update On-policy prefix once, then four arms continue to total
update 275. This is five cache tasks, five prefix tasks and twenty continuations.
Existing verified caches and completed tasks are skipped. A seed can proceed
as soon as its own dependencies finish; other seeds need not finish first.

## MATH or MBPP

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
MATH writes to `srgc_rebuttal/runs/additional-seeds/`; MBPP writes to
`srgc_rebuttal/runs/mbpp-seeds/`. Inputs, cache receipts, checkpoints, task
leases and result reports are separate; device leases are shared between
the built-in datasets. `--plan` selects a custom cohort; an explicitly
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

The next node can resume a task from its saved checkpoint after its old
process and lock have gone away. Prefix checkpoints are saved every five
updates; continuations every 25. Completed tasks are skipped before loading
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
