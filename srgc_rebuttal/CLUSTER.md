# Multi-node execution for rebuttal seeds

Run one four-GPU worker on each allocated node. Workers share a queue and take
independent tasks as soon as dependencies finish. No cross-node gradient
all-reduce is needed: the four replicas of one task stay on one node.

The fixed plan has five new seeds (5–9). Each seed runs a shared 25-update
On-policy prefix once, then four arms continue to total update 275. This is
five prefix jobs and twenty continuation jobs, not twenty separate prefixes.

| Available nodes (four GPUs each) | Scheduling |
| --- | --- |
| 1 | Executes all tasks in sequence, reusing each prefix |
| 5 | All five prefixes can run together, then five continuations at a time |
| 10 | Up to ten ready continuations at a time |
| 20 | Up to all twenty continuations after their respective prefixes finish |

Five nodes are a practical starting allocation; more nodes shorten the
continuation queue. Prefix work initially has only five independent tasks.
Actual speedup depends on rollout lengths, hardware and the longest arm;
no measured 7B speedup or completion time is claimed.

## Prepare the shared run directory

Use a fixed checkout of `offpolicy-misranking` branch
`experiments/srgc-cost-replication` on storage accessible at the **same path**
on every node. Inputs, outputs and locks must be shared too. The filesystem
must implement coherent POSIX `flock` across nodes and atomic rename; local
scratch copies with separate lock directories cannot coordinate these workers.
The automated multi-process tests exercise local filesystem locking, not
your cluster's filesystem. Use the site's validated shared-lock filesystem.

Install `requirements.txt` in an environment available at the same path on
each node, with a CUDA-capable PyTorch build. Stage model weights in the local
or shared Hugging Face cache before starting workers to avoid simultaneous
downloads. Use the same packages and GPU type for comparable measurements.
Do not edit Python code, plan or inputs while the queue is active.

Prepare the five real input bundles described in [README.md](README.md), then:

```bash
python -m srgc_rebuttal.plan --check-inputs
```

The real MATH-train input bundles for seeds 5–9 are prepared in `inputs/`.
See [REBUTTAL_READY.md](REBUTTAL_READY.md) for the fixed split, explicit 50-prompt
online validation set, cache-generation commands and preparation manifest.
Generate the missing initial-policy rewards on GPUs before this input check
and before launching the training queue. Different seeds' caches can be built
on different four-GPU nodes in parallel.

## Start workers

From the shared repository root on each allocated node:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m srgc_rebuttal.cluster worker
```

Replace the device list with that node's four allocated GPUs. The worker
checks their occupancy, holds an exclusive device lease and starts four local
processes with `torchrun`. It does not stop unrelated GPU jobs. Use the same
`--node-lock-root` for different queues sharing the same devices; its default
is `srgc_rebuttal/runs/gpu-node-locks` next to the output cohort directory.

For SSH-accessible allocated nodes, first print the commands without launching:

```bash
python -m srgc_rebuttal.cluster commands \
  --hosts node-a node-b node-c node-d node-e \
  --repo /shared/offpolicy-misranking \
  --python /shared/env/bin/python
```

After replacing those placeholders with real nodes and paths, change
`commands` to `launch` to start all workers. This SSH mode uses each remote
environment's `CUDA_VISIBLE_DEVICES`, or devices 0–3 when unset. For scheduler
allocations, start `worker` within each allocation so its GPU visibility is
preserved. `launch` starts workers only; it does not wait for training to finish.

## Progress and restart

```bash
python -m srgc_rebuttal.cluster status
python -m srgc_rebuttal.summarize
```

Outputs are in `srgc_rebuttal/runs/additional-seeds/seed-N/`.
Queue receipts and per-task append-only logs are under the cohort's `.queue/`.
Each seed's prefix has a checksum receipt; all arms record that same checksum.
Task and execution locks protect against duplicate workers and overlapping
manual starts. Code, plan and input hashes reject incompatible resumes.

The next node can resume a task from its saved checkpoint after its old
process and lock have gone away. Prefix checkpoints are saved every five
updates; continuations every 25. Completed tasks are skipped before loading
a model. Explicit failures are retained for inspection; after resolving an
environmental failure, restart a worker with `--retry-failed`. A code or input
change requires a new output directory and plan, not an in-place resume.

SIGTERM/SIGINT is forwarded to the worker's own child process group; it waits
for that group to exit before releasing leases. Preserve all attempt logs:
durable phase receipts count completed repeated work across attempts. An
interrupted timer prevents reporting a complete GPU-time total for that arm.
The summary requires every planned seed and all
four arms, including unfavorable outcomes.

## Verified scope

Four CPU worker processes completed all 25 synthetic tasks exactly once,
overlapped independent work, and waited for each seed's own prefix. Tests also
cover failure/retry behavior, locking, hash rejection and numerical equivalence
of the computation changes. See [VALIDATION.md](VALIDATION.md). No real remote
node or 7B GPU training job was launched while preparing this code.
