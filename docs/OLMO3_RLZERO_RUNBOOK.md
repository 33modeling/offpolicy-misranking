# OLMo-3 RL-Zero Runbook

## Registered experiment

- Model: `allenai/Olmo-3-1025-7B` at
  `a81bae42db3975be1671e27b9c9a56da1a9f980f`.
- Starting policy: raw OLMo-3 base. No SFT, DPO, or prior RLVR checkpoint is
  loaded.
- Objective: online verifier-reward GRPO with four DDP ranks and eight samples
  per prompt. LoRA is the trainable policy parameterization; it is not an SFT
  objective. `policy_train.json` must record `supervised_loss=false`.
- Domains: MATH-500 at
  `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be` and MBPP at
  `4bb6404fdc6cacfda99d4ac4205087b89d32030c`.
- Candidate/validation counts: MATH-500 `400/100`, MBPP `512/100`.
- Seeds: `0 1 2 3 4`. Cumulative policy steps: `0 25 100 400`.
- Generation: temperature `1`, top-p `1`, group size `8`, maximum `2048` new
  tokens. GRPO uses one optimizer epoch per newly sampled group.
- Prompts: the released OLMo RL-Zero math/code templates are rendered directly.
  The base tokenizer has no chat template, so `apply_chat_template` is not used.

The immutable machine-readable contract is `configs/olmo3_rlzero.json`.

## Cluster network: download only (recorded 2026-09-09)

The cluster can pull (`git pull`, dataset and model downloads) but cannot push
or upload: `git push` from a cluster node fails and other uploads are heavily
restricted. Consequences, so that nobody builds on the wrong assumption again:

- Nothing that runs on the cluster may depend on pushing, uploading, or
  reaching a service to deliver a result. Results are plain-text files under
  `$OM_WORK/exports` (digests, `why`, reward checks, rescoring logs, status
  history) and leave the cluster only by the operator's own path, usually from
  the PC, into the `33modeling/transfer` repository.
- `bash scripts/digest_family.sh` therefore does not push. It writes the digest
  and prints its KEY NUMBERS table to the terminal, so a screenshot or a short
  paste already carries the result; `scripts/handover.sh` (copy, commit, push)
  is for the PC side, or `DIGEST_HANDOVER=1` on a machine that can push.
- The operator often works from a phone with no clipboard: commands handed over
  must be one line with nothing in front and no arguments; a needed mode is
  made the script default instead.

## Reference reliability versus budget (2026-09-09)

The 16:18 KST status readout put the split-half floor of every d0 point near
chance (MATH-500 .10-.175, MBPP .06-.12 against 2k/n = .20), so plan section 9
row 1 applies: no drift point is interpreted, and the registered remedy is a
more reliable reference, not a lower gate. Two scripts serve that, both outside
the registered matrix (no registered point is written, no label is defined):

- Sizing (CPU, any node, minutes):

  ```bash
  bash scripts/reliability_budget.sh
  ```

  reads `oracle_micro_groups.pt` and `val_groups.pt` of every d0 point (root copy
  or the copy parked under `pinned-scoring/`), re-draws the two reference halves at
  every observable half size (4, 8, 12, 16 responses; 25 or 50 validation prompts),
  separates the candidate-rollout axis from the validation-direction axis, and
  extrapolates with Spearman-Brown through the frozen Gaussian lookup of
  `src/measurement_ceiling.py`. The KEY lines say how many responses per half the
  gate floor would need and which side limits it now. Optional arguments
  `<dataset> <drift>`; the report goes to `$OM_WORK/exports/reliability-budget-*.txt`.
  It reuses the locked rollouts descriptively; plan section 7 forbids using such a
  curve as a registered result, which is why the next script exists.

- New reference groups (one node with the visible GPUs, about half a day per
  dataset at 64 responses per prompt on 4xH100):

  ```bash
  bash scripts/run_reliability_budget.sh math500
  ```

  `<dataset> [fresh_k=64] [val_k=32] [seed=100]`; `RB_DRY=1` prints the plan. It
  copies `prompts.json` and the run-config template from a finished d0 point,
  generates `fresh_k` responses per candidate prompt and `val_k` per validation
  prompt from the base policy in a new RNG domain under
  `$OM_WORK/runs/reliability-budget-v1/<dataset>-fk<K>-vk<VK>-s<seed>/`, computes
  the micro-group gradients with the registered stages of `src/experiment.py`, and
  writes the floor-versus-budget report next to the registered d0 point's to
  `$OM_WORK/exports/reliability-budget-run-*.txt`. Rerunning the same command
  resumes. `fresh_k` must be a multiple of 16 (micro-groups of four, R/A/B
  partition). The run does not join the supervisor queue, but it holds the same
  node-local `primary.lock` as the registered launcher (a busy node is refused) and
  one lease per run directory (the same dataset, budget and seed cannot start on
  two nodes). Resuming compares the whole effective contract (model, prompts, budgets,
  token cap, projection, verifier, prompt format) and refuses a mismatch; TERM/INT
  stop the run's own stage processes. The sizing report marks its prediction
  `unsupported` (no budget claim) when the held-out self-check misses, the registered
  cell carries no signal, exact ties dominate the selection boundary, or the stored
  A/B scores do not reproduce from the artifacts (review 2026-09-09, six defects fixed).

## Training hyperparameters

| Component | Registered value |
|---|---|
| Distributed batch | 4 DDP ranks, 1 prompt per rank, 8 responses per prompt (32 responses/update) |
| Reward/advantage | Binary domain verifier; within-prompt population standard deviation; denominator epsilon `1e-4` |
| Policy loss | PPO-form GRPO surrogate, ratio range `[0.8, 1.2]`, one optimizer epoch per fresh group |
| Reference regularization | KL coefficient `0` |
| Optimizer | AdamW, learning rate `1e-5`, no scheduler, gradient-norm cap `1` |
| AdamW | `betas=(0.9, 0.999)`, `eps=1e-8`, `weight_decay=0.0` (explicit since 2026-09-06; runs generated on earlier commits used torch's default 0.01 and are pinned to that code), `amsgrad=false` |
| LoRA | `q_proj,v_proj`, rank `16`, alpha `32`, dropout `0`, bias `none` |
| Numeric/attention mode | BF16 model load and eager attention |
| Sampling | temperature `1`, top-p `1`, top-k `0`, no repetition penalty, maximum `2048` new tokens |
| Checkpointing | Every 5 updates; cumulative targets `0/25/100/400` with adapter and optimizer resume |
| Evaluation | behavior/current/validation rollouts `8/32/8`; 4-rollout micro-groups; CountSketch dimension `4096`; final 4 decoder layers |
| Selection/inference | top `10%`; 10,000 bootstrap replicates for final labels |

The trainer constructs `torch.optim.AdamW(trainable, lr=1e-5)` and does not
install a learning-rate scheduler. The AdamW values above are therefore the
PyTorch defaults rather than separately passed command-line fields; the run
manifest records the software versions needed to interpret them.

The stored old log probabilities and the `clip_epsilon=0.2` field are part of
the implemented GRPO surrogate. Because this matrix performs only one epoch on
each freshly sampled group, current and old log probabilities are compared
before the sole optimizer step and their ratio is one up to numerical error.
There is no post-update second pass on that group, so clipping is not described
as an effective trust-region constraint for this experiment. This differs from
the previous two-epoch Qwen matrix; the separate generalization matrices also use
one epoch and keep GRPO fixed.

## Prepare

Use a checkout that can reach the public Hugging Face repositories and mounts
the same `GROUP_VOLUME` as the compute nodes:

```bash
git pull
bash scripts/run_olmo3_rlzero.sh prepare
```

No token is read or requested. The model and datasets are public and all
inherited Hugging Face token variables are removed. An existing
model directory, including one uploaded separately without a Hugging Face
`.cache` directory, is checked against the registered official file sizes and
hashes and sealed automatically without redownloading. If that snapshot is outside the default
`$MODELS_DIR/Olmo-3-1025-7B`, set `OM_OLMO3_MODEL_PATH` to its exact local path
on all three nodes.

Separately uploaded MATH-500 and MBPP data may be JSONL, parquet, an HF saved
dataset, a flat file, or an arbitrarily named nested directory. `run` searches
recursively and identifies data by parsed schema, official row count, and an
order-independent official content fingerprint. Original MATH rows containing
`problem/solution` are accepted by extracting the final boxed answer. The
verified rows are atomically materialized at the standard local paths before a
commit-pinned continuation. Run mode never contacts the Hub.

`run` bootstraps the pinned `math-verify==0.9.0`,
`latex2sympy2-extended==1.11.0`, ANTLR 4.13.2, SymPy 1.14.0, and mpmath 1.3.0
wheels from `vendor/wheels` into `$OM_WORK/runtime-deps`. This does not call
pip, modify the shared venv, or require network access. Import-source and
symbolic-equivalence smoke tests run before GPU admission.

`transformers>=4.57.0` is required by OLMo-3. Update the shared venv from
`requirements.txt` before allocating GPUs if static checking reports an older
version.

## Run on three clusters

All three clusters see the same checkout and `.git` directory. From that shared
checkout, update once before starting any worker:

```bash
git pull --ff-only
```

Then execute the same command on every 4xH100 node, without another pull:

```bash
bash scripts/run_olmo3_rlzero.sh run
```

There are ten independent families: two datasets times five seeds. A shared
`flock` assigns one whole family to a cluster. Within a family, one node owns the
continuous `base -> 25 -> 100 -> 400` chain, so adapters and optimizer state are
never handed between simultaneously running nodes. After finishing a family,
the node claims the next unowned family. Three workers therefore process up to
three families concurrently without hard-coding a cluster index or hostname.

Node admission uses a lock under local `/tmp`. Only family and collection locks
are on `GROUP_VOLUME`, so identical cloned hostnames do not collapse three
clusters into one worker.

Before family assignment, every node must pass:

1. Model revision, architecture, tokenizer, LoRA target, shard completeness, and
   per-file SHA verification in offline mode.
2. Dataset revision, row count, file SHA, deterministic disjoint split, and real
   Math-Verify/code verifier checks. Code uses bubblewrap when namespace creation
   works, otherwise a resource-limited subprocess with a strict AST/import/file/
   process deny policy.
3. Cached real OLMo generations with positive, negative, and within-prompt mixed
   verifier rewards for both MATH-500 and MBPP.
4. A four-rank GRPO step followed by an adapter/optimizer resume to step two on
   that physical node.

No family is claimed when any gate fails.

## Reading one finished family early

```bash
bash scripts/family_readout.sh h100 mbpp 0          # provisional (1,000 replicates)
bash scripts/family_readout.sh h100 mbpp 0 10000    # final-quality bootstrap
```

Runs `regime_map.py` on that family's four completed points only, prints the
regime report, and writes it under `$OM_WORK/readouts/family-<dataset>-s<seed>-<git>-boot<N>/`.
It is a preview: the registered result is the full 40-point collection.

## CPU Rescoring And Separate GPU Evaluation Status (2026-09-09)

`scripts/rescore_math500.sh` performs CPU-only reward rewriting from saved
responses. The follow-up gradients and model-based scores require a GPU worker.
These are separate workloads, although they still use the same family queue.
Setting `OM_RLZERO_ONLY_FAMILIES` restricts that worker, not every other worker.

After rescoring parks the old DONE, status distinguishes:

- `prior completion`: current DONE or nonempty archived DONE, counted once.
  This is historical completion evidence, not a new checkpoint validation.
- `final accepted`: current nonempty DONE only. The actual completion gates
  are unchanged; historical completion does not authorize final collection.
- `R / GPU evaluation queued`: reward-derived evaluation without a live owner.
  This is not an automatic retry or evidence that a separate worker is running.
- `GPU RE-EVALUATION`: an owned reopened point. Watchdog, runtime-contract,
  and current-attempt errors remain visible. A combined ETA is not estimated
  from a mixture of primary completion and reopened evaluation points.

Inspect with `bash scripts/run_olmo3_rlzero.sh status h100`. On a separate idle
4xH100 node sharing the same work root, the selected evaluations can be assigned
without restarting healthy primary workers:

```bash
OM_RLZERO_ONLY_FAMILIES="math500/s0 math500/s1" bash scripts/run_olmo3_rlzero.sh run h100
```

CPU rescoring now checks duplicate response identity before reusing the
comparison across the merged file and shards. Conflicting tokens, response
boundaries, gold answers or pinned rewards abort before the family's write
phase. The timeout budgets and scoring rules are unchanged. Progress logs show
`[rescore-cpu]`, `scored`, `reused`, `timeout_events`, and elapsed seconds, after
each file and at row boundaries at least ten seconds apart (or every 1000 rows).
A single slow verification may exceed that reporting interval. Both legacy
stdout timeout notices and current math-verify logger warnings are counted;
other warnings are preserved. Timeout events are not CUDA errors and are not
the same metric as `timeout_sensitive` reward reproduction.

Pulling code does not replace an already-running Python rescoring process.
Do not rerun an apply operation just to update logging or inspect status. Its
existing output is under `$OM_WORK/exports/rescore-math500-apply-*.txt`; inspect
that output before interrupting a running apply. No GPU worker restart is
required for this status change.

## Parallel d0 evaluation on spare OLMo nodes (2026-09-09)

For an explicit pair, use separate roles instead of starting two unrestricted
workers. Both commands require an existing family and generation pin:

```bash
# Chain owner, e.g. node 4: resume only this family, with a shareable lease.
bash scripts/run_olmo3_rlzero.sh resume-family h100 mbpp 4

# Spare GPU node, e.g. node 3: only this family's d0 evaluation, never its GRPO.
bash scripts/run_olmo3_rlzero.sh assist h100 mbpp 4
```

Transition procedure for the September 9 14:21 allocation:

1. Prepare a new, unused checkout on the shared mount before interrupting any
   work. From the existing checkout, one example is
   `git clone "$(git remote get-url origin)" ../offpolicy-misranking-pair`.
   Do not update a checkout still read by a running launcher. If the example
   directory already exists, choose a different unused directory. Preserve the
   existing `OM_WORK`, profile, model/data settings and node-local lock namespace.
2. On node 4, stop its current launcher with Ctrl-C, change to the new checkout,
   then run `resume-family h100 mbpp 4` as above. This is a targeted restart:
   saved checkpoints and rollout partials stay in their existing directories
   and must pass the normal reuse checks. In-flight unsaved work and model/
   preflight startup time are not free. Do not remove the generation pin.
3. On node 3, stop its idle launcher with Ctrl-C, change to the same new checkout,
   then run `assist h100 mbpp 4`. Nodes 1, 6, 7 and 8 are left running. These
   commands do not start Qwen or change the scientific matrix.

The helper can start first: it waits for a shared training owner and cannot
take the GRPO chain. It does not attach through an old exclusive family lease.
The owner command does not fall back to being a helper. Once d0 is accepted,
the helper exits and releases its node lease; it does not wait for the whole
family, take another family or publish the full collection. A d0 that already
has DONE is reported before GPU admission without starting compute.

Ctrl-C is not a promise that every detached child has stopped. Explicit roles
refuse an occupied local node lock instead of killing its owner. After acquiring
the lock, they clean only same-node, target-family orphan processes. Any other
GPU process causes admission to fail without being killed. Do not remove lock
files or launch multiple workers on the same physical node to bypass a refusal;
inspect the reported remaining processes. Ordinary `run` behavior is unchanged.

Success is visible as `[worker-role] role=resume-family ... parallel_control=1`
on the owner, followed by `[control-assist] claim=mbpp/s4/d0` on the helper.
`[assist-wait]` explicitly distinguishes an exclusive lease from no shared
training owner. Owner JSON includes `parallel_control` and `supervisor_git` so
future status evidence can identify the actual supervisor mode, not infer it
from the generation pin. Preflight qualification remains enabled. Cross-node
flock coherence and actual GPU timing still require verification on the cluster.

The older automatic-help opt-in remains available:

Opt in on the participating nodes from an updated idle checkout:

```bash
OM_RLZERO_PARALLEL_CONTROL=1 bash scripts/run_olmo3_rlzero.sh run h100
```

Workers still prioritize normal OLMo families. When no training family can be
claimed, a spare worker searches the most-work-remaining families for an
unfinished d0 evaluation whose behavior pool is already validated. It can
evaluate that d0 while another upgraded worker continues d25 -> d100 -> d400.
Once finished it returns to the queue and can help another eligible family.
This increases useful parallelism within OLMo; it never starts Qwen.

For the September 9 08:01 snapshot, mbpp/s4 was furthest behind. The useful
initial pairing is its current owner plus a genuinely spare four-H100 node.
Both must run the command above. Stop/re-enter only those selected launchers
after checking their current ownership; preserve the run root and generation
pin. Existing checkpoint and rollout partials resume through the normal pinned
pipeline, with model/preflight startup overhead. Do not restart all healthy
workers. Old exclusive family owners cannot be helped in place: their locks
remain respected until those owners stop. A git pull cannot hot-patch a frozen
running supervisor. Expand participation at later natural restart boundaries.

The math/MBPP protocol, data sizes, K values, GRPO world size and generation
commit do not change. D0 is not a prerequisite for GRPO once behavior data is
valid. The sequential GRPO checkpoint dependency still cannot be skipped.
The existing legacy serial validation path remains serial within each point;
this change does not silently deploy new generation code to the current run.

Locking: upgraded participants share the existing family lease (legacy EX
owners remain excluded), one exclusive training lease allows only one GRPO
chain, and one exclusive point lease covers repair, generation, validation and
publication. Helpers have a separate control lease and owner record, displayed
as `<dataset>/s<seed>/d0` in worker claims. Family-wide runtime repair is disabled
in this mode; the matrix repairs only its locked point. A helper never writes
the family-complete stamp or publishes the full collection. Completed control
artifacts are reused by the original chain; the final 40-point checks remain.

Use `[control-assist] claim=.../d0` and `completed .../d0` in the worker log to
confirm actual assignment. No compatible upgraded chain or no unfinished valid
d0 means waiting, not extra GPU training. All participants must see coherent
cross-node shared-volume flock behavior; local tests cannot validate the mount.
This overlaps independent evaluation work but is not a measured H100 speedup
or a promise that the whole remaining matrix scales with GPU count.

## Failure loops

OLMo3 is the first priority. A blocked primary worker never hands off to Qwen
or another model. It tries other eligible OLMo3 families; if all remaining
families are `LOOPING`, it reports `[primary-blocked]` and rechecks the primary
queue without retrying the known-broken points. GPU keepalive is stopped in
this fully blocked state; it is not training progress or a throughput fix.

Additional `--run` and `--check` commands (including direct Qwen launchers and
the rotation runner) require the shared OLMo3 generation/config binding,
all 40 `DONE` points, all 10 family completion stamps, and the final collection.
The gate defaults to the H100 v2 matrix; `OM_PRIMARY_PROFILE=baseline` selects
the baseline matrix. Existing `OM_OLMO3_ROOT`, `OM_OLMO3_RESULTS`,
`OM_OLMO3_MODEL_TAG`, and `OM_RLZERO_CONFIG` overrides must identify that same
primary matrix. Missing or stale evidence returns 75 before GPU preflight.
`prepare` remains an offline-cluster-independent download operation. The 9B
wrapper no longer rotates automatically to other models.

Explicit exception: `bash scripts/run_qwen35_9b.sh run-idle` allows only Qwen
9B on the operator-selected current hostname before primary completion. The
local primary lock and four-idle-H100 admission still apply. `restart-idle`
first cleans up only that node's previous Qwen 9B namespace and children.
OLMo workers keep running; a residual lock owner must be identified, not
assumed to be OLMo. The primary never invokes this exception automatically.

This supervisor change does not alter generation pins or checkpoint formats.
It does not hot-patch an already running frozen supervisor or stop a remote
Qwen process. Do not restart healthy OLMo3 workers merely to update this policy.

A point that dies of the same error on every attempt is not retried forever.
After `OM_RLZERO_MAX_FAMILY_FAILURES` (default 4) consecutive failures the
launcher writes `.families/<dataset>-s<seed>.loop` with the last error, prints
`[family-loop] ...`, and every worker skips that family. `status` shows the row
as `LOOPING` with the error. After fixing the cause, relaunch with
`OM_RLZERO_CLEAR_LOOPS=1 bash scripts/run_olmo3_rlzero.sh run h100`.

## Dead-worker alerts

Every worker's heartbeat process watches the other workers' heartbeat files
under `$ROOT/.workers/`. When one stops updating for five minutes (node lost,
launcher killed) every surviving worker prints, on its own terminal and in its
log, once per hour:

```text
[WORKER DEAD] run275509-... on h100-b: no heartbeat for 4d; it held math500/s0. Start a worker on h100-b again: ...
```

The same line is appended to `$ROOT/logs/ALERTS.log`, and `status` shows the
last alerts and names dead workers in its DECISION line.

## Parallel fresh validation (2026-09-08)

For a matrix whose **generation commit contains this change**, fresh train
and fresh validation both run on the selected GPUs. With four GPUs and 100
validation prompts, each GPU generates 25 validation prompts after its train
shard, instead of GPU0 generating all 100. The shell waits for all shards and
merges into the same canonical `rollouts_fresh_val.jsonl`. It validates exact
prompt/K coverage and provenance before publishing the canonical manifest and
discarding redundant shards. Downstream gradient
and scoring inputs, K, token cap, policy adapter, and per-prompt seed domains
are unchanged. There are no extra model loads when a train worker continues
directly to its validation shard.

Scheduling is recorded in the point's `.fresh-val-layout.json` under a lock.
A pre-existing serial validation partial retains the serial path and resumes
its missing prompts; it is not discarded to get parallelism. Finished
validation is reused. An interrupted new shard resumes from its own partial.
Mixed layouts or a changed GPU count for an unfinished sharded validation
fail without silently reinterpreting or discarding those artifacts. Preserve
the recorded layout rather than deleting it to force a resume.

The September 8 MBPP log showed roughly 90-95 minutes for serial validation.
Four balanced shards would theoretically reduce that part to 23-24 minutes
before overhead. This is not a measured H100 speedup or a fourfold speedup of
the whole matrix. Inspect `fresh-shard*.log` for each worker's
`validation sharded ... prompts=[lo,hi)` line and generation/verification
timings when validating the new generation on an available allocation.

**Existing pinned matrices do not acquire this change from `git pull` or a
supervisor restart.** Their `.queue/generation.git` and point code identities
remain unchanged. Do not interrupt the seven running primary workers, edit
their local clones, rewrite Git/hash fields, or regenerate finished work just
to activate this optimization. The patch is available for a new compatible
generation; migration of an already-running pinned generation is a separate
operation, not implemented or implicitly authorized by this change.

CPU regression command (use a test environment with PyTorch and pytest):

```bash
PYTHONPATH=src python -m pytest -q tests/test_fresh_validation_shards.py
```

## Runtime batch sizes per dataset (H100 profile, 2026-09-07)

Measured on the running matrix: every response runs to the 2048-token cap and a
batch-8 decode step is overhead-bound (about 280 tok/s per GPU, 233 s per
prompt for 32 samples), and mbpp validation gradients die at gradient
micro-batch 4 (`torch.OutOfMemoryError` in the fp32 attention softmax, 9 GiB per
layer at 4.3k tokens). The h100 launcher therefore applies, per family:

| dataset | generation batch (`OM_GEN_BATCH`) | gradient micro-batch |
|---|---|---|
| math500 | 32 | 4 |
| mbpp | 16 | 1 |

Both are execution-only knobs: neither is in `regime_contract.RUN_CONFIG_FIELDS`,
the rollout partial manifest does not record the batch, and OOM recovery
already lowers the batch mid-stage. They are recorded per point in
`run_config.json`; because the pinned `run_point.sh` refuses to re-enter a point
whose record differs from its environment, the launcher rewrites the two fields
(and the digest) of every unfinished point of a family before claiming it and
logs each change as `[repair] <point>: <field> <old> -> <new>`
(`src/repair_run_config.py`; finished points are never touched). `status`
expects the same per-dataset values and prints them as `runtime_per_dataset`.
Override or disable with `OM_RLZERO_GEN_BATCH_BY_DATASET="math500=32 mbpp=16"`
and `OM_RLZERO_GRADIENT_MICRO_BATCH_BY_DATASET="mbpp=1"` (an empty string
disables). The baseline profile applies no overrides.

## Static split per node (optional)

By default every node pulls from the shared family queue. To pin a node to one
dataset, add it as the third argument (this is the form to type on a phone):

```bash
bash scripts/run_olmo3_rlzero.sh run h100 math500
```

To pin a node to an arbitrary fixed list instead, set:

```bash
OM_RLZERO_ONLY_FAMILIES="math500/s0 mbpp/s0 math500/s1 mbpp/s1" bash scripts/run_olmo3_rlzero.sh run h100
```

Families outside the list are neither claimed nor waited for on that node.
Give every family to exactly one node; the final collection still needs all
ten. The per-family lock stays in force, so an overlapping list is safe (the
second node skips a family already held).

## Restart and Git updates

An interrupted rollout keeps only exact-K complete prompt groups in `.partial`
and continues from the next prompt. The partial file is bound to its generation
manifest; incompatible restart state is quarantined instead of mixed. If the
worker stops after publishing JSONL but before its manifest rename, the next run
validates the exact rows and finishes that publication automatically. Interrupted
GRPO loads the newest adapter/optimizer/statistics checkpoint, including a
complete target-step checkpoint when only final publication remained. A family
failure no longer exits the launcher: it releases that family lock, preserves
the partial artifacts, and (since 2026-09-07) the same worker retries that
family after `OM_RLZERO_FAMILY_RETRY_SECONDS` (60 s) until it succeeds or the
failure-loop guard marks it LOOPING. It no longer moves on to a fresh family
and leaves the failed one for "the next free worker": on 2026-09-06 that left
math500/s1 (three points done, last generation stage) unowned for nine hours
while every worker was busy for a day. Workers also claim the most-progressed
family first (DONE points, then started families, then registered order), so
resuming beats starting. `OM_RLZERO_FAMILY_ATTEMPTS` controls immediate
attempts inside one claim.

The supervisor GPU keepalive pauses its bursts on a GPU whose utilization is
already above `OM_GPU_KEEPALIVE_BUSY_PERCENT` (40) as sampled by nvidia-smi
every `OM_GPU_KEEPALIVE_SAMPLE_SECONDS` (5); a GPU running a rollout is not
idle. Without a working sampler the fixed 15% duty applies as before.

Only a user signal or loss of the worker itself requires rerunning the command:

```bash
bash scripts/run_olmo3_rlzero.sh run
```

The first worker atomically writes the experiment-wide generation commit to:

```text
$OM_WORK/runs/olmo3-1025-7b-base-rlzero-grpo-v1/.queue/generation.git
```

After all launchers exit, one `git pull --ff-only` on the shared checkout is
allowed. At startup each worker copies the selected supervisor and generation
commits into a node-local standalone clone under
`${OM_LOCAL_LOCK_DIR:-/tmp/offpolicy-misranking-UID}/pipelines`. Each clone owns
its `.git` directory; no worker adds, locks, or prunes shared worktrees. A newer
supervisor therefore runs all unfinished and not-yet-started families using the
recorded generation code without being affected by a later shared-checkout
change. Do not delete the generation marker, family completion stamps, or
partial checkpoints. A missing local Git object aborts before a GPU family is
claimed.

Prompt/contract failures use exit code `43` to skip immediate same-family
attempts. The worker stays allocated, moves to other work, and retries that
family on a later queue pass instead of terminating. Repeated attempts reuse
only contract-valid durable artifacts.

## Is it training? (2026-09-07)

Liveness is not progress. Every worker prints, on its own terminal and in its
worker log, one line every 10 minutes (`OM_RLZERO_PROGRESS_SECONDS`) computed
from durable artifacts only: DONE points, GRPO steps (`grpo_stats.jsonl`
lines), rollout bytes and the newest artifact write. Logs, keepalive and
telemetry do not count.

```text
[progress] TRAINING  points 7/40  grpo 1025 steps  rollouts 812 MB  last 30m: +0 points +25 grpo steps +18 MB rollouts  last write 2m ago (rollouts_fresh_train.shard1.partial)
[NOT TRAINING] NOT TRAINING for 47m  points 7/40  grpo 1025 steps  rollouts 812 MB  last 30m: +0 points +0 grpo steps +0 MB rollouts  last write 47m ago (...)
```

`[NOT TRAINING]` also goes to `logs/ALERTS.log`. `status` prints the same as
its `PROGRESS` line and, when workers are alive but nothing durable changed for
`OM_PROGRESS_STALL_MINUTES` (30), its DECISION is `ERROR: NOT TRAINING ...` and
STATE is `NOT TRAINING`, whatever the heartbeats and GPU duty say. The probe
history is under `<root>/.progress/history.jsonl` (`src/training_progress.py`).

## Observe and collect

Queue state:

```bash
bash scripts/run_olmo3_rlzero.sh status h100
```

`status` prints one screen: a VERDICT/ACTION header, worker and point counts,
and one row per family (state, current point, pipeline stage `k/8`, GRPO
steps, time since last activity, worker, and the latest current-attempt error
or recovery note). Add `verbose` as the third argument
(`status h100 verbose`) for the full evidence dump described below: per-point
rows, telemetry, error attribution and log tails. Every `status` run is also
appended to `$ROOT/logs/status-history.log` (shared volume), so the timeline of
verdicts can be read later from any node.

`status` is an active health check, not only a queue listing. By default it
samples the shared run for 20 seconds and combines five independent signals:
family locks, worker heartbeats, node-local watchdog telemetry, durable
artifact changes, and the configured/runtime batch contract. It also scans
every log belonging to an active family for CUDA/runtime errors. Family states
include `PROGRESSING`, `COMPUTING`, `ALIVE`, `IDLE`, `UNKNOWN`, `STUCK`, `DEAD`,
`RETRYING`, `STOPPED`, `PENDING`, and `COMPLETE`.

- `PROGRESSING`: a real log or durable artifact changed during the probe.
- `COMPUTING`: fresh watchdog telemetry observed CPU or GPU activity even if
  the stage emitted no log line.
- `ALIVE`: the worker/pipeline heartbeat is fresh but no progress was observed
  during the short probe.
- `IDLE`: one inactive watchdog window was observed; the watchdog will verify
  it again before terminating anything.
- `HUNG`: telemetry claims CPU/GPU activity but no artifact or log has changed
  for more than six hours (or eight stall windows). Supervisors launched before
  2026-09-06 counted the pipeline's own GPU keepalive as compute, so a hung
  point looked `COMPUTING` indefinitely. Ctrl-C that worker, `git pull`, and
  relaunch `run h100`; the `.partial` rollouts and checkpoints resume.
- `UNKNOWN`: a legacy or broken worker holds the lock but provides no fresh
  telemetry, or a CPU/GPU probe failed. This is not proof that the process is
  stuck, and a probe failure suppresses watchdog termination.
- `STUCK`: the node-local watchdog measured no log, CPU, or GPU activity for
  consecutive idle windows and recorded termination evidence.
- `DEAD`: the family lock was released while an owner record remained.

The final diagnosis checks whether all three expected worker heartbeats are
observable. Completion additionally requires the exact generation commit,
config hash, model revision, dataset, and seed in the family stamp; non-empty
stamp files are not accepted by themselves. For a non-waiting snapshot set
`OM_RLZERO_STATUS_PROBE_SECONDS=0`; change the thresholds only through
`OM_RLZERO_STATUS_STUCK_SECONDS`, `OM_RLZERO_STATUS_WORKER_STALE_SECONDS`, and
`OM_RLZERO_STATUS_HEARTBEAT_STALE_SECONDS`, and
`OM_RLZERO_STATUS_EXPECTED_WORKERS`.

The H100 generation batch is 8. Only a confirmed OOM can lower it, and recovery
steps down geometrically (`8 -> 4 -> 2`). Batch 2 is the supervisor safety
floor and does not change the immutable experiment config hash; a further OOM
aborts with explicit recovery exhaustion instead of falling back to batch 1.
CUDA runtime/context errors restart at the configured batch 8. Each
attempt records the starting byte offset of every stage log; recovery reads
only newly appended bytes for the missing rollout stage. Status output likewise
separates all historical error matches from current-attempt matches.

Worker logs are under:

```text
$OM_WORK/runs/olmo3-1025-7b-base-rlzero-grpo-v1/logs/
```

The final locked aggregation requires all 40 points and writes:

```text
$OM_WORK/results/olmo3-1025-7b-base-rlzero-grpo-v1/FINAL_REPORT.md
$OM_WORK/results/olmo3-1025-7b-base-rlzero-grpo-v1/COMPLETE
```

The Qwen primary and prior additional-study roots are neither read nor modified.
