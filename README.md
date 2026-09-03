# Off-policy Misranking RLVR Experiments

This repository runs the paper experiment with a real verifier-reward GRPO
policy update. The earlier positive-rollout SFT drift is retired and cannot be
entered through the canonical runner.

## Clean OLMo-3 Restart

The clean causal experiment starts from the raw, non-SFT
`allenai/Olmo-3-1025-7B` base checkpoint and uses verifier-reward GRPO on
MATH-500 and MBPP. The three nodes use one shared checkout. Update that checkout
exactly once, before starting any worker:

```bash
git pull --ff-only
```

Then run this command on each 4xH100 node, without another pull:

```bash
bash scripts/run_olmo3_rlzero.sh run
```

The command performs all offline model/data checks, real reward-signal
qualification, a four-GPU GRPO update, and checkpoint-resume smoke before it
claims long work. Prepare missing public snapshots once from a machine with the
same shared volume using `bash scripts/run_olmo3_rlzero.sh prepare`. Inspect the
shared queue with `bash scripts/run_olmo3_rlzero.sh status`.

This restart has disjoint run/result roots and does not consume artifacts from
the Qwen or additional-study matrices. See
[`docs/OLMO3_RLZERO_RUNBOOK.md`](docs/OLMO3_RLZERO_RUNBOOK.md) for its immutable
model/data contract, family assignment, restart behavior, and completion checks.

The main matrix uses five seeds, policy steps `0/25/100/400`, four DDP ranks,
eight responses per prompt and a single optimizer epoch per fresh group. It uses
the PPO-form GRPO surrogate with `clip_epsilon=0.2`, but the one-epoch setting has
no post-update reuse of a group; the clip is therefore not claimed as an
effective trust-region constraint. AdamW uses a constant `1e-5` learning rate,
and query/value LoRA uses rank `16`, alpha `32`, and dropout `0`.

The running baseline remains on the command and `...-grpo-v1` roots above. For
the next clean H100 run, use the same launcher with its isolated runtime profile:

```bash
bash scripts/run_olmo3_rlzero.sh run h100
```

Diagnose that profile from any machine mounting the shared volume. The command
waits 20 seconds, checks every active-family log and lock, and reports whether
each worker is progressing, merely alive, stuck, or dead:

```bash
bash scripts/run_olmo3_rlzero.sh status h100
```

The `h100` profile preserves the model, datasets, seeds, checkpoints, GRPO
groups, and optimizer contract. It changes only execution batching: rollout
generation uses batches of eight and response log-probability passes use
micro-batches of four. Projected-gradient scoring also batches four stored
responses per forward pass. Its artifacts go to the separate
`...-grpo-h100-v2` root, and every GRPO step records throughput plus peak GPU
memory so the profile can be measured before any further tuning.

## Previous Qwen Matrix

After one shared-checkout update, run this on each of the three 4xH100 nodes:

```bash
bash scripts/run_rlvr.sh
```

Run the same command on all three nodes. The nodes claim work from shared
`flock` queues, so a seed/dataset family runs once. Each claimed family uses all
four local H100s for GRPO. Interrupted training resumes from a hash-validated
adapter, optimizer, and statistics checkpoint; a complete target-step checkpoint
republishes without repeating the last interval. Rollout partials are bound to
their generation manifest, and a JSONL/manifest rename interruption repairs on
the next run. The launcher admission lock is stored on each node's
local `/tmp`, never on `GROUP_VOLUME`; cloned hostnames therefore cannot collapse
three independent clusters into one worker. Only family and collection locks
are shared.

Do not pull the shared checkout while any launcher is running. After all workers
exit, update it once and start the same command on each node. Each matrix stores
one atomic `generation.git` marker; if the supervisor is newer than a partial
matrix, it creates a node-local standalone clone with its own `.git` directory
at the recorded commit and uses that immutable `run_point.sh` for the remaining
generation. It never adds or prunes entries in the shared repository's
`.git/worktrees`. Mixed or malformed commit provenance aborts before a family is
claimed.

Every positive-drift point inherits the exact `prompts.json` owned by its d0
behavior source. For legacy generation revisions, the supervisor materializes a
hash- and seed-addressed loader snapshot for GSM8K, MATH-500, MBPP,
Knights-and-Knaves, ARC-Challenge, or the registered MMLU-Pro slice. The old pipeline therefore reconstructs
the same split and order for seeds 0-2 before it compares prompts. A mismatched
partial target is moved to quarantine once and rebuilt; a qualification or
base-prompt mismatch is a non-retryable contract error and does not enter the
launcher restart loop.

The canonical launcher preserves the existing `v1` exact/float mathematical
verifier and result roots, so jobs started at revision `295dfea` can resume from
their existing flat local datasets and artifacts. Pinned manifests and symbolic
Math-Verify are requirements only of the separate additional-study launcher.

The command runs:

- 27B primary: GSM8K and MATH-500, seeds 0-4, policy steps 0/25/100/400.
- 7B scale replication: GSM8K and MATH-500, seeds 0-2, the same policy steps.
- Independent R/A/B evaluation: R ranks, the mean of A/B scalar scores is the
  held-out utility reference, and A versus B sets the reliability floor. Final
  labels use 10,000 hierarchical bootstrap replicates per run.
- A hash-bound harvest that packages existing matrix reports without rerunning
  rollouts or bootstrap analysis. Publication revalidates the analysis-cache
  hashes, exact registered cells and selectors, final bootstrap status, and
  JSON/CSV agreement before atomically replacing the readout.

The only user-facing result bundle is replaced in place at:

```text
$OM_WORK/readouts/rlvr-grpo/
```

It contains exactly `REPORT.md`, `RESULTS.json`, `RESULTS.csv`, and
`MANIFEST.sha256`. Unchanged inputs reuse that bundle; extra, partial, stale,
or internally inconsistent files cause rejection or a clean replacement.
Additional experiments write only to their separate domain-specific roots.

Generation provenance is a grouping boundary, not report metadata. Regime
analysis records generation and analysis revisions separately, partitions
diagnostic summaries by generation revision, and refuses to publish the 27B
and 7B matrices together when their generation revisions differ. The canonical
launcher binds both matrices to one shared suite revision before either phase
can claim work; partial matrices force the other phase to resume that revision.
The retired 2026-08-24 mixed-revision readout is documented in
[`docs/results/2026-08-24/PROVENANCE_STATUS.md`](docs/results/2026-08-24/PROVENANCE_STATUS.md).

## Previous Qwen Objective Contract

For every positive policy step, completion requires `policy_train.json` with:

- `training_objective = grpo`
- `policy_update = clipped_policy_gradient`
- `reward_source = verifier`
- `supervised_loss = false`
- `positive_only_filter = false`
- exactly four distributed workers
- a matching adapter hash and complete per-step group/sample/reward accounting

LoRA is the policy parameterization used to fit previous 27B training on each
80 GB H100.
It does not change the objective: gradients come only from the clipped GRPO
surrogate over online verifier-reward samples.

## Generalization Matrix

The additional study has one purpose: test whether the primary ranking conclusion
transfers beyond the OLMo-3/math/code setting. It fixes the GRPO objective and
uses two raw, non-instruction base architectures: sparse
`allenai/OLMoE-1B-7B-0125` (1B active/7B total) and dense
`Qwen/Qwen2.5-14B`. Three separate single-domain matrices cover structured logic
(Knights-and-Knaves), science (ARC-Challenge), and non-math professional
knowledge (a deterministic balanced MMLU-Pro slice). No domain is pooled and no
mixed-domain policy is trained.

Every matrix uses seeds 0-2, the `0/25/100/400` checkpoint chain, one GRPO
optimizer epoch per fresh group, 512 candidate prompts, and 100 validation
prompts. Model and dataset revisions are immutable. This design deliberately
changes architecture, capacity, and domain while holding the learning objective
fixed; it does not add a second paper objective or reuse the main math/code data.

Do not pull or modify a checkout running the primary `295dfea` jobs. First use
one separate online checkout with the same shared volume to prepare and qualify
every model and dataset snapshot without requiring a GPU:

```bash
git clone https://github.com/33modeling/offpolicy-misranking.git \
  ~/offpolicy-misranking-additional
cd ~/offpolicy-misranking-additional
bash scripts/run_additional_experiments.sh --prepare
```

Then run `bash scripts/run_additional_experiments.sh` from a separate clean
checkout on each of the three new 4xH100 nodes. The default run rechecks the
snapshots, blocks on any existing node-specific primary lock without signalling
that process, and starts only after all four GPUs are idle. It runs the logic,
science, and professional-knowledge GRPO matrices in that order.

Each node runs an independent model/runtime smoke test, then claims complete
seed/dataset families from shared queues. The generalization roots, contracts,
quarantine, and reports are separate from the primary Qwen matrix. A dirty
checkout, missing dataset manifest, wrong model revision, non-four-H100 node,
or contract mismatch aborts before a long run starts. MMLU-Pro qualification
also enforces the exact 612-row derived selection, category quotas, unique
questions, answer-index consistency, and its order-independent content hash.

See [docs/EXPERIMENT.md](docs/EXPERIMENT.md) for the algorithm and artifact
layout, and [docs/INCIDENT_RLVR_OBJECTIVE_2026-08-31.md](docs/INCIDENT_RLVR_OBJECTIVE_2026-08-31.md)
for the SFT substitution root-cause record.

## Scripts

- `run_olmo3_rlzero.sh`: main raw OLMo-3 base RL-Zero entry point.
- `run_rlvr.sh`: previous Qwen primary/scale-replication entry point.
- `run_additional_experiments.sh`: wait for primary, then run all additions.
- `run_matrix.sh`: shared multi-node family queue and retry supervisor.
- `run_point.sh`: one training/evaluation point.
- `harvest_results.sh`: report packaging only; no experiment recomputation.
- `setup_env.sh`, `provision.sh`, `fetch_datasets.sh`, `fetch_27b.sh`: setup.
- `check_data.sh`, `check_27b_fla.py`, `diagnose_run_failure.sh`: preflight and diagnosis.
