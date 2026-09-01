# Off-policy Misranking RLVR Experiments

This repository runs the paper experiment with a real verifier-reward GRPO
policy update. The earlier positive-rollout SFT drift is retired and cannot be
entered through the canonical runner.

## Clean OLMo-3 Restart

The clean causal experiment starts from the raw, non-SFT
`allenai/Olmo-3-1025-7B` base checkpoint and uses verifier-reward GRPO on
MATH-500 and MBPP. On each of the three new 4xH100 nodes:

```bash
git pull
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

## Previous Qwen Matrix

On each of the three independent 4xH100 nodes, after pulling the same commit:

```bash
git pull
bash scripts/run_rlvr.sh
```

Run the same command on all three nodes. The nodes claim work from shared
`flock` queues, so a seed/dataset family runs once. Each claimed family uses all
four local H100s for GRPO. Interrupted training resumes from the latest adapter
and optimizer checkpoint. The launcher admission lock is stored on each node's
local `/tmp`, never on `GROUP_VOLUME`; cloned hostnames therefore cannot collapse
three independent clusters into one worker. Only family and collection locks
are shared.

Do not pull a checkout while its launcher is running. After that process exits,
the checkout may be updated and the same command run again. Each matrix stores
one atomic `generation.git` marker; if the supervisor is newer than a partial
matrix, it creates a node-local detached worktree at the recorded commit and
uses that immutable `run_point.sh` for the remaining generation. Mixed or
malformed commit provenance aborts before a family is claimed.

Every positive-drift point inherits the exact `prompts.json` owned by its d0
behavior source. For legacy generation revisions, the supervisor materializes a
hash- and seed-addressed loader snapshot for GSM8K, MATH-500, MBPP,
Knights-and-Knaves, or ARC-Challenge. The old pipeline therefore reconstructs
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
Additional experiments write only to their separate method-specific roots.

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

The additional study has one purpose: test whether the primary conclusion
generalizes. It varies two distinct factors without pooling them:

- Model/domain transfer fixes GRPO and varies Mistral versus OLMo 2 and GSM8K,
  MATH-500, MBPP, Knights-and-Knaves, and ARC-Challenge.
- Method transfer fixes the two models and math/code domains and varies GRPO,
  Dr.GRPO, and sequence-level RLOO.

Both strata use seeds 0-2 and the `0/25/100/400` checkpoint chain. Model and
dataset revisions are immutable. There is one operational entry point for the
entire additional study. Candidate counts are registered per dataset: MATH-500
uses 400, the other four use 512, and every dataset uses 100 validation prompts.

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
that process, and starts only after all four GPUs are idle. It runs the GRPO
model/domain stratum, then the Dr.GRPO and RLOO method strata.

Each node runs an independent model/runtime smoke test, then claims complete
seed/dataset families from shared queues. The generalization roots, contracts,
quarantine, and reports are separate from the primary Qwen matrix. A dirty
checkout, missing dataset manifest, wrong model revision, non-four-H100 node,
or contract mismatch aborts before a long run starts.

The method stratum is a compute-bounded slice. It compares
the full-matrix GRPO cells against Dr.GRPO and sequence-level RLOO on both model
families and the GSM8K and MBPP domains with the same seeds and checkpoint chain.

Dr.GRPO keeps online verifier rewards and the clipped old-policy ratio, but
removes per-question reward-standard-deviation scaling and replaces
response-length normalization with the fixed generation budget. RLOO treats a
complete response as one action, uses the other seven online rewards as its
baseline, and takes one unclipped sequence-level REINFORCE epoch. Neither is
SFT or a relabeled GRPO artifact; each method is bound into its own config,
paths, matrix contracts, checkpoints, and reports.

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
