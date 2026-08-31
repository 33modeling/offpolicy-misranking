# Off-policy Misranking RLVR Experiments

This repository runs the paper experiment with a real verifier-reward GRPO
policy update. The earlier positive-rollout SFT drift is retired and cannot be
entered through the canonical runner.

## Run

On each of the three independent 4xH100 nodes, after pulling the same commit:

```bash
git pull
bash scripts/run_rlvr.sh
```

Run the same command on all three nodes. The nodes claim work from shared
`flock` queues, so a seed/dataset family runs once. Each claimed family uses all
four local H100s for GRPO. Interrupted training resumes from the latest adapter
and optimizer checkpoint.

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

## Objective Contract

For every positive policy step, completion requires `policy_train.json` with:

- `training_objective = grpo`
- `policy_update = clipped_policy_gradient`
- `reward_source = verifier`
- `supervised_loss = false`
- `positive_only_filter = false`
- exactly four distributed workers
- a matching adapter hash and at least one nonzero-advantage reward group

LoRA is the policy parameterization used to fit 27B training on each 80 GB H100.
It does not change the objective: gradients come only from the clipped GRPO
surrogate over online verifier-reward samples.

## Generalization Matrix

The additional study has one purpose: test whether the primary conclusion
generalizes. It varies two distinct factors without pooling them:

- Model/domain transfer fixes GRPO and varies Mistral versus OLMo 2 and GSM8K,
  MBPP, Knights-and-Knaves, and ARC-Challenge.
- Method transfer fixes the two models and math/code domains and varies GRPO,
  Dr.GRPO, and sequence-level RLOO.

Both strata use seeds 0-2 and the `0/25/100/400` checkpoint chain. Model and
dataset revisions are immutable. There is one operational entry point for the
entire additional study.

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

- `run_rlvr.sh`: canonical Qwen primary/scale-replication entry point.
- `run_additional_experiments.sh`: wait for primary, then run all additions.
- `run_matrix.sh`: shared multi-node family queue and retry supervisor.
- `run_point.sh`: one training/evaluation point.
- `harvest_results.sh`: report packaging only; no experiment recomputation.
- `setup_env.sh`, `provision.sh`, `fetch_datasets.sh`, `fetch_27b.sh`: setup.
- `check_data.sh`, `check_27b_fla.py`, `diagnose_run_failure.sh`: preflight and diagnosis.
