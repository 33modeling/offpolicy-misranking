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

The audited launcher writes a new `v2-mathverify` matrix and disables implicit
analysis upgrades. The three jobs already started at revision `295dfea` remain
`v1` artifacts and must finish under that revision; they are never opened,
relabelled, or harvested as `v2` results.

The command runs:

- 27B primary: GSM8K and MATH-500, seeds 0-4, policy steps 0/25/100/400.
- 7B scale replication: GSM8K and MATH-500, seeds 0-2, the same policy steps.
- Independent R/A/B evaluation: R ranks, the mean of A/B scalar scores is the
  held-out utility reference, and A versus B sets the reliability floor. Final
  labels use 10,000 hierarchical bootstrap replicates per run.
- A hash-bound harvest that packages existing matrix reports without rerunning
  rollouts or bootstrap analysis.

The only user-facing result bundle is replaced in place at:

```text
$OM_WORK/readouts/rlvr-grpo-v2-mathverify/
```

It contains exactly `REPORT.md`, `RESULTS.json`, `RESULTS.csv`, and
`MANIFEST.sha256`. The legacy `v1` launcher retains its own
`$OM_WORK/readouts/rlvr-grpo/` bundle. Unchanged inputs reuse their versioned
bundle; no protocol version overwrites another.

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

The preregistered cross-domain replication uses two independently pretrained
7B instruction-model families, Mistral and OLMo 2, and four verifier domains:
GSM8K arithmetic, MBPP code execution, Knights-and-Knaves logic, and
ARC-Challenge science multiple choice. It uses seeds 0-2 and the same
0/25/100/400 policy-step chain. Model and dataset revisions are immutable and
the complete matrix is defined in `configs/domain_transfer.json`.

Provision the pinned shared snapshots once from an online shell:

```bash
git pull
bash scripts/provision_generalization.sh
```

Then run this same command on all three 4xH100 nodes:

```bash
git pull
bash scripts/run_generalization.sh
```

Each node runs an independent model/runtime smoke test, then claims complete
seed/dataset families from shared queues. The generalization roots, contracts,
quarantine, and reports are separate from the primary Qwen matrix. A dirty
checkout, missing dataset manifest, wrong model revision, non-four-H100 node,
or contract mismatch aborts before a long run starts.

Training-method robustness is a separate, compute-bounded slice. It compares
the full-matrix GRPO cells against Dr.GRPO and sequence-level RLOO on both model
families and the GSM8K and MBPP domains with the same seeds and checkpoint
chain:

```bash
bash scripts/run_method_robustness.sh
```

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
- `run_generalization.sh`: pinned cross-model and cross-domain replication.
- `run_method_robustness.sh`: separate Dr.GRPO and RLOO math/code slices.
- `provision_generalization.sh`: pinned transfer models/data plus qualification.
- `run_matrix.sh`: shared multi-node family queue and retry supervisor.
- `run_point.sh`: one training/evaluation point.
- `harvest_results.sh`: report packaging only; no experiment recomputation.
- `setup_env.sh`, `provision.sh`, `fetch_datasets.sh`, `fetch_27b.sh`: setup.
- `check_data.sh`, `check_27b_fla.py`, `diagnose_run_failure.sh`: preflight and diagnosis.
