# RLVR Experiment Contract

## GRPO policy update

At optimizer step `t`, every one of four distributed ranks selects one prompt
and samples `K=8` responses from the current policy. The mathematical
equivalence, structured-answer, or isolated runtime verifier produces rewards
`r_i`. Within each prompt group:

```text
A_i = (r_i - mean(r)) / (std(r) + 1e-4)
ratio_i,u = exp(log pi_theta(a_i,u|h_i,u) - log pi_old(a_i,u|h_i,u))
L = -(1/G) sum_i (1/|o_i|) sum_u
    min(ratio_i,u A_i, clip(ratio_i,u, 0.8, 1.2) A_i)
```

The old token log probabilities are captured when responses are generated.
Gradients are averaged by DDP across all four ranks. No reference answer text is
used as a supervised label, and the registered reference-model KL coefficient
is zero.

The main OLMo-3 RL-Zero matrix uses exactly one optimizer epoch for each freshly
sampled group. Current and old log probabilities are therefore evaluated before
the only optimizer step and their ratio is one up to numerical error. The code
still evaluates the PPO-form clipped surrogate with `clip_epsilon=0.2` and
records `policy_update=clipped_policy_gradient`, but this single-epoch design
does not reuse the group after a policy update. Consequently, the paper does not
claim that ratio clipping supplies an effective trust region in the OLMo-3
matrix; policy movement is indexed by the cumulative update-step sweep.

The previous Qwen matrix uses two optimizer epochs per sampled group. Its second
epoch evaluates the frozen sampling-policy ratio after one update, so clipping
can bind. This two-epoch statement does not apply to the main OLMo-3 matrix or
the registered generalization matrices, all of which use one epoch. Dr.GRPO and
RLOO remain implementation-supported diagnostics but are not registered paper
experiments and must not be pooled with the fixed-objective GRPO extension.

## Checkpoint chain

Within each seed/dataset family, policy steps are a single continuous chain:

```text
base -> step 25 -> step 100 -> step 400
```

The later point loads both the preceding adapter and optimizer state. A crash
within a point resumes from the newest fully published `checkpoint-N` and trims
statistics beyond that durable step. The main OLMo-3 matrix uses this chain for
both datasets and all five seeds. The previous Qwen and registered
additional-study matrices retain their own seed counts and disjoint artifact
roots.

Policy step zero is an unchanged base-policy control. Behavior rollouts are
generated once in that control run and contract-validated before reuse by all
positive steps in the same family. Current-policy rollout and score artifacts
are never reused across policy steps.

## Evaluation split

The 32 current-policy rollouts for each candidate form eight four-rollout
micro-groups. They are assigned once as `R=4`, `A=2`, and `B=2` groups. The 100
validation-prompt gradients are independently assigned as `R=50`, `A=25`, and
`B=25`. R alone ranks candidates. The mean of the two scalar scores from A and B
is the held-out utility reference, while A-versus-B top-k agreement defines the
reliability floor. Final regime labels require 10,000 hierarchical bootstrap
replicates per run; fewer replicates are explicitly provisional.

## Completion

Positive-step directories contain:

```text
policy_step_N/
  adapter_config.json
  adapter_model.safetensors
  optimizer.pt
  grpo_stats.jsonl
  policy_train.json
```

`DONE` is written only after the policy contract, rollout contracts, score and
oracle protocols, exact prompt coverage, and required artifacts pass.

The canonical primary launcher retains its `v1` exact/float mathematical
verifier, flat dataset compatibility, and analysis-only migration path so an
existing run can resume. The additional-study launcher alone enables symbolic
Math-Verify and immutable dataset qualification. It uses disjoint run roots and
disables analysis migration across reward, dataset, policy, or generation
protocol boundaries.

## Multi-node execution

`run_olmo3_rlzero.sh run` requires exactly four H100s on its node. Three nodes may
execute the same command simultaneously against the same `GROUP_VOLUME`;
`run_matrix.sh` locks an entire seed/dataset family, preserving the ordered
checkpoint chain and preventing duplicate jobs. Per-node launcher admission
uses a node-local `/tmp` lock rather than hostname-derived state on the shared
volume, so cloned cluster hostnames do not reject valid workers. Report
collection is separately locked and content-addressed. The previous Qwen runner
uses the same family-locking mechanism but writes to a disjoint root.

Before any family claim, the shared matrix is atomically bound to one full Git
commit in `.queue/generation.git`. A supervisor updated after an interruption
automatically materializes that commit as a node-local standalone clone with an
independent `.git` directory and runs the remaining generation there. It does
not mutate the shared checkout's worktree registry. An explicit checkout at the
wrong commit, mixed run-config commits, a malformed marker, or a dirty
generation checkout fails before new work begins.

The previous Qwen matrix and already-running `295dfea` jobs retain the
`readouts/rlvr-grpo` target. Additional-study outputs never enter that bundle.
The OLMo-3 main matrix instead writes below
`$OM_WORK/results/olmo3-1025-7b-base-rlzero-grpo-v1`. None of these paths uses
timestamped harvest directories or pools artifacts across matrices.

The single `run_additional_experiments.sh` entry point has a GPU-free
`--prepare` mode that downloads and qualifies every immutable shared snapshot
under a global lock. Its default run revalidates those inputs, waits on the
node-local primary lock, and applies the same queue and artifact
contracts to every additional stratum. The extension fixes online verifier-reward
GRPO and compares a raw sparse OLMoE-1B-7B base model with a raw dense
Qwen2.5-14B base model. Logic, science, and non-math professional knowledge are
run as separate Knights-and-Knaves, ARC-Challenge, and balanced MMLU-Pro
matrices. Each dataset contributes 512 candidate and 100 validation prompts;
each model/domain cell uses seeds 0-2 and checkpoints 0/25/100/400. The extension
does not train a mixed-domain policy and does not pool domain estimates.

All three nodes run the same command. Each family is claimed once, and every
positive checkpoint resumes from its immediate predecessor. The launcher binds
the clean Git commit, full model revision and file hashes, dataset snapshot
hashes, normalized prompt split hashes, verifier runtime self-tests, and all
hyperparameters into an immutable per-model matrix document. Results remain in
domain-specific roots below `$OM_WORK/results/{generalization-logic-grpo-v1,
generalization-science-grpo-v1,generalization-knowledge-grpo-v1}/`; no policy
checkpoint or generated artifact is reused across domains or mixed with either
the OLMo-3 main matrix or the previous Qwen bundle.
