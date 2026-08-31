# RLVR Experiment Contract

## Policy update

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
Each online batch receives two optimization epochs, so the second epoch uses the
clipped ratio against the frozen sampling policy. Gradients are averaged by DDP
across all four ranks. No reference answer text is used as a supervised label.
The registered reference-model KL coefficient is zero; policy movement is
controlled by the old-policy ratio clip and the policy-step sweep.

The training-method robustness slice uses Dr.GRPO. For the same sampled group,
it sets `A_i = r_i - mean(r)` and divides each response's summed clipped token
surrogate by the fixed `max_new_tokens=512` budget rather than its realized
length:

```text
L_Dr = -(1/G) sum_i (1/512) sum_u
       min(ratio_i,u A_i, clip(ratio_i,u, 0.8, 1.2) A_i).
```

This changes two optimization normalizers only. Sampling, verifier rewards,
four-rank DDP, policy-ratio clipping, optimizer, LoRA parameterization, and the
0/25/100/400 chain remain fixed. The implementation records
`training_objective=dr_grpo`, and the policy validator rejects it unless the
caller explicitly requests that method.

The second robustness method is sequence-level RLOO. For $G=8$ online
responses it uses

```text
b_i = (1/(G-1)) sum_(j != i) r_j
L_RLOO = -(1/G) sum_i (r_i - b_i) sum_u log pi_theta(a_i,u | h_i,u).
```

RLOO uses exactly one epoch over each freshly sampled group, no old-policy
ratio clipping, and no response-length normalization. The zero reference-KL
coefficient is retained. Its manifest records
`training_objective=rloo` and `policy_update=reinforce_leave_one_out`.

## Checkpoint chain

Within each seed/dataset family, policy steps are a single continuous chain:

```text
base -> step 25 -> step 100 -> step 400
```

The later point loads both the preceding adapter and optimizer state. A crash
within a point resumes from the newest fully published `checkpoint-N` and trims
statistics beyond that durable step. The 27B primary and 7B replication use the
same chain; the primary uses five seeds and the replication uses three.

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

`run_rlvr.sh` requires exactly four H100s on its node. Three nodes may execute it
simultaneously against the same `GROUP_VOLUME`; `run_matrix.sh` locks an entire
seed/dataset family, preserving the ordered checkpoint chain and preventing
duplicate jobs. Report collection is separately locked and content-addressed.
The Qwen primary and already-running `295dfea` jobs retain the
`readouts/rlvr-grpo` target. Additional-study outputs never enter that bundle.
Neither path uses timestamped harvest directories.

The single `run_additional_experiments.sh` entry point has a GPU-free
`--prepare` mode that downloads and qualifies every immutable shared snapshot
under a global lock. Its default run revalidates those inputs, waits on any
existing node-specific primary lock, and applies the same queue and artifact
contracts to every additional stratum. The GRPO stratum varies
Mistral-7B-Instruct-v0.3 and
OLMo-2-1124-7B-Instruct over GSM8K, MBPP, Knights-and-Knaves, and ARC-Challenge.
The method stratum uses those models and the GSM8K/MBPP pair to compare GRPO,
Dr.GRPO, and RLOO. These are distinct generalization factors, not separate paper
objectives and not pooled estimates.

All three nodes run the same command. Each family is claimed once, and every
positive checkpoint resumes from its immediate predecessor. The launcher binds
the clean Git commit, full model revision and file hashes, dataset snapshot
hashes, normalized prompt split hashes, verifier runtime self-tests, and all
hyperparameters into an immutable per-model matrix document. Results remain in
method-specific roots below `$OM_WORK/results/{generalization-grpo-v1,
method-dr-grpo-v1,method-rloo-v1}/`; no policy checkpoint or generated artifact
is reused across methods or mixed with the primary Qwen bundle.
