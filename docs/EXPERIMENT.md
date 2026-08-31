# RLVR Experiment Contract

## Policy update

At optimizer step `t`, every one of four distributed ranks selects one prompt
and samples `K=8` responses from the current policy. The exact-answer or runtime
verifier produces rewards `r_i`. Within each prompt group:

```text
A_i = (r_i - mean(r)) / (std(r) + 1e-4)
ratio_i,u = exp(log pi_theta(a_i,u|h_i,u) - log pi_old(a_i,u|h_i,u))
L = -mean_i,u min(ratio_i,u A_i, clip(ratio_i,u, 0.8, 1.2) A_i)
```

The old token log probabilities are captured when responses are generated.
Each online batch receives two optimization epochs, so the second epoch uses the
clipped ratio against the frozen sampling policy. Gradients are averaged by DDP
across all four ranks. No reference answer text is used as a supervised label.
The registered reference-model KL coefficient is zero; policy movement is
controlled by the old-policy ratio clip and the policy-step sweep.

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

## Multi-node execution

`run_rlvr.sh` requires exactly four H100s on its node. Three nodes may execute it
simultaneously against the same `GROUP_VOLUME`; `run_matrix.sh` locks an entire
seed/dataset family, preserving the ordered checkpoint chain and preventing
duplicate jobs. Report collection is separately locked and content-addressed.
It updates the single `readouts/rlvr-grpo` bundle in place and does not retain
timestamped harvest directories.
