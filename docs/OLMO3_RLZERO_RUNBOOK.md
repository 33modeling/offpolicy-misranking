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

## Training hyperparameters

| Component | Registered value |
|---|---|
| Distributed batch | 4 DDP ranks, 1 prompt per rank, 8 responses per prompt (32 responses/update) |
| Reward/advantage | Binary domain verifier; within-prompt population standard deviation; denominator epsilon `1e-4` |
| Policy loss | PPO-form GRPO surrogate, ratio range `[0.8, 1.2]`, one optimizer epoch per fresh group |
| Reference regularization | KL coefficient `0` |
| Optimizer | AdamW, learning rate `1e-5`, no scheduler, gradient-norm cap `1` |
| AdamW defaults | `betas=(0.9, 0.999)`, `eps=1e-8`, `weight_decay=0.01`, `amsgrad=false` |
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

Execute the same command from a clean checkout on every 4xH100 node:

```bash
git pull
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

## Restart and Git updates

An interrupted rollout keeps only exact-K complete prompt groups in `.partial`
and continues from the next prompt. The partial file is bound to its generation
manifest; incompatible restart state is quarantined instead of mixed. If the
worker stops after publishing JSONL but before its manifest rename, the next run
validates the exact rows and finishes that publication automatically. Interrupted
GRPO loads the newest adapter/optimizer/statistics checkpoint, including a
complete target-step checkpoint when only final publication remained. A family
failure no longer exits the launcher: it releases that
family lock, preserves the partial artifacts, processes other available
families, and retries failed families after 60 seconds while the GPU keepalive
remains active. `OM_RLZERO_FAMILY_ATTEMPTS` controls immediate attempts and
`OM_RLZERO_FAMILY_RETRY_SECONDS` controls the outer retry delay.

Only a user signal or loss of the worker itself requires rerunning the command:

```bash
bash scripts/run_olmo3_rlzero.sh run
```

The first worker atomically writes the experiment-wide generation commit to:

```text
$OM_WORK/runs/olmo3-1025-7b-base-rlzero-grpo-v1/.queue/generation.git
```

After a launcher exits, `git pull` is allowed. A newer supervisor restores a
node-local detached worktree at the recorded generation commit and runs all
unfinished and not-yet-started families with that same code. Do not delete the
generation marker, family completion stamps, or partial checkpoints. A missing
local Git object aborts before a GPU family is claimed.

Prompt/contract failures use exit code `43` to skip immediate same-family
attempts. The worker stays allocated, moves to other work, and retries that
family on a later queue pass instead of terminating. Repeated attempts reuse
only contract-valid durable artifacts.

## Observe and collect

Queue state:

```bash
bash scripts/run_olmo3_rlzero.sh status
```

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
