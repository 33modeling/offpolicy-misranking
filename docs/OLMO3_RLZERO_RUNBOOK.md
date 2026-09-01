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
   Math-Verify/bubblewrap verifier checks.
3. Cached real OLMo generations with positive, negative, and within-prompt mixed
   verifier rewards for both MATH-500 and MBPP.
4. A four-rank GRPO step followed by an adapter/optimizer resume to step two on
   that physical node.

No family is claimed when any gate fails.

## Restart and Git updates

An interrupted rollout keeps only complete prompt groups in `.partial` and
continues from the next prompt. Interrupted GRPO loads the newest complete local
checkpoint. Re-run the same command:

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

Permanent prompt/contract failures use exit code `43` and are not placed in an
infinite retry loop. Other point failures receive bounded retries and preserve
durable artifacts for the next launcher invocation.

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
