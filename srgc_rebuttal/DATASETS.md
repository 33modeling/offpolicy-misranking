# Running the switching experiment on other datasets

## Integrated MBPP option

```bash
python scripts/run_srgc_rebuttal.py prepare --dataset mbpp
python scripts/run_srgc_rebuttal.py worker --dataset mbpp
python scripts/run_srgc_rebuttal.py status --dataset mbpp
python scripts/run_srgc_rebuttal.py costs --dataset mbpp
```

Run `prepare` once, then `worker` on every allocated four-GPU node. Cache
generation, shared-prefix training and the four arms are automatically queued.
See [CLUSTER.md](CLUSTER.md) for two-node and SSH launch commands.

The integrated option pins MBPP to the operational repository's release
`4bb6404fdc6cacfda99d4ac4205087b89d32030c`, full configuration, all published
splits. It makes one new 400/100/300 split with split seed 0 shared by training
seeds 5–9, and fixes 50 ranking-validation prompts. This is a new reproduction
cohort, not the official MBPP test split or a relabeling of historical results.
The verifier runs the assertion tests, not the math verifier. `prepare --rows`
accepts local question/answer JSONL and records its source hash. Existing
different inputs are never overwritten. Prepared rewards remain empty until
GPU generation; this option does not claim new experimental results.

The code verifier's isolated Python process and resource limits are not a
security sandbox. Run generated-code experiments on disposable compute with
no credentials or sensitive files accessible to the verifier account.

The lower-level per-bundle tools below remain available. Their default split
seed follows `--seed`, unlike the integrated cohort's fixed split seed 0.

The V7 reference implementation only needs an `srgc-inputs-v1` bundle per seed
and a plan that names the verifier. The tools below build both for MATH train,
GSM8K, MBPP or any JSONL file of `{question, answer}` rows. The current protocol
uses a 25-update shared prefix, total 275, 40 fresh scored candidates plus the
next four unused SR prompts every 25 updates, retaining the On-policy top four
for training, checks every 25 updates and projection 4096. SR-GC compares the
selected four on each side. Random and SR use the whole candidate pool without
replacement within each pass; SR follows cached score order. These rules apply
to both MATH and MBPP and differ from the older fixed-subset protocol.

## 1. Build the bundle (CPU)

```bash
# MBPP (code verifier) or gsm8k / math_train (math verifier); Hugging Face `datasets` required
python -m srgc_rebuttal.build_inputs --dataset mbpp --seed 5 \
    --output srgc_rebuttal/inputs/mbpp-seed-5.json

# Any JSONL file: {"question": ..., "answer": ...} per line; --kind picks prompt + verifier
python -m srgc_rebuttal.build_inputs --dataset jsonl --rows my.jsonl --kind math --seed 5 \
    --output srgc_rebuttal/inputs/mine-seed-5.json
```

The split seed defaults to the experiment seed (`--split-seed` overrides it),
so each seed draws its own 400 / 100 / 300 split from the shuffled distinct
questions. `ranking_validation_ids` are the first 50 of the shuffled
validation pool (`--ranking-validation N` changes the size). Prompts use the
released OLMo RL-Zero math or code format verbatim. For MBPP the question
embeds the test assertions, as in the original experiment, and `answer` holds
the assertions the verifier executes. Datasets with fewer than 800 distinct
questions (MATH-500 alone) are rejected; use math_train or supply at least 800 suitable JSONL records instead.

## 2. Cache the initial-policy rewards (one node, four GPUs)

```bash
torchrun --standalone --nproc_per_node=4 -m srgc_rebuttal.build_cache \
    --bundle srgc_rebuttal/inputs/mbpp-seed-5.json \
    --verifier srgc_rebuttal.verifiers:code_reward \
    --model allenai/Olmo-3-1025-7B --model-revision <40-hex revision from the plan>
```

Eight sampled responses per candidate (temperature 1, top-p 1, no top-k,
2048 new tokens) from the base model are verified and written into
`cached_rewards`; raw responses go to `<bundle>.cache-responses.jsonl`. The
run is resumable (already cached candidates are skipped). Reuse an existing
cache with `build_inputs ... --cache other-bundle.json` when the candidate
IDs overlap. Validate any finished bundle with `--check`:

```bash
python -m srgc_rebuttal.build_inputs --dataset mbpp --seed 5 \
    --output srgc_rebuttal/inputs/mbpp-seed-5.json --check
```

## 3. Write the plan

```bash
python -m srgc_rebuttal.plan_dataset --dataset mbpp --seeds 5 6 7 8 9 \
    --verifier srgc_rebuttal.verifiers:code_reward
# -> srgc_rebuttal/experiments/mbpp_seeds.json, inputs at ../inputs/mbpp-seed-{seed}.json,
#    outputs at ../runs/mbpp-seeds/
```

Only `dataset`, `seeds`, `verifier`, `input_pattern`, `output_root` and the
outcome sentence change; the generated file is re-read through the strict
`load_plan` so every frozen key is present and valid.

## 4. Run exactly as for seeds 5–9, with `--plan`

```bash
PLAN=srgc_rebuttal/experiments/mbpp_seeds.json
python -m srgc_rebuttal.plan --plan $PLAN --check-inputs
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m srgc_rebuttal.cluster worker --plan $PLAN   # per node
python -m srgc_rebuttal.cluster status --plan $PLAN
python -m srgc_rebuttal.summarize --plan $PLAN
```

Verifiers (`srgc_rebuttal/verifiers.py`):

- `math_reward`: the manuscript's math-verify check (re-exported from `run_experiment`).
- `code_reward`: extracts the last fenced Python block, appends the record's
  assertions and runs them in an isolated interpreter (`python -I`) with a
  10 s wall clock, CPU and 1 GB memory limits (`SRGC_CODE_TIMEOUT`,
  `SRGC_CODE_MEMORY_MB`); any failure, timeout or missing code gives 0.

Tests: `python -m unittest srgc_rebuttal.tests.test_build_inputs`.

## Seeds 5–9 from the seed 3 and 4 source data (no GPU)

`import_pair_inputs.py` rebuilds MATH bundles from the selector-pair runs on
the cluster volume: `prompts.json` (400 train = candidates, 100 val =
validation pool, R = first 50), `rollouts_behavior_train.jsonl` (eight
initial-policy rewards per candidate = cache) and the 300 held-out
`evaluation.test` questions recorded in `branches/*/switch.json`.

```bash
# on the cluster, from the shared V7 checkout
python -m srgc_rebuttal.import_pair_inputs --pair-root $PAIR_ROOT --output-dir srgc_rebuttal/inputs
# default assignment 5=3 6=4 7=3 8=4 9=3; override with --assign 5=3 6=3 ...
python -m srgc_rebuttal.plan --check-inputs
```

Without a pair root, pass `--source 3=<run dir> --source 4=<run dir>
--evaluation <test.json>`. Every bundle is validated with `validate_inputs`
before it is written and records the source hashes in `provenance`.

The prepared default MATH cohort uses a single new split shared by training seeds 5–9.
Importing the historical seed-3/4 splits defines a different data cohort: use a
separate input directory and plan/output root. The importer refuses to overwrite
different existing bundles. Its explicit first-50 validation choice for new runs
does not establish what online validation set the submitted runs used.
