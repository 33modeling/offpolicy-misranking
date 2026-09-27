# SR-GC rebuttal: online switching and extra-seed cost experiments

Canonical code repository: `33modeling/offpolicy-misranking`, branch
`master`. Executable code, tests and input bundles
are maintained here, not in the manuscript repository. Use the Python entry
point `scripts/run_srgc_rebuttal.py`; see [COSTS.md](COSTS.md) for commands,
stage definitions and accounting boundaries.

Node-parallel execution now includes missing-cache generation. Run
`python scripts/run_srgc_rebuttal.py worker --dataset math` on every allocated
four-GPU node. MBPP is selected with `--dataset mbpp`; its real seed-5--9 input
bundles are included in the checkout and need no separate preparation. Dataset
queues and results are separate. See [CLUSTER.md](CLUSTER.md) for two-node
execution, SSH launch, status, stop/resume and bounded retries.

Independent implementation for new rebuttal runs, using the author's
2026-09-27 clarification: refresh 40 candidates every 25 updates and train
on their top four until the next refresh.
It implements new runs; it does not certify the implementation or numerical
results of the previously reported experiments. The manuscript and its data
are not modified by these programs.

V7 adds a shared-storage multi-node queue and batches gradient computation.
For the fastest available parallel execution, use [CLUSTER.md](CLUSTER.md).
The existing runner files and methods consulted are recorded in
[EXISTING_CODE_REVIEW.md](EXISTING_CODE_REVIEW.md). V6 remains the submission archive.

## Exact online computation

1. Rank the full 400-prompt pool using eight cached binary rewards per candidate:
   `score = -abs(mean(rewards) - 0.5)`, with seeded tie breaking. Cached scores
   remain fixed. For SR training, draw 40 distinct random candidates from the
   full pool and take the four highest cached scores within that draw.
2. While On-policy selection is active, draw 40 candidates uniformly without
   replacement within each draw from the 400-prompt pool every 25 updates.
   Previously drawn candidates may reappear at a later refresh. Compute fresh
   current-policy gradients for these 40 candidates **and the next 40 unused
   SR prompts**. Generate eight responses per distinct prompt. Reuse a single
   gradient for overlap; the union has at most 80 prompts. Merely scoring an
   SR prompt does not mark it as trained.
3. Candidate scoring uses two leave-one-out groups of four responses. Each
   gradient sums response-token contributions and averages the eight responses.
   Each validation prompt uses one group of eight. Average validation gradients.
   A fixed 4096-dimensional CountSketch uses the final four dense layers and
   final normalization. These are scoring gradients, not LoRA optimizer gradients.
4. Rank the 40 On-policy candidates by cosine alignment with the validation
   gradient. Select the four highest-scoring **problems**, with seeded tie breaking.
   Keep these four for the next 25 optimizer updates, generating fresh training
   responses on every update. Save candidate and selected IDs for resume.
5. At a scheduled check, compute

   ```text
   on_mean = mean(projected_gradients[i] for i in all_40_on_candidates)
   sr_mean = mean(projected_gradients[i] for i in next_unused_sr_40)
   D = dot(projected_validation_mean, on_mean - sr_mean)
   ```

   D compares 40 prompts on each side, independently of the four-prompt training
   batch. The On-policy 40 are randomly sampled candidates, not a gradient-selected
   top 40 from the entire pool. D is an inner product, not a cosine. No extra generation, differentiation or A/B
   diagnostic reference batch is requested to compute D. Its sign favors SR
   when negative.
6. Check every 25 updates. Two consecutive negative checks trigger switching.
   For negative/nonnegative/negative checks, switch only if their three-value
   mean is negative. Equality does not switch. If the middle value offsets
   both negatives, retain only the latest negative as a new window. A
   nonnegative third check clears the window. Missing scheduled measurements
   break confirmation; this is not a cumulative mean.
7. Generate a separate eight training responses for each of the four selected
   problems. GRPO uses population-standardized rewards, epsilon `1e-4`, token
   means, clip `[0.8,1.2]`, one epoch and zero reference KL. RLOO uses eight-response
   leave-one-out advantages and token sums. AdamW uses learning rate `1e-5`,
   betas `(0.9,0.999)`, epsilon `1e-8`, zero weight decay, and gradient-norm clip 1.
8. On switching, retain model and optimizer state, stop gradient scoring and D
   checks. On that update and every subsequent SR update, draw 40 distinct
   random candidates from the full 400 and train on their SR-score top four.
   The SR arm uses this same rule from the start of its continuation. Its
   random training candidates are separate from the diagnostic SR comparison.
9. Random draws 40 distinct random candidates from the full 400, then chooses
   four of those 40 uniformly without replacement, on **every optimizer update**.
   There is no fixed 40-prompt subset or 25-update batch reuse, and no gradient scoring.

For both MATH and MBPP, Random/SR redraw their 40 candidates every update;
On-policy redraws its 40 at each scheduled refresh. No draw contains duplicate
IDs. A prompt may reappear in a later draw, including one previously trained.
This supersedes the intermediate global non-repeating-pass training rule.
On-policy still retains its selected batch for 25 updates; its selection rule
and the temporal confirmation rule are unchanged.

These author-requested Random/SR changes date to 2026-09-28. SR-GC retains
the original 40-vs-40 comparison; the interim four-vs-four change is superseded.
The new 400-to-40-to-four sampling defines a new protocol, not a relabeling of old
results. Old checkpoints cannot resume under it. Keep in-flight/archived runs
on their original code and do not bypass implementation-hash checks.
Checkpoints persist trained IDs for the unchanged unused-SR diagnostic preview.
Per-update records retain all 40 training candidate IDs and the selected four
for Random/SR; refresh records retain the On-policy and diagnostic SR sets.
Sampling/ranking CPU time is included in the measured Random/SR training phase.

The shell launcher now uses the shared fresh cohort `candidate40-v2` for both
datasets. Stop old workers before updating and restarting. The new cohort
does not import old caches, checkpoints or queue state; existing files remain
untouched. Repeated launches of this cohort join/resume it, including on a
second node. Reports follow the active cohort.

The controller's `step=t` denotes **t completed optimizer updates**. A check
at t scores that checkpoint's policy before update t+1; a triggered transition
affects update t+1. The first Switch check after the shared prefix is at t=25.
This explicitly fixes an update/check ordering that must also be used when
comparing new logs against old checkpoints.
The shared prefix scores at t=0; continuations refresh at t=25,50,...,250
while using On-policy. Step 275 is the final evaluation, with no further
selection for an update that will never be performed.

## Files and manuscript mapping

| File | Responsibility | Manuscript |
| --- | --- | --- |
| `srgc.py` | 40+40 diagnostic, random candidate-40 draws, method-specific top/random four, four arms, resume | New sampling protocol for V7; not the frozen submission |
| `objectives.py` | LOO, GRPO/RLOO equations, cosine, fixed projection reference | Section 2, Appendix C |
| `torch_backend.py` | Fresh generation, dense scoring derivatives, LoRA optimization, distributed reductions | Appendix C |
| `run_experiment.py` | One seed, common prefix, four continuations, endpoint evaluation/checkpoints | Online experiments |
| `cluster.py`, `runtime.py` | Cross-node task leases, prefix dependencies, restart and experiment identity checks | New execution infrastructure |
| `experiments/additional_seeds.json` | Fixed additional-seed protocol | New experiments, no outcomes yet |
| `summarize.py` | Per-seed outcomes, paired differences, mean and sample SD | Analysis of new runs |
| `timing.py`, `cost_ledger.py`, `cost_report.py` | Exclusive stage timers, rank aggregation, durable receipts, paired cost comparisons | New measurements only |
| `toy_backend.py`, `run_demo.py` | Executable CPU example using a small categorical policy | Software validation only |

The separate retrospective A/B analysis, full-pool diagnostics and historical
off-policy experiments are not replayed by this online implementation.

## Quick CPU verification

From the repository root:

```bash
python -m pip install numpy
python -m unittest discover -s srgc_rebuttal/tests -v
python -m srgc_rebuttal.run_demo --seeds 5 6 --updates 275 --prefix 25 --output /tmp/srgc-demo.json
```

The example samples responses and updates an analytic policy, but its rewards
and runtimes are synthetic CPU results, never paper results. Its 50 validation
prompts are a demo choice. PyTorch/OLMo tests are skipped without optional dependencies.

For the model adapter and tiny randomly initialized OLMo-3 tests:

```bash
# Use the existing Pair/MBPP Python environment; do not reinstall its packages.
python -m unittest discover -s srgc_rebuttal/tests -v
torchrun --standalone --nproc_per_node=2 -m srgc_rebuttal.tests.distributed_smoke
```

These tests do not download pretrained weights. The distributed smoke test
uses two CPU processes to compare the global four-prompt update against a
single-process result. Install a CUDA-capable PyTorch build for production.
API references: [OLMo-3 in Transformers](https://huggingface.co/docs/transformers/model_doc/olmo3),
[PyTorch autograd.grad](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad.html).

## Additional seeds 5–9

The real MATH-train inputs are now prepared. Start with
[REBUTTAL_READY.md](REBUTTAL_READY.md) and the
[preparation manifest](experiments/prepared_inputs.json). Initial-policy reward
caches still require GPU generation; no new scientific outcomes are present.

Five new seeds are configured separately from the existing seeds 3 and 4.
Each uses a 25-update shared On-policy prefix, then Random, SR, On-policy and
Switch continue to total update 275. The threshold and temporal rule are fixed.
Report every new seed, including unfavorable or non-switching outcomes; do not
treat questions or checkpoints as independent training seeds.

Prepare `inputs/seed-5.json` through `inputs/seed-9.json` with this structure:

```json
{
  "schema": "srgc-inputs-v1",
  "records": {
    "example-id": {"question": "raw question", "prompt": "exact formatted RL-Zero prompt", "answer": "$2$"}
  },
  "candidate_ids": [],
  "validation_pool_ids": [],
  "ranking_validation_ids": [],
  "evaluation_ids": [],
  "cached_rewards": {"example-id": [0, 1, 0, 1, 0, 1, 0, 1]},
  "provenance": {"split_seed": 0, "prompt_format": "record the source", "cache": "record the source", "verifier": "record the configuration"}
}
```

This is a schema illustration, not usable experimental data: supply all 400
candidate, 100 validation-pool and 300 evaluation records, disjoint by ID and
normalized question text, and eight existing cache rewards per candidate.
The same input is used by all four arms within each seed.

The current manuscript does not separately specify the online ranking-validation
subset: the 50/25/25 split is stated for diagnostics. Consequently the runner
requires explicit `ranking_validation_ids` rather than silently inferring the
online subset from the diagnostic description. Also provide the original
formatted prompts and confirm the verifier extraction settings. The default
`math_reward` uses math-verify 0.9.0; the plan accepts a `module:function` verifier
to match the original configuration. These input details are needed before a
new 7B run can be claimed to reproduce the original protocol exactly.

Inspect commands and input status without loading a model or launching jobs:

```bash
python -m srgc_rebuttal.plan
python -m srgc_rebuttal.plan --check-inputs
```

Run one seed on four allocated GPUs, after the input check succeeds:

```bash
torchrun --standalone --nproc_per_node=4 -m srgc_rebuttal.run_experiment --seed 5
```

Repeat with seeds 6–9 using the commands emitted by `plan`. Generation uses
BF16/eager OLMo-3 7B, temperature 1, top-p 1, no top-k truncation, and 2048 new
tokens maximum. LoRA has rank 16, alpha 32, zero dropout and query/value targets.
Each rank trains one of four problems with eight fresh responses. Scoring and
evaluation prompts are sharded across ranks and gathered by prompt ID.

Outputs go to `runs/additional-seeds/seed-N/`: an input/plan-hashed manifest,
common-prefix model+optimizer checkpoint, arm checkpoints every 25 updates,
progress with each decision and selected IDs, and final question-wise rewards.
Use `--resume` only with the same plan and inputs. Existing results are not
overwritten. Checkpoints are local trusted files generated by this runner.

The cluster queue generates a missing cache first, freezes its final input
hash, runs `--task prefix` once per seed and dispatches
`--task random|sr|on_policy|switch` to independent nodes after that seed's
verified prefix completes. All four arms load the identical saved model and
optimizer. Each endpoint records its prefix hash; the summary rejects arms
with different prefix, input, plan or implementation hashes.

The default log-probability micro-batch is two responses. Candidate counts,
rollout counts, objectives and update counts are unchanged. Scoring accumulates
the response gradients before one projection per prompt; unrelated adapter
derivatives are disabled during scoring and restored for training. Chunked
LM-head computation with activation recomputation bounds vocabulary-sized
temporaries. These optimizations are numerically checked on a tiny model;
their 7B throughput and peak GPU memory have not been measured.

```bash
python -m srgc_rebuttal.summarize
```

The summary requires all planned seeds and all four arms. It reports paired
Switch-minus-On-policy and Switch-minus-SR differences, their mean and sample
standard deviation across seeds. New seeds are reported separately; no
significance claim or favorable-seed filtering is automated.

## Time accounting and validation limits

Selection timers include the distinct-prompt union, validation generation,
gradient computation, projection and ranking. They run on each 25-update refresh,
and stop after switching. Training is timed separately. Timers synchronize
the device; GPU-seconds are elapsed seconds multiplied once by allocated GPU
count. CPU examples report zero GPU-seconds. No recorded total is multiplied
by the number of updates.

The D arithmetic timer is an informational subset of selection time, not a
second additive cost. The detailed report meters actual preparation on each
task; the legacy SR preparation field remains informational and is not added
again to measured preparation. Cache creation and the shared prefix are
reported separately from each continuation. Endpoint evaluation, model setup,
checkpoint snapshots/writes/loads and complete invocations are also measured.
No historical paper time is embedded in the implementation.

Durable receipts retain completed charges across restarts, including repeated
work after a checkpoint. A start without a finish marks incomplete measurement;
it is never imputed as zero. Rank-local nested stages are exclusive and
reconcile to their synchronized phase with explicit waiting/unattributed time.
Inclusive invocation totals are not added to their component phases. See
[COSTS.md](COSTS.md) for allocation, cache and shared-prefix accounting.
Additional-seed jobs are prepared, not executed by this commit.
Tiny-model CPU tests establish numerical and software behavior; four-GPU
OLMo-3 7B performance and the new scientific outcomes remain to be measured.
