# Reduced E5: does the selected data actually train better? (2026-09-10)

Question: after selecting the top 10% of MATH-500 prompts with a stale
selector, does further GRPO training on that subset reach the same held-out
reward as training on the fresh selection, and does either beat a random
subset? Gradient alignment is a proxy; this experiment measures reward.

## Design

| item | value |
|---|---|
| source points | OLMo-3 matrix, MATH-500, checkpoint d400, seeds 0 1 2 (`family-math500-s<seed>/…-d400`) |
| arms | `random` (uniform subset), `fresh_r` (fresh top-k from split R), `g11` (stale top-k, full correction) |
| training | 100 further GRPO updates per arm from the point's `policy_step_400` adapter and optimizer, same objective configuration (4 ranks, one epoch, clip 0.2, lr 1e-5, LoRA q/v 16/32) |
| evaluation | 300 MATH-train problems that share no question with the 400 candidates or the 100 ranking-validation prompts, 8 responses each, Math-Verify reward, before (d400 policy) and after every arm |
| readout | mean reward after each arm, paired difference vs fresh_r with a 10,000-draw prompt bootstrap, overlap of each subset with fresh_r |

Cost: 900 updates in total (22% of the 4,000 in the matrix) plus 12 policies
x 300 prompts x 8 responses. Measured matrix throughput on 4xH100 (status
history 09-08/09, MATH-500 d400: 103->142 steps in 45 min, 311->392 in 95 min)
is about 70 s per GRPO update, so one arm takes about 2 h and one seed's three
arms about 6 h. Evaluation at 8 responses per prompt runs about 60 s per prompt
per GPU (responses run to the 2048-token cap), so one policy takes about 1.3 h
and one seed's four evaluations about 5-6 h. One seed is therefore about 12 h:
three nodes finish in about half a day, one node in about 1.5 days.

## Commands (phone-typable)

Once, in a shell with Hub access (login node):

```
bash scripts/fetch_math_train.sh
```

On each idle 4xH100 node (no OLMo or Qwen launcher on it), the same line:

```
git pull --ff-only && bash scripts/run_e5.sh
```

Progress from any node, no GPU:

```
bash scripts/run_e5.sh status
```

`bash scripts/run_e5.sh plan` prints the contracts and commands without
touching a GPU. Rerunning after a kill resumes from the newest five-step
checkpoint or the completed evaluation shards; arms are leased per seed so
several nodes share one seed's arms without duplicating work.

The 2026-09-11 launcher repair also handles an invalid final publication when
a compatible, hash-verified checkpoint remains. It uses the existing trainer's
recovery path without clearing the experiment output. An arm with no verified
repair checkpoint is preserved before continuing to other arms. The recovery
probe is CPU-only.
Scientific source files hashed by `experiment.json` are unchanged, so existing
E5 contracts, subsets, checkpoints and completed evaluations remain reusable.

Do not interrupt healthy training to install this repair or pull over a live
shared launcher. After that launcher exits, pull and rerun the same command;
no new output root or experiment restart is required. OLMo/Qwen status commands
in the repaired revision no longer update the checkout automatically. On the
older revision, use `bash scripts/run_e5.sh status` for E5 and
`bash scripts/status_qwen35.sh` for Qwen without the auto-updating wrapper.

`bash scripts/run_e5.sh status` retains detailed per-shard response counts,
log ages/tails and training checkpoint steps through a separate diagnostic
module. Five-minute console progress does not impose a five-minute wait:
evaluation completion is checked every second. Status labels describe files
present, not independently validated scientific results.

Outputs: `$OM_WORK/runs/e5-reduced/math500-d400/s<seed>/downstream_results.csv`
(and `.json`), one row per arm. Upload the three CSVs when they exist.

Environment knobs (defaults in parentheses): `E5_SEEDS` ("0 1 2"),
`E5_SELECTORS` ("random fresh_r g11"), `E5_STEPS` (100), `E5_EVAL_K` (8),
`E5_TEST_COUNT` (300), `E5_DRIFT` (400).

## What the result means

- fresh_r and g11 both above random, and close to each other: reuse keeps the
  training value of selection at this budget, although its overlap with the
  fresh set is low.
- fresh_r above random, g11 not: reuse loses training value at d400.
- all three similar: the selection signal at this budget does not translate
  into reward within 100 updates; the paper then reports that boundary.

Intervals are prompt-bootstrap intervals conditional on the trained seed; they
describe evaluation noise, not training-seed variability.

This three-arm experiment measures the training value of fixed data selections.
It does not choose among g00/g10/g01/g11 using A/B alignment or fresh-R overlap.
The manuscript's method-choice proposal is a distinct comparison, not a label
for these runs. Do not relabel d400/three-seed/100-update/300-question results
as d100/five-seed/200-update/500-question results, or expand `E5_SELECTORS` in an
already frozen output. Preserve the current experiment and its negative or
positive results alike.
