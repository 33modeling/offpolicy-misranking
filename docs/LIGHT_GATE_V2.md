# V2 One-Shot Screen

V2 adopts the v3 measurement-first direction with corrected theory, cost
accounting and a training-reward endpoint. This is a new additive experiment,
not a change to the running E5, Qwen, OLMo matrix or the v3 draft.

## What Runs

One existing 400-prompt, eight-response behavior cache is scanned once.
For each prompt the score is `-abs(mean_reward - 0.5)`, exactly the existing
`passrate_beta` difficulty score. There is no new predictor, gradient scoring,
rollout generation or bootstrap in the screen. The name `beta` refers to the
behavior policy; this implementation does not add Bayesian smoothing.

The default exploratory rule requires positive actual-score half correlation,
score spread, mixed-group fraction and selected-minus-pool mixed fraction.
The scan must fit 1% of the allocated budget and a 30-second wall cap. Cache
age must be at most 400 updates in the fixed OLMo protocol. Undefined or
negative correlation, constant scores, invalid coverage, failed measurement
and excessive cost choose random. These are unvalidated heuristic thresholds,
not a guarantee that selection helps. A single random split is descriptive;
no confidence bound or nonlinear Spearman--Brown conversion is claimed.

The decision and subset are frozen before training. Both arms generate fresh
GRPO responses during continuation. They start with the same source adapter
and optimizer. The random arm never scans the cache; if the screen rejects,
both arms use the same seeded random subset, but the gated arm pays its cost.
Every completed branch records allocated GPU-seconds, including failed work,
CPU work while GPUs are allocated, training and checkpointing. A hard phase
timeout kills the entire child process group. The trainer stops before the
next update that cannot fit with its save reserve. Unknown cost after an
unclean kill is never reset to zero. Other unlocked tasks continue after a
failed task; there is no retry loop within a launcher invocation.

## Commands

Use the existing `offpolicy-misranking` checkout, not another clone.

```bash
git pull
bash scripts/run_light_gate.sh cpu
bash scripts/run_light_gate.sh plan
bash scripts/run_light_gate.sh
```

Run the last command on each separately allocated four-H100 node. Shared task
leases divide the ten default continuations (MATH d100, seeds 0..4, two arms).
Preparation is idempotent and source artifacts are read-only. Node admission
preserves unrelated running suites. Relaunch cleanup is limited to this
suite on the same physical node; it never kills Qwen/E5 or another node.
Outputs default to `$OM_WORK/runs/selection-gate-light-v2`, separate from
all existing results. Set `LIGHT_GATE_ROOT` only to start a distinct protocol.

```bash
bash scripts/run_light_gate.sh status
bash scripts/run_light_gate.sh live
bash scripts/run_light_gate.sh summarize
bash scripts/run_light_gate.sh stop
```

The default budget is the source median step time times 100 times four GPUs,
rounded upward to a minute. Explicit options are accepted during first
preparation, for example `--budget-gpu-seconds 72000`. Changing a frozen
contract requires a new output root; it does not mutate existing runs.
Independent evaluation uses 300 disjoint questions with eight responses,
and is charged to a separate reporting ledger. This is incremental
continuation accounting: shared source training/cache production and launcher
admission are prerequisites, not silently free end-to-end training.

## Development And Held-Out Testing

The default suite is **development**, not final evidence. After exporting a
complete validated `light_results.json`, freeze the rule without fitting a
tree or changing thresholds:

```bash
bash scripts/run_light_gate.sh freeze
```

This writes `frozen-rule.json` under the suite output root. The summary records
the actual source trajectory IDs. Testing requires a new
output root, `--role test --rule frozen-rule.json`, and different source
trajectories. Another checkpoint of a development seed does not qualify.
If source seeds 0..4 are all development data, additional independent source
runs are required for held-out validation. Do not relabel those five seeds.
Freezing does not certify improvement; negative findings remain negative.

## Evidence And Comparison

- Old published v2: choose reuse estimators by alignment rather than overlap.
  Retained as supplementary diagnostics, not the new main method.
- Previous one-shot v2 prototype: eight features and a learned decision tree,
  three branches per seed. Preserved in `run_selection_gate_gpu.sh`; it is
  not the new default experiment and is not required to run this suite.
- V3: reliability from existing training responses. Adopted as the starting
  idea, but a shared-baseline gradient split is not independent replication,
  a 40-prompt training subset is not the 400-prompt pool, and extra work is
  not free just because no generation was added.
- Current v2: one whole-pool scan, actual difficulty-score statistics,
  activity checks, explicit fallback and two equal-allocation continuations.

CPU exact examples refute a distribution-free square-root reliability bound
and show perfect pass-rate repeatability with zero GRPO signal. They are
not language-model benchmark results. Local GPU execution is unavailable;
the four-H100 path requires cluster testing. Outcome claims must use the
paired independent reward difference, not positive cache statistics.
