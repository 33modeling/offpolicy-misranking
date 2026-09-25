# Figure 3 Prospective Cost Measurement

This is a new experiment, not a rewrite of the old cost export. It measures
Random, cached SR, Switch, and On-policy from the same saved step-25 model
**and optimizer**, to the same final step. Defaults: seeds 3 and 4, final step
275, selection and checks every 25 steps, one timing replicate per seed.
On-policy refreshes its gradient-ranked subset throughout training. Switch
does so only until its single-reference signal triggers the SR transition.
This repeated-selection comparison is different from the historical experiment
that ranked once and retained fixed subsets. Its costs and rewards are new.
All outputs go to a new sibling root. Existing Pair jobs and files are read-only.

## Commands

Run seeds on two separate nodes, each with four matching GPUs. Within each
seed, the four methods run sequentially on that node's same four GPUs:

```bash
# Node 1: seed 3
bash scripts/run_selector_pair_cost_measure.sh run --seed 3

# Node 2: seed 4
bash scripts/run_selector_pair_cost_measure.sh run --seed 4

# Inspect/export independently (replace 3 with 4 for the second node).
bash scripts/run_selector_pair_cost_measure.sh status --seed 3
bash scripts/run_selector_pair_cost_measure.sh results --seed 3
```

This automatically separates roots and run locks into
`runs/selector-pair-cost-measure-s3-v1` and `runs/selector-pair-cost-measure-s4-v1`.
Home exports are also separate: `selector-pair-cost-measure-s3-results.txt`
and `selector-pair-cost-measure-s4-results.txt`. Use the same `--seed` for
plan, run, status, results, and resumption. `--seed=3` also works. The earlier
`PAIR_COST_MEASURE_SEED` environment setting remains supported, but do not
combine it with `--seed`. Explicit `PAIR_COST_MEASURE_ROOT` overrides the root, so use
distinct overrides when running on two nodes. Existing unseeded runs are
unchanged and must not run concurrently with new copies of the same experiment.

The original single-node, both-seed mode is still available:

```bash
bash scripts/run_selector_pair_cost_measure.sh plan
bash scripts/run_selector_pair_cost_measure.sh run
bash scripts/run_selector_pair_cost_measure.sh status
bash scripts/run_selector_pair_cost_measure.sh results
```

The default command is `plan`, not GPU execution. `run` trains and evaluates
four new continuations per seed (eight in unseeded mode); it is not a quick CPU export. The per-arm safety cap
is 160 allocated GPU-hours, including evaluation and failed attempts. It is a
cap, not a runtime prediction. All arms execute sequentially on the same
four GPUs; their order rotates across seeds and replicas. The root is leased,
so two nodes cannot write it at once. Busy GPUs are not cleared or preempted.

```bash
# Measure only seed 3; use the same options when resuming.
bash scripts/run_selector_pair_cost_measure.sh run --seed 3

# Three timing replicas (new output root; substantial additional GPU work).
PAIR_COST_MEASURE_ROOT="$OM_WORK/runs/selector-pair-cost-measure-r3-v1" \
  bash scripts/run_selector_pair_cost_measure.sh run --replicates 3

# Inspect an every-update selection protocol before committing GPU time.
PAIR_COST_MEASURE_ROOT="$OM_WORK/runs/selector-pair-cost-measure-every-update-v1" \
  bash scripts/run_selector_pair_cost_measure.sh plan --selection-interval 1
```

Overrides: `OM_WORK`, `PAIR_ROOT` (existing Pair inputs),
`PAIR_COST_MEASURE_ROOT` (new output), `PAIR_PYTHON`, `CUDA_VISIBLE_DEVICES`.
`--end-step`, `--interval` (online checks), `--selection-interval` (reranking),
and `--max-gpu-hours` are frozen in the new plan. Both intervals default to 25;
the selection interval is identical for On-policy and pre-transition Switch.
Every-update reranking is much more expensive; it may exhaust the safety cap.
Changing a bound setting requires a new output root; no old ledger is reset.

## What Is Measured

- Random: fresh uniform subset selection, then training.
- SR: fresh selection from the existing success-rate cache, then training.
- On-policy: fresh full-pool ranking, including response generation and
  gradient computation, at the start and every `--selection-interval` updates
  until the final training block. Each ranking uses the current model and
  refreshes the selected subset; no ranking is run after the final update.
- Switch: independently paid initial ranking plus cached-subset selection;
  reranking at the same selection interval until the SR transition. Actual
  single-reference checks run on its own evolving policy every `--interval`
  updates. If ranking and checking coincide, rank first, then check the new
  On-policy subset against cached SR. The trigger check and its preceding
  ranking are charged. No ranking or check runs after the transition.
- Every arm trains through the same endpoint. Switching stops only selection
  and checking gradients, not the backward passes required for training.
- Evaluation: fresh common-question endpoint evaluation for every arm,
  metered separately from operating cost. Do not reuse the A/B run's rewards.

One reference uses `validation-a` and `candidate-a` from the existing SR-GC
worker. No B draw is launched, no A/B mean is read, and no A/B duration is
halved. Both candidate subsets use the same current gradients, so overlapping
prompts cancel. Switch after two adjacent negative checks, or a
negative/nonnegative/negative triplet with negative mean. Missing/nonfinite
checks stop execution rather than silently choosing an arm.

Every arm is restarted at the union of selection and check boundaries. This matches restart and
checkpoint schedules instead of adding boundary startup only to Switch. The
existing trainer uses the global step to seed training and continues the saved
optimizer. This is a segmented measurement protocol, not continuous-service
latency; startup/checkpoint cost can be compared with update-only timing.

## Units and Scope

The outer allocation meter measures elapsed seconds multiplied by **four
reserved GPUs once**, including idle reserved GPUs during CPU subset selection,
startup, checkpoint writes, and closed failed attempts. Operating GPU-hours
are selection + online checks + training, divided by 3600. Evaluation is separate.
The report also contains the nested update-timer cost, not added a second time.

Each scoring worker records CUDA-synchronized function durations for model
setup, response generation, and gradient computation. Each worker owns one
GPU, so the four shard durations are summed, **not multiplied by four again**.
The gradient function includes projection and CPU work; it is not isolated
backward-kernel time. These breakdowns cover successful scoring attempts and
are already contained in allocation cost. Failed work remains in the allocation
total even when a component trace could not be completed.

The shared training prefix, original cache acquisition, source/hash preflight,
orchestration gaps, and queue waits are outside this measurement. Original cache
acquisition remains unknown, not zero. No cost or reward ordering is guaranteed.
Random can have longer actual training than SR; online checks can make Switch
more expensive than On-policy. Report the observations even if they contradict
the manuscript's pending schematic layout.

## Outputs and Recovery

- `plan.json`: source and execution-code hashes, settings, and GPU identities.
- `s<seed>-r<replica>-<arm>/phases/`: allocation ledgers, raw function timing,
  gradients, training outputs, and completion receipts.
- `results.json` / `results.csv`: per-seed/replica costs and fresh rewards.
- `~/selector-pair-cost-measure-results.txt`: uploadable text with CSV and JSON.

Reports include ranking/check steps and counts, plus separate gradient-function
times for selection and online checks. These nested timings are not added to
the outer allocation total again. On-policy's selection cost is never copied
to Switch: each arm has its own measured phases, even at the common start.

Successful phases are checked and reused without being charged again. A failed
closed attempt is retained and retried in a new directory, with its time still
charged. An unclosed cost event blocks execution and leaves cost unknown; it is
not silently discarded. A missing optimizer is an error, not a cold restart.
Source artifacts and relevant code must stay unchanged during a measurement.
Resumption uses the same GPU allocation recorded in the plan.

Local CPU tests exercise full orchestration using a stand-in GPU backend,
actual policy-lineage validation, source preservation, idempotent resume,
single-reference switching, disjoint update intervals, and cost accounting.
They do not substitute for running the GPU experiment.
