# Fixed-Checkpoint Reuse Gate

This is the September 13 implementation amendment. It does not change the
historical `gate_passrate` experiment into a different experiment. The old
training-pilot outputs, existing E5/Qwen/OLMo jobs and their checkpoints are
preserved. No model is updated during the new diagnostic.

## Run

From the existing checkout on each idle node with four allocated GPUs:

```sh
git pull
bash scripts/run_fixed_gate.sh
```

Use the same command on up to four nodes. The default is OLMo MATH d400 and
d0, seeds 0, 1, 2: six independently leased points. A busy point is skipped
while the controller looks for another one. Source paths, E5 inputs and
hyperparameters come from the existing workspace and frozen E5 contracts.
No clone, new repository or manual source-checkpoint selection is required.

```sh
bash scripts/run_fixed_gate.sh status
bash scripts/run_fixed_gate.sh live
bash scripts/run_fixed_gate.sh export
```

`status` shows all six points, outcome availability, cost availability and
stale heartbeats. `live` follows all node launcher logs, including files
created after the command starts. `export` writes a dated text bundle under
`$OM_WORK/exports/` and prints its path. Upload that file through transfer.
It includes the rule, source contracts, decision, component cost events and
paired reward results, plus progress and the last eighty lines of each worker
log. The CSV/JSON summaries remain under the suite root.

Optional branch restriction: append `d0` or `d400` to any run/plan command.
`E5_SEEDS` retains its existing meaning. `FIXED_GATE_ROOT` and
`FIXED_GATE_RULE` are explicit overrides, not required arguments.

```sh
bash scripts/run_fixed_gate.sh plan
bash scripts/run_fixed_gate.sh cpu
bash scripts/run_fixed_gate.sh stop
```

The CPU mode uses the existing local environment and temporary test data; it
does not launch cluster training. `stop` targets this suite on the current
node, not unrelated jobs. The run launcher replaces only its own previous
local controller, then uses the existing physical-node admission lock.

## Exact Experiment

The fixed selector is the existing **g11**, not a search over selectors.

1. Freeze the source policy and optimizer identity, candidate pool, the
   existing ranking-validation direction R, original eight-response behavior
   cache, scoring configuration, pilot indices and rule.
2. Uniformly choose forty distinct candidate prompts before reading outcomes.
   On those prompts only, generate an independent group of eight responses
   using the **behavior/base policy**. These are not current-policy training
   rollouts. New response generation and verification count as diagnostic work.
3. At the fixed current-policy checkpoint, compute g11 from the original
   eight responses and independently from the new eight responses. Both use
   the same R validation direction, leave-one-out advantages and clipping.
   Neither four-response nonlinear halves nor A/B's smaller validation target
   are substituted for the deployed selector's score.
4. Assess one forty-pair correlation interval. Retain g11 only if it meets
   the frozen threshold and estimated remaining scoring fits the finite
   budget. Missing costs, invalid measurements, failed measurement, expired
   deadlines and unresolved intervals cause random fallback.
5. If retained, score only the remaining candidates, reuse the primary pilot
   scores and apply the unchanged g11 top-k rule. Actual remaining scoring
   also has a hard allocation limit. A timeout cancels those workers and
   switches to random before any continuation begins.
6. Validate that the selected IDs exactly match the appropriate frozen E5
   subset. Reuse its completed training/evaluation outcome when available.
   If random or g11 comparison outcomes are missing, call the existing E5
   driver to finish those arms from their original source checkpoint, with
   100 updates and eight test responses. Existing arm locks/checkpoints are
   honored. No ten-update pilot policy is used as the training parent.
7. After the decision is frozen, independently time an ungated g11 scoring
   pass over the full candidate pool. This is research-only cost measurement;
   its scores and runtime cannot change the gate decision.

The two measurements have the same response budget and scoring target.
They are conditionally independent response draws given the policy and fixed
R direction. The Gaussian gain interpretation and Fisher interval remain
model-dependent approximations; the code does not certify future reward.

## Limits And Costs

Defaults in `config/fixed_gate_rule.json` are operational limits for this
exploratory amendment, not experimentally optimized thresholds:

| Quantity | Default |
|---|---|
| Pilot | 40 distinct prompts, 8 original + 8 replica responses each |
| Retention threshold | lower endpoint >= 0.25 |
| Interval | two-sided 90% Fisher-z approximation |
| Pilot + assessment deadline | 3,600 wall seconds on four allocated GPUs |
| Remaining scoring budget | 28,800 GPU-seconds, at most 7,200 wall seconds |
| Remaining-cost estimate | 1.5 times fixed model loads plus measured primary per-prompt work |
| Research-only ungated timing cap | 7,200 wall seconds |
| Missing E5 comparison completion cap | 43,200 wall seconds |

These are **caps, not runtime forecasts**. A failed/timeout measurement is
not automatically retried until it produces a favorable statistic. The
decision is frozen, and subsequent runs reuse it. Finished E5 checkpoints
and evaluation shards are not retrained. Interrupted or partial cost events
remain explicitly unknown; they are never counted as zero.

Measured phases charge all four allocated GPUs, including CPU assessment
while the GPUs wait, subprocess startup/model loading and cancellation
overrun. Source-contract preparation is separately charged to deployment.
Research-only baseline scoring and counterfactual training/evaluation use
separate ledgers. Post-hoc result export is not selector deployment work.
Historical cache generation and R-validation construction are shared inputs;
their absence from the incremental ledger does not mean they were free.
This is not an end-to-end from-scratch cost comparison.

`measured_net_scoring_gpu_seconds_saved` is the separately timed ungated
g11 scoring allocation minus measured gated preparation/diagnostic/remaining
scoring allocation. It may be negative. It is null if the paired baseline
or allocation accounting is incomplete. The research baseline always runs
after the frozen decision, so timing can still reflect order/cache effects;
report hardware and this order rather than claiming a randomized timing trial.

`forgone_reward` is the validated g11 fixed-arm reward minus the reward of
the chosen fixed arm. A retain decision has zero forgone reward by this
counterfactual mapping; random fallback uses the existing paired g11-minus-
random interval. This is valid only under the matching source, optimizer,
subset, learner and evaluation contracts. It is **not** a newly trained gate
policy, a test on an independent training seed, an equal-total-compute reward
comparison, or a preregistered test of the old September 12 rule.

## Outputs And Figure Inputs

New files go only under `$OM_WORK/runs/fixed-checkpoint-gate-v1/d<drift>/s<seed>/`:

- `contract.json`: rule, source identities and pilot indices.
- `pilot/`: newly generated behavior-policy replica responses and paired scores.
- `assessment.json`, `diagnostic.json`: the single statistical assessment.
- `remaining/`: conditional primary scores, absent after early fallback.
- `decision.json`: checksummed action and exact subset binding.
- `baseline/`: separate full-pool scoring for measured cost comparison.
- `cost.jsonl`, `progress.json`: phase allocation events and progress.
- `result.json`: contract-checked outcome reuse and cost comparison status.

The suite's `results.csv` provides three figure panels without parsing logs:

1. Decision panel: one row per checkpoint/seed, correlation with its interval,
   fixed threshold, action and fallback reason. Do not count descriptive
   moving-policy windows as additional trials.
2. Cost panel: diagnostic and remaining-scoring components against measured
   ungated scoring; show signed net saving. Missing cost is a missing point,
   not a zero-height bar. Historical common costs and research-only work are
   disclosed separately.
3. Outcome panel: forgone reward with paired intervals, grouped by actual
   action. A zero on a retained branch is definitional outcome reuse, not
   evidence that the measurement predicts benefit on a new seed.

## Repairs To Existing Utilities

- Historical training-pilot logs now require every expected step/rank/prompt;
  identical replays are deduplicated and conflicting replays rejected. This
  repairs aggregation, not the old experiment's fixed-policy design mismatch.
- A budgeted decision cannot retain a signal with unknown remaining cost.
- Adding public benchmark sets uses a sidecar extension; the original
  `benchmarks.json` hash and finished shard contracts stay unchanged. Local
  dataset hashes/row counts are verified and new downloads pin revisions.
- Conflicting stale-half shard parameters are rejected before merge.
- The Gaussian top-k Monte Carlo constant is actually cached.
- CPU export wrappers propagate analysis failures instead of ending with a
  successful message. Partial output is preserved for diagnosis.

Legacy unbound stale-half artifacts and mtime-based cost-accounting tables
are not upgraded into trustworthy evidence by these changes. The new gate
does not consume the old stale-half scores or old per-prompt cost suggestions.

## Local Verification

On September 13, `bash scripts/run_fixed_gate.sh cpu` passed **104 tests**
in 36.48 seconds with CUDA disabled. Coverage includes real g11 gradients on
a tiny CPU model, fixed source micro-batch settings, subprocess execution and
deadline cancellation, per-point leases, immutable decisions, E5 outcome
reuse, missing allocation records, status after partial results, benchmark
extensions, and export-shell failure propagation. Subprocess fixtures test
control flow; their timings are not GPU experiment results.

The repository's five bundled benchmark manifests also passed hash/row-count
validation: AIME24 30, AIME25 30, AMC23 40, GSM8K 1,319 and MATH-rest 4,500.
The CPU `status` command runs without CUDA or local cluster artifacts.
Actual H100 model loading, cluster filesystem locking and end-to-end GPU
training have not been exercised on this machine. Existing remote jobs were
not stopped or restarted.
