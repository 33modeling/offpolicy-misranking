# Pre-selection gate: experiment and implementation design

Date: 2026-09-12 KST. Status: design, not an implemented controller.

## Decision and work record

The user clarified the proposed contribution: compare selection and random
TRAINING during research, identify when paying for selection stops helping,
and put an inexpensive predictor BEFORE expensive selection in actual use.
Do not train both alternatives at every deployed decision. The user also
authorized replacing old manuscript arguments and proofs where necessary.

The primary outcome is independent benchmark reward at matched TOTAL compute,
not rank overlap, agreement, or alignment alone. Low-order reuse scoring is
one candidate selector, not the required central contribution. No existing
OLMo, Qwen, E5, or method-choice contract is changed by this document.

Completed before this design:

- `7c36283`: isolated low-order comparison implementation, CPU regression
  tests, and prior-art review. The recorded focused suite passed 137 tests;
  no full GPU run was performed locally.
- `e6542f8`: initial random-fallback direction. Its paired-pilot operating
  sketch is superseded: paired training is OFFLINE research supervision.
- This design: a cost-aware mathematical objective, proof draft, causal switch
  experiment, measurement ablation, figure plan, and concrete integration
  requirements. No new gate runner or automatic GPU job is introduced.

The canonical [theory and proofs](https://github.com/33modeling/offpolicy-misranking-paper-v2/blob/main/research/2026-09-12/SELECTION_GATE_THEORY.md)
live in the existing v2 manuscript repository. This file is the operational
specification, not a duplicate manuscript tree. The published PDF/reader
remain the previous draft until a complete synchronized manuscript revision.

## 1. What the controller does

At a declared training-block boundary, before new full-pool scoring:

1. If the frozen controller has not been validated for this model, dataset,
   selector, verifier, and budget scope, use no-measurement random sampling.
2. Read bounded pre-decision features. If the recipe allows a probe, check its
   hard cost/time cap BEFORE dispatch. Never add a probe simply to chase an
   uncertain decision. Feature extraction and I/O are charged.
3. Apply the frozen predictor/threshold. If it predicts sufficient selection
   benefit under its declared validation protocol, score/select/train for
   one block. Otherwise stop using the selector and continue random training
   from the CURRENT checkpoint for the remaining budget.
4. In the first version, random fallback is absorbing within this run
   contract. There is no select-random-select loop or hidden repeated pilot.
5. Stop training only at its total budget or an actual common-input/training
   failure. A selector-specific failure may fall back to valid random work;
   a corrupt model, dataset, or verifier is not cured by changing the sampler.

Output includes action, reason, predictor/version, source hashes, features and
their timestamps, consumed/remaining budget, next eligible boundary (if any),
and whether selection has permanently stopped. A statistical certificate is
reported only when its assumptions and sample size actually justify it.

Suggested reason codes: `select`, `predicted_no_useful_gain`,
`unsupported_scope`, `measurement_budget_exhausted`, `probe_timeout`,
`invalid_selector`, `random_absorbing`, and `common_input_failure`.
An uncertain prediction or missing measurement is not an observed negative
benchmark reward.

## 2. The targets and the random controls

For state x and remaining budget B, compare independent terminal rewards.
Selection includes its own scoring cost. If a gate costs c, the selected
training branch only gets B-c. Random without a gate gets B. The theory writes

`gate gain = E[gate * selection advantage at B-c] - E[random budget effect]`.

The latter is the reward difference between random at B and random at B-c,
NOT seconds subtracted from accuracy. Even rejection has already paid c.

Primary baseline: `random_k`, a uniform subset of the same size k from the
same eligible pool, refreshed on the same declared block schedule. Preserve
the learner's update objective, generation protocol, and optimizer settings.
Additional baseline: `uniform_pool`, using the full eligible prompt pool
without selector scoring. State the actual sampling/ordering implementation;
the current trainer uses a deterministic seed/step-indexed traversal, not
independent uniform sampling at every update.

Current E5 random is an existing fixed-subset arm. It must retain that label
and cannot silently become the new block-refreshed baseline. Both arms
generate current-policy responses during training. The old cached answers
are inputs to some scoring methods, not mandatory training targets.

## 3. Research labels: same-state interventions

Independent always-random and always-selector learning curves are useful
baselines. Their crossing is NOT a causal switch label: their model and
optimizer states have diverged.

At a nominated checkpoint on a development trajectory, preserve one complete
parent state. For a fixed remaining budget B, compare:

| Branch | Action from that same parent | Purpose |
| --- | --- | --- |
| R-full | Random for remaining B, no gate | Actual alternative to measuring/selecting. |
| R-reduced | Random for B-c after the declared measurement | Isolate the signed opportunity cost of measuring. |
| S-reduced | Declared selection continuation for B-c | Single-decision conditional-benefit target. |
| S-block-then-R | Select one block, then random until the same B is spent | Label whether another selection block is worthwhile before stopping. |

Branch costs include loads, scoring, refresh, generation, optimizer steps,
checkpoint I/O, and any probe. All arms may finish different numbers of
updates because the comparison is compute-matched. Keep equal-update curves
as a secondary mechanism analysis, not an efficiency result.

Reserve budget before dispatching a block and retain the last valid completed
state. Charge any partial or over-budget work. A run that exceeds its nominal
cap is not evidence for an exact at-most-B guarantee: report its actual cost,
and use valid common budget endpoints for comparison. Do not silently crop
the ledger while keeping the extra training's reward.

Keep two distinct labels. The predictor's conditional action-advantage target
is `S-block-then-R` versus `R-reduced`, both AFTER paying the same check cost.
The net intervention target is `S-block-then-R` versus `R-full`, charging that
check against no-measurement random. The random budget contrast connects the
two. Do not conflate them, train on whole-horizon always-selector labels, or
claim those labels estimate the value of one more block. Measure complete
gated trajectories later; a locally useful block can change future decisions.

Use existing runs only where exact parent, optimizer, evaluation, and cost
lineage match. Reusing earlier points on an R curve to read B-c is valid only
when durable checkpoints and actual cost boundaries exist. Do not interpolate
a missing checkpoint's benchmark score or subtract time from a fixed final
score. Discovery over many post-hoc times is exploratory.

## 4. Features and a deliberately small predictor

First candidate: a shallow tree (maximum depth two) on pre-decision features,
fit only on development trajectories. Leaf values estimate the conditional
action advantage Delta: one selection block then random versus random at the
same POST-measurement budget. A frozen threshold determines the action. The
whole-controller comparison must separately cover measurement on rejected
cases. Compare the tree with a time-only threshold and a constant decision.
Use an established tree implementation if this candidate reaches coding.
The small tree is a concrete starting hypothesis, not a proved novel method.

Candidate features, subject to availability and timing audit:

- Remaining budget and checkpoint/update age.
- Cached reward success fractions, mixed-reward group fraction, and their
  coverage/missingness over the eligible pool.
- Already-produced learner statistics: active-advantage group rate, reward
  trend, and gradient norm, with the sampling source recorded.
- Existing cache age and valid previously computed drift/support summaries.

Do not call these free: parsing large rollout files repeatedly can be costly.
Maintain bounded summaries with provenance when the underlying data is
created. Nonrandomly sampled training histories can be biased and require
held-out validation, not an IID interpretation of all logged prompts.

KL/ESS requiring new current-policy likelihoods are a paid probe, NOT a cached
feature just because the old responses are on disk. Candidate gradients,
full-pool alignment scores, and fresh reference selection cannot appear as
pre-gate features if they require doing the expensive selection first.

Feature ablation: history-only; bounded forward-only probe; full scoring as
an expensive diagnostic comparator. Paid-probe sizes and check schedules are
chosen from a small predeclared menu after timing a pilot, then frozen before
calibration. Record actual costs and cap overshoot; no unmeasured speed claim.

Do not infer benefit from low overlap or low ESS by definition. The feature
must predict the independent downstream continuation contrast. Its transfer
to other seeds, stages, and model families is an empirical question.

## 5. Splits, schedules, and what can be claimed

Separate development trajectories, whole-controller calibration trajectories,
and final reporting trajectories. Keep all checkpoints of one parent run in
one split. Also keep diagnostic prompts separate from final benchmark reporting
and the selector's own fitting/validation data where independent roles are
required. Do not tune thresholds on final test reward.

Five existing seeds cannot automatically fill all these roles with useful
statistical power. The proof draft shows the small-sample limitation explicitly.
Start with exploratory intervention/timing pilots, not a claim of certified
superiority. Existing 40 score-measurement cells are not 40 independent gated
training experiments.

At first, check only at fixed block boundaries, with a finite maximum number
of checks and an absorbing random fallback. Choose the block size using
observed step/scoring durations, not a guessed completion-time promise.
Existing `checkpoint_every=5` is a persistence setting, not a justified
measurement frequency. A time-only switch learned on development data is a
mandatory baseline: otherwise a complex detector may merely rediscover time.

Report always random, always selector, a fixed-switch policy, the proposed
gate, and full-pool random. Compare feature budgets and checking frequency.
Include a delayed-start baseline if selection is unhelpful early but useful
later; absorbing fallback deliberately cannot recover that later gain.
An oracle chosen from TEST outcomes is only a clearly labeled retrospective
upper bound, never a deployable method or a fair fitted baseline.

Use paired per-seed results and observed dispersion. Do not add a large nested
bootstrap loop by default. Any interval must name its sample unit and inferential
target. Prompt-level resampling does not represent training-seed uncertainty.
The analytic finite-family bound is a conditional theoretical benchmark, not
a reason to report a useful confidence certificate from insufficient seeds.

## 6. Cost experiment and economic decision

Every phase writes start/end UTC and monotonic elapsed time, allocated device
count/type, accelerator-seconds, CPU seconds when available, input/output
tokens, status, parent/action IDs, cache hit/miss, and artifact hashes. Allocated
accelerator-seconds include time that assigned GPUs wait; utilization is a
separate diagnostic, not a reason to remove charged time.

Measure both warm/cached deployment and cold start. Failed/aborted probes,
retries, unused results, model loading, and idle gaps belong in the cost
ledger. Shared setup costs need a declared allocation rule. Externally queued
time is reported separately from allocated resource time.

Separate three ledgers:

1. Research: counterfactual branch training, gate fitting, and calibration.
2. Deployment: actual gate checks, scoring, training, and selected trajectory.
3. Reporting: final benchmark evaluation never used for decisions.

Primary operational plots include every deployment cost; any online reward
evaluation is deployment cost, not reporting. Also disclose the research and
reporting totals. The cold-start/full-project analysis charges the research
cost; amortization over many runs is an explicit scenario, not a hidden saving.

For a PREDECLARED quality target q, let C_R(q) and C_G(q) be measured random
and gate costs to reach q, and C_fit the gate-development/calibration cost.
If C_R(q)>C_G(q), the elementary amortization condition is

`N * (C_R(q) - C_G(q)) > C_fit`.

This is not an estimate until those costs are observed; unreached targets are
censored, not assigned infinite savings. Report reward at fixed budget even
when no arm reaches q. Avoid an accuracy/GPU-hour ratio as the sole outcome.

## 7. Integration in THIS repository

Proposed modules, not files already supplied:

| Component | Responsibility |
| --- | --- |
| `src/selection_gate.py` | Pure decision/state transition logic; absorbing fallback, scope validation, caps. |
| `src/selection_gate_study.py` | Same-parent intervention preparation, development splits, controller evaluation and aggregation. |
| `scripts/run_selection_gate.sh` | Existing shell style: plan/check/status/live; run only after an immutable contract is prepared. |

Reuse `evidence_downstream` and the GRPO trainer, existing leases, atomic JSON,
generation validation, and phase cost tracking. Do not copy a new trainer,
split v1/v2 code again, or put this runner in another repository.

Concrete code observations and required changes BEFORE execution:

- `low_order_experiment.py` currently prepares random alongside scored subsets
  and waits for scoring completion. The gate path must prepare valid random
  work independently. It must still validate shared model/data/verifier inputs.
- `method_choice.py` chooses from score-derived estimates; it is not a
  pre-scoring stopping gate. Do not relabel it as the new method.
- `train_policy_grpo._save_checkpoint` currently keeps only the LAST TWO
  `checkpoint-*` directories. Earlier nominal five-step checkpoints may no
  longer exist. Save nominated branch parents durably for the NEW study before
  retention removes them; do not assume all old switch points can be restored.
- Intermediate checkpoints contain `checkpoint_state.json`; external parent
  loading validates `policy_train.json`. A raw checkpoint path is not already
  a valid published branch parent. Add a validated export/branch contract,
  not fabricated final manifests or disabled lineage checks.
- Resume validation binds optimizer/configuration. A subset switch needs a
  new segment contract linking the old parent and new prompt-subset hashes.
  Preserve optimizer state and learner configuration. Do not modify a running
  experiment's input files to force its resume path to accept new data.
- Per-step training seeds are derived from seed/step/rank. Record that mapping
  and branch stream choices; common random numbers aid pairing but do not make
  distinct policies produce identical responses or prove bitwise replay.
- Scheduler work is leased per valid task, not a new blanket physical-node
  lock. Failed selector/probe work must not prevent other valid work from
  using the node. A common-input failure remains visible rather than spinning.

Immutable artifacts should bind parent state, sampler/pool, selector version,
feature availability cutoff, gate model, split role, budget, branch target,
decision history, and cost/evaluation lineage. CPU aggregate reports must
reject missing parents, duplicate IDs, future features, and mixed budget units.

## 8. Tests and execution order

Before any cluster launch:

1. CPU contract tests: no-score random eligibility, absorbing state, threshold
   ties, missing/NaN features, unsupported scope, caps and invalid shared input.
2. Tiny deterministic mathematics checks: cost identity, weighted decision
   regret, prediction-error inequality, and sequential telescoping. These
   check examples; the algebraic proofs are in the paper repository.
3. Mock subprocess/clock tests: stalled probe terminates its own process
   group; a timeout produces one bounded decision, not an infinite relaunch.
4. Resume/branch tests: parent retention/export, optimizer preservation,
   changed-subset segment manifests, interrupted atomic writes, concurrent
   workers, and next-valid-task scheduling.
5. Cost tests: failed work, warm/cold setup, three-versus-four assigned GPUs,
   no double counting, budget reservation and end-of-step overshoot.
6. Leakage tests: checkpoint siblings never cross splits; post-decision
   features and final test labels are rejected by the gate fitter.
7. On the remote GPU cluster only, a single short paired intervention/timing
   pilot validates manifests, costs, memory, and actual output before expansion.

No new run command is claimed usable yet. `run_low_order.sh` still runs the
old fixed-arm experiment. There is no basis yet for a reliable hours/node-count
estimate for the new gate study; derive it from the timed pilot and the final
number of independent branch parents, not from the number of score cells.

## 9. Results that would support or reject this direction

Support requires a nontrivial gate that recognizes both helpful and unhelpful
selection on held-out trajectories, improves reward at matched total cost or
reduces cost at matched reward, and improves over a fixed-time switch. Report
all negative seeds and the research-cost amortization threshold.

If no selector helps, always random is the practical recommendation, but that
does not validate a useful detector. If the signal fails to transfer, a global
cost/timing rule may be all the data support. If measurement consumes the
saving, remove it from deployment; do not claim benefit after excluding its
cost. The old theory and score matrix cannot decide these outcomes in advance.
