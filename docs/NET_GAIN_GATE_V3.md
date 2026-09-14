# V3 Cost-Inclusive Continuation Gate

Date: 2026-09-14. Experimental implementation, not a validated performance claim.

## What The Existing Results Establish

The received E5 export `e5-results-20260913T232454Z.txt` contains conditions
where gradient selection improves downstream reward. In particular, the
fresh-gradient control is favorable on the d0 public evaluation, and the
reused-gradient control is favorable on the d100 MATH evaluation. This does
not establish that gradient selection always wins, or that its scoring cost
is recovered at equal total compute.

These are separate continuations of 100 updates from specified parents.
A favorable d100 continuation is evidence about that starting state and
that continuation, not a recommendation to stop selection at update 100.
Comparisons at d0, d100 and d400 do not identify the best intervening switch
step. No optimal stopping time is reported by this implementation.

The old correlation gate can reject a useful selector. Conversely, retaining
a reproducible selector does not establish a reward benefit. The new target
is therefore downstream reward after accounting for the diagnostic and
selection work, not correlation, set agreement, or alignment alone.

## Isolation And Scope

- New code: `src/net_gain_gate.py`, `src/net_gain_gate_gpu.py`.
- Launcher: `bash scripts/run_net_gain_gate.sh` in the existing checkout.
- Output: `$OM_WORK/runs/net-gain-gate-v3`, or `NET_GATE_ROOT`.
- Existing E5, Qwen, GRPO, light-gate and v2 launchers are unchanged.
- The discarded `run_fixed_gate.sh` experiment is not imported or revived.
- No mixed-pool experiment is launched.
- GPU support is positive-drift OLMo MATH, four GPUs per admitted node,
  K=G=8, top 10% fixed prompt subset, one GRPO epoch per sampled batch.
- Supported selectors: `low_order` (default), `pair_u2`, and `difficulty`.
  The first two use the existing costed gradient-direction scoring backend.
  Difficulty uses `-abs(cached_pass_rate - 0.5)` with seeded tie-breaking.
- **These are not the E5 `g11` or `fresh_r` selectors.** Their E5 wins cannot
  be reused as labels for a different selector. There is no new d0, Qwen,
  MBPP, g11 or fresh_r GPU implementation in this change.

## Decision And Horizon

At a frozen parent checkpoint, scan the existing whole candidate-pool reward
cache once. Summarize its success distribution and append the last 20
already-recorded GRPO updates' reward, active-group fraction and allocated
time per update. Also record the decision step and cache age. The cache is
not described as a measurement of current-policy usefulness. Current training
statistics must end at the parent step; future or duplicate rows are rejected.

A depth-at-most-two regression tree predicts the following fixed-budget
contrast, using development trajectories only:

```
N(x, B) = R_selection(B - c) - R_random(B)
```

Here `c` is the one-shot diagnostic allocation. The selection branch's
scoring, fresh-response GRPO and checkpoint writing all fit inside `B-c`.
Select only if predicted `N` exceeds the frozen nonnegative reward margin.
Outside the development feature ranges, choose random. A failed diagnostic
also chooses random, without repeating the diagnostic or refunding its cost.

The decision and chosen prompt IDs remain fixed for **one nominated compute
budget B**. There is no gate in the epoch loop, no repeated bootstrap, no
new diagnostic rollout, and no diagnostic gradient computation. Training
still generates new responses on the fixed selected prompt subset.

The controller does not estimate an optimal switching time. A model fitted
only at d100 will not silently extrapolate to d400. Joint checkpoint studies
can compare decisions at both states, but are still separate matched-parent
continuations, not a validated sequential switching policy.

## What Is Compared

Development runs collect three branches from the same parent and optimizer:

| Branch | Diagnostic | Subsequent allocation | Fixed subset |
| --- | --- | --- | --- |
| `random_full` | none | B | seeded random |
| `random_reduced` | c | B-c | same seeded random |
| `selection_reduced` | c | B-c, including scoring | selected |

The diagnostic is executed once and charged to each counterfactual that
would need it. It is not physically executed twice. Selected indices from
the diagnostic cache scan can be reused for difficulty scoring; gradient
scoring remains an additional charged phase.

For a paid diagnostic followed by action indicator A, the accounting identity
is

```
Delta = R_selection(B-c) - R_random(B-c)
L     = R_random(B) - R_random(B-c)
G     = A*Delta - L = A*N - (1-A)*L
```

This is bookkeeping, not a proof of positive gain. In particular, predicting
negative N and choosing random still pays c. The N threshold tests whether
selection can beat the no-diagnostic baseline; it is not the reward-optimal
threshold after c has already been sunk. The paid-action comparison would
instead use Delta. We deliberately report both contrasts and measure actual
gate reward rather than claiming the threshold is optimal.

The **held-out test** executes three real continuations:
`random_full`, `selection_full`, and `gated`. Always-selected training pays
its own scoring but no gate diagnosis. The gate pays diagnosis even on a
random fallback. Scoring directories are private to each branch: one
branch's paid gradient scores cannot become another branch's free cache.
Independent evaluation questions and response counts match within a point.

## Costs And Statistical Boundaries

- An allocated four-GPU node is charged during CPU diagnosis and waiting
  inside a metered phase, not only during CUDA kernels. Worker startup,
  scoring, training, checkpointing and failed attempts are recorded.
- Diagnosis is capped at the smaller of 30 wall seconds and 1% of B by
  default. Process termination overhead is still charged, not clipped away.
- Training uses the existing budget-limited GRPO driver. An update-count
  stop or measured overrun is not accepted as a completed equal-budget result.
- Shared source construction, initial source hashing, model fitting and
  independent reward evaluation are outside deployment continuation caps.
  Report them separately; missing historical cache cost is not zero.
- The reported cost is the sum of metered allocation phases, not a profiler
  estimate of GPU utilization or an end-to-end cluster invoice. Launcher,
  admission and offline preparation overhead must not be silently included
  in a claim of measured end-to-end speedup.
- A started cost interval without a finish is unknown, not free. Such an
  interrupted task is invalid until its external allocation evidence is
  reconciled. The worker records the issue and tries other available tasks.
- Fit uses at least three independent development trajectories, depth <=2,
  and at least two distinct trajectories per leaf. Repeated checkpoints are
  weighted within trajectories, not treated as new independent seeds.
- Neither development trajectories nor exact parent/optimizer pairs may be
  renamed as held-out data. Synthetic fitted models cannot control GPU runs.
- Source artifacts and the experiment's relevant code hashes are frozen.
  Keep the original code while a suite runs; changed methods require a new
  suite rather than silently altering partially completed comparisons.
- Out-of-range fallback is an explicit design choice, not a confidence bound.
  No bootstrap count, confidence guarantee or benchmark gain is invented.

## Commands

Use the same repository on each node. No additional clone is needed.

```bash
git pull
bash scripts/run_net_gain_gate.sh cpu
bash scripts/run_net_gain_gate.sh plan
```

CPU fitting/tests use `python3` by default. `NET_GATE_CPU_PYTHON` can select
an existing environment with `requirements-gate.txt`. GPU work uses the
existing `.venv-cu126`; it does not need scikit-learn to apply a frozen tree.

Small development run, d100, seeds 0/1/2, nine continuations total:

```bash
bash scripts/run_net_gain_gate.sh
```

Run that same command on up to four admitted four-GPU nodes. File leases
assign different tasks; they do not launch one duplicate suite per node.
An active unrelated E5/Qwen allocation is not killed to make room.

To cover both d100 and d400 from the outset, use a separate root and the
following command on each node (18 continuations):

```bash
NET_GATE_ROOT="$OM_WORK/runs/net-gain-gate-v3-two-checkpoints" \
  bash scripts/run_net_gain_gate.sh run --drifts 100 400
```

The common compute cap is frozen across both checkpoints. Without an explicit
`--budget-gpu-seconds`, preparation uses the median source update duration
times four GPUs times 100 equivalent updates, rounded up to 60 GPU-seconds.
This specifies a compute horizon, not exactly 100 subsequent updates. A rough
training allocation for P points on N four-GPU nodes is `3*P*B/(4*N)` wall
seconds; evaluation, scoring imbalance and idle tail time must be added.

Status, summary and a shell-based home-directory export:

```bash
bash scripts/run_net_gain_gate.sh status
bash scripts/run_net_gain_gate.sh summarize
bash scripts/run_net_gain_gate.sh export
```

For a nondefault root, retain the same `NET_GATE_ROOT` prefix on these commands.
The export contains status, study/results, diagnostic decisions, costs and
log tails. It does not copy model checkpoints, rollouts or another repository.

After all declared development points are valid, freeze the model on CPU:

```bash
bash scripts/run_net_gain_gate.sh fit
```

This requires real matched-budget results and refuses an existing model
output. It does not manufacture a model from the old equal-update E5 export.

Actual held-out test, using a new root and the default development model:

```bash
NET_GATE_ROOT="$OM_WORK/runs/net-gain-gate-v3-test" \
  bash scripts/run_net_gain_gate.sh run --mode test \
  --model "$OM_WORK/runs/net-gain-gate-v3/model.json" --seeds 3 4
```

Seeds 3/4 are usable only if they were not used to develop the fitted rule.
The code verifies frozen model provenance, not the researcher's undocumented
prior inspection. Do not call an adaptively chosen split preregistered.
The selector, budget and measurement settings must match the fitted model.
For a two-checkpoint model/test, also specify `--drifts 100 400` and its actual
model path. Unknown source points are reported, not recreated implicitly.

Completed old three-branch gate results may be read with `import-legacy
--source-root ROOT --out NEW.json`. This preserves measured old branch costs
and adds current-state features from their original checkpoint logs. It is
marked `legacy_replay_development_only`; its replay CPU cost is research cost.
It cannot certify the new deployed diagnostic or be relabeled as held-out
evidence. Discarded fixed-checkpoint gate and equal-update E5 results are not
converted. There is no reason to restart running E5/Qwen jobs for this code.

## Verification Record

Local CPU tests exercise feature time boundaries, binary cache coverage,
net-gain labels, paid fallback, independent-trajectory splits, frozen model
and artifact bindings, private scoring costs, update-count rejection,
single-attempt diagnosis, process interruption and four-worker task leasing.
The shell export is tested for read-only collection and overwrite refusal.

Final local verification: 160 tests passed across the new tests and existing
selection-gate, light-gate and budget-driver regression tests. `bash -n` and
Ruff's undefined/unused-name checks passed. This is CPU/fake-worker coverage,
not an H100 training result.

The local machine has not executed H100 scoring, GRPO continuation or
independent benchmark generation for this suite. No fitted observed v3 model,
GPU completion claim or optimal switch step is included in this commit.
