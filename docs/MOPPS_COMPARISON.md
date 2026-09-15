# MoPPS online-selector comparison

Registered: 2026-09-15. Implementation and CPU verification only; no local
cluster access, GPU smoke result or measured training outcome is claimed.

## Named prior work

Yun Qu, Qi Wang, Yixiu Mao, Vincent Tao Hu, Bjorn Ommer and Xiangyang Ji.
**Can Prompt Difficulty be Online Predicted for Accelerating RL Finetuning of
Reasoning Models?** KDD 2026, Vol. 1, pp. 1240-1250.
[DOI](https://doi.org/10.1145/3770854.3780263),
[paper v5](https://arxiv.org/html/2507.04632v5),
[institutional publication record](https://epub.ub.uni-muenchen.de/135053/).

The selector is **Model Predictive Prompt Selection (MoPPS)**, not the cached
`passrate_beta` heuristic. This implementation follows the paper's uniform-prior,
top-k MATH variant and the authors'
[sampler](https://github.com/thu-rllab/MoPPS/blob/4110c5ab40fc9a2b9d09c70989b87c9e74bfdd17/recipe/ours/mopps.py).
The inspected upstream revision is `4110c5ab40fc9a2b9d09c70989b87c9e74bfdd17`.

Frozen settings: Beta(1,1), target success probability 0.5, decay 1.0,
candidate multiplier 16, eight binary verifier rewards per selected prompt.
Draw a success probability from each candidate's Beta posterior, rank by
squared distance to 0.5, and retain the best batch. After the update, add
success/failure counts to the selected prompts' posteriors exactly once.
These are the MATH settings in Appendix C.3, not values tuned on our results.

## Question and experimental units

**Primary comparison: always-online reward-based MoPPS versus the actually
executed GATE**, at the same starting policy, optimizer and total remaining
compute budget. The primary contrast is **GATE minus MoPPS** final reward;
positive values favor GATE. Count GATE's diagnostic and any fresh scoring
against that same total cap. Do not substitute the winning fixed-action
control for the executed GATE result.

Online random and the original fixed-subset fresh-r/random arms are secondary
controls. Their completion does not establish completion of the primary
comparison when the actual GATE result is missing.

- Reuse certified **fresh-r-selected** prefixes at updates 25, 50 and 100.
- Held-out seeds are fixed to 3 and 4. No favorable seed/checkpoint selection.
- Add `mopps` and `random_online` at all six states: **12 new continuations**.
- Keep the existing 48-branch experiment and its fitted gate unchanged.
- Import each state only after its original gate decision barrier is frozen.
- MoPPS starts with a uniform prior at each branch. It receives only that
  branch's subsequent training rewards, never cached gradients/rewards,
  validation labels, other branches' outcomes or future observations.
- Preserve the source learner, model, optimizer, generation settings, verifier,
  independent evaluation questions and evaluation RNG seeds.

This is an **online selector replacement from a fresh-r prefix**. It does not
test when to stop an already-running MoPPS selector, and does not reproduce
the authors' from-base-model training curves or their full verl recipe.
Aggregate prefix logs cannot reconstruct a trustworthy per-prompt posterior;
we do not invent one or treat a base-model cache as current-policy feedback.

## Matching and implementation differences

Both new arms use the full original training pool and select four distinct
prompts per update, one per GPU, with eight responses each. Each step draws
`min(pool_size, 16 * 4)` distinct candidates uniformly. MoPPS uses posterior
top-k; `random_online` chooses four uniformly from the same candidate proposal.
The candidate RNG is independent of selector and response-generation RNGs.
Both arms regenerate responses on-policy; neither reuses trajectories.

The authors' large batch/epoch data loader is adapted to our frozen four-prompt
learner and per-step candidate sampling. Existing `selection_full` and
`random_full` instead retain a fixed 10% subset. Therefore the online-random
control is required to separate posterior-guided selection from online pool
access. Do not describe all arms as having identical subset persistence.

`src/train_mopps_grpo.py` is an isolated copy of the budgeted driver at
`7dc108a`, with selection, feedback and posterior evidence added. It shares
the original loss, rollout, optimizer-check and checkpoint helpers. The
scientific files frozen by `selection_switch_gpu.CODE` are not changed.

Every rank applies the same selected batch and gathered reward groups.
NCCL feedback is moved from the verifier's CPU tensor to the model device.
Checkpoint statistics contain each selection and reward group. Resume replays
only the checkpoint's committed history and verifies posterior fingerprints;
it does not reset the sampler or retain feedback from rolled-back updates.
Final policy metadata binds the posterior state and full training-pool file.

## Cost and reporting

Use the original `switch.json` total GPU-second cap without overriding it.
MoPPS and online random pay input verification, process/model startup,
selection, feedback, training, checkpoint saving and failed attempts.
Selection time is inside the metered training process even when it runs on
CPU while four GPUs are reserved. No additional scoring rollouts are taken.
Final evaluation uses the original separate reporting allocation.
Source import/provenance preparation is shared research work, not a selector
deployment operation. Its `import-cost` ledger records four reserved GPUs on
an admitted node, zero GPUs for CPU-only import, and wall time in both cases.
Source costs and interrupted research costs are not relabeled as zero.

Unknown interrupted deployment cost blocks the affected branch. Exact atomic
finish receipts can be recovered; otherwise a confirmed termination duration
and its log reference are required. A heartbeat is only a lower bound.

`summarize` validates results and writes `comparison-report.json`, containing
per-question rewards, completed updates, stop reasons, actual ledgers and
the primary Gate-minus-MoPPS contrast and paired question-bootstrap intervals
conditional on each trained policy pair. The main table shows both rewards
and actual total deployment costs **including gate diagnosis**. Secondary
MoPPS-minus-online-random/fresh-r/random contrasts remain in the JSON report.
It also reports means across checkpoints **within each seed**. There are two
independent trajectories, not six independent replicates. Missing, invalid,
failed and over-budget conditions must remain visible; no population-level
claim follows from this small extension.

## Cluster commands

The operator runs these inside the secure cluster. Existing switch jobs can
continue. The new launcher has a separate root and refuses occupied nodes;
it never stops other jobs or modifies the parent experiment.

```bash
git pull --ff-only
bash scripts/run_mopps_comparison.sh prepare
bash scripts/run_mopps_comparison.sh run
```

Run the same `run` command on each **free** allocated four-H100 node. Four
nodes can process the queue concurrently. At most 12 tasks are independent
once all source states are ready; fewer are available before prefix and gate
dependencies finish. There is no reason to reserve 16 nodes for this extension.
Do not launch it on the four nodes already occupied by the switch experiment.

Defaults use `$OM_WORK/runs/selection-switch-v1` as the read-only parent and
`$OM_WORK/runs/mopps-comparison-v1` as output. Override with `SWITCH_ROOT` and
`MOPPS_ROOT`; they must be disjoint from sources, models and the repository.
`MOPPS_PYTHON` overrides the existing experiment interpreter.

```bash
bash scripts/run_mopps_comparison.sh status
bash scripts/run_mopps_comparison.sh errors --phase train --limit 1
bash scripts/run_mopps_comparison.sh recover-cost
bash scripts/run_mopps_comparison.sh summarize
```

Status is a read-only 12-row view. DONE verifies the result receipt, not the
full scientific contract; `summarize` performs the latter. WAIT names missing
prefix/gate dependencies; RUNNING shows host and phase; stale heartbeats are
not assumed alive. No automatic failure retry loop consumes the budget.

After diagnosing a failed branch, explicitly retry only that branch:

```bash
bash scripts/run_mopps_comparison.sh retry --seed 3 --step 25 --arm mopps
```

If a hard stop left an unknown cost, recover it first using actual evidence:

```bash
bash scripts/run_mopps_comparison.sh recover-cost \
  --directory states/s3-t25/mopps --event-id EVENT_ID \
  --seconds ACTUAL_ELAPSED --reason 'scheduler termination log reference'
```

## Verification boundary

CPU tests cover upstream-equivalent ranking/count updates, fixed configuration,
binary feedback, exact posterior replay, corrupt-history rejection, budget
failure accounting, input immutability, and four simultaneous queue processes
claiming all 12 tasks without duplication. A direct in-memory comparison with
the inspected official sampler matched all selections and posterior values
over 200 updates across 10 seeds. That is algorithm conformance, not GPU
training replication. Run `bash scripts/run_mopps_comparison.sh cpu` locally.

Local regression result: **248 passed, 1 skipped** (optional plotting).
Both selectors also passed actual tiny OLMo3/LoRA CPU gradient updates and
optimizer/posterior restoration after an injected checkpoint interruption,
using synthetic token/reward groups. This does not test real verifier rollout
throughput or NCCL on the cluster.

The operator should first run one task on a free cluster node and inspect its
training/evaluation logs before adding nodes. H100 memory, NCCL and cluster
filesystem behavior cannot be certified by the local CPU tests.
