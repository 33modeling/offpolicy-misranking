# One-shot selection gate

Implementation and protocol amendment, 2026-09-12.

This document supersedes the periodic checks, update-interval defaults, and
online stopping controller in `SELECTION_GATE_DESIGN_2026-09-12.md`. The user's
revised requirement is one initial decision from the entire candidate-pool
distribution, with measurement cost included. No per-epoch gate is implemented.

## Repository and isolation

The operational checkout is `33modeling/offpolicy-misranking`. This is a v2
paper experiment, not a change to the separate `offpolicy-misranking-v2`
checkout. The manuscript remains in `33modeling/offpolicy-misranking-paper-v2`.
Do not create another clone or copy scripts between v1 and v2 environments.

All implementation and tests in this change are new files. The original
`train_policy_grpo.py`, E5/Qwen/OLMo launchers, source matrices, and running
experiments are not modified. `train_selection_gate_grpo.py` has an independent
budgeted loop derived from the original trainer at `bf4c7f8`; loss, sampling,
optimizer checks, and artifact validators are imported from the existing code.
The later opt-in E5 reliability logger is not enabled in this driver.

## Decision and target

1. Before continuation training, scan the existing binary-reward cache for
   **every candidate prompt**, once. Check exact prompt/response coverage.
2. Record the mean, standard deviation, quartiles, zero-success/all-success/
   mixed-group fractions, and the complete success-count histogram. The gate's
   eight approved features are fixed before any continuation outcome is known.
3. A depth-at-most-two tree predicts the full-horizon benchmark contrast after
   paying measurement and selection costs. A positive prediction selects the
   declared method; otherwise use random. No fitted model skips the cache scan.
4. Compute selection scores once only if selected. Freeze the top 10% prompt
   subset. Random also uses one frozen, uniformly sampled 10% subset.
5. Continue ordinary on-policy GRPO on those prompts. New responses are generated
   during training; cached responses are not substituted for training rollouts.
   Never rerun the gate or rerank prompts between epochs or updates.

The candidate reward distribution describes the cached **behavior-policy**
responses, not current-policy performance. It is a proposed predictor whose
usefulness must be learned and tested, not a theorem identifying good prompts.
The expensive scorer remains the existing `low_order` method (or explicitly
`pair_u2`); the gate does not claim a new gradient estimator.

The fitted target is `selection_reduced - random_full`, not just the difference
between two equally shortened branches. A selection method can beat shortened
random training yet still lose to random training that avoids the measurement.
Model support, model/dataset/verifier/hardware scope, and training budget must
match. Unsupported cases use random; a synthetic fit cannot control GPU runs.

## CPU First

From the existing operational checkout:

```bash
git pull --ff-only
bash scripts/run_selection_gate.sh cpu
bash scripts/run_selection_gate.sh plan
```

The CPU environment needs `requirements-gate.txt`; inference from the exported
JSON tree does not require scikit-learn, PyTorch, or a GPU. No bootstrap loop is
used. CPU-only tests do not launch real training or alter source artifacts.

Checks completed locally on 2026-09-12:

| Test group | Result | Meaning |
| --- | --- | --- |
| Gate, pool distribution, fitting, accounting, process lifecycle | 83 passed in 2.37 s | System Python, CUDA hidden |
| New trainer integration and existing GRPO/low-order/E5 regression | 91 passed in 6.39 s | Torch environment, CUDA hidden |

Some lifecycle tests occur in both groups; these are not 174 distinct tests.
The integration checks include CPU toy-model scoring, fixed-subset reuse,
budget expiration before sampling, matched-parent commands, independent reward
coverage, and rejection of inconsistent budget/result records.

The deliberately synthetic counterfactual fixture gives +0.03 when selection is
accepted and -0.01 when rejected after paying measurement. Its held-out fixture
average is +0.01. These numbers verify accounting and decision code **only**;
they are not OLMo/Qwen results, evidence of generalization, or paper results.
With five independent runs, the implemented finite-family Hoeffding certificate
is vacuous (radius about 1.384 for three candidates at alpha 0.05). Do not claim
a certified improvement or manufacture more independent units by resampling.

## GPU Study

On each already allocated four-GPU node, in the same existing checkout:

```bash
git pull --ff-only
bash scripts/run_selection_gate_gpu.sh run
```

The same command works on multiple nodes. Nodes lease distinct point/arm tasks;
the admission helper distinguishes physical nodes. It replaces only prior
processes from **this new suite on this node**, not E5, Qwen, or OLMo jobs.

Defaults:

- Source: existing OLMo MATH d100 points, seeds 0 through 4, K=G=8, one optimizer
  epoch, top 10%. Source adapters and optimizers are read-only.
- Output: `$OM_WORK/runs/selection-gate-one-shot-v1`, separate from source runs.
  `GATE_ROOT` overrides it without modifying old launchers or their outputs.
- Shared per-branch budget B: source median update wall time times 100 times
  four GPUs, rounded up to 60 GPU-seconds. `--budget-gpu-seconds` overrides it.
  This is a budget rule, not a promised completion time.
- Independent test: 300 disjoint MATH questions, eight responses each. The same
  test and sampling seed are used across arms. Evaluation timeout: four hours
  per four-GPU phase; timed-out outcomes never become successful labels.
- Protocol: 15 branch jobs, three per source seed. The initial distribution is
  measured once per seed, not once per branch or epoch.

| Arm | Available budget | Purpose |
| --- | --- | --- |
| `random_full` | B | No gate measurement or scoring prerequisite |
| `random_reduced` | B-c | Random training after paying the same measurement cost |
| `selection_reduced` | B-c | Score once, then train with the remaining budget |

Here c is measured initial-distribution allocation time. Selection scoring,
model loading, validation direction, merging, checkpoints, failed attempts, and
training consume the selection arm's remaining allocation. Shortened random
does not pay selection scoring. Fixed update counts are not matched compute.
The trainer checks remaining time, not the gate, before each update; it reserves
30 seconds for saving and uses the last update duration to avoid starting a
block that appears not to fit. An external watchdog bounds the process phase.
Timing variation can still overrun a cap; those results are excluded from gate
training labels, not silently relabeled as matched-budget observations.

All three branches are **offline research supervision**. They are not run as a
deployment-time probe on every dataset. The default role is `development`, not
held-out validation. Separate whole source trajectories must be declared for
calibration/test; the same checkpoint under another name is not an independent
unit. A five-seed pilot does not establish a universal gate.

```bash
bash scripts/run_selection_gate_gpu.sh status
bash scripts/run_selection_gate_gpu.sh live
bash scripts/run_selection_gate_gpu.sh summarize
```

`status` lists all arms, including pending/failed ones; `--json` gives full data.
`live` follows all node launcher logs. Worker details are in each arm's phase
logs. `summarize` writes `study.json` and explicitly lists excluded points.
An incomplete point is not converted to a zero reward or a successful result.

Fit only after observed matched-budget labels exist:

```bash
bash scripts/run_selection_gate.sh fit --study PATH/study.json --out PATH/gate.json
bash scripts/run_selection_gate.sh analyze --study PATH/heldout-study.json --model PATH/gate.json --out PATH/report.json
```

The paths are the actual output files shown by preparation. Dataset/hardware/
budget and declared roles must agree. Reusing development trajectories as test
data is rejected. There is currently no observed fitted model shipped here.

## Deployment and recovery

Prepare a separate new output root with `--mode deploy --model PATH/gate.json`,
then use `run` on that frozen root. Deployment executes only the initially chosen
continuation. Without a model it uses random, without scanning the reward cache.
A failed selected scorer falls back to random once and retains the cost already
spent; it does not relaunch the gate. The research selection arm instead stays
failed so that it cannot masquerade as a successful selection counterfactual.

Normal restart reuses the gate decision, profile, selected IDs, checkpoints, and
finished evaluation shards. Completed results are not retrained. A failure is
recorded once per launcher attempt and other arms are tried; there is no endless
automatic retry loop. Unknown cost after an unclean kill is reported explicitly
and that arm cannot enter a matched-cost comparison until reconciled. Other
independent arms remain runnable. Do not delete the cost ledger to make it pass.

Costs are allocated GPU-seconds, including GPUs idle during measured CPU phases,
with research/deployment/reporting ledgers kept separate. Independent test
generation is reporting cost and must still appear in the study's resource
report. Source-cache production, offline fitting, launcher setup, scheduling
gaps, and allocation outside instrumented phases are **not** free and are not
fully measured by this pilot. Obtain job-accounting logs before claiming an
end-to-end savings or amortization result.

Real four-H100 execution, distributed checkpoint timing, cluster cancellation,
and observed benchmark improvement have not been tested on this local machine.
Start with an allocated node; expand to other available nodes once the first
branch produces valid training and cost records. Existing jobs need no restart.
