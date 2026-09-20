# Matched RLOO Continuations

The GPU entry point is `scripts/run_rloo.sh`. With no arguments it automatically
prepares and validates inputs, admits an idle four-GPU node, and runs the actual
experiment. There is no smoke experiment, smoke prerequisite, or separate
preparation command required. `status`, `plan`, `prepare`, `check`, `report` and
`cpu` remain explicit non-GPU modes.

## Scope

- Existing MATH d0/d400 source points, seeds 0/1/2, 100 additional RLOO updates per arm.
- Arms: `random`, cached difficulty proxy `passrate_beta`, and `fresh_r`.
- Eighteen training runs, plus six parent-policy evaluations. No smoke run.
- Reuse the original top-10% selections and frozen, disjoint 300-question E5 test.
- Evaluate eight responses per question with identical sampling seeds across arms.
- RLOO uses the existing sequence-sum REINFORCE loss, leave-one-out baseline,
  and exactly one epoch per batch. No new loss implementation is introduced.
- `fresh_r` denotes the original GRPO-study score, not a newly computed RLOO score.
  This tests transfer of fixed selections to a different learner. It does not
  establish RLOO-native selector quality, switching behavior, or an H boundary.
- d0 starts from the base model. d400 loads the exact original GRPO adapter
  and optimizer state and continues steps 401 through 500. The original d400
  parent is also the evaluation baseline. No extra prefix is trained.
- Base model, selected subsets, optimizer configuration, generation settings,
  LoRA configuration, seed, update count and evaluation match the GRPO study.
  Only the continuation objective changes. d400 is an algorithm-switch study,
  not an assertion that its entire training history was RLOO.

`train_policy_rloo.py` adapts only the canonical trainer's parent-objective
checks, including final/resumed lineage validation. Parents remain GRPO;
children are RLOO. All parent hashes and configuration checks remain active.
The in-memory adaptation is structurally checked before GPU admission and
fails closed if canonical validation changes. Existing shared trainer, Pair
and MBPP files are untouched.

## Entry Point

`scripts/run_rloo.sh` defaults to GPU `run`, with automatic CPU preparation.
Modes: `plan`, `prepare`, `check`, `status`, `run`, `report`, `cpu`.

Defaults use `OM_WORK`, the existing `olmo3-1025-7b-base-rlzero-grpo-h100-v2`
source families, and `inputs/e5-reduced/test-math500-d{0,400}.json`.
`RLOO_ROOT` defaults to `$OM_WORK/runs/rloo-selector-v2`, with
`math500-d{0,400}/s{0,1,2}` beneath it. The earlier d0-only v1 preparation is not
overwritten or treated as a completed matched experiment.
`RLOO_PYTHON` may select the existing environment. Preparation accepts
`--source-root` and `--eval-prompts` when the server paths differ. Subsequent
default launches reuse these recorded input paths.

Preparation validates all six source points before writing per-point contracts.
It requires completed source scores, the model snapshot, and the actual test
file. Missing server files are errors, not synthetic replacements.
Source artifacts, selections, evaluation questions, and source-code hashes are
bound to the contract. A changed contract requires a new output root.

No separate preparation or smoke invocation is needed. The default per-phase
hang guard is 86400 seconds, overridable with `RLOO_MAX_PHASE_SECONDS` or
`run --max-phase-seconds`. This is not the experiment's resource budget,
a time estimate or a stopping target: every arm requests 100 updates and full
evaluation. Timeout remains failure, never DONE.

Multiple explicitly launched workers claim distinct seed/arm locks. Each worker
requires one free four-GPU allocation. Shared node admission refuses occupied
GPUs; no cleanup or cancellation of existing experiments is performed.
Interrupted training resumes the canonical trainer's checkpoints. Completed
evaluation shards are reused only after checking hashes and exact coverage.
Run `status` for global completion, not a single worker's exit message.
The launcher status uses the same dashboard as MBPP and Selector Pair:
summary counts, FULL STATUS per seed/checkpoint, CURRENT RUN, node assignments,
and phase progress. `status --all`, `status --json`, and `status --watch [SECONDS]`
are supported; watch defaults to 15 seconds. The 18 continuation arms are the
training total. Six shared baseline evaluations appear in the Before column,
outside that total. Training without all four sealed evaluation shards is not
DONE, and stale heartbeats or a lock file alone never establish RUN.
Status is read-only and does not create worker locks or touch GPU admission.
It verifies evaluation receipt bindings, rollout hashes and question coverage;
full source/model/optimizer validation remains in `check` and `report`. The
dashboard lives outside frozen training sources so this display update does not
invalidate existing RLOO contracts.

## Readout And Costs

Completed points automatically write `results.json`; `report` also allows
explicit regeneration after all trained arms and parent evaluations validate. Per-seed results
contain mean reward and paired differences against base, random, and cached
`passrate_beta`, with prompt-bootstrap intervals. These are conditional on a
training seed, not across-seed confidence intervals.

Training and evaluation receipts include failed attempts and retries. They cover
new allocated GPU phase time only. Historical selection costs are neither
remeasured nor treated as zero. No total-cost advantage or H value is reported.
Matched update counts define the comparison; a safety timeout is a failure,
not a budget-completed experiment or a valid zero-performance observation.

Do not update the manuscript with RLOO result claims until real results have
passed artifact validation. MBPP is outside this comparison. Local CPU checks
are not evidence of an actual GPU run.
