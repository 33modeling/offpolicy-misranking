# RLOO Frozen-Selection Comparison

Status: execution tooling prepared; real server inputs and GPU smoke not yet verified.
No GPU work is launched by preparation. Existing Pair, MBPP, and E5 runners are unchanged.
Local verification: 58 CPU tests passed across the new workflow, canonical
policy trainer, independent evaluation, and budgeted-training contract suites.

## Scope

- MATH-500 source point d0, seeds 0/1/2, 100 native RLOO updates per arm.
- Arms: `random`, cached difficulty proxy `passrate_beta`, and `fresh_r`.
- Nine training runs, plus three base-model evaluations and a separate smoke run.
- Reuse the original top-10% selections and frozen, disjoint 300-question E5 test.
- Evaluate eight responses per question with identical sampling seeds across arms.
- RLOO uses the existing sequence-sum REINFORCE loss, leave-one-out baseline,
  and exactly one epoch per batch. The trainer is not replaced or patched.
- `fresh_r` denotes the original GRPO-study score, not a newly computed RLOO score.
  This tests transfer of fixed selections to a different learner. It does not
  establish RLOO-native selector quality, switching behavior, or an H boundary.
- d400 is intentionally rejected. Its parent is GRPO and the trainer rejects
  cross-objective resume. A future, explicit algorithm-handoff protocol must
  define optimizer handling and provenance before that comparison is enabled.
  No extra 400-step prefix is silently scheduled.

## Entry Point

`scripts/run_rloo.sh` defaults to CPU-only `status`, never to GPU execution.
Modes: `plan`, `prepare`, `check`, `status`, `smoke`, `run`, `report`, `cpu`.

Defaults use `OM_WORK`, the existing `olmo3-1025-7b-base-rlzero-grpo-h100-v2`
source families, and `inputs/e5-reduced/test-math500-d0.json`.
`RLOO_ROOT` defaults to `$OM_WORK/runs/rloo-selector-v1`.
`RLOO_PYTHON` may select the existing environment. Preparation accepts
`--source-root` and `--eval-prompts` when the server paths differ.

Preparation validates all three source points before writing per-seed contracts.
It requires completed source scores, the model snapshot, and the actual test
file. Missing server files are errors, not synthetic replacements.
Source artifacts, selections, evaluation questions, and source-code hashes are
bound to the contract. A changed contract requires a new output root.

Run `smoke` before `run`. Both require an explicit positive
`--max-phase-seconds` safety timeout; no GPU time allowance is guessed here.
The smoke uses seed 0, the random subset, two updates, and four test questions
with two responses each. Its output is isolated and excluded from the report.
The full worker refuses to run until smoke training and evaluation validate.
Choose the full safety timeout after observing smoke, allowing for the full
300-question evaluation; a two-step smoke is not a reliable total-time estimate.

Multiple explicitly launched workers claim distinct seed/arm locks. Each worker
requires one free four-GPU allocation. Shared node admission refuses occupied
GPUs; no cleanup or cancellation of existing experiments is performed.
Interrupted training resumes the canonical trainer's checkpoints. Completed
evaluation shards are reused only after checking hashes and exact coverage.
Run `status` for global completion, not a single worker's exit message.

## Readout And Costs

`report` requires all trained arms and their base evaluations. Per-seed results
contain mean reward and paired differences against base, random, and cached
`passrate_beta`, with prompt-bootstrap intervals. These are conditional on a
training seed, not across-seed confidence intervals. Smoke is never pooled in.

Training and evaluation receipts include failed attempts and retries. They cover
new allocated GPU phase time only. Historical selection costs are neither
remeasured nor treated as zero. No total-cost advantage or H value is reported.
Matched update counts define the comparison; a safety timeout is a failure,
not a budget-completed experiment or a valid zero-performance observation.

Do not update the manuscript with RLOO result claims until real results have
passed artifact validation. MBPP and d400 are outside this prepared pilot.
