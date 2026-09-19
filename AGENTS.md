# Experiment safety and incident memory

Before changing experiment budgets, accounting, resume/restart, status, or result
publication, read [the MBPP budget incident](docs/INCIDENT_MBPP_BUDGET_2026-09-19.md)
and the relevant current runbook. These are standing requirements from the user,
not permission to change a running experiment.

## Do not repeat the budget incident

- The goal is preserved, scientifically usable experiment results, not merely a
  process that starts or consumes its allocation. Budget exhaustion is NOT
  completion. Require the protocol's valid evaluation/results and any required
  curve/report artifacts before reporting `DONE`.
- Establish the comparison objective and read the actual frozen MBPP manifest
  and calibration before choosing or changing a cap. Do not copy a MATH cap
  merely because the runner is shared. Do not change a frozen cap/accounting
  contract silently or add optional experiment conditions without authorization.
- Distinguish selection, diagnosis, training, and evaluation costs. A training
  allocation is neither total GPU cost nor an end-to-end completion estimate.
  Selection charged separately is not free. Unknown cost is not zero.
- Before a budget/resume fix, demonstrate the complete path: cap reached ->
  saved policy/checkpoint -> required evaluation -> validated result publication
  -> repeated launch skips completed work. Include crash/restart tests. CPU
  tests are not evidence that the remote GPU experiment completed successfully.
- Reuse verified Selection AND Random work. Never reset manifests, ledgers,
  checkpoints, selected data, completed results, or status to make a run start.
  Distinguish a display bug from missing data using read-only evidence first.
- If no training allocation remains, do not repeatedly run selection/training
  or grant a fresh allocation. Finish only protocol-authorized evaluation from
  valid saved work; otherwise explain the exact blocker and obtain direction
  for a scientifically explicit protocol change.
- Preserve the simple status vocabulary `READY`, `RUN`, `WAIT`, `DONE`.
  Put budget/phase details in Remarks. Show full node names, assignments, and
  evidence-based progress; time-limit usage is not experiment completion.
- Once the user says a run is working, do not restart, clean processes, change
  its configuration, or deploy a change just to verify it. Use read-only status.
  The plain MBPP command currently restarts this node even with unchanged code;
  never present it as a monitoring command. Repository updates can also trigger
  launcher reloads; recording this incident does not authorize a deployment.
- Give one simple command when possible; no environment-variable setup or
  `tail` instructions for the user. Keep diagnostic attachments bounded.
- Distinguish implementation, local verification, user-reported progress, and
  independently verified cluster outcomes. Do not promise that all CUDA errors
  are fixed or give a total-time estimate based only on a training cap.

## Budget incident follow-through

- Before claiming an exhausted run can finish, verify the saved policy ->
  reporting-only evaluation -> curve -> sealed result path. Do not charge
  deployment/input verification again merely to resume an already stopped
  training run at its cap. Preserve all legitimate prior costs.
- No training checkpoint does not mean no reusable work: selection can have
  saved rollout groups and prompt gradients. Inspect their provenance before
  repeating work; do not reset or waive their costs to create fresh allocation.
- Record dataset, frozen cap, charged ledger categories, and calibration basis
  before discussing budget adequacy. Equal training caps do not imply equal
  total compute or completed update counts, and cap/GPU count is not an ETA.
- Preserve the user's last reported MBPP status: 21 DONE and 27 remaining,
  48 total. This is user-reported, not independent artifact validation.
  The accompanying "6" was not clarified; do not assume it means nodes or GPUs.
- Incident documentation is not permission to restart a working experiment.
  Check automatic pull/reload consumers before pushing. Documentation commits
  can trigger revision-based reloads too; use a separate incident branch when
  the running branch auto-pulls unless that operational change is authorized.
