# Repository debugging and launcher diagnostics — 2026-10-31

The user requested a whole-repository debugging pass and permanent bug-board update after repeated `MODEL CHAIN WAIT`, then asked what `model launcher active; dataset worker not registered` means and whether the child keeps restarting.

## Scope and source state

The audit started on existing `master` at `72a9a33128c5d095c45fc11f8adf6180da732e9e`. Six preexisting modifications belonged to a concurrent Gemma selection/NCCL fix; they were preserved and subsequently committed by that work in `cd0cb20503bc0d7294b8be3dca138d18c180b262` and `db93271fd2be44a971bd60756b69d130576cc613`. Their source hashes match the audit-start snapshot. Nine unrelated untracked paths remain untouched.

Static inspection covered all 857 tracked Python files, 98 JSON files and 118 shell files. All parsed successfully. Repository-wide Ruff checks for undefined/exported/unbound local names (`F821,F822,F823`) found no diagnostics. Manual review followed the public shell entrypoints, model coordinator, task admission and claims, shared backlog, GPU admission/cleanup guards, checkpoint/rollout persistence, results export, status merging and the permanent dashboard renderer. This does not mean every line has been manually inspected or every runtime condition has been proven correct.

## Reproduced defects

### Requested MBPP queue admitted despite an unclaimable task

`resume_drain.runnable` checked the task-level early-MBPP rules only when classifying a sibling queue. A requested MBPP queue could therefore be called runnable while the claim rejected its training arm because MATH was unfinished or a completed MATH endpoint was still leased. Resume-only requested prefixes could likewise be admitted without the panel required by their claim.

The fix applies the same `task_allowed` predicate to every candidate before GPU admission, using the existing normal/resume-only panel policy. It retains the MATH barrier, early cache/prefix permission, retry delay, task leases and model order. Twelve regressions cover normal/resume-only modes, unfinished/leased MATH, early-panel states and full worker rejection before any GPU query/admission; checkpoint bytes remain identical.

Running those twelve tests with the exact old `resume_drain.py` from `db93271` loaded in a separate CPU process produced **7 failures and 5 passing controls**. The four original admission/claim disagreement probes failed before the fix as well. The fault is locally reproduced; it has not been established as the cause of the user's actual remote WAIT or LAUNCH message.

### Dashboard evidence schemas lost report text and file links

The dashboard assumed every evidence item had `label` and `url`, but 21 entries used `user_report/text` or `source/commit/path`. Those entries produced empty labels and `#` links, hiding the actual report and source file. The real incident renderer failed both initial schema probes before repair.

The renderer now shows escaped report text, preserves line breaks and constructs GitHub file links from validated commit/path evidence. Existing labeled links are retained, and incomplete/invalid evidence stays readable without manufacturing a `#` link. Nine Node-backed regressions execute the actual renderer, including every stored incident.

### Checkpoint diagnosis silently imported another runtime's recovery helper

The full repository run exposed a failing missing-runtime diagnostic. It remained reproducible as the repository owner after the Git ownership failures were removed. `selector_pair_checkpoint_audit` prepended the requested paths but used an unrestricted normal import: when the requested helper was absent, ambient `PYTHONPATH` could supply a different checkout; a cached module from another runtime could also be reused. A new cached-helper regression run against the old function returned success (`0`) instead of the required runtime error (`2`).

The diagnostic now requires the helper under the requested runtime and checks the loaded module's resolved filename before inspecting any contract. The standalone shell carries the identical audited Python. The missing-runtime test explicitly plants an ambient helper and proves it is never imported, and the cached-helper test proves no contract is inspected with foreign code. Existing report persistence, return codes, paths with spaces, saved artifact bytes and read-only behavior remain covered. The relevant checkpoint/deployment/Pair launch/results/handoff modules pass **244 tests and 5 subtests in 38.26 seconds**. These are diagnostics and CPU control checks, not new model experiments.

## Launcher diagnostics and unresolved remote report

`LAUNCH` with no dataset worker is evidence that the coordinator has a child launch in progress and no matching worker receipt has been selected. It does not establish a training phase, GPU utilization, a healthy child, or an infinite pending state. The existing coordinator can launch the same model again after Busy/Yield/interruption, but prior receipts lost the attempt count and previous exit when the next launch began.

Coordinator receipts now retain each model/dataset launch cycle, launch start and previous exit code/time. Heartbeats keep the current child PID. Read-only status and `MODEL CHAIN WAIT` display recorded PID, cycle, previous exit and launch age, allowing repeated exits to be distinguished from a long current launch. Seven regressions cover repeated Busy/Yield/SIGKILL, unknown legacy fields and the three model status paths. These fields describe coordinator cycles, not optimizer updates or proof that a PID is alive; cycle counts restart for each model/dataset drain. Existing running coordinators cannot acquire new fields through `git pull`; their missing history remains unknown.

The node lease remains authoritative and is not deleted or bypassed. The earlier `2f9d9e6` patch changes immediate Busy exit to waiting; it does not release a continually held lease. The original remote hostname cannot be resolved from this environment and `/group-volume` is not mounted here. The user's actual lock holder and worker-registration stall remain unresolved. Their reports must remain open on the dashboard.

## Verification

- Initial targeted repair checks: **316 passed in 73.71 seconds**.
- Combined admission, model chain, actual CPU subprocess/signals, node status, dashboard evidence and ledger retention: **409 passed in 103.84 seconds**. These overlap the first set and are not additive totals.
- The first dispatch/status/dashboard change passes scoped Ruff excluding existing test-only `RUF059`/`PLW1510`; the later checkpoint diagnostic has no new Ruff diagnostics compared with its exact pre-repair source (existing `EXE001`, `TRY004`, `PLW1510` remain). Dashboard JavaScript, standalone shell syntax and Git whitespace checks pass. Final tracked Python syntax coverage is 858 files. A SHA-256 comparison confirms 282 protected learner/runtime/adapter Python files match the audit-start snapshot.
- Full-repository CPU regression results will be appended after the active run finishes. The initial environment lacked the explicitly declared `scikit-learn` dependency from `requirements-gate.txt`; its failures are tracked separately and rerun after installing it in the CPU test virtual environment. No installed package is used as evidence of actual GPU performance.

Frozen learner/runtime/adapter files, original inputs, live queues, checkpoints and results are unchanged by this audit. No remote process was killed or restarted and no real GPU experiment was launched. Local shared-filesystem probes do not verify remote NFS/H100/NCCL behavior.
