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
- The first dispatch/status/dashboard change passes scoped Ruff excluding existing test-only `RUF059`/`PLW1510`; the later checkpoint diagnostic has no new Ruff diagnostics compared with its exact pre-repair source (existing `EXE001`, `TRY004`, `PLW1510` remain). Dashboard JavaScript, standalone shell syntax and Git whitespace checks pass. Final tracked/new Python syntax coverage is 860 files (plus 98 JSON and 118 shell files). A SHA-256 comparison confirms 282 protected learner/runtime/adapter Python files match the audit-start snapshot.
- Initial full-repository serial CPU regression: **6,914 passed, 42 failed, 18 skipped, 330 subtests passed** in 1,774.96 seconds. Thirteen failures required the explicitly declared scikit-learn dependency and 24 required running Git-backed tests as repository owner kms. Both gate modules pass 53 checks after correcting those conditions. A 21-case ownership retry left only the actual runtime-diagnosis defect; its complete affected modules pass 244 checks and 5 subtests after repair. The remaining four failures were nonisolated default-interpreter fixtures and selection-cache fixtures assuming one module alias/NumPy tokens. Those fixture assumptions are corrected without changing model code. The final full serial run with declared dependencies, repository ownership and dedicated in-memory temporary files completed with **7,025 passed, zero failed, 18 skipped, 330 subtests passed and 150 warnings in 1,824.81 seconds**. The skips are 15 explicitly gated CUDA/NCCL checks, two unavailable official tokenizer checks and one inapplicable nested RLOO-meter case. These skipped checks are not represented as passed. CPU packages/tests are not evidence of actual GPU performance.

Frozen learner/runtime/adapter files, original inputs, live queues, checkpoints and results are unchanged by this audit. No remote process was killed or restarted and no real GPU experiment was launched. Local shared-filesystem probes do not verify remote NFS/H100/NCCL behavior.


## Follow-up: GPU accounting rebuild and clearer node explanations

The user explicitly requested rebuilding GPU time calculation and another whole-code check. A CPU fault injection proves that the old `PhaseLedger`/cost-report reader counted identical receipt copies as additional GPU work: one 10-second, four-GPU operation became 80 instead of 40 GPU-seconds when its receipt was copied. The replacement read-only accounting reader deduplicates by operation identity, preserves distinct real replays, validates measurements, separates inclusive invocations from contained phases, and retains unknown interrupted durations. The original writer/learner files stay unchanged. See [the v2 accounting guide](SRGC_GPU_TIME_ACCOUNTING_V2.md).

The first integration run passed 156 checks. Extended core/public-model checks passed 32 cases on local disk. A later combined 276-case run found one obsolete `Wait reason` header expectation (275 passed); the expectation is updated for the requested clearer status column. The stable-source full run passed 7,025 tests and 330 subtests with 18 explicit skips; it completed before the later partial-display correction below. Exact original logs are retained separately from follow-up checks.

Node explanations now identify the recorded cache/Prefix/arm, last recorded stage, worker-response gaps and known child-exit reason. `worker 시작 미확인` explicitly leaves the cause unknown. Ownership alone is not described as healthy training or an inferred Prefix wait. Saved endpoints with a held task lease are shown as waiting for exit/lease release. The raw WAIT and worker-registration incidents stay open.

## Final partial-display correction

A further CPU reproduction using both the exact `ea70189` time-display function and
the original frozen `PhaseLedger` found a narrower missing-versus-zero defect:
when startup alone was measured and selection only had an unclosed start, the
partial detail table displayed selection/training as zero because it copied the
ledger's defaults. This does not establish that the old code filled the complete
interrupted allocation interval with zero; that interval was already unknown.

Eight fault-injection cases fail before repair. Partial detail values now require
completed phase evidence: measured zero remains zero, absent/unclosed phases stay
unknown, and checkpoint and primary totals follow the same rule. Ambiguous zero
defaults in legacy partial summaries without per-phase completion evidence also
stay unknown; nonzero recorded evidence is retained. This changes derived display
only. Original receipts, endpoints, learner bytes and scientific comparison rules
remain unchanged.

After the full run completed, only `scripts/srgc_time_status.py` and its new accounting regression module changed. The eight affected GPU/status/results modules then passed **178 tests in 82.52 seconds**, including all eight new fault-injection cases. These overlap the full suite and are not added to its 7,025-test count. The remaining 977 source files match the full-run source fence. New reader/display/tests pass Ruff; the earlier preexisting model-status `ISC004` and node-test `PLW1510` diagnostics remain unchanged.
