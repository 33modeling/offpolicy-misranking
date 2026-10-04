# Experiment runtime fixes October 4 2026

Follow-up to [the four reproduced audit findings](V7_EXPERIMENT_CODE_AUDIT_2026-10-04.md).
Base commit: `0d44bbe9fa48cb0e2646f6b4378f1a42477c775a` on `master`.
No remote experiment was restarted. Existing plans, inputs, checkpoints,
results, active pointers, manuscript claims and historical evidence are unchanged.

## Repairs

| Finding | Change | Regression coverage |
| --- | --- | --- |
| F1: unrelated shared-memory deletion | Disable the filename/UID-based sweep. Unknown `torch_*` and `nccl-*` files are never removed automatically. Owned process-tree termination and GPU admission checks remain. | Live foreign handle, unknown files, directories, empty process table; existing process-guard tests |
| F2: resume changes attention | Read saved attention before model construction, inside the runner's execution lease. Prefer the arm checkpoint, then the shared prefix; absent legacy metadata means eager. Reject a mismatch before engine restoration. | Saved eager/SDPA, prefix inheritance, arm precedence, legacy default, invalid metadata, model-loader hook cleanup |
| F3: interrupts exhaust retries | Keep monotonic `attempt` and immutable attempt receipts; use `retry_attempt` for the budget. Refund intentional interruption, not an actual failure or an invalid run. Base, multi-dataset and Qwen workers use this budget during retry cooldown. | Six consecutive interrupts, prefix dependency release, interleaved real failures, all three worker loops |
| F4: forged MBPP completion | Child evaluates test expressions and exports bounded literal values. Parent holds expected answers and performs equality checks; the child cannot award a pass by sending a completion token. | Forged completion, wrong values, equality hooks, normal literal values, nested constructor inputs, malformed/oversized output, timeout and descriptor cleanup |

F1 deliberately leaves unidentified stale shared-memory files in place. File
ownership must be established before manual cleanup; matching a UID or filename
is insufficient. This repair does not certify intermittent NCCL failures fixed.

F2 keeps the checkpoint kernel even when `SRGC_ATTENTION` differs. Fresh runs
still use that setting, defaulting to eager. The log names the effective kernel
and source checkpoint. The base OLMo wrapper is the affected path; extra arms
already restore their saved attention, and Qwen pins its setting separately.

F3 does not discard failure history or grant unlimited retries after real
errors. Legacy receipts without `retry_attempt` refund only the last explicitly
recorded interruption; missing earlier history is not reconstructed by guessing.
Frozen old runtimes retain their recorded queue semantics.

## MBPP protocol boundary

New verifier version: `parent-checked-values-v3`.
Old version: `assertion-completion-v2`.

The prepared MBPP corpus contains 800 problems and 2,400 assertions. Every check
is equality against a literal expected value. The new parser covers all of them.
The candidate sees the expression/input, but not the verifier's expected-value
list or a pass token. The parent decodes returned strings with `ast.literal_eval`,
never `eval` or pickle, and compares them with its own expected values.
`assert False` cannot be overridden by a child report, including an empty one.

Supported results are literal builtin values/containers. Custom objects,
subclasses, executable equality hooks and unsupported assertion syntax are not
silently accepted as equivalent. Unsupported records raise an input error;
invalid candidate outputs score zero. This is a versioned reward-contract
change, not a claim that every old generated response receives the same reward.
This checks returned values, not the provenance of each internal function call.
Known answers can still be hardcoded or directly emitted; the protocol does not
provide test secrecy or Python execution attestation.

Output is capped at 1 MiB. Time, memory, process-group cleanup and descriptor
closure are retained. This is not an OS security sandbox: generated programs
still need isolated, disposable compute without credentials or valuable writable
mounts. No inspection of actual archived rollout responses was performed, so
the fix does not establish that historical rewards were exploited or corrupted.

Old caches cannot enter a v3 experiment through cache resume or input validation.
Do not relabel old rewards, combine different verifier versions in one comparison,
or regenerate rewards in an existing run directory.

## Historical continuation

The new root-package digest is
`b5dd86cfbe7b3b50691834233578f81d45fb0ca82a843d96dab474154be33ac2`.
Changing package code intentionally changes experiment identity.

Existing extra-arm launches can still use the two verified saved implementations:
`f581eb043e89409e` and `9435e80003f41e1f` (abbreviated digests). Six changed support
modules have byte-exact release copies under `scripts/frozen_srgc/pre_20261004/`.
The manifest pins all original package files; activation verifies the full digest,
loads the original engine and support modules, and refuses an unknown or modified
runtime. The preexisting frozen engine file is unchanged.

An extra arm from an old MBPP prefix deliberately retains **v2**, not v3. That
path preserves reproducibility; it does not receive the corrected reward protocol.
The base/Qwen queues refuse a changed implementation on an existing root. Their
root and active pointer are not silently switched, and completed P0 is not rerun.
Continue historical work with its pinned runtime. Only explicitly new experiments
may use corrected code with separate outputs and newly verified rewards.

Read-only status/results keep showing old MBPP endpoints with their recorded
verifier version and an explicit warning. Structural input validation and endpoint
identity checks remain mandatory; only current-training eligibility is bypassed
for display. Qwen preparation can copy question splits from old OLMo inputs, but
always discards their rewards and preserves their source provenance. These two
read paths do not relax cache admission for training or regenerate old data.

## Verification

Final SRGC suite: **461 passed, no skips**, plus 301 passed subtests, in 94.53
seconds. The subtests are not added to the top-level unique-test count. The five
warnings concern multiprocessing `fork()` in multithreaded test fixtures.

Static checks: **334 tracked Python files** compiled and **111 tracked shell
scripts** passed `bash -n`; Ruff `F821,F823,F811` and `git diff --check` passed.
All six new archived module copies match the source commit byte for byte.
The preexisting frozen engine, input bundles, plans and audit evidence have no diff.

Historical tests: **4,579 passed, 16 skipped**. The complete run started before
the final read-only/Qwen compatibility edits and finished without interruption:
5,035 passed, 16 skipped and three failed identity assertions in 1,163.33 seconds.
Those three assertions had already loaded the intermediate `c0a28f...` digest
before the package changed to `b5dd86...`. They are not counted as passing from
that run. A fresh final-code SRGC run passed all 461 tests, and a further explicit
rerun of the three identity tests passed all three in 0.71 seconds.

Deduplicating by test ID gives **5,040 passed, 16 skipped, no remaining failures**
across 5,056 top-level tests. This is aggregate coverage, not a claim that one
uninterrupted full-suite invocation passed against the final tree. Sixteen passing
tests belong to the preexisting untracked CFCS file and are not in this commit;
the tracked-only totals are 5,024 passed and 16 skipped. Fifteen skips require an
explicit CUDA device; one nested-curve-meter case is inapplicable to RLOO.

Environment: CPU PyTorch 2.13.0, Transformers 5.14.1, PEFT 0.20.0 and
math-verify 0.9.0, with supplemental test dependencies under `/tmp` only.
Local records: `/tmp/srgc-fixes-20261004-full.{log,xml}`,
`/tmp/srgc-fixes-20261004-suite-final.{log,xml}` and
`/tmp/srgc-fixes-20261004-identity-final.xml`.

Recheck from the repository root in the complete test environment:

```sh
env PYTHONPATH=src:scripts:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m pytest tests srgc_rebuttal/tests -q --tb=short
```

Tests use temporary fixtures and tiny CPU OLMo/Qwen models, not production training.
No H100, production FLA, remote shared-volume load or multi-node NCCL validation
was performed. The preexisting untracked research/hotfix files are not part of
this change.
