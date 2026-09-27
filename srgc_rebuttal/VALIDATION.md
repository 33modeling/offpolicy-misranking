# Existing-runtime repair on 2026-09-28

The additional-study entry point now reuses Pair/MBPP interpreter selection,
the operational OLMo runtime gate and local model loader, and the existing
offline verifier bundle. SSH workers use the same entry point. Exact new
Transformers/PEFT pins no longer reject an otherwise compatible working
environment. No existing experiment environment was modified or reinstalled.

All 96 unit/integration tests passed with no skips. The six added regression
tests cover interpreter selection, existing runtime/model-loader delegation,
local snapshot provenance, and guarded recovery of an admission-only queue.
Queues with tasks, cache receipts, checkpoints or live workers still reject
implementation changes. The two-rank CPU distributed smoke also passed both
global-update/timing parity and failure propagation without hanging.

Verification used temporary CPU tooling with Transformers 4.57.6 and PEFT
0.20.0; reuse of other compatible versions is covered by mocked compatibility
tests, not a claim of GPU validation on every version. The user's exact
Transformers traceback was not supplied. No H100 training or scientific
result was produced by these tests. Existing inputs and protocol are unchanged.

## Master publication on 2026-09-28

The three experiment commits through `937b949` were fast-forwarded onto
`master` at the author's request. Current execution guides now point to
`master`; the branch names below describe historical verification runs.
No experiment protocol, input data or historical result was changed by
this publication. User-owned untracked files were left untouched.

The full 90-test suite passed again on `master` with no skips, using CPU
PyTorch, Transformers 4.57.6 and the deployment-pinned PEFT 0.20.0. The
runtime code, tests and input bundles match `937b949` exactly; only branch
guidance and repository instructions changed. No H100 workload was launched.

## Node-parallel and MBPP verification on 2026-09-27

Branch: `experiments/srgc-cost-replication`. Changes are confined to the code
repository. The manuscript repository and scientific outcomes are unchanged.

- 76 unit/integration tests passed with no skips, including the tiny OLMo-3
  generation, gradient, optimizer and timing tests from the migration suite.
- Two spawned CPU worker processes executed all 30 synthetic cache/prefix/arm
  tasks exactly once and respected per-seed dependencies. Four workers also
  passed the existing 25-task already-cached case.
- Tested immutable cache handoff, input-change rejection, completed-task skip,
  bounded failed retries, preserved attempts and abandoned-task recovery.
- Tested partially overlapping GPU-UUID leases and real subprocess inheritance
  of a task lock after the parent descriptor closed. Immediate stop terminated
  the owned test process and released its lease.
- Tested worker-loop orchestration, mocked multi-host SSH preflight/startup
  acknowledgements, failed remote-directory handling and final paired reports.
- Tested MBPP input preparation from 800 synthetic fixture records, consistent
  split across seeds, no overwrite of different data, independent plan/output
  selection and dataset/plan mismatch rejection. Fixtures are temporary test
  data, not experimental inputs or reported results.
- CPU `status --dataset math` reports five ready cache tasks, five prefixes
  waiting for cache and twenty continuations waiting for prefix; no worker was
  launched by this inspection.

Environment: Python 3.12, CPU PyTorch 2.14.0, Transformers 4.57.6, PEFT 0.21.0.
No real remote SSH node, shared cross-node filesystem, CUDA allocation, MBPP
download or 7B experiment was exercised. Actual node access is still required
to validate cluster hardware and measure the new scientific outcomes.

## Code-repository migration verification on 2026-09-27

Branch: `experiments/srgc-cost-replication` in `33modeling/offpolicy-misranking`.
The active implementation, tests and five input bundles are maintained here,
not in the manuscript repository. New CLI: `scripts/run_srgc_rebuttal.py`.

Migration and instrumentation checks:

- 60 unit/integration tests passed, with no skips, including tiny OLMo-3
  generation/scoring/training/evaluation and exclusive timing reconciliation.
- Unequal two-rank CPU scoring shards complete without deadlock; distributed
  four-prompt training matches the single-rank update. Raw rank receipts are
  recorded and collected successfully.
- Tests cover nested timers, straggler accounting, CPU versus allocated GPU
  time, completed retries, interrupted phases/invocations, missing totals and
  Python launcher help from outside the repository.
- Five existing real input bundles were checked against the pinned source
  rows; the preparation manifest was regenerated for the new paths/hashes.
  Cached rewards remain empty. No model training result was invented.

Environment: Python 3.12, CPU PyTorch 2.14.0, Transformers 4.57.6, PEFT 0.21.0,
math-verify 0.9.0. The deployment requirement pins PEFT 0.20.0; this local run
does not establish GPU performance or exact deployment-package equivalence.
No pretrained weights, real GPU jobs or multi-node experiment were launched.

## Pre-migration verification on 2026-09-27

The author clarified selection refresh to every 25 updates. The new runner
keeps the top four prompts until the next refresh and generates fresh training
responses at each update. V6 stays frozen. This verifies implementation and
input preparation, not the submitted experiments or new 7B outcomes.

Current checks:

- Full unit/integration suite, including periodic selection, mid-block resume,
  shared model/optimizer prefix, actual candidate IDs, reused 40-vs-40 D,
  both temporal conditions, split leakage, queue leases and complete paired
  summaries. Exact final counts are recorded in the review report.
- Tiny OLMo-3 generation, dense scoring gradients, GRPO/RLOO LoRA updates,
  micro-batching, chunked-head gradients and optimizer restore.
- Two CPU distributed processes: gathered projected gradients and the global
  four-prompt optimizer update match a single process.
- Durable phase receipts preserve completed retry costs. Open timers mark
  measurement incomplete and the summary omits a complete cost total.
- Cache resume uses stable prompt-specific seeds and atomic response receipts;
  source/model/settings changes and duplicate imported rollout indices fail.
- Five real MATH-train bundles: 400/100/300 disjoint splits, 50 explicit online
  validation prompts, all 800 selected gold answers self-verify. Input hashes
  match prepared_inputs.json. Pending caches are rejected by training checks.
- Three dataset plans retain selection_interval=25. All 271 submitted V6 files
  match archive 3d144ee666e480766e253d06221f6e29feaef25e.

Test environment: Python 3.12, CPU PyTorch 2.13.0, Transformers 4.57.6,
PEFT 0.20.0, huggingface-hub 0.36.2, math-verify 0.9.0. The same numerical
suite also passed with the local Transformers 5.14.1 installation before
validating the pinned 4.57.6 deployment requirement. No pretrained 7B weights
were downloaded. Prepared real inputs contain no fabricated cached rewards.

Not run: real multi-node GPU work, initial-policy cache generation, additional
7B training seeds, GPU peak memory or throughput measurements. The current
computer cannot use its NVIDIA driver; GPU node access remains necessary.

## Historical verification (2026-09-26, superseded protocol)

The record below describes the original every-update implementation before
the author's interval clarification; it is not the current execution status.

# Verification on 2026-09-26

The source protocol is the V6 manuscript at `e649d2e`. This is a newly written
implementation, not a verification of archived training code or a reproduction
of the paper's reward/cost values.

Passed:

- 30 unit/integration tests: full 40-vs-40 D, overlap cancellation, cosine
  versus inner-product behavior, LOO subgroups, GRPO token normalization,
  temporal boundaries and missing checks, per-step scoring, no extra D model
  call, stopping scoring on switching, shared-prefix and optimizer restoration,
  resume determinism, split leakage, complete paired-seed summaries, and tiny
  OLMo-3 fresh generation/dense gradients/LoRA GRPO and RLOO updates.
- Four spawned CPU workers: all 25 synthetic queue tasks executed exactly
  once, independent work overlapped, and each continuation waited for its own
  prefix. Restart skipped completed work. Failure/retry, duplicate leases,
  changed-input rejection and recovery after final-endpoint publication passed.
- Tiny OLMo-3: micro-batches of one and four yield matching scoring gradients
  and optimizer updates; eight active responses use four batched forwards at
  micro-batch two, with only one projection per parameter per prompt. Chunked
  LM-head log-probabilities and derivatives agree with full model forwards.
- Summaries reject mixed implementation and prefix hashes across paired arms.
- Two-process CPU distributed test: gathered per-prompt gradients and the
  global four-prompt update agree with the one-process implementation.
- math-verify adapter: a correct boxed answer receives 1 and an incorrect
  answer receives 0.
- CPU categorical-policy demo: two seeds, four arms each, common prefix 25,
  total update 30; output explicitly labeled synthetic and kept outside Git.
- Python compilation of all implementation modules.
- Additional-seed plan: seeds 5–9, four arms, shared prefix 25, endpoint 275;
  single-node and SSH worker commands generated without launching experiments.
- V7 manuscript: 94 existing tests, all artifact generators, source/PDF/text
  bindings and LaTeX build passed (9 main pages, 29 total). All 271 submitted
  V6 files match the final archive baseline `3d144ee`; V7 TeX/figure sources
  retain that manuscript without new scientific changes.

Test environment: Python 3.12, NumPy 2.5.2, PyTorch 2.14.0+cpu,
Transformers 4.57.6, PEFT 0.21.0, math-verify 0.9.0. Dependencies were installed
in a temporary isolated environment. No pretrained model weights were downloaded.

Not run: remote-node execution or the four-GPU OLMo-3 7B experiments. The five real input bundles are
not present. Their exact prompt formatting, cached rewards, verifier extraction
settings and online ranking-validation IDs must be supplied before execution.
No new experimental result or measured 7B runtime is claimed. Cross-node
filesystem locking and hardware peak memory require cluster-level validation;
local multi-process tests are not a substitute for that validation.
