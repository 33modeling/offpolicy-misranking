# Experiment code audit October 4 2026

Follow-up: [runtime repairs and verification](V7_EXPERIMENT_FIXES_2026-10-04.md).
The findings and reproduction evidence below describe the pre-repair code and
are preserved as an audit record, not the current repair status.

Reviewed code: `master` at `d22d533c8ae3698327e088cbd0582bbfff1520a3`.
Four defects were independently reproduced. This is a review record, not a
repair or certification of archived experiment results. Training code, frozen
runtimes, inputs, results, manuscript claims and cluster jobs are unchanged.
Preexisting untracked research and hotfix files were left untouched.

## Findings

| ID | Priority | Affected path | Reproduced behavior |
| --- | --- | --- | --- |
| F1 | P1 | Shared SRGC process guard | Deletes a live unrelated job's shared-memory file |
| F2 | P1 | Base OLMo checkpoint wrapper | Accepts a different attention kernel on resume and overwrites the saved policy |
| F4 | P1 | OLMo and Qwen MBPP verifier | Candidate code can forge assertion completion and receive reward 1 without passing tests |
| F3 | P2 | Base OLMo and Qwen task queue | User interrupts consume the retry budget and eventually prevent resumption |

P1 means a correctness or cross-job safety issue to address before affected
new runs. P2 means an operational defect that can block otherwise valid work.
The reproductions establish conditions under which the code is wrong; they do
not establish that existing MATH/MBPP results encountered those conditions.

### F1 Shared memory cleanup can delete another live job's storage

Location: `scripts/srgc_process_guard.py:292-305`, invoked at `:317` before
waiting for free GPUs.

`clean_shm` checks whether a recognized SRGC child exists, then removes every
same-UID `torch_*` or `nccl-*` entry. A live PyTorch job outside the recognized
SRGC launchers does not prevent deletion. There is no check that the entry
belongs to this experiment or is no longer in use. GPU leases do not establish
ownership of shared-memory files.

The reproduction keeps an audit-owned `torch_*` file open, supplies an unrelated
live Python process in the process table, and invokes cleanup on a temporary
directory. The pathname disappears although the open descriptor remains
readable. Existing mappings can survive an unlink; subsequent attachments can
fail. This is not evidence that every running process immediately crashes.

Correction needed: delete only provably owned stale segments. Preserve unknown
or still-open entries, including those belonging to non-SRGC jobs. Add coverage
for a live foreign job, not just live SRGC children and matching filename/UID.

### F2 Base OLMo resumes under the current attention setting

Locations: `scripts/srgc_step_checkpoints.py:51-52`, `:97`, `:107-116`;
`srgc_rebuttal/srgc.py:417`.

The wrapper applies the current `SRGC_ATTENTION` setting before loading saved
state. The engine validates its scientific config but does not validate
`checkpoint_policy.attention`. The next save replaces that policy with the
current setting. An SDPA checkpoint is therefore accepted under eager attention
without rejection or preservation of its original setting. Kernel choice can
affect numerical behavior and timing while the experiment identity is unchanged.

The toy-backend reproduction restores step 1 with saved attention `sdpa` into
the real checkpoint wrapper configured as `eager`; the next state reports
`eager`. It exercises metadata validation, not an empirical comparison of GPU
kernel outputs.

This finding is specific to the base OLMo wrapper. Extra arms already restore
attention through `extra_checkpoint_policy` in `srgc_sr_refresh.py`; Qwen pins
its attention setting. Those paths should not be described as sharing F2.

Correction needed: restore or reject the saved attention policy before model
construction, including continuation from the shared prefix. Define an explicit
legacy default for old checkpoints without the field and test both resume paths.

### F3 User interruption counts as a failed attempt

Location: `srgc_rebuttal/cluster_queue.py:187-199`, `:210-223`.

Each claim increments `attempt`. `finish(..., interrupted=True)` records the
interruption but does not distinguish it from failures when applying the attempt
limit. Three claim/interrupt cycles with `max_attempts=3` leave an otherwise
resumable prefix in `attempts_exhausted`, even with `retry_failed=True`. Its
dependent arms cannot start. The resume command only removes the stop marker;
it does not repair this budget.

The Qwen default is three attempts. The standard OLMo shell launcher overrides
the limit to 50, so it has the same mechanism but not the same three-interrupt
threshold. The extra-arm worker already refunds interrupted attempts and is not
affected by this specific defect.

Correction needed: separate failed-attempt accounting from launch history, or
refund intentional interruptions without erasing attempt receipts. Cover repeated
interrupt/resume and prefix dependency release in both base and Qwen workers.

### F4 MBPP candidates can forge the completion signal

Locations: `srgc_rebuttal/verifiers.py:68-78`,
`srgc_rebuttal/code_check.py:14-31`.

The prior early-exit repair rejects ordinary `SystemExit(0)` and `os._exit(0)`.
However, the completion descriptor and secret are passed in the same process's
`sys.argv`, which generated code can read. A candidate can write the secret to
that descriptor and exit zero before any assertion executes. Against the real
verifier with `answer="assert False"`, ordinary `pass` receives 0, while the
audit candidate receives 1.

The trigger requires candidate code to use these runtime internals. No scan of
real rollout responses was performed, so the frequency and effect on published
results are unknown. This concerns reward correctness, not merely the already
documented absence of a security sandbox. It affects caches, training rewards
and evaluation that use this verifier, including the Qwen adaptation.

Correction needed: separate the untrusted candidate from the authority that
attests successful tests. Merely removing the token from `sys.argv` is not a
complete Python isolation boundary. Test forged completion, early termination,
normal pass/fail, timeouts and descriptor cleanup. Preserve old verifier
provenance and use a new version for any corrected rewards.

## Reproduction

Run from the repository root with the experiment's Python environment:

```sh
python docs/audit_evidence/2026-10-04/reproduce_srgc.py
```

The script uses the existing toy backend and synthetic queue fixtures. Every
write stays in a temporary directory. It does not touch `/dev/shm`, real queues,
GPUs or unrelated processes. Exit zero means the four reviewed defects were
observed, not that runtime correctness tests passed. After repairs, convert the
reproductions into regression tests asserting the intended behavior.

## Verification scope

Execution guides were read before code review. Coverage includes the base
Random/SR/On-policy/Switch engine, gradient and rollout backend, cache creation,
shared prefix, checkpointing, queue leases, additional/fixed/replicate/support
arms, Qwen adaptation, reward verifiers, cost ledgers and result exports.
Historical experiment tests were also discovered and run; this is not a claim
of a manual line-by-line review of every historical implementation.

Static checks passed for 437 tracked Python/shell files selected from `scripts`,
`src` and `srgc_rebuttal`: 326 Python compilation checks and 111 shell syntax
checks. This selection includes 48 files under `srgc_rebuttal/tests`.
Ruff checks for undefined names, referenced-before-assignment locals and duplicate
definitions (`F821,F823,F811`) passed. Passing static checks does not resolve F1-F4.

Test environment: CPU-only PyTorch 2.13.0, Transformers 5.14.1, PEFT 0.20.0,
math-verify 0.9.0. Supplemental test dependencies were installed under `/tmp`,
not in the experiment environment. No full-size H100 training, remote shared
filesystem behavior, intermittent NCCL failures or production FLA kernels were
verified. Existing experiment outcomes were not regenerated or relabeled.

### Completed targeted checks

- New SRGC suite after dependency setup: **443 passed, no skips**, plus 266
  passed subtests, in 101.86 seconds. Real tiny OLMo/Qwen models exercise
  generation, dense scoring gradients, updates and checkpoint restoration.
  Queue tests use simulated nodes, not a remote GPU cluster.
- Separate Qwen extension/hardening run: **33 passed**, plus two passed
  subtests, in 21.25 seconds. These tests overlap the 443 above and must not
  be added to that unique-test count.
- Net-gain gate and GPU orchestration tests: **53 passed** after installing
  the declared gate dependencies in the temporary test path. The initial
  13 failures were missing-dependency errors, not demonstrated training bugs.
- The first new-suite run had nine Qwen skips while temporary dependencies
  were incomplete. That run is superseded by the complete 443-test rerun,
  not presented as evidence that Qwen was tested before setup finished.
- The four independent reproductions match
  [observed.json](audit_evidence/2026-10-04/observed.json). They expose behavior
  not rejected by the existing regression suite and remain unfixed.

### Full workspace run and audit interference

Discovery collected 5,038 top-level tests from `tests` and
`srgc_rebuttal/tests`, including 16 tests in the preexisting untracked CFCS
test file. That file was inspected but is not included in this audit commit.

The first combined run reported 3,945 passed, 23 failed, 1,055 errors and 15
skipped, plus 271 passed subtests, in 976.58 seconds. It is not a clean full-suite
pass. All 23 failures were missing `scikit-learn`, including two shell-child
failures. The other errors resulted from the audit's own attempt to interrupt
the long run: an installed SIGINT handler raised `SystemExit(130)` during
`tmp_path` setup, followed by 1,054 pytest fixture-finalizer assertion errors.
The audit watchdog was suspended so pytest could finish and write its JUnit
record, then resumed and reaped. These setup errors are not 1,055 independent
experiment defects.

Every failed/error node was selected from that JUnit record for a fresh-process
rerun with completed dependencies, without changing tests or runtime code.
The rerun completed with **1,077 passed and one skipped** in 291.25 seconds.
The skip is an inapplicable nested-curve-meter case for RLOO, not a failed
training case hidden by the rerun.

Deduplicating the initial run and this rerun gives **5,022 passed, 16 skipped,
no remaining failed/error test IDs** out of 5,038 collected top-level tests.
This is aggregate coverage across runs, not an uninterrupted green full-suite
run. Of these, 16 passing CFCS tests belong to the preexisting untracked file;
the tracked-only totals are 5,006 passed and 16 skipped. Passed subtests are
reported separately and are not added to these unique top-level counts.
Fifteen skips require an explicit `SWITCH_TEST_CUDA` device and were not waived.

Local verification records:

- `/tmp/srgc-audit-20261004-full.{log,xml}`
- `/tmp/srgc-audit-20261004-retry.{log,xml}`
- `/tmp/srgc-audit-20261004-retry-selection.json`
- `/tmp/srgc-audit-20261004-new-fixed-env.{log,xml}`
- `/tmp/srgc-audit-20261004-qwen.{log,xml}`
- `/tmp/srgc-audit-20261004-gate-fixed-env.{log,xml}`

For a new uninterrupted run, use an environment containing the experiment and
gate dependencies, including the appropriate Transformers version for Qwen:

```sh
env PYTHONPATH=src:scripts:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m pytest tests srgc_rebuttal/tests -q --tb=short
```

The audited `srgc_rebuttal/*.py` digest remains
`9435e80003f41e1f65cb9dcc4074f1b06f063880d4823f433d5cd8fb5e2d74a7`.
Only this audit record and its two reproduction artifacts are published.
