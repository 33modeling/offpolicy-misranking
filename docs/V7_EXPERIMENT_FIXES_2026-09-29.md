# V7 experiment repairs - 2026-09-29

Scope: follow-up implementation for the
[V7 audit](V7_EXPERIMENT_CODE_AUDIT_2026-09-29.md), including existing runtime
regressions. No archived result, frozen runtime, production historical hash
allowlist, manuscript claim or running GPU job was changed.

## Corrected behavior

| Finding | Correction | Regression coverage |
| --- | --- | --- |
| F1: fixed switch was one update late | Charge the checkpoint-N selection refresh, then switch before training N+1; checkpoint protocol `fixed-boundary-before-training-v2` | Boundaries 25/75/100/125/200, resume before/at/after the boundary, matching adaptive Switch at 50, rejection of old fixed checkpoints |
| F2: secondary dataset used the wrong cohort | Both datasets call the same active Pair/prepared-plan resolver | Both dataset orders, Pair and prepared cohorts |
| F3: MBPP early exit could receive reward 1 | Separate candidate/assertion execution; require assertion completion on a dedicated channel and exit 0 | Passing/failing code, SystemExit/sys.exit/os._exit, timeout, exceptions, stdout spoofing, exec shadowing, descriptor cleanup |
| F4: arbitrary fixed controls disappeared from results | Discover saved fixed-N endpoints for planned seeds, retaining identity validation | fixed75/fixed200 JSON and text exports; incompatible endpoint rejection |
| F5: missing On-policy diagnostic D | Reconstruct from finite gradient norms/cosines only when decision D is absent; label its origin | On-policy and fixed-control export, zero norm, missing/nonfinite terms, recorded decision and raw-history preservation |

MBPP cache receipts and exported provenance include
`code_verifier_version=assertion-completion-v2`. A cache identified as produced
by `srgc_rebuttal.verifiers:code_reward` without that version cannot be resumed
or used as corrected cached rewards. Bundle-to-bundle CLI imports validate the
source and preserve verifier metadata instead of discarding it. Direct external
reward maps remain an explicitly supplied input without a verifier certification.
Historical Pair caches use a distinct verifier and retain their original
provenance; this repair does not claim to have regraded them.

The completion runner is not an OS security sandbox. Generated code still
requires disposable, credential-free compute, as documented in `DATASETS.md`.
No saved rollout scan has established the frequency of the old false positive.

## Existing runtime tests

The 209 historical failures in the audit were test-contract issues, not 209
demonstrated training bugs. The repair keeps production compatibility guards:

- Historical Switch/Pair/MoPPS migration fixtures now hash the released trainer
  blobs from `c0c38d6`, including MoPPS, consistently in parent and child
  processes. A dedicated test checks every pinned trainer hash against Git.
- Historical Pair migration fixtures use the released launcher identity.
  Legacy WAIT/retry tests execute the actual released shell body; current
  dispatcher/resume tests continue to exercise the current launcher separately.
- The RLOO d100 test restores the report module's separately loaded point list,
  preventing order-dependent corruption of subsequent report tests.
- The simulated Qwen two-node test synchronizes first task execution instead
  of assuming two 25 ms sleeps will overlap on a loaded CPU. Task uniqueness,
  dependencies, runtime binding and two-node execution assertions remain.

## Run separation

The reviewed package digest is
`f581eb043e89409e0e68d8ed77201fa030e34bd93d4d5babbab65304c24a5d6b`,
intentionally different from the pre-repair package. Do not edit old receipts
or hashes to make their queues/checkpoints pass the new identity checks. This
applies to both OLMo and Qwen adapters and to existing fixed-control endpoints.

Preserve the old checkout and output root for any in-flight or archived run.
Use a separate current checkout and a new output root for corrected experiments.
For the OLMo integrated launcher, after source inputs exist, a new active cohort
can be prepared without starting training:

```sh
python scripts/run_srgc_rebuttal.py storage --dataset math --fresh verifier-v2-20260929
python scripts/run_srgc_rebuttal.py storage --dataset mbpp --fresh verifier-v2-20260929
```

These commands change the active-cohort pointer, so use them only for the new
experiment, not while workers are following an old active cohort. Prepared
caches are regenerated; verified historical Pair imports keep their separate
source provenance. Qwen must likewise be prepared in a separate output root.
None of these operational commands was executed against real cluster storage
during this repair.

## Verification

Completed checks:

- Complete historical pytest suite: 4,577 passed, 16 skipped, no failures,
  plus 5 passed subtests (1,108.37 s). Fifteen skips require an explicitly
  selected CUDA device; one is an inapplicable nested RLOO meter case.
- Two newly added release-pin tests passed separately (added after full-suite
  collection): the frozen Git hashes are exact and unrelated files are not
  hidden by the fixture. Total unique historical/pin tests passed: 4,579.
- New experiment suite, Transformers 5: 268 passed, no skips (52.341 s).
- New experiment suite, Transformers 4: 259 passed, 9 Qwen-version skips
  (268 discovered, 48.571 s). Those nine pass in the Transformers 5 suite.
- Historical focused regression suite: 430 passed.
- Current Pair resume/dispatcher plus release-pin checks: 19 passed.
- Manuscript V7 scripts: 128 passed; compute-comparison and SR-GC artifact
  `--check` commands passed. All 271 submitted V6 files remain unchanged.
- AST parsing: 575 Python files; `bash -n`: 110 shell files; Ruff
  `F821,F822,F823` and `git diff --check`: passed.

Tests use CPU/tiny models and simulated workers, not full-size H100 training.
Passing tests does not establish new scientific results or prove absence of all
bugs. Historical experiment numerical outputs were not rerun or relabeled.

Reproduction from the code repository:

```sh
env PYTHONPATH=src:scripts:tests:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .work/.venv-cu126/bin/python -m pytest tests -q --disable-warnings --tb=short
env PYTHONPATH=src:scripts:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/nsh/.venvs/proto-ml/bin/python -m unittest discover -s srgc_rebuttal/tests -v
```

Local evidence: `/tmp/v7-fixed-historical-20260929.{log,xml}`,
`/tmp/v7-fixed-new-final-tf5.log`, `/tmp/v7-fixed-new-final-tf4.log`, and
`/tmp/v7-fixed-paper-tests.log`. These temporary logs are local verification,
not remotely archived experiment outcomes.
