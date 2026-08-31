# End-to-End RLVR Audit Result

Date: 2026-08-31 KST

## Revisions and verdict

- Running three-node primary source: `295dfea`. Its `v1` exact-answer artifacts
  remain attributed to that revision and are not migrated or relabelled.
- Pre-audit checklist revision: `6873002`.
- Audited implementation revision: `e54077fa50ffaafba70c2ec93914a83c7efb2770`.
- Single-launcher consolidation revision: `c8095c8`.
- GPU-free snapshot preflight revision: `756a09f`.
- Canonical harvest re-audit and fix revision:
  `8812a52ac7e098e1109b078c9fa5d5e159f50cf9`.
- Multi-cluster admission fix revision:
  `fb48565dda10d35aa58452b0b37696ae122558aa`.
- Audited paper revision: `e53fe4ac9437ad702eb558213adad55e3125f83c`.
- Final verdict: the existing primary remains on its compatible `v1` protocol;
  the separate generalization and method protocols are ready after their
  cluster preflights pass. No result is claimed before registered cells finish.
- The retired positive-only LoRA cross-entropy path is SFT, not RLVR. It is
  excluded from canonical launch, harvest, main results, and generalization
  claims. The root cause is recorded in
  `docs/INCIDENT_RLVR_OBJECTIVE_2026-08-31.md`.

## Checklist outcome

| Checklist section | Outcome | Recorded evidence |
|---|---|---|
| 0. Provenance/objective | PASS WITH PRIOR DEFECT | SFT substitution isolated; positive steps require verifier reward, no supervised loss/filter, and a method-bound policy manifest. |
| 1. Policy update | PASS, GPU RUNTIME PENDING | GRPO sign/clipping/std, Dr.GRPO normalizers, RLOO leave-one-out sequence loss, DDP/resume lineage, optimizer and manifest paths inspected and unit tested. Four-H100 execution remains a launch preflight. |
| 2. Data | PASS FOR REGISTERED SETS | Immutable revisions/manifests, explicit loader roots, schema/dedupe/split checks, prompt set/order hashes, and runtime qualification added. APPS remains evaluator-only and is excluded from the registered matrix because its existing converted snapshot is not yet immutable-qualified. |
| 3. Verifiers/sandbox | PASS WITH FIXES | Additional-study symbolic equivalence uses pinned Math-Verify; primary remains exact/float compatible. MBPP blocks process-control escapes and FD token forgery. Local MBPP/KK/ARC qualification passed. |
| 4. Rollout/contracts | PASS, GPU RUNTIME PENDING | Token/log-prob boundaries and manifest checks covered by the full suite; prompt identities and method/source hashes are bound. Actual new-model generation is gated by cluster smoke tests. |
| 5. R/A/B independence | PASS | Ranking split R and disjoint reference splits A/B are enforced in contracts and paper text; exact set/order checks and independent score artifacts remain mandatory. |
| 6. Estimators/selection | PASS | Four estimator cells, clipping/ESS, LOO advantages, ranking baselines, stable top-k, and selection contracts were rechecked against executable tests and manuscript equations. |
| 7. Utility/statistics | PASS | Utility remains validation-gradient alignment, not downstream reward. Reliability/fresh-gain gates and 10,000-replicate hierarchical bootstrap remain unchanged and are not overclaimed. |
| 8. Distributed/publication | PASS WITH FIX, CLUSTER RUNTIME PENDING | Exactly four H100s, clean Git state, shared-volume location, family/collection locks, retries, deep validation, and content-addressed outputs fail closed. The final harvester now carries those checks through publication. No four-H100 node is attached to this audit host. |
| 9. Generalization | REGISTERED | Two independent 7B model families, four verifier domains, three seeds, four checkpoints, full GRPO matrix, and math/code Dr.GRPO plus RLOO slice use disjoint roots/contracts. |
| 10. Mathematics/claims | PASS | Estimator identities, counterexamples, normalization lemma, recovery proposition, reliability ceiling, certification bound, and claim scopes were checked against code/tests; generalization is explicitly non-universal. |
| 11. References/build | PASS | 65 citation keys resolve to 65 unique bibliography entries with zero missing, duplicate, or uncited entries. Clean 18-page letter PDF has no undefined refs/citations, LaTeX errors, or overfull boxes. |
| 12. Regression | PASS WITH RESIDUALS | Exact command evidence and unresolved runtime/statistical risks are below. |

## Defects and corrections

1. **SFT substituted for RLVR.** Parameter-efficient LoRA was confused with the
   optimization objective. Canonical training now samples online verifier
   rewards and validates GRPO-family manifests; old SFT artifacts cannot enter
   RLVR results.
2. **Mathematical protocol separation.** Exact string/float matching can reject
   equivalent symbolic answers, but changing it in place breaks primary
   compatibility. The primary explicitly retains exact/float matching;
   Math-Verify 0.9.0 is enabled only by the additional launcher in disjoint roots.
3. **MBPP reward bypass.** Appended assertions could be skipped by
   `SystemExit(0)`. An initial completion pipe was also forgeable through an
   inherited file descriptor. The final harness separates trusted tests,
   rejects interpreter/process-control escape APIs, restores builtins, catches
   `BaseException`, and includes both attacks as regressions.
4. **Dataset shadowing/provenance.** An obsolete flat data file could precede a
   pinned snapshot, and primary fetches were not uniformly revision-bound.
   Explicit qualified roots now win, and manifests bind repository revision,
   artifact hash, row count, and normalized prompt split hashes.
5. **Narrow empirical coverage.** A Qwen/math/GRPO-only matrix could not support
   broad generalization. The registered extension adds Mistral and OLMo 2,
   arithmetic/code/logic/science verifiers, and distinct GRPO, Dr.GRPO, and
   sequence-level RLOO objectives without pooling unlike methods.
6. **Fragmented additional launch.** Model/domain and method checks are distinct
   generalization factors, but separate public launch/provision wrappers made
   them look like unrelated objectives. `scripts/run_additional_experiments.sh`
   now prepares, qualifies, queues, and runs the entire registered extension;
   `--prepare` performs all dependency work without requiring an H100.
7. **Final harvest trusted file presence.** `harvest_results.sh` previously
   required five nonempty analysis files but did not verify the report-cache
   hashes, registered matrix dimensions, 27B/7B role assignment, final bootstrap
   status, or JSON/CSV agreement. A stale or internally inconsistent report set
   could therefore be packaged after the stronger matrix checks had run. The
   v2 harvester reads a coherent hash-bound snapshot, validates every registered
   cell and selector, rejects provisional/non-finite/duplicate output, checks
   summary and CSV agreement, serializes concurrent publishers, preserves the
   prior bundle on failure, and enforces the exact four-file published layout.
8. **Shared launcher lock rejected independent clusters.** The canonical
   launcher placed a hostname-derived node-admission lock on `GROUP_VOLUME`.
   Independent cloned clusters with the same hostname therefore contended on
   one lock: the first entered the family queue and the other two exited. Node
   admission now uses a fixed lock in node-local `/tmp`; random worker IDs keep
   shared logs distinct, while only family and collection locks remain shared.

## Verification evidence

Executed after the final fixes:

```text
python -m pytest -q
78 passed in 11.44s

python -m pytest -q tests/test_generalization_launcher.py \
  tests/test_model_matrix.py tests/test_grpo_policy.py \
  tests/test_regime_contract.py tests/test_rlvr_launcher.py
24 passed in 2.11s

python -m compileall -q src tests
ruff check --select E9,F63,F7,F82 src tests
bash -n scripts/*.sh
jq empty configs/*.json
git diff --check
All passed.
```

Additional runtime/data evidence:

- MBPP FD-forgery and `SystemExit(0)` attacks both receive reward `0.0`.
- All 974 published MBPP reference programs pass the new static admission
  rules; three sampled references pass their real hidden assertions in
  bubblewrap.
- Local immutable snapshots qualified: MBPP 974 rows, Knights-and-Knaves 6,900
  rows, ARC-Challenge 1,418 rows. Train/validation duplicates and overlap are
  zero, and set/order hashes were recorded by the qualifier.
- Paper: `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` passed;
  `main.pdf` is 18 pages, US Letter, 306,167 bytes.
- Citation audit: 65 cited keys, 65 bibliography keys, zero missing, duplicate,
  or uncited keys.

### Harvest re-audit on 2026-09-01

Executed against fix revision `8812a52`:

```text
python -m pytest -q
83 passed in 12.14s

python -m pytest -q tests/test_harvest_results.py
7 passed in 0.89s

python -m compileall -q src tests
ruff check --select E9,F63,F7,F82 src tests
bash -n scripts/*.sh
jq empty configs/*.json
python -m pip check
git diff --check
All passed.
```

The harvest fixture covers a valid full 27B/7B matrix, unchanged reuse, removal
of unmanifested files, three simultaneous harvesters, stale analysis hashes,
JSON/CSV disagreement, a missing registered cell, swapped primary/replication
roots, and preservation of the last valid bundle after rejection. The running
source revision `295dfea` was checked directly and emits the v2 regime schema,
five selectors, 10,000-replicate final rows, and report-cache marker format
accepted by the new validator.

### Multi-cluster admission re-audit on 2026-09-01

Executed against fix revision `fb48565`:

```text
python -m pytest -q
86 passed in 13.43s

launcher + queue + additional + harvest integration
17 passed in 10.55s

same-hostname three-worker launcher + shared family queue
2 passed, repeated 5/5 times

python -m compileall -q src tests
ruff check --select E9,F63,F7,F82 src tests
bash -n scripts/*.sh
jq empty configs/*.json
python -m pip check
git diff --check
All passed.
```

The regression starts three launchers reporting the identical hostname but
using independent node-local filesystems; all three enter both registered
matrix phases. The shared queue regression records worker identity and requires
all three workers to claim work, every dataset/seed/drift point exactly once,
one transient hung point to be retried, and aggregate analysis to publish once.
A separate test requires a second process on the same physical node to fail and
rejects any `OM_LOCAL_LOCK_DIR` placed below `GROUP_VOLUME`.

## Launch boundaries

The audit host has one 6 GB RTX 3050, not four H100s, so it must not start or
pretend to validate a production training cell. Do not modify the checkout that
is running `295dfea`. From a separate clean checkout on each four-H100 node, run:

```bash
bash scripts/run_additional_experiments.sh
```

The script waits for the primary node lock and idle GPUs before it touches
snapshots or starts work. The shared `flock` family queue assigns each
seed/dataset family once. Every family consumes all four GPUs on its node and
preserves the ordered `0 -> 25 -> 100 -> 400` checkpoint lineage.

## Residual risks

- Mistral/OLMo weight download integrity, tokenizer compatibility, LoRA target
  availability, memory fit, and real generation are checked by provisioning
  and per-node smoke tests but were not executable on this host.
- Failure recovery and shared locking are covered by fixture tests; actual
  preemption, filesystem latency, and three-node scheduling remain operational
  observations to record during the run.
- This host has no primary cluster result volume, so the new harvester was
  exercised with a structurally complete fixture rather than the in-progress
  H100 artifacts. Run it after those jobs finish and the checkout is updated;
  rejection leaves the last valid readout unchanged and does not rerun training.
- New statistical outcomes and cross-stratum generalization are unknown until
  every registered cell and reliability gate completes. A pooled or universal
  claim remains prohibited.
- APPS is not part of the registered extension. Adding it requires an immutable
  source revision, manifest, full qualifier, and a separately versioned
  protocol rather than silently widening this matrix.
