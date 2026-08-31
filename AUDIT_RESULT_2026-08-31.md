# End-to-End RLVR Audit Result

Date: 2026-08-31 KST

## Revisions and verdict

- Running three-node primary source: `295dfea`. Its `v1` exact-answer artifacts
  remain attributed to that revision and are not migrated or relabelled.
- Pre-audit checklist revision: `6873002`.
- Audited implementation revision: `e54077fa50ffaafba70c2ec93914a83c7efb2770`.
- Audited paper revision: `e53fe4ac9437ad702eb558213adad55e3125f83c`.
- Final verdict: ready to launch the new `v2-mathverify`, generalization, and
  method-robustness protocols after their cluster preflights pass. No empirical
  result is claimed for those new matrices before all registered cells finish.
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
| 3. Verifiers/sandbox | PASS WITH FIXES | Symbolic math equivalence uses pinned Math-Verify. MBPP catches `SystemExit` and blocks process-control/dynamic-introspection escapes; inherited-FD token forgery regression added. Local MBPP/KK/ARC reward qualification passed. |
| 4. Rollout/contracts | PASS, GPU RUNTIME PENDING | Token/log-prob boundaries and manifest checks covered by the full suite; prompt identities and method/source hashes are bound. Actual new-model generation is gated by cluster smoke tests. |
| 5. R/A/B independence | PASS | Ranking split R and disjoint reference splits A/B are enforced in contracts and paper text; exact set/order checks and independent score artifacts remain mandatory. |
| 6. Estimators/selection | PASS | Four estimator cells, clipping/ESS, LOO advantages, ranking baselines, stable top-k, and selection contracts were rechecked against executable tests and manuscript equations. |
| 7. Utility/statistics | PASS | Utility remains validation-gradient alignment, not downstream reward. Reliability/fresh-gain gates and 10,000-replicate hierarchical bootstrap remain unchanged and are not overclaimed. |
| 8. Distributed/publication | PASS, CLUSTER RUNTIME PENDING | Exactly four H100s, clean Git state, shared-volume location, family/collection locks, retries, deep validation, and content-addressed outputs fail closed. No four-H100 node is attached to this audit host. |
| 9. Generalization | REGISTERED | Two independent 7B model families, four verifier domains, three seeds, four checkpoints, full GRPO matrix, and math/code Dr.GRPO plus RLOO slice use disjoint roots/contracts. |
| 10. Mathematics/claims | PASS | Estimator identities, counterexamples, normalization lemma, recovery proposition, reliability ceiling, certification bound, and claim scopes were checked against code/tests; generalization is explicitly non-universal. |
| 11. References/build | PASS | 65 citation keys resolve to 65 unique bibliography entries with zero missing, duplicate, or uncited entries. Clean 18-page letter PDF has no undefined refs/citations, LaTeX errors, or overfull boxes. |
| 12. Regression | PASS WITH RESIDUALS | Exact command evidence and unresolved runtime/statistical risks are below. |

## Defects and corrections

1. **SFT substituted for RLVR.** Parameter-efficient LoRA was confused with the
   optimization objective. Canonical training now samples online verifier
   rewards and validates GRPO-family manifests; old SFT artifacts cannot enter
   RLVR results.
2. **Mathematical false negatives.** Exact string/float matching rejected
   equivalent symbolic answers. Math-Verify 0.9.0 was pinned, and the changed
   reward protocol uses separate `v2-mathverify` run/readout roots with implicit
   analysis migration disabled.
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
6. **Missing launch provisioning.** The generalization launcher assumed its
   snapshots existed while the old provisioner prepared only the primary Qwen
   study. `scripts/provision_generalization.sh` now downloads pinned model/data
   snapshots and qualifies them before GPU launch.

## Verification evidence

Executed after the final fixes:

```text
python -m pytest -q
76 passed in 10.38s

python -m pytest -q tests/test_generalization_launcher.py \
  tests/test_model_matrix.py tests/test_grpo_policy.py \
  tests/test_regime_contract.py tests/test_rlvr_launcher.py
22 passed in 1.17s

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

## Launch boundaries

The audit host has one 6 GB RTX 3050, not four H100s, so it must not start or
pretend to validate a production training cell. On an online shared-volume
shell, prepare the new snapshots once:

```bash
git pull
bash scripts/provision_generalization.sh
```

Then run the same selected command on each of the three four-H100 nodes:

```bash
bash scripts/run_generalization.sh
# After the full GRPO matrix, for the registered method slice:
bash scripts/run_method_robustness.sh
```

The shared `flock` family queue assigns each seed/dataset family once. Every
family consumes all four GPUs on its node and preserves the ordered
`0 -> 25 -> 100 -> 400` checkpoint lineage. Do not pull the new revision into
the already-running `295dfea` worktrees.

## Residual risks

- Mistral/OLMo weight download integrity, tokenizer compatibility, LoRA target
  availability, memory fit, and real generation are checked by provisioning
  and per-node smoke tests but were not executable on this host.
- Failure recovery and shared locking are covered by fixture tests; actual
  preemption, filesystem latency, and three-node scheduling remain operational
  observations to record during the run.
- New statistical outcomes and cross-stratum generalization are unknown until
  every registered cell and reliability gate completes. A pooled or universal
  claim remains prohibited.
- APPS is not part of the registered extension. Adding it requires an immutable
  source revision, manifest, full qualifier, and a separately versioned
  protocol rather than silently widening this matrix.
