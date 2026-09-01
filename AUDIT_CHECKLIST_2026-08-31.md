# End-to-End RLVR Audit Checklist

Date: 2026-08-31 KST

This document freezes the audit scope before inspection. It is updated with
commands, evidence, defects, and final verdicts only after every required check
has run. A checked box means evidence was recorded, not merely that code was
read.

The frozen per-item scope remains below. Final outcomes, commands, defects, and
residual risks are recorded in [AUDIT_RESULT_2026-08-31.md](AUDIT_RESULT_2026-08-31.md).

## Status legend

- `[ ]` pending
- `[x]` verified with recorded evidence
- `[!]` defect found; the final record must link the fix and regression test
- `[~]` intentionally out of scope with a written reason

## 0. Provenance and objective separation

- [ ] Record the code and paper revisions audited.
- [ ] Confirm the three running clusters remain attributed to code revision
  `295dfea`; later audit commits must not be substituted into their manifests.
- [ ] Confirm every positive-step canonical run declares verifier-reward GRPO,
  clipped policy-gradient updates, no supervised loss, and no positive-only
  response filter.
- [ ] Confirm the retired positive-only LoRA SFT runs cannot enter the GRPO
  launcher, aggregate bundle, regime table, or headline claim.
- [ ] Reconstruct and document why the earlier SFT substitution occurred.

## 1. GRPO policy update

- [ ] Trace prompt sampling, `K=8` online response generation, verifier rewards,
  group normalization, and loss construction end to end.
- [ ] Verify the implementation uses population standard deviation plus
  `1e-4`, including all-correct and all-incorrect groups.
- [ ] Verify old-policy token log probabilities are captured at sampling time
  and remain frozen for both optimization epochs.
- [ ] Verify the likelihood ratio sign, token mask, reduction, and `[0.8,1.2]`
  clipping match the documented GRPO loss.
- [ ] Verify gradients come only from the policy-gradient surrogate and not
  reference answers, teacher forcing, or a hidden cross-entropy term.
- [ ] Verify four-rank DDP averaging, accumulation, optimizer stepping,
  gradient clipping, and zeroing semantics.
- [ ] Verify LoRA targets, trainable parameter counts, dtype, optimizer state,
  and checkpoint restoration for every supported model family.
- [ ] Verify the continuous `0 -> 25 -> 100 -> 400` lineage and recovery after
  interruption without repeating or skipping optimizer steps.

## 2. Dataset acquisition and preprocessing

- [ ] Audit immutable source repository/revision/hash contracts for every
  dataset used or proposed.
- [ ] Audit all loaders: GSM8K, MATH-500, DAPO-Math-17K, MBPP, APPS, Knights and
  Knaves, and ARC-Challenge.
- [ ] Verify schema fallbacks cannot silently select the wrong field or dataset.
- [ ] Verify exact duplicate removal, conflicting-label rejection, and empty or
  corrupt local-copy fallback behavior.
- [ ] Verify deterministic seeded splits, exact train/validation counts, and
  train/validation identity disjointness.
- [ ] Verify benchmark-test leakage rules and document datasets that may be used
  only for evaluation rather than training selection.
- [ ] Verify prompt templates preserve the task and do not expose hidden tests
  or gold answers beyond intended public examples.
- [ ] Verify answer normalization before and after generation for numeric,
  symbolic, multiple-choice, structured-logic, function-code, and stdin/stdout
  tasks.

## 3. Verifiers and sandboxing

- [ ] Test correct, incorrect, malformed, adversarial, empty, and multiple-answer
  outputs for every verifier branch.
- [ ] Test numeric formatting, commas, signs, fractions, equivalent symbolic
  forms, boxed answers, and accidental substring matches.
- [ ] Test ARC and Knights-and-Knaves structured output against label/name
  ambiguities and partial answers.
- [ ] Test MBPP and APPS extraction, syntax failure, timeout, memory/process
  limits, nondeterminism, stdout normalization, and hidden-test isolation.
- [ ] Confirm verifier exceptions fail closed and are visible in artifacts.

## 4. Rollout generation and contracts

- [ ] Verify tokenizer chat templates, generation prompt boundaries, EOS/pad
  handling, maximum token count, temperature, top-p, top-k, repetition penalty,
  and deterministic seed derivation.
- [ ] Verify generated token IDs align exactly with stored token log
  probabilities and masks, including truncation and EOS.
- [ ] Verify behavior rollouts are generated once at step zero and reused only
  inside the same seed/dataset/model family.
- [ ] Verify current-policy rollouts are regenerated at every positive step and
  cannot be reused across policy checkpoints.
- [ ] Verify rollout manifests bind model/tokenizer hashes, policy adapter hash,
  generation settings, prompt IDs, sample IDs, and source revision.
- [ ] Verify interrupted generation resumes without duplicate or missing samples.
- [ ] Verify exact candidate coverage and fail-closed handling of partial,
  corrupt, stale, or incompatible artifacts.

## 5. R/A/B independence and gradient features

- [ ] Verify 32 current-policy rollouts form eight four-rollout micro-groups and
  are assigned once as R=4 groups, A=2 groups, B=2 groups.
- [ ] Verify 100 validation prompts split deterministically and disjointly as
  R=50, A=25, B=25.
- [ ] Verify ranking inputs never appear in A/B evaluation references.
- [ ] Verify independent tie streams and sample identifiers prevent shared-noise
  agreement.
- [ ] Verify gradients cover the final four decoder layers plus final
  normalization and use one fixed 4096-dimensional CountSketch.
- [ ] Verify projection seeds, parameter ordering, dtype, serialization shape,
  and distributed aggregation are stable across resume and model family.

## 6. Off-policy estimators and selection

- [ ] Derive and match `g00`, `g10`, `g01`, and `g11` token by token, including
  the current-token, prefix, and suffix ratio boundaries.
- [ ] Verify each completed ratio product is clipped once to `[0.1,10]` and that
  ESS is computed from the intended unnormalized or clipped weights.
- [ ] Verify leave-one-out advantages, zero-variance groups, masks, and response
  length normalization.
- [ ] Verify `g11` equals the intended full-trajectory change of measure on
  exhaustive toy trajectories.
- [ ] Verify behavior-diversity, expected-uniform, split-R, and random baselines
  match their paper definitions and compute budgets.
- [ ] Verify top-k size, stable ordering, independent ties, subset filters, and
  the minimum-20-prompt label rule.

## 7. Utility, uncertainty, and regime labels

- [ ] Verify held-out utility is validation-gradient alignment, not realized
  downstream reward improvement.
- [ ] Verify A/B averaging, A-versus-B reliability, selection gain, retained
  fraction, regret, and boundary margin formulas.
- [ ] Verify the reliability gate uses the one-sided 95% lower confidence bound
  and the fixed `2k/n` threshold.
- [ ] Verify the fresh-gain gate, three-way label logic, and inconclusive paths.
- [ ] Verify 10,000 hierarchical bootstrap replicates, resampling units, seed
  aggregation, endpoint handling, and deterministic bootstrap seeds.
- [ ] Verify threshold grids are descriptive and no nested-sample budget curve
  is presented as independent evidence.

## 8. Distributed execution and artifact publication

- [ ] Verify four-H100 fail-fast checks and that CPU-only validation does not
  allocate or interfere with running GPUs.
- [!] The OLMo live launch repeatedly aborted despite component fixture passes;
  see `docs/INCIDENT_MULTI_CLUSTER_LOCK_2026-09-01.md`. Add one composition test
  that executes the actual launcher, worktree supervisor, matrix supervisor,
  point entry point, cleanup, and partial resume without replacing an internal
  supervisor boundary with a fake script.
- [ ] On the target cluster, verify preflight, rollout, CPU verifier, checkpoint
  transition, retry, queue wait, and cleanup under utilization-based GPU
  reclamation. Record per-GPU process/utilization evidence and prove exactly one
  keepalive owner per GPU.
- [ ] Test clean, missing, locked, invalid-directory, dirty, and wrong-HEAD
  pipeline worktree cache states. Verify repair quarantines cache bytes, closes
  inherited lock descriptors, preserves experiment artifacts, and resumes the
  immutable generation commit after a supervisor pull.
- [ ] Verify all nodes require the same physical `GROUP_VOLUME` and family locks
  prevent duplicate work while preserving checkpoint order.
- [ ] Test stale locks, process crashes, preemption, retry limits, and quarantine
  behavior without deleting valid work.
- [ ] Verify `DONE` is published only after every policy, rollout, score, oracle,
  coverage, and schema contract passes.
- [ ] Verify analysis-only migration cannot alter training provenance.
- [ ] Verify harvest is separately locked, refuses partial matrices, and emits
  exactly `REPORT.md`, `RESULTS.json`, `RESULTS.csv`, and `MANIFEST.sha256`.
- [ ] Verify the final manifest covers every byte and unchanged inputs reproduce
  the same bundle.

## 9. Generalization matrix

- [ ] Survey recent primary RLVR papers for model families, parameter scales,
  training objectives, verifier types, datasets, selection signals, and
  out-of-domain evaluations.
- [ ] Separate training-data diversity from evaluation-only benchmark breadth.
- [ ] Cover at least mathematical exact answer, executable code, symbolic or
  structured logic, and scientific multiple choice with distinct verifiers.
- [ ] Cover at least two independent base-model families in addition to the
  Qwen scale comparison, with immutable model revisions.
- [ ] Compare a minimal set of materially different RLVR update families rather
  than renaming minor GRPO variants.
- [ ] Prespecify a compute-feasible factorial or fractional-factorial design and
  prevent optional results from contaminating the running primary matrix.

## 10. Paper mathematics and claims

- [ ] Re-derive the 2-by-2 estimator decomposition and both difference identities.
- [ ] Exhaustively enumerate the two-token counterexamples for `c=10` and
  `c=01`, including KL, ESS, indistinguishability, and ranking reversal.
- [ ] Verify the binary group-normalization lemma with and without the
  denominator stabilizer.
- [ ] Verify the top-k margin recovery proposition, split-half Gaussian ceiling,
  Spearman-Brown mapping, and exact-certification lower bound.
- [ ] Check every empirical statement against executable code and distinguish
  selection-proxy utility from downstream improvement.
- [ ] Check theorem scope, assumptions, quantifiers, nontrivial-certificate
  definition, and limitations for overclaiming.

## 11. References and paper build

- [ ] Resolve every citation key to exactly one bibliography item and detect
  uncited, duplicated, or missing entries.
- [ ] Verify every title, author list, year, venue status, arXiv identifier or
  DOI, and primary-source URL.
- [ ] Verify dataset and method descriptions against the cited primary papers.
- [ ] Force a clean LaTeX rebuild and check undefined references/citations,
  overfull boxes, page size, page count, and anonymization.
- [ ] Render and inspect the title, framework figure, empirical protocol,
  main-text page boundary, proofs, validity audit, tables, and final page.

## 12. Final regression record

- [ ] Run the complete CPU test suite from a clean environment and record exact
  pass/fail/skip counts and runtime.
- [ ] Classify every verification result as unit, simulated integration, local
  real-model, or target-cluster runtime. Never report fixture pass counts as
  proof of H100 scheduling or shared-volume launch readiness.
- [ ] Require target-cluster evidence for the exact canonical command before
  declaring a GPU experiment launcher ready. If cluster access is unavailable,
  record the launch gate as unverified rather than inferring success.
- [ ] Run syntax/import/static checks for every Python and shell entry point.
- [ ] Record every defect, root cause, affected artifacts, fix commit, and
  regression test.
- [ ] Re-run all relevant checks after fixes.
- [ ] Record residual GPU-only, model-download, cluster, and statistical risks.
- [ ] Commit and push the completed audit record without rewriting the source
  revision stored by already-running jobs.
