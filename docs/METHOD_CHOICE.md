# Method Choice And Measurement Checks

September 11, 2026. Exploratory extension after the main matrix; not a new
registration of the existing results. The reduced E5, its output root, and
all previously hashed scientific files are unchanged.

## Commands

Use the existing `offpolicy-misranking` checkout, not a new clone or v2 worker.

```bash
bash scripts/run_measurement_audit.sh
bash scripts/run_measurement_audit.sh --calibrate
bash scripts/run_method_choice.sh plan
bash scripts/run_method_choice.sh prepare
bash scripts/run_method_choice.sh
bash scripts/run_method_choice.sh status
bash scripts/run_method_choice.sh summarize
```

The first two commands are CPU-only and do not acquire a node lock or source
the cluster setup script. Reports go to `$OM_WORK/exports/measurement-audit-*`.
`--run /path/to/completed/point` adds a read-only current-artifact audit; no
fallback to parked pre-rescore artifacts is allowed. Each report has its own
directory; existing reports are not overwritten. The default calibration is
20 independently generated datasets per scenario and 200 inner draws. It is
a smoke test, not publication evidence for nominal coverage. Both the
population-cosine and expected finite-budget targets are evaluated. This uses
the production half-score bootstrap, not the full ranking/label procedure.

`plan`, `prepare`, `status`, and `summarize` do not load a GPU model. The plan
prints the exact budget without preparing inputs. Prepare requires all five
completed MATH d100 points and the existing MATH-train pool fetched by
`bash scripts/fetch_math_train.sh`. It freezes a 500-question test set excluding
the candidate and ranking-validation questions of every source seed, then
freezes all five choices and subsets before any training or test evaluation.
Existing downstream outcomes cannot be imported as prospective choices.

The bare method-choice command **does launch GPU work** on an allocated
four-GPU node. Each seed has the original seven subset arms plus a full-pool
control, 200 additional GRPO updates, and 32 responses per test question.
This is 40 trained arms / 8,000 updates plus 45 policy evaluations, including
five shared baselines. It is not the reduced E5 and is not a short CPU check.
Multiple nodes may use the same command; per-seed/arm leases skip busy work.
Failures retain checkpoints and continue to another arm. A repeated launch
cleans previous processes for this suite on this node only; it does not stop
OLMo, Qwen, or reduced E5. Admission still requires free allocated GPUs.

Inputs use `OM_OLMO3_ROOT`, `OM_WORK`, and `DATASETS_DIR` as in the existing
launchers. Output defaults to `$OM_WORK/runs/method-choice-v1`, overridable
with `METHOD_CHOICE_ROOT`. Do not point it at reduced E5 or a source run.
Once frozen, source hashes, choices, test inputs, subsets, and scientific code
are checked before resumption. A conflicting input is an error, not a reason
to erase the original experiment. Interrupted preparation can resume from its
bound intent before training starts.

## Question And Controls

Both criteria choose among g00/g10/g01/g11. One maximizes overlap with fresh R;
the other maximizes mean independent A/B cosine above the pool mean. They
share a seeded hash tie rule and record every tied maximum. A/B has now been
used for method choice; its winning score is not independent outcome evidence.
Choosing by overlap is our comparator, not a claim that all prior methods do so.

Every selected arm and full-pool control starts from the same seed-specific
adapter/optimizer and source GRPO settings. Full-pool training samples from
all 400 candidates; matched updates do not guarantee equal exposure to every
candidate or equal token cost. Random selects 40 candidates and is therefore
not a substitute for the full-pool arm. Fresh R and pass-rate selection remain
additional controls. No new response-replay optimizer or selection method is
claimed as a contribution.

## Inference And Cost

The primary contrast is test reward(alignment choice) minus test reward(overlap
choice), reported for each of five seeds. Equal choices contribute exactly zero;
negative differences stay in the report. A missing seed blocks the aggregate
mean rather than silently changing the denominator. Paired-prompt intervals
are conditional on each trained seed; the descriptive t interval over the five
seed means is conditional on the shared test set. Neither an interval crossing
zero nor a large resampling count establishes equivalence or coverage.

No overlap gate is used for this independent downstream endpoint. Historical
registered labels and thresholds are not changed. Independent evaluation does
not remove nonlinear cosine bias; the CPU examples and stored A/B sensitivity
check that distinct issue without asserting measured real-model bias.

`cost.jsonl` records phase starts/finishes, host, seed, arm, return code, elapsed
seconds, and allocated GPU-seconds. Failed attempts count; a hard kill leaves
an unfinished start, which the report identifies instead of treating as zero.
Training includes online generation and verification; test evaluation includes
generation and verification too. These phase totals must not be double-counted
as separate subtasks. The ledger is research-run resource expenditure, not
yet all-in cost for either decision rule.

Historical behavior generation, R/A/B generation, gradient computation, and
verification costs must come from measured, attributable source records.
They cannot be inferred from overlap, response counts alone, or file mtimes.
`comparison.json` explicitly leaves end-to-end cost incomplete until that
accounting exists; it does not claim an efficiency advantage. Report cold-start
and incremental reuse costs separately. Never compare alignment's training
time alone against another method's selection-plus-training cost.

## Plot Contract

- Seed-level paired reward differences, horizontal zero line, including equal
  and negative choices; show seed points, not a bar implying 40 independent runs.
- Reward by chosen method and fresh/random/full-pool controls, same reward axis.
- Reward versus total GPU-hours only with complete method-specific accounting;
  missing cost is an explicit missing panel, not x=0.
- Analytic cosine-budget curve, fixed-set coverage stress check, and real
  stored-gradient sensitivity in separate panels. Labels distinguish exact,
  simulated, and descriptive observations.

Local verification is CPU-only. Four-GPU execution must be checked on the
cluster; no local test or successful paper build substitutes for those results.

## Local Verification

The focused suite passed 88 tests, including the existing first-interval,
reliability-budget, regime-map, downstream resumption, and node-ownership
tests. New tests cover exact counterexamples, deterministic production-core
resampling, all-five-seed preparation, input tampering, full-pool/subset sizes,
test independence, interrupted preparation, equal/negative choices, missing
seed aggregation, busy-arm skipping, failure continuation, and cleanup of a
late-export controller while preserving another experiment's processes.
Ruff and shell syntax checks passed. The shell-launched calibration smoke run
completed on CPU; no cluster training or real-artifact result is claimed.
