# MBPP Five-Branch Follow-Up

The original matched/convergence MBPP run contains 37 complete branches,
five unavailable canonical results and six gated branches waiting on development
data. Exit 80 reports this blocked original run; it is not a completion result.

The authorized follow-up uses a separate `selection-switch-mbpp-quality-repair-v1`
root. It preserves the original root and its costs. It copies the 37 completed
results and their metadata, retains certified prefix checkpoints, and executes
only these five original-condition retries:

| Seed | Switch Step | Branch |
| --- | --- | --- |
| 2 | 50 | selection_reduced |
| 3 | 25 | selection_full |
| 4 | 25 | random_full |
| 4 | 50 | selection_reduced |
| 4 | 100 | random_reduced |

The recovered development result enables fitting the gate and running the six
held-out gated branches. The original per-attempt budgets, decisions, evaluation
sets, selected prefixes and learner are unchanged. This is a **separate retry**,
not resumed work inside the exhausted original allocation. Failed original
attempts remain in `original-attempts/`; their cost is never refunded or reported
as zero. New and old costs remain separate in the results export.

Run on an idle four-GPU allocation, from the updated repository:

```bash
bash scripts/run_mbpp_repair.sh
```

Read-only monitoring and a single bounded TXT export:

```bash
bash scripts/run_mbpp_repair.sh status
bash scripts/run_mbpp_repair.sh results
```

Preparation is CPU-only, serialized and atomic. It rejects changed source
evidence, a live original lease, unexpected completed retries and overlapping
output roots. Repeated launches preserve new retry progress. Every worker checks
the fixed branch allowlist and copied evidence; reused branches cannot train.
The original controller and other GPU processes are never stopped by preparation.
GPU execution must be verified on the experiment server; local CPU regression
tests do not establish that a remote training job has started.
