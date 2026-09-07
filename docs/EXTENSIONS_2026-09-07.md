# Registered extensions E1-E6 (2026-09-07)

Registration and rationale: `EXPERIMENT_ADDITIONS_2026-09-07.md` in the paper
repository. Every module below is a new file; no registered pipeline file was
changed. All modules were exercised end to end on a CPU harness (tiny random
OLMo-3, synthetic prompts, four gloo ranks) that runs the real
`run_matrix.sh -> run_point.sh` orchestration, and have unit tests in
`tests/test_additional_experiments.py`.

| Extension | Entry point | Needs | Output |
|---|---|---|---|
| E1 drift curve | `bash scripts/run_drift_curve.sh configs/olmo3_rlzero.json <curve root> <curve results> [h100]` then `python src/drift_curve.py --registered <runs> --curve <runs> --output-dir <dir>` | new training: one fine chain `0 5 10 25 50 100 200 400` per seed (0, 1), MATH-500 | `drift_curve_summary.csv`, `<source>_<dataset>_<metric>_vs_drift.dat` |
| E2 reversal frequency | `python src/reversal_matrix.py <runs> --output-dir <dir>` | completed runs only | `reversal_matrix.csv/.md` (overall and boundary-band rates, anchor) |
| E3 margin condition | `python src/margin_condition.py <runs> --output-dir <dir>` | completed runs only | `margin_condition.csv`, summary JSON; exit 2 on a violation |
| E4 synthetic pool | `python src/synthetic_pool_study.py --output-dir <dir>` | CPU, ~1 min | `synthetic_pool_summary.csv`, `*_vs_delta.dat`, `*_vs_kb.dat` |
| E5 downstream update | `bash scripts/run_downstream_compare.sh <completed d>0 run> <out root> [steps=50]` | training: 7 selectors x 50 GRPO updates per seed from `policy_step_d` | `downstream_summary.csv` |
| E6 sensitivity | `python src/rescore_variants.py --run <run> --model <base> --variants bk2 bk4 clip3 clip30` (GPU re-scoring, no generation) then `python src/sensitivity_tables.py <runs> --output-dir <dir>`; `python src/diagnostics_vs_retention.py <runs> --output-dir <dir>` | completed runs; re-scoring loads base and policy | `sensitivity_summary.csv/.md`, `diagnostics_spearman.csv` |

Notes.

- `rescore_variants.py` writes `scores_offpolicy.variant-<name>.json` and
  `variant_protocol.json` next to the registered scores and never rewrites
  `scores_offpolicy.json`.
- `run_drift_curve.sh` trains a separate chain because the registered trainer
  keeps only the two most recent durable checkpoints; it never reads or
  writes a registered run directory.
- `run_downstream_compare.sh` resumes the registered trainer from the point's
  adapter and optimizer with the same objective configuration; the held-out
  reward is the mean verifier reward over the validation prompts
  (`DOWNSTREAM_EVAL_K` samples each, default 8).
- Registered labels are never recomputed by these modules; every output is
  descriptive unless the registration states its own criterion (E3).

## One-shot cluster runner

```bash
git pull --ff-only
bash scripts/go_extensions.sh h100            # all stages: synthetic rescore analyze curve downstream
bash scripts/go_extensions.sh h100 analyze    # only the analysis tables over completed runs
```

Stages are idempotent and resumable; rerun the same command after an
interruption. Output: `$OM_WORK/results/extensions-<model tag>/` and the
console log under `$OM_WORK/console-logs/`. The `curve` stage trains the E1
chain (400 updates per seed) and the `downstream` stage the E5 updates (7
selectors x 50 updates per seed), so run those two on a node that has finished
its share of the registered matrix.
