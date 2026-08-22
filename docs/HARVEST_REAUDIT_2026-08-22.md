# Harvest re-audit (2026-08-22)

## Scope

`scripts/harvest.sh` and every directly invoked report generator were checked
again against the v4 finalization layout and the preregistered P4-0 rule. The
review covered run discovery, exit-code propagation, nonempty publication,
partial-output preservation, result-table copying, and shell regressions.

## Findings and fixes

1. `KCURVE.md` intentionally uses only `v2-*` runs. This is not a generation
   discovery bug: it preserves the preregistered P4-0 voting population.
2. The harvest bundle included only that preregistered report, however, even
   though v3/v4/27B runs must appear as non-voting extended evidence. The bundle
   now publishes `KCURVE_ALL.md` from `src/kcurve_all.py` separately. The v2
   decision rule and population are unchanged.
3. Existing zero-byte `TABLES.md` or `FRONTIER.md` files were silently skipped.
   A partial result directory could therefore yield a successful-looking
   harvest. Existing empty files and copy failures now abort the harvest and
   are named in `HARVEST_FAILURES.md`; absent optional reports remain optional.
4. The integration test previously replaced `kcurve_floor.py` with a stub and
   checked only its exit code, so it could not detect omission of the all-run
   report. The test now requires separate nonempty preregistered and extended
   reports and exercises extended-report failure and empty result-table cases.
5. A received screenshot contained only 11 of the expected 20 v4 runs, but a
   manually invoked harvest still wrote `HARVEST_STATUS.md`. When any non-smoke
   v4 run is visible, harvest now requires nonempty completion, configuration,
   manifest, score/oracle protocol, and report artifacts for all
   2 models x 5 seeds x 2 datasets. An incomplete matrix exits nonzero and
   records every missing artifact in `V4_MATRIX.err`.
6. `readout_summary.py` grouped conclusions only by generation and dataset.
   Consequently the screenshot's `v4/gsm8k 0/5` combined one 27B run with four
   7B runs, and `v4/math500 0/6` combined one 27B run with five 7B runs. Model
   families are now separate tags such as `v4/27b/gsm8k` and `v4/7b/gsm8k`.

## Screenshot triage

The screenshot is a partial readout, not a final confirmatory result. It shows
11 rows: two of ten expected 27B conditions and nine of ten expected 7B
conditions. Missing rows are eight 27B conditions and `v4-7b-s0`. Across the
visible rows the preregistered one-sided-loss criterion is met in zero runs.
The two visible 27B rows have `g00` highest and `g11` lowest, but each represents
only one seed/dataset condition and cannot establish a model-level ordering.
No paper claim should be updated until the complete artifacts are merged and
the corrected model-separated reports, confidence intervals, and reversal
audits are available.

`scripts/collect_v4.sh` is the single-command collection path. It does not run
or stop GPU work. Before aggregation it lists and counts three states
separately: a missing run directory (experiment result absent), a run without a
nonempty `DONE` marker (experiment incomplete), and missing postprocess
artifacts in an otherwise present run. Only a complete 20-run matrix proceeds
to model-separated TABLES/FRONTIER generation and final harvest.

If GPU workers have exited and these checks show absent or incomplete runs, the
affected cluster slots must resume `go_v4.sh` from the same generation commit.
`go_v4.sh` now delegates to `resume_v4.sh`, which reads the original commit from
the existing run configs and executes that snapshot in an isolated worktree.
This permits pulling analysis-only fixes without changing the immutable run
contract or quarantining multi-day partial artifacts. Mixed recorded commits
fail closed instead of merging incompatible runs.
After all slots finish, `bash scripts/collect_v4.sh` performs the collection;
the user does not copy run directories manually.

## Verification

- `bash -n scripts/harvest.sh`: pass
- `python3 -m py_compile tests/test_harvest.py`: pass
- `python3 tests/test_harvest.py`: pass, 39 checks
- `python3 tests/test_readout_summary.py`: pass, 13 checks
- `PYTHONPATH=src python3 tests/test_v4_resume_commit.py`: pass
- `python3 tests/test_v4_resume_shell.py`: pass (temporary Git worktree integration)
- Full local `pytest` cannot be collected with the system Python because this
  checkout has no real PyTorch installation. GPU/torch-dependent tests must be
  run in `$VENV_DIR`; this does not affect the shell regression above, which
  uses its explicit fake Python runner.

## Output contract

- `KCURVE.md`: preregistered P4-0 report; scientific exits 3 and 4 are valid
  report outcomes rather than process failures.
- `KCURVE_ALL.md`: all eligible generations and model conditions; extended
  rows do not enter the preregistered vote.
- `READOUT.md`, `REVERSAL.md`, and `STATS.md`: final only after a zero exit and
  nonempty output.
- Failed stdout/stderr remain as `*.partial.md` and `*.err`; the harvest exits
  nonzero and writes `HARVEST_FAILURES.md`.
