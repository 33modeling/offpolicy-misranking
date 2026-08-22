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

## Verification

- `bash -n scripts/harvest.sh`: pass
- `python3 -m py_compile tests/test_harvest.py`: pass
- `python3 tests/test_harvest.py`: pass, 29 checks
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
