# V7 additional experiment audit — 2026-10-20

Reviewed the experiment repository on `master` starting at
`8e06b1e4b63bd2cf1fc387b9627a293740c1d1bb`. The request was to verify
implementation coverage and check existing additional-experiment code for bugs.

## Implementation coverage

| Group | Current code |
| --- | --- |
| Backbone replication | Qwen, Llama and Gemma launch/status/result paths |
| Existing controls | Stage mechanism, fixed switching, confirmation rules, matched/candidate/pool SR refresh, repeated switching, direction controls and same-prefix replicates |
| N01–N08 | Dedicated research trajectories, diagnostics and analysis in `srgc_research/` |
| N12/N13 | Source-bound CPU analysis, checkpoint inspection, inference and gradient-scoring diagnostics in `srgc_research/followup/` |
| N09–N11, N14–N18 | Designs remain; no new dedicated GPU experiment runners implemented by this audit |

Existing launch definitions for fixed200 are preserved. The author's exclusion
of fixed200 from adopted comparisons remains in force; this audit does not
restart, assign or reinterpret that experiment.

## Reproduced defects and repairs

1. **Timing/rule reports could not publish.** The unified dispatcher and switch
   reporter supported `timing` and `rules`, but `srgc_result_collection.publish`
   accepted only `mechanism`, `support` and `switch_validation`. Both scoped
   result commands therefore exited 1 with `unknown result collection scope`
   even after successful collection. The publisher now accepts both scopes,
   retaining their separate `results/timing/` and `results/rules/` directories.
   They do not overwrite each other, combined validation or P0 latest files.
2. **Default text output hid the original-file bundle path.** `print_paths`
   skipped `COLLECTED FILES` unless JSON mode was selected. Both human-readable
   locations now appear in normal output. JSON mode keeps the JSON payload on
   stdout and both location messages on stderr.

The regression tests run the real CLI reporter and publisher using injected,
already-collected synthetic reports. They verify exact raw snapshots, preserved
source bytes, separate scope/latest paths and output streams. They do not
replace endpoint validation with a claimed real experiment.
Before the repair, six of the seven new cases failed; only the existing JSON
path behavior passed. The isolated repaired publisher passes all seven.

## Verification

- Before repair: both full experiment packages passed **1,964 tests and 325 subtests**
  in 542.25 seconds, with no failures or skips. This demonstrates the two defects
  were coverage gaps rather than failures already caught by that suite.
- After repair: all eight affected CLI/report/control test modules, including
  the new regressions, passed **305 tests and 154 subtests** in 35.73 seconds.
- New regressions also passed **7/7** against the canonical code using ordinary
  disk fixtures, independently of the isolated shared-memory test directory.
- The 67 full-suite warnings were existing Python 3.12 multiprocessing `fork`
  deprecation warnings in CPU distributed tests, not test failures.


The additional repeated-switch probe restores at 13 boundaries, including
return to On-policy and a second SR transition, and compares selected IDs,
weights and Adam moments with an uninterrupted run. They are identical.
Static parsing passed for 288 tracked experimental Python files and syntax
checks passed for 12 launcher shell files.

Only result publication/output, its regression tests and this audit record
change. The scientific Engine, stage-mechanism implementation and Llama/Gemma
adapter hashes remain identical. Frozen snapshots, live jobs, queues, original
inputs/checkpoints/results and the V7 manuscript are unchanged. This CPU audit
does not establish H100 memory use, CUDA/NCCL execution or remote filesystem
behavior; the available Torch build is 2.13.0+cpu.

Evidence: `/tmp/v7-additional-audit-20261020/` contains the pre-repair source
hashes, failing regressions, isolated repair result, baseline/final logs,
static checks and repeated-switch resume probe. Full-suite and targeted counts
are separate observations with overlapping tests, not additive unique totals.
