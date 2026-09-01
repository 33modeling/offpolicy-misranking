# 2026-08-24 v4 Readout Provenance Status

## Verdict

`READOUT.md` is a retired partial progress report. Its automatic conclusions
group runs by generation/model/dataset but not by the generation commit stored
in each run's `run_config.json`. Runs created by different code revisions can
therefore appear in one seed count. Those aggregate counts are inadmissible.

## What Can Be Recovered

The retained report contains derived numbers and run names, but not the source
`run_config.json`, `manifest.json`, or score-protocol documents for each run.
The exact revision partition cannot be reconstructed from this directory.
The separate 2026-08-21 progress archive records commit `abe9dbb` for its five
incomplete runs, but that fact cannot be transferred to the later 2026-08-24
bundle.

If the original run directories remain available, regenerate the diagnostic
with `src/readout_summary.py`. It validates `run_config`, manifest, and score
protocol provenance, prints the generation SHA for every row, and groups
automatic conclusions by SHA. Missing or contradictory provenance fails the
readout instead of assigning a revision by majority.

## Current Rule

New regime matrices bind one generation commit before the first family is
claimed. The canonical RLVR launcher also binds its 27B primary and 7B
replication roots to one suite marker before either matrix starts. A partial
root fixes the commit for an empty peer; conflicting existing roots abort.
Analysis records `generation_git` separately from `analysis_git` and partitions
summaries by generation commit. Final harvest requires exactly one generation
commit per matrix and the same generation commit for both matrices.
