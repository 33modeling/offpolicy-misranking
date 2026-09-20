# Current Paper Data

Run from the repository on the experiment server:

```bash
bash scripts/run_paper_results.sh results rloo
bash scripts/run_paper_results.sh results pair
bash scripts/run_paper_results.sh results mbpp
```

Each command writes one fixed-name file in the home directory:
`rloo-results.txt`, `selector-pair-results.txt`, or `mbpp-results.txt`.
Repeated exports atomically replace that file, rather than creating numbered
parts or timestamped copies. Each file is limited to 1,900,000 UTF-8 bytes.
Oversize exports fail explicitly without truncation or replacing the old file.
An optional `--out /path/name.txt` selects a different destination.

RLOO includes sealed per-question rewards, available complete-arm comparisons
against cached selection (`passrate_beta`), confidence intervals, raw cost events,
and explicit missing arms/shards. An incomplete `before` no longer blocks other
measurements. Partial-shard averages are descriptive, not final benchmark scores.
Invalid experiment points are labeled and excluded while other points are kept;
the command exits nonzero if any point fails validation. Canonical `results.json`
and frozen training code are not changed by the progress exporter.

Pair regenerates its validated partial report, then packs its development/test
measurements, costs, missing states, and curve CSV into the TXT. A failed report
does not re-export a stale report as new data. States without a published result
remain missing; ongoing unfinished curves are not certified benchmark results.

MBPP combines all configured, existing observed suites into one TXT, preserving
the full results export's per-question rewards and comparisons. Missing suites
and per-suite export errors are listed. Saved-result snapshots are not a new
validation of policy/optimizer lineage. Experiment conditions remain separate.

These commands use CPU reporting only. They do not launch training, stop workers,
change budgets, remove GPU processes, or delete existing experiment artifacts.
