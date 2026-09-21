# Results export audit

## Uploaded MBPP file

`~/mbpp_results_1.txt` reports exporter checkout `6624305`. After line-ending
normalization, comparison with `~/mbpp-results.txt` changes only the three
timestamp and checkout headers. Parsing both files produces identical data:
37/48 quality endpoints, 21/42 inclusive endpoints and 3/6 cached endpoints.
The quality full-control comparison remains 4/6 measured pairs, not a newly
completed 48-branch export. This does not establish the remote repair's status.

Input SHA-256 values:

- Previous: `dd7d34095df13711063cf8fc82be67cb3470a85d2e861cb0883ac7c905ad3deb`.
- Uploaded: `cd901b22d0a44d5b2e1e1363379bc89b98fb66dedeb2c738049f3bf1d5a94f5f`.

The defect is in root selection: default run/restart can route to the prepared
repair, but default results observed only the original and historical roots.
Commit `6722208` adds the existing repair to both default MBPP results entrypoints.
The single `~/mbpp-results.txt` now includes a separate `repair_runs` section,
48 repair inventory entries, reused/rerun/dependent origin labels, question
rewards, curves and distinct original/new cost records. It does not merge the
repair with the original scientific protocol or count reused branches as new
training. Explicit repair roots also use the origin-separated exporter.

## Pair and RLOO

- Pair source/observation reads now reject nonregular files without waiting for
  a FIFO writer. Source reads are bounded to 8 MiB. A damaged branch is flagged;
  valid measurements from other branches remain exportable. This edge case was
  locally reproduced, not observed in the user's live Pair training run.
- A missing RLOO root now produces a fresh error TXT and nonzero exit instead of
  leaving a previous result TXT that could be mistaken for current output.
- Missing references, incomplete curves, invalid seals and pending branches
  remain distinct from completed comparisons. No missing value is imputed.

## Verification

- Four real-Bash MBPP repair-routing regressions failed before the change and
  passed afterward, covering both entrypoints and partial/complete fixtures.
- Initial MBPP export/repair/launcher suite: 56 passed.
- Combined MBPP/Pair/RLOO result and validation tests: 211 passed.
- Three real-process Pair FIFO regressions and the RLOO stale-file regression
  failed before their fixes; they are retained in the regression suite.
- No GPU job was launched or stopped. No experiment data, checkpoint, target,
  training budget or scientific source file was modified.

## Commands

From an updated checkout, each command writes one stable TXT under home:

```bash
bash scripts/run_mbpp_experiments.sh results
bash scripts/run_selector_pair_results.sh
bash scripts/run_rloo.sh results
```

The existing repair-specific command remains available on older checkouts:

```bash
bash scripts/run_mbpp_repair.sh results
```

It writes `~/mbpp-repair-results.txt`. Actual completed repair measurements are
still needed before updating the paper's endpoint tables and learning figures.
Figure 2 is a different score-calibration experiment: the current archived
data contain no MBPP off-policy rows, and `run_stale_splithalf.sh` targets MATH.
MBPP continuation rewards cannot be substituted for those calibration inputs.
