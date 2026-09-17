# MBPP Selection Experiments

Run from the original `offpolicy-misranking` checkout on an idle, allocated
four-H100 node, inside tmux:

```bash
bash scripts/run_mbpp_experiments.sh
```

The command runs the existing switch experiment on MBPP, in this order:

| Suite | Continuation selection | Accounting | Gate |
| --- | --- | --- | --- |
| `fresh` | Fresh gradient scores | Scoring and training share the allocation | Final reward |
| `quality` | Fresh gradient scores | Scoring recorded separately from training allocation | Updates saved minus selection cost in update units |
| `difficulty` | Cached success rate closest to 0.5 | Scoring and training share the allocation | Updates saved minus selection cost in update units |

All suites include random controls. The first suite creates five fresh-selected
training prefixes. The other two reuse these exact prefixes, states at
25/50/100 updates, and the same evaluation questions. They do not import MATH
prefixes. Each suite uses the existing 18 development and 30 held-out
continuations, with development seeds 0/1/2 and held-out seeds 3/4.
Final evaluation uses eight responses per question; convergence curves use
three archived checkpoints with four responses per question.

This is a port of the current switch suites, not a new difficulty definition,
an E5 fixed-checkpoint run, or a gate that directly chooses fresh versus
difficulty in one branch. The original gates choose their suite's selector
versus random; the shared states permit the fresh/difficulty comparison.

## Commands

```bash
bash scripts/run_mbpp_experiments.sh plan       # settings only, no writes/GPU work
bash scripts/run_mbpp_experiments.sh check      # read-only local input checks
bash scripts/run_mbpp_experiments.sh status
bash scripts/run_mbpp_experiments.sh results    # one report per suite, also copied home
bash scripts/run_mbpp_experiments.sh why
bash scripts/run_mbpp_experiments.sh run fresh
bash scripts/run_mbpp_experiments.sh run quality
bash scripts/run_mbpp_experiments.sh run difficulty
```

Each stage waits for its full queue before advancing. Ordinary failed tasks
are retried by the existing launcher, starting with a 600-second hold;
`MBPP_HOLD_SECONDS` changes that interval. Completed artifacts are reused and
interrupted work resumes under the existing checkpoint and cost-ledger rules.
Admission failures and terminal signals stop the sequence. A missing cost
receipt is not silently erased or waived. Ctrl-C in the tmux pane stops the
foreground run; no subsequent suite is started. No auto-pull or other
experiment launch is enabled by this entrypoint.

## Inputs And Outputs

Use a clean, committed checkout and the existing training environment. The
runtime snapshot refuses dirty executable files, including untracked ones;
the launcher does not discard them. Required inputs:

- Completed OLMo MBPP matrix d0 points for seeds 0 through 4.
- MBPP seed-0 d100 `grpo_stats.jsonl`, to derive the same update-equivalent
  allocation using MBPP timings rather than MATH timings.
- The pinned `mbpp.jsonl` and its SHA-256 manifest, obtained online with
  `bash scripts/fetch_datasets.sh mbpp` if not already installed.
- The matrix's model snapshot and four-GPU training environment.

The original MBPP matrix trains on a subset of the merged **full MBPP pool**.
Evaluation excludes the union of all five seeds' training candidates and
ranking-validation questions. The remaining questions are shared across
all arms and suites. The preflight prints the actual available count; the
existing protocol takes at most 300 and refuses fewer than four. A small
remaining set limits evaluation precision. This is an internal held-out
MBPP experiment, **not the official MBPP test-split benchmark**. The prompt
and assertion-execution reward are unchanged from the original MBPP matrix.

Default roots under `$OM_WORK/runs`:

- `selection-switch-mbpp-v1`
- `selection-switch-mbpp-quality-v1`
- `selection-switch-mbpp-difficulty-v1`

Override with `SWITCH_MBPP_ROOT`, `SWITCH_MBPP_QUALITY_ROOT`, and
`SWITCH_MBPP_DIFFICULTY_ROOT`. Roots must be separate, non-nested directories.
`OM_WORK`, `OM_OLMO3_ROOT`, `DATASETS_DIR`, and `VENV_DIR` retain their existing
meanings. Status and results never start training. Existing math outputs,
trainers, scoring code, and frozen experiment contracts are unchanged.
