# Executed SR-GC Switch Rewards

## Run on All Available Nodes

Update the existing code checkout, then run the same command on each allocated
four-GPU node sharing the experiment filesystem:

```bash
git pull origin master
bash scripts/run_selector_pair_switch_rewards.sh
```

Start with four allocated four-GPU nodes: two train the independent seed
trajectories and two evaluate. Other nodes automatically
claim checkpoint evaluations, including checkpoints published while the learners
are still running. Training world size remains four for comparability; adding
nodes does not change the optimizer, batch size, sampling or update sequence.
Final reward evaluations take priority over intermediate curve evaluations.
There is no new selector scoring, D measurement, regression or control training.
If the old trigger optimizer is missing, the learner first replays only the
required On-policy interval from the existing full step-25 parent, in a new
directory. Seed 3 needs 100 replay updates (25 to 125); seed 4 needs 75 (25 to
100). The SR suffix follows automatically. No manual stage change is needed.

The launcher does not provision nodes. Launch it on each available allocation.
If only one node is available it handles both seeds and evaluations sequentially.
`--seed 3` or `--seed 4` restricts a worker, but leave it unset for automatic
load balancing across all nodes.

## Inputs and Outputs

Defaults are `$OM_WORK/runs/selector-pair-v1` for the read-only source and
`$OM_WORK/runs/selector-pair-srgc-switch-v1` for the new experiment. `OM_WORK`
defaults to `/group-volume/${OM_USER:-minsoo3.kim}/offpolicy-misranking`.
Optional overrides are `PAIR_ROOT`, `PAIR_SWITCH_ROOT` and `PAIR_PYTHON`.

```bash
bash scripts/run_selector_pair_switch_rewards.sh plan
bash scripts/run_selector_pair_switch_rewards.sh results
```

If the trigger directory contains only archived adapter weights, the launcher
also searches the Pair tree, sibling run/backup directories, and the work area's
`checkpoints/` directory for the original optimizer. It accepts only the saved
checkpoint's exact hash; other seeds/steps and optimizer resets are not substitutes.
Missing checkpoint statistics may be recovered from the exact byte prefix of the
original training log, again only when its saved hash matches. Source files are
never changed. A missing seed does not prevent another resumable seed from running.

For a CPU-only search of both seeds and a report at
`~/selector-pair-switch-checkpoints.txt`:

```bash
bash scripts/run_selector_pair_switch_rewards.sh checkpoints
```

An additional mounted backup location can be supplied through the optional
`PAIR_CHECKPOINT_SEARCH_ROOTS` environment variable (colon-separated paths).
If only the trigger optimizer is missing, the script explicitly reports
`replay_on_policy_from_25` and validates the saved step-25 model AND optimizer
before admitting GPU work. It does not reset the optimizer, take a later-step
optimizer, or overwrite the old On-policy branch. Missing/corrupt parent files
remain an error, not permission to restart from zero.

The ordinary `run_selector_pair.sh` does not regenerate deleted intermediate
optimizers of already completed branches. Use the switch command above: it
reuses that experiment's saved parent, subsets, controls and D measurements.

`plan` is read-only and reports the stored-D trigger, full source checkpoint,
common terminal step, and separate median/p90 prefix-replay and suffix-training
estimates from the corresponding saved step timings. These are not end-to-end completion guarantees: evaluation,
allocation and interruptions are additional. Workers default to a 24-hour
allocation window, retain progress on expiry, and exit after 120 minutes with
no claimable work rather than wait forever. Reuse the same command to resume.

`results` writes `switch-rewards.txt` and `switch-rewards.json` under the new
experiment root, including all four measured curves, final rewards at a common
step, missing points, last saved training step and per-attempt cost ledgers.
Copy `switch-rewards.txt` for manuscript ingestion. Unknown interrupted cost
remains unknown, not zero. Historical source costs remain in the source ledgers.
When all curves are complete, optional `matplotlib` also writes
`switch-rewards.pdf` and `switch-rewards.png`. To enable plots:

```bash
# In the experiment's Python environment:
python -m pip install -r requirements-switch-plots.txt
bash scripts/run_selector_pair_switch_rewards.sh results
```

## Scientific Contract

The first two consecutive stored `D < 0` checks, 25 steps apart, determine the
trigger. Missing checks cannot be skipped; future D or rewards do not choose it.
Existing measurements imply seed 3 at step 125 and seed 4 at step 100, but the
implementation computes and validates these rather than hard-coding them.

All branches share the existing On-policy prefix from step 0 to 25. The controls
then continue with Random, On-policy or SR. Thus the SR control is **SR from step
25**, not SR-only from initialization. The new Switch curve shares the actual
On-policy prefix until the trigger when its exact optimizer exists, then trains
on the existing cached SR subset. If that optimizer was deleted by an older
trainer, the new curve instead uses the genuinely replayed On-policy prefix
and its regenerated trigger model/optimizer. Never splice either the old
On-policy rewards or SR control rewards onto the regenerated trajectory.

`s<seed>/replay-audit.json` compares original/replayed model, optimizer and log
hashes. Byte identity is not assumed: GPU replay can differ. A non-identical
replay tests continuation at the previously frozen original-trajectory trigger;
it is not evidence of a newly evaluated D controller on that replay. The result
JSON carries this distinction and the audit. Replay training costs are recorded
separately in the research ledger, not hidden in evaluation or counted as zero.

The terminal step is the latest checkpoint saved by all three controls, selected
from checkpoint availability, not observed reward. No target35 or fixed step100
cap is imposed. Existing curve shards are verified and reused. Missing reporting
evaluations use the original held-out prompts, response count and sampling
recipe. Switch checkpoints are evaluated every 25 steps plus the terminal step.
The four thin curves show Random in gray, continued On-policy in light blue,
SR control in light green, and Switch in dark blue then dark green.

This is a genuinely executed suffix replay under a prefix-only decision rule,
not a claim that the rule was designed on an untouched prospective test set.

## Interruptions and Isolation

Before claiming work, each worker runs the existing Pair four-rank NCCL/DDP
admission probe. It records original rank errors and only carries a transport
override into training after the corresponding probe succeeds. Bounded failed
probes stop the node without claiming training or changing source checkpoints.
Evidence is saved under the new output root's `node-preflight/`, not the old
Pair experiment. This does not guarantee recovery from every NCCL failure;
hardware, driver and training-time errors still require their original logs.

Full model/optimizer checkpoints are retained every five updates, both under
`s<seed>/replay/policy/` and `s<seed>/policy/`. Completing a stage does not delete
them. The existing trainer resumes the latest validated checkpoint and
repairs interrupted final publication. Work interrupted before the first durable
checkpoint is moved to `interrupted-attempts/`, never deleted. Evaluation shards
have independent completion hashes and restart only incomplete work.

Shared task leases exclude duplicate learners and evaluations. GPU children
inherit the task lock, so killing a controller does not admit another writer
while its workers remain alive. Each attempt has a separate cost ledger; a
previous interrupted ledger cannot block resumption or become a zero-cost run.
Original Pair checkpoints, subsets, results and running jobs are not modified.

## Local Verification

Replay and exact-resume paths are exercised with fixture policies, including
separate replay rewards, optimizer lineage, repeated invocation, mid-replay
interruption and retained checkpoints. These checks are not a remote GPU run.
The unrelated legacy migration suite `test_checkpoint_retention_runtime.py`
has 11 historical code-fingerprint fixture failures, reproduced unchanged on
the pre-fix commit `be42f0b`; this new runner does not use that migration path.
