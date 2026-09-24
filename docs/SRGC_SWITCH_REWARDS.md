# Executed SR-GC Switch Rewards

## Run on All Available Nodes

Update the existing code checkout, then run the same command on each allocated
four-GPU node sharing the experiment filesystem:

```bash
git pull origin master
bash scripts/run_selector_pair_switch_rewards.sh
```

Two nodes can train the two independent suffixes. Other nodes automatically
claim checkpoint evaluations, including checkpoints published while the learners
are still running. Training world size remains four for comparability; adding
nodes does not change the optimizer, batch size, sampling or update sequence.
Final reward evaluations take priority over intermediate curve evaluations.
There is no new selector scoring, D measurement, regression or control training.

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

`plan` is read-only and reports the stored-D trigger, full source checkpoint,
common terminal step, and median/p90 suffix-training estimates from saved SR
step timings. These are not end-to-end completion guarantees: evaluation,
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
On-policy prefix until the trigger, then trains on the existing cached SR subset
from the exact trigger checkpoint, including its optimizer state. Never splice
the SR control's rewards onto this trajectory.

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

Full model/optimizer checkpoints are retained every five updates in the new
directory. The existing trainer resumes the latest validated checkpoint and
repairs interrupted final publication. Work interrupted before the first durable
checkpoint is moved to `interrupted-attempts/`, never deleted. Evaluation shards
have independent completion hashes and restart only incomplete work.

Shared task leases exclude duplicate learners and evaluations. GPU children
inherit the task lock, so killing a controller does not admit another writer
while its workers remain alive. Each attempt has a separate cost ledger; a
previous interrupted ledger cannot block resumption or become a zero-cost run.
Original Pair checkpoints, subsets, results and running jobs are not modified.
