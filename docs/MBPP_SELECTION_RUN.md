# MBPP Selection Experiments

Run this same command from the original `offpolicy-misranking` checkout on
**every allocated four-H100 node**, including replacement nodes:

```bash
bash scripts/run_mbpp_experiments.sh
```

The command joins the existing node controller's shared MBPP queue:

| Suite | Continuation selection | Accounting | Gate |
| --- | --- | --- | --- |
| `fresh` | Fresh gradient scores | Scoring and training share the allocation | Final reward |
| `quality` | Fresh gradient scores | Scoring recorded separately from training allocation | Updates saved minus selection cost in update units |
| `difficulty` | Cached success rate closest to 0.5 | Scoring and training share the allocation | Updates saved minus selection cost in update units |

All suites include random controls. The fresh suite creates five fresh-selected
training prefixes. The other two reuse these exact prefixes, states at
25/50/100 updates, and the same evaluation questions. They do not import MATH
prefixes. Each suite uses the existing 18 development and 30 held-out
continuations, with development seeds 0/1/2 and held-out seeds 3/4.
Once all shared prefix certificates are ready, quality and difficulty can start
even while fresh continuations are still running elsewhere. No node is assigned
permanently to one suite. Final evaluation uses eight responses per question;
convergence curves use three archived checkpoints with four responses per question.

This is a port of the current switch suites, not a new difficulty definition,
an E5 fixed-checkpoint run, or a gate that directly chooses fresh versus
difficulty in one branch. The original gates choose their suite's selector
versus random; the shared states permit the fresh/difficulty comparison.

## Commands

```bash
bash scripts/run_mbpp_experiments.sh plan       # settings only, no writes/GPU work
bash scripts/run_mbpp_experiments.sh check      # read-only local input checks
bash scripts/run_mbpp_experiments.sh status
bash scripts/run_mbpp_experiments.sh progress   # shared experiment/node view
bash scripts/run_mbpp_experiments.sh stop       # stop/clean THIS node, not peer nodes
bash scripts/run_mbpp_experiments.sh results    # one report per suite, also copied home
bash scripts/run_mbpp_experiments.sh why
bash scripts/run_mbpp_experiments.sh run fresh
bash scripts/run_mbpp_experiments.sh run quality
bash scripts/run_mbpp_experiments.sh run difficulty
```

## Nodes Joining And Failing

`run` and `stop` use `scripts/run_experiments.sh` for the existing queue,
detached console, keepalive, watchdog, automatic Git updates, stale-event
recovery and GPU-fault waiver policy. A small MBPP ownership guard now holds
one controller lease per node and supervises shutdown; it does not schedule
another queue or change the learner, selectors, costs, or checkpoints.
The inner worker yields its idle peer-wait to this shared controller, instead
of waiting inside one suite while another has work. Owned training/evaluation
finishes normally before yielding; no active branch is interrupted for fairness.

- On the same node, a duplicate command reports `already running` and leaves
  the healthy controller working. It does not stop it or create another worker.
  Use `stop` only when an intentional interruption is needed.
- After a controller is killed, run the same command. Its durable owner token
  identifies its own surviving children, including separately-sessioned ranks.
  Those children are stopped before the replacement starts. The node lock is
  held by the guard only, never inherited by rollout/scoring workers.
- On a new node, the same command reads the shared queue and claims available
  tasks using the existing leases. Busy branches are skipped, not duplicated.
- If a node dies, its locks are released; surviving nodes use the existing
  heartbeat/stale-cost recovery and recorded GPU-fault waiver rules before
  retrying its work. Completed results and valid checkpoints remain reusable.
- If a branch fails, the controller attempts other available work and retries
  later. Missing source inputs or prefixes wait in the queue; they do not
  prevent cleanup or terminate the whole launcher.
- A node that repeatedly fails GPU/NCCL admission is released under the
  original policy. A replacement node must be allocated by the cluster/user;
  this script does not request a new cluster allocation itself.
- The node exits successfully only when every requested MBPP suite is complete,
  including suites whose inputs were initially pending. An unrelated unfinished
  MoPPS run does not keep this MBPP allocation alive.

The MBPP hold defaults to 600 seconds (`MBPP_HOLD_SECONDS`), with
`EXPERIMENTS_HOLD_SECONDS` taking precedence when set. Holds poll for newly
available work and exit early when peers complete the queue. Setting
`EXPERIMENTS_HELP_SIBLINGS=0` explicitly restricts a node to the first root;
leave it enabled to serve all requested suites.
Terminal launches detach as before: Ctrl-C stops the log view, not the workers.
Use `stop` to stop this node's MBPP controller and its token-bound children.
**A busy lock no longer triggers a node-wide process or GPU sweep.** Other
experiments are not cleanup targets merely because they use the same account,
work volume, or GPUs. MBPP recovery/watchdog roots exclude unrelated math and
MoPPS roots. PID files are checked for an actual MBPP launcher before signalling.
Existing untagged legacy workers cannot be safely adopted as dead just because
they hold a lock; their ownership must be checked rather than deleting locks.
Completed outputs and cost ledgers are not reset by this ownership fix.

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
meanings. Status and results never start training. Trainers, scoring code,
data splits, gate calculations, and frozen experiment contracts are unchanged.
