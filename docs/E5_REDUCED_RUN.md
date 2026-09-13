# Reduced E5: does the selected data actually train better? (2026-09-10)

Question: after selecting the top 10% of MATH-500 prompts with a stale
selector, does further GRPO training on that subset reach the same held-out
reward as training on the fresh selection, and does either beat a random
subset? Gradient alignment is a proxy; this experiment measures reward.

## Design

| item | value |
|---|---|
| source points | OLMo-3 matrix, MATH-500, seeds 0 1 2; two branch points: d400 (`…-d400`, arms resume its adapter and optimizer) and d0 (`…-d0`, arms start from the base model) |
| arms | `random` (uniform subset), `passrate_beta` (stored-reward difficulty band, zero selection cost), `fresh_r` (fresh top-k from split R), `g11` (stale top-k, full correction) |
| training | 100 further GRPO updates per arm from the point's `policy_step_400` adapter and optimizer, same objective configuration (4 ranks, one epoch, clip 0.2, lr 1e-5, LoRA q/v 16/32) |
| evaluation | 300 MATH-train problems that share no question with the 400 candidates or the 100 ranking-validation prompts, 8 responses each, Math-Verify reward, before (d400 policy) and after every arm |
| readout | mean reward after each arm, paired difference vs fresh_r with a 10,000-draw prompt bootstrap, overlap of each subset with fresh_r |

Cost: 900 updates in total (22% of the 4,000 in the matrix) plus 12 policies
x 300 prompts x 8 responses. Measured matrix throughput on 4xH100 (status
history 09-08/09, MATH-500 d400: 103->142 steps in 45 min, 311->392 in 95 min)
is about 70 s per GRPO update, so one arm takes about 2 h and one seed's three
arms about 6 h. Evaluation at 8 responses per prompt runs about 60 s per prompt
per GPU (responses run to the 2048-token cap), so one policy takes about 1.3 h
and one seed's four evaluations about 5-6 h. One seed is therefore about 12 h:
three nodes finish in about half a day, one node in about 1.5 days.

## Commands (phone-typable)

Once, in a shell with Hub access (login node):

```
bash scripts/fetch_math_train.sh
```

On each idle 4xH100 node (no OLMo or Qwen launcher on it), the d400 branch:

```
git pull --ff-only && bash scripts/run_e5.sh
```

and the d0 branch (same arms, starting from the base model):

```
git pull --ff-only && bash scripts/run_e5.sh d0
```

Progress of every branch from any node, no GPU:

```
bash scripts/run_e5.sh status
```

Reliability-logging run (random arm only, training only, separate root
`e5-reduced/math500-d<drift>-rlog`):

```
git pull --ff-only && bash scripts/run_e5.sh rlog
```

(`bash scripts/run_e5.sh rlog d0` for the d0 branch.) The trainer option
`--reliability-log` keeps the update unchanged and writes, per rank and step,
the two half-group pass rates and the cosines of the two half-group gradients
with the mean gradient of the other prompts in the batch
(`reliability_log.rank*.jsonl`). After training,
`src/reliability_trajectory.py` turns them into `reliability_trajectory.csv`
and `.dat`: split-half reliability of the pass-rate and gradient signals in
sliding 20-step windows, with Spearman--Brown full-group values, bootstrap
intervals, and the fraction of prompts with mixed rewards. About 2 h per seed.

Arms added after a seed was prepared (for example `passrate_beta` on a d400
seed that started with three arms) are recorded in `arms.json` next to the
frozen contract; completed shards stay valid and the new arm is trained and
evaluated on the next pass.

### Node ownership repair (2026-09-11)

The normal `run_e5.sh` command now stops the previous E5 seed loop and its
children on this node before taking the node lock. A shell's late `OUT_ROOT`
export is proved through its children; killing only the downstream child left
the older seed loop able to start another child and reacquire the lock.
Cleanup failures are no longer suppressed or reported as successful cleanup.

One controller keeps node ownership through all three seeds. Its child
launchers borrow that ownership without reopening the same lock or repeating
cleanup. Node-lock descriptors are not inherited by training, evaluation or
logging children. The `.before.lock` and per-arm locks in each shared seed
output remain unchanged: three nodes may still work on distinct arms/seeds.
Nothing is deleted or moved, and no healthy OLMo/Qwen launcher is stopped.

Node diagnostics print hostname, controller PID, lock path and filesystem.
An unavailable lock is not automatically attributed to OLMo/Qwen. The per-host
fallback requires an identified shared filesystem and no visible local owner;
an unsupported `flock`, a busy local lock with an unknown owner, or a failed
per-host lock cannot silently authorize GPU work. `force` remains explicit and
does not override a visible matrix launcher. Persistent GPU memory occupancy
after cleanup stops admission instead of merely warning and loading more models.
On shared filesystems, the controller also holds its per-host lock when the
legacy shared lock was initially free, preventing duplicate admission when a
different node later releases that shared lock.

Use the unchanged command on each of the three allocated E5 nodes after
updating its checkout. No new work root, arm selection or manual PID selection
is needed. CPU tests exercise the real Bash controllers and real file locks,
with mocked model work and simulated node-local process visibility. Both
separate node-lock directories and a shared-filesystem configuration are tested.
Live GPU execution is not accessible from the development host.

The follow-up addresses reports of many `--pickler=torch._inductor...` and
`/bin/sleep 15` entries. These are process command lines, not CUDA tracebacks.
The former owner listing expanded a lock opener into every descendant, so the
list did not establish that each compiler worker itself retained the lock.
Admission diagnostics now group actual file openers by their owning ancestors,
show PID/PPID and a parent command when available, and print at most eight
groups. Normal E5 cleanup output is bounded as well.

Unlabelled orphan compiler/sleep/tee families are reclaimed by the default
command, including those predating the E5 `OUT_ROOT` marker. Recovery requires
the exact node-lock file to be open, same-user local process visibility, and
an ancestry consisting only of these helpers ending at PID 1. A live training
PID named by the compiler's `--parent` option also prevents automatic cleanup.
A live or unreadable parent, an unrelated lock, or a non-helper process is not
automatically stopped. The existing force option is not needed for verified
orphan helpers. No lock file is deleted and no Torch compilation setting changes.
Real CPU orphan-process tests cover compiler-shaped workers and sleeps without
E5 markers; unit tests cover live parents, unreadable ancestry, other locks and
bounded output for a 101-process pool. The reported remote process tree is
not yet available, so these command fragments alone do not prove its owner is
orphaned.

Provenance note: the concurrent `5b44ab4` change relaxed code/runtime matching
in `evidence_downstream.py`. This node-ownership repair does not modify that
scientific file or broaden that relaxation. The older fixed-file-hash status
regression still fails against `5b44ab4`; that provenance compatibility issue
is separate from node admission and remains to be reviewed. The earlier
unchanged-scientific-source statement below describes the earlier repair,
not that concurrent change.

`bash scripts/run_e5.sh plan` prints the contracts and commands without
touching a GPU. Rerunning after a kill resumes from the newest five-step
checkpoint or the completed evaluation shards; arms are leased per seed so
several nodes share one seed's arms without duplicating work.

The 2026-09-11 launcher repair also handles an invalid final publication when
a compatible, hash-verified checkpoint remains. It uses the existing trainer's
recovery path without clearing the experiment output. An arm with no verified
repair checkpoint is preserved before continuing to other arms. The recovery
probe is CPU-only.
Scientific source files hashed by `experiment.json` are unchanged, so existing
E5 contracts, subsets, checkpoints and completed evaluations remain reusable.

Do not interrupt healthy training to install this repair or pull over a live
shared launcher. After that launcher exits, pull and rerun the same command;
no new output root or experiment restart is required. OLMo/Qwen status commands
in the repaired revision no longer update the checkout automatically. On the
older revision, use `bash scripts/run_e5.sh status` for E5 and
`bash scripts/status_qwen35.sh` for Qwen without the auto-updating wrapper.

`bash scripts/run_e5.sh status` retains detailed per-shard response counts,
log ages/tails and training checkpoint steps through a separate diagnostic
module. Five-minute console progress does not impose a five-minute wait:
evaluation completion is checked every second. Status labels describe files
present, not independently validated scientific results.

Outputs: `$OM_WORK/runs/e5-reduced/math500-d400/s<seed>/downstream_results.csv`
(and `.json`), one row per arm. Upload the three CSVs when they exist.

Environment knobs (defaults in parentheses): `E5_SEEDS` ("0 1 2"),
`E5_SELECTORS` ("random fresh_r g11"), `E5_STEPS` (100), `E5_EVAL_K` (8),
`E5_TEST_COUNT` (300), `E5_DRIFT` (400).

## What the result means

- fresh_r and g11 both above random, and close to each other: reuse keeps the
  training value of selection at this budget, although its overlap with the
  fresh set is low.
- fresh_r above random, g11 not: reuse loses training value at d400.
- all three similar: the selection signal at this budget does not translate
  into reward within 100 updates; the paper then reports that boundary.

Intervals are prompt-bootstrap intervals conditional on the trained seed; they
describe evaluation noise, not training-seed variability.

This three-arm experiment measures the training value of fixed data selections.
It does not choose among g00/g10/g01/g11 using A/B alignment or fresh-R overlap.
The manuscript's method-choice proposal is a distinct comparison, not a label
for these runs. Do not relabel d400/three-seed/100-update/300-question results
as d100/five-seed/200-update/500-question results, or expand `E5_SELECTORS` in an
already frozen output. Preserve the current experiment and its negative or
positive results alike.

## Executed gate arm and offline gate decisions (2026-09-13)

The manuscript's bounded diagnostic is executed as an arm of the same benchmark:

```
git pull --ff-only && bash scripts/run_e5.sh gate        # d400 branch
git pull --ff-only && bash scripts/run_e5.sh gate d0     # d0 branch
```

`gate_passrate` joins the seeds of the branch through `arms.json` (the frozen
contract is unchanged). Phase 1 trains a uniform pilot block (the whole
candidate pool in a seeded order, `pilot_size / 4` updates) with
`--reliability-log`; the decision applies the frozen rule
`config/gate_rule.json` (pilot size 40, `r_min` 0.25, Fisher-z bound at
two-sided 0.90) to the difficulty score `-|p-1/2|` of the two half groups of
every pilot visit; phase 2 resumes from the pilot policy and trains the
retained selector's subset (`passrate_beta`) or the random subset up to the
same total of 100 updates, then evaluates like every other arm.
`<seed>/gate_passrate/decision.json` records decision, reason, r, bounds,
pilot steps and pilot seconds; `downstream_results.csv` gains the columns
`gate_decision … forgone_vs_selector` (reward of the unchanged selector minus
reward of the gate arm, paired interval). The rule file is frozen per seed
directory (`gate_rule.json`); changing it afterwards invalidates the arm.

Offline decisions on the stored half scores (fresh a/b, difficulty from the
behavior responses, and the reuse estimators once `scores_stale_splithalf.json`
exists), mapped to the fixed arms' rewards, CPU only:

```
bash scripts/run_gate_decision.sh          # both branches; export under $OM_WORK/exports
```

`bash scripts/run_e5.sh rlog400` runs the reliability-logging random arm for
400 updates under the separate root `math500-d<drift>-rlog400`.

## Public benchmarks for the trained policies (2026-09-13)

The five sets are committed in the repository under `data/benchmarks/`
(`{aime24,aime25,amc23,gsm8k,math_rest}.jsonl` with manifests recording the
Hub revisions and hashes; AIME 2024/2025, AMC 2023, GSM8K test, and the 4,500
MATH test problems outside MATH-500). The cluster needs no download;
`scripts/fetch_benchmarks.sh` only regenerates them on an online machine.
On an idle 4xH100 node:

```
git pull --ff-only && bash scripts/run_e5_bench.sh          # d400 branch
git pull --ff-only && bash scripts/run_e5_bench.sh d0       # d0 branch
bash scripts/run_e5_bench.sh status | results | plan        # no GPU
```

Per seed the source checkpoint (`before`) and every arm whose policy is
complete (gate arms included) are evaluated on all five sets with the E5
sampling, prompt format and verifier; GSM8K and MATH-rest are frozen
200-problem subsamples (`E5_BENCH_COUNT`), `E5_BENCH_K` responses per prompt
(default 8). One process per (arm, GPU shard) loads the policy once and runs
every set; arms are leased per seed; completed shards are reused. Results:
`<seed>/benchmark_results.csv` (per set: mean reward, paired difference
against the source checkpoint and against the random arm, GPU seconds) and a
`macro` row per arm. `bash scripts/run_e5.sh export` bundles them.
Cost at the measured E5 evaluation rate (about 60 s per prompt per GPU at 8
responses): 500 prompts per policy, about 2 h per policy on four GPUs.

## Reuse-estimator half scores, gain law, and cost accounting (2026-09-13)

```
git pull --ff-only && bash scripts/run_stale_splithalf.sh        # d400 points, four GPUs; d0 with the word d0
bash scripts/run_gain_law.sh                                      # CPU: cross-half gain against rho*c on every point
bash scripts/run_cost_accounting.sh                               # CPU: GPU-seconds per arm, logging overhead, stage costs
bash scripts/run_gate_decision.sh                                 # CPU: frozen rule on fresh, difficulty and reuse halves
```

`run_stale_splithalf.sh` writes `scores_stale_splithalf.json` (g00, g10, g01,
g11 on the first and second half of every stored response group) into each
MATH-500 point, with a protocol file that compares a few recomputed
full-group scores with `scores_offpolicy.json`. The gate decision and the
gain law then include the reuse estimators. All CPU scripts write under
`$OM_WORK/exports/` and print the export path; `bash scripts/run_e5.sh export`
also bundles the gate decisions and benchmark results.

## Mixed pool: a positive control where selection should matter (2026-09-13)

Half MATH-500 candidates, half off-task prompts (MBPP by default); the ranking
validation set and the independent test set stay MATH. One new d0 point is
built with the matched configuration of the existing MATH d0 point of the same
seed (`scripts/run_point.sh` with `OM_POOL_FILE`, pre-split pool), then the
reduced E5 arms and the gate arm run on it with `MIX_STEPS` updates (default
200). A random subset spends part of its budget on prompts whose rewards carry
no signal for MATH; the difficulty and gradient scores can exclude them.

```
bash scripts/run_mixed_pool.sh pool      # CPU: pool file under $OM_WORK/inputs/mixed
bash scripts/run_mixed_pool.sh point     # GPU node, about a day: rollouts, gradients, scores
bash scripts/run_mixed_pool.sh e5        # GPU node: random / difficulty / fresh / reused arms
bash scripts/run_mixed_pool.sh gate      # GPU node: executed gate arm
bash scripts/run_mixed_pool.sh status | results
```

Outputs: point under `family-math500mix-s0/…-math500mix-d0`, arms under
`runs/e5-reduced/math500mix-d0/s0` (same test set as the MATH d0 branch,
`E5_TEST_DATASET=math500`); `run_e5.sh export` bundles them. Knobs:
`MIX_OTHER`, `MIX_MATH`, `MIX_N_OTHER`, `MIX_STEPS`, `MIX_SEED`.

## Queue of the remaining steps and its one-screen status (2026-09-13)

```
bash scripts/run_queue.sh          # GPU node: every remaining step in order (leased; skips finished or claimed work)
bash scripts/run_queue.sh status   # CPU: one line per step with a state word, seed detail below
```

`status` (`src/queue_status.py`) reads the shared filesystem only. State words:
DONE (all artifacts exist), RUNNING (a lease of the step is held right now on
some node; `*[node]` marks the leased arm and the node holding it), PARTIAL
(artifacts exist, nothing running), WAITING (prerequisite step unfinished),
TODO (nothing started). Below the steps, a `nodes` section lists every node
that ran the queue with its current step, the time it started it, and whether
its heartbeat is alive (`$OM_WORK/queue/<host>.txt` and `.beat`, written by
`run_queue.sh`; a node killed by the scheduler shows NO HEARTBEAT), plus the
leases each node holds (`scripts/_lease.sh` writes host/pid/time into every
lease file on acquisition; leases taken by older code show `*[?]`). The
`this node` line lists the queue processes on the current machine by their
OUT_ROOT marker. The per-step commands keep their detailed `status` modes.
