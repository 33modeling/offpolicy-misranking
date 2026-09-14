# Selected-prefix switching experiment

Implemented: 2026-09-14. Manuscript target: v4. No GPU result is claimed here.

## Run

Use four nodes with four H100 GPUs each, sharing the same group-volume root.
Two nodes also work; neither the labels nor gate fitting require four nodes.
The queue uses nonblocking task locks across nodes. A node does not own a
permanent seed or arm.

After `git pull`, first run one registered smoke continuation on one node:

```bash
bash scripts/run_selection_switch.sh smoke
```

This creates seed 0's selected prefix through update 25, measures once, then
executes CONTINUE_D including fresh_r scoring, training and independent
evaluation. It is a real registered experiment, not a disposable artificial
GPU benchmark. Its outputs and charges are reused by the full queue. Its
time limits are the prefix timeout, frozen branch allocation and evaluation
timeout; it is not a promise of a short, seconds-long check.

After checking smoke completion, run the same command on each of four nodes:

```bash
bash scripts/run_selection_switch.sh
```

No seed, stage or arm flags are needed. The default queue builds five selected
prefixes, collects 18 development continuations, freezes the gate automatically,
then enables 30 held-out continuations. Test prefixes can be trained before
gate fitting; their continuation rewards cannot enter fitting.

```bash
bash scripts/run_selection_switch.sh status
bash scripts/run_selection_switch.sh live
bash scripts/run_selection_switch.sh export
bash scripts/run_selection_switch.sh why
```

`export` and `why` write text files and print their full paths. Defaults:

- Results: `/group-volume/minsoo3.kim/offpolicy-misranking/runs/selection-switch-v1`.
- Logs: `logs/launcher.<host>.log` under that result directory; phase logs
  reside beside the branch's `cost.jsonl` and `progress.json`.
- Reports: `/group-volume/minsoo3.kim/offpolicy-misranking/reports/selection-switch`.
- Override storage with `OM_WORK` or `SWITCH_ROOT`, Python with `SWITCH_PYTHON`.
- `selection-switch-v1` is this experiment's first protocol, not a new paper
  version, repository, or replacement for the old v1/v2/v3 experiments.

Rerun the same command to resume. Completed branches are skipped. A failed
task is attempted at most once per invocation; other eligible tasks continue.
It never kills another E5/Qwen/net-gain process or bypasses an occupied node.
Waiting for other owners is bounded at ten minutes without local progress,
then the launcher exits rather than looping indefinitely. Rerun it to rejoin.
SIGINT/TERM stops the current worker tree. A hard kill with an unclosed cost
event requires an audited recovery: unknown failed work is not reset to zero.

## What is frozen

Sources are completed base-policy MATH points for seeds 0--4 in the original
OLMo matrix. Initial subsets use the verified `fresh_r` score and E5 tie rule.
Initial score generation is historical shared work with unknown historical
cost, not free deployment scoring. New prefix training costs are recorded.

Preparation compatibility fix (2026-09-14): the first launcher rejected legacy
`oracle_protocol.json` metadata at `selection_switch_gpu.py:145`, even after
live rollout validation. If the saved protocol lacks the current validation
record/schema, preparation now reconstructs the exact fresh_r scalar scores
on CPU from `oracle_micro_groups.pt` and `val_groups.pt`. It verifies shapes,
prompt coverage, finite values and any surviving recorded input hashes, and
records the recovery source hashes. No original matrix artifact is overwritten,
no response or gradient is regenerated, and no unverifiable old scalar score
is silently accepted. If these saved gradients are absent, the error names
the exact required paths. Preparation failures now enter the launcher log;
`why` and `export` also work before `switch.json` has been published.

Prefix segments preserve the same initial selected subset and full optimizer
lineage: 0 to 25, 25 to 50, 50 to 100. A generic drift checkpoint is rejected.
Dedicated read-only input views reference the original pool and new prefix;
they do not modify matrix checkpoints or claim those checkpoints were selected.

Fresh_r renewal samples eight candidate responses, computes two four-response
LOO gradient groups, averages the vectors, and takes their cosine with the
ranking-validation direction. It uses only the first half of the original
validation pool. This matches `experiment.score_oracle_microgroups(...)[1]['r']`;
it does not average individual cosines or substitute a low-order estimator.
Independent A/B diagnostic gradients are not computed for selection. Gradients
use the original projection/layer definition, micro-batch one, eval-mode
activation checkpointing and durable per-prompt partials.

At each branch state, all decisions are frozen before any continuation starts.
The diagnostic scans cached rewards and the last 20 completed prefix updates
once. Its charge is assigned once to GATE, CONTINUE_D and SWITCH_D. A failed
test diagnostic produces a charged random fallback and still permits the
fixed controls; a failed development diagnostic cannot become a training label.

The gate is development-only standardized ridge regression, alpha=1, on four
registered inputs. CONTINUE when predicted paid-control reward difference is
positive; SWITCH otherwise. The checkpoint-only ablation is fit separately.
GATE trains and evaluates its own policy; no control result is copied into it.
This is a state-specific decision test, not a globally optimal stopping-time
claim or a repeatedly invoked per-epoch controller.

## Allocation and reporting

Unless explicitly supplied during first preparation, the common branch cap is
100 updates times four GPUs times the median original development seed-0 d100
update duration, rounded up to 60 GPU-seconds. Only timing, not reward, enters
this convention. The cap and timing-file hash are frozen in `switch.json`.
It is an experimental allocation, **not a wall-time completion estimate**.

Use `--budget-gpu-seconds N` on the first `prepare`/`smoke`/`run` invocation to
override it before outcomes. Later conflicting preparation flags are rejected.
Scoring, verification, training and failed attempts consume the branch cap.
Evaluation has an identical separate reporting allocation. Prefixes, initial
cached scores, preparation and fitting are research overhead, separately
identified; fitting records occupied GPU allocation when run on an admitted node.

`summarize` writes development and held-out reports plus observed-result plots.
Missing/invalid states stay listed. Reports include both directional decision
losses, gate-minus-random/continue rewards, per-question paired contrasts,
actual versus intended action, fallback status, completed updates and ledgers.
Plots retain visible seed marks and zero references; no connected curve is
presented as an executed switching trajectory. The initial two test seeds are
not enough for a strong population-level guarantee. Five-test-seed expansion
requires separately registered source support; seeds 5--7 are not silently
invented or replaced with existing seeds.

## Verification

CPU tests exercise exact fresh_r score equality, 12/13-prompt validation-shard
weighting, four-feature ridge fitting, split leakage, decision-before-control
ordering, diagnostic failure fallback, real gated evaluation calls, shared
task exclusion and all 48 queued continuations with mocked GPU phases.
Related legacy gate/ledger/process tests are run as regression checks.
Local verification: 111 tests passed, including 38 new switch/plot tests;
Bash syntax and staged whitespace checks passed. Figure checks rendered
synthetic test fixtures only, not invented manuscript results.

```bash
bash scripts/run_selection_switch.sh cpu
```

Use the existing experiment environment with torch, NumPy and pytest. Plotting
also requires matplotlib. Local tests use CPU torch; they do not establish
H100 memory fit, real generation throughput or cluster filesystem behavior.
GPU smoke validation and actual experiment outcomes remain outstanding. Existing
jobs and the manuscript's accepted experimental numbers are not changed.
