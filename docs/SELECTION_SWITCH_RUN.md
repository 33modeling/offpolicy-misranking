# Selected-prefix switching experiment

Implemented: 2026-09-14. Manuscript target: v4. No GPU result is claimed here.

Incident history, repair commits, verification limits, and unresolved mid-run
interruptions are tracked in the dedicated
[bug-fix log](SELECTION_SWITCH_BUGFIX_LOG.md). A locally tested fix is not a
verified successful cluster run.

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
prefixes, collects 18 development continuations and freezes the gate
automatically. Held-out states (seeds 3 and 4) are published as soon as their
prefix exists: the 24 control arms (selection/random, full/reduced) run
immediately, and only the 6 GATE arms wait for `model.json`. Each held-out
state binds the fitted gate in `states/s{seed}-t{step}/gate.json`; the control
decisions are frozen in `decisions-frozen.json` before any control starts, and
the GATE decision is frozen in `gate-frozen.json` from that same shared
diagnostic and the bound model, so control outcomes cannot change it. Held-out
rewards never enter gate fitting.

```bash
bash scripts/run_selection_switch.sh status
bash scripts/run_selection_switch.sh status --watch 5
bash scripts/run_selection_switch.sh live
bash scripts/run_selection_switch.sh errors
bash scripts/run_selection_switch.sh check-code
bash scripts/run_selection_switch.sh export
bash scripts/run_selection_switch.sh why
```

`status` from either launcher (or `bash scripts/run_experiments.sh status`) is one
read-only screen for both experiments: the selection-switch view below, then the
MoPPS comparison view, then this node's GPUs once. `--all`, `--json` (one object
with `selection_switch` and `mopps_comparison`) and `--watch [seconds]` apply to
the whole screen; `EXPERIMENTS_COMBINED=0` shows only this experiment. The
selection-switch view shows
active/waiting/stale nodes, prefix and continuation completion counts, the current
worker on each node (PID, phase, elapsed time, limit and heartbeat age), five
prefix rows, a 15-state continuation grid, and concise failures requiring attention.
The prefix table distinguishes the last logged training step from a published
prefix certificate. Waiting branches name their first unmet dependency, including
a failed or stale prefix and, for GATE arms only, a pending development gate.

`why` from either launcher (or `bash scripts/run_experiments.sh why`) writes one
report under `reports/experiments/` for both experiments: the combined status
screen, then each experiment's own why report; `EXPERIMENTS_COMBINED=0` writes
only this experiment's report under `reports/selection-switch/`.

`bash scripts/run_switch_long.sh` runs the longer-horizon variant: a separate
root (`runs/selection-switch-long-v1`) that imports the five certified prefixes
and the evaluation set of `runs/selection-switch-v1` (`prepare --prefix-source`)
and gives every continuation three times the allocation (87,120 GPU-seconds, a
300-update equivalent), so fresh gradient scoring is about a quarter of a branch
instead of four fifths. It has its own development labels, gate, ledgers and
status; the MoPPS pass is skipped. Same modes as the switch launcher.

`bash scripts/run_switch_difficulty.sh` and `bash scripts/run_switch_hard.sh`
run the cached-selector variants: the same prefixes, states, evaluation set and
29,040 GPU-second allocation (`prepare --prefix-source`), but the selection
arms rank the pool from the pre-continuation reward cache instead of
rescoring (`prepare --selector difficulty|hard`, recorded in `switch.json`
and every state contract and protocol). `difficulty` keeps the 10% of prompts
closest to a 0.5 cached success rate, with the diagnostic's tie-break;
`hard` keeps the 10% with the lowest cached success rate among prompts solved
at least once (never-solved prompts give GRPO no signal). Selection is
charged as a metered read (`<selector>-select`, a few GPU-seconds), so the
selection arms complete about as many updates as random and the gate's
decision rests on the per-update gain. The shared diagnostic freezes the same
ranking in `selection.json` next to `measurement.json`, and a paid arm's
selection must match it. Roots are `runs/selection-switch-difficulty-v1` and
`runs/selection-switch-hard-v1`, each with its own labels, gate and ledgers;
the MoPPS pass is skipped. Same modes as the switch launcher.

`bash scripts/run_switch_mbpp.sh` runs the code variant: the same protocol on
the OLMo MBPP matrix family (`family-mbpp-s<seed>`, 512-prompt pool, top 10%
= 51, execution-verified rewards) in its own root
(`runs/selection-switch-mbpp-v1`). `prepare --dataset mbpp` resolves the five
d0 sources from that family, derives the branch budget from the MBPP seed-0
d100 update timings, builds the evaluation pool from `$DATASETS_DIR/mbpp/mbpp.jsonl`
in the prompt format of the source runs, and draws up to 300 test problems
disjoint from every run's candidate and validation prompts (all remaining
ones if fewer). Contracts record dataset `mbpp` and verifier
`code_execution`; MATH roots are published through the unchanged MATH path.

`waive` (from either launcher) returns the allocation of attempts that stalled
after a GPU fault: a rank that dies with a CUDA "unspecified launch failure"
leaves the trainer hung until the phase's allocation limit, the whole
allocation is charged, and every retry ends with "branch allocation exhausted
before a valid checkpoint". The waiver moves that attempt's ledger lines to
`cost-waived.jsonl`, writes a receipt under `waivers/` with the fault line, and
removes `failure.json` so the next pass retries the branch. It also moves the
attempt's `policy/`, `evaluation/`, progress and result files to
`discarded/<utc>/`: the trainer resumes from checkpoints in its output
directory, so a retry that found them would add the discarded attempt's
updates to its own allocation. It refuses branches with a published result,
branches whose phase log shows no fault, and branches a worker holds.
`reset-waived` handles branches waived before that discard existed (their retry
resumed the discarded checkpoints and exceeded the allocation): it appends the
branch's whole ledger to `cost-discarded.jsonl`, moves the outputs aside, writes
a receipt under `discards/`, and leaves the frozen decision, execution record
and subset, so the queue reruns the branch from the state's parent policy.
Stop the node running such a branch first; a held branch is skipped. The node launcher also runs a stall watchdog: a training phase
whose worker logs are silent for `EXPERIMENTS_STALL_SECONDS` (default 1500) is
terminated, the attempt is charged for those minutes only, and the host is
recorded under `runs/experiments/node-faults/`, after which both launchers
refuse GPU work on it (rc 78) and the node launcher releases it.

`status --watch 5` refreshes every five seconds; Ctrl-C stops only the status
viewer. `--all` adds per-task paths and reasons, and `--json` exposes the snapshot
for scripting. A running heartbeat older than 60 seconds is STALE, not RUNNING
or READY. Waiting-node counts are inferred only from fresh launcher log entries;
nodes without recent evidence are not assumed alive. A failed held-out diagnostic
does not falsely block the registered random fallback.

Status does not acquire task locks, replay cost receipts, create runtime migration
files, load models, or validate large source/weight artifacts. DONE in this view
means that the published result and its receipt match; scientific result validation
still belongs to the reporting workflow. Invalid JSON is reported locally without
hiding the other nodes. This status-only update does not change frozen runtime
code, training, budgets, or scheduling, and does not require restarting workers.

`errors` prints the latest three recorded failures and their actual worker log
tails directly to the terminal, without model loading, GPU allocation, hash
migration or changes to run artifacts. For a `prefix-train worker failed`
wrapper exception, use `errors --phase prefix-train`. The launcher's line number
identifies the process supervisor, not the underlying training exception. A
worker failure now includes the failed child's log tail in the raised exception
immediately, including when other queue tasks continue afterward. A
failed launcher also prints these tails before exiting; it preserves its original
nonzero exit status. Use `--limit` and `--lines` to adjust the diagnostic output.
Without `--phase`, it also shows the latest bounded launcher log tails, including
controller exceptions that have no task `failure.json`. New launcher start/exit
markers include revision, PID and exit code. A missing exit marker does not prove
normal completion; hard kills cannot reliably write one.

`export` and `why` write text files and print their full paths. Defaults:

- Results: `/group-volume/minsoo3.kim/offpolicy-misranking/runs/selection-switch-v1`.
- Logs: `logs/launcher.<host>.log` under that result directory; phase logs
  reside beside the branch's `cost.jsonl` and `progress.json`.
- Reports: `/group-volume/minsoo3.kim/offpolicy-misranking/reports/selection-switch`.
- Override storage with `OM_WORK` or `SWITCH_ROOT`, Python with `SWITCH_PYTHON`.
- `selection-switch-v1` is this experiment's first protocol, not a new paper
  version, repository, or replacement for the old v1/v2/v3 experiments.

When started from a terminal, `run` and `smoke` detach themselves first: the
controller runs in its own session with `logs/console.<host>.log` as its console,
so a dropped phone or SSH session cannot kill it (the 2026-09-13 queue incident:
a dead terminal killed the controller and left GPU ranks running). The terminal
only follows that console; Ctrl-C ends the view and the run continues. Use
`bash scripts/run_selection_switch.sh stop` on that node to stop it: it sends
TERM to the detached session, the worker reaps its GPU ranks and closes cost
receipts, and the command waits up to three minutes for the exit. Starting
`run` again while a detached launcher is alive only prints its PID. Without a
terminal (tests, pipelines) or with `SWITCH_FOREGROUND=1`, the launcher keeps the
direct foreground behaviour, where Ctrl-C stops the worker as before.

Node admission (`scripts/selection_nccl_preflight.py`) runs a tiny four-rank NCCL
probe before any task is claimed. A CUDA 802 "system not yet initialized" failure
means single-process CUDA works but the NVSwitch fabric is not ready, so the probe
is retried with one fabric-dependent transport disabled at a time:
`NCCL_NVLS_ENABLE=0`, then `NCCL_CUMEM_ENABLE=0`, then `NCCL_P2P_DISABLE=1`
(shared-memory transport). Only the overrides of the probe that passed are
exported to the training workers; settings already present in the environment
are never changed. If the failure persists through the whole ladder the node is
refused (exit 78) with the administrator diagnosis and nothing is recorded
against any branch.

`bash scripts/run_selection_switch.sh recover-cost --stale` (also for the MoPPS
root via `run_mopps_comparison.sh recover-cost --stale`) closes open cost events
whose owner has shown no life for 15 minutes. Operator decision, 2026-09-15:
hard-killed attempts never write a finish receipt and this cluster kills jobs
routinely, so such events are closed from evidence instead of blocking a branch
forever. If an atomic finish receipt exists it is used; otherwise the charged
duration is the last observed evidence of the attempt (meter heartbeat or the
ranks' phase-log writes, capped at any later attempt's start) minus the recorded
start, plus a 60-second margin so the estimate over-counts the interrupted
attempt rather than under-counting it. The evidence is recorded in the ledger as
`stale_owner_last_evidence`. Live local owners and recent evidence are left
untouched; without `--stale` the command only lists open events.

The allocation on this cluster lives only while the launcher process does, so
an exit after "no claimable task" or a failed pass used to give the GPUs back
and force a new allocation plus the same command again. In `run` mode an
operator launch (detached or on a terminal) now keeps the node: after each
worker pass it re-checks completion (18 development and 30 held-out
continuations published), otherwise prints `[hold] pass N ended rc=...` and
starts the next pass after `SWITCH_HOLD_SECONDS` (default 600; doubled up to
3600 after a failed pass, reset after a clean one). Every pass re-admits the
node through the NCCL probe and re-attempts failed tasks once. Node admission
failure (78) and stop signals still end the launcher; `stop` works during the
hold. Callers without a terminal (tests, pipelines) keep the single pass;
`SWITCH_HOLD_SECONDS=0` forces it anywhere. MoPPS `run` holds the same way,
checking for its 12 published continuations.

Holding is not enough on its own: the cluster also reclaims an allocation whose
GPUs sit idle, which is what waiting or holding looks like. Operator launches
therefore start `scripts/_gpu_keepalive.py` right after the occupancy check and
keep it for the launcher's lifetime: a tiny fp16 matmul on every visible GPU a
few times a second (a few percent of utilisation, well under 1 GB per device),
logged to `logs/keepalive.<host>.log`, killed on exit and by `stop`. It is not a
metered cost event and never touches the run directory. `SWITCH_KEEPALIVE=0`
disables it; `SWITCH_KEEPALIVE_PERIOD` (seconds, default 0.25) sets the pace.

From a terminal, a plain `bash scripts/run_selection_switch.sh` (or the MoPPS
launcher) now hands the node to `scripts/run_experiments.sh`: every cycle it
closes stale cost events of both roots, runs one switch queue pass, then one
MoPPS pass (which retries recorded failures first), and holds the node with the
keepalive for `EXPERIMENTS_HOLD_SECONDS` (default 300) before the next cycle.
Nodes are not assigned to an experiment; whichever has claimable work gets it.
`bash scripts/run_experiments.sh stop` (or either launcher's `stop`) stops the
node; `status` shows both views. `EXPERIMENTS_COMBINED=0` runs one queue only,
`EXPERIMENTS_AUTO_PULL=1` pulls before each cycle.

Rerun the same command to resume. Completed branches are skipped. A failed
task is attempted at most once per invocation; other eligible tasks continue.
It never kills another E5/Qwen/net-gain process or bypasses an occupied node.
Waiting continues while locked tasks have fresh running heartbeats from peers,
even after ten minutes without local work. Without an active peer or local
progress, the ten-minute idle limit still applies. Rerun the launcher to rejoin.
State publication now uses a nonblocking task lock: a node skips a state that
another node is preparing and continues through other ready states and prefixes.
Missing development labels do not contend for the fitting lock. `waiting`
lists occupied tasks and, when a fresh heartbeat exists, their host, PID and
phase. `[claimed]` identifies the node and task that actually started. It means
the scan found no claimable task; it does not mean parallel execution is disabled.
Same-seed prefix segments remain sequential; different seeds and ready branches
run concurrently. Failed-task counts are printed separately from active peers.

SIGINT/TERM stops the current worker tree. Phase startup is covered by cleanup,
and an atomic `cost-events/<event-id>.json` completion receipt is persisted before
the finish is appended to `cost.jsonl`. On resume, `spent()` replays a matching
receipt under the cost writer lock. It does not infer completion from an old
heartbeat. The exact `cb01401`, `a63e69d`, `96ad9ed`, `bb32da3`, `69bec8d`,
and `4798f93` runtimes are accepted with added runtime bindings. `7dc108a` and
`857c3fa` have the same switch code hashes as `4798f93`. Existing manifests,
runtime receipts, policies and cost journals stay unchanged. The original
`cb01401` runtime was previously missing from the compatibility list, so valid
experiments frozen by that version could incorrectly fail at the code check.
The fix records `code-compat-runtime.json`; it never resets the run or treats an
unknown deployment cost as zero. An already frozen `857c3fa` MoPPS sidecar is
also accepted with its own receipt, without changing the source switch run.

`check-code` is a read-only CPU preflight that prints the frozen and current code
fingerprints and changed file names. It requires no GPU, node lock, environment
bootstrap or run writes. The run/smoke launcher invokes it before node admission.
Unknown scientific changes still stop execution, with the differing file names
and hashes in the error. Do not edit `switch.json` or remove receipts to bypass
that check; provide the `check-code` output and complete traceback for diagnosis.

For the first runtime-isolation deployment, stop old live-checkout launchers
gracefully and confirm their worker trees have stopped, then pull and relaunch
on each node. Do not leave old-version controllers reading the changing checkout.

Run/smoke/prepare/fit/summarize now enter an isolated detached local clone before
GPU admission or preparation. `[runtime] commit=... pinned=...` identifies it.
Controller and subprocess code, including `OM_REPO` and Python imports, stay on
that revision even if the original checkout is updated later. The existing
`SWITCH_ROOT`, input/model paths, Python environment and budgets are unchanged.
The scientific code hash map did not change in this launcher-only repair.
Status, errors, check-code and cost inspection do not create runtime clones.

The default cache is `/tmp/offpolicy-misranking-<uid>/switch-runtimes/<commit>`;
`SWITCH_RUNTIME_CACHE` can select another node-local path outside the repository.
Do not edit or delete a runtime cache while its workers are alive. Dirty source
or a modified cached runtime is rejected without discarding changes. New
scientific revisions still need the frozen-run compatibility checks; pinning is
not permission to mix different algorithms across nodes.

An interrupted **research prefix** no longer blocks training merely because its
historical duration is unknown. Its seed/task lock and cost writer lock must be
free, and a recorded local owner must not still be alive. The old start and last
progress are preserved in `pending-costs/<event-id>.json` before a new phase can
replace `progress.json`. Training resumes from its existing checkpoint machinery;
completed prefix policies are reused. The original cost event stays open, never
zeroed or assigned a guessed duration. Status and reports expose the research
total as unknown (`total_gpu_seconds: null`) until evidence closes the event.
This exception does **not** apply to deployment/measurement costs, whose exact
accounting is still required to enforce the frozen continuation budget.

For an older interrupted phase without a completion receipt, inspect its record
without allocating GPUs:

```bash
bash scripts/run_selection_switch.sh recover-cost
```

This lists open event IDs, relative directories, and matching progress records,
including archived prefix progress even after another phase has run.
After confirming that the affected job has stopped, close one event using its
elapsed seconds from the termination/scheduler log:

```bash
bash scripts/run_selection_switch.sh recover-cost \
  --directory states/s0-t25/points/view-25/selection_reduced \
  --event-id EVENT_ID --seconds ELAPSED_SECONDS --reason 'termination log reference'
```

Use the directory and event ID from the listing. An existing completion receipt
can be replayed with just `--directory` and `--event-id`. Legacy recovery requires
a stated duration and evidence; the last heartbeat is only a lower bound.
Recovery refuses occupied task/cost locks, a still-live local legacy owner,
underreported duration, changed allocation, and already published cost totals.
It appends a failed completion with evidence and the original ledger hash;
prior records, partial outputs and budgets remain in place. Repeating recovery
does not charge the event twice. Failed work is not reset to zero.

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

KV-cache repair (2026-09-15): both teacher-forced scoring paths now pass
`use_cache=False`. With the model's default cache enabled, eval-mode decoder
checkpointing could append to the same mutable KV cache again during backward,
causing attention-shape or checkpoint-recomputation failures. Generation keeps
its original cache settings. The CPU suite now exercises the actual tiny OLMo3
architecture with cache enabled, merged adapters, frozen early layers, both
logit paths, gradient equivalence, and generation after scoring.

The exact pre-fix runtime from `a63e69d` can resume with the same command. A
`kv-cache-runtime.json` receipt binds the original manifest and patched code;
the frozen protocol, saved prefixes, rollout partials and cost ledgers retain
their original contents. Unknown code changes are rejected. Failed work stays
charged, so an exhausted branch budget is not restored by this repair. The
receipt is included in `why` and `export`.

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

For the separately registered **MoPPS (KDD 2026)** online reward-selection
comparison, see [MOPPS_COMPARISON.md](MOPPS_COMPARISON.md) and
`scripts/run_mopps_comparison.sh`. It adds 12 held-out continuations in another
output root, imports existing prefixes read-only, and does not alter this
running experiment's arms, gate fit or frozen scientific runtime.

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
