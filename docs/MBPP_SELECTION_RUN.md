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
bash scripts/check_mbpp_storage.sh             # saves <=4 KiB TXT in home; never starts training
bash scripts/run_mbpp_experiments.sh plan       # settings only, no writes/GPU work
bash scripts/run_mbpp_experiments.sh check      # read-only local input checks
bash scripts/run_mbpp_experiments.sh status
bash scripts/run_mbpp_experiments.sh progress   # shared experiment/node view
bash scripts/run_mbpp_experiments.sh saved      # READ-ONLY saved-work/archived-work inventory, <=4 KiB stdout
bash scripts/run_mbpp_experiments.sh stop       # stop/clean THIS node, not peer nodes
bash scripts/run_mbpp_experiments.sh results    # one report per suite, also copied home
bash scripts/run_mbpp_experiments.sh why        # ONE diagnostic TXT, at most 16 KiB
bash scripts/run_mbpp_experiments.sh run fresh
bash scripts/run_mbpp_experiments.sh run quality
bash scripts/run_mbpp_experiments.sh run difficulty
```

## Nodes Joining And Failing

Before `run` or `restart` can enter the controller (including stopping an old
controller during restart), the wrapper now runs `check_mbpp_storage.sh`.
The controller repeats the check before every recovery/queue pass, including
in-place automatic code reloads. A live task lease defers mutable checkpoint
inspection so a peer's in-progress checkpoint is not mistaken for file loss.
Missing work storage, lost result payloads with surviving seals/DONE log lines,
archived completed results, incomplete checkpoint metadata without a complete
alternative, and successful training costs with missing final policy/stop
records block startup with exit 2. `stop` and read-only commands remain available.
An absent root is not labelled "deleted"; if none of the requested existing run
manifests can be found, recovery refuses to initialize replacement runs silently.
Unprepared quality/difficulty roots are allowed alongside the existing fresh root.

The separate audit saves a unique `~/mbpp-storage-*.txt` (at most 4 KiB), prints
its full path, and prints exact configured roots. Send that TXT file for diagnosis;
no archive, model or rollout upload is needed. A blocked audit still saves its TXT
and exits 2. Automatic startup/pass checks save a file only when blocked, avoiding
report accumulation during successful queue passes.
It distinguishes active saved
results from `discarded/`, waiver and explicit-reset receipts. It reads bounded
metadata and short historical log tails, not model/optimizer/rollout payloads,
and changes no experiment files. It cannot prove who deleted a file or detect
historical deletion with no remaining evidence. Tensor hashes and checkpoint
lineage still require trainer validation; passing metadata checks is not that
certification. Already-running workers are not stopped by this read-only audit.

Published training with a missing convergence curve is `EVAL`, not `READY`:
only evaluation remains. `RESUME` means a checkpoint candidate exists and must
pass the trainer's original hash/contract validation; it is not a fresh start.
`REVIEW` means saved training evidence remains without a complete checkpoint
candidate. The trainer refuses to fall back to parent weights if all local
checkpoints fail validation, and automatic waivers never archive saved training
or completion evidence. These protections preserve existing files and costs;
they do not restore files already missing or certify damaged checkpoints.

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

The MBPP hold defaults to 15 seconds (`MBPP_HOLD_SECONDS`), with
`EXPERIMENTS_HOLD_SECONDS` taking precedence within the enforced 1–60 second
range. This cap also applies to inherited settings after an in-place code reload,
idle passes, busy locks, and GPU-error retry backoff. Polling is at most 5 seconds;
dependency-free failed/stale tasks wake a hold as well as READY tasks, and peer
completion ends it early. The countdown includes time spent checking status.
Pending siblings do not erase a failed pass's retry state.

A first recorded MBPP GPU fault now waits 60 seconds, then must pass the existing
NCCL/CUDA admission probe before any training resumes. An explicit
`EXPERIMENTS_FAULT_TTL_SECONDS` still overrides that cooldown, but expiry is not
followed by another long exponential delay. A second recorded GPU fault or an
invalid receipt blocks admission; two blocked passes release the controller
instead of holding indefinitely. Legacy receipts without `time` expire from
their file modification time, not from the time they are read. Fault receipts
are atomically published and never cleared by MBPP `run` or `restart`.
Zero/invalid poll intervals are rejected. Setting
`EXPERIMENTS_HELP_SIBLINGS=0` explicitly restricts a node to the first root;
leave it enabled to serve all requested suites.
Terminal launches detach as before: Ctrl-C stops the log view, not the workers.
Use `stop` to stop this node's MBPP controller and its token-bound children.
After pulling an update, `bash scripts/run_mbpp_experiments.sh restart` stops
this node's verified MBPP controller and starts it with the new code in one
command. Valid checkpoints, selections, completed results, and fault receipts
remain on disk. Ordinary `run` still leaves a live controller alone.
**A busy lock no longer triggers a node-wide process or GPU sweep.** Other
experiments are not cleanup targets merely because they use the same account,
work volume, or GPUs. MBPP recovery/watchdog roots exclude unrelated math and
MoPPS roots. PID files are checked for an actual MBPP launcher before signalling.
Existing untagged legacy workers cannot be safely adopted as dead just because
they hold a lock; their ownership must be checked rather than deleting locks.
Completed outputs and cost ledgers are not reset by this ownership fix.

The holding regression audit traced the behavior through `d52ed90` (fault TTL),
`d59d6fd` (separate cooldown exit), `32813f0` (READY-only wakeup), `888e99d`
(duplicate-controller guard), and `e0f091d` (ordinary-failure-only hold cap).
CPU controller tests cover all five pass outcomes (0/1/75/78/79), legacy 600s
settings, cooldown expiry and invalid receipts, failed/stale wakeups, and
checkpoint-preserving restart. They do not certify the health of a live GPU node.

`why` now writes one attachment across the requested suites, capped at 16 KiB.
It includes the newest two saved failures per suite, selection/publication file
presence, the original CUDA/NCCL warning context, the latest node admission and
two short MBPP node-console tails. File presence is not a hash-validation result,
and a saved failure is not proof that the current retry is failing. Full rollouts,
model data, cost ledgers and repeated full-suite exports are excluded. Existing
experiment logs are only read, never truncated, deleted or repaired.

### Selection retries and saved work

The controller no longer automatically waives/resets a failed `fresh-r-*`
selection stage. Selection can save rollouts and per-prompt gradients before
there is any training checkpoint; "no training checkpoint" is not evidence
that the selection accomplished nothing. Both the saved work and its charges
stay in place. With allocation remaining, completed shards are skipped, rollout
collection resumes from complete prompt groups, and hash-bound prompt gradients
are reused. `fresh-r-candidate` covers both rollout and gradient work; the phase
label alone does not mean all candidates are being regenerated.

An exhausted allocation is rejected before metering another attempt, reported
as `BUDGET` rather than an ordinary retryable `FAIL`. It does not get a new budget,
silently switch selectors, or count as a completed experimental result. Existing
publication recovery is still allowed. The runtime migration preserves the
frozen `switch.json`, prior migration receipts, decisions, costs and policies.

`saved` prints active checkpoint/adapter presence, partial selection counts,
archived attempts under `discarded/`, and budget usage for two failed branches
per suite. It is at most 4 KiB, opens no model/rollout contents, writes nothing,
and starts no controller. Presence is not integrity/resume certification.
Archived selection work whose costs were waived cannot be restored for free;
its provenance and the corresponding charges require review first. Do not run
`reset-waived`, remove experiment directories, or increase a frozen allocation
to make an exhausted branch appear successful.

These fixes prevent repeat reset/retry damage; they cannot certify or restore
files on an unmounted GPU server. Keep the original root and its `discarded/`
and `waivers/` directories for recovery inspection.

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
