# MBPP Selection Experiments

Run this same command from the original `offpolicy-misranking` checkout on
**every allocated four-H100 node**, including replacement nodes:

```bash
bash scripts/run_mbpp_experiments.sh
```

The command joins the existing node controller with one default experiment,
reusing the existing quality root and its validated training work:

| 표시명 | 선택 방식 | 분기 예산에 차감하는 비용 | Gate 판단 기준 |
| --- | --- | --- | --- |
| On-policy · 선택비용 별도 | 현재 정책에서 계산한 gradient | 진단 + 학습; 선택 비용은 별도 계측 | 비용 보정 학습 효율 |

Default `run`/`all` schedules `quality`, with frozen settings
`fresh_r/matched/convergence` (selector/accounting/gate). This applies the
paper's **Separating learning quality from selection cost** protocol to MBPP.
It does not force a MATH GPU-second number onto MBPP or change an existing cap:
the MBPP calibration and frozen root contract remain authoritative.

On-policy gradients are computed under the current policy. “선택비용 별도”
means selection GPU time goes to the `reporting` ledger outside the common
diagnostic/training allocation. Selection is **not free**: report actual
selection, diagnosis, training and evaluation costs separately and include all
four in total compute. Equal training-allocation caps do not mean equal total
GPU cost or identical completed update counts. The convergence gate uses
updates saved to a common reward target minus the separately recorded selection
cost expressed in random-training update units.

This is a deliberate accounting difference from the paper's MATH fixed-total-
budget comparisons. An MBPP learning-quality advantage under this protocol
would not by itself establish a total-compute advantage or validate a direct
on-policy-to-difficulty switch.

The older `fresh` (cost-inclusive on-policy), `difficulty` and `long` conditions
remain available only through explicit execution commands. Their saved files,
historical results and active-node evidence are retained; no root is renamed
or deleted. A code update does not kill their in-flight work, and an explicitly
scoped controller can continue until its requested scope is changed. This guide
does not claim any remote controller has already stopped.

The main condition includes random controls and reuses the five certified
on-policy-selected MBPP prefixes from the original `fresh` root, at 25/50/100
updates, and the same MBPP evaluation questions. It does not import MATH prefixes
or relabel cost-inclusive continuation results as quality results. Existing
quality training is resumed and validated, not copied to a new experiment root.
There are **48 planned branches: 18 development and 30 held-out**, with
development seeds 0/1/2 and held-out seeds 3/4. Final evaluation uses eight
responses per question; convergence curves evaluate three archived checkpoints
with four responses per question. Prefix availability remains a prerequisite.

This is the existing matched-training-budget switch protocol, not an E5
fixed-checkpoint run or a gate that directly chooses on-policy versus difficulty.
The gate compares on-policy selection with random.
The display names do not rename existing CLI keys (`fresh`, `quality`, `difficulty`),
saved directories, scoring phase names, or any frozen protocol fields.

### Result tables

For the main quality condition, compare **Full selection, Full random and Gate
policy** at matched seed/checkpoint states using their **final held-out reward**.
The gate's development target comes from the additional convergence measurements.
Report valid-result count / planned count, completed training updates, measured
reward, diagnostic/selection/training GPU time, and evaluation GPU time separately.
Retain the additional diagnostic-paid Selection/Random controls: 18 development
and 30 held-out continuations, as in the registered protocol. Results from the
older cost-inclusive, difficulty and long conditions belong in separately
labeled blocks outside the main condition's 48-branch denominator.
Include interrupted and failed attempts in an explicit cost/failure audit.
An unevaluated budget-exhausted branch has no measured reward: show “미완료”,
not zero. If no valid paired outcomes exist, a runtime/cost table is possible,
but there is no completed selection-versus-random performance comparison.
This scope update supplies no new GPU measurements; unfinished branches must
not be presented as completed runs or assigned invented rewards.

## Commands

```bash
bash scripts/check_random_storage.sh           # ALL discovered suites: RF/RR/RO separately; <=4 KiB TXT in home
bash scripts/check_mbpp_storage.sh             # saves <=4 KiB TXT in home; never starts training
bash scripts/run_mbpp_experiments.sh plan       # settings only, no writes/GPU work
bash scripts/run_mbpp_experiments.sh check      # read-only local input checks
bash scripts/run_mbpp_experiments.sh status
bash scripts/run_mbpp_experiments.sh status --watch
bash scripts/run_mbpp_experiments.sh progress   # main quality progress and retained saved-work evidence
bash scripts/run_mbpp_experiments.sh saved      # READ-ONLY saved-work/archived-work inventory, <=4 KiB stdout
bash scripts/run_mbpp_experiments.sh stop       # stop/clean THIS node, not peer nodes
bash scripts/run_mbpp_experiments.sh results    # one report per suite, also copied home
bash scripts/run_mbpp_experiments.sh why        # ONE diagnostic TXT, at most 16 KiB
bash scripts/run_mbpp_experiments.sh run quality       # same main condition as the default
bash scripts/run_mbpp_experiments.sh status quality    # main condition only; no training
bash scripts/run_mbpp_experiments.sh results quality   # main results; no training
bash scripts/run_mbpp_experiments.sh run fresh         # optional old cost-inclusive condition
bash scripts/run_mbpp_experiments.sh run difficulty    # optional cached-selector condition
bash scripts/run_mbpp_experiments.sh run long          # optional old longer-budget condition
```

For a single MBPP status snapshot, use `bash scripts/run_mbpp_experiments.sh status`.
Add `--watch` only when continuous refresh is wanted. The view shows the main
quality experiment and retains observation of earlier
MBPP roots and their nodes; observation is not permission to schedule them.
The view includes all seed/step
rows and the full names Selection, Random, Full selection, Full random and
Gate policy. The top line shows planned / verified completed / remaining counts
for the requested roots, and each condition has the same numeric columns.
Each registered condition has 48 continuation branches; prefix, selection and
evaluation phases are not extra experiments. Remaining means planned minus
verified completed, including unverified records. Progress uses the same fixed
planned denominator, not elapsed time or just the subset of records found.
Display states are `READY`, `DONE`, `WAIT`, and `RUN`; Remarks
explains evaluation, checkpoint validation, budget exhaustion, and failures.
Numbered full node names map to their experiment and phase. Inactive node
history is hidden, not deleted. A missing or unreadable condition shows
planned 48 / verified completed 0 / remaining 48, with `WAIT` and
“미확인 48개”. This does not claim its past work disappeared or must restart.
The totals describe the current view, not a requirement to run every optional
condition; viewing status does not change the execution queue.
The `NODES ... current` table pads the Node, Experiment, Status, Progress and
Remarks columns using terminal character widths, including Korean text. Long
experiment/remark text wraps inside its own column. If the terminal is too
narrow for a full node name and useful work columns, the complete node name
appears above its aligned work row instead of being truncated or mixed into
other columns.
Unassigned nodes do not all mean `WAIT`: `CHECK` is pre-task validation or
queue inspection, `ADMIT` is device admission, `COOL` is GPU-fault cooldown,
and `BLOCK` / `FAIL` / `EXIT` show blocked, failed or exited controllers.
The Remarks column preserves the controller exit code and the most recent
failure reason even when cleanup messages follow it. Code 80 means remaining
work requires checkpoint review; it is not experiment completion. The compact
`why` attachment preserves this exit reason as well.
When an interactive launch's detached controller exits, the launcher now returns
that actual exit code instead of unconditional success. Closing the log viewer
still leaves the controller and its workers running.
`status --all` adds exact roots and individual task states;
`status quality --watch 5` watches only **On-policy · 선택비용 별도**.
This is read-only and does not restart training. The generic
`run_experiments.sh status` defaults to the math-root overview, not this MBPP view.

The bottom of ordinary `bash scripts/run_mbpp_experiments.sh status` separately
lists **작업 없는 노드**; `--watch` is not required. It shows full, numbered node names
with recent explicit waiting/holding evidence and no live task in any observed
MBPP condition (including retained conditions). Admission, recovery, cooldown,
unknown and stale nodes are not called idle. This is a read-only scheduling
view, not a node-wide GPU/process inspection or automatic cleanup.

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
The main quality condition retains the shared-prefix prerequisite checks.
Observing an absent optional root does not schedule a replacement experiment.

One narrow exception lets independent work continue: an automatic startup whose
only errors are `CHECKPOINT_MISSING` in canonical MBPP continuation branches
quarantines those branches instead of blocking the entire queue. The audit still
prints `BLOCKED` and the exact affected paths; this is **not** recovered training
or a passing integrity check. Manual storage checks continue to exit 2. Missing
prefixes/checkpoints in shared prefixes, lost results, wrong storage, malformed
metadata and all other audit error types still block startup globally.

The worker rechecks the affected branch while holding its task lease, before
selection, decision work or metering. It leaves that branch in `WAIT` for
checkpoint review and continues independent branches and pending curve evaluations.
Automatic stale-cost recovery also leaves the quarantined branch's ledger and
finish receipts unchanged, including on repeated launches. Full tensor/hash and
lineage validation is still required if a checkpoint candidate becomes available.
No budget, result, denominator, checkpoint or parent-policy restart is reset.
When every other scoped task is complete and only these branches remain, exit 80
releases this node without a holding/GPU-admission retry loop. Exit 80 is **not**
experiment completion; the affected branches still require storage recovery.

The reviewed runtime upgrade preserves original manifests and previous runtime
receipts, adding `mbpp-branch-quarantine-runtime.json` with the storage guard's
hash. Shared-code compatibility for Selector Pair and MoPPS is recorded separately;
their experimental designs and scheduling are unchanged. This update does not
release an existing Selector Pair controller's exclusive lock.

For missing random-control work, `check_random_storage.sh` scans every switch and
MoPPS root under the configured work directory, not just MBPP. It separates
`random_full`, `random_reduced`, and `random_online` from selector results, lists
sealed random results, and prioritizes missing stops, archives and conflicting
parent-only stops. It saves `~/random-storage-*.txt` (at most 4 KiB), never resets
or restores anything, and does not certify tensor contents. An absent result is
not proof of deletion. Existing roots, including the active `quality` root, keep their
storage names for compatibility; no saved paths or protocol keys are renamed.

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

Published training with a missing convergence curve has internal state `EVAL`,
not `READY`: evaluation/publication remains. The MBPP dashboard shows `WAIT`
with “최종 평가 저장됨; 곡선 평가 남음 (재학습 없음)”, or `RUN` while that work
has a fresh heartbeat. This distinguishes a sealed final evaluation awaiting
its curve from a saved policy still awaiting final evaluation. Once the bound
curve is published, the next refresh shows `DONE`; old failure/heartbeat records
do not override that completion. No restart is needed to refresh this view.
Internal state `RESUME` means a checkpoint candidate exists and must
pass the trainer's original hash/contract validation; it is not a fresh start.
`REVIEW` means saved training evidence remains without a complete checkpoint
candidate. The trainer refuses to fall back to parent weights if all local
checkpoints fail validation, and automatic waivers never archive saved training
or completion evidence. These protections preserve existing files and costs;
they do not restore files already missing or certify damaged checkpoints.

The saved-work views separate archived history from current `DONE`, `EVAL`
and `RESUME` records; the MBPP dashboard uses the four display states above.
An archive alone never downgrades a valid current result.
Default status shows live nodes only, grouped by state and then node number;
inactive node logs and records remain untouched. The generic
`bash scripts/run_experiments.sh status --all` includes node history, while
`--json` always retains the full snapshot. Display names distinguish the main
quality condition from earlier cost-inclusive and long conditions; `status --all`
includes exact roots. Changing the default does not erase or replace completed results.

Saved switch points are resolved from `suite.json`, matching the worker, rather
than inferred from directory count. Ambiguous paths are `REVIEW`, not `READY`.
MoPPS checkpoints and final policies are also displayed as `RESUME`/`EVAL`
candidates, subject to runtime validation. None of these views restores missing
files or turns unvalidated work into `DONE`.

After updating, reopen any previously running status watcher once; subsequent
viewer-code changes reload only the read-only viewer, never GPU workers.
The controller logs `[dispatch]` evidence before each queue root: exact path,
loaded revision, frozen protocol, task counts, and up to three relevant task
reasons. `checkpoint_step` and `logged_step` are distinct. Metadata logging is
CPU-only, time-bounded, and cannot change worker exit codes.

A structurally consistent final policy and budget-stop record awaiting result
publication is internally `EVAL`, subject to full worker validation, not a new
`READY` training task. The MBPP dashboard explains it in Remarks. The combined
status and progress screens show RF/RR/RO random
counts separately from selector counts; display labels use on-policy while
preserving the original storage paths and CLI keys.

A missing `budget_stop.json` can now be reconstructed from an intact final
policy only after full lineage, tensor and budget-record validation. A surviving
result must also agree with the exact restored file hash. This repair does not
restart training or refund previous costs. A parent-only stop that conflicts
with saved local training is blocked without overwriting either record.
Automatic refunds require interruption evidence tied to the failed event;
an old CUDA log line or absence of a checkpoint alone is not sufficient.

`run` and `stop` use `scripts/run_experiments.sh` for the existing queue,
detached console, keepalive, watchdog, automatic Git updates, stale-event
recovery and GPU-fault waiver policy. A small MBPP ownership guard now holds
one controller lease per node and supervises shutdown; it does not schedule
another queue or change the learner, selectors, costs, or checkpoints.
The inner worker yields its idle peer-wait to this shared controller, instead
of waiting inside one suite while another has work. Owned training/evaluation
finishes normally before yielding; no active branch is interrupted for fairness.

- On the same node, re-running the plain command stops its previous MBPP
  controller and resumes from saved checkpoints, even with unchanged code.
  Use `logs` for a read-only view or `stop` to stop without restarting.
- After a controller is killed, run the same command. Its durable owner token
  identifies its own surviving children, including separately-sessioned ranks.
  Those children are stopped before the replacement starts. The node lock is
  held by the guard only, never inherited by rollout/scoring workers.
- On a new node, the same command reads the shared queue and claims available
  tasks using the existing leases. Busy branches are skipped, not duplicated.
- If a node dies, its locks are released; surviving nodes use the existing
  heartbeat/stale-cost recovery and recorded GPU-fault waiver rules before
  retrying its work. Completed results and valid checkpoints remain reusable.
- Stale-cost recovery includes the archived-checkpoint `curve/` reporting
  ledger and takes its parent branch's task lease. Shared `curve-parent/`
  recovery takes the same point lease as its evaluator. A malformed ledger is
  reported as blocked without preventing independent ledgers from recovering;
  an incomplete scan never reports all costs as known. Existing result seals,
  training allocations and saved work are not reset.
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
idle passes and recoverable retry backoff. Busy GPU/node locks and failed
GPU/NCCL admission exit without holding. Polling is at most 5 seconds;
dependency-free failed/stale tasks wake a hold as well as READY tasks, and peer
completion ends it early. The countdown includes time spent checking status.
Pending siblings do not erase a failed pass's retry state.

A first recorded MBPP GPU fault now waits 60 seconds, then must pass the existing
NCCL/CUDA admission probe before any training resumes. An explicit
`EXPERIMENTS_FAULT_TTL_SECONDS` still overrides that cooldown, but expiry is not
followed by another long exponential delay. A second recorded GPU fault or an
invalid receipt blocks admission. A busy GPU/node lock or failed NCCL/CUDA
admission releases the MBPP controller immediately, with the original cause
in its log, rather than entering a holding/retry loop. Legacy receipts without `time` expire from
their file modification time, not from the time they are read. Fault receipts
are atomically published and never cleared by MBPP `run` or `restart`.
Zero/invalid poll intervals are rejected. Setting
`EXPERIMENTS_HELP_SIBLINGS=0` explicitly restricts a node to the first root;
leave it enabled to serve all requested suites.
Terminal launches detach as before: Ctrl-C stops the log view, not the workers.
Use `stop` to stop this node's MBPP controller and its token-bound children.
Just run `bash scripts/run_mbpp_experiments.sh`: it checks for a fast-forward
update before inspecting the running controller. Re-running the command always
requests an owner-scoped restart after the storage audit, including unchanged
code. No separate `git pull` or `restart` is needed. `logs` follows the existing
log without interrupting workers. Valid
checkpoints, selections, completed results, and fault receipts remain on disk;
work after the last saved checkpoint may need repeating. An offline pull uses
the local checkout. `restart` remains available for an intentional forced reload.
Shutdown verifies that the previous controller's token-bound processes have
exited and checks the driver's process list for old CUDA PIDs for up to ten
seconds before admitting a replacement. Surviving processes or an unverifiable
GPU query block new GPU work;
neither a node-wide kill nor a GPU reset is attempted. The existing four-GPU
memory check and NCCL/DDP probe remain mandatory before a training task is claimed.
While a controller is stopping, the terminal displays elapsed shutdown time,
owned process IDs/roles and bounded GPU memory/owner queries at roughly five-second
intervals. Unknown/container-hidden owners are not assumed to belong to MBPP.
These observations are saved in `cleanup.mbpp.<node>.log`; `why` includes the
newest short cleanup excerpt without exceeding its 16 KiB attachment limit.
No new worker is started while the old controller is still alive.

Each node's `Progress` describes its current phase, not the suite-wide completed
branch fraction. Four observed rollout/gradient shard counters provide processed
item progress when available; otherwise the column explicitly reports phase-time
allocation usage (not result completion), with completed training updates in the
remarks. Missing evidence stays `확인 중`. The reader uses bounded log tails only;
it does not run GPU queries or load model/rollout payloads during status refresh.
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

An exhausted allocation is rejected before metering another attempt, recorded
internally as `BUDGET`; the MBPP dashboard shows `WAIT` with “예산 소진으로 중단”.
It does not get a new budget,
silently switch selectors, or count as a completed experimental result. Existing
publication recovery is still allowed. The runtime migration preserves the
frozen `switch.json`, prior migration receipts, decisions, costs and policies.
Without a valid published evaluation, report this branch as **incomplete**, not
as reward zero or `DONE`; keep its actual costs and saved artifacts in the report.
A normally budget-limited training run with a validated saved policy may still
finish evaluation/publication. Only valid result receipts (and the required
convergence curve) establish completion; budget exhaustion alone does not.

The operational MBPP queue can now evaluate a fully validated saved continuation
after allocation exhaustion, without training or changing its cap. These
posthoc measurements, curve points and their separate reporting costs are saved
under `<branch>/budget-recovery/`, never substituted for canonical results or
development gate labels. Already sealed matching evaluation shards are reused.
The queue defers exhausted-branch recovery until it has attempted every
independent runnable branch. Recovery reacquires the original task lease
without waiting; peer-owned recovery yields, and a recovery failure cannot
prevent other branches from being assigned. No exhausted training is restarted.
The dashboard keeps the branch in WAIT with a recovery-evaluation remark, not
DONE. If only review branches remain, the existing exit-80 path releases the
node rather than retrying training. Missing or corrupt checkpoints still need
review. See `docs/MBPP_BUDGET_RECOVERY_2026-09-20.md` for the observed incident,
cost constraints and validation/deployment boundaries.

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

Main root under `$OM_WORK/runs`:

- `selection-switch-mbpp-quality-v1` — On-policy · 선택비용 별도 (`quality`)

Override with `SWITCH_MBPP_QUALITY_ROOT`. The shared-prefix source is the existing
`selection-switch-mbpp-v1` (`SWITCH_MBPP_ROOT`), not a newly initialized training
history. The retained optional roots are `selection-switch-mbpp-difficulty-v1`
(`SWITCH_MBPP_DIFFICULTY_ROOT`) and `selection-switch-mbpp-long-v1`
(`SWITCH_MBPP_LONG_ROOT`); the original `fresh` root also remains explicitly runnable.
Roots must be separate, non-nested directories.
`OM_WORK`, `OM_OLMO3_ROOT`, `DATASETS_DIR`, and `VENV_DIR` retain their existing
meanings. Status and results never start training. Trainers, scoring code,
data splits, gate calculations, and frozen experiment contracts are unchanged.
