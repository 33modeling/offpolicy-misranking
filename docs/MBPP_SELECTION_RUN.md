# MBPP Selection Experiments

Run this same command from the original `offpolicy-misranking` checkout on
**every allocated four-H100 node**, including replacement nodes:

```bash
bash scripts/run_mbpp_experiments.sh
```

The command joins the existing node controller's shared MBPP queue:

| 표시명 | 선택 방식 | 분기 예산에 차감하는 비용 | Gate 판단 기준 |
| --- | --- | --- | --- |
| On-policy · 선택비용 포함 | 현재 정책에서 계산한 gradient | 선택 + 진단 + 학습 | 최종 보상 차이 |
| On-policy · 선택비용 별도 | 같은 on-policy gradient | 진단 + 학습; 선택 비용은 별도 기록 | 비용 보정 학습 효율 |
| Difficulty · 선택비용 포함 | 저장된 정답률이 0.5에 가까운 문제 | 선택 + 진단 + 학습 | 비용 보정 학습 효율 |

Compatibility keys remain `fresh` → `fresh_r/budget/final`, `quality` →
`fresh_r/matched/convergence`, and `difficulty` → `difficulty/budget/convergence`
(selector/accounting/gate). These are saved identifiers, not extra methods.

The first two suites use the **same on-policy gradient selector**, with gradients
computed under the current policy. `quality` is not another selector: it changes
the cost accounting and gate criterion. Difficulty ranks cached success rates
by closeness to 0.5. `final` uses final-reward differences; `convergence` uses
updates saved to a common reward target, minus selection cost in update units.

“선택비용 포함” charges selection GPU time to the diagnostic/training branch
allocation. “선택비용 별도” records selection GPU time on the `reporting` ledger,
outside that allocation; it is **not free** and remains part of total actual
compute. Evaluation has a common, separate reporting allocation in all three
suites. Reporting-ledger selection and evaluation costs are distinguished by
phase, not omitted or combined into a supposedly free selector.

All suites include random controls. `fresh` creates five on-policy-selected
training prefixes. The other two reuse these exact prefixes, states at
25/50/100 updates, and the same evaluation questions. They do not import MATH
prefixes. Each suite uses the existing 18 development and 30 held-out
continuations: **48 per suite, 144 across the three**, with development seeds
0/1/2 and held-out seeds 3/4. These are separate continuation-training branches,
not merely re-evaluations of the first suite's trained results. Shared prefixes
do not make the later continuation training identical or free.
Once all shared prefix certificates are ready, `quality` and `difficulty` can start
even while `fresh` continuations are still running elsewhere. No node is assigned
permanently to one suite. Final evaluation uses eight responses per question;
convergence curves use three archived checkpoints with four responses per question.

This is a port of the current switch suites, not a new difficulty definition,
an E5 fixed-checkpoint run, or a gate that directly chooses on-policy versus
difficulty in one branch. The original gates choose their suite's selector
versus random; the shared states permit the on-policy/difficulty comparison.
The display names do not rename CLI keys (`fresh`, `quality`, `difficulty`),
saved directories, scoring phase names, or any frozen protocol fields.

### Result tables

Keep the cost-inclusive and separate-cost conditions in distinct blocks;
they are not the same total-compute comparison. Within each block compare
Full selection, Full random and Gate policy at matched seed/checkpoint states.
Report valid-result count / planned count, completed training updates, measured
reward, diagnostic/selection/training GPU time, and evaluation GPU time separately.
Include interrupted and failed attempts in an explicit cost/failure audit.
An unevaluated budget-exhausted branch has no measured reward: show “미완료”,
not zero. If no valid paired outcomes exist, a runtime/cost table is possible,
but there is no completed selection-versus-random performance comparison.

## Commands

```bash
bash scripts/check_random_storage.sh           # ALL discovered suites: RF/RR/RO separately; <=4 KiB TXT in home
bash scripts/check_mbpp_storage.sh             # saves <=4 KiB TXT in home; never starts training
bash scripts/run_mbpp_experiments.sh plan       # settings only, no writes/GPU work
bash scripts/run_mbpp_experiments.sh check      # read-only local input checks
bash scripts/run_mbpp_experiments.sh status
bash scripts/run_mbpp_experiments.sh status --watch
bash scripts/run_mbpp_experiments.sh progress   # MBPP suites only (fresh, quality, difficulty), unprepared ones listed
bash scripts/run_mbpp_experiments.sh saved      # READ-ONLY saved-work/archived-work inventory, <=4 KiB stdout
bash scripts/run_mbpp_experiments.sh stop       # stop/clean THIS node, not peer nodes
bash scripts/run_mbpp_experiments.sh results    # one report per suite, also copied home
bash scripts/run_mbpp_experiments.sh why        # ONE diagnostic TXT, at most 16 KiB
bash scripts/run_mbpp_experiments.sh run fresh
bash scripts/run_mbpp_experiments.sh run quality
bash scripts/run_mbpp_experiments.sh run difficulty
```

For MBPP monitoring, use `bash scripts/run_mbpp_experiments.sh status --watch`.
It refreshes one dashboard for the three named MBPP suites, with all seed/step
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
`status --all` adds exact roots and individual task states;
`status quality --watch 5` watches only **On-policy · 선택비용 별도**.
This is read-only and does not restart training. The generic
`run_experiments.sh status` defaults to the math-root overview, not this MBPP view.

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
Unprepared `quality`/`difficulty` roots are allowed alongside the existing `fresh` root.

For missing random-control work, `check_random_storage.sh` scans every switch and
MoPPS root under the configured work directory, not just MBPP. It separates
`random_full`, `random_reduced`, and `random_online` from selector results, lists
sealed random results, and prioritizes missing stops, archives and conflicting
parent-only stops. It saves `~/random-storage-*.txt` (at most 4 KiB), never resets
or restores anything, and does not certify tensor contents. An absent result is
not proof of deletion. Both on-policy suites keep their existing storage names
for compatibility; no saved paths or protocol keys are renamed.

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
with “평가·결과 저장 남음”, or `RUN` while that work has a fresh heartbeat.
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
`--json` always retains the full snapshot. Distinct cost-qualified display names
identify the two on-policy suites; `status --all` includes their exact roots.
A new `quality` root does not erase or replace completed `fresh` results.

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

- `selection-switch-mbpp-v1` — On-policy · 선택비용 포함 (`fresh`)
- `selection-switch-mbpp-quality-v1` — On-policy · 선택비용 별도 (`quality`)
- `selection-switch-mbpp-difficulty-v1` — Difficulty · 선택비용 포함 (`difficulty`)

Override with `SWITCH_MBPP_ROOT`, `SWITCH_MBPP_QUALITY_ROOT`, and
`SWITCH_MBPP_DIFFICULTY_ROOT`. Roots must be separate, non-nested directories.
`OM_WORK`, `OM_OLMO3_ROOT`, `DATASETS_DIR`, and `VENV_DIR` retain their existing
meanings. Status and results never start training. Trainers, scoring code,
data splits, gate calculations, and frozen experiment contracts are unchanged.
