# Qwen3.5-9B replication — 2026-09-06

This is the selected additional experiment; 27B remains an optional preserved
configuration, not a dependency or automatically scheduled job.

Official model: [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B), pinned
to `c202236235762e1c871ad0ccb60c8ee5ba337b9a` (Hub `main`, re-verified against
the Hub tree on 2026-09-07). It is the released post-trained multimodal model,
used text-only, not a pretrained base. Model identity is by file content
(`model_matrix.PINNED_OFFICIAL_FILES`), never by folder name: the folder can be
called anything (`Qwen3.5-9B-pinned` is only the default download name). The
pretrained base, [Qwen/Qwen3.5-9B-Base](https://huggingface.co/Qwen/Qwen3.5-9B-Base)
at `68c46c4b3498877f3ef123c856ecfde50c39f404`, is registered in the same table;
switching to it is a config change (`repository`, `revision`,
`prompt_format: olmo_rlzero`, new run id) plus its 19.3 GB of weights on the
volume. The two 9B repositories ship the same `config.json`, so discovery also
compares shard sizes. The official config uses `Qwen3_5ForConditionalGeneration`
and 32 text blocks. Vision, embeddings and output head are excluded from ranking
gradients and LoRA targets.

The 27B runbook's scientific caveats also apply: initialization, capacity,
prompt template and adapter parameter counts are not controlled individually.
The fused DeltaNet QKV adapter includes keys, unlike attention Q/V-only LoRA.

## Execute

Safety correction (2026-09-06): wrappers never fetch/merge/reset Git or terminate
other jobs. Update explicitly in an idle checkout. Uploaded directory names may
be arbitrary, but config and weight identity must match the pinned revision;
`OM_TRUST_LOCAL_SNAPSHOT`/`OM_ALLOW_UNPINNED_SNAPSHOT` no longer bypass that check.
Discovery/doctor do not rename model folders or rebuild indexes. Shard/index
names must be a valid pinned layout; prepare that layout separately if uploads
were renamed. Multiple matching uploads require explicit `OM_SNAPSHOT_PATH`.
A read-only upload without a verified manifest must be prepared in a writable
staging copy, rather than silently treated as verified.

`prepare` exists only because compute nodes have no Hub access: it downloads the
pinned model and datasets on a networked machine into the shared volume. If the
snapshot was uploaded by hand instead (any folder under `$MODELS_DIR`; the
default download folder is `Qwen3.5-9B-pinned`),
skip `prepare`: `check`/`run` seal the uploaded files offline against the pinned
official sizes/hashes (`model_matrix.PINNED_OFFICIAL_FILES`) and adopt uploaded
MATH-500/MBPP copies by content, then continue. A failed launch prints one
Korean diagnosis line and one action line; the full stream is in the session log.

From a clean committed checkout, with the same shared volume/environment as
the other registered experiments:

```bash
# Internet-connected preparation machine; downloads actual weights and data.
bash scripts/run_qwen35_9b.sh prepare
# Optional post-primary check; normal launch already includes its checks.
bash scripts/run_qwen35_9b.sh check
# Idle four-H100 compute node; run only the 40-point 9B matrix.
bash scripts/run_qwen35_9b.sh
```

No argument, `run`, and the compatibility alias `run-idle` all select 9B on
this node without waiting for OLMo3 completion on other nodes. The wrapper
does not rotate to 2B, 4B, 27B or another OLMo experiment, even after a failure.
It preserves existing work and still requires free local locks, idle GPUs,
the pinned snapshot, valid contracts and successful preflight. It does not
terminate an existing job by default. `prepare`, `check` and `status` remain
9B-only. The optional `check` and the generic additional/rotation launchers
retain their primary-completion gate; normal 9B launch needs no separate check.

Two datasets (MATH-500 / MBPP), five seeds and four GRPO checkpoints
(0/25/100/400) are preserved. Generation batch is 32; gradient and training
log-prob microbatches are 4 (Qwen3.5-9B has 8 full-attention layers with 4 KV
heads, so the 2048-token KV cache is ~40 MB per sequence). This is a starting configuration, not a
measured speed/memory guarantee. Changing it changes the contract hash.
`check` uses a short prompt and does not qualify 2048-token or four-rank
training. Full weights have not been trained on the local audit host.

Results: `$OM_WORK/results/qwen35-9b-posttrained-math-code-grpo-v1/`.
Models/configuration/outputs are separate from the retained 27B experiment.
Do not reuse or relabel 27B checkpoints as 9B results.

## Contract conflict recovery (2026-09-08)

### Recovered model-path errors still shown as current (2026-09-11)

A later status report showed `4/8 fresh-rollout 400x32 + val` together with
`ERROR (current): ... rollouts_behavior_train.shard0.manifest.json ...
Qwen3.5-9B-pinned`. The status reader could produce this exact combination
after a successful model-alias repair: it ignored `regime-attempt-1-alias-1.log`
and kept reading the failed `regime-attempt-1.log` as current. It also sorted
by retry number, so an old attempt 3 could outrank attempt 1 of a new launch.

The status fix includes alias-resume logs, orders attempts by their recorded
start time (file mtime for legacy logs), and uses the recorded `main.log` byte
offset to separate earlier errors from new failures. New `ValueError` and
contract failures remain visible. Verbose status names the current attempt
log. Missing or invalid start metadata falls back conservatively; it does not
discard errors on that basis.

Apply this **read-only status fix without restarting experiments**:

```bash
git pull --ff-only
bash scripts/run_qwen35_9b.sh status
```

Use `bash scripts/run_qwen35_9b.sh log` for the live console, or
`bash scripts/run_qwen35_9b.sh status verbose` to see the chosen attempt log
and watchdog telemetry. Do not restart healthy workers just because the old
status showed this error. The repaired display is not proof that every remote
worker is healthy: check whether a *new* error remains, rollout writes grow,
or fresh watchdog telemetry reports computation. At d0, zero GRPO steps is
expected; fresh rollout generation is still GPU work.

CPU regression tests reproduce the exact misleading error before the fix,
cover alias resumes, reset retry numbers, new failures, invalid/truncated
logs, and verify that status leaves all input files unchanged. Existing tests
also exercise alias recovery against the pinned `2e96090` validator. Actual
H100 execution cannot be checked on this audit host.

### Earlier model-path failure loop (2026-09-11)

The uploaded `status-qwen35-history (1).log`, sampled at
`2026-09-11T00:22:42Z`, reports 0/40 completed points and 22h34m without
changes to completion, GRPO steps or rollout bytes. Repeated failures end in
`rollouts_behavior_train.shard0.manifest.json: model mismatch`, with the
recorded basename `Qwen3.5-9B-pinned`. This is not evidence of useful GPU
progress, and this exception is not a CUDA error. The abbreviated status line
does not establish whether the two model paths actually identify the same
snapshot; the recovery now checks that on the compute node.

The supervisor handles the old pinned generator without modifying its source
or the E5 source-hash contract:

- Before entering a partial point, and after a matching manifest failure,
  prove that the recorded absolute model paths resolve to the same accessible
  canonical directory and that recorded model metadata hashes still match.
- Validate completed rollout hashes, prompt/K coverage, merged-versus-shard
  rows, sampling parameters and existing policy/RNG bindings in a temporary
  view. Only the model-path spelling changes in this validation view.
- Publish the canonical merged manifest and its ready cache. Preserve rollout
  bytes, original shards, partials, run configuration and training checkpoints.
  Original manifests and a hash receipt are retained under each point's
  `logs/model-alias-repair/`. Already bound analysis results are not rewritten.
- Re-enter the cached point immediately after a successful repair, with at
  most three repair passes for its three primary rollout sources. Each pass
  keeps a separate attempt log. Unprovable or repeated mismatches block that
  family; the queue tries other eligible families instead of running CUDA
  recovery or repeating the same matrix hundreds of times. Explicit Qwen
  launchers still do not switch to OLMo or another Qwen model.

Install the pushed update and use the existing node-local relaunch command
on the **stalled Qwen nodes only**:

```bash
git pull --ff-only
bash scripts/run_qwen35_9b.sh restart-idle
```

Keep the existing work root and environment. Do not reset the matrix, remove
rollouts, rebuild model folders, or restart healthy E5/OLMo work. A running
old launcher does not acquire this fix merely because another checkout was
updated. Look for `[model-alias-repair]` followed by a new stage, and confirm
actual rollout/step/completion progress with
`bash scripts/run_qwen35_9b.sh status`. If identity cannot be proved, the
launcher prints the full offending paths or failed hash and leaves artifacts
untouched. CPU tests exercise the recorded `2e96090` validator through all
three source transitions; the audit host cannot verify live H100 execution.

### Earlier matrix contract conflict

The uploaded `additional-qwen35-run-20260908T072635Z-eQlPEK.log` ended on
`matrix contract mismatch`, after a successful single-GPU smoke. It does not
show a CUDA failure. The updated launcher checks the contract before FLA/smoke,
labels the stage `matrix-contract-*`, and prints the actual differing fields.
It writes the expected JSON beside the original as
`*.json.expected-<digest>.json`; the recorded contract is not overwritten.
The exact remote mismatch cannot be resolved without comparing those files.

Directly invoking `bash scripts/run_qwen35_9b.sh` is the operator's assignment
of this node to Qwen. It no longer requires OLMo3 completion on other nodes.
Automatic OLMo-to-Qwen handoff and the generic rotation worker still require
all 40 primary points and final collection. Do not restart healthy primary
OLMo workers for this launcher change. Contract validation is not bypassed.

### Default node-local run (2026-09-09)

Run `bash scripts/run_qwen35_9b.sh` on the allocated idle node. The remaining
nodes continue OLMo3; `run-idle` is retained only for compatibility.
Ctrl-C alone does not guarantee that detached CUDA children exited. To clean
up this user's previous Qwen 9B processes in the same work root on this node,
then launch again from an updated idle checkout:

```bash
git pull --ff-only
bash scripts/run_qwen35_9b.sh restart-idle
```

Recovery correction (2026-09-11): `restart-idle` now recognizes a named Qwen
launcher whose `OM_WORK` was exported after Bash started, using its children's
initial environment as evidence. It also recognizes old children by the exact
work-root/9B session-log namespace. It tracks selected children by PID and
process start time even after reparenting, escalates TERM to KILL when needed,
waits for actual exit, and verifies that both node locks can be acquired before
launching again. The previous implementation could miss the launcher or forget
a TERM-ignoring orphan and then fail with `additional suite already queued`.
Cleanup prints the selected PIDs; an unrelated remaining owner is listed, not
killed. Lock files are never deleted. CPU tests use real Bash processes and
real `flock` locks, including late exports and a TERM-ignoring orphan.

`restart-idle` uses the existing process-namespace cleanup with a 15-second
TERM grace followed by KILL for remaining matches. It selects the same
`OM_WORK` and 9B run namespace/launcher plus descendants, on this machine and
under this user only. It excludes its caller/ancestors, other model namespaces
and other work roots. It does not delete artifacts, contracts or lock files.
Unrecognized survivors still block normal lock/GPU admission rather than being
killed indiscriminately. The no-argument command does not perform this cleanup.

All run modes bind the assignment to this hostname and only the 9B profile. Each
requires the same exclusive local locks, four idle H100 GPUs, pinned snapshot,
and dataset/matrix contracts as normal admission. No mode waits behind a
primary lock, resets a contract, or rotates to another model. None stops
OLMo workers. A remaining lock owner must be identified rather than assumed
to be OLMo just because the shared lock file is named `primary.lock`.
The OLMo-to-Qwen automatic handoff remains disabled. A pre-existing Qwen
contract conflict can still reject admission and must be diagnosed separately.

If inspection confirms an obsolete, metadata-only Qwen root, stop the Qwen
launchers using that root before this separate recovery command:

```bash
bash scripts/reset_qwen35_root.sh
bash scripts/run_qwen35_9b.sh
```

The reset refuses any point directory, partial checkpoint, nonempty result
directory, or live matrix/local-primary lock by default. Do not force a reset
when it refuses existing work: retain the artifacts and compare the contract
fields. An accepted reset moves the old metadata to a unique directory under
`$OM_WORK/quarantine/`; nothing is deleted, and external contract lock files
keep their inodes. It never runs automatically during a launch.

Updated launchers hold a shared lifecycle lock across preflight and training;
reset needs that lock exclusively. Legacy workers do not hold this new lease,
so stop their Qwen launchers explicitly before resetting, including workers
on other nodes that are still in preflight. Old queue locks are also checked,
but their absence alone does not establish that a legacy preflight is idle.

Contract-stage failures skip the redundant automatic model doctor. For other
Qwen failures it is bounded by `ADDITIONAL_FAILURE_DOCTOR_TIMEOUT` (30 seconds,
plus at most 2 seconds for forced termination); advisory failure does not
replace the launcher's original exit code. Manual `doctor` is unchanged.

## Is it training? (2026-09-07)

While the matrix runs, the launcher prints one `[progress]` line every 10
minutes (`OM_PROGRESS_INTERVAL_SECONDS`) computed from durable artifacts only
(DONE points, GRPO steps, rollout bytes, newest artifact write); the line reads
`TRAINING ...`, `NOT STARTED ...` or `NOT TRAINING for <age> ...`. This remains
a launcher diagnostic. Status now derives its DECISION and tables together
from the whole matrix: per-family locks, durable writes, stage progress and
session records. A PID by itself does not establish training progress; a
silent unclaimed launcher is a warning, not a progressing family. The old
single-session/global-PROGRESS override is no longer used by status.

## Reading progress on a phone

```bash
bash scripts/run_qwen35_9b.sh status            # completion grid, current work and errors
bash scripts/run_qwen35_9b.sh status verbose    # all points, launchers, scores and attempt details
```

Since 2026-09-13 the default is a ten-row completion grid, covering all forty
registered points. Each seed/dataset row has explicit `d0`, `d25`, `d100` and
`d400` cells: `DONE`, `RUN`, `WAIT`, `ERROR`, `CHECK` or `STOP`, plus its done
count. The first line totals the point states. Current work appears below
the grid with its stage, node and last write; current errors remain visible.
`DONE` still requires a nonempty completion record, not an exited launcher.
`CHECK` means quiet or unverified activity, not a confirmed dead process.

Verbose mode retains the full per-family states, the `ALL POINTS (40)` table,
score tables and attempt log details. `LAUNCHERS` includes all sessions without an exit record
and every exit from the last three days, with no eight-row cap. PID liveness
is verified only on the local node; a silent remote session is unverified,
not declared dead. KEY NUMBERS cover every scored point, followed by
`overall_verdict=` / `recommended_action=`. An extra
profile word (`status h100`) is accepted and ignored. Status always uses the
installed code and prints its revision. It never fetches or merges, regardless
of which experiment owns the shared checkout or whether a local launcher is
present. Update explicitly only after launchers using that checkout have
exited; running E5, OLMo and Qwen stages can all read its files.

The full design is printed even before any run directory exists. In verbose
mode, `work` and `matrix` show the actual paths being inspected. A missing/broken renderer
produces `[status-error]` and a nonzero exit, never a silent six-point fallback.
Output, including errors, remains in
`$OM_WORK/console-logs/status-qwen35-history.log`. No GPU or experiment restart
is required to inspect status; the training configuration and code are unchanged.

CPU verification on 2026-09-13: 75 tests passed across
`tests/test_matrix_status.py` and `tests/test_status_reward_audit.py`, including
compact/verbose entrypoints, all forty registered cells, current versus old
errors, empty completion records, remote liveness and shell failure codes.
This change does not launch or interrupt any experiment.

The terminal shows tagged lines only (`[stage]`, `[progress]`, `[abort]`,
`[model]`, `[regime-*]`, `START/OK/FAILED/DIAGNOSIS/ACTION`); tracebacks and
library output go to the session log. A point prints one `[progress]` line per
stage, `<run>  k/8 <stage>  +<min>`. GRPO prints one console line every 5 steps
with reward, active groups, loss, seconds per step and ETA; `grpo_stats.jsonl`
still records every step.

## Logging and failures

Live output, without starting or stopping an experiment:

```bash
bash scripts/run_qwen35_9b.sh log
```

This follows the most recently modified Qwen 9B run session on the current
node. From a node without its own session, it follows the newest shared run
log and prints that choice. `Ctrl+C` stops only the viewer. `live` and `logs`
are aliases. The viewer does not source setup, update Git, or acquire GPU locks.

All additional profiles now capture both stdout and stderr after environment
setup, including prepare, admission, snapshot, FLA, smoke and matrix failures:

```bash
ls -t "$OM_WORK"/console-logs/additional-qwen35-*.log
tail -F /absolute/path/to/the-session.log
```

Every invocation gets a unique log (no overwrite), with UTC start/stage/end
markers, host, PID, source commit, final exit status and last stage. Initial
setup messages before logging starts remain terminal-only. Original matrix
logs and per-run attempt logs remain available. Session stderr includes
tracebacks absent from the old phase-only logs. No environment/token dump or
shell tracing is enabled. A failed log writer cannot report successful exit.

Training writes per-step reward, loss, gradient norm, ratio, clip fraction,
approximate KL, response tokens, time and peak memory to `grpo_stats.jsonl`.
Non-finite loss/gradient or active zero-gradient updates are rejected across
ranks before any optimizer update. A failed step has no success metric row;
the traceback and launcher exit marker identify the failed attempt.

Ranking OOM retries discard all accumulated gradients for the prompt and restart
at a smaller microbatch. Failed generation attempts discard partial outputs and
restore the RNG state before trying a smaller batch. Different batch schedules
can still produce different sampled tokens; no bitwise equivalence is claimed.
Backoff events are recorded in the full session log. GRPO loss/backward OOM is
not independently retried inside one DDP rank; normal job recovery handles it.

## Audit handoff

The separate `offpolicy-misranking-final-audit` now uses the producer's v4
validator, registered dimensions and all five selectors. Set `CODE_REPO` and
`AUDIT_MATRIX_CONFIG` to this checkout and `configs/qwen35_9b_grpo.json` when
freezing a completed matrix. Freeze is staging, not submission approval:
raw-run lineage and explicit manuscript claim review remain required.
For a nonstandard model directory on a different machine, set
`AUDIT_MODEL_SNAPSHOT` to a local copy with the verified snapshot manifest.
The manuscript's registered extension must be amended before reporting 9B
as its completed replication; this code change does not silently amend it.
