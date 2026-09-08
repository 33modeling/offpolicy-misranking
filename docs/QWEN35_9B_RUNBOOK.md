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
# Idle four-H100 compute node; offline, no matrix training.
bash scripts/run_qwen35_9b.sh check
# Try the 40-point 9B matrix first, then other selected independent matrices.
bash scripts/run_qwen35_9b.sh run
```

`run` (also the no-argument default) now retains a rotation worker: it tries
`qwen35`, then `qwen35_2b`, `qwen35_4b`, and `olmo3_domains`. A failed or busy
profile keeps its artifacts and yields to the next one. No 27B job is selected
implicitly. Override the fallback list with `OM_RLZERO_FALLBACK_PROFILES`;
9B is still tried first. `OM_SNAPSHOT_PATH`, if set, applies only to the first
profile, not to different fallback models. Each fallback requires its own
registered, offline model and data. If none is runnable, the worker waits;
it does not download weights or fabricate GPU activity.

`prepare` and `check` remain 9B-only. To execute only 9B without a rotation
worker, use `bash scripts/run_additional_experiments.sh --run qwen35`.
`status` reports the 9B matrix, not the currently selected fallback; use the
rotation terminal's `[fallback]` lines and each profile's session log for that.

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

The uploaded `additional-qwen35-run-20260908T072635Z-eQlPEK.log` ended on
`matrix contract mismatch`, after a successful single-GPU smoke. It does not
show a CUDA failure. The updated launcher checks the contract before FLA/smoke,
labels the stage `matrix-contract-*`, and prints the actual differing fields.
It writes the expected JSON beside the original as
`*.json.expected-<digest>.json`; the recorded contract is not overwritten.
The exact remote mismatch cannot be resolved without comparing those files.

Normal `run` requires OLMo3 to finish first, including all 40 points and the final collection.
An idle node or a blocked OLMo3 family does not authorize a Qwen run. The shared
launcher returns 75 before GPU preflight when primary completion is missing.
Only after OLMo3 completion, update an idle Qwen checkout with
`git pull --ff-only`, then run `bash scripts/run_qwen35_9b.sh`. This wrapper
does not rotate to other models. Do not restart healthy primary OLMo workers
for this launcher change. The recovery commands below are also post-primary.

### Explicit idle-node parallel run (2026-09-09)

The operator may assign a separate node to Qwen 9B while the remaining nodes
continue OLMo3. Ctrl-C alone does not guarantee that detached CUDA children
exited. To clean up this user's previous Qwen 9B processes in the same work
root on this node, then launch again from an updated idle checkout:

```bash
git pull --ff-only
bash scripts/run_qwen35_9b.sh restart-idle
```

`restart-idle` uses the existing process-namespace cleanup with a 15-second
TERM grace followed by KILL for remaining matches. It selects the same
`OM_WORK` and 9B run namespace/launcher plus descendants, on this machine and
under this user only. It excludes its caller/ancestors, other model namespaces
and other work roots. It does not delete artifacts, contracts or lock files.
Unrecognized survivors still block normal lock/GPU admission rather than being
killed indiscriminately. Use `run-idle` when no previous Qwen cleanup is needed.

Both modes bind the exception to this hostname and only the 9B profile. Each
requires the same exclusive local locks, four idle H100 GPUs, pinned snapshot,
and dataset/matrix contracts as normal admission. Neither waits behind a
primary lock, resets a contract, nor rotates to another model. Neither stops
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
`TRAINING ...`, `NOT STARTED ...` or `NOT TRAINING for <age> ...`. `status`
prints the same as its `PROGRESS` line and its DECISION is `ERROR: NOT
TRAINING ...` whenever the launcher is alive but nothing durable changed for
`OM_PROGRESS_STALL_MINUTES` (30). A launcher that is alive without training
must never look like a running experiment (18 hours were lost that way on
2026-09-06/07).

## Reading progress on a phone

```bash
bash scripts/run_qwen35_9b.sh status     # one screen: stage, points done/started, current point, last error
```

The terminal shows tagged lines only (`[stage]`, `[progress]`, `[abort]`,
`[model]`, `[regime-*]`, `START/OK/FAILED/DIAGNOSIS/ACTION`); tracebacks and
library output go to the session log. A point prints one `[progress]` line per
stage, `<run>  k/8 <stage>  +<min>`. GRPO prints one console line every 5 steps
with reward, active groups, loss, seconds per step and ETA; `grpo_stats.jsonl`
still records every step.

## Logging and failures

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
