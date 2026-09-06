# Qwen3.5-9B replication — 2026-09-06

This is the selected additional experiment; 27B remains an optional preserved
configuration, not a dependency or automatically scheduled job.

Official model: [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B), pinned
to `c202236235762e1c871ad0ccb60c8ee5ba337b9a`. It is the released post-trained
multimodal model, used text-only, not a pretrained base. The official config
uses `Qwen3_5ForConditionalGeneration` and 32 text blocks. Vision, embeddings
and output head are excluded from ranking gradients and LoRA targets.

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
snapshot was uploaded by hand instead (scp into `$MODELS_DIR/Qwen3.5-9B-pinned`),
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
# After target-node qualification, explicitly launch the 40-point matrix.
bash scripts/run_qwen35_9b.sh run
```

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
