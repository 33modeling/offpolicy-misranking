# Qwen3.5 execution safety repair

Reviewed source: `a046c09`. No experiment, model download, external job
termination, or checkout reset was performed during this repair.

## Closed defects

- Prep success/failure is tested on `run_stage`, not the progress logger.
- A partially failed backward discards the entire prompt's accumulated gradient
  before restarting at a smaller microbatch. Regression injects OOM after a
  successful earlier chunk and a partial later chunk; the final sum matches the
  no-OOM result. Unrecoverable failures return no gradient result.
- Generation retries release failed frames/outputs and restore RNG state before
  restarting. Batch scheduling can still change individual sampled tokens.
- Qwen/follow-up wrappers never fetch, merge or reset Git. They retain the
  current committed revision and existing operator-selected runtime settings.
- Additional launch admission no longer kills processes by command substring.
  Node/provision locks remain in place. Diagnostics no longer recommend broad
  process termination or destructive checkout commands.
- Discovery requires pinned config content or exact Hub metadata, rejects wrong
  model sizes and ambiguous candidates, and is bounded across directory cycles.
  It does not rename folders, create links or reconstruct indexes.
- Model folders may have arbitrary names. Snapshot check and matrix creation
  require a manifest covering the current files and pinned content. Old trust
  bypasses do not authorize a run. Previously self-sealed manifests are checked
  against official weight hashes rather than accepted by revision text alone.
- Writer status 1 is now fatal; only grep's no-match status is normalized.
  The final exit record is produced after the log writer drains.
- The separate audit accepts nonstandard paths only with a locally verified
  snapshot (optionally supplied through AUDIT_MODEL_SNAPSHOT), recording its
  manifest/validator hashes.

## Verification

- Full code suite: **216 passed**, 132.29 s.
- Final logging/repair focused suite: **18 passed**.
- Separate audit suite: **13 passed**.
- Fatal-error lint, compileall, shell syntax and git diff checks passed.

CPU mocks/fixtures exercised failure branches; no full-size 9B weights were
loaded. Four-H100 runtime, long-sequence peak memory and performance remain
unverified. The registered generation/gradient/logprob batches remain 32/4/4.
Use a separate clean checkout for new execution; do not update an active one.
