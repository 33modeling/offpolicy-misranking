# Complete Qwen Status

Baseline: `0728570`. Scope: status display only, not training or experiment
contracts. The public command remains:

```bash
bash scripts/run_qwen35_9b.sh status
```

## Findings And Repairs

- Default output now lists all 40 registered points, including completed and
  unstarted points, plus the ten-family grid and scored-point values.
- The launcher list no longer drops everything after eight sessions. All open
  sessions and exits from the last three days are shown. An older live launcher
  can no longer disappear behind eight newer exit logs.
- The shell's independent, local-PID-based DECISION was removed. The same
  full-matrix renderer now produces DECISION, tables and the overall verdict.
  A silent remote session is unverified, not confirmed dead. A live but silent
  unclaimed launcher does not establish training progress.
- An empty work path still prints the whole design. The inspected work and
  matrix paths are visible. A broken/missing renderer returns an explicit error
  instead of silently substituting a six-point sample.
- Completion counts use the configured dataset/seed/drift combinations, not
  arbitrary directories containing DONE files. Historical launcher failures do
  not override a fully completed registered matrix.
- `status verbose` adds attempt details and stage-log tails; it is not required
  to see all points. Standard output and errors remain in the status history.

## Verification

CPU regression coverage includes the real public shell command, all 40 point
rows, more than eight sessions, remote liveness, silent local preflight, current
errors, completion, empty paths, paths with spaces, explicit renderer failures,
history-writer errors, and unchanged E5 scientific-code hashes. Related OLMo
status, Qwen launch contracts, pinned files and training-progress tests also run.

Result: 126 tests passed in 11.06 seconds with CUDA disabled. Bash syntax checks
and `git diff --check` passed.

No cluster GPU run was inspected or restarted. No training source, launch
configuration, checkpoint, experiment contract or saved result was changed.
