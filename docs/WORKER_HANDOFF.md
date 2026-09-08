# OLMo3 first, independent experiments only after completion

The OLMo launcher preserves blocked family checkpoints and rollouts. It finishes
other eligible primary families first. When every remaining selected primary
family is marked LOOPING, it does not exit or collect a partial matrix as complete.
It keeps its heartbeat and primary lock, stops GPU keepalive, and reports
`[primary-blocked]` while rechecking OLMo3 work. It never hands the process to
Qwen or another independent matrix. Repaired, explicitly unblocked primary
families can be resumed by the same worker.

## Independent work

The default registered profile order is:

```text
olmo3_domains qwen35_2b qwen35_4b qwen35 qwen38
```

Only after OLMo3 completes all 40 points and the final collection, explicitly
start a separate rotation if needed:

```bash
OM_RLZERO_FALLBACK_PROFILES="olmo3_domains qwen35_2b" \
  bash scripts/run_available_experiments.sh
```

These are the existing `run_additional_experiments.sh --run <profile>` matrices,
not new experiment definitions. Each keeps its own registered model/data checks,
generation binding, GPU admission lock, run root and result validation. The
fallback never invokes `--prepare` or downloads models/data. It removes inherited
primary generation overrides and implicit authentication tokens from the child
environment.

A failed/unavailable profile yields immediately to the next selected profile.
Automatic matrix restarts are disabled in this path and each point gets one
attempt. Failure cooldown is 900 seconds, configurable with
`OM_RLZERO_FALLBACK_COOLDOWN_SECONDS`. A profile completed successfully is skipped
for the lifetime of this rotation worker; on restart its normal artifact checks
determine what needs work.

If every selected profile is unavailable, cooling down or complete, the worker
stays alive and checks every 30 seconds. It cannot perform useful GPU computation
without ready work; it does not disguise keepalive as training. Upload/prepare the
registered snapshots or select other registered profiles when none are available.

The rotation and direct additional compute launchers check the shared primary
completion binding, family stamps, all 40 DONE files and final report outputs
before any GPU preflight. Missing or stale completion returns 75. This guard
does not stop a process already running from an older frozen checkout; do not
restart healthy OLMo3 workers just to update the policy.

## Completion postcondition

If a pipeline exits zero but `run_complete` fails, the supervisor writes a
`[point-failed] ... rc=43: completion validation failed: <reason>` event to the
terminal/worker log and point `logs/supervisor.log`. It returns immediately without
another point retry, CUDA recovery or sleep. Exit code 43 is the existing permanent
contract-failure code. The OLMo wrapper marks that family blocked on its first such
failure, even if historical CUDA text remains in its logs. Other workers preserve
that marker on startup unless explicitly instructed to clear loops.

Neither DONE nor checkpoints are deleted or rewritten by this failure handling.
DONE alone is not proof that strict completion validation succeeded.

## Verification scope

CPU fixtures cover postcondition failure after a successful pipeline, immediate
shared blocking, marker preservation across worker startup, rotation after an
unavailable profile, completed-profile skipping, environment isolation, and
primary-only waiting/resumption and cluster-wide completion admission. Real H100 execution and snapshot availability
must be checked on the cluster; local tests use fake GPU/model launchers.

## Bounded waits and retry ownership

The primary defaults to one matrix invocation per family claim
(`OM_RLZERO_FAMILY_ATTEMPTS=1`). The matrix still owns its configured point retry
budget. If that budget fails, the primary rotates to another eligible family
before reclaiming the failed family after cooldown. This avoids multiplying a
three-attempt point budget by three immediate family attempts.

The last failed point attempt records diagnostics but does not run a recovery
stage or sleep when no subsequent pipeline attempt remains. Both matrix-contract
and completion-postcondition failures return 43 without another quarantine/retry
cycle. A new unknown failure is not reclassified using historical CUDA errors.

Watchdog shutdown interrupts its timer immediately. The additional launcher
stops its progress writer before draining the log pipe, including on failure;
otherwise the open pipe can prevent exit and therefore prevent fallback rotation.

Fallback matrices export `REGIME_YIELD_WHEN_BUSY=1`: when all remaining families
are owned by other nodes, the matrix returns 75 instead of holding an idle node
in the queue. The additional launcher propagates that status without restarting
the same matrix. Ordinary matrix launchers retain their existing queue-wait
behavior unless this setting is explicitly enabled.

Additional-launcher limits (seconds):

| Variable | Default | Scope |
| --- | ---: | --- |
| `ADDITIONAL_QUALIFICATION_LOCK_SECONDS` | 60 | Shared dataset qualification lock |
| `ADDITIONAL_DATA_TIMEOUT` | 1800 | Dataset qualification command |
| `ADDITIONAL_MODEL_TIMEOUT` | 1800 | Each snapshot discovery/check/seal command |
| `ADDITIONAL_FLA_TIMEOUT` | 120 | Kernel preflight |
| `ADDITIONAL_SMOKE_TIMEOUT` | 600 | GPU smoke check |
| `ADDITIONAL_GPU_WAIT_SECONDS` | 600; fallback 60 | Total GPU-release polling budget |

Preflight timeout sends TERM, then KILL after five seconds if necessary. A timed
out snapshot check does not trigger a second long sealing operation. GPU polling
uses a total deadline rather than multiplying a slow query by 120 iterations;
its kill grace is two seconds. Non-numeric GPU memory readings cannot admit work.
The fallback removes the primary's external-keepalive flag because that primary
helper has already been stopped.
Missing model snapshots are rejected before spending time qualifying datasets;
qualification is still required once per registered matrix before any training.

These limits do not shorten GRPO steps, rollout budgets, artifact validation,
bootstrap samples or the existing NCCL collective timeout. They are not a fix for
an unhealthy CUDA driver or an uninterruptible kernel/storage operation.
