# Blocked primary work and independent experiment rotation

The OLMo launcher preserves blocked family checkpoints and rollouts. It finishes
other eligible primary families first. When every remaining selected primary
family is marked LOOPING, it does not exit or collect a partial matrix as complete.
It stops its own heartbeat/keepalive, releases the node's primary lock, and hands
the process to `scripts/run_available_experiments.sh` from the supervisor snapshot.

## Independent work

The default registered profile order is:

```text
olmo3_domains qwen35_2b qwen35_4b qwen35 qwen38
```

Override the selection before launching OLMo when only some snapshots are prepared:

```bash
OM_RLZERO_FALLBACK_PROFILES="olmo3_domains qwen35_2b" \
  bash scripts/run_olmo3_rlzero.sh run h100
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

The primary remains incomplete. Fix its recorded cause and deliberately relaunch
it with `OM_RLZERO_CLEAR_LOOPS=1` to resume preserved artifacts. Do not launch it on
a node while an independent experiment is using the GPUs. This handoff does not
automatically resume a blocked primary while the independent matrix runs.

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
primary-lock/heartbeat handoff. Real H100 execution and snapshot availability
must be checked on the cluster; local tests use fake GPU/model launchers.
