# Reference Sampling: Existing Cluster Checkout

Use the existing `offpolicy-misranking` repository on an idle allocated node.
No v2 repository, new virtual environment, or primary-worker restart is needed.

```bash
git pull --ff-only
bash scripts/run_reference_axes.sh math500 0
```

This starts five conditions, not a dry run. `--plan` prints the design without
starting Python or GPU work; `--check` validates through the existing launcher's
dry-run path. Source inputs and the model must be mounted on the compute node.
Real GPU execution has not been tested on the development machine.

The defaults come from the transferred cluster logs:

- Environment: `/group-volume/minsoo3.kim/offpolicy-misranking/.venv-cu126`.
- Model: `/group-volume/models/Olmo-3-1025-7B`.
- Input: `/group-volume/minsoo3.kim/offpolicy-misranking/runs/olmo3-1025-7b-base-rlzero-grpo-h100-v2`.
- New runs: `/group-volume/minsoo3.kim/offpolicy-misranking/runs/reference-axes`.
- New exports: `/group-volume/minsoo3.kim/offpolicy-misranking/exports/reference-axes`.

The `h100-v2` substring is the existing experiment tag recorded in the logs,
not the separate v2 code repository. The command overrides inherited v2/Qwen
Python, source, config and output settings inside its own process. The caller's
shell is unchanged. It preserves allocated `CUDA_VISIBLE_DEVICES` and the
existing node-local GPU admission lock. Do not run on top of a primary worker.

Conditions `(fresh_k, val_k)` are `(32,8), (64,8), (128,8), (32,16), (32,32)`.
They use the same independent seed formula as the documented paper design.
Replicates are 0..4; use a different replicate or dataset on each additional
node. The common baseline is run once per replicate. One condition's failure
does not block the remaining conditions; rerunning resumes completed work.
Final exit status reports any failures. INT/TERM is forwarded to the active
existing launcher, which owns and cleans up its stage children.

For other installations only, explicit `REFERENCE_WORK`, `REFERENCE_VENV`,
`REFERENCE_MODEL`, and `REFERENCE_SOURCE_ROOT` override those defaults. Generic
inherited `OM_WORK`/`VENV_DIR`/`OM_OLMO3_ROOT` do not silently select another
project. No packages are installed or updated by this entry point.

This change moves the immediate reference-experiment execution path into the
operational repository. It does not replace the existing E5/E6 launchers with
the separately developed E5/E6 implementation or change primary scheduling.

## Status and Diagnostic Logs

From the existing operational checkout:

```bash
git pull --ff-only
bash scripts/run_olmo3_rlzero.sh status h100
bash scripts/why.sh
```

These are read-only inspection commands; do not restart a productive worker
to refresh its status. Status includes primary work and reference experiments
in one snapshot, appended under a lock to the primary `logs/status-history.log`.
`status_checkout` identifies the reporting code, while `primary_generation`
identifies the pinned primary experiment code. They need not match.

New reference invocations save their complete launcher console under
`$OM_WORK/console-logs/reference-*.log` and atomic state records under
`$OM_WORK/reference-workers/`. Records include node, PID, dataset, replicate,
condition, seed, output directory and final exit state. Remote PID liveness
is unverified; a RUNNING record alone does not establish current GPU activity.
Older invocations without records remain UNKNOWN, not stopped. Stage tails
show file activity, not proof that an error belongs to the latest attempt.
Stage/shard ETAs are not whole-experiment completion estimates.

`why.sh` writes one report under `$OM_WORK/exports/why-*.txt`, including reference
status. Its default error section uses the latest modified attempt log, not
the highest attempt number. Historical stage errors are not automatically
attributed to that attempt. To include explicitly labelled older evidence:

```bash
WHY_HISTORY=1 bash scripts/why.sh
```

The separately prepared `run_mbpp_hotfix.sh` is not called by any of these
commands. Its `go` mode starts a new five-seed MBPP matrix without reusing
the existing matrix outputs; it is not a status/logging update.
