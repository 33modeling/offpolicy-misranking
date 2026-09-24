# Pair checkpoint/hash diagnosis

`checkpoint contract or hashes invalid` is a failed checkpoint validation,
not proof that a kernel lock is stuck. Its existing validator returns the same
failure for a changed contract, missing file, empty artifact, or hash mismatch.

## Standalone shell script

Download `scripts/diagnose_selector_pair_hashes.sh` and run it from the existing
worker checkout:

```bash
bash /path/to/diagnose_selector_pair_hashes.sh
```

This single file embeds the diagnostic; it does not download dependencies or
update the checkout. It selects `PAIR_PYTHON`, then the existing `VENV_DIR` or
`$OM_WORK/.venv-cu126`, then `python3`. It respects `PAIR_ROOT` and `OM_WORK`.
Use `--repo`, `--root`, or repeated `--directory` arguments to override paths.
Only the report is printed to stdout. Exit 1 means diagnostic issues were found;
exit 2 means the runtime could not be inspected. Neither triggers repair.
The embedded Python must remain identical to `selector_pair_checkpoint_audit.py`;
the tests check this and exercise a downloaded copy with unchanged source files.

## Python entry point

The read-only `scripts/selector_pair_checkpoint_audit.py` reports these causes
separately. Use the Python environment and runtime checkout of the failed worker:

```bash
PYTHONDONTWRITEBYTECODE=1 "$OM_WORK/.venv-cu126/bin/python" -B \
  scripts/selector_pair_checkpoint_audit.py --repo "$PWD" \
  --root "$OM_WORK/runs/selector-pair-v1"
```

The default inspects the two branches explicitly listed for supplemental
recovery in the existing code: On-policy seed 1/t50 and Random seed 4/t100.
This default does not establish that these are the user's two failing branches.
For a different branch, pass `--directory /absolute/path/to/selection_full`
(the directory containing `policy/`), repeating the option as needed.

The script writes its report only to stdout. It does not acquire or remove
locks, change metadata/hashes/results, select a replacement recovery checkpoint,
stop a process, start training, or execute the recovery workflow. It imports
the existing recovery helper only to construct the exact expected contract;
Python bytecode writes are disabled. Full policy lineage and server-side run
completion are not certified by this report.

Read `contract_differences` for incompatible fields, `issues` for missing or
corrupt files, `planned_checkpoint` for a stale pinned path, and
`runner_hash_matches` for a recovery-helper version mismatch. A newer valid
checkpoint is listed for inspection but never substituted for the pinned one.

No resume fix or hash compatibility exception is justified until the actual
failed paths and mismatch report are available. Existing experiment files must
remain unchanged; do not delete a plan or lock, regenerate hashes, or disable
validation to get past this error.

Validation: `python3 -B tests/test_selector_pair_checkpoint_audit.py`.

All ten diagnostic and standalone-launcher tests pass. The earlier broader recovery check reported 54
passing tests and one existing fixture failure:
`test_overrun_retries_immediately_after_original_hits_cap` writes an empty
`decision.json`, then raises `KeyError: budget_gpu_seconds`. This also fails
when run alone without importing the new diagnostic. Existing recovery code
and fixtures were not changed; the complete recovery suite is not passing.
