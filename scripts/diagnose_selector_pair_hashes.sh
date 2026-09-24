#!/usr/bin/env bash
# Standalone read-only Pair checkpoint/hash diagnosis.
# Embedded Python must match scripts/selector_pair_checkpoint_audit.py.
# No downloads, git updates, locks, training, or experiment writes.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PAIR_AUDIT_WORK="${OM_WORK:-/group-volume/${OM_USER:-minsoo3.kim}/offpolicy-misranking}"
PAIR_AUDIT_ROOT="${PAIR_ROOT:-$PAIR_AUDIT_WORK/runs/selector-pair-v1}"
PAIR_AUDIT_REPO="$PWD"
if [[ ! -f "$PAIR_AUDIT_REPO/scripts/mbpp_budget_recovery.py" ]]; then
    PAIR_AUDIT_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    if [[ -f "$PAIR_AUDIT_SCRIPT_DIR/../scripts/mbpp_budget_recovery.py" ]]; then
        PAIR_AUDIT_REPO="$(cd -- "$PAIR_AUDIT_SCRIPT_DIR/.." && pwd)"
    else
        PAIR_AUDIT_REPO="$PAIR_AUDIT_WORK"
    fi
fi
if [[ -n "${PAIR_PYTHON:-}" ]]; then
    PAIR_AUDIT_PYTHON="$PAIR_PYTHON"
elif [[ -x "${VENV_DIR:-$PAIR_AUDIT_WORK/.venv-cu126}/bin/python" ]]; then
    PAIR_AUDIT_PYTHON="${VENV_DIR:-$PAIR_AUDIT_WORK/.venv-cu126}/bin/python"
else
    PAIR_AUDIT_PYTHON=python3
fi
pair_run_audit() {
"$PAIR_AUDIT_PYTHON" -B - --repo "$PAIR_AUDIT_REPO" --root "$PAIR_AUDIT_ROOT" "$@" <<'PAIR_HASH_AUDIT_PYTHON'
#!/usr/bin/env python3
"""Read-only diagnosis of Pair checkpoint contract/hash failures.

No recovery, locks, training, metadata repair, or experiment writes. The
expected contract is built by the existing recovery helper in --repo.
Run using the same Python environment and checkout as the failed worker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True

FILES = {
    "adapter_model.safetensors": "adapter_sha256",
    "optimizer.pt": "optimizer_sha256",
    "grpo_stats.jsonl": "grpo_stats_sha256",
}
DEFAULT_BRANCHES = (
    "branches/on_policy/states/s1-t50/points/view-50/selection_reduced",
    "branches/on_policy/states/s4-t100/points/view-100/random_full",
)


def read_json(path):
    with path.open("rb") as handle:
        raw = handle.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise ValueError(f"metadata exceeds 1 MiB: {path}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"metadata is not an object: {path}")
    return value


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def differences(saved, expected, prefix=""):
    result = []
    for key, value in expected.items():
        name = f"{prefix}.{key}" if prefix else key
        if key not in saved:
            result.append({"field": name, "problem": "missing", "expected": value})
        elif isinstance(value, dict) and isinstance(saved[key], dict):
            result.extend(differences(saved[key], value, name))
            for extra in sorted(saved[key].keys() - value.keys()):
                result.append({"field": f"{name}.{extra}", "problem": "unexpected",
                               "recorded": saved[key][extra]})
        elif saved[key] != value:
            result.append({"field": name, "problem": "mismatch",
                           "recorded": saved[key], "expected": value})
    return result


def checkpoint_report(path, expected):
    report = {"checkpoint": str(path), "issues": []}
    names = ["checkpoint_state.json", "adapter_config.json", *FILES]
    before = {}
    for name in names:
        try:
            info = (path / name).stat()
            before[name] = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
        except OSError as exc:
            report["issues"].append({"file": name, "problem": type(exc).__name__})
    try:
        state = read_json(path / "checkpoint_state.json")
    except (OSError, ValueError) as exc:
        report["issues"].append({"file": "checkpoint_state.json", "problem": str(exc)})
        return report
    report["completed_steps"] = state.get("completed_steps")
    report["contract_differences"] = differences(state, expected)
    if type(state.get("completed_steps")) is not int:
        report["issues"].append({"field": "completed_steps", "problem": "not an integer"})
    elif not expected["start_step"] < state["completed_steps"] <= expected["target_steps"]:
        report["issues"].append({"field": "completed_steps", "problem": "outside frozen interval"})
    for name in names[1:]:
        if name not in before:
            continue
        if before[name][2] == 0:
            report["issues"].append({"file": name, "problem": "empty"})
        if name not in FILES:
            continue
        try:
            actual = digest(path / name)
            recorded = state.get(FILES[name])
            if actual != recorded:
                report["issues"].append({"file": name, "problem": "hash mismatch",
                                         "recorded": recorded, "actual": actual})
        except OSError as exc:
            report["issues"].append({"file": name, "problem": type(exc).__name__})
    for name, signature in before.items():
        try:
            info = (path / name).stat()
            after = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
        except OSError:
            after = None
        if signature != after:
            report["issues"].append({"file": name, "problem": "changed during read; inconclusive"})
    report["contract_and_hashes_match"] = not report["issues"] and not report["contract_differences"]
    return report


def branch_report(directory, contract_builder, runner):
    report = {"directory": str(directory), "issues": [], "checkpoints": []}
    out = directory.parent
    try:
        contract = read_json(out / "contract.json")
        expected = contract_builder(out, contract, directory.name)
        policy = (directory / "policy").resolve()
        plan_path = directory / "budget-recovery/plan.json"
        selected = None
        if plan_path.is_file():
            plan = read_json(plan_path)
            selected = Path(plan["points"][-1]["adapter"]).resolve()
            report["planned_checkpoint"] = str(selected)
            report["runner_hash_matches"] = plan.get("runner_sha256") == digest(runner)
            if selected != policy and selected.parent != policy:
                raise ValueError("plan selects a checkpoint outside the original policy")
        candidates = sorted(
            (p for p in policy.glob("checkpoint-*") if p.name[11:].isdigit()),
            key=lambda p: int(p.name[11:]), reverse=True)[:2]
        paths = list(dict.fromkeys(([selected] if selected else []) + candidates))
        if not paths:
            report["issues"].append("no saved checkpoint or recovery plan found")
        for path in paths:
            if path == policy and (policy / "policy_train.json").is_file():
                report["issues"].append("plan selects final policy; use policy lineage diagnostics")
                continue
            report["checkpoints"].append(checkpoint_report(path, expected))
        report["recovery_result_exists"] = (directory / "budget-recovery/result.json").is_file()
        report["canonical_result_exists"] = (directory / "result.json").is_file()
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        report["issues"].append(f"{type(exc).__name__}: {exc}")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd(),
                        help="checkout/runtime used by the failed worker")
    parser.add_argument("--root", type=Path,
                        default=Path(os.environ.get("PAIR_ROOT", os.environ.get("OM_WORK",
                            "/group-volume/minsoo3.kim/offpolicy-misranking") + "/runs/selector-pair-v1")))
    parser.add_argument("--directory", type=Path, action="append",
                        help="specific failed branch; repeat for multiple branches")
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    sys.path[:0] = [str(repo / "scripts"), str(repo / "src")]
    try:
        import mbpp_budget_recovery as recovery
        runner = repo / "scripts/selector_pair_budget_recovery.py"
        directories = args.directory or [args.root / name for name in DEFAULT_BRANCHES]
        rows = [branch_report(p.resolve(), recovery.checkpoint_contract, runner) for p in directories]
        print(json.dumps({"read_only": True, "scope": "checkpoint contract and hash diagnosis, not full lineage certification",
                          "runtime": str(repo), "branches": rows}, indent=2))
        return 1 if any(row["issues"] or row.get("runner_hash_matches") is False or
                        any(not p.get("contract_and_hashes_match", False) for p in row["checkpoints"])
                        for row in rows) else 0
    except (ImportError, OSError, ValueError) as exc:
        print(f"Cannot inspect runtime: {type(exc).__name__}: {exc}. "
              "Use the failed worker's Python environment and --repo. No run files changed.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
PAIR_HASH_AUDIT_PYTHON
}

# Always create a new report outside the experiment; never overwrite a file.
PAIR_AUDIT_REPORT="$(mktemp /tmp/pair-hash-diagnostic-XXXXXXXX.txt)"
printf '[report] %s\n' "$PAIR_AUDIT_REPORT" >&2
set +e
pair_run_audit "$@" 2>&1 | tee -- "$PAIR_AUDIT_REPORT"
PAIR_AUDIT_STATUS=("${PIPESTATUS[@]}")
set -e
if [[ "${PAIR_AUDIT_STATUS[1]}" -ne 0 ]]; then
    printf '[report] TXT write failed: %s\n' "$PAIR_AUDIT_REPORT" >&2
    exit "${PAIR_AUDIT_STATUS[1]}"
fi
printf '[report] saved: %s (diagnostic exit %s)\n' "$PAIR_AUDIT_REPORT" "${PAIR_AUDIT_STATUS[0]}" >&2
exit "${PAIR_AUDIT_STATUS[0]}"
