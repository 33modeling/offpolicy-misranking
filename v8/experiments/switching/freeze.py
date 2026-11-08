"""Freeze artifacts: budget (cost-only), frozen study, sampled schedules (before test)."""

from __future__ import annotations

from pathlib import Path

from .common import read_json, sha256_file, sha256_json, utc_now, write_once_json


def freeze_budget(rows, *, policy_id="srgc_current", scope="cold_start_algorithm", files_read=()):
    """B from cost-only data. Refuses if any evaluation file was read to compute it."""
    if any("evaluations" in Path(f).parts for f in files_read):
        raise ValueError("budget freeze must not read evaluation outputs")
    picks = [r for r in rows if r["policy_id"] == policy_id and r.get("status") in ("triggered", "not_triggered")]
    if not picks:
        raise ValueError(f"no {policy_id} rows with composed costs")
    key = "cold_start_gpu_seconds" if scope == "cold_start_algorithm" else "continuation_gpu_seconds"
    values = [r.get(key) for r in picks]
    used_scope = scope
    if any(v is None for v in values):
        values = [r.get("continuation_gpu_seconds") for r in picks]
        used_scope = "continuation_algorithm (cold-start incomplete)"
    if any(v is None for v in values):
        raise ValueError("budget cannot be frozen: composed costs are incomplete")
    inputs = [{"seed": r.get("seed"), "run_id": r.get("run_id"), "switch_update": r.get("switch_update"),
               "continuation_gpu_seconds": r.get("continuation_gpu_seconds"),
               "cold_start_gpu_seconds": r.get("cold_start_gpu_seconds")} for r in picks]
    return {"policy_id": policy_id, "requested_scope": scope, "scope": used_scope, "unit": "gpu_seconds",
            "value": sum(values) / len(values), "n_seeds": len(values), "per_seed": values,
            "inputs_sha256": sha256_json(sorted(inputs, key=lambda x: (str(x["seed"]), str(x["run_id"])))),
            "rule": "mean composed cost of srgc_current at the matched horizon; frozen before evaluation outputs",
            "frozen_at": utc_now()}


SEMANTIC_BUDGET_FIELDS = ("policy_id", "requested_scope", "scope", "unit", "value", "n_seeds", "per_seed",
                          "inputs_sha256", "rule", "gpu_type", "gpu_count")


def write_frozen_budget(path, budget):
    """Write-once with idempotent re-runs: identical semantics -> reuse the ORIGINAL file (frozen_at, B)."""
    path = Path(path)
    if path.exists():
        old = read_json(path)
        diff = [k for k in SEMANTIC_BUDGET_FIELDS if old.get(k) != budget.get(k)]
        if diff:
            raise ValueError(f"frozen budget {path} already exists with different content in {diff}; "
                             "a changed budget needs a new study version")
        return old, "reused"
    write_once_json(path, budget)
    return budget, "frozen"


def freeze_study(path, *, cfgs, seeds, tuning_path=None, budget_path=None, schedules=None, code_identity=None,
                 audit_path=None):
    body = {"cohorts": {c: {"config": cfg, "config_hash": sha256_json(cfg)} for c, cfg in cfgs.items()},
            "seeds": seeds, "code_identity": code_identity,
            "frozen_tuning_sha256": sha256_file(tuning_path) if tuning_path else None,
            "frozen_budget_sha256": sha256_file(budget_path) if budget_path else None,
            "sampled_schedules": schedules, "audit_sha256": sha256_file(audit_path) if audit_path else None,
            "frozen_at": utc_now()}
    if tuning_path:
        tuning = read_json(tuning_path)
        if tuning.get("test_access_status") != "clean":
            raise ValueError("frozen tuning is not clean; the study cannot be frozen for confirmatory test")
    write_once_json(path, body)
    return body
