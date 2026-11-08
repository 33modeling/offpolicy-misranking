"""Configuration: base -> cohort -> arm merge with visible overrides, hashing and fail-fast validation."""

from __future__ import annotations

import copy
from pathlib import Path

from .common import read_json, sha256_json

HERE = Path(__file__).resolve().parents[1]
CONFIGS = HERE / "configs"

# Fields that must be non-null before a confirmatory (test) run; development needs a subset.
REQUIRED = {
    "test": ["model.repository", "model.revision", "data.train_pool_hash", "data.ranking_reference_hash",
             "data.development_eval_hash", "data.final_eval_hash", "budget.value", "budget.gpu_type",
             "budget.gpu_count", "seeds.training", "analysis.tie_tolerance_pp", "tuning.frozen_tuning_sha256",
             "seeds.unused_test_seed_audit"],
    "development": ["model.repository", "model.revision", "data.train_pool_hash", "data.final_eval_hash",
                    "seeds.training"],
    "rebuttal_bank": ["model.repository", "model.revision", "seeds.training"],
    "pilot": [], "smoke": [],
}


def deep_merge(base, overlay, *, source, path="", overrides=None):
    overrides = [] if overrides is None else overrides
    out = copy.deepcopy(base)
    for k, v in overlay.items():
        key = f"{path}.{k}" if path else k
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v, source=source, path=key, overrides=overrides)
        else:
            if k in out and out[k] is not None and out[k] != v:
                overrides.append({"field": key, "from": out[k], "to": v, "source": source})
            out[k] = copy.deepcopy(v)
    return out


def load_cohort(cohort_id):
    return read_json(CONFIGS / "cohorts" / f"{cohort_id}.json")


def resolve(cohort_id, *, role, extra=None, base=None):
    base = base or read_json(CONFIGS / "base.json")
    overrides = []
    cfg = deep_merge(base, load_cohort(cohort_id), source=f"cohorts/{cohort_id}.json", overrides=overrides)
    if extra:
        cfg = deep_merge(cfg, extra, source="arm/cli", overrides=overrides)
    cfg["identity"] = dict(cfg.get("identity", {}), cohort_id=cohort_id, role=role)
    cfg["resolved_overrides"] = overrides
    return cfg


def config_hash(cfg) -> str:
    return sha256_json({k: v for k, v in cfg.items() if k not in ("resolved_overrides", "status", "notes")})


def get(cfg, dotted):
    cur = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def validate(cfg, *, mode) -> dict:
    errors, warnings = [], []
    for field in REQUIRED[mode]:
        if get(cfg, field) in (None, "", [], "pending"):
            errors.append(f"required field is null/pending for {mode}: {field}")
    t = cfg.get("training", {})
    if t.get("matched_update_endpoint") is not None and t.get("prefix_updates") is not None:
        if t["matched_update_endpoint"] <= t["prefix_updates"]:
            errors.append("horizon must exceed the prefix")
    if get(cfg, "budget.value") is not None and get(cfg, "budget.unit") != "gpu_seconds" and get(cfg, "budget.unit") != "gpu_hours":
        errors.append("budget unit must be gpu_seconds or gpu_hours")
    for grid in ("bank.core_grid", "bank.extended_grid"):
        for tau in get(cfg, grid) or []:
            if tau % t.get("selection_interval", 25) or not t.get("prefix_updates", 25) <= tau < t.get("matched_update_endpoint", 275):
                errors.append(f"{grid}: {tau} must be a refresh boundary in [prefix, horizon)")
    if mode == "test":
        if get(cfg, "data.ranking_reference_hash") and get(cfg, "data.ranking_reference_hash") == get(cfg, "data.final_eval_hash"):
            errors.append("final evaluation split equals the ranking reference")
        if get(cfg, "tuning.test_access_status") not in ("clean",):
            errors.append("frozen tuning test_access_status is not clean")
    for o in cfg.get("resolved_overrides", []):
        warnings.append(f"override {o['field']}: {o['from']!r} -> {o['to']!r} ({o['source']})")
    return {"mode": mode, "ok": not errors, "errors": errors, "warnings": warnings, "config_hash": config_hash(cfg)}
