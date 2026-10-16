"""Display recorded GPU costs without changing experiment runtimes or receipts."""

import copy
import json
import math
import statistics
from pathlib import Path

from srgc_rebuttal.cost_ledger import PhaseLedger
from srgc_rebuttal.cost_report import seed_costs
from srgc_rebuttal.plan import input_path

COMPONENTS = ("preparation", "selection", "training", "evaluation", "checkpoint", "startup")
KEYS = (*COMPONENTS, "primary", "phases")
ARM_FIELDS = {key: f"{key}_gpu_seconds" for key in COMPONENTS}
ARM_FIELDS.update(primary="selection_training_preparation_gpu_seconds",
                  phases="continuation_phases_gpu_seconds")
EXECUTION_SETTINGS = ("world_size", "generation_micro_batch", "logprob_micro_batch", "logit_chunk_tokens",
                      "gradient_checkpointing", "attention", "max_new_tokens", "responses")
STAGE_COLUMNS = (("Gen", ("generation",)), ("Fwd", ("forward",)), ("Bwd", ("backward",)),
                 ("Project", ("gradient_projection",)), ("Comm", ("communication", "gradient_reduction")),
                 ("Residual", ("unattributed_and_wait",)))
COUNT_COLUMNS = (("Prompts", "prompts"), ("Responses", "responses"), ("Tokens", "generated_tokens"),
                 ("FwdResp", "forward_responses"), ("BwdResp", "backward_responses"))


def finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("GPU cost must be numeric")
    if not math.isfinite(value) or value < 0:
        raise ValueError("GPU cost must be a finite nonnegative measurement")
    return float(value)


def timing(record="MISSING", values=None, *, error=None, accounting="exclusive phase costs", details=None):
    values = values or {}
    seconds = {key: None if values.get(key) is None else finite(values[key]) for key in KEYS}
    result = {"record": record, "gpu_seconds": seconds,
              "gpu_hours": {key: None if value is None else value / 3600 for key, value in seconds.items()},
              "lower_bound": record == "PARTIAL", "accounting": accounting}
    if error:
        result["error"] = str(error)
    if details is not None and record in {"COMPLETE", "PARTIAL"}:
        for field in ("exclusive_stages", "counts"):
            if field not in details:
                continue
            measurements = details[field]
            if not isinstance(measurements, dict):
                raise TypeError(f"{field} must be an object")
            for name, value in measurements.items():
                if not isinstance(name, str) or "." not in name:
                    raise ValueError(f"{field} must identify the recorded phase and metric")
                if field == "exclusive_stages":
                    if not isinstance(value, dict):
                        raise TypeError("stage measurement must be an object")
                    finite(value["gpu_seconds"])
                    for key in ("rank_wall_seconds", "calls"):
                        if key in value:
                            finite(value[key])
                else:
                    finite(value)
            result[field] = copy.deepcopy(measurements)
        if "exclusive_stages" in result:
            result["exclusive_stage_gpu_hours"] = {
                name: value["gpu_seconds"] / 3600 for name, value in result["exclusive_stages"].items()}
    return result


def from_arm(cost, *, completed):
    if not cost:
        return timing()
    complete = bool(completed and cost.get("complete"))
    known = cost.get("known_completed_phase_gpu_seconds", {})
    if not isinstance(known, dict):
        raise TypeError("completed phase costs must be an object")
    known = {key: finite(value) for key, value in known.items()}
    if complete:
        return timing("COMPLETE", {key: cost.get(field) for key, field in ARM_FIELDS.items()}, details=cost)
    if not any(known.values()) and not cost.get("unfinished_phases") and not any(
            cost.get(field) for field in ("exclusive_stages", "counts")):
        return timing()
    values = {key: known.get(f"{key}_gpu_seconds") for key in COMPONENTS}
    values["checkpoint"] = sum(known.get(f"{key}_gpu_seconds", 0) for key in
                                ("checkpoint_load", "checkpoint_save"))
    # Zero defaults in the frozen ledger do not prove a missing phase cost is zero.
    values = {key: value if value else None for key, value in values.items()}
    primary = sum(known.get(f"{key}_gpu_seconds", 0) for key in ("preparation", "selection", "training"))
    total = sum(known.values())
    values.update(primary=primary if primary else None, phases=total if total else None)
    return timing("PARTIAL", values, details=cost)


def from_ledger(ledger, *, completed, accounting="exclusive phase costs"):
    if not ledger or not ledger.get("recorded_phases"):
        return timing(accounting=accounting)
    known = ledger.get("known_gpu_seconds", {})
    if not isinstance(known, dict):
        raise TypeError("recorded GPU costs must be an object")
    known = {key: finite(value) for key, value in known.items()}
    complete = bool(completed and ledger.get("complete"))
    values = {key: known.get(f"{key}_gpu_seconds", 0 if complete else None) for key in COMPONENTS}
    values["checkpoint"] = sum(known.get(f"{key}_gpu_seconds", 0) for key in
                                ("checkpoint_load", "checkpoint_save")) if complete or any(
        f"{key}_gpu_seconds" in known for key in ("checkpoint_load", "checkpoint_save")) else None
    values["primary"] = sum(known.get(f"{key}_gpu_seconds", 0) for key in
                            ("preparation", "selection", "training")) if complete or any(
        known.get(f"{key}_gpu_seconds") for key in ("preparation", "selection", "training")) else None
    values["phases"] = ledger.get("total_gpu_seconds") if complete else sum(known.values()) or None
    return timing("COMPLETE" if complete else "PARTIAL", values, accounting=accounting, details=ledger)


def invalid(row, message=None):
    row["timing"] = timing("INVALID", error=message)
    return row


def attach_scope(row, directory, *, arm=None):
    """Extra/research/information tasks meter their own separate output scopes."""
    if row.get("status", row.get("state")) == "invalid":
        return invalid(row)
    directory = Path(directory)
    completed = row.get("status", row.get("state")) in {"complete", "DONE"}
    try:
        ledger_path = directory / "cost-receipts"
        if arm is not None:
            ledger_path = ledger_path / arm
        ledger = PhaseLedger(ledger_path).totals()
        endpoint_path = directory / (f"{arm}-endpoint.json" if arm is not None else "endpoint.json")
        endpoint = json.loads(endpoint_path.read_text()) if endpoint_path.is_file() else {}
        if not isinstance(endpoint, dict):
            raise TypeError("timing endpoint must be an object")
        recorded = endpoint.get("cost_receipts")
        if not ledger["recorded_phases"] and isinstance(recorded, dict):
            ledger = recorded
        measured_complete = endpoint.get("cost_measurement_complete", ledger.get("complete", False))
        row["timing"] = from_ledger(ledger, completed=completed and measured_complete)
    except (OSError, ValueError, KeyError, TypeError) as error:
        invalid(row, error)
    return row


def attach(report, plan_path, plan):
    """Use the same validated cost reader as results; incomplete costs stay partial."""
    report["recorded_execution_settings"] = {key: copy.deepcopy(plan[key]) for key in EXECUTION_SETTINGS if key in plan}
    by_seed = {row["seed"]: row for row in report.get("costs", [])}
    root = Path(report["output_root"])
    shared = []
    for seed in plan["seeds"]:
        rows = [row for row in report["tasks"] if row["seed"] == seed]
        if any(row.get("status") == "invalid" for row in rows) or any(
                f"seed {seed} " in error or f"seed {seed}:" in error for error in report.get("errors", [])):
            for row in rows:
                invalid(row, "input, endpoint or cost identity is invalid")
            continue
        try:
            cost = by_seed.get(seed)
            if cost is None:
                cost = seed_costs(root / f"seed-{seed}", input_path(plan_path, plan, seed), plan["arms"])
            for row in rows:
                completed = row["status"] == "complete"
                if row["arm"] == "cache":
                    cache = cost.get("cache_build") or {}
                    row["timing"] = from_ledger(cache.get("invocations"),
                        completed=completed and cache.get("complete", False),
                        accounting="inclusive cache invocation; shared once per seed")
                elif row["arm"] == "prefix":
                    row["timing"] = from_ledger(cost.get("shared_prefix"), completed=completed,
                        accounting="shared prefix; separate from continuation arms")
                else:
                    row["timing"] = from_arm(cost.get("arms", {}).get(row["arm"]), completed=completed)
            ledger = cost.get("shared_all_task_startup_and_load")
            if ledger:
                shared.append({"seed": seed, "arm": "shared-startup",
                    "timing": from_ledger(ledger,
                        completed=all(row["status"] == "complete" for row in rows),
                        accounting="shared startup/load; charged once, never per arm")})
        except (OSError, ValueError, KeyError, TypeError) as error:
            for row in rows:
                invalid(row, error)
            report.setdefault("warnings", []).append(f"seed {seed} GPU timing unavailable: {error}")
    report["shared_timing"] = shared
    report["timing_units"] = "allocated GPU-seconds and GPU-hours; 1 GPU-hour = 3600 GPU-seconds"
    report["timing_accounting"] = (
        "primary = preparation + selection + training; phases include evaluation, checkpoints and startup; "
        "cache and prefix are shared once; inclusive invocations are never added to phase totals; "
        "partial values are lower bounds from finished receipts, not live elapsed estimates")
    return report


def table_lines(rows):
    headers = ("Seed", "Arm", "Record", "Prep", "Select", "Train", "Eval", "Check", "Start", "S+T+P", "Phases")
    cells = []
    for row in rows:
        value = row.get("timing", timing())
        if row.get("status", row.get("state")) == "invalid":
            value = timing("INVALID")
        numbers = ["-" if value["gpu_hours"].get(key) is None else
                   (">=" if value["lower_bound"] else "") + f"{value['gpu_hours'][key]:.3f}" for key in KEYS]
        cells.append([str(row.get("seed", "-")), str(row.get("arm", row.get("task", "-"))), value["record"], *numbers])
    return aligned_table(headers, cells)


def aligned_table(headers, cells):
    widths = [max([len(str(header)), *(len(str(row[i])) for row in cells)]) for i, header in enumerate(headers)]
    return ["  ".join(str(cell).ljust(width) for cell, width in zip(headers, widths)),
            "  ".join("-" * width for width in widths),
            *("  ".join(str(cell).ljust(width) for cell, width in zip(row, widths)) for row in cells)]


def detail_lines(rows):
    """Show existing measurements by phase/section, without adding them to totals."""
    stages, counts = [], []
    for row in rows:
        value = row.get("timing", {})
        if row.get("status", row.get("state")) == "invalid" or value.get("record") not in {"COMPLETE", "PARTIAL"}:
            continue
        prefix = ">=" if value["lower_bound"] else ""
        identity = [row.get("seed", "-"), row.get("arm", row.get("task", "-")), value["record"]]
        scopes = {}
        for name, measurement in value.get("exclusive_stages", {}).items():
            scope, stage = name.rsplit(".", 1)
            scopes.setdefault(scope, {})[stage] = measurement["gpu_seconds"]
        covered = {name for _, names in STAGE_COLUMNS for name in names}
        for scope, metrics in sorted(scopes.items()):
            groups = [names for _, names in STAGE_COLUMNS] + [tuple(name for name in metrics if name not in covered)]
            cells = ["-" if not any(name in metrics for name in names) else
                     prefix + f"{sum(metrics[name] for name in names if name in metrics) / 3600:.3f}" for names in groups]
            stages.append([*identity, scope, *cells])
        scopes = {}
        for name, measurement in value.get("counts", {}).items():
            scope, counter = name.rsplit(".", 1)
            scopes.setdefault(scope, {})[counter] = measurement
        covered = {name for _, name in COUNT_COLUMNS}
        for scope, metrics in sorted(scopes.items()):
            cells = ["-" if name not in metrics else prefix + str(metrics[name]) for _, name in COUNT_COLUMNS]
            other = ", ".join(f"{name}={prefix}{count}" for name, count in sorted(metrics.items()) if name not in covered)
            counts.append([*identity, scope, *cells, other or "-"])
    lines = []
    if stages:
        lines += ["", "GPU STAGES (GPU-hours; recorded phase/section only)",
                  *aligned_table(("Seed", "Arm", "Record", "Phase/section", *(name for name, _ in STAGE_COLUMNS), "Other"), stages),
                  "Comm includes gradient reduction; Residual = unattributed_and_wait, not measured GPU idle time.",
                  "Other sums remaining recorded stages. Stage coverage may be incomplete; details/calls are retained in JSON."]
    if counts:
        lines += ["", "RECORDED WORK (finished receipts; prompt/response counts include repeated exposures)",
                  *aligned_table(("Seed", "Arm", "Record", "Phase/section", *(name for name, _ in COUNT_COLUMNS), "Other"), counts)]
    return lines


def render(report, *, label=None):
    rows = [*report.get("tasks", report.get("rows", [])), *report.get("shared_timing", [])]
    if not rows:
        return []
    lines = ["", f"{label or report.get('label', report.get('dataset', ''))} GPU TIMES (GPU-hours)".strip(),
             *table_lines(rows)]
    settings = report.get("recorded_execution_settings", {})
    for keys in (EXECUTION_SETTINGS[:4], EXECUTION_SETTINGS[4:]):
        recorded = [f"{key}={json.dumps(settings[key], ensure_ascii=False, sort_keys=True)}" for key in keys if key in settings]
        if recorded:
            lines.append("RECORDED PLAN SETTINGS: " + " | ".join(recorded))
    lines.extend(detail_lines(rows))
    for row in rows:
        if row.get("timing", {}).get("error"):
            lines.append(f"TIME ERROR seed {row.get('seed', '-')} {row.get('arm', row.get('task', '-'))}: "
                         f"{row['timing']['error']}")
    arms = sorted({row["arm"] for row in rows if row.get("arm") not in {None, "cache", "prefix", "shared-startup"}})
    means = []
    for arm in arms:
        selected = [row for row in rows if row.get("arm") == arm]
        if not selected or any(row.get("status", row.get("state")) == "invalid" or
                row.get("timing", {}).get("record") != "COMPLETE" or
                row["timing"]["gpu_hours"]["primary"] is None for row in selected):
            continue
        values = [row["timing"]["gpu_hours"]["primary"] for row in selected]
        sd = statistics.stdev(values) if len(values) > 1 else None
        means.append(f"{arm} {statistics.mean(values):.3f}" + (f" +/- {sd:.3f}" if sd is not None else "") + f" (n={len(values)})")
    if means:
        lines.append("S+T+P mean +/- sample SD (GPU-h): " + " | ".join(means))
    lines += ["COMPLETE = measured scope; PARTIAL/>= = finished-receipt lower bound; - = unmeasured; INVALID = unreadable/invalid timing.",
              "S+T+P = prep + selection + training. Phases include evaluation/checkpoints/startup; cache/prefix are separate shared costs.",
              "GPU-h is allocated GPU time, not wall hours; nested invocation and phase totals are never added."]
    return lines
