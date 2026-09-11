"""CPU-only pre-selection decisions, durable state, and compute accounting."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import struct
from dataclasses import asdict, dataclass
from pathlib import Path

SCHEMA = "offpolicy-selection-gate/one-shot-v1"
FEATURES = ("success_rate", "success_rate_std", "mixed_group_fraction",
            "zero_success_fraction", "all_success_fraction",
            "success_rate_p25", "success_rate_p50", "success_rate_p75")
TARGET = "full_horizon_selection_after_measurement_minus_no_gate_random"


def number(value, name, low=None, high=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric data")
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f"{name} outside [{low}, {high}]")
    return float(value)


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def read(path):
    def invalid(value):
        raise ValueError(f"nonfinite JSON constant: {value}")
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with tmp.open("w") as handle:
            handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        tmp.unlink(missing_ok=True)


def validate_scope(scope):
    keys = {"model", "dataset", "selector", "verifier", "pool_sha256", "gpu_type"}
    if not isinstance(scope, dict) or set(scope) != keys or any(
            not isinstance(v, str) or not v.strip() for v in scope.values()):
        raise ValueError(f"scope needs exactly {sorted(keys)}")
    return scope


def compatible_scope(left, right):
    """A fitted dataset-level gate may see a different eligible prompt pool."""
    validate_scope(left)
    validate_scope(right)
    return all(left[k] == right[k] for k in left if k != "pool_sha256")


def validate_model(model):
    if model.get("schema") != SCHEMA or model.get("target") != TARGET:
        raise ValueError("unsupported gate model or prediction target")
    if model.get("data_kind") not in {"observed", "synthetic"}:
        raise ValueError("model needs observed/synthetic provenance")
    number(model.get("budget_gpu_seconds"), "model training budget", 1e-12)
    validate_scope(model.get("scope"))
    if model.get("model_id") != fingerprint({k: v for k, v in model.items() if k != "model_id"}):
        raise ValueError("gate model hash changed")
    names = model.get("features")
    if not isinstance(names, list) or not names or len(set(names)) != len(names) or set(names)-set(FEATURES):
        raise ValueError("unsupported or duplicate gate features")
    nodes = model.get("nodes")
    if not isinstance(nodes, list) or not 1 <= len(nodes) <= 7:
        raise ValueError("gate must be a tree of depth at most two")
    visited = set()

    def visit(idx, depth):
        integer(idx, "node index")
        if idx >= len(nodes) or idx in visited or depth > 2:
            raise ValueError("invalid gate tree structure")
        visited.add(idx)
        node = nodes[idx]
        if "value" in node:
            if set(node) != {"value"}:
                raise ValueError("ambiguous leaf")
            number(node["value"], "predicted advantage", -1, 1)
        else:
            if set(node) != {"feature", "threshold", "left", "right"} or node["feature"] not in names:
                raise ValueError("invalid tree split")
            number(node["threshold"], "tree threshold")
            visit(node["left"], depth + 1)
            visit(node["right"], depth + 1)
    visit(0, 0)
    if len(visited) != len(nodes):
        raise ValueError("unreachable tree nodes")
    ranges = model.get("feature_ranges", {})
    if set(ranges) != set(names):
        raise ValueError("missing training feature support")
    for name in names:
        lo, hi = ranges[name]
        number(lo, name, 0, 1)
        number(hi, name, lo, 1)
    return model


def predict(model, values):
    validate_model(model)
    converted = {}
    for name in model["features"]:
        value = number(values.get(name), name, 0, 1)
        lo, hi = model["feature_ranges"][name]
        if not lo <= value <= hi:
            raise ValueError(f"outside development support: {name}")
        # sklearn trees convert input to float32 before comparing thresholds.
        converted[name] = struct.unpack("f", struct.pack("f", value))[0]
    idx = 0
    while "value" not in model["nodes"][idx]:
        node = model["nodes"][idx]
        idx = node["left"] if converted[node["feature"]] <= node["threshold"] else node["right"]
    return float(model["nodes"][idx]["value"])


@dataclass(frozen=True)
class GateConfig:
    scope: dict
    total_gpu_seconds: float
    measurement_gpu_seconds: float = 0.
    measurement_wall_seconds: float = 30.
    start_step: int = 0
    threshold: float = 0.
    model_id: str | None = None
    data_kind: str = "observed"

    def validate(self):
        validate_scope(self.scope)
        number(self.total_gpu_seconds, "total budget", 1e-12)
        number(self.measurement_gpu_seconds, "measurement budget", 0, self.total_gpu_seconds)
        number(self.measurement_wall_seconds, "measurement wall-time cap", 1e-12)
        integer(self.start_step, "initial step")
        number(self.threshold, "acceptance threshold", 0, 1)
        if self.data_kind not in {"observed", "synthetic"}:
            raise ValueError("invalid gate data kind")
        if self.model_id is not None and (not isinstance(self.model_id, str) or not self.model_id):
            raise ValueError("invalid model id")


@dataclass(frozen=True)
class GateState:
    mode: str = "ready"
    checks: int = 0
    last_step: int = -1
    total_used: float = 0.
    measurement_used: float = 0.
    measurement_wall_used: float = 0.

    def validate(self):
        if self.mode not in {"ready", "select", "random", "done"}:
            raise ValueError("invalid gate state")
        integer(self.checks, "checks")
        if self.checks > 1:
            raise ValueError("one-shot gate cannot have multiple checks")
        integer(self.last_step, "last step", -1)
        number(self.total_used, "spent total", 0)
        number(self.measurement_used, "spent measurement", 0, self.total_used)
        number(self.measurement_wall_used, "spent measurement wall time", 0)


def decide(config, state, observation, model=None):
    """Choose once before continuation training, then keep that choice.

    The prediction target already includes measurement/selection opportunity
    cost. Rejected cases still pay any measurement that actually occurred.
    """
    config.validate()
    state.validate()
    step = integer(observation["step"], "step")
    used = number(observation["total_used"], "total used", state.total_used)
    measurement = number(observation.get("measurement_used_now", 0), "measurement used now", 0)
    wall = number(observation.get("measurement_wall_now", 0), "measurement wall time", 0)
    if step < state.last_step or measurement > used-state.total_used+1e-9:
        raise ValueError("step went backwards or measurement exceeds newly spent total")
    common_valid = observation.get("common_inputs_valid", True)
    if type(common_valid) is not bool or not common_valid:
        raise ValueError("common training inputs invalid; changing the sampler cannot repair them")
    updated = {**asdict(state), "last_step": step, "total_used": used,
               "measurement_used": state.measurement_used + measurement,
               "measurement_wall_used": state.measurement_wall_used + wall}

    def result(mode, reason, prediction=None, check=False):
        new = GateState(**{**updated, "mode": mode, "checks": state.checks + int(check)})
        new.validate()
        return new, {"action": mode, "reason": reason, "step": step,
                     "prediction": prediction, "threshold": config.threshold,
                     "next_check_step": None, "decision_schedule": "once_before_training",
                     "remaining_gpu_seconds": max(0., config.total_gpu_seconds-used),
                     "overshoot_gpu_seconds": max(0., used-config.total_gpu_seconds),
                     "measurement_used": new.measurement_used, "checks": new.checks,
                     "measurement_wall_seconds": new.measurement_wall_used,
                     "evidence_scope": "experimental prediction; no individual reward guarantee"}

    if state.mode != "ready" and (measurement or wall):
        raise ValueError("measurement repeated after the one-shot decision")
    if used >= config.total_gpu_seconds or state.mode == "done":
        return result("done", "total_budget_exhausted")
    if state.mode in {"select", "random"}:
        if observation.get("scope") != config.scope:
            raise ValueError("training scope changed after the frozen decision")
        if state.mode == "select" and observation.get("selector_valid", True) is not True:
            return result("random", "invalid_selector")
        return result(state.mode, "decision_frozen")
    if step != config.start_step:
        return result("random", "initial_decision_window_missed")
    if observation.get("scope") != config.scope:
        return result("random", "unsupported_scope")
    if model is None or config.model_id is None:
        return result("random", "no_fitted_model")
    if (model.get("model_id") != config.model_id or not compatible_scope(model.get("scope"), config.scope)
            or model.get("data_kind") != config.data_kind
            or model.get("budget_gpu_seconds") != config.total_gpu_seconds):
        return result("random", "unsupported_model")
    validate_model(model)
    if updated["measurement_used"] > config.measurement_gpu_seconds:
        return result("random", "measurement_budget_exhausted", check=True)
    if updated["measurement_wall_used"] > config.measurement_wall_seconds:
        return result("random", "measurement_wall_budget_exhausted", check=True)
    if observation.get("selector_valid", True) is not True:
        return result("random", "invalid_selector", check=True)
    if observation.get("measurement_status", "ok") != "ok":
        return result("random", observation["measurement_status"], check=True)
    if observation.get("feature_step") != step:
        return result("random", "invalid_feature_time", check=True)
    if observation.get("full_pool_coverage") is not True:
        return result("random", "incomplete_pool_distribution", check=True)
    try:
        value = predict(model, observation.get("features", {}))
    except (ValueError, TypeError, KeyError):
        return result("random", "invalid_or_unsupported_features", check=True)
    if value <= config.threshold:
        return result("random", "predicted_no_useful_gain", value, check=True)
    return result("select", "predicted_useful_gain", value, check=True)


def durable_decide(path, config, observation, model=None):
    """Serialize duplicate launchers; retrying the same event never bills it twice."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    event = observation.get("event_id")
    if not isinstance(event, str) or not event.strip():
        raise ValueError("decision needs an event_id")
    config.validate()
    contract = fingerprint(asdict(config))
    request = fingerprint(observation)
    with path.with_suffix(path.suffix + ".lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        saved = read(path) if path.exists() else {"schema": SCHEMA, "contract": contract,
                                                  "state": asdict(GateState()), "events": {}}
        if saved.get("schema") != SCHEMA or saved.get("contract") != contract:
            raise ValueError("gate state contract changed")
        if event in saved["events"]:
            previous = saved["events"][event]
            if previous["request"] != request:
                raise ValueError("decision event replay has changed inputs")
            if saved.get("last_event") != event:
                raise ValueError("stale decision replay after a newer decision")
            return previous["decision"]
        state, result = decide(config, GateState(**saved["state"]), observation, model)
        saved["events"][event] = {"request": request, "decision": result}
        saved["last_event"] = event
        saved["state"] = asdict(state)
        atomic_json(path, saved)
        return result


def measurement_allowed(config, state, *, reserved_gpu_seconds, step, reserved_wall_seconds=0.):
    config.validate()
    state.validate()
    reserve = number(reserved_gpu_seconds, "reserved measurement", 0)
    wall = number(reserved_wall_seconds, "reserved measurement wall time", 0)
    integer(step, "step")
    return (state.mode == "ready" and step == config.start_step
            and config.model_id is not None and state.checks == 0
            and wall <= config.measurement_wall_seconds
            and state.measurement_used+reserve <= config.measurement_gpu_seconds
            and state.total_used+reserve < config.total_gpu_seconds)


def cost_summary(events):
    """Account for failures; incomplete events are visible, never charged as zero."""
    seen, hardware = {}, set()
    for row in events:
        event_id, state = row["event_id"], row["state"]
        if not isinstance(event_id, str) or not event_id or state not in {"started", "finished"}:
            raise ValueError("invalid cost event identity/state")
        key = (event_id, state)
        if key in seen and seen[key] != row:
            raise ValueError("conflicting duplicate cost event")
        seen[key] = row
    sums = {k: {"gpu_seconds": 0., "wall_seconds": 0., "failed_events": 0}
            for k in ("research", "deployment", "reporting")}
    pending, missing_starts = [], []
    for event_id in sorted({k[0] for k in seen}):
        start, finish = seen.get((event_id, "started")), seen.get((event_id, "finished"))
        if finish is None:
            pending.append(event_id)
            continue
        if start is None:
            missing_starts.append(event_id)
        elif any(start.get(k) != finish.get(k) for k in ("ledger", "gpus", "gpu_type", "phase")):
            raise ValueError("cost event allocation changed")
        ledger = finish["ledger"]
        if ledger not in sums:
            raise ValueError("invalid cost ledger")
        gpus = integer(finish["gpus"], "allocated devices")
        seconds = number(finish["seconds"], "phase duration", 0)
        charged = number(finish["allocated_gpu_seconds"], "charged GPU seconds", 0)
        if not math.isclose(charged, seconds*gpus, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("GPU cost does not match allocated devices and duration")
        if gpus:
            if not isinstance(finish.get("gpu_type"), str) or not finish["gpu_type"]:
                raise ValueError("missing GPU type")
            hardware.add(finish["gpu_type"])
        integer(finish["exit_code"], "exit code", -255)
        sums[ledger]["gpu_seconds"] += charged
        sums[ledger]["wall_seconds"] += seconds
        sums[ledger]["failed_events"] += int(finish["exit_code"] != 0)
    if len(hardware) > 1:
        raise ValueError("mixed GPU types cannot be added as comparable compute")
    return {"ledgers": sums, "incomplete_events": pending, "missing_starts": missing_starts,
            "complete": not pending and not missing_starts,
            "gpu_types": sorted(hardware), "wall_scope": "sum of phase durations, not concurrent makespan"}


def certificate(gains, *, family_size=1, alpha=.05, useful_gain=0.):
    integer(family_size, "family size", 1)
    number(alpha, "alpha", 1e-12, 1-1e-12)
    number(useful_gain, "useful gain", 0, 1)
    if not gains:
        raise ValueError("no independent calibration units")
    values = [number(x, "calibration gain", -1, 1) for x in gains]
    mean = sum(values)/len(values)
    radius = math.sqrt(2*math.log(2*family_size/alpha)/len(values))
    lower = mean-radius
    return {"n": len(values), "mean": mean, "radius": radius, "lower": lower,
            "accepted": lower > useful_gain, "vacuous": radius >= 1,
            "assumptions": "fixed finite family; independent whole-training units; same deployment distribution"}
