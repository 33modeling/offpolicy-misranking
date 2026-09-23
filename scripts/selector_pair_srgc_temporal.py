"""Retrospective temporal gate on a complete, saved-On D series; no GPU work."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

from paper_result_text import write_export

SOURCE_SCHEMA = "offpolicy-selector-pair/sr-gc-all-d-v1"
SCHEMA = "offpolicy-selector-pair/sr-gc-temporal-v1"


def decide(points, interval):
    """Return a signal, never an executed policy switch or its reward."""
    if type(interval) is not int or interval <= 0 or not points:
        raise ValueError("positive interval and measured D points required")
    rows = []
    candidate = None
    bounce = None
    signal_step = None
    previous_step = None
    for point in points:
        step, d = point["step"], point["d"]
        if (type(step) is not int or (previous_step is not None and step != previous_step + interval)
                or isinstance(d, bool) or not isinstance(d, (int, float)) or not math.isfinite(d)):
            raise ValueError("D checks must be finite and consecutive")
        previous_step = step
        mean = None
        if signal_step is not None:
            action = "post_signal_diagnostic"
        elif candidate is None:
            if d < 0:
                candidate = (step, d)
                action = "candidate"
            else:
                action = "retain_on"
        elif bounce is None:
            if d < 0:
                signal_step = step
                mean = (candidate[1] + d) / 2
                action = "switch_signal"
                candidate = None
            else:
                bounce = d
                action = "await_third_check"
        else:
            mean = (candidate[1] + bounce + d) / 3
            if d < 0 and mean < 0:
                signal_step = step
                action = "switch_signal"
                candidate = None
            elif d < 0:
                candidate = (step, d)
                action = "candidate"
            else:
                candidate = None
                action = "retain_on"
            bounce = None
        rows.append({"step": step, "d": d, "action": action, "window_mean": mean})
    return {"checks": rows, "signal_step": signal_step,
            "pending_confirmation": signal_step is None and candidate is not None,
            "executed_switch": False, "switched_policy_rewards": None}


def evaluate(source):
    if source.get("schema") != SOURCE_SCHEMA or source.get("status") != "complete":
        raise ValueError("a complete all-D export is required before temporal decisions")
    interval = source.get("interval")
    trajectories = []
    for row in source.get("trajectories", []):
        scheduled = row["scheduled_steps"]
        points = row["points"]
        if ([point["step"] for point in points] != scheduled or row["pending"] or row["errors"]
                or any(point["status"] != "measured" for point in points)):
            raise ValueError("all scheduled D checks must be measured")
        trajectories.append({"state": row["state"], **decide(points, interval)})
    if not trajectories:
        raise ValueError("all-D export has no trajectories")
    return {"schema": SCHEMA, "source_schema": SOURCE_SCHEMA, "interval": interval,
            "scope": "Retrospective signal on fully measured fixed-On D series; no adaptive "
                     "continuation or switched-policy reward was executed.",
            "trajectories": trajectories}


def table(report):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(("state", "step", "d", "action", "window_mean", "signal_step"))
    for trajectory in report["trajectories"]:
        for point in trajectory["checks"]:
            writer.writerow((trajectory["state"], point["step"], point["d"],
                             point["action"], point["window_mean"], trajectory["signal_step"]))
    return output.getvalue()


def parse_export(raw):
    lines = raw.decode("utf-8-sig").splitlines()
    if "DATA_JSON" not in lines:
        raise ValueError("input is not an all-D TXT export")
    return json.loads("\n".join(lines[lines.index("DATA_JSON") + 1:]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path.home() / "srgc-all-d.txt")
    parser.add_argument("--out", type=Path, default=Path.home() / "srgc-temporal.txt")
    args = parser.parse_args()
    raw = args.input.read_bytes()
    try:
        report = evaluate(parse_export(raw))
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    report["source_sha256"] = hashlib.sha256(raw).hexdigest()
    write_export("selector-pair-srgc-temporal", report, table(report), args.out)


if __name__ == "__main__":
    main()
