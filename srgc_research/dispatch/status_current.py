"""Display current measured results and GPU times without saving status files."""

import math
import sys


def render_results(report):
    from .qwen_status import table

    def measured(value):
        return type(value) in (int, float) and math.isfinite(value) and value >= 0

    cells = []
    for row in report["tasks"]:
        shared = row["arm"] in {"cache", "prefix"}
        timing = row.get("timing" if shared else "submission_timing", {})
        record = timing.get("record", "MISSING")
        if row["status"] not in {"complete", "running", "invalid", "failed"} and record == "MISSING":
            continue
        ready = row.get("endpoint_ready") is True and row["status"] == "running"
        reward = row.get("saved_reward_percent") if ready else row.get("reward_percent")
        reward = (f"{reward:.1f}" if (row["status"] == "complete" or ready)
                  and measured(reward) and reward <= 100 else "-")
        hours, partial = None, False
        if row["status"] != "invalid" and record in {"COMPLETE", "PARTIAL"}:
            hours = timing.get("gpu_hours", {}).get("phases" if shared else "primary")
            if hours is None and not shared:
                subtotal = timing.get("known_recorded_subtotal_gpu_seconds")
                if measured(subtotal):
                    hours, partial = subtotal / 3600, True
            partial = partial or record == "PARTIAL"
        gpu = ((">=" if partial else "") + f"{hours:.3f}") if measured(hours) else "-"
        cells.append([row["seed"], row["arm"], reward, gpu])
    lines = ["", f"{report['label']} RESULTS / GPU TIME"]
    if cells:
        lines.extend(table(("Seed", "Arm", "Result (%)", "GPU-h"), cells))
        lines.append("GPU-h: retained training + selection + prep; Cache/Prefix shared. >= recorded subtotal; - unmeasured.")
    else:
        lines.append("No recorded results or GPU times yet.")
    return lines


def print_current(text):
    if sys.stdout.isatty():
        print("\x1b[H\x1b[2J", end="", flush=True)
    print(text, flush=True)
