#!/usr/bin/env python3
"""How long one V8 evaluation task takes, and how many remain, per cohort (read only).

    sh scripts/v8_eval_time.sh   -> screen + ~/v8-eval-time.txt

From .multi-node/jobs/*.json: wall time of each finished evaluation (last attempt ended_at - started_at),
the pending/running/failed counts, and a rough finish estimate = remaining tasks x median / tasks running now."""
import json, os, socket, statistics, time
from pathlib import Path

COHORTS = ("olmo_math", "qwen_mbpp", "olmo_mbpp", "qwen_math")


def hours(s):
    return f"{s / 3600:.2f}h" if s >= 3600 else f"{s / 60:.0f}min"


def cohort_lines(queue):
    times, states, failed_attempts = [], {}, 0
    for path in sorted((queue / "jobs").glob("*.json")):
        try:
            job = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        task = job.get("task") if isinstance(job.get("task"), dict) else {}
        if task.get("kind") != "evaluate":
            continue
        state = job.get("state", "pending")
        states[state] = states.get(state, 0) + 1
        attempts = [a for a in job.get("attempts") or [] if isinstance(a, dict)]
        failed_attempts += sum(a.get("outcome") == "failed" for a in attempts)
        last = attempts[-1] if attempts else {}
        a, b = last.get("started_at"), last.get("ended_at")
        if state == "done" and isinstance(a, (int, float)) and isinstance(b, (int, float)) and b > a:
            times.append(b - a)
    total = sum(states.values())
    if not total:
        return ["no evaluation tasks yet"]
    out = [f"evaluation tasks: {total}  " + "  ".join(f"{k} {v}" for k, v in sorted(states.items()))
           + (f"  (failed attempts so far {failed_attempts})" if failed_attempts else "")]
    if times:
        times.sort()
        med = statistics.median(times)
        out.append(f"one evaluation (wall time, n={len(times)}): median {hours(med)}  "
                   f"min {hours(times[0])}  max {hours(times[-1])}")
        left = total - states.get("done", 0)
        running = states.get("running", 0)
        if left and running:
            out.append(f"remaining {left}, running now {running} -> roughly {hours(left * med / running)} "
                       "if the same number keep running")
        elif left:
            out.append(f"remaining {left}, none running now")
    else:
        out.append("no finished evaluation yet")
    return out


def main():
    work = os.environ.get("OM_WORK") or f"{os.environ.get('GROUP_VOLUME', '/group-volume')}/" \
                                         f"{os.environ.get('OM_USER', 'minsoo3.kim')}/offpolicy-misranking"
    root = Path(os.environ.get("V8_ROOT") or f"{work}/v8")
    text = [f"# V8 evaluation time  host={socket.gethostname()}  at={time.strftime('%Y-%m-%d %H:%M %Z')}  root={root}"]
    for c in COHORTS:
        text += ["", f"## {c}"]
        queue = root / c / ".multi-node"
        text += cohort_lines(queue) if (queue / "jobs").is_dir() else ["not started"]
    body = "\n".join(text) + "\n"
    dest = Path(os.environ.get("V8_EVAL_TIME_OUT") or Path.home() / "v8-eval-time.txt")
    dest.write_text(body)
    print(body + f"\n[v8] saved: {dest}")


if __name__ == "__main__":
    main()
