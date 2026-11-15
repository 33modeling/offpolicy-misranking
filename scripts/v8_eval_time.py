#!/usr/bin/env python3
"""How long one V8 evaluation takes, and how many remain, per cohort (read only).

    sh scripts/v8_eval_time.sh   -> screen + ~/v8-eval-time.txt

Thin view of scripts/v8_status.py task_model(): the same evaluation tasks (existing + certain to be created),
the same states and the same measured median as the status screen's eval column (user 11-16 'status가 정보가
하나도 안맞는다고 여러개가 있는데'). Time left here is evaluations only: remaining evaluations x median / evaluations
running now."""
from __future__ import annotations

import os
from pathlib import Path
import socket
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v8_status  # noqa: E402

COHORTS = v8_status.COHORTS


def hours(s):
    return f"{s / 3600:.2f}h" if s >= 3600 else f"{s / 60:.0f}min"


def model_lines(model):
    evals = [t for t in model['tasks'] if t['phase'] == 'eval']
    if not evals:
        return ["no evaluation tasks yet"]
    count = {s: sum(t['state'] == s for t in evals) for s in v8_status.STATES}
    done = count['done']
    out = [f"evaluations {done}/{len(evals)} done  run {count['running']}  ready {count['ready']}  "
           f"retry {count['retrying']}  wait {count['waiting']}  ✗ {count['person']}"]
    out += ['  + ' + note for note in model['more'] if '평가' in note]
    times = sorted(t['row']['ended'] - t['row']['started'] for t in evals if t['state'] == 'done' and t['row'] is not None
                   and v8_status.number(t['row'].get('started')) and v8_status.number(t['row'].get('ended'))
                   and t['row']['ended'] > t['row']['started'])
    if model['spe']:
        out.append(f"one evaluation (wall time): median {hours(model['spe'])}" +
                   (f"  n={len(times)}  min {hours(times[0])}  max {hours(times[-1])}" if times else
                    "  (from running evaluations; none finished yet)"))
        left = sum(t['left_units'] for t in evals)
        if left and count['running']:
            out.append(f"evaluations left {left:.1f}, running now {count['running']} -> ~{hours(left * model['spe'] / count['running'])}"
                       " if the same number keep running")
        elif left:
            out.append(f"evaluations left {left:.1f}, none running now")
    else:
        out.append("no measured evaluation time yet")
    return out


def main():
    work = os.environ.get("OM_WORK") or f"{os.environ.get('GROUP_VOLUME', '/group-volume')}/" \
                                         f"{os.environ.get('OM_USER', 'minsoo3.kim')}/offpolicy-misranking"
    root = Path(os.environ.get("V8_ROOT") or f"{work}/v8")
    now = time.time()
    text = [f"# V8 evaluation time  host={socket.gethostname()}  at={time.strftime('%Y-%m-%d %H:%M %Z')}  root={root}"]
    for c in COHORTS:
        model = v8_status.task_model(root, c, now)
        text += ["", f"## {c}"] + (model_lines(model) if model['started'] else ["not started"])
    body = "\n".join(text) + "\n"
    dest = Path(os.environ.get("V8_EVAL_TIME_OUT") or Path.home() / "v8-eval-time.txt")
    dest.write_text(body)
    print(body + f"\n[v8] saved: {dest}")


if __name__ == "__main__":
    main()
