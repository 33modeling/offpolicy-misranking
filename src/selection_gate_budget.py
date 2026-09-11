"""Cheap resource stopping for a frozen training choice, independent of the gate."""

import math


def stop_before_step(deadline, now, reserve, last_step_seconds):
    if deadline is None:
        return False
    if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v)
           for v in (deadline, now, reserve, last_step_seconds)):
        raise ValueError("training deadline values must be finite")
    if reserve < 0 or last_step_seconds < 0:
        raise ValueError("negative training reserve or step duration")
    return now + reserve + last_step_seconds >= deadline
