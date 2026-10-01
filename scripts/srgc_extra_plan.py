"""Find a saved prefix displaced by a recorded automatic restart, without writes."""

from itertools import chain
import sys

from srgc_rebuttal.plan import load_plan
from srgc_rebuttal.runtime import run_root
from scripts.srgc_run_history import previous_plans


def select_plan(active, seed=None):
    active = active.resolve()
    original = load_plan(active)
    seeds = [seed] if seed is not None else original["seeds"]
    if any(s not in original["seeds"] for s in seeds):
        raise ValueError("seed is not in the frozen plan")
    for candidate in chain((active,), previous_plans(active)):
        plan = load_plan(candidate)
        if plan != original:
            raise ValueError(f"previous extra-arm plan has different settings: {candidate}")
        root = run_root(candidate, plan)
        # Never bypass a present but corrupt or unfinished prefix. Its ordinary
        # verification must explain the problem instead of choosing another run.
        occupied = any(any((root / f"seed-{s}" / name).exists()
                           for name in ("prefix-ready.json", "prefix.pt", "prefix-latest.pt"))
                       for s in seeds)
        if occupied:
            if candidate != active:
                print(f"[sr-refresh] using saved prefix run: {candidate}; "
                      f"active replacement has no prefix: {active}", file=sys.stderr, flush=True)
            return candidate
    return active
