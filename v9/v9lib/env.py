"""Paths and imports shared by every V9 command.

V8 results are read from $V8_ROOT (default $WORK/v8) and never written. V9 writes only below
$V9_ROOT (default $WORK/v9). The V8 runner (v8/experiments/switching) and the frozen trainer
(V8_CODE_ROOT/srgc_rebuttal) are imported, not copied.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
V9_DIR = REPO / "v9"
V8_RUNNER = REPO / "v8" / "experiments"

# Cohorts with a V8 Branch Bank (the V8 queue builds them) and the Llama cohorts (new training in V9).
BANK_COHORTS = ("olmo_math", "olmo_mbpp", "qwen_mbpp", "qwen_math")
LLAMA_COHORTS = ("llama_math", "llama_mbpp")
X1_COHORTS = BANK_COHORTS + LLAMA_COHORTS
X2_COHORTS = ("olmo_math", "olmo_mbpp")
X3_COHORTS = ("olmo_math",)
X4_COHORT = "olmo_math"
X4_SEEDS = (10, 11, 12, 13, 14)
X5_UPDATES = (25, 50, 100, 175, 275)
DEV_COHORT = "olmo_math"           # tau is selected on this cohort only
TAU_TOLERANCE_PP = 0.5
HORIZON_X3 = 400
# X8 refresh-frequency control: SR-hold, Random-hold (from the 25-update prefix), Staged-hold (from the On
# trajectory at tau0); the batch chosen at each Alignment refresh is held for the same 25 updates.
X8_COHORTS = ("olmo_math", "olmo_mbpp")
# X9 online SR: SR whose reward cache is rebuilt from the current policy at every refresh (V8 sr_early_refreshed).
X9_COHORT = "olmo_math"
# X10 budget sensitivity of the switch-time rule: every tau, SR and Alignment scored at these fractions of the
# frozen V8 OLMo-3 MATH budget (fixed before any V9 result; 1.0 is the manuscript's B).
X10_COHORT = "olmo_math"
X10_FRACTIONS = (0.75, 1.25)
# X11 baseline: DAPO-style dynamic sampling (V8 method dynsample_early) from each seed's 25-update prefix.
X11_COHORTS = ("olmo_math", "olmo_mbpp")
# X12 public benchmarks (v9lib/bench.py): the update-275 snapshots of these arms in every cohort.
X12_ARMS = {"x1": ("random", "alignment", "sr", "staged"), "x11": ("dynamic_sampling",)}
# X13 warm-up control: Random until tau0, then SR (branch of the X1 Random run). X14 order reversal: SR until tau0
# (V8 SR branch snapshot), then Alignment. Design: paper repo v9/X13_X15_DESIGN.md (decision rules fixed before running).
X13_COHORTS = ("olmo_math", "olmo_mbpp")
X14_COHORTS = ("olmo_math",)
# X15 scale: 16 prompts per update, 160 candidates per draw, OLMo MATH seeds 5-9, branched design from the initial model.
X15_COHORT = "olmo_math"
STAGES = ("freeze-tau", "x1", "x2", "x5", "x3", "x4", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "collect")


def work():
    group = os.environ.get("GROUP_VOLUME") or "/group-volume"
    user = os.environ.get("OM_USER") or "minsoo3.kim"
    return Path(os.environ.get("OM_WORK") or f"{group}/{user}/offpolicy-misranking")


def v8_root():
    return Path(os.environ.get("V8_ROOT") or work() / "v8")


def v9_root():
    return Path(os.environ.get("V9_ROOT") or work() / "v9")


def code_root():
    """The frozen trainer checkout (same discovery order as the V8 launchers)."""
    for name in ("V8_CODE_ROOT", "OM_CODE_ROOT"):
        if os.environ.get(name) and (Path(os.environ[name]) / "srgc_rebuttal" / "srgc.py").is_file():
            return Path(os.environ[name]).resolve()
    for cohort in BANK_COHORTS:
        loc = v8_root() / cohort / "locator.json"
        if loc.is_file():
            try:
                saved = json.loads(loc.read_text()).get("code", {}).get("code_root")
            except (OSError, ValueError):
                saved = None
            if saved and (Path(saved) / "srgc_rebuttal" / "srgc.py").is_file():
                return Path(saved).resolve()
    if (REPO / "srgc_rebuttal" / "srgc.py").is_file():
        return REPO
    raise FileNotFoundError("the trainer checkout (srgc_rebuttal/srgc.py) was not found; set V8_CODE_ROOT")


def activate():
    """Make `switching` (V8 runner) and `srgc_rebuttal` (frozen trainer) importable."""
    root = code_root()
    os.environ.setdefault("V8_CODE_ROOT", str(root))
    for p in (str(V9_DIR), str(V8_RUNNER), str(root / "scripts"), str(root), str(root / "src")):
        if p not in sys.path:
            sys.path.insert(0, p)
    return root


def cohort_python(cohort, fallback):
    """Same precedence as the V8 launchers (srgc_v8_nodes.cohort_python)."""
    names = ["V8_PYTHON_" + cohort.upper()]
    if cohort.startswith("qwen_"):
        names.append("QWEN_PYTHON")
    if cohort.startswith("llama_"):
        names.append("LLAMA_PYTHON")
    names.append({"math": "PAIR_PYTHON", "mbpp": "SWITCH_PYTHON"}.get(cohort.rsplit("_", 1)[-1], ""))
    names.append("V8_PYTHON")
    for name in names:
        if name and os.environ.get(name):
            return os.environ[name]
    return fallback
