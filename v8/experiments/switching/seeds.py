"""Seed registry (plan Code §11.1): usage history, not just numbers.

Training seeds whose RLVR-switching outcomes were already inspected (0-9) are never
called fresh test seeds. Seeds >= 10 have no RLVR training use in the code repo, the
manuscript evidence, or the run plans; docs/CFCS_PROPOSAL.md mentions 'seeds 10..'
for synthetic CFCS smoke runs (a different, non-RLVR study) - recorded as a note.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

COHORTS = ("olmo_math", "olmo_mbpp", "qwen_math", "qwen_mbpp", "gemma_math", "gemma_mbpp", "llama_math", "llama_mbpp")

KNOWN = [
    {"seeds": [0, 1, 2], "cohorts": ["olmo_math", "olmo_mbpp"], "role": "historical_pre_online",
     "protocol": "selector-pair / selection-switch protocols before the 2026-09-27 online protocol",
     "inspected": True},
    {"seeds": [3, 4], "cohorts": ["olmo_math", "olmo_mbpp"], "role": "online_seed34_half_retrospective",
     "protocol": "A/B diagnostic on the continued On-policy path (Appendix C.2)", "inspected": True},
    {"seeds": [5, 6, 7, 8, 9], "cohorts": list(COHORTS), "role": "reported_additional_seeds",
     "protocol": "srgc_rebuttal online protocol (candidate40-v3), four arms to 275", "inspected": True},
]


def scan_sources(code_root: Path | None, manuscript_root: Path | None) -> list[dict]:
    found = []
    if code_root and code_root.exists():
        for p in sorted((code_root / "srgc_rebuttal" / "experiments").glob("*.json")) + sorted((code_root / "configs").glob("*.json")):
            try:
                seeds = json.loads(p.read_text()).get("seeds")
            except (ValueError, AttributeError):
                continue
            if isinstance(seeds, list):
                found.append({"source": str(p.relative_to(code_root)), "seeds": seeds})
        doc = code_root / "docs" / "CFCS_PROPOSAL.md"
        if doc.exists() and re.search(r"seeds 10\.\.", doc.read_text()):
            found.append({"source": "docs/CFCS_PROPOSAL.md", "seeds": "10..", "note": "synthetic CFCS smoke runs, not RLVR training"})
    if manuscript_root and manuscript_root.exists():
        for p in sorted((manuscript_root / "v7" / "evidence").rglob("*seeds5-9.csv")):
            with open(p) as fh:
                seeds = sorted({int(r["seed"]) for r in csv.DictReader(fh)})
            found.append({"source": str(p.relative_to(manuscript_root)), "seeds": seeds})
    return found


def registry(code_root=None, manuscript_root=None, upto=24):
    scanned = scan_sources(code_root, manuscript_root)
    used_numbers = set()
    for s in scanned:
        if isinstance(s["seeds"], list):
            used_numbers |= {int(x) for x in s["seeds"]}
    rows = []
    for cohort in COHORTS:
        for seed in range(upto + 1):
            roles = [k for k in KNOWN if seed in k["seeds"] and cohort in k["cohorts"]]
            inspected = any(k["inspected"] for k in roles)
            note = ""
            if seed >= 10:
                note = "CFCS synthetic smoke mentions seeds 10.. (non-RLVR)" if seed >= 10 else ""
            if seed in used_numbers and not roles:
                note = (note + "; " if note else "") + "number appears in another run plan"
            rows.append({"training_seed": seed, "cohort": cohort,
                         "prior_roles": ";".join(k["role"] for k in roles),
                         "prior_method_versions": ";".join(k["protocol"] for k in roles),
                         "outcomes_inspected": inspected,
                         "eligible_as_fresh_test": not inspected and seed not in used_numbers,
                         "reason": "outcomes already inspected" if inspected else (note or "no recorded RLVR use")})
    return rows, scanned


def choose(rows, cohort, *, n_test=5, n_dev=3, start=10):
    eligible = [r["training_seed"] for r in rows if r["cohort"] == cohort and r["eligible_as_fresh_test"]
                and r["training_seed"] >= start]
    if len(eligible) < n_test + n_dev:
        raise ValueError("not enough unused seeds in the registry range")
    return {"test": eligible[:n_test], "development": eligible[n_test:n_test + n_dev],
            "rebuttal_bank": [5, 6, 7, 8, 9],
            "note": "rebuttal_bank seeds 5-9 were already inspected: same-seed hindsight + LOSO, not a fresh test"}


def write_registry(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
