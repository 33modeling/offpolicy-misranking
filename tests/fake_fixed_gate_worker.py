"""CPU subprocess fixture for the fixed gate controller; never used by launchers."""

import argparse
import os
import time
from pathlib import Path

import evidence_downstream as ed
import fixed_gate as fg

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--phase", required=True)
parser.add_argument("--shard", type=int, required=True)
args = parser.parse_args()
time.sleep(float(os.environ.get("FAKE_FIXED_GATE_SLEEP", "0")))
c = ed.read(args.out / "contract.json")
if args.phase == "assess":
    rows, parts = fg.read_phase(args.out, "pilot")
    result = fg.assess(c, rows, fg.projected_remaining_cost(c, parts))
    result["record_sha256"] = fg.core.fingerprint(result)
    fg.bind(args.out / "assessment.json", result)
else:
    stored = ed.read(Path(c["run"]) / "scores_offpolicy.json")["g11"]
    rows = []
    for i in fg.expected_ids(c, args.phase, args.shard):
        row = {"prompt_idx": i, "primary": stored[str(i)]["score"]}
        if args.phase == "pilot":
            row["replica"] = row["primary"]
        rows.append(row)
    fg.bind(args.out / args.phase / f"shard-{args.shard}.json", {
        "contract_sha256": ed.digest(args.out / "contract.json"), "phase": args.phase,
        "shard": args.shard, "rows": rows, "training_updates": 0,
        "timing": {"model_load_seconds": 1., "primary_seconds": max(1, len(rows))}})
