"""Collect the current validated Pair report and complete curves in one TXT."""

import argparse
import json
import os
from pathlib import Path
import subprocess

from paper_result_text import write_export


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    repo = Path(__file__).resolve().parents[1]
    # Regenerate first. Never package a stale report after validation failed.
    result = subprocess.run(["bash", "scripts/run_selector_pair.sh", "report"], cwd=repo,
                            env={**os.environ, "PAIR_ROOT": str(root), "CUDA_VISIBLE_DEVICES": ""})
    if result.returncode:
        raise SystemExit(result.returncode)
    data = json.loads((root / "report.json").read_text())
    data["source_root"] = str(root)
    data["complete"] = not (data["missing_states"] or data["missing_development_states"])
    write_export("selector-pair", data, (root / "curves.csv").read_text(), args.out)


if __name__ == "__main__":
    main()
