"""Register diagnostic accounting before executing the verified frozen rank."""

import argparse
import json
import runpy
import sys
from pathlib import Path
from unittest.mock import patch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank-script", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = json.loads((args.output / "manifest.json").read_text())
    runtime = Path(manifest["runtime"]).resolve()
    target = runtime / "srgc_research/information_rank.py"
    if args.rank_script.resolve() != target:
        raise ValueError("information rank differs from the frozen measurement runtime")
    from srgc_research.storage import verify_runtime
    verify_runtime(runtime, manifest["identity"]["measurement_sha256"])
    # This sibling adapter is deliberately outside the hashed scientific
    # package. Imports made by the rank continue to use its frozen PYTHONPATH.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from information_gradients import aligned_problem_gradients
    from information_meter import diagnostic_costs
    with diagnostic_costs(), aligned_problem_gradients(), \
            patch.object(sys, "argv", [str(target), "--output", str(args.output)]):
        runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
