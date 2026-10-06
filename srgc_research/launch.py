"""Use each dataset's existing Python for admission, preserving the task lease."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from srgc_research.cli import launch_current
from srgc_research.design import Condition

if __name__ == "__main__":
    folder, condition_path = map(Path, sys.argv[1:3])
    manifest = json.loads((folder / "manifest.json").read_text())
    condition = Condition(**json.loads(condition_path.read_text()))
    with os.fdopen(os.dup(int(sys.argv[3])), "a+") as handle:
        raise SystemExit(launch_current(folder, condition, manifest, handle))
