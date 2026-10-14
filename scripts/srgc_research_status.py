"""Read-only research status with GPU times; frozen training modules are intact."""

import argparse
import fcntl
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.srgc_time_status import attach_scope, render
from srgc_rebuttal.runtime import Busy
from srgc_research.cli import status
from srgc_research.design import SCOPES
from srgc_research.storage import root


@contextmanager
def observed_lease(path, **kwargs):
    """Observe an existing lock without creating status-only experiment files."""
    try:
        handle = Path(path).open("r")  # noqa: SIM115 -- closed by the context manager below
    except FileNotFoundError:
        yield None
        return
    with handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise Busy(str(path)) from None
        try:
            yield handle
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("math", "mbpp", "all"))
    parser.add_argument("scope", choices=SCOPES)
    parser.add_argument("action", choices=("status",))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.scope == "n08":
        parser.error("N08 analyzes existing result archives; use results/json instead of status")
    datasets = ("math", "mbpp") if args.dataset == "all" else (args.dataset,)
    location = root()
    with patch("srgc_research.cli.lease", observed_lease):
        rows = status(location, datasets, args.scope)
    for row in rows:
        row["arm"] = row["task"]
        attach_scope(row, Path(row["result"]).parent)
    if args.json:
        print(json.dumps({"root": str(location), "scope": args.scope, "rows": rows}, indent=2, allow_nan=False))
    else:
        print(f"RESULT ROOT {location}")
        for row in rows:
            print(f"{row['dataset']} seed={row['seed']} {row['task']} {row['status']} "
                  f"step={row['step']} phase={row['phase']} result={row['result']}" +
                  (f" error={row['error']}" if row['error'] else ""))
        for dataset in datasets:
            print("\n".join(render({"rows": [row for row in rows if row["dataset"] == dataset]}, label=dataset)))
    return int(any(row["status"] == "invalid" for row in rows))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"RESEARCH STATUS ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from None
