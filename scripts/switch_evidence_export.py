#!/usr/bin/env python3
"""Compact evidence export of the selection-switch roots: only the records the
paper's audit reads, all roots in one file.

The full `why` report carries every log tail and every record of every root, which
is tens of megabytes and cannot be moved off the cluster comfortably. The paper's
importer keeps only `switch.json` and, per branch, `result.json`, `decision.json`,
`cost.jsonl` and `policy/budget_stop.json`. This writes exactly those, for one
root, with the `UTC:` and `COMMIT:` header lines the importer reads for
provenance. On the 09-16 export the same content is 13.4 MB as a why report and
about 1 MB here.

Read-only: it opens nothing but the root's own records and needs no GPU.

Every block name is prefixed with its root's directory name, so one file can hold
several roots whose branch paths would otherwise collide. The importer wants one
root at a time with unprefixed names; scripts/switch_evidence_split.py writes those
back out on the machine that runs it.

  python3 scripts/switch_evidence_export.py --root RUNS/a --root RUNS/b --out one.txt
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# The arms and file names the importer accepts; anything else is not evidence.
# The five continuation arms sit under a state's point; mopps and random_online sit
# directly under the state, which is why they are listed apart.
POINT_ARMS = ("selection_full", "selection_reduced", "random_full", "random_reduced", "gated")
STATE_ARMS = ("mopps", "random_online")
FILES = ("result.json", "decision.json", "cost.jsonl", "policy/budget_stop.json")


def blocks(root: Path, prefix: str = ""):
    """(block name, text) for every record the importer accepts, in a stable order.

    Names are root-relative so they match the importer's anchored pattern; nothing
    under discards/, discarded/ or waivers/ can match it and none is emitted.
    """
    switch = root / "switch.json"
    if not switch.is_file():
        raise SystemExit(f"[abort] not a selection-switch root: {root}")
    yield prefix + "switch.json", switch.read_text()
    for state in sorted((root / "states").glob("s[0-9]-t[0-9]*")):
        directories = [point / arm for point in sorted((state / "points").glob("view-[0-9]*"))
                       for arm in POINT_ARMS]
        directories += [state / arm for arm in STATE_ARMS]
        for directory in directories:
            for name in FILES:
                path = directory / name
                if path.is_file():
                    yield prefix + str(path.relative_to(root)), path.read_text()


def commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       cwd=Path(__file__).resolve().parents[1]).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def report(roots) -> str:
    roots = [Path(r).resolve() for r in ([roots] if isinstance(roots, (str, Path)) else roots)]
    if len({r.name for r in roots}) != len(roots):
        raise SystemExit("[abort] roots must have distinct directory names")
    out = [f"SELECTION SWITCH EVIDENCE\nUTC: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
           f"COMMIT: {commit()}",
           "CONTENT: per root, switch.json plus each branch's result.json, decision.json, "
           "cost.jsonl and policy/budget_stop.json; no logs",
           "NAMES: every block name is prefixed with its root's directory name"]
    for root in roots:
        out.append(f"ROOT: {root}")
    out.append("")
    total = 0
    for root in roots:
        count = 0
        for name, text in blocks(root, prefix=root.name + "/"):
            out.append(f"===== {name} =====")
            out.append(text if text.endswith("\n") else text + "\n")
            count += 1
        if count < 2:
            raise SystemExit(f"[abort] no branch records under {root}")
        print(f"[records] {root.name}: {count}", file=sys.stderr)
        total += count
    # No trailer: anything after the last marker is that block's body and would be
    # parsed as part of its JSON. Counts go to stderr instead.
    print(f"[records] total {total}", file=sys.stderr)
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, action="append", required=True,
                        help="a switch root; repeat to put several roots in one file")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    text = report(args.root)
    if args.out:
        args.out.write_text(text)
        print(f"[saved] {args.out} ({len(text.encode())//1024} KB)")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
