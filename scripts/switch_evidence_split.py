#!/usr/bin/env python3
"""Split one evidence export back into the per-root files the paper's importer wants.

The node writes a single file so only one file has to be moved. Its block names
carry the root's directory name as a prefix; the importer's pattern is anchored and
expects names relative to one root. This strips the prefix and writes one file per
root, leaving the header lines the importer reads for provenance.

  python3 scripts/switch_evidence_split.py --export one.txt --out-dir .
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

MARKER = re.compile(r"^===== (.+) =====\s*$", re.M)


def split(text: str):
    """{root name: file text} for each root found in the export."""
    parts = MARKER.split(text)
    head = parts[0]
    utc = re.search(r"^UTC: (\S+)", head, flags=re.M)
    commit = re.search(r"^COMMIT: (\S+)", head, flags=re.M)
    roots: dict[str, list[str]] = {}
    for name, body in zip(parts[1::2], parts[2::2]):
        if "/" not in name:
            raise SystemExit(f"[abort] block name has no root prefix: {name}")
        root, relative = name.split("/", 1)
        roots.setdefault(root, []).append(f"===== {relative} =====\n"
                                          + (body if body.endswith("\n") else body + "\n"))
    if not roots:
        raise SystemExit("[abort] no blocks in this export")
    out = {}
    for root, blocks in roots.items():
        header = ["SELECTION SWITCH EVIDENCE",
                  f"UTC: {utc.group(1) if utc else 'unknown'}",
                  f"ROOT: {root}",
                  f"COMMIT: {commit.group(1) if commit else 'unknown'}", ""]
        out[root] = "\n".join(header) + "".join(blocks)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for root, text in sorted(split(args.export.read_text()).items()):
        target = args.out_dir / f"switch-evidence-{root}.txt"
        target.write_text(text)
        print(f"[saved] {target} ({len(text.encode())//1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
