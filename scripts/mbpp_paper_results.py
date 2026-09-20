"""Combine all current MBPP suites into one uploadable TXT, without GPU work."""

import argparse
from pathlib import Path

from paper_result_text import write_export
import switch_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    sections, coverage = [], []
    failed = False
    for root in dict.fromkeys(p.resolve() for p in args.root):
        if not (root / "switch.json").is_file():
            coverage.append({"root": str(root), "status": "unprepared"})
            continue
        try:
            sections.append(f"SUITE {root}\n" + switch_results.report(root))
            coverage.append({"root": str(root), "status": "exported",
                             "validation": "saved result snapshot; not full checkpoint lineage validation"})
        except (ValueError, OSError, KeyError, TypeError) as exc:
            failed = True
            coverage.append({"root": str(root), "status": "error", "error": str(exc)})
    write_export("mbpp", {"suites": coverage}, "\n\n".join(sections), args.out)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
