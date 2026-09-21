"""Combine all current MBPP suites into one uploadable TXT, without GPU work."""

import argparse
from pathlib import Path

from paper_result_text import write_export
import mbpp_repair_results
import switch_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--repair-root", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    sections, coverage, repairs = [], [], []
    failed = False
    repair_roots = list(args.repair_root)
    for root in dict.fromkeys(p.resolve() for p in args.root):
        if (root / "repair.json").exists() or (root / "repair.json").is_symlink():
            repair_roots.append(root)
            continue
        if not (root / "switch.json").is_file():
            coverage.append({"root": str(root), "status": "unprepared"})
            continue
        try:
            sections.append(f"SUITE {root}\n" + switch_results.report(root))
            coverage.append({"root": str(root), "status": "exported",
                             "validation": "saved result snapshot; not full checkpoint lineage validation"})
        except (ValueError, OSError, KeyError, TypeError, AttributeError) as exc:
            failed = True
            coverage.append({"root": str(root), "status": "error", "error": str(exc)})
    for root in dict.fromkeys(p.resolve() for p in repair_roots):
        data = mbpp_repair_results.export(root)
        repairs.append(data)
        errors = data.get("errors") or any(row["measurement"]["issues"] for row in data["branches"])
        failed = failed or bool(errors)
        coverage.append({"root": str(root), "status": "error" if errors else "exported",
                         "kind": "repair_follow_up", "complete": data["complete"],
                         "measured": data.get("measured", {}),
                         "validation": "origin-separated repair export; not pooled with original runs"})
        sections.append(f"REPAIR FOLLOW-UP {root}\n"
                        + "\n".join(data["notes"]) + "\n"
                        + f"MEASURED {data.get('measured', {})}; COMPLETE {data['complete']}\n"
                        + mbpp_repair_results.table(data))
    write_export("mbpp", {"suites": coverage, "repair_runs": repairs}, "\n\n".join(sections), args.out)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
