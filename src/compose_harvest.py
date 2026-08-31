"""Collapse harvest markdown components into two delivery documents."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


HEADING = re.compile(r"^(#{1,6})([ \t]+)", re.MULTILINE)


def demote_headings(markdown: str, levels: int = 2) -> str:
    def replace(match: re.Match[str]) -> str:
        return "#" * min(6, len(match.group(1)) + levels) + match.group(2)

    return HEADING.sub(replace, markdown).strip()


def write_document(path: Path, title: str, sections: list[tuple[str, Path]]) -> None:
    lines = [f"# {title}", ""]
    for label, source in sections:
        body = demote_headings(source.read_text(encoding="utf-8"))
        if not body:
            raise ValueError(f"empty markdown component: {source}")
        lines.extend((f"## {label}", "", body, ""))
    if not sections:
        raise ValueError(f"no sections for {path}")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--appendix", required=True, type=Path)
    parser.add_argument("--primary", action="append", nargs=2, default=[], metavar=("LABEL", "PATH"))
    parser.add_argument("--detail", action="append", nargs=2, default=[], metavar=("LABEL", "PATH"))
    args = parser.parse_args()
    primary = [(label, Path(path)) for label, path in args.primary]
    details = [(label, Path(path)) for label, path in args.detail]
    write_document(args.results, "Off-policy Experiment Results", primary)
    write_document(args.appendix, "Analysis Appendix", details)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
